"""sc-2643: Q4/Q8 quantization goldens for FLUX.2-klein-9b. The fork quantizes the WHOLE model
(transformer + Qwen3 text encoder + VAE) via `nn.quantize(predicate=hasattr to_quantized, bits)`
at the default `ModelConfig.precision = bfloat16` (so it packs bf16 weights, group_size 64). This
script dumps, for a chosen bit-width:

  (1) byte-parity refs — the fork's in-memory quantized `weight`/`scales`/`biases` for three
      representative modules, one per distinct packing scenario:
        * transformer  `transformer_blocks.0.attn.to_q`  (bias-less, bf16-native Linear)
        * text encoder `embed_tokens`                      (nn.Embedding → QuantizedEmbedding)
        * vae          encoder mid-block `to_q`            (f32-on-disk Linear, WITH bias)
      The Rust port loads weights f32 but casts to bf16 before packing (sc-2604 chokepoint), so the
      packing must be byte-identical to these.

  (2) e2e render — the full quantized pipeline at 256²/4 steps (seeded noise, prompt embeds/ids,
      step-0 velocity, decoded image), so the Rust `load(Q).generate()` can be gated end-to-end.

Gitignored output. Run from the mflux fork venv, once per bit-width:
    cd ~/repos/mflux && BITS=8 .venv/bin/python ~/repos/mlx-gen/tools/dump_flux2_quant_golden.py
    cd ~/repos/mflux && BITS=4 .venv/bin/python ~/repos/mlx-gen/tools/dump_flux2_quant_golden.py
"""

import os
import re

import mlx.core as mx
from mlx.utils import tree_flatten

from mflux.models.common.config import ModelConfig as MC
from mflux.models.common.config.config import Config
from mflux.models.flux2.variants import Flux2Klein

from _paths import fixture

BITS = int(os.environ.get("BITS", "8"))
PROMPT = "a red fox resting in fresh snow under soft winter light"
SEED, STEPS, SIZE, GUIDANCE = 0, 4, 256, 1.0

# Default precision = bf16 (do NOT force f32): the fork packs bf16 weights, which the Rust
# bf16-cast-before-quantize chokepoint must byte-match.
print(f"FLUX.2-klein-9b quant golden: bits={BITS}, precision={MC.precision}")
model = Flux2Klein(quantize=BITS, model_config=MC.flux2_klein_9b())

out = {}


def grab(component, flat, exact=None, pattern=None, label=""):
    """Dump <path>.{weight,scales,biases} for the unique module matching `exact` or `pattern`."""
    keys = {k for k, _ in flat}
    if exact is not None:
        path = exact
        assert f"{path}.scales" in keys, f"{label}: {path}.scales not found (not quantized?)"
    else:
        rx = re.compile(pattern)
        matches = sorted({m.group(1) for k in keys if (m := rx.match(k))})
        assert len(matches) == 1, f"{label}: expected 1 match for {pattern!r}, got {matches}"
        path = matches[0]
    d = dict(flat)
    for leaf in ("weight", "scales", "biases"):
        out[f"{label}_{leaf}"] = d[f"{path}.{leaf}"]
    print(f"  {label}: {path}  wq{tuple(d[f'{path}.weight'].shape)} "
          f"scales{tuple(d[f'{path}.scales'].shape)} ({d[f'{path}.scales'].dtype})")


t_flat = tree_flatten(model.transformer.parameters())
e_flat = tree_flatten(model.text_encoder.parameters())
v_flat = tree_flatten(model.vae.parameters())

grab(model.transformer, t_flat, exact="transformer_blocks.0.attn.to_q", label="t_to_q")
grab(model.text_encoder, e_flat, exact="embed_tokens", label="te_embed")
grab(model.vae, v_flat, pattern=r"^(encoder\..*\.to_q)\.scales$", label="vae_enc_q")

# --- (2) full quantized e2e render (manual pipeline, mirrors dump_flux2_e2e_golden.py) ---
prompt_embeds, text_ids, neg_embeds, neg_ids = model._encode_prompt_pair(
    prompt=PROMPT, negative_prompt=" ", guidance=GUIDANCE
)


def render(size, capture_v0):
    config = Config(
        model_config=model.model_config,
        num_inference_steps=STEPS,
        height=size,
        width=size,
        guidance=GUIDANCE,
        scheduler="flow_match_euler_discrete",
    )
    latents, latent_ids, lat_h, lat_w = model._prepare_generation_latents(seed=SEED, config=config)
    noise0 = latents
    predict = model._predict(model.transformer)
    v0 = None
    for t in range(config.init_time_step, config.num_inference_steps):
        noise = predict(
            latents=latents,
            latent_ids=latent_ids,
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            negative_prompt_embeds=neg_embeds,
            negative_text_ids=neg_ids,
            guidance=GUIDANCE,
            timestep=config.scheduler.timesteps[t],
        )
        if v0 is None:
            v0 = noise
        latents = config.scheduler.step(noise=noise, timestep=t, latents=latents, sigmas=config.scheduler.sigmas)
        mx.eval(latents)
    packed = latents.reshape(latents.shape[0], lat_h, lat_w, latents.shape[-1]).transpose(0, 3, 1, 2)
    decoded = model.vae.decode_packed_latents(packed)  # NCHW [1,3,size,size]
    mx.eval(decoded)
    if capture_v0:
        out.update({
            "noise0": noise0.astype(mx.float32),
            "prompt_embeds": prompt_embeds.astype(mx.float32),
            "text_ids": text_ids.astype(mx.float32),
            "latent_ids": latent_ids.astype(mx.float32),
            "v0": v0.astype(mx.float32),
        })
    return decoded.astype(mx.float32)


# 256² carries the chaos-free v0 gate (cheap forward); 512² is the higher-res render check (the
# story's @512²) where the 4-step sampler is less chaos-sensitive, so the f32-vs-bf16 render gap
# shrinks toward parity.
out["decoded"] = render(SIZE, capture_v0=True)
out["decoded_512"] = render(512, capture_v0=False)

path = fixture(f"tools/golden/flux2_quant_q{BITS}.safetensors")
mx.save_safetensors(path, out, metadata={"bits": str(BITS)})
print(f"wrote {path}  decoded{tuple(out['decoded'].shape)}  decoded_512{tuple(out['decoded_512'].shape)}")
