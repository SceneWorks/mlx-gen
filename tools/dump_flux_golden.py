"""Real-weights FLUX.1 golden run — the reference for the mlx-gen FLUX e2e parity (sc-2345).

Run with the fork's venv, e.g.:
  /Users/michael/Repos/mflux/.venv/bin/python tools/dump_flux_golden.py

Loads the real black-forest-labs/FLUX.1-{schnell,dev} weights and runs a fixed
(prompt, seed, steps, size) generation BY HAND — mirroring the fork's
`Transformer.__call__` + `LinearScheduler.step` + `FluxLatentCreator` — dumping every
intermediate so the Rust port can be validated stage-by-stage:
the CLIP/T5 input_ids, the T5 `prompt_embeds`, the CLIP `pooled_prompt_embeds`, the
seeded init noise, the first velocity `v0`, the final latents, the fork sigmas, and the
decoded image. Real bf16 path (matches production / the fork's `ModelConfig.precision`).

Env-overridable: FLUX_VARIANT (schnell|dev), FLUX_PROMPT, FLUX_SEED, FLUX_STEPS, FLUX_W, FLUX_H.
"""

import os

import mlx.core as mx
import numpy as np
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux.flux_initializer import FluxInitializer
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.utils.image_util import ImageUtil

_GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(_GOLDEN_DIR, exist_ok=True)

VARIANT = os.environ.get("FLUX_VARIANT", "schnell")
# Optional f32 reference: force the fork's compute precision to float32 so the golden reflects an
# all-f32 path (the mlx-gen quality target), instead of the production bf16 conditioning. The
# transformer/text/vae weights stay as shipped; only the compute dtype changes.
if os.environ.get("FLUX_PRECISION", "").lower() in ("f32", "float32", "fp32"):
    import mlx.core as _mx

    ModelConfig.precision = _mx.float32
PROMPT = os.environ.get("FLUX_PROMPT", "a red fox")
SEED = int(os.environ.get("FLUX_SEED", "7"))
W = int(os.environ.get("FLUX_W", "256"))
H = int(os.environ.get("FLUX_H", "256"))
STEPS = int(os.environ.get("FLUX_STEPS", "4" if VARIANT == "schnell" else "20"))
GUIDANCE = float(os.environ.get("FLUX_GUIDANCE", "0.0" if VARIANT == "schnell" else "3.5"))

_PREC_SUFFIX = "_f32" if os.environ.get("FLUX_PRECISION", "").lower() in ("f32", "float32", "fp32") else ""
# QUANTIZE=8/4 dumps the fork's Q8/Q4 golden (FluxInitializer quantize=N), like dump_z_image_golden.py.
# For the e2e quant parity gate, ALSO set FLUX_PRECISION=f32: the Rust transformer runs f32 activations,
# so the reference must be quantized AND f32-compute (a bf16-precision Q golden conflates quant with the
# fork's bf16 modulation precision — large for FLUX.1-dev's guidance term). Filename gets _q{N}_f32.
QUANTIZE = int(os.environ["QUANTIZE"]) if os.environ.get("QUANTIZE") else None
_Q_SUFFIX = f"_q{QUANTIZE}" if QUANTIZE else ""
OUT = os.path.join(_GOLDEN_DIR, f"flux1_{VARIANT}{_Q_SUFFIX}{_PREC_SUFFIX}_golden.safetensors")
PNG = os.path.join(_GOLDEN_DIR, f"flux1_{VARIANT}{_Q_SUFFIX}{_PREC_SUFFIX}_golden.png")

model_config = ModelConfig.schnell() if VARIANT == "schnell" else ModelConfig.dev()


class Holder:
    pass


model = Holder()
FluxInitializer.init(model, model_config=model_config, quantize=QUANTIZE)

config = Config(
    model_config=model_config,
    num_inference_steps=STEPS,
    height=H,
    width=W,
    guidance=GUIDANCE,
)

t5_tok = model.tokenizers["t5"]
clip_tok = model.tokenizers["clip"]
t5_out = t5_tok.tokenize(PROMPT)
clip_out = clip_tok.tokenize(PROMPT)
prompt_embeds = model.t5_text_encoder(t5_out.input_ids)
pooled_prompt_embeds = model.clip_text_encoder(clip_out.input_ids)

print("variant:", VARIANT, "steps:", STEPS, "size:", f"{W}x{H}", "guidance:", GUIDANCE)
print("t5 ids shape:", tuple(t5_out.input_ids.shape), "clip ids shape:", tuple(clip_out.input_ids.shape))
print("prompt_embeds:", tuple(prompt_embeds.shape), prompt_embeds.dtype)
print("pooled:", tuple(pooled_prompt_embeds.shape), pooled_prompt_embeds.dtype)

sigmas = config.scheduler.sigmas
print("sigmas:", np.array(sigmas).tolist())

# --- transformer sub-stage intermediates (for Rust bisection) at step 0 ---
from mflux.models.flux.model.flux_transformer.transformer import Transformer  # noqa: E402

tr = model.transformer
init_dbg = FluxLatentCreator.create_noise(SEED, H, W)
hidden0 = tr.x_embedder(init_dbg)
encoder0 = tr.context_embedder(prompt_embeds)
text_embeddings0 = Transformer.compute_text_embeddings(0, pooled_prompt_embeds, tr.time_text_embed, config)
rope0 = Transformer.compute_rotary_embeddings(prompt_embeds, tr.pos_embed, config)
mx.eval(hidden0, encoder0, text_embeddings0, rope0)
print("hidden0:", tuple(hidden0.shape), hidden0.dtype)
print("encoder0:", tuple(encoder0.shape), encoder0.dtype)
print("text_embeddings0:", tuple(text_embeddings0.shape), text_embeddings0.dtype)
print("rope0:", tuple(rope0.shape), rope0.dtype)

# block-0 output (isolate one joint block)
b0_enc, b0_hid = tr.transformer_blocks[0](
    hidden_states=hidden0,
    encoder_hidden_states=encoder0,
    text_embeddings=text_embeddings0,
    rotary_embeddings=rope0,
)
# after ALL joint blocks
hs, ehs = hidden0, encoder0
for blk in tr.transformer_blocks:
    ehs, hs = blk(hidden_states=hs, encoder_hidden_states=ehs, text_embeddings=text_embeddings0, rotary_embeddings=rope0)
joint_hidden = hs
encoder_joint = ehs
# after ALL single blocks (img part), capturing the img part after block 0 too
hs_cat = mx.concatenate([ehs, hs], axis=1)
single_b0_img = None
for bi, blk in enumerate(tr.single_transformer_blocks):
    hs_cat = blk(hidden_states=hs_cat, text_embeddings=text_embeddings0, rotary_embeddings=rope0)
    if bi == 0:
        single_b0_img = hs_cat[:, ehs.shape[1]:, ...]
single_img = hs_cat[:, ehs.shape[1]:, ...]

# single block 0 internals (norm / attn / ff) fed the exact joint output
from mlx import nn as _nn  # noqa: E402

sb0 = tr.single_transformer_blocks[0]
single_in = mx.concatenate([encoder_joint, joint_hidden], axis=1)
sb0_norm, _sb0_gate = sb0.norm(single_in, text_embeddings0)
sb0_attn = sb0.attn(sb0_norm, rope0)
sb0_ff = _nn.gelu_approx(sb0.proj_mlp(sb0_norm))
mx.eval(sb0_norm, sb0_attn, sb0_ff)
mx.eval(b0_enc, b0_hid, joint_hidden, single_img)
print("DTYPES b0_hid:", b0_hid.dtype, "joint_hidden:", joint_hidden.dtype, "single_img:", single_img.dtype)
print("DTYPES hidden0:", hidden0.dtype, "encoder0:", encoder0.dtype, "text_embeddings0:", text_embeddings0.dtype)

init = FluxLatentCreator.create_noise(SEED, H, W)
latents = init
v0 = None
for t in range(STEPS):
    noise = model.transformer(
        t=t,
        config=config,
        hidden_states=latents,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
    )
    if t == 0:
        v0 = noise
    latents = config.scheduler.step(noise, t, latents, sigmas=sigmas)
    mx.eval(latents)

unpacked = FluxLatentCreator.unpack_latents(latents, H, W)
decoded = model.vae.decode(unpacked)
mx.eval(decoded)

img = ImageUtil._numpy_to_pil(ImageUtil._to_numpy(ImageUtil._denormalize(decoded)))
img.save(PNG)

tensors = {
    "t5_input_ids": t5_out.input_ids.astype(mx.int32),
    "clip_input_ids": clip_out.input_ids.astype(mx.int32),
    "prompt_embeds": prompt_embeds.astype(mx.float32),
    "pooled_prompt_embeds": pooled_prompt_embeds.astype(mx.float32),
    "init": init.astype(mx.float32),
    "v0": v0.astype(mx.float32),
    "hidden0": hidden0.astype(mx.float32),
    "encoder0": encoder0.astype(mx.float32),
    "text_embeddings0": text_embeddings0.astype(mx.float32),
    "block0_encoder": b0_enc.astype(mx.float32),
    "block0_hidden": b0_hid.astype(mx.float32),
    "joint_hidden": joint_hidden.astype(mx.float32),
    "encoder_joint": encoder_joint.astype(mx.float32),
    "single_b0_img": single_b0_img.astype(mx.float32),
    "sb0_norm": sb0_norm.astype(mx.float32),
    "sb0_attn": sb0_attn.astype(mx.float32),
    "sb0_ff": sb0_ff.astype(mx.float32),
    "rope0": rope0.astype(mx.float32),
    "single_img": single_img.astype(mx.float32),
    "final_latents": latents.astype(mx.float32),
    "decoded": decoded.astype(mx.float32),
    "sigmas": sigmas.astype(mx.float32),
}
meta = {
    "variant": VARIANT,
    "prompt": PROMPT,
    "seed": str(SEED),
    "steps": str(STEPS),
    "w": str(W),
    "h": str(H),
    "guidance": str(GUIDANCE),
    "quantize": str(QUANTIZE),
}
mx.save_safetensors(OUT, tensors, meta)
print(f"\nwrote {OUT} + {PNG}")
print(f"final_latents {tuple(latents.shape)}, decoded {tuple(decoded.shape)}")
