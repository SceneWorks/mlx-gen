"""sc-2644: Q8/Q4 **img2img** render golden for FLUX.2-klein-9b. The img2img latent prep (VAE encode
+ blend) and the quantization scope are both already proven elsewhere — the dense img2img golden
(`dump_flux2_img2img_golden.py`) gates the chaos-free latent-prep chain bit-tight, and the sc-2643
quant goldens gate the packing byte-parity + quantized forward. This script closes the explicit
"Q8 img2img" parity ask end-to-end: the fork's quantized img2img render, for the Rust
`load(Q).generate(Reference{strength})` render gate.

Default precision = bf16 (do NOT force f32): the fork packs bf16 weights and runs bf16 activations,
so the Rust f32-activation render is a bounded coherence floor vs this (like the sc-2643 render
gate), not bit-parity.

Gitignored output. Run from the mflux fork venv, once per bit-width:
    cd ~/repos/mflux && BITS=8 .venv/bin/python ~/repos/mlx-gen/tools/dump_flux2_img2img_quant_golden.py
"""

import os
import tempfile

import mlx.core as mx
import numpy as np
import PIL.Image

from mflux.models.common.config import ModelConfig as MC
from mflux.models.common.config.config import Config
from mflux.models.flux2.variants import Flux2Klein

from _paths import fixture, mflux_asset

BITS = int(os.environ.get("BITS", "8"))
PROMPT = "a red fox resting in fresh snow under soft winter light"
SEED, STEPS, SIZE, GUIDANCE, STRENGTH = 0, 4, 256, 1.0, 0.6
ASSET = mflux_asset("flux2_klein_edit.jpg")

# Byte-identical u8 init pixels (LANCZOS resize to 256² then a lossless PNG; the in-pipeline resize
# is a no-op).
init = PIL.Image.open(ASSET).convert("RGB").resize((SIZE, SIZE), PIL.Image.LANCZOS)
init_u8 = np.array(init, dtype=np.uint8)
tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
init.save(tmp.name)

print(f"FLUX.2-klein-9b img2img quant golden: bits={BITS}, precision={MC.precision}")
model = Flux2Klein(quantize=BITS, model_config=MC.flux2_klein_9b())

config = Config(
    model_config=model.model_config,
    num_inference_steps=STEPS,
    height=SIZE,
    width=SIZE,
    guidance=GUIDANCE,
    image_path=tmp.name,
    image_strength=STRENGTH,
    scheduler="flow_match_euler_discrete",
)
prompt_embeds, text_ids, neg_embeds, neg_ids = model._encode_prompt_pair(
    prompt=PROMPT, negative_prompt=" ", guidance=GUIDANCE
)
latents, latent_ids, lat_h, lat_w = model._prepare_img2img_latents(seed=SEED, config=config)

predict = model._predict(model.transformer)
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
    latents = config.scheduler.step(noise=noise, timestep=t, latents=latents, sigmas=config.scheduler.sigmas)
    mx.eval(latents)

packed = latents.reshape(latents.shape[0], lat_h, lat_w, latents.shape[-1]).transpose(0, 3, 1, 2)
decoded = model.vae.decode_packed_latents(packed)  # NCHW [1,3,256,256]
mx.eval(decoded)

out = {
    "init_u8": mx.array(init_u8.astype(np.int32)),  # [256,256,3]
    "decoded": decoded.astype(mx.float32),  # NCHW [1,3,256,256]
}
path = fixture(f"tools/golden/flux2_img2img_q{BITS}.safetensors")
mx.save_safetensors(path, out, metadata={"bits": str(BITS)})
print(f"wrote {path}  decoded{tuple(out['decoded'].shape)}  init_time_step {config.init_time_step}")
