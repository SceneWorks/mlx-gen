"""Real-weights end-to-end golden for FLUX.2-klein single-reference EDIT (sc-2346 S5), for the
#[ignore]d Rust test. Forces f32, resizes the fork's edit asset to 256² (so the Rust test can pass
byte-identical reference pixels and the LANCZOS resize is a no-op), and runs the full edit pipeline
manually (no KV-cache — that's the 9b-kv variant) to capture the seeded noise, the decoded image,
and the resized reference pixels.

Gitignored output. Run from the mflux fork venv:
    cd ~/repos/mflux && .venv/bin/python ~/repos/mlx-gen/tools/dump_flux2_edit_golden.py
"""

import tempfile

import mlx.core as mx
import numpy as np
import PIL.Image

from mflux.models.common.config.model_config import ModelConfig

ModelConfig.precision = mx.float32

from mflux.models.common.config import ModelConfig as MC  # noqa: E402
from mflux.models.common.config.config import Config  # noqa: E402
from mflux.models.flux2.variants import Flux2KleinEdit  # noqa: E402
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers  # noqa: E402

from _paths import fixture, mflux_asset  # noqa: E402

PROMPT = "make it look like a cold winter morning"
SEED, STEPS, SIZE, GUIDANCE = 0, 4, 256, 1.0
ASSET = mflux_asset("flux2_klein_edit.jpg")

# Resize the reference to 256² (LANCZOS) and persist as a lossless PNG so both the fork and the Rust
# test consume byte-identical u8 pixels.
ref = PIL.Image.open(ASSET).convert("RGB").resize((SIZE, SIZE), PIL.Image.LANCZOS)
ref_u8 = np.array(ref, dtype=np.uint8)  # [256,256,3]
tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
ref.save(tmp.name)

model = Flux2KleinEdit(model_config=MC.flux2_klein_9b())
config = Config(
    model_config=model.model_config,
    num_inference_steps=STEPS,
    height=SIZE,
    width=SIZE,
    guidance=GUIDANCE,
    image_path=tmp.name,
    scheduler="flow_match_euler_discrete",
)

prompt_embeds, text_ids, neg_embeds, neg_ids = model._encode_prompt_pair(
    prompt=PROMPT, negative_prompt=" ", guidance=GUIDANCE
)
latents, latent_ids, lat_h, lat_w = _Flux2KleinEditHelpers.prepare_generation_latents(
    seed=SEED, height=SIZE, width=SIZE
)
noise0 = latents
image_latents, image_latent_ids = _Flux2KleinEditHelpers.prepare_reference_image_conditioning(
    vae=model.vae, tiling_config=None, image_paths=[tmp.name], height=SIZE, width=SIZE, batch_size=1
)

predict = model._predict(model.transformer)
v0 = None
for t in range(config.init_time_step, config.num_inference_steps):
    noise = predict(
        latents=latents,
        image_latents=image_latents,
        latent_ids=latent_ids,
        image_latent_ids=image_latent_ids,
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
decoded = model.vae.decode_packed_latents(packed)
mx.eval(decoded)

out = {
    "ref_u8": mx.array(ref_u8.astype(np.int32)),  # [256,256,3]
    "noise0": noise0.astype(mx.float32),
    "v0": v0.astype(mx.float32),  # [1, seq_tgt, 128]
    "latents": latents.astype(mx.float32),
    "decoded": decoded.astype(mx.float32),  # NCHW [1,3,256,256]
    "image_latents": image_latents.astype(mx.float32),  # [1, seq_ref, 128]
}
path = fixture("tools/golden/flux2_edit.safetensors")
mx.save_safetensors(path, out)
print(f"wrote {path}")
print(f"  ref_u8 {tuple(ref_u8.shape)}  image_latents {tuple(image_latents.shape)}  v0 {tuple(v0.shape)}")
