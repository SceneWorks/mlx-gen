"""Real-weights end-to-end golden for FLUX.2-klein txt2img **img2img** (image_path + image_strength,
sc-2644), for the #[ignore]d Rust parity test. Forces `ModelConfig.precision = float32` (the Rust
pipeline runs f32) and runs the fork's `_prepare_img2img_latents` + manual denoise loop so we can
capture the seeded noise, the clean (pre-blend) init latents, the blended initial latents, the
step-start velocity, the final latents, and the decoded image. Small (256², 4 steps, strength 0.6,
guidance 1.0) to keep the f32 run feasible.

Resizes the fork's edit asset to 256² (LANCZOS) and persists it as a lossless PNG so both the fork
and the Rust test consume byte-identical u8 init pixels (the LANCZOS resize is then a no-op).

Gitignored output. Run from the mflux fork venv:
    cd ~/repos/mflux && .venv/bin/python ~/repos/mlx-gen/tools/dump_flux2_img2img_golden.py
"""

import tempfile

import mlx.core as mx
import numpy as np
import PIL.Image

from mflux.models.common.config.model_config import ModelConfig

ModelConfig.precision = mx.float32  # the Rust pipeline runs f32 activations

from mflux.models.common.config import ModelConfig as MC  # noqa: E402
from mflux.models.common.config.config import Config  # noqa: E402
from mflux.models.common.latent_creator.latent_creator import LatentCreator  # noqa: E402
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator  # noqa: E402
from mflux.models.flux2.variants import Flux2Klein  # noqa: E402
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers  # noqa: E402

from _paths import fixture  # noqa: E402

PROMPT = "a red fox resting in fresh snow under soft winter light"
SEED, STEPS, SIZE, GUIDANCE, STRENGTH = 0, 4, 256, 1.0, 0.6
ASSET = "/Users/michael/repos/mflux/src/mflux/assets/flux2_klein_edit.jpg"

# Resize the init image to 256² (LANCZOS) and persist as a lossless PNG so the fork and the Rust
# test consume byte-identical u8 pixels (and the in-pipeline LANCZOS resize is a no-op).
init = PIL.Image.open(ASSET).convert("RGB").resize((SIZE, SIZE), PIL.Image.LANCZOS)
init_u8 = np.array(init, dtype=np.uint8)  # [256,256,3]
tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
init.save(tmp.name)

model = Flux2Klein(model_config=MC.flux2_klein_9b())
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

# Authoritative blended initial latents straight from the fork's img2img path. `init_latents`
# captures them before the denoise loop rebinds `latents` (mlx arrays are immutable).
latents, latent_ids, lat_h, lat_w = model._prepare_img2img_latents(seed=SEED, config=config)
init_latents = latents

# The seeded noise (for the Rust `create_noise` gate) and the clean pre-blend init latents (for the
# encode-chain gate) — recomputed with the same params the fork's img2img path used.
noise0, _, _, _ = Flux2LatentCreator.prepare_packed_latents(seed=SEED, height=SIZE, width=SIZE, batch_size=1)
encoded = LatentCreator.encode_image(
    vae=model.vae, image_path=tmp.name, height=SIZE, width=SIZE, tiling_config=model.tiling_config
)
encoded = _Flux2KleinEditHelpers.ensure_4d_latents(encoded)
encoded = _Flux2KleinEditHelpers.crop_to_even_spatial(encoded)
encoded = Flux2Klein._match_latent_spatial_size(encoded=encoded, target_height=lat_h * 2, target_width=lat_w * 2)
encoded = Flux2LatentCreator.patchify_latents(encoded)
encoded = _Flux2KleinEditHelpers.bn_normalize_vae_encoded_latents(encoded, vae=model.vae)
clean_latents = Flux2LatentCreator.pack_latents(encoded)

init_time_step = config.init_time_step  # max(1, int(STEPS*STRENGTH)) = 2 here
start_sigma = float(config.scheduler.sigmas[init_time_step])

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
        v0 = noise  # step-start velocity — the chaos-free real-weights transformer gate
    latents = config.scheduler.step(noise=noise, timestep=t, latents=latents, sigmas=config.scheduler.sigmas)
    mx.eval(latents)

packed = latents.reshape(latents.shape[0], lat_h, lat_w, latents.shape[-1]).transpose(0, 3, 1, 2)
decoded = model.vae.decode_packed_latents(packed)  # NCHW [1,3,H,W]
mx.eval(decoded)

out = {
    "init_u8": mx.array(init_u8.astype(np.int32)),  # [256,256,3]
    "noise0": noise0.astype(mx.float32),  # [1, seq, 128] seeded noise
    "clean_latents": clean_latents.astype(mx.float32),  # [1, seq, 128] pre-blend init
    "init_latents": init_latents.astype(mx.float32),  # [1, seq, 128] blended (1-σ)·clean + σ·noise
    "v0": v0.astype(mx.float32),  # [1, seq, 128] step-start velocity
    "latents": latents.astype(mx.float32),  # [1, seq, 128] final
    "decoded": decoded.astype(mx.float32),  # NCHW [1,3,256,256]
    "init_time_step": mx.array([init_time_step]).astype(mx.int32),
    "start_sigma": mx.array([start_sigma]),
}
path = fixture("tools/golden/flux2_img2img.safetensors")
mx.save_safetensors(path, out)
print(f"wrote {path}")
print(f"  init_u8 {tuple(init_u8.shape)}  clean {tuple(clean_latents.shape)}  v0 {tuple(v0.shape)}")
print(f"  init_time_step {init_time_step}  start_sigma {start_sigma:.6f}")
