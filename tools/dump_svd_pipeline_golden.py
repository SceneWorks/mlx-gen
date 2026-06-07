"""Dump a deterministic e2e golden for the SVD pipeline (epic 3040 / sc-3375) from the real
diffusers `StableVideoDiffusionPipeline` components. Validates `SvdPipeline::{denoise,decode}` — the
frame-wise CFG v-prediction Euler loop (image-latent channel-concat) + chunked temporal decode —
against the reference, fed identical conditioning + init noise so the only variable is the math.

The image preprocessing (`_resize_with_antialiasing` + CLIP normalize) is intentionally bypassed:
we feed pre-computed `image_embeds`/`image_latents` (the deterministic core), and the encoder +
VAE-encode are validated separately (S2/S1 parity).

Small (latent 8×8 = 64px, F=3, 2 steps) so the deep UNet runs cheaply.

Run: /Users/michael/repos/mflux/.venv-0312/bin/python tools/dump_svd_pipeline_golden.py
Writes `mlx-gen-svd/tests/fixtures/svd_pipeline_golden.safetensors`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from diffusers import StableVideoDiffusionPipeline
from safetensors.numpy import save_file

from _paths import fixture, hf_hub_cache

SNAP = (
    hf_hub_cache()
    / "models--stabilityai--stable-video-diffusion-img2vid-xt"
    / "snapshots"
)
snap_dir = next(SNAP.iterdir())

pipe = StableVideoDiffusionPipeline.from_pretrained(snap_dir, torch_dtype=torch.float32)
pipe.to("cpu")
vae, unet, image_encoder, scheduler = pipe.vae, pipe.unet, pipe.image_encoder, pipe.scheduler

rng = np.random.default_rng(3375)
F, hw, steps = 3, 8, 2
min_g, max_g = 1.0, 3.0


def randn(*shape):
    return rng.standard_normal(shape).astype(np.float32)


with torch.no_grad():
    # --- conditioning (deterministic; bypasses the antialiased resize) ---
    clip_px = torch.from_numpy(randn(1, 3, 224, 224))  # pre-CLIP-normalized pixel_values
    image_embeds = image_encoder(clip_px).image_embeds.unsqueeze(1)  # [1,1,1024]

    vae_img = torch.from_numpy(randn(1, 3, hw * 8, hw * 8))  # preprocessed + noise-aug, [-1,1]
    image_latents = vae.encode(vae_img).latent_dist.mode()  # [1,4,8,8]
    image_latents = image_latents.unsqueeze(1).repeat(1, F, 1, 1, 1)  # [1,F,4,8,8]

    added_time_ids = torch.tensor([[6.0, 127.0, 0.02]], dtype=torch.float32)  # fps-1, motion, noise_aug

    # --- init latents (seeded) scaled by init_noise_sigma ---
    scheduler.set_timesteps(steps, device="cpu")
    timesteps = scheduler.timesteps
    init_latents = torch.from_numpy(randn(1, F, 4, hw, hw)) * scheduler.init_noise_sigma  # [1,F,4,8,8]

    # --- frame-wise CFG denoise loop (mirrors __call__) ---
    guidance = torch.linspace(min_g, max_g, F).unsqueeze(0)  # [1,F]
    guidance = guidance[..., None, None, None]  # _append_dims to ndim 5

    embeds_cfg = torch.cat([torch.zeros_like(image_embeds), image_embeds])
    img_lat_cfg = torch.cat([torch.zeros_like(image_latents), image_latents])
    atid_cfg = torch.cat([added_time_ids, added_time_ids])

    lat = init_latents.clone()
    for t in timesteps:
        inp = torch.cat([lat] * 2)
        inp = scheduler.scale_model_input(inp, t)
        inp = torch.cat([inp, img_lat_cfg], dim=2)
        pred = unet(inp, t, encoder_hidden_states=embeds_cfg, added_time_ids=atid_cfg).sample
        uncond, cond = pred.chunk(2)
        pred = uncond + guidance * (cond - uncond)
        lat = scheduler.step(pred, t, lat).prev_sample
    final_latents = lat  # [1,F,4,8,8]

    frames = pipe.decode_latents(final_latents, F, decode_chunk_size=F)  # [1,3,F,64,64]

tensors = {
    "image_embeds": image_embeds.numpy().astype(np.float32),
    "image_latents": image_latents.numpy().astype(np.float32),
    "added_time_ids": added_time_ids.numpy().astype(np.float32),
    "init_latents": init_latents.numpy().astype(np.float32),
    "final_latents": final_latents.numpy().astype(np.float32),
    "frames": frames.numpy().astype(np.float32),  # [1,3,F,64,64]
    "meta": np.array([F, steps], dtype=np.int32),
    "guidance": np.array([min_g, max_g], dtype=np.float32),
}
out_path = fixture("mlx-gen-svd/tests/fixtures/svd_pipeline_golden.safetensors")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, out_path)
print(f"wrote {out_path}")
print("  final_latents:", final_latents.shape, " frames:", frames.shape)
print("  frames range:", float(frames.min()), float(frames.max()))
