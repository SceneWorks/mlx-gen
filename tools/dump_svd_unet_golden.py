"""Dump a golden for the SVD UNet (epic 3040 / sc-3374) from the real diffusers
`UNetSpatioTemporalConditionModel` (the SVD `unet`). Validates the Rust `SvdUnet::forward`
(full down/mid/up spatiotemporal stack + micro-conditioning) byte-close in f32.

Small latent (B=1, F=2, 16×16) so the 3 downsamples + temporal Conv3d path are exercised cheaply.

Run: /Users/michael/repos/mflux/.venv-0312/bin/python tools/dump_svd_unet_golden.py
Writes `mlx-gen-svd/tests/fixtures/svd_unet_golden.safetensors`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from diffusers import UNetSpatioTemporalConditionModel
from safetensors.numpy import save_file

from _paths import fixture, hf_hub_cache

SNAP = (
    hf_hub_cache()
    / "models--stabilityai--stable-video-diffusion-img2vid-xt"
    / "snapshots"
)
snap_dir = next(SNAP.iterdir())
unet_dir = snap_dir / "unet"

unet = UNetSpatioTemporalConditionModel.from_pretrained(unet_dir, torch_dtype=torch.float32)
unet.eval()

rng = np.random.default_rng(3374)
b, f, hw = 1, 2, 16
sample = rng.standard_normal((b, f, 8, hw, hw)).astype(np.float32)  # [B,F,8,H,W]
timestep = np.float32(1.0)
image_embeds = rng.standard_normal((b, 1, 1024)).astype(np.float32)  # encoder_hidden_states
added_time_ids = np.array([[6.0, 127.0, 0.02]], dtype=np.float32)  # [fps-1, motion_bucket, noise_aug]

with torch.no_grad():
    out = unet(
        torch.from_numpy(sample),
        torch.tensor(float(timestep)),
        torch.from_numpy(image_embeds),
        added_time_ids=torch.from_numpy(added_time_ids),
    ).sample
    out = out.cpu().numpy().astype(np.float32)  # [B,F,4,H,W]

# --- Isolated TransformerSpatioTemporalModel (down_blocks.0.attentions.0) for bisecting parity. ---
tf = unet.down_blocks[0].attentions[0]
tf_c = 320
t_in = rng.standard_normal((b * f, tf_c, hw, hw)).astype(np.float32)
t_ctx = rng.standard_normal((b * f, 1, 1024)).astype(np.float32)
ioi = np.zeros((b, f), dtype=np.float32)
with torch.no_grad():
    t_out = tf(
        torch.from_numpy(t_in),
        encoder_hidden_states=torch.from_numpy(t_ctx),
        image_only_indicator=torch.from_numpy(ioi),
        return_dict=False,
    )[0]
    t_out = t_out.cpu().numpy().astype(np.float32)

# --- Isolated SpatioTemporalResBlock (down_blocks.0.resnets.0) for bisecting parity. ---
rb = unet.down_blocks[0].resnets[0]
r_in = rng.standard_normal((b * f, tf_c, hw, hw)).astype(np.float32)
r_temb = rng.standard_normal((b * f, 1280)).astype(np.float32)
with torch.no_grad():
    r_out = rb(
        torch.from_numpy(r_in),
        torch.from_numpy(r_temb),
        image_only_indicator=torch.from_numpy(ioi),
    )
    r_out = r_out.cpu().numpy().astype(np.float32)

tensors = {
    "sample": sample,
    "timestep": np.array([timestep], dtype=np.float32),
    "image_embeds": image_embeds,
    "added_time_ids": added_time_ids,
    "out": out,
    "num_frames": np.array([f], dtype=np.int32),
    # transformer isolation
    "tf_in": t_in,
    "tf_ctx": t_ctx,
    "tf_out": t_out,
    # resnet isolation
    "rb_in": r_in,
    "rb_temb": r_temb,
    "rb_out": r_out,
}
out_path = fixture("mlx-gen-svd/tests/fixtures/svd_unet_golden.safetensors")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, out_path)
print(f"wrote {out_path}")
print("  sample:", sample.shape, " out:", out.shape, " timestep:", float(timestep))
print("  out[0,0,:, 0,0]:", out[0, 0, :, 0, 0])
