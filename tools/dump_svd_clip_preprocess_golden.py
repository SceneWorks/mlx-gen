"""Dump a golden for the SVD CLIP-image antialiased preprocess (epic 3040 / sc-3412) directly from
diffusers' `_resize_with_antialiasing` (`pipelines/stable_video_diffusion`). This is the exact
pre-`feature_extractor` path of `StableVideoDiffusionPipeline._encode_image`:

    image = pil_to_numpy(image)         # [1,H,W,3] in [0,1]
    image = numpy_to_pt(image)          # [1,3,H,W]
    image = image * 2 - 1               # [-1,1]
    image = _resize_with_antialiasing(image, (224, 224))
    image = (image + 1) / 2             # [1,3,224,224] in [0,1]

It needs only the pipeline's static functions (no checkpoint, no model weights), so the resulting
golden gates the Rust `mlx_gen_svd::resize_with_antialiasing_unit` in a fast non-ignored test. A
moderately-sized non-square downscale (448×800 → 224²) exercises an asymmetric blur (kx=5, ky=3).

Run: /Users/michael/repos/mflux/.venv-0312/bin/python tools/dump_svd_clip_preprocess_golden.py
Writes `mlx-gen-svd/tests/fixtures/svd_clip_preprocess_golden.safetensors`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _resize_with_antialiasing,
)

from _paths import fixture

IN_H, IN_W = 448, 800
OUT_H, OUT_W = 224, 224

# A deterministic RGB8 image (smooth gradients + a high-frequency checker patch so the gaussian
# blur + antialias actually do work, and edges exercise the reflect padding / border clamp).
rng = np.random.default_rng(3412)
yy, xx = np.mgrid[0:IN_H, 0:IN_W]
img = np.zeros((IN_H, IN_W, 3), dtype=np.float32)
img[..., 0] = (xx / (IN_W - 1)) * 255.0
img[..., 1] = (yy / (IN_H - 1)) * 255.0
img[..., 2] = ((np.sin(xx / 7.0) * np.cos(yy / 5.0) * 0.5 + 0.5) * 255.0)
# A high-frequency checkerboard patch (aliasing stress) + a little noise.
checker = (((xx // 2) + (yy // 2)) % 2) * 255.0
img[100:200, 120:260, :] = checker[100:200, 120:260, None]
img = np.clip(img + rng.integers(-4, 5, size=img.shape), 0, 255).astype(np.uint8)

# Replicate `_encode_image`'s pre-feature_extractor steps exactly.
arr = img.astype(np.float32) / 255.0          # [H,W,3] in [0,1]
pt = torch.from_numpy(arr[None].transpose(0, 3, 1, 2))  # [1,3,H,W]
pt = pt * 2.0 - 1.0
with torch.no_grad():
    resized = _resize_with_antialiasing(pt, (OUT_H, OUT_W))
    resized_unit = (resized + 1.0) / 2.0      # [1,3,224,224] in [0,1]
resized_unit = resized_unit.cpu().numpy().astype(np.float32)

tensors = {
    "input_image": img.astype(np.float32),    # HWC [448,800,3], integer-valued 0..255
    "resized_unit": resized_unit,             # NCHW [1,3,224,224] in [0,1]
}
out_path = fixture("mlx-gen-svd/tests/fixtures/svd_clip_preprocess_golden.safetensors")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, out_path)
print(f"wrote {out_path}")
print("  input_image:", img.shape, " resized_unit:", resized_unit.shape)
print("  resized_unit[0,:,0,0]:", resized_unit[0, :, 0, 0])
print("  resized_unit mean/std:", resized_unit.mean(), resized_unit.std())
