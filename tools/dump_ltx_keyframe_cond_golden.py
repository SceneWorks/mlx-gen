"""Dump a golden for the LTX **keyframe-append** (IC-LoRA in-context) conditioning op (epic 3040 /
sc-3052), straight from the torch reference `ltx_core` so the Rust `append_keyframe_clip` /
`keyframe_append_positions` port can be byte-validated against the *actual* pipeline code (not a
re-derived formula).

It exercises `VideoConditionByKeyframeIndex.apply_to` (the `video_conditioning` path) over a tiny
base `LatentState` and dumps the **appended** tokens / positions / denoise-mask (the slice past the
base tokens) for a couple of frame indices. The base state is trivial (zeros), so the appended slice
fully isolates the new math.

Reference: `<LTX2_SRC>/ltx2/ltx_core/conditioning/types/keyframe_cond.py`.

Run (needs torch + einops in the reference venv); LTX2_SRC points at your Wan2GP `models` dir:
    LTX2_SRC=<path-to-Wan2GP>/models \
      python tools/dump_ltx_keyframe_cond_golden.py
Writes `mlx-gen-ltx/tests/fixtures/ltx_keyframe_cond_golden.safetensors`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

from _paths import fixture, require_env

LTX2_SRC = require_env(
    "LTX2_SRC", "set it to your Wan2GP `models` dir (the one holding `ltx2/ltx_core`)"
)
sys.path.insert(0, LTX2_SRC)

from ltx2.ltx_core.components.patchifiers import VideoLatentPatchifier  # noqa: E402
from ltx2.ltx_core.conditioning.types.keyframe_cond import (  # noqa: E402
    VideoConditionByKeyframeIndex,
)
from ltx2.ltx_core.types import (  # noqa: E402
    LatentState,
    SpatioTemporalScaleFactors,
    VideoLatentShape,
)
from ltx2.ltx_core.tools import VideoLatentTools  # noqa: E402

# Small deterministic case: C=4 channels, keyframe of cf=1 latent frame, h=2, w=3.
C, CF, H, W = 4, 1, 2, 3
FPS = 24.0
SCALE = SpatioTemporalScaleFactors(time=8, width=32, height=32)
patchifier = VideoLatentPatchifier(patch_size=1)
# target_shape is unused by apply_to but required by the dataclass.
tools = VideoLatentTools(
    patchifier=patchifier,
    target_shape=VideoLatentShape(batch=1, channels=C, frames=4, height=H, width=W),
    fps=FPS,
    scale_factors=SCALE,
    causal_fix=True,
)

# A trivial single-token base state (zeros); the appended slice is what we validate.
base = LatentState(
    latent=torch.zeros(1, 1, C),
    denoise_mask=torch.ones(1, 1, 1),
    positions=torch.zeros(1, 3, 1, 2),
    clean_latent=torch.zeros(1, 1, C),
)

# Deterministic keyframe latent (B, C, cf, h, w).
keyframe = torch.arange(C * CF * H * W, dtype=torch.float32).reshape(1, C, CF, H, W)

tensors: dict[str, np.ndarray] = {
    "keyframe": keyframe.cpu().numpy().astype(np.float32),
}
meta = {"C": str(C), "cf": str(CF), "h": str(H), "w": str(W), "fps": str(FPS)}

for tag, frame_idx, strength in [("f0", 0, 1.0), ("f5", 5, 0.8)]:
    cond = VideoConditionByKeyframeIndex(keyframes=keyframe, frame_idx=frame_idx, strength=strength)
    out = cond.apply_to(base, tools)
    n_base = base.latent.shape[1]
    # Appended slice (past the base token).
    tensors[f"{tag}_latent"] = out.latent[:, n_base:].cpu().numpy().astype(np.float32)
    tensors[f"{tag}_mask"] = out.denoise_mask[:, n_base:].cpu().numpy().astype(np.float32)
    tensors[f"{tag}_positions"] = out.positions[:, :, n_base:].cpu().numpy().astype(np.float32)
    meta[f"{tag}_frame_idx"] = str(frame_idx)
    meta[f"{tag}_strength"] = str(strength)

out_path = fixture("mlx-gen-ltx/tests/fixtures/ltx_keyframe_cond_golden.safetensors")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, out_path, metadata=meta)
print(f"wrote {out_path}")
for k, v in tensors.items():
    print(f"  {k}: {v.shape} {v.dtype}")
