"""sc-5136: golden for the Bernini planner's ViT (Qwen2.5-VL) image preprocessing.

Two exactly-matchable pieces of `Qwen2VLImageProcessor`:
  - **smart_resize** — the target `(h_bar, w_bar)` (factor 28, area clamp [min,max], aspect kept,
    Python banker's `round`). grid_thw derives from these, so they must be **bit-exact**.
  - **patch packing + rescale + normalize** — `_preprocess` with `do_resize=False` (so the
    non-bit-identical PIL bicubic resize is excluded): rescale 1/255 → CLIP normalize → temporal-pad to
    `temporal_patch_size` → the 9-axis reshape/transpose → `pixel_values [seq, 1176]` + `grid_thw`.

`smart_resize` is called directly and `Qwen2VLImageProcessor._preprocess` is the oracle (run on a
fixed uint8 image whose dims are multiples of 28), so the reference is exact.

Run:
  ~/Repos/mflux/.venv/bin/python tools/dump_bernini_vit_preprocess_golden.py
Fixture -> mlx-gen-bernini/tests/fixtures/vit_preprocess_golden.safetensors
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
from safetensors.torch import save_file
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor, smart_resize


# verbatim copy of data_utils.smart_video_nframes (importing the module pulls decord/torchvision).
def smart_video_nframes(total_frames, video_fps, fps=2.0, frame_factor=None,
                        min_frames=None, max_frames=None, add_one=False):
    nframes = total_frames / video_fps * fps
    if frame_factor is not None:
        nframes = math.floor(nframes / frame_factor) * frame_factor + int(add_one)
        nframes = max(nframes, frame_factor + int(add_one))
        if video_fps == fps:
            total_frames = math.floor(total_frames / frame_factor) * frame_factor + int(add_one)
    else:
        nframes = int(nframes + int(add_one))
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    if min_frames is not None:
        if frame_factor is not None:
            min_frames = math.ceil(min_frames / frame_factor) * frame_factor
        nframes = max(min_frames + int(add_one), nframes)
    while len(idx) < int(nframes):
        idx.append(idx[-1])
    if max_frames is not None:
        if frame_factor is not None:
            max_frames = math.floor(max_frames / frame_factor) * frame_factor
        nframes = min(max_frames + int(add_one), nframes)
    if len(idx) > int(nframes):
        idx = idx[: int(nframes)]
    return idx


# (total_frames, video_fps, fps, frame_factor, max_frames, add_one) — ViT (fps2/factor2/!add_one) +
# VAE (fps16/factor4/add_one -> 4k+1), short clips, and the video_fps==fps total-frames path.
NFRAMES_CASES = [
    (50, 25.0, 2.0, 2, 81, False),   # ViT
    (50, 25.0, 16.0, 4, 81, True),   # VAE -> 4k+1
    (8, 8.0, 2.0, 2, 81, False),     # short ViT
    (100, 30.0, 16.0, 4, 81, True),  # VAE longer
    (50, 2.0, 2.0, 2, 81, False),    # video_fps == fps branch
]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(
    REPO_ROOT, "mlx-gen-bernini", "tests", "fixtures", "vit_preprocess_golden.safetensors"
)

PATCH = 14
TEMPORAL = 2
MERGE = 2
FACTOR = PATCH * MERGE  # 28
IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]
MIN_PIXELS = 3136
MAX_PIXELS = 12845056

# smart_resize cases: identity, up-clamp (tiny), down-clamp (huge), banker's-round half.
RESIZE_CASES = [
    (56, 84),       # multiples of 28, mid-range -> identity
    (40, 40),       # below min_pixels -> up-clamp
    (4000, 3000),   # above max_pixels -> down-clamp
    (42, 70),       # 42/28=1.5 -> round-to-even=2 (56); 70/28=2.5 -> 2 (56)
    (98, 28),       # 98/28=3.5 -> 4 (112)
    (100, 200),     # generic
]


def main() -> None:
    out = {}

    # 1) smart_resize table (input hw, output hw_bar).
    inp = torch.tensor(RESIZE_CASES, dtype=torch.int32)
    res = torch.tensor(
        [list(smart_resize(h, w, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS))
         for (h, w) in RESIZE_CASES],
        dtype=torch.int32,
    )
    out["smart_resize.in"] = inp.contiguous()
    out["smart_resize.out"] = res.contiguous()

    # 2) packing + normalize, do_resize=False on a fixed uint8 image (dims multiples of 28).
    proc = Qwen2VLImageProcessor(
        do_resize=False, do_rescale=True, rescale_factor=1 / 255,
        do_normalize=True, image_mean=IMAGE_MEAN, image_std=IMAGE_STD,
        patch_size=PATCH, temporal_patch_size=TEMPORAL, merge_size=MERGE, do_convert_rgb=False,
    )
    rng = np.random.RandomState(0)
    H, W = 56, 84  # multiples of 28
    img = rng.randint(0, 256, size=(H, W, 3), dtype=np.uint8)  # HWC RGB
    feat = proc(images=[img], return_tensors="pt", do_resize=False)
    pixel_values = feat["pixel_values"].float()  # [seq, 1176]
    grid_thw = feat["image_grid_thw"].to(torch.int32)  # [1, 3]

    out["pack.image_hwc_u8"] = torch.from_numpy(img.astype(np.int32)).contiguous()  # [H,W,3]
    out["pack.pixel_values"] = pixel_values.contiguous()
    out["pack.grid_thw"] = grid_thw.contiguous()

    # 3) smart_video_nframes frame-index sampling.
    for i, (tf, vfps, fps, ff, mx, add) in enumerate(NFRAMES_CASES):
        idx = smart_video_nframes(tf, vfps, fps=fps, frame_factor=ff, max_frames=mx, add_one=add)
        out[f"nframes.{i}"] = torch.tensor(idx, dtype=torch.int32).contiguous()

    meta = {
        "patch": str(PATCH), "temporal": str(TEMPORAL), "merge": str(MERGE),
        "factor": str(FACTOR), "min_pixels": str(MIN_PIXELS), "max_pixels": str(MAX_PIXELS),
        "image_mean": ",".join(repr(x) for x in IMAGE_MEAN),
        "image_std": ",".join(repr(x) for x in IMAGE_STD),
        "pack_h": str(H), "pack_w": str(W),
        "nframes_cases": ";".join(",".join(str(int(x) if isinstance(x, bool) else x) for x in c) for c in NFRAMES_CASES),
    }
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    save_file(out, FIXTURE, metadata=meta)
    print(f"wrote {FIXTURE}  ({len(out)} tensors)")
    print(f"  smart_resize cases: {RESIZE_CASES} -> {res.tolist()}")
    print(f"  pack: img {(H, W, 3)} -> pixel_values {tuple(pixel_values.shape)} grid {grid_thw.tolist()}")


if __name__ == "__main__":
    main()
