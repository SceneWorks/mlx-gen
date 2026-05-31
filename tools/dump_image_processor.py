"""Generate the Qwen2-VL image-processor parity fixture for the Rust port (sc-2341).

Ground truth from the fork's hand-rolled `QwenImageProcessor` (qwen_image_processor.py):
smart_resize → PIL BICUBIC resize → /255 → CLIP normalize → temporal-repeat → patchify
→ (N, 1176) pixel_values + (1,3) grid_thw.

Three deterministic synthetic images exercise the branches:
  - A: 56×84  → already multiples of 28, smart_resize is a no-op (isolates normalize+patchify
                from the interpolation — the EXACT path);
  - B: 200×150 → smart_resize downscales to 196×140 (PIL bicubic with antialiasing);
  - C: 20×20   → smart_resize upscales to 56×56 (PIL bicubic upscale).

We dump the input as a uint8 HWC array (not a PNG) so the Rust test feeds identical pixels
with no image-decode dependency or decode-parity variable.

Run from the mflux fork venv:
    cd ~/repos/mflux && uv run python ~/repos/mlx-gen/tools/dump_image_processor.py
"""

import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.models.qwen.tokenizer.qwen_image_processor import QwenImageProcessor

rng = np.random.default_rng(0)
proc = QwenImageProcessor()

CASES = {"a": (56, 84), "b": (200, 150), "c": (20, 20)}  # (H, W)

out = {}
for name, (h, w) in CASES.items():
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    pixel_values, grid_thw = proc.preprocess(img)
    out[f"{name}.input"] = mx.array(arr)  # uint8 (H, W, 3)
    out[f"{name}.pixel_values"] = mx.array(pixel_values.astype(np.float32))
    out[f"{name}.grid_thw"] = mx.array(grid_thw.astype(np.int32))
    print(f"{name}: in=({h},{w}) pixel_values={pixel_values.shape} grid_thw={grid_thw.tolist()}")

path = "/Users/michael/repos/mlx-gen/mlx-gen-qwen-image/tests/fixtures/qwen_image_processor.safetensors"
mx.save_safetensors(path, out)
print(f"wrote {path} ({len(out)} tensors)")
