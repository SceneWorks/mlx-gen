"""Dump PIL BICUBIC resize goldens so the Rust `resize_u8` can be regression-tested for bit-exact
PIL parity (sc-2465 — the fixed-point resampler that closed the Edit e2e gap). Two 512->384 cases:
a sawtooth gradient (maximizes the cliff disagreement that exposed the f64-vs-fixed-point bug) and a
smooth ramp. Image/weight-free.

cd ~/repos/mflux && uv run python /path/to/mlx-gen/tools/dump_pil_resize_golden.py
Output (gitignored): tools/golden/pil_resize_golden.safetensors
"""

import os

import mlx.core as mx
import numpy as np
from PIL import Image

saw = np.empty((512, 512, 3), np.uint8)
smo = np.empty((512, 512, 3), np.uint8)
for y in range(512):
    for x in range(512):
        b = (x + y) % 256
        saw[y, x] = [b, (b * 2) % 256, (b * 3) % 256]
        smo[y, x] = min((x + y) // 4, 255)

pil384 = np.asarray(Image.fromarray(saw).resize((384, 384), Image.BICUBIC), np.int32)
pil384_smooth = np.asarray(Image.fromarray(smo).resize((384, 384), Image.BICUBIC), np.int32)

golden_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(golden_dir, exist_ok=True)
path_out = os.path.join(golden_dir, "pil_resize_golden.safetensors")
mx.save_safetensors(path_out, {"pil384": mx.array(pil384), "pil384_smooth": mx.array(pil384_smooth)})
print(f"wrote {path_out}  saw row0 ch0 {pil384[0, :6, 0].tolist()}")
