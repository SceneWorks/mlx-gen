"""sc-3708 — package the spike's real-photo baselines into a parity golden.

Bundles the spike sc-3635 artifacts (real RGB photos + box prompts + the PyTorch and ONNX baseline
masks — the quality source-of-truth) into one gitignored golden safetensors. The Rust
`tests/photo_parity.rs` runs the MLX `Sam2Segmenter` on the same photos+boxes and asserts the mask
IoU lands in the spike's ort-vs-PyTorch band (zidane ~0.99, bus ~0.93) — the engine-side GO gate.

No model is run here: it just repackages files already on disk from the spike
(`/tmp/sc3635/{zidane,bus}/`: `*.jpg`, `inputs.json`, `ref_pt_mask.png`, `mask_ortcpu.png`).

Run:
  ~/mlx-flux-venv/bin/python tools/dump_sam2_photo_golden.py
"""

from __future__ import annotations

import argparse
import json
import os

import mlx.core as mx
import numpy as np
from PIL import Image

IMAGES = ("zidane", "bus")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spike-dir", default="/tmp/sc3635")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "golden"))
    args = ap.parse_args()

    golden: dict[str, mx.array] = {}
    for name in IMAGES:
        d = os.path.join(args.spike_dir, name)
        rgb = np.asarray(
            Image.open(os.path.join(args.spike_dir, f"{name}.jpg")).convert("RGB"), dtype=np.uint8
        )
        meta = json.load(open(os.path.join(d, "inputs.json")))
        box = np.asarray(meta["box_orig"], dtype=np.float32)
        pt = (np.asarray(Image.open(os.path.join(d, "ref_pt_mask.png"))) > 127).astype(np.uint8)
        ort = (np.asarray(Image.open(os.path.join(d, "mask_ortcpu.png"))) > 127).astype(np.uint8)
        assert rgb.shape[:2] == pt.shape == ort.shape, f"{name} shape mismatch"
        golden[f"rgb_{name}"] = mx.array(rgb)
        golden[f"box_{name}"] = mx.array(box)
        golden[f"pt_{name}"] = mx.array(pt)
        golden[f"ort_{name}"] = mx.array(ort)
        print(f"[{name}] rgb={rgb.shape} box={box.tolist()} pt_fg={int(pt.sum())} ort_fg={int(ort.sum())}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "sam2_photo_golden.safetensors")
    mx.save_safetensors(out, golden, metadata={"format": "mlx", "source": "sc3635-spike"})
    print(f"[written] {out} ({len(golden)} tensors)")


if __name__ == "__main__":
    main()
