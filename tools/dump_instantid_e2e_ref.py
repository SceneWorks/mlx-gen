#!/usr/bin/env python
"""Reference face image for the InstantID e2e identity test (sc-3115).

Dumps a single-face reference photo (insightface's `Tom_Hanks_54745.png`) as raw RGB so the Rust
e2e test can run the native face stack (SCRFD detect → ArcFace embed) on it, generate, then re-detect
the output and measure the ArcFace-cosine identity preservation — all in Rust, no torch at test time.

Run from a venv with PIL + safetensors (e.g. the dwpose-spike venv):
    ~/.dwpose-spike/venv/bin/python ~/Repos/mlx-gen/tools/dump_instantid_e2e_ref.py
"""
import os
from pathlib import Path

import insightface
import numpy as np
import PIL.Image
from insightface.app import FaceAnalysis
from safetensors.numpy import save_file

OUT = Path(__file__).resolve().parent / "golden" / "instantid_e2e_ref.safetensors"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # t1.jpg — the canonical insightface test image (a real scene). We crop the largest face with
    # generous margin → a single, large, centered face WITH context: SCRFD detects it reliably, and
    # the generated portrait (driven by its kps) has a large detectable face for the identity metric.
    # (The whole group photo gives a small off-center face; the aligned 112² Tom_Hanks crop fills the
    # frame with no margin and isn't detectable.)
    img_path = os.path.join(
        os.path.dirname(insightface.__file__), "data", "images", "t1.jpg"
    )
    full = PIL.Image.open(img_path).convert("RGB")
    fw, fh = full.size

    app = FaceAnalysis(name="antelopev2", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    faces = app.get(np.asarray(full)[:, :, ::-1])  # BGR for insightface
    assert faces, "no face detected in t1.jpg"
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    x1, y1, x2, y2 = face.bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * 2.2  # margin around the face
    half = side / 2
    l, t = int(round(cx - half)), int(round(cy - half))
    r, b = int(round(cx + half)), int(round(cy + half))
    l, t = max(0, l), max(0, t)
    r, b = min(fw, r), min(fh, b)
    img = full.crop((l, t, r, b))
    arr = np.asarray(img, dtype=np.uint8)  # [h, w, 3]
    h, w = arr.shape[:2]
    save_file(
        {
            "ref_img": np.ascontiguousarray(arr),
            "ref_wh": np.array([w, h], dtype=np.int32),
        },
        str(OUT),
    )
    print(f"wrote {OUT}")
    print(f"  ref {w}x{h} from {img_path}")


if __name__ == "__main__":
    main()
