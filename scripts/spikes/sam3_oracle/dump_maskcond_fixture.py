#!/usr/bin/env python
"""SAM3 **mask-conditioned (detection-seeded) frame decode** parity oracle — epic 4910, sc-4924
(Phase F2.5a-ii).

When the video pipeline seeds a new object from a detection mask, `_tracker_add_new_objects` runs the
tracker with `run_mem_encoder=True`, which calls `_run_single_frame_inference` with `mask_inputs` set →
`_use_mask_as_output` (modeling_sam3_tracker_video.py ~2136). That turns the binary detection mask into
high-res `±` logits (for the memory encoder) and runs the SAM decoder prompted with
`mask_embed(mask_downsample(mask))` on the **raw** image embedding (no `no_memory_embedding`) to get the
object pointer stored in the bank.

We wrap `tracker_model._use_mask_as_output` on a real 2-frame `Sam3VideoModel` PCS run and capture the
first call: inputs (backbone_features = raw pix_feat, high_res_features, mask_inputs) and outputs
(high_res_masks, object_pointer, object_score_logits, pred_masks).

The Rust test feeds the captured pix_feat + high-res features + mask into
`Sam3Tracker::decode_mask_conditioning_frame` and compares high_res + object_pointer + object_score
(cosine gate >0.9999; the low-res antialias downsample is not bit-reproduced — unused on this path).

Run:  /tmp/sam3ref/.venv/bin/python dump_maskcond_fixture.py
"""

import hashlib
import json
import os
import urllib.request
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import Sam3VideoModel, Sam3VideoProcessor

OUT = os.path.dirname(os.path.abspath(__file__))
MODEL = "facebook/sam3"
torch.manual_seed(0)

URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg"


def stats(t):
    t = t.detach().float().cpu()
    return {
        "shape": list(t.shape),
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "sha1_5dp": hashlib.sha1(np.ascontiguousarray(t.numpy().round(5).astype(np.float32)).tobytes()).hexdigest()[:16],
    }


def main():
    print("loading", MODEL)
    model = Sam3VideoModel.from_pretrained(MODEL, dtype=torch.float32).eval()
    processor = Sam3VideoProcessor.from_pretrained(MODEL)

    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        image = Image.open(BytesIO(r.read())).convert("RGB")
    video = [np.array(image), np.array(image)]

    session = processor.init_video_session(video=video, inference_device="cpu", dtype=torch.float32)
    processor.add_text_prompt(session, "person")

    tracker = model.tracker_model
    cap = {}
    orig = tracker._use_mask_as_output

    def wrapped(backbone_features, high_res_features, mask_inputs, *a, **k):
        out = orig(backbone_features, high_res_features, mask_inputs, *a, **k)
        if "out" not in cap:
            cap["pix_feat"] = backbone_features
            cap["high_res_features"] = high_res_features
            cap["mask_inputs"] = mask_inputs
            cap["out"] = out
        return out

    tracker._use_mask_as_output = wrapped
    try:
        with torch.no_grad():
            for _out in model.propagate_in_video_iterator(session):
                pass
    finally:
        tracker._use_mask_as_output = orig

    assert "out" in cap, "no _use_mask_as_output call captured"
    det = lambda t: t.detach().cpu().float().contiguous().clone()
    pix_feat = cap["pix_feat"]
    feat_s0, feat_s1 = cap["high_res_features"]
    mask_inputs = cap["mask_inputs"].float()
    out = cap["out"]
    print(
        f"  pix_feat {list(pix_feat.shape)} feat_s0 {list(feat_s0.shape)} feat_s1 {list(feat_s1.shape)} "
        f"mask {list(mask_inputs.shape)} (sum>0={float((mask_inputs>0).float().sum())})"
    )
    print(
        f"  high_res {list(out.high_res_masks.shape)} obj_ptr {list(out.object_pointer.shape)} "
        f"obj_score {float(out.object_score_logits.reshape(-1)[0]):.4f}"
    )

    manifest = {
        "model": MODEL,
        "image_url": URL,
        "stages": {
            "pix_feat": stats(pix_feat),
            "feat_s0": stats(feat_s0),
            "feat_s1": stats(feat_s1),
            "mask_inputs": stats(mask_inputs),
            "high_res_masks": stats(out.high_res_masks),
            "object_pointer": stats(out.object_pointer),
            "object_score_logits": stats(out.object_score_logits),
        },
    }
    with open(os.path.join(OUT, "maskcond_fixture_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    save_file(
        {
            "pix_feat": det(pix_feat),  # [1,256,72,72]
            "feat_s0": det(feat_s0),  # [1,32,288,288]
            "feat_s1": det(feat_s1),  # [1,64,144,144]
            "mask_inputs": det(mask_inputs),  # [1,1,288,288]
            "high_res_masks": det(out.high_res_masks),  # [1,1,1008,1008]
            "object_pointer": det(out.object_pointer),  # [1,1,256]
            "object_score_logits": det(out.object_score_logits),  # [1,1,1]
        },
        os.path.join(OUT, "maskcond_fixture.safetensors"),
    )
    print("wrote maskcond_fixture_manifest.json + .safetensors to", OUT)


if __name__ == "__main__":
    main()
