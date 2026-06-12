#!/usr/bin/env python
"""SAM3 → MLX parity oracle (spike sc-4911, epic 4910).

Runs the public `transformers` SAM3 reference (facebook/sam3) on real photos with a
text concept and dumps the full numerical contract as fixtures for the MLX port:
preprocessing, key staged intermediates (vision / text / detr-enc / detr-dec), the
raw detector outputs, and the post-processed instances. No MLX here — this is the
torch oracle the Rust port validates against (no Python at validation, like sc-3635).

Run with the spike venv:
    /tmp/sam3ref/.venv/bin/python run_oracle.py
"""

import hashlib
import json
import os
import urllib.request
from io import BytesIO

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import Sam3Model, Sam3Processor

OUT = os.path.dirname(os.path.abspath(__file__))
MODEL = "facebook/sam3"
torch.manual_seed(0)

# (label, url, text concept) — sam-car is the model-card example; the COCO shot exercises multi-instance "person".
CASES = [
    ("car", "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-car.png", "car"),
    # zidane (2 people) + bus (several people) — same images the SAM2 spike sc-3635 used
    ("zidane", "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg", "person"),
    ("bus", "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg", "person"),
]


def load_image(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return Image.open(BytesIO(r.read())).convert("RGB")


def stats(t):
    t = t.detach().float().cpu()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "std": float(t.std()),
        # sha of rounded bytes → stable cross-run identity check for the port
        "sha1_5dp": hashlib.sha1(np.ascontiguousarray(t.numpy().round(5)).tobytes()).hexdigest()[:16],
    }


def main():
    print("loading", MODEL)
    model = Sam3Model.from_pretrained(MODEL, dtype=torch.float32).eval()
    processor = Sam3Processor.from_pretrained(MODEL)

    # capture staged intermediates via hooks
    caught = {}
    h = []
    h.append(model.vision_encoder.register_forward_hook(lambda m, i, o: caught.__setitem__("vision", o)))
    h.append(model.detr_encoder.register_forward_hook(lambda m, i, o: caught.__setitem__("detr_encoder", o)))
    h.append(model.detr_decoder.register_forward_hook(lambda m, i, o: caught.__setitem__("detr_decoder", o)))

    manifest = {"model": MODEL, "cases": {}}

    for label, url, text in CASES:
        print(f"\n=== case {label!r}  text={text!r} ===")
        try:
            image = load_image(url)
        except Exception as e:
            print(f"  SKIP (image fetch failed: {e})")
            manifest["cases"][label] = {"error": f"image fetch failed: {e}"}
            continue
        W, H = image.size
        inputs = processor(images=image, text=text, return_tensors="pt")
        caught.clear()
        with torch.no_grad():
            out = model(**inputs)

        # post-process to final instances
        results = processor.image_processor.post_process_instance_segmentation(
            out, threshold=0.5, mask_threshold=0.5, target_sizes=[(H, W)]
        )[0]
        n = int(len(results["scores"]))
        combined = (out.pred_logits.sigmoid() * out.presence_logits.sigmoid())[0]
        topk = combined.topk(min(5, combined.numel())).values.tolist()
        print(f"  image {W}x{H} -> {n} instances @0.5; top5 raw scores={[round(x,3) for x in topk]}; presence={out.presence_logits.sigmoid().item():.3f}")

        case = {
            "image_size_wh": [W, H],
            "text": text,
            "input_ids": inputs["input_ids"][0].tolist(),
            "attention_mask": inputs["attention_mask"][0].tolist(),
            "pixel_values": stats(inputs["pixel_values"]),
            "raw": {
                "pred_logits": stats(out.pred_logits),
                "pred_boxes": stats(out.pred_boxes),
                "presence_logits": stats(out.presence_logits),
                "presence_logits_value": out.presence_logits.flatten().tolist(),
                "pred_masks": stats(out.pred_masks),
                "semantic_seg": stats(out.semantic_seg),
            },
            "intermediates": {},
            "num_instances": n,
            "instance_scores": results["scores"].tolist(),
            "instance_boxes_xyxy": results["boxes"].tolist(),
        }
        # staged intermediates (stats only; full tensors saved to npz below)
        v = caught.get("vision")
        if v is not None and hasattr(v, "fpn_hidden_states"):
            case["intermediates"]["fpn_hidden_states"] = [stats(t) for t in v.fpn_hidden_states]
        for k in ("detr_encoder", "detr_decoder"):
            o = caught.get(k)
            if o is not None and hasattr(o, "last_hidden_state"):
                case["intermediates"][k + ".last_hidden_state"] = stats(o.last_hidden_state)

        manifest["cases"][label] = case

        # full-fidelity tensors for the port to diff against
        npz = {
            "pixel_values": inputs["pixel_values"].numpy(),
            "input_ids": inputs["input_ids"].numpy(),
            "pred_logits": out.pred_logits.numpy(),
            "pred_boxes": out.pred_boxes.numpy(),
            "presence_logits": out.presence_logits.numpy(),
            "semantic_seg": out.semantic_seg.float().numpy(),
        }
        if n > 0:
            npz["instance_masks"] = results["masks"].cpu().numpy().astype(np.uint8)
        np.savez_compressed(os.path.join(OUT, f"fixture_{label}.npz"), **npz)

        # overlay PNG sanity render
        ov = image.convert("RGBA")
        dr = ImageDraw.Draw(ov)
        for b in results["boxes"].tolist():
            dr.rectangle(b, outline=(255, 0, 0, 255), width=3)
        ov.convert("RGB").save(os.path.join(OUT, f"overlay_{label}.png"))

    for x in h:
        x.remove()
    with open(os.path.join(OUT, "oracle_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("\nwrote oracle_manifest.json + fixture_*.npz + overlay_*.png to", OUT)


if __name__ == "__main__":
    main()
