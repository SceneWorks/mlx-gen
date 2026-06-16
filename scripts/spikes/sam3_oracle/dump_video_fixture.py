#!/usr/bin/env python
"""SAM3 **end-to-end multi-object video PCS** parity oracle — epic 4910, sc-4924 (Phase F2.6).

Drives the full `Sam3VideoModel` on a short clip (init_video_session + add_text_prompt +
propagate_in_video_iterator) and dumps, per frame, the per-`obj_id` low-res (288²) mask logits +
object id list, plus the preprocessed `pixel_values` and the tokenized prompt — everything the Rust
`Sam3VideoModel::propagate` pipeline needs to reproduce the run.

The Rust test feeds the captured frames + input_ids into the port and compares per-frame
per-`obj_id` masks (cosine) and the object-id sets.

Run:  /tmp/sam3ref/.venv/bin/python dump_video_fixture.py

sc-4995: the `kernels` cv-utils ops (`generic_nms` / `cc_2d`) are GPU-only and unavailable on this
Mac, so the stock reference runs in its no-`kernels` fallback (detection NMS off, hole-fill off). To
produce a fixture that reflects the *kernels-enabled* mask quality the Rust port replicates, set both
`SAM3_EMULATE_NMS=1` and `SAM3_EMULATE_HOLEFILL=1` to install CPU emulations of `generic_nms` and the
`cc_2d` connected-components before the run. Keep them set to match the Rust port's enabled
post-processing, otherwise `video_parity` will diff the port against a fixture that skipped it.
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
NUM_FRAMES = 8  # exercises full memory bank (num_maskmem 7) + object-pointer accumulation; tight-parity horizon (long-horizon is cross-backend-chaos-limited, see story)


def stats(t):
    t = t.detach().float().cpu()
    return {
        "shape": list(t.shape),
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "sha1_5dp": hashlib.sha1(np.ascontiguousarray(t.numpy().round(5).astype(np.float32)).tobytes()).hexdigest()[:16],
    }


def install_kernel_emulation():
    """sc-4995: install CPU emulations of the GPU-only `kernels` cv-utils ops, gated by env, so the
    fixture reflects the kernels-enabled behavior the Rust port replicates.

    SAM3_EMULATE_NMS=1 → emulate `generic_nms` with Meta's reference `generic_nms_cpu` greedy pass
    (descending score; suppress IoU **>** threshold; ties → ascending index via a stable sort, which
    the Rust `nms_dedup` matches). The detector's `det_nms_thresh` gate (> 0) is already satisfied by
    the default config, so patching the module-global `nms_masks` is enough to turn dedup on.

    SAM3_EMULATE_HOLEFILL=1 → emulate the GPU-only `cc_2d` connected-components with Meta's reference
    CPU path (`skimage.measure.label`, 8-connectivity) by patching `_get_connected_components_with_padding`.
    The stock `fill_holes_in_mask_scores` logic is left intact — only the kernel-dependent CC primitive
    is replaced — so hole-fill (`fill_hole_area`) turns on exactly as the Rust `fill_holes_in_mask`.
    """
    import transformers.models.sam3_video.modeling_sam3_video as m

    if os.environ.get("SAM3_EMULATE_NMS") == "1":

        def _generic_nms_cpu(ious, scores, iou_threshold):
            ious_np = ious.float().cpu().numpy()
            scores_np = scores.float().cpu().numpy()
            order = np.argsort(-scores_np, kind="stable")  # descending; ties → ascending index
            kept = []
            while order.size > 0:
                i = int(order[0])
                kept.append(i)
                rest = order[1:]
                order = rest[ious_np[i, rest] <= iou_threshold]
            return torch.tensor(kept, dtype=torch.int64)

        def _nms_masks(pred_probs, pred_masks, prob_threshold, iou_threshold):
            is_valid = pred_probs > prob_threshold
            probs = pred_probs[is_valid]
            masks_binary = pred_masks[is_valid] > 0
            if probs.numel() == 0:
                return is_valid
            ious = m.mask_iou(masks_binary, masks_binary)
            kept_inds = _generic_nms_cpu(ious, probs, iou_threshold)
            valid_inds = torch.where(is_valid, is_valid.cumsum(dim=0) - 1, -1)
            return torch.isin(valid_inds, kept_inds)

        m.nms_masks = _nms_masks
        print("  [emulation] NMS = generic_nms_cpu (SAM3_EMULATE_NMS=1)")

    if os.environ.get("SAM3_EMULATE_HOLEFILL") == "1":
        from skimage.measure import label as sk_label

        def _ccwp(mask):
            # mirror Meta's connected_components_cpu_single: skimage 8-connected labels + per-pixel
            # component size (foreground only; background pixels get count 0). mask is (B,1,H,W).
            mu = mask.to(torch.uint8)
            b, _, h, w = mu.shape
            arr = mu[:, 0].cpu().numpy()
            labels = np.zeros((b, h, w), dtype=np.int32)
            counts = np.zeros((b, h, w), dtype=np.int32)
            for i in range(b):
                lab, num = sk_label(arr[i], return_num=True)  # connectivity=2 (8-conn) by default
                labels[i] = lab.astype(np.int32)
                if num > 0:
                    sizes = np.bincount(lab.ravel())  # sizes[0] = background; sizes[k] = comp k area
                    cnt = sizes[lab].astype(np.int32)
                    cnt[lab == 0] = 0  # background pixels keep count 0 (reference fills fg only)
                    counts[i] = cnt
            labels_t = torch.from_numpy(labels).unsqueeze(1).to(mask.device)
            counts_t = torch.from_numpy(counts).unsqueeze(1).to(mask.device)
            return labels_t, counts_t

        m._get_connected_components_with_padding = _ccwp
        print("  [emulation] hole-fill CC = skimage 8-conn (SAM3_EMULATE_HOLEFILL=1)")


def main():
    print("loading", MODEL)
    model = Sam3VideoModel.from_pretrained(MODEL, dtype=torch.float32).eval()
    processor = Sam3VideoProcessor.from_pretrained(MODEL)
    install_kernel_emulation()

    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        image = Image.open(BytesIO(r.read())).convert("RGB")
    video = [np.array(image) for _ in range(NUM_FRAMES)]

    session = processor.init_video_session(video=video, inference_device="cpu", dtype=torch.float32)
    processor.add_text_prompt(session, "person")

    det = lambda t: t.detach().cpu().float().contiguous().clone()
    tensors = {}
    # preprocessed frames + prompt tokens
    for f in range(NUM_FRAMES):
        tensors[f"frame_{f}"] = det(session.get_frame(f).unsqueeze(0))  # [1,3,1008,1008]
    input_ids = session.prompt_input_ids[0]
    attn = session.prompt_attention_masks[0]
    tensors["input_ids"] = input_ids.detach().cpu().to(torch.int64).contiguous().clone()  # [1,32]
    tensors["attention_mask"] = attn.detach().cpu().to(torch.int64).contiguous().clone()  # [1,32]
    tensors["num_frames"] = torch.tensor([NUM_FRAMES], dtype=torch.int64)

    per_frame = []
    with torch.no_grad():
        for out in model.propagate_in_video_iterator(session):
            f = out.frame_idx
            obj_ids = list(out.object_ids)
            # stack masks in object-id order → [num_obj, 288, 288] logits
            masks = [out.obj_id_to_mask[o].reshape(288, 288) for o in obj_ids]
            stacked = torch.stack(masks, 0) if masks else torch.zeros(0, 288, 288)
            tensors[f"masks_{f}"] = det(stacked)
            tensors[f"obj_ids_{f}"] = torch.tensor(obj_ids, dtype=torch.int64)
            per_frame.append({"frame": f, "obj_ids": obj_ids, "num_obj": len(obj_ids)})
            print(f"  frame {f}: obj_ids={obj_ids}")

    manifest = {
        "model": MODEL,
        "image_url": URL,
        "num_frames": NUM_FRAMES,
        "input_ids": tensors["input_ids"].tolist(),
        "attention_mask": tensors["attention_mask"].tolist(),
        "frames": per_frame,
        "frame_stats": {f"masks_{p['frame']}": stats(tensors[f"masks_{p['frame']}"]) for p in per_frame},
    }
    with open(os.path.join(OUT, "video_fixture_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    save_file(tensors, os.path.join(OUT, "video_fixture.safetensors"))
    print("wrote video_fixture_manifest.json + .safetensors to", OUT)


if __name__ == "__main__":
    main()
