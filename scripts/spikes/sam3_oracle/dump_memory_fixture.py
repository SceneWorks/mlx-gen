#!/usr/bin/env python
"""SAM3 **tracker memory encoder** parity oracle — epic 4910, sc-4924 (Phase F2).

Drives the public `transformers` SAM3 tracker memory encoder (`facebook/sam3` →
`Sam3VideoModel.tracker_model`, a `Sam3TrackerVideoModel`) on one real frame + box-prompt mask,
replicating `_encode_new_memory` (modeling_sam3_tracker_video.py ~2658):

    pix_feat       = current_vision_feats[-1].permute(1,2,0).view(B,256,72,72)   # raw 72² image emb
    high_res_masks = interp(low_res_decoder_mask -> 1008²)                       # image-res mask logits
    mask_for_mem   = interp(high_res_masks -> 1152², bilinear, align_corners=False, antialias=True)
                       -> sigmoid()  (or >0 binarize if is_mask_from_pts)
                       * sigmoid_scale_for_mem_enc (20) + sigmoid_bias_for_mem_enc (-10)
    feats, pos     = memory_encoder(pix_feat, mask_for_mem)                      # [1,64,72,72] each
    feats         += (1 - (obj_score>0)) * occlusion_spatial_embedding_parameter # occlusion add

NB: high_res_masks is 1008² and mask_mem is 1152² → this is **upsampling**, so PyTorch's
`antialias=True` is a documented no-op (antialias only affects downsampling). The Rust port replicates
it as a plain separable bilinear resize (align_corners=False).

Dumps the staged inputs + intermediates + outputs as the parity fixtures the Rust memory encoder
validates against (cosine gate). No MLX here.

Run:  /tmp/sam3ref/.venv/bin/python dump_memory_fixture.py
"""

import hashlib
import json
import os
import urllib.request
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file
from transformers import Sam3Processor, Sam3VideoModel

OUT = os.path.dirname(os.path.abspath(__file__))
MODEL = "facebook/sam3"
torch.manual_seed(0)

URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg"
BOX_1008 = [430.0, 90.0, 700.0, 980.0]  # same tall-person box as the F1 tracker fixture


def stats(t):
    t = t.detach().float().cpu()
    return {
        "shape": list(t.shape),
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "std": float(t.std()),
        "sha1_5dp": hashlib.sha1(np.ascontiguousarray(t.numpy().round(5)).tobytes()).hexdigest()[:16],
    }


def main():
    print("loading", MODEL)
    model = Sam3VideoModel.from_pretrained(MODEL, dtype=torch.float32).eval()
    tracker = model.tracker_model
    processor = Sam3Processor.from_pretrained(MODEL)
    cfg = tracker.config

    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        image = Image.open(BytesIO(r.read())).convert("RGB")
    W, H = image.size
    pixel_values = processor(images=image, text="person", return_tensors="pt")["pixel_values"]
    print(f"  image {W}x{H} -> pixel_values {list(pixel_values.shape)}")

    with torch.no_grad():
        # ---- shared PE backbone -> tracker features (no own vision encoder) -------------------
        vision_embeds = model.detector_model.vision_encoder(pixel_values)
        feats, _pos = model.get_vision_features_for_tracker(vision_embeds)  # [s0, s1, pix] HWxBxC
        sizes = tracker.backbone_feature_sizes  # [[288,288],[144,144],[72,72]]
        high_res = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(feats[:-1], sizes[:-1])
        ]
        B, C = feats[-1].size(1), feats[-1].size(2)
        h, w = sizes[-1]
        # raw 72² image embedding (NO no_memory_embedding here — that's only the no-mem cond path)
        pix_feat = feats[-1].permute(1, 2, 0).view(B, C, h, w)  # [1,256,72,72]
        pix_with_nomem = (feats[-1] + tracker.no_memory_embedding).permute(1, 2, 0).view(B, C, h, w)
        image_pe = tracker.get_image_wide_positional_embeddings()

        # ---- single-frame box decode -> a real high-res mask logit map ------------------------
        box = torch.tensor(BOX_1008, dtype=torch.float32).view(1, 1, 4)
        sparse, dense = tracker.prompt_encoder(
            input_points=None, input_labels=None, input_boxes=box, input_masks=None
        )
        low_res, iou, _tok, obj = tracker.mask_decoder(
            image_embeddings=pix_with_nomem,
            image_positional_embeddings=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
            high_resolution_features=high_res,
        )
        iou_flat = iou.reshape(-1)
        best = int(torch.argmax(iou_flat).item())
        low_res_best = low_res.reshape(low_res.shape[-3], low_res.shape[-2], low_res.shape[-1])[best]
        # high-res masks at image resolution (1008²), the `_encode_new_memory` input.
        pred_masks_high_res = F.interpolate(
            low_res_best.view(1, 1, *low_res_best.shape),
            size=(cfg.image_size, cfg.image_size),
            mode="bilinear",
            align_corners=False,
        )  # [1,1,1008,1008]
        object_score_logits = obj.reshape(1, 1)
        print(f"  best={best} iou={iou_flat[best]:.4f} obj={float(object_score_logits):.4f}  "
              f"low_res {list(low_res_best.shape)} high_res {list(pred_masks_high_res.shape)}")

        # ---- replicate _encode_new_memory (f32, capturing intermediates) ----------------------
        msz_h, msz_w = tracker.prompt_encoder.mask_input_size  # (288, 288)
        mem_h, mem_w = msz_h * 4, msz_w * 4  # (1152, 1152)
        resized = F.interpolate(
            pred_masks_high_res.float(),
            size=(mem_h, mem_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,  # no-op on upsampling; kept to match reference call exactly
        )
        scale = cfg.sigmoid_scale_for_mem_enc
        bias = cfg.sigmoid_bias_for_mem_enc
        # is_mask_from_pts=False -> sigmoid path (the propagation case); also dump the binarize path.
        mask_for_mem = torch.sigmoid(resized) * scale + bias
        mask_for_mem_bin = (resized > 0).float() * scale + bias

        feats_raw, pos_enc = tracker.memory_encoder(pix_feat, mask_for_mem)  # [1,64,72,72] each

        # occlusion: add the spatial embedding where the object is predicted absent (obj_score<=0)
        occ = tracker.occlusion_spatial_embedding_parameter  # [1,64]
        is_obj_appearing = (object_score_logits > 0).float()  # [1,1]
        feats_final = feats_raw + (1 - is_obj_appearing[..., None]) * occ[..., None, None].expand(
            *feats_raw.shape
        )

    print(f"  mask_for_mem {list(mask_for_mem.shape)}  feats {list(feats_raw.shape)}  "
          f"pos {list(pos_enc.shape)}  occlusion_active={float(1 - is_obj_appearing):.0f}")

    manifest = {
        "model": MODEL,
        "image_url": URL,
        "box_1008": BOX_1008,
        "best_index": best,
        "object_score": float(object_score_logits),
        "is_obj_appearing": float(is_obj_appearing),
        "sigmoid_scale_for_mem_enc": float(scale),
        "sigmoid_bias_for_mem_enc": float(bias),
        "mask_input_size": [msz_h, msz_w],
        "mask_mem_size": [mem_h, mem_w],
        "image_size": int(cfg.image_size),
        "stages": {
            "pix_feat": stats(pix_feat),
            "pred_masks_high_res": stats(pred_masks_high_res),
            "mask_for_mem": stats(mask_for_mem),
            "mask_for_mem_bin": stats(mask_for_mem_bin),
            "maskmem_features_raw": stats(feats_raw),
            "maskmem_pos_enc": stats(pos_enc),
            "maskmem_features_final": stats(feats_final),
        },
    }
    with open(os.path.join(OUT, "memory_fixture_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    det = lambda t: t.detach().cpu().float().contiguous().clone()
    save_file(
        {
            "pix_feat": det(pix_feat),  # [1,256,72,72] NCHW
            "pred_masks_high_res": det(pred_masks_high_res),  # [1,1,1008,1008]
            "object_score_logits": det(object_score_logits),  # [1,1]
            "mask_for_mem": det(mask_for_mem),  # [1,1,1152,1152] sigmoid path
            "mask_for_mem_bin": det(mask_for_mem_bin),  # [1,1,1152,1152] binarize path
            "maskmem_features_raw": det(feats_raw),  # [1,64,72,72]
            "maskmem_pos_enc": det(pos_enc),  # [1,64,72,72]
            "maskmem_features_final": det(feats_final),  # [1,64,72,72] post-occlusion
        },
        os.path.join(OUT, "memory_fixture.safetensors"),
    )
    print("wrote memory_fixture_manifest.json + .safetensors to", OUT)


if __name__ == "__main__":
    main()
