#!/usr/bin/env python3
"""sc-3070 — EVA02-CLIP-L-14-336 visual-tower parity golden (torch f32 reference).

Builds the *reference* EVA visual tower (the exact `create_model_and_transforms(...).visual` PuLID
uses), runs it in float32 on a fixed input, and dumps a single safetensors the Rust parity test
loads: the f32 mlx-named weights, the input, the 5 captured hidden states, the projected
`id_cond_vit`, and the deterministic RoPE buffers (for the weight-free RoPE-construction gate).

f32 (not the fp16/bf16 production dtype) is the numerically-correct ground truth; the Rust port is
gated near-bit in f32 and at the bf16 floor separately (repo convention).

xformers is intentionally absent in pulidenv -> `XFORMERS_IS_AVAILBLE=False` -> the reference runs the
explicit-softmax attention path (deterministic), which the MLX SDPA port matches.

Run:
    cd /Users/michael/Repos/SceneWorks/apps/worker/scene_worker/_vendor/pulid_flux && \
      PYTHONPATH=. /private/tmp/pulidenv/bin/python \
      /path/to/mlx-gen/tools/dump_eva_clip_golden.py
Output: tools/golden/eva_clip_golden.safetensors
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from convert_eva_clip import remap_visual  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "golden")
HIDDEN_BLOCKS = [4, 8, 12, 16, 20]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from eva_clip import create_model_and_transforms
    from eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import normalize, resize

    torch.manual_seed(0)
    model, _, _ = create_model_and_transforms("EVA02-CLIP-L-14-336", "eva_clip", force_custom_clip=True)
    visual = model.visual.float().eval()

    out = {}

    # --- f32 weights (same remap as the production converter) ---
    vis_sd = {k: v.float() for k, v in visual.state_dict().items()}
    for k, v in remap_visual(vis_sd).items():
        out[f"w.{k}"] = v.astype(np.float32)

    # --- deterministic RoPE buffers (weight-free construction gate) ---
    out["rope.freqs_cos"] = visual.rope.freqs_cos.detach().cpu().float().numpy().astype(np.float32)
    out["rope.freqs_sin"] = visual.rope.freqs_sin.detach().cpu().float().numpy().astype(np.float32)

    # --- encoder parity: fixed 336^2 input fed straight to the tower (NCHW for torch) ---
    g = torch.Generator().manual_seed(1234)
    enc_in = torch.randn(1, 3, 336, 336, generator=g, dtype=torch.float32)
    with torch.no_grad():
        id_cond_vit, hidden = visual(enc_in, return_all_features=False, return_hidden=True, shuffle=False)
    assert len(hidden) == len(HIDDEN_BLOCKS), (len(hidden), HIDDEN_BLOCKS)
    # dump input NHWC so the Rust patch-embed (channels-last conv) consumes it directly
    out["enc_in_nhwc"] = enc_in.permute(0, 2, 3, 1).contiguous().numpy().astype(np.float32)
    out["id_cond_vit"] = id_cond_vit.float().numpy().astype(np.float32)  # [1,768]
    for i, h in enumerate(hidden):
        out[f"hidden_{i}"] = h.float().numpy().astype(np.float32)  # [1,577,1024]

    # --- input-transform parity (sc-3073 nugget folded here): 512^2 [0,1] -> 336^2 resize+normalize ---
    g2 = torch.Generator().manual_seed(99)
    # smooth-ish [0,1] image (low-freq grid upsampled) to mimic face_features_image, not white noise
    coarse = torch.rand(1, 3, 32, 32, generator=g2)
    ffi = torch.nn.functional.interpolate(coarse, size=(512, 512), mode="bilinear", align_corners=False)
    ffi = ffi.clamp(0, 1).float()
    with torch.no_grad():
        resized = resize(ffi, visual.image_size, InterpolationMode.BICUBIC)
        transformed = normalize(resized, OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
    out["ffi_512_nhwc"] = ffi.permute(0, 2, 3, 1).contiguous().numpy().astype(np.float32)
    out["tf_resized_nhwc"] = resized.permute(0, 2, 3, 1).contiguous().numpy().astype(np.float32)
    out["tf_normalized_nhwc"] = transformed.permute(0, 2, 3, 1).contiguous().numpy().astype(np.float32)
    out["eva_mean"] = np.array(OPENAI_DATASET_MEAN, dtype=np.float32)
    out["eva_std"] = np.array(OPENAI_DATASET_STD, dtype=np.float32)
    out["image_size"] = np.array([visual.image_size], dtype=np.int32)

    from safetensors.numpy import save_file

    path = os.path.join(OUT_DIR, "eva_clip_golden.safetensors")
    save_file(out, path)
    n_w = sum(1 for k in out if k.startswith("w."))
    print(f"wrote {len(out)} tensors ({n_w} weights) -> {path}")
    print("id_cond_vit", out["id_cond_vit"].shape, "hidden", out["hidden_0"].shape, "x5")
    print("rope.freqs_cos", out["rope.freqs_cos"].shape)


if __name__ == "__main__":
    main()
