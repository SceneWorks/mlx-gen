"""Tiny-config parity golden for the FLUX.2-dev **Pixtral vision tower + Mistral3 projector** (sc-5918).

dev's `text_encoder` is a `Mistral3ForConditionalGeneration`; for edit/reference conditioning a
Pixtral ViT encodes reference images and a projector maps the features into the Mistral token space.
The frozen mflux fork has neither, so the reference here is **transformers' PyTorch
`PixtralVisionModel` + `Mistral3MultiModalProjector`** (the authoritative arch). A tiny synthetic
config keeps the committed fixture small and CI fast while exercising every path the port adds:

  * bias-less split q/k/v/o + RMSNorm + SwiGLU, block-diagonal attention,
  * the 2-D Pixtral RoPE (θ=10000, `rotate_half`), and
  * the projector's `norm → 2×2 patch-merge (unfold) → linear_1 → gelu → linear_2`.

Random weights, f32 throughout (the Rust port runs f32 activations). Run from a venv with
transformers + torch:

    ~/mlx-flux-venv/bin/python ~/Repos/mlx-gen/tools/dump_flux2_dev_pixtral_vision_golden.py
"""

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import Mistral3Config, MistralConfig, PixtralVisionConfig
from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector
from transformers.models.pixtral.modeling_pixtral import PixtralVisionModel

from _paths import fixture

torch.manual_seed(0)

# Tiny dims. head_dim = V_HIDDEN/V_HEADS = 8; IMG/PATCH = 16 ≥ both patch-grid sides (4, 6); the
# patch grid (4×6) is divisible by spatial_merge 2 → 2×3 merged tokens. T_HIDDEN is the projector
# output width (= the Mistral input-embed dim it scatters into).
V_HIDDEN, V_LAYERS, V_HEADS, V_INTER, PATCH, IMG = 32, 2, 4, 64, 2, 32
T_HIDDEN = 40
SPATIAL_MERGE = 2
IMG_H, IMG_W = 8, 12  # → patch grid 4×6

vcfg = PixtralVisionConfig(
    hidden_size=V_HIDDEN,
    num_hidden_layers=V_LAYERS,
    num_attention_heads=V_HEADS,
    intermediate_size=V_INTER,
    patch_size=PATCH,
    image_size=IMG,
    num_channels=3,
    rope_theta=10000.0,
    hidden_act="silu",
    attention_dropout=0.0,
)
tcfg = MistralConfig(hidden_size=T_HIDDEN, rms_norm_eps=1e-5)
mcfg = Mistral3Config(
    vision_config=vcfg.to_dict(),
    text_config=tcfg.to_dict(),
    image_token_index=10,
    spatial_merge_size=SPATIAL_MERGE,
    projector_hidden_act="gelu",
    multimodal_projector_bias=False,
    vision_feature_layer=-1,
)

vision = PixtralVisionModel(vcfg).eval().to(torch.float32)
projector = Mistral3MultiModalProjector(mcfg).eval().to(torch.float32)


def randomize(model):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            else:
                p.copy_(0.1 * torch.randn_like(p))


randomize(vision)
randomize(projector)

# Single reference image. NCHW for the torch reference; NHWC for the Rust patch-Conv2d.
pixel_values = 0.5 * torch.randn(1, 3, IMG_H, IMG_W, dtype=torch.float32)
image_sizes = torch.tensor([[IMG_H, IMG_W]], dtype=torch.long)

with torch.no_grad():
    vout = vision(pixel_values=pixel_values, image_sizes=image_sizes)
    image_features = vout.last_hidden_state.squeeze(0)  # [gh·gw, V_HIDDEN]
    projected = projector(image_features, image_sizes)  # [(gh/2)·(gw/2), T_HIDDEN]

gh, gw = IMG_H // PATCH, IMG_W // PATCH

tensors = {}
for prefix, model in (("vision_tower", vision), ("multi_modal_projector", projector)):
    for k, v in model.state_dict().items():
        if "inv_freq" in k or "rotary" in k or "positional_embedding" in k:
            continue  # RoPE recomputed in Rust
        tensors[f"{prefix}.{k}"] = v.detach().cpu().numpy().astype(np.float32)

# NHWC pixel values + the patch grid the Rust forward needs.
tensors["pixel_values"] = pixel_values.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32)
tensors["grid_hw"] = np.array([gh, gw], dtype=np.int32)
tensors["image_features"] = image_features.detach().cpu().numpy().astype(np.float32)
tensors["projected"] = projected.detach().cpu().numpy().astype(np.float32)

path = fixture("mlx-gen-flux2/tests/fixtures/pixtral_vision_golden.safetensors")
save_file(tensors, path)
print(f"wrote {path} ({len(tensors)} tensors)")
print(f"  patch grid: {gh}x{gw} = {gh * gw} patches → {(gh // 2) * (gw // 2)} merged tokens")
print(f"  image_features: {tuple(image_features.shape)}  (expect ({gh * gw}, {V_HIDDEN}))")
print(f"  projected: {tuple(projected.shape)}  (expect ({(gh // 2) * (gw // 2)}, {T_HIDDEN}))")
weight_keys = sorted(k for k in tensors if "." in k)
print(f"  weight keys ({len(weight_keys)}): {weight_keys}")
