"""Parity fixture for the Chroma DiT (epic 3531) — a TINY synthetic config so it commits cheaply
and CI stays fast, while exercising every Chroma-specific path: the `ChromaApproximator` /
`ChromaCombinedTimestepTextProjEmbeddings` modulation generator, the pruned-adaLN slice offsets,
the double (joint img+txt) + single blocks, FluxPosEmbed RoPE, and the pruned `norm_out`.

Chroma has no mflux port, so the reference is the **torch `diffusers`** `ChromaTransformer2DModel`
(0.39-dev). Random weights, f32 throughout (the Rust port runs f32 activations for the transformer).

Run from the SceneWorks torch venv (which has diffusers + ChromaPipeline):
    "/Users/michael/Library/Application Support/SceneWorks/python/venv/bin/python" \
        tools/dump_chroma_golden.py

Outputs (consumed by mlx-gen-chroma/tests):
    mlx-gen-chroma/tests/fixtures/chroma_tiny_weights.safetensors   # model state_dict (diffusers keys)
    mlx-gen-chroma/tests/fixtures/chroma_tiny_io.safetensors        # inputs + pooled_temb + output
"""

from __future__ import annotations

import numpy as np
import torch
from safetensors.torch import save_file

from diffusers.models.transformers.transformer_chroma import ChromaTransformer2DModel

from _paths import fixture  # noqa: E402

torch.manual_seed(0)

# --- tiny config: inner_dim = 2*8 = 16; mod out_dim = 3*1 + 2*6*1 + 2 = 17 ---
IN_CH, HEADS, HEAD_DIM, JOINT = 4, 2, 8, 12
NUM_LAYERS, NUM_SINGLE = 1, 1
m = ChromaTransformer2DModel(
    patch_size=1,
    in_channels=IN_CH,
    num_layers=NUM_LAYERS,
    num_single_layers=NUM_SINGLE,
    attention_head_dim=HEAD_DIM,
    num_attention_heads=HEADS,
    joint_attention_dim=JOINT,
    axes_dims_rope=(2, 2, 4),
    approximator_num_channels=8,
    approximator_hidden_dim=16,
    approximator_layers=2,
).to(torch.float32)
m.eval()

# Re-randomize with a controlled scale so norms/affines behave (mirrors the flux2 golden script).
with torch.no_grad():
    for k, v in m.named_parameters():
        if any(t in k for t in ("norm_q", "norm_k", "norm_added", "norms.")):
            v.copy_(1.0 + 0.1 * torch.randn_like(v))
        else:
            v.copy_(0.1 * torch.randn_like(v))

# --- inputs: seq_img = 4 (2x2 grid), seq_txt = 3, with an attention mask that drops the last txt token ---
SEQ_IMG, SEQ_TXT = 4, 3
hidden = 0.5 * torch.randn(1, SEQ_IMG, IN_CH)
encoder = 0.5 * torch.randn(1, SEQ_TXT, JOINT)
timestep = torch.tensor([0.7], dtype=torch.float32)  # scaled *1000 inside forward
img_ids = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]], dtype=torch.float32)
txt_ids = torch.zeros(SEQ_TXT, 3, dtype=torch.float32)
# MMDiT mask over the full [txt|img] sequence. Exercise a real 0 to lock masking parity.
attention_mask = torch.ones(1, SEQ_TXT + SEQ_IMG, dtype=torch.float32)
attention_mask[0, SEQ_TXT - 1] = 0.0

with torch.no_grad():
    # Intermediate: the distilled-guidance modulation tensor (sc-3836 parity target).
    input_vec = m.time_text_embed(timestep.to(torch.float32) * 1000)
    pooled_temb = m.distilled_guidance_layer(input_vec)

    out = m(
        hidden_states=hidden,
        encoder_hidden_states=encoder,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )[0]

io = {
    "hidden": hidden,
    "encoder": encoder,
    "timestep": timestep,
    "img_ids": img_ids,
    "txt_ids": txt_ids,
    "attention_mask": attention_mask,
    "input_vec": input_vec,
    "pooled_temb": pooled_temb,
    "output": out,
}
save_file({k: v.contiguous() for k, v in m.state_dict().items()},
          fixture("mlx-gen-chroma/tests/fixtures/chroma_tiny_weights.safetensors"))
save_file({k: v.contiguous() for k, v in io.items()},
          fixture("mlx-gen-chroma/tests/fixtures/chroma_tiny_io.safetensors"))

print("inner_dim", m.inner_dim)
print("pooled_temb", tuple(pooled_temb.shape), "output", tuple(out.shape))
print("wrote chroma_tiny_weights.safetensors + chroma_tiny_io.safetensors")
