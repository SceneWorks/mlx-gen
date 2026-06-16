"""Tiny-config parity golden for the FLUX.2-dev **Mistral** text encoder (sc-5915).

The dev `text_encoder` is a `Mistral3ForConditionalGeneration`; the T2I path consumes only its
Mistral language tower. The frozen mflux fork has no Mistral encoder, so the reference here is
**transformers' PyTorch `MistralModel`** (the authoritative arch). A tiny synthetic config keeps the
committed fixture small and CI fast while exercising every code path the dev encoder adds over the
already-proven klein Qwen3 path:

  * **no per-head q/k-norm** (Mistral's delta from Qwen3),
  * HF half-split RoPE at the dev theta (1e9),
  * the causal + padding mask,
  * `hidden_size != num_heads * head_dim` (head_dim override — dev is 5120 vs 32*128=4096), and
  * the multi-layer hidden-state concatenation (embed / layer-0 / layer-1).

Random weights, f32 throughout (the Rust port runs f32 activations). Run from any venv with
transformers + torch:

    ~/repos/mflux/.venv/bin/python ~/Repos/mlx-gen/tools/dump_flux2_te_dev_golden.py
"""

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import MistralConfig
from transformers.models.mistral.modeling_mistral import MistralModel

from _paths import fixture

torch.manual_seed(0)

# Tiny config. hidden(80) != num_heads*head_dim(64) mirrors dev's head_dim override (5120 vs 4096);
# theta/eps are the dev values. 4 layers, picks (0,1,2) = embed / layer-0 / layer-1 — all strictly
# interior (< num_hidden_layers), matching the real dev usage `hidden_states[10,20,30]` (all < 40,
# all RAW). HF applies the final RMSNorm only to `hidden_states[num_layers]`, which we never pick.
VOCAB, HIDDEN, LAYERS, HEADS, KV, INTER, HEAD_DIM = 64, 80, 4, 4, 2, 128, 16
cfg = MistralConfig(
    vocab_size=VOCAB,
    hidden_size=HIDDEN,
    num_hidden_layers=LAYERS,
    num_attention_heads=HEADS,
    num_key_value_heads=KV,
    intermediate_size=INTER,
    head_dim=HEAD_DIM,
    rope_theta=1_000_000_000.0,
    rms_norm_eps=1e-5,
    hidden_act="silu",
    attention_bias=False,
    sliding_window=None,
    max_position_embeddings=512,
)
model = MistralModel(cfg).eval().to(torch.float32)

# Randomize weights (norms near 1.0 to exercise the affine scale); leave rotary buffers alone.
with torch.no_grad():
    for name, p in model.named_parameters():
        if "norm" in name:
            p.copy_(1.0 + 0.1 * torch.randn_like(p))
        else:
            p.copy_(0.1 * torch.randn_like(p))

# 6 tokens, last 2 padded — tests both the causal and padding masks.
input_ids = torch.tensor([[3, 17, 42, 8, 0, 0]], dtype=torch.long)
attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)

with torch.no_grad():
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )

# hidden_states: tuple of len LAYERS+1; [0] = embeddings, [k] = output of layer k-1. The dev
# pipeline stacks layers (10,20,30) then permute+reshape -> per-token feature concat. Tiny: (0,1,2).
hs = out.hidden_states
pick = (0, 1, 2)
stacked = torch.stack([hs[k] for k in pick], dim=1)  # (1, 3, seq, HIDDEN)
b, n, s, h = stacked.shape
prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(b, s, n * h)  # (1, seq, 3*HIDDEN)

tensors = {}
for k, v in model.state_dict().items():
    if "inv_freq" in k or "rotary" in k or k == "norm.weight":
        continue  # rope recomputed in Rust; final norm discarded by the prompt path
    tensors[k] = v.detach().cpu().numpy().astype(np.float32)
tensors["input_ids"] = input_ids.cpu().numpy().astype(np.int32)
tensors["attention_mask"] = attention_mask.cpu().numpy().astype(np.int32)
tensors["prompt_embeds"] = prompt_embeds.detach().cpu().numpy().astype(np.float32)

path = fixture("mlx-gen-flux2/tests/fixtures/te_dev_golden.safetensors")
save_file(tensors, path)
print(f"wrote {path} ({len(tensors)} tensors)")
print(f"  prompt_embeds: {tuple(prompt_embeds.shape)}  (expect (1, 6, {3 * HIDDEN}))")
print(f"  keys: {sorted(k for k in tensors if k not in ('input_ids','attention_mask','prompt_embeds'))}")
