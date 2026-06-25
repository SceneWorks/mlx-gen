#!/usr/bin/env python
"""Dump a tiny Gemma-2 decoder parity fixture for sc-7843 component 5 (PiD caption encoder).

Builds a small `Gemma2Model` (the decoder `AutoModelForCausalLM().get_decoder()` returns — embed →
norm-sandwich layers → final RMSNorm → last-hidden), with **eager** attention so the logit
soft-capping path runs, then dumps input_ids + the last-hidden states. The Rust `Gemma2` port loads
the bare state_dict (prefix="") and must match.

Run: /Users/michael/Repos/mlx-gen/_vendor/pid/.venv-pid/bin/python tools/dump_pid_gemma.py
"""

import os
import sys

import torch
from safetensors.torch import save_file
from transformers import Gemma2Config
from transformers.models.gemma2.modeling_gemma2 import Gemma2Model

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "mlx-gen-pid", "tests", "fixtures", "gemma2_tiny.safetensors",
)

CFG = Gemma2Config(
    vocab_size=100,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    query_pre_attn_scalar=8,        # == head_dim, mirroring the real 2-2b (256==256)
    attn_logit_softcapping=50.0,
    final_logit_softcapping=30.0,
    rope_theta=10000.0,
    rms_norm_eps=1e-6,
    max_position_embeddings=128,
    sliding_window=128,             # > seq -> every layer is full-causal (the PiD regime)
    hidden_activation="gelu_pytorch_tanh",
    attn_implementation="eager",    # force the soft-cap path (sdpa can't soft-cap)
)


def main():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(202)
    model = Gemma2Model(CFG).eval()
    with torch.no_grad():
        for _, p in model.named_parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.2)

    ids = torch.randint(0, CFG.vocab_size, (1, 6), generator=g)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    last_hidden = out.last_hidden_state  # [1, 6, 32]

    tensors = {k: v.detach().contiguous().float() for k, v in model.state_dict().items()}
    tensors["__io__.ids"] = ids.to(torch.int32).contiguous()
    tensors["__io__.last_hidden"] = last_hidden.detach().contiguous().float()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(tensors, OUT)
    print(f"wrote {OUT}")
    print(f"  state_dict tensors: {len(model.state_dict())}")
    print(f"  last_hidden {tuple(last_hidden.shape)} mean={last_hidden.mean():.5f} std={last_hidden.std():.5f}")
    print("  sample keys:")
    for k in list(tensors)[:10]:
        if not k.startswith("__io__"):
            print(f"    {k}  {tuple(tensors[k].shape)}")


if __name__ == "__main__":
    main()
