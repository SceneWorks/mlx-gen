"""Generate the committed Krea 2 text-encoder parity fixture (sc-7569).

Runs the **transformers** `Qwen3VLTextModel` (the independent reference forward) at TINY dims, text-only,
and saves the stacked select-layer hidden states `[B, L-prefix, n_select, hidden]` plus the `input_ids`
/ `attention_mask` the Rust parity test feeds verbatim. Text-only → standard RoPE (Qwen3-VL's
interleaved MRoPE sections all index the same sequential position when there are no image tokens), so
the tiny model exercises the exact block math (GQA, per-head q/k RMSNorm, half-split RoPE, SwiGLU,
the hidden-state stack + prefix slice) the real 4B uses.

Random norm weights (Qwen3 RMSNorm init = ones, which hides the weight-multiply) exercise every path.

Run from a torch venv:  ~/Repos/mflux/.venv/bin/python tools/dump_krea_te_golden.py
"""

from __future__ import annotations

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

from _paths import fixture

torch.manual_seed(0)

# Tiny dims; head_dim (32) != hidden/heads (16), mirroring the real 4B (128 != 2560/32).
VOCAB, HIDDEN, INTER, LAYERS = 128, 64, 128, 6
HEADS, KVHEADS, HEAD_DIM = 4, 2, 32
EPS, THETA = 1e-6, 5_000_000.0
SELECT = [2, 4]  # HF hidden_states indices (after 2 / 4 layers)
PREFIX = 3  # template-prefix tokens dropped
SEQ = 8


@torch.no_grad()
def main():
    cfg = Qwen3VLTextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KVHEADS,
        head_dim=HEAD_DIM,
        rms_norm_eps=EPS,
        rope_theta=THETA,
        max_position_embeddings=512,
    )
    model = Qwen3VLTextModel(cfg).eval()

    # Randomize every RMSNorm weight (init ones → would hide the weight-multiply).
    for name, p in model.named_parameters():
        if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
            p.data = 1.0 + 0.1 * torch.randn_like(p.data)

    input_ids = torch.randint(0, VOCAB, (1, SEQ))
    attention_mask = torch.ones_like(input_ids)
    out = model(
        input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
    )
    hiddens = torch.stack([out.hidden_states[i] for i in SELECT], dim=2)  # [1, SEQ, n, hidden]
    hiddens = hiddens[:, PREFIX:]  # drop the template prefix

    # Remap the text-submodule keys (`layers.*`, `embed_tokens`, `norm`) under `language_model.*`.
    tensors = {f"language_model.{k}": v for k, v in model.state_dict().items()}
    tensors["in.input_ids"] = input_ids.to(torch.int32)
    tensors["in.attention_mask"] = attention_mask.to(torch.int32)
    tensors["out.hiddens"] = hiddens
    tensors = {
        k: (v if v.dtype == torch.int32 else v.to(torch.float32)).contiguous()
        for k, v in tensors.items()
    }

    from safetensors.torch import save_file

    path = fixture("mlx-gen-krea/tests/fixtures/te_golden.safetensors")
    save_file(tensors, path)
    print(f"wrote {path}  ({len(tensors)} tensors, hiddens {tuple(hiddens.shape)})")


if __name__ == "__main__":
    main()
