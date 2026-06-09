"""sc-3187: synthetic-fixture golden for the SenseNova-U1 AR text-generation runtime.

Builds the same *small* `NEOLLMConfig` as the backbone golden (dense dual-path Qwen3, tri-axis RoPE,
GQA), then exercises the reference's incremental-decode machinery end to end, weight-free:

  1. Prefill a text prefix into an HF `DynamicCache` via the block-causal-mask path
     (`attention_mask={"full_attention": create_block_causal_mask(indexes[0])}`, the reference
     `_t2i_prefix_forward` / `_build_t2i_text_inputs` route that honours the passed `indexes`).
  2. Greedy-decode N tokens exactly as `_generate_think` does — set `model.model.current_index =
     t_idx`, forward the single token with `past_key_values=cache, use_cache=True` (no indexes / no
     mask, so the forward assigns temporal index `t_idx+1` and an all-attend causal mask), argmax.

Dumps weights + the prefix + the greedy token stream + per-step logits to a safetensors fixture the
Rust parity test (`tests/runtime_parity.rs`) replays through `Qwen3Backbone::forward_cached` /
`generate`. Validates the KV cache, the position bookkeeping, and greedy sampling without the 35 GB
checkpoint. float32 throughout for a clean near-bit comparison.

Run (uses the vendored reference env):
  cd _vendor/sensenova_u1 && PYTHONPATH=src .venv/bin/python \
      ../../tools/dump_sensenova_runtime_golden.py
Fixture → mlx-gen-sensenova/tests/fixtures/runtime_golden.safetensors
"""

from __future__ import annotations

import os
import sys

import torch
from safetensors.torch import save_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from sensenova_u1.models.neo_unify.configuration_neo_chat import NEOLLMConfig
from sensenova_u1.models.neo_unify.modeling_qwen3 import (
    Qwen3ForCausalLM,
    create_block_causal_mask,
)


def build_config() -> NEOLLMConfig:
    # Identical to dump_sensenova_backbone_golden.py so the Rust config-from-metadata helper is shared.
    cfg = NEOLLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-6,
        rope_theta=5_000_000.0,
        rope_theta_hw=10_000.0,
        max_position_embeddings=4096,
        max_position_embeddings_hw=4096,
        vocab_size=64,
        attention_bias=False,
        hidden_act="silu",
        use_sliding_window=False,
        max_window_layers=0,
        tie_word_embeddings=False,
    )
    cfg._attn_implementation = "eager"
    if not getattr(cfg, "layer_types", None) or len(cfg.layer_types) != cfg.num_hidden_layers:
        cfg.layer_types = ["full_attention"] * cfg.num_hidden_layers
    return cfg


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    cfg = build_config()
    model = Qwen3ForCausalLM(cfg).to(torch.float32).eval()

    S = 5  # prefix length
    N = 8  # decode steps
    prefix_ids = torch.randint(0, cfg.vocab_size, (1, S), dtype=torch.long)

    # ---- Prefill (understanding path, block-causal over text-like positions) ----
    indexes = torch.zeros(3, S, dtype=torch.long)
    indexes[0] = torch.arange(S)
    attn_mask = {"full_attention": create_block_causal_mask(indexes[0])}
    prefill = model(
        input_ids=prefix_ids,
        indexes=indexes,
        attention_mask=attn_mask,
        use_cache=True,
    )
    cache = prefill.past_key_values
    t_idx = int(indexes[0].max().item())

    # ---- Greedy decode (mirrors _generate_think's single-token step) ----
    next_token = torch.argmax(prefill.logits[:, -1, :], dim=-1)  # [1]
    decode_tokens = []
    decode_logits = []
    for _ in range(N):
        decode_tokens.append(int(next_token.item()))
        model.model.current_index = t_idx
        out = model(
            input_ids=next_token.unsqueeze(0),  # [1, 1]
            past_key_values=cache,
            use_cache=True,
        )
        cache = out.past_key_values
        t_idx += 1
        row = out.logits[:, -1, :]  # [1, vocab] — predicts the next token
        decode_logits.append(row)
        next_token = torch.argmax(row, dim=-1)

    out_t = {}
    for k, v in model.state_dict().items():
        out_t[f"language_model.{k}"] = v.contiguous().to(torch.float32)
    out_t["prefix.input_ids"] = prefix_ids.to(torch.int32)
    out_t["prefix.indexes"] = indexes.to(torch.int32)
    out_t["prefix.logits"] = prefill.logits.to(torch.float32)
    out_t["decode.tokens"] = torch.tensor(decode_tokens, dtype=torch.int32)
    out_t["decode.logits"] = torch.cat(decode_logits, dim=0).to(torch.float32)  # [N, vocab]

    dst = os.path.join(
        REPO_ROOT, "mlx-gen-sensenova", "tests", "fixtures", "runtime_golden.safetensors"
    )
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    meta = {
        "hidden_size": str(cfg.hidden_size),
        "intermediate_size": str(cfg.intermediate_size),
        "num_hidden_layers": str(cfg.num_hidden_layers),
        "num_attention_heads": str(cfg.num_attention_heads),
        "num_key_value_heads": str(cfg.num_key_value_heads),
        "head_dim": str(cfg.head_dim),
        "rms_norm_eps": repr(cfg.rms_norm_eps),
        "rope_theta": repr(cfg.rope_theta),
        "rope_theta_hw": repr(cfg.rope_theta_hw),
        "vocab_size": str(cfg.vocab_size),
        "prefix_len": str(S),
        "decode_steps": str(N),
    }
    save_file(out_t, dst, metadata=meta)
    print(f"wrote {dst}")
    print(f"  prefix_ids {tuple(prefix_ids.shape)}  decode.tokens {decode_tokens}")
    print(f"  decode.logits {tuple(out_t['decode.logits'].shape)}  tensors: {len(out_t)}")


if __name__ == "__main__":
    sys.exit(main())
