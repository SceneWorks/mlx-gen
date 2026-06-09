"""sc-3182: synthetic-fixture golden for the SenseNova-U1 (NEO-Unify) DENSE dual-path Qwen3 backbone.

Builds a *small* `NEOLLMConfig` (structurally faithful: head_dim split into temporal/H/W, GQA, dual
`*_mot_gen` paths) with random weights, runs the reference `Qwen3ForCausalLM` through both the pure
understanding path (`forward_und`, text-like positions) and the pure generation path (`forward_gen`,
image-grid positions, bidirectional-within-block), and dumps weights + inputs + outputs to a
safetensors fixture the Rust parity test loads. No 41 GB checkpoint or torchvision-heavy plumbing —
just the backbone math, in float32 for a clean near-bit comparison.

Run (uses the vendored reference env):
  cd _vendor/sensenova_u1 && PYTHONPATH=src .venv/bin/python \
      ../../tools/dump_sensenova_backbone_golden.py
Fixture → mlx-gen-sensenova/tests/fixtures/backbone_golden.safetensors
"""

from __future__ import annotations

import os
import sys

import torch
from safetensors.torch import save_file

# tools/ is a sibling of _vendor; resolve the repo root from this file.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from sensenova_u1.models.neo_unify.configuration_neo_chat import NEOLLMConfig
from sensenova_u1.models.neo_unify.modeling_qwen3 import Qwen3ForCausalLM


def build_config() -> NEOLLMConfig:
    # Tiny but structurally faithful → the committed fixture stays well under 1 MB.
    cfg = NEOLLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,  # -> temporal 16 (q_norm dim 16), spatial 16 -> H 8 + W 8
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
    cfg._attn_implementation = "eager"  # deterministic; matches the port's manual attention
    # NEOLLMConfig (unlike NEOMoELLMConfig) does not backfill layer_types — do it here.
    if not getattr(cfg, "layer_types", None) or len(cfg.layer_types) != cfg.num_hidden_layers:
        cfg.layer_types = ["full_attention"] * cfg.num_hidden_layers
    return cfg


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    cfg = build_config()
    model = Qwen3ForCausalLM(cfg).to(torch.float32).eval()

    S = 6
    H = cfg.hidden_size
    embeds = torch.randn(1, S, H, dtype=torch.float32)

    out = {}

    # Persist every weight under the real checkpoint's `language_model.` prefix so the Rust loader's
    # canonical keys apply unchanged.
    for k, v in model.state_dict().items():
        out[f"language_model.{k}"] = v.contiguous().to(torch.float32)

    # ---- Understanding path: text-like positions (temporal = arange, H = W = 0) ----
    und_indexes = torch.zeros(3, S, dtype=torch.long)
    und_indexes[0] = torch.arange(S)
    und_mask = torch.zeros(1, S, dtype=torch.bool)  # all understanding tokens
    und = model(
        inputs_embeds=embeds,
        indexes=und_indexes,
        image_gen_indicators=und_mask,
        use_cache=False,
    )

    # ---- Generation path: a 2x3 image grid sharing one temporal index (bidirectional block) ----
    gen_indexes = torch.zeros(3, S, dtype=torch.long)
    gen_indexes[0] = 0  # all image tokens share the temporal index
    gen_indexes[1] = torch.tensor([0, 0, 0, 1, 1, 1])  # height (rows of a 2x3 grid)
    gen_indexes[2] = torch.tensor([0, 1, 2, 0, 1, 2])  # width  (cols)
    gen_mask = torch.ones(1, S, dtype=torch.bool)  # all generation tokens
    gen = model(
        inputs_embeds=embeds,
        indexes=gen_indexes,
        image_gen_indicators=gen_mask,
        use_cache=False,
    )

    out["input.embeds"] = embeds
    out["und.indexes"] = und_indexes.to(torch.int32)
    out["und.hidden"] = und.hidden_states.to(torch.float32)
    out["und.logits"] = und.logits.to(torch.float32)
    out["gen.indexes"] = gen_indexes.to(torch.int32)
    out["gen.hidden"] = gen.hidden_states.to(torch.float32)
    out["gen.logits"] = gen.logits.to(torch.float32)

    dst = os.path.join(
        REPO_ROOT, "mlx-gen-sensenova", "tests", "fixtures", "backbone_golden.safetensors"
    )
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # Flat metadata records the config the Rust side rebuilds.
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
        "seq_len": str(S),
    }
    save_file(out, dst, metadata=meta)
    print(f"wrote {dst}")
    print(f"  und.hidden {tuple(und.hidden_states.shape)}  gen.hidden {tuple(gen.hidden_states.shape)}")
    print(f"  und.logits {tuple(und.logits.shape)}  gen.logits {tuple(gen.logits.shape)}")
    print(f"  tensors: {len(out)}")


if __name__ == "__main__":
    sys.exit(main())
