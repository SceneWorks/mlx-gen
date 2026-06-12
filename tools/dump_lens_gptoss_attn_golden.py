#!/usr/bin/env python
"""Dump a single-layer golden for the Lens gpt-oss **attention core** (mlx-gen sc-3165).

Runs the authoritative `transformers.models.gpt_oss.GptOssAttention` (forced **eager** so the
attention-sink path is exercised) on layer 0 of the cached `microsoft/Lens-Turbo` text encoder, in
**float32**, and writes a self-contained safetensors golden the Rust parity test consumes:

  - `model.layers.0.self_attn.{q,k,v,o}_proj.{weight,bias}`, `…self_attn.sinks` — the dense weights
    (the attention modules are NOT MXFP4: `modules_to_not_convert` keeps `self_attn` bf16);
  - `x`            — the RMSNorm'd hidden state fed to the attention (`[1, L, hidden]`, fixed seed);
  - `attn_out`     — the reference attention output before the residual add (`[1, L, hidden]`);
  - `ref_inv_freq` — the reference YaRN `inv_freq` (`[head_dim/2]`), to cross-check the Rust derivation;
  - metadata: `L`, `attention_scaling`, `layer0_is_sliding`, `sliding_window`.

Run (from repo root):
  ~/Repos/mflux/.venv/bin/python tools/dump_lens_gptoss_attn_golden.py

Writes `tools/golden/lens_gptoss_attn_golden.safetensors` (gitignored real-weights golden).
"""

from __future__ import annotations

import glob
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention,
    GptOssRotaryEmbedding,
)

LAYER = 0
SEQ_LEN = 16
HOME = os.path.expanduser("~")
SNAP_GLOB = f"{HOME}/.cache/huggingface/hub/models--microsoft--Lens-Turbo/snapshots/*/text_encoder"
OUT = os.path.join(os.path.dirname(__file__), "golden", "lens_gptoss_attn_golden.safetensors")


def text_encoder_dir() -> str:
    matches = sorted(glob.glob(SNAP_GLOB))
    if not matches:
        raise SystemExit(f"no Lens-Turbo text_encoder snapshot at {SNAP_GLOB}")
    return matches[-1]


def load_self_attn_tensors(te_dir: str, layer: int) -> dict[str, torch.Tensor]:
    """Load just `model.layers.{layer}.self_attn.*` (dense bf16) from the sharded checkpoint."""
    prefix = f"model.layers.{layer}.self_attn."
    out: dict[str, torch.Tensor] = {}
    for shard in sorted(glob.glob(os.path.join(te_dir, "*.safetensors"))):
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                if key.startswith(prefix):
                    out[key] = f.get_tensor(key)
    if not out:
        raise SystemExit(f"no {prefix}* tensors found in {te_dir}")
    return out


def main() -> None:
    te_dir = text_encoder_dir()
    config = AutoConfig.from_pretrained(te_dir)
    config._attn_implementation = "eager"  # force the sink-aware eager path
    torch.manual_seed(0)

    raw = load_self_attn_tensors(te_dir, LAYER)
    prefix = f"model.layers.{LAYER}.self_attn."

    # Build the reference attention + rotary, load the layer's dense weights as float32.
    attn = GptOssAttention(config, layer_idx=LAYER).to(torch.float32).eval()
    rotary = GptOssRotaryEmbedding(config).to(torch.float32)
    module_sd = {k[len(prefix):]: v.to(torch.float32) for k, v in raw.items()}
    missing, unexpected = attn.load_state_dict(module_sd, strict=False)
    # `sinks` + the 4 projections (weight+bias) must all be present.
    assert not [m for m in missing if "rotary" not in m], f"missing attn weights: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"

    hidden = config.hidden_size
    x = torch.randn(1, SEQ_LEN, hidden, dtype=torch.float32)
    position_ids = torch.arange(SEQ_LEN).unsqueeze(0)
    cos, sin = rotary(x, position_ids)

    # Full causal additive mask (layer 0 is sliding, but SEQ_LEN < sliding_window so it equals causal).
    neg = torch.finfo(torch.float32).min
    causal = torch.full((SEQ_LEN, SEQ_LEN), neg, dtype=torch.float32)
    causal = torch.triu(causal, diagonal=1).reshape(1, 1, SEQ_LEN, SEQ_LEN)

    with torch.no_grad():
        attn_out, _ = attn(x, position_embeddings=(cos, sin), attention_mask=causal)

    layer_types = getattr(config, "layer_types", None)
    is_sliding = bool(layer_types and layer_types[LAYER] == "sliding_attention")

    tensors = {k: v.to(torch.float32).contiguous() for k, v in raw.items()}
    tensors["x"] = x.contiguous()
    tensors["attn_out"] = attn_out.to(torch.float32).contiguous()
    tensors["ref_inv_freq"] = rotary.inv_freq.to(torch.float32).contiguous()

    meta = {
        "L": str(SEQ_LEN),
        "hidden_size": str(hidden),
        "attention_scaling": repr(float(rotary.attention_scaling)),
        "layer0_is_sliding": str(is_sliding),
        "sliding_window": str(getattr(config, "sliding_window", 0)),
        "transformers_version": __import__("transformers").__version__,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(tensors, OUT, metadata=meta)
    print(f"wrote {OUT}")
    print(f"  L={SEQ_LEN} hidden={hidden} attention_scaling={meta['attention_scaling']}")
    print(f"  layer0_is_sliding={is_sliding} sliding_window={meta['sliding_window']}")
    print(f"  attn_out: mean={attn_out.mean():.5f} std={attn_out.std():.5f}")


if __name__ == "__main__":
    main()
