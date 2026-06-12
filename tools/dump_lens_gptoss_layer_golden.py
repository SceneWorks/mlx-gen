#!/usr/bin/env python
"""Dump a single full-decoder-layer golden for the Lens gpt-oss encoder (mlx-gen sc-3166).

Runs the authoritative eager `transformers.GptOssDecoderLayer` (attention + **MoE** + residuals) on
layer 0 of the cached `microsoft/Lens-Turbo` text encoder, in **float32**, and writes a
self-contained safetensors golden the Rust parity test consumes. The MoE experts are MXFP4 in the
checkpoint; we embed the raw `*_blocks`/`*_scales` (compact uint8) + dense attn/router/norm weights so
the Rust side exercises its own MXFP4 dequant — no 12 GB snapshot needed at test time.

Golden contents:
  - `model.layers.0.*` — the layer's weights as stored (attn bf16 + biases f32, router bf16, norms
    bf16, experts `*_blocks`/`*_scales` uint8 + `*_bias` f32);
  - `x` / `layer_out`   — full-layer input/output (`[1, L, hidden]`);
  - `moe_in` / `moe_out`— MoE-only input/output (isolates the MoE from attention);
  - `ref_inv_freq`      — reference YaRN inv_freq (cross-check);
  - metadata: `L`, `hidden_size`.

Run (from repo root):
  ~/Repos/mflux/.venv/bin/python tools/dump_lens_gptoss_layer_golden.py
Writes `tools/golden/lens_gptoss_layer_golden.safetensors` (gitignored real-weights golden).
"""

from __future__ import annotations

import glob
import os

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.integrations.mxfp4 import convert_moe_packed_tensors
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssDecoderLayer,
    GptOssRotaryEmbedding,
)

LAYER = 0
SEQ_LEN = 16
HOME = os.path.expanduser("~")
SNAP_GLOB = f"{HOME}/.cache/huggingface/hub/models--microsoft--Lens-Turbo/snapshots/*/text_encoder"
OUT = os.path.join(os.path.dirname(__file__), "golden", "lens_gptoss_layer_golden.safetensors")


def text_encoder_dir() -> str:
    matches = sorted(glob.glob(SNAP_GLOB))
    if not matches:
        raise SystemExit(f"no Lens-Turbo text_encoder snapshot at {SNAP_GLOB}")
    return matches[-1]


def load_layer_tensors(te_dir: str, layer: int) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer}."
    out: dict[str, torch.Tensor] = {}
    for shard in sorted(glob.glob(os.path.join(te_dir, "*.safetensors"))):
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                if key.startswith(prefix):
                    out[key] = f.get_tensor(key)
    if not out:
        raise SystemExit(f"no {prefix}* tensors in {te_dir}")
    return out


def main() -> None:
    te_dir = text_encoder_dir()
    config = AutoConfig.from_pretrained(te_dir)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"  # the GptOssExperts loop path (dense, no triton)
    if hasattr(config, "quantization_config"):
        del config.quantization_config  # eager (dense) experts, not Mxfp4GptOssExperts
    torch.manual_seed(0)

    raw = load_layer_tensors(te_dir, LAYER)
    p = f"model.layers.{LAYER}."

    layer = GptOssDecoderLayer(config, layer_idx=LAYER).to(torch.float32).eval()
    rotary = GptOssRotaryEmbedding(config).to(torch.float32)

    # Attention (dense) + norms + router.
    attn_sd = {
        k[len(p + "self_attn.") :]: v.to(torch.float32)
        for k, v in raw.items()
        if k.startswith(p + "self_attn.")
    }
    layer.self_attn.load_state_dict(attn_sd, strict=False)
    layer.input_layernorm.weight = nn.Parameter(raw[p + "input_layernorm.weight"].to(torch.float32))
    layer.post_attention_layernorm.weight = nn.Parameter(
        raw[p + "post_attention_layernorm.weight"].to(torch.float32)
    )
    layer.mlp.router.load_state_dict(
        {
            "weight": raw[p + "mlp.router.weight"].to(torch.float32),
            "bias": raw[p + "mlp.router.bias"].to(torch.float32),
        }
    )

    # Experts: dequantize MXFP4 → dense params matching the eager GptOssExperts layout.
    gate_up = convert_moe_packed_tensors(
        raw[p + "mlp.experts.gate_up_proj_blocks"],
        raw[p + "mlp.experts.gate_up_proj_scales"],
        dtype=torch.float32,
    )  # [E, hidden, 2*inter]
    down = convert_moe_packed_tensors(
        raw[p + "mlp.experts.down_proj_blocks"],
        raw[p + "mlp.experts.down_proj_scales"],
        dtype=torch.float32,
    )  # [E, inter, hidden]
    layer.mlp.experts.gate_up_proj = nn.Parameter(gate_up)
    layer.mlp.experts.gate_up_proj_bias = nn.Parameter(
        raw[p + "mlp.experts.gate_up_proj_bias"].to(torch.float32)
    )
    layer.mlp.experts.down_proj = nn.Parameter(down)
    layer.mlp.experts.down_proj_bias = nn.Parameter(
        raw[p + "mlp.experts.down_proj_bias"].to(torch.float32)
    )

    hidden = config.hidden_size
    position_ids = torch.arange(SEQ_LEN).unsqueeze(0)
    cos, sin = rotary(torch.zeros(1, SEQ_LEN, hidden), position_ids)
    neg = torch.finfo(torch.float32).min
    causal = torch.triu(torch.full((SEQ_LEN, SEQ_LEN), neg), diagonal=1).reshape(1, 1, SEQ_LEN, SEQ_LEN)

    x = torch.randn(1, SEQ_LEN, hidden, dtype=torch.float32)
    moe_in = torch.randn(1, SEQ_LEN, hidden, dtype=torch.float32)
    with torch.no_grad():
        layer_out = layer(x, position_embeddings=(cos, sin), attention_mask=causal)
        if isinstance(layer_out, tuple):
            layer_out = layer_out[0]
        moe_out, _ = layer.mlp(moe_in)

    # Embed the layer weights as stored (compact: experts stay uint8 blocks).
    tensors = {k: v.contiguous() for k, v in raw.items()}
    tensors["x"] = x.contiguous()
    tensors["layer_out"] = layer_out.to(torch.float32).contiguous()
    tensors["moe_in"] = moe_in.contiguous()
    tensors["moe_out"] = moe_out.to(torch.float32).contiguous()
    tensors["ref_inv_freq"] = rotary.inv_freq.to(torch.float32).contiguous()

    meta = {
        "L": str(SEQ_LEN),
        "hidden_size": str(hidden),
        "transformers_version": __import__("transformers").__version__,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(tensors, OUT, metadata=meta)
    print(f"wrote {OUT}")
    print(f"  layer_out: mean={layer_out.mean():.5f} std={layer_out.std():.5f}")
    print(f"  moe_out:   mean={moe_out.mean():.5f} std={moe_out.std():.5f}")


if __name__ == "__main__":
    main()
