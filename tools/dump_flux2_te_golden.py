"""Parity fixture for the FLUX.2 Qwen3 text encoder (sc-2346 S1) — a TINY synthetic config so it
commits cheaply and CI stays fast, while exercising every code path: bias-less GQA, per-head q/k
RMSNorm, HF half-split RoPE, the causal+padding mask, and the multi-layer (embed/layer-0/layer-1)
hidden-state concatenation. Random weights, f32 throughout (the Rust port runs f32 activations).

Run from the mflux fork venv:
    cd ~/repos/mflux && .venv/bin/python ~/repos/mlx-gen/tools/dump_flux2_te_golden.py
"""

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder

from _paths import fixture

mx.random.seed(0)

# Tiny config (head_dim=16, 4 q / 2 kv heads, hidden 64, 2 layers, intermediate 128).
VOCAB, HIDDEN, LAYERS, HEADS, KV, INTER, HEAD_DIM = 64, 64, 2, 4, 2, 128, 16
te = Qwen3TextEncoder(
    vocab_size=VOCAB,
    hidden_size=HIDDEN,
    num_hidden_layers=LAYERS,
    num_attention_heads=HEADS,
    num_key_value_heads=KV,
    intermediate_size=INTER,
    rope_theta=1_000_000.0,
    rms_norm_eps=1e-6,
    head_dim=HEAD_DIM,
    attention_bias=False,
)

# Randomize every weight (norms around 1.0 to exercise the affine scale). Leave the rotary
# `inv_freq` buffer alone — the Rust port recomputes it from theta/head_dim.
flat = tree_flatten(te.parameters())
new = []
for k, v in flat:
    if "inv_freq" in k:
        new.append((k, v.astype(mx.float32)))
    elif "norm" in k or "layernorm" in k:
        new.append((k, (1.0 + 0.1 * mx.random.normal(v.shape)).astype(mx.float32)))
    else:
        new.append((k, (0.1 * mx.random.normal(v.shape)).astype(mx.float32)))
te.update(tree_unflatten(new))

# Inputs: 6 tokens, last 2 padded (tests both the causal and padding masks).
input_ids = mx.array(np.array([[3, 17, 42, 8, 0, 0]], dtype=np.int32))
attention_mask = mx.array(np.array([[1, 1, 1, 1, 0, 0]], dtype=np.int32))

prompt_embeds = te.get_prompt_embeds(
    input_ids=input_ids,
    attention_mask=attention_mask,
    hidden_state_layers=(0, 1, 2),  # embed, layer-0, layer-1 (list has LAYERS+1 entries)
)

out = {}
for k, v in tree_flatten(te.parameters()):
    if "inv_freq" in k:
        continue  # recomputed in Rust; not loaded
    out[k] = v.astype(mx.float32)
out["input_ids"] = input_ids.astype(mx.int32)
out["attention_mask"] = attention_mask.astype(mx.int32)
out["prompt_embeds"] = prompt_embeds.astype(mx.float32)  # [1, 6, 3*64=192]

path = fixture("mlx-gen-flux2/tests/fixtures/te_golden.safetensors")
mx.save_safetensors(path, out)
print(f"wrote {path} ({len(out)} tensors)")
print(f"  prompt_embeds: {tuple(prompt_embeds.shape)}")
