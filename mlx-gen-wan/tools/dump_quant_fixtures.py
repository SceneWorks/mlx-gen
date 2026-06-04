#!/usr/bin/env python3
"""Dump the Q4 + Q8 transformer-forward parity goldens (sc-2682) from the `mlx_video` Wan reference.

This is the quantized twin of `dump_s3_fixtures.py`: it quantizes the bf16 5B DiT **in memory** the
same way `convert_wan.py::_quantize_saved_model` / `loading.py` do — `nn.quantize` with the reference
`_quantize_predicate` (attention `q/k/v/o` + `ffn.fc1/fc2`, group_size 64, skipping
embeddings/norms/head/modulation) — then runs the identical forward and dumps the output. The Rust
gate (`tests/quant_parity.rs`) quantizes the *same* bf16 checkpoint via `WanTransformer::quantize` and
must reproduce these bit-for-bit (the group scales are byte-identical because both quantize the same
bf16 weights with the same MLX op).

Run with the SceneWorks venv:
    "$HOME/Library/Application Support/SceneWorks/python/venv/bin/python" \
        mlx-gen-wan/tools/dump_quant_fixtures.py

Writes (committed; small — output is [48,1,16,16] f32 per bits):
  mlx-gen-wan/tests/fixtures/q4_dit_golden.safetensors
  mlx-gen-wan/tests/fixtures/q8_dit_golden.safetensors
each with: latent, context_raw, t, output (self-contained inputs + golden output).
"""
import glob
import os

import mlx.core as mx
import mlx.nn as nn
import mlx.utils

from mlx_video.convert_wan import (
    _quantize_predicate,
    load_safetensors_weights,
    sanitize_wan_transformer_weights,
)
from mlx_video.models.wan.config import WanModelConfig
from mlx_video.models.wan.model import WanModel

HOME = os.path.expanduser("~")
HF = os.path.join(HOME, ".cache/huggingface/hub")
OUT_DIR = os.environ.get(
    "WAN_5B_DIR",
    os.path.join(HOME, "Library/Application Support/SceneWorks/data/models/mlx/wan_2_2_ti2v_5b"),
)
GROUP_SIZE = 64  # the reference + mflux default

# Same small grid as dump_s3_fixtures.py (t_lat=1, h_lat=16, w_lat=16 → grid (1,8,8) → L=64).
C, T_LAT, H_LAT, W_LAT = 48, 1, 16, 16
L_TEXT = 12
T_VAL = 500.0


def ensure_model(config):
    out = os.path.join(OUT_DIR, "model.safetensors")
    if not os.path.exists(out):
        snap = sorted(glob.glob(os.path.join(HF, "models--Wan-AI--Wan2.2-TI2V-5B/snapshots/*")))[-1]
        print(f"Converting 5B DiT from {snap} ...")
        weights = sanitize_wan_transformer_weights(load_safetensors_weights(snap))
        weights = {k: v.astype(mx.bfloat16) for k, v in weights.items()}
        mx.save_safetensors(out, weights)
        print(f"  wrote {len(weights)} tensors → {out}")
    return out


def build_quantized(config, model_path, bits):
    """Replicate `_quantize_saved_model`'s in-memory steps: load the bf16 model, then nn.quantize the
    predicate layers (bits, group 64). Returns the quantized WanModel."""
    model = WanModel(config)
    weights = mx.load(model_path)  # bf16 on disk
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    nn.quantize(
        model,
        group_size=GROUP_SIZE,
        bits=bits,
        class_predicate=lambda path, m: _quantize_predicate(path, m),
    )
    mx.eval(model.parameters())
    n_q = sum(1 for k, _ in mlx.utils.tree_flatten(model.parameters()) if ".scales" in k)
    print(f"  {bits}-bit: {n_q} layers quantized")
    return model


def main():
    config = WanModelConfig.wan22_ti2v_5b()
    model_path = ensure_model(config)

    # Fixed seeded inputs (identical to dump_s3 so the dense + quant goldens share geometry).
    mx.random.seed(0)
    latent = mx.random.normal((C, T_LAT, H_LAT, W_LAT)).astype(mx.float32)
    context_raw = mx.random.normal((L_TEXT, config.text_dim)).astype(mx.float32)
    grid = (T_LAT // config.patch_size[0], H_LAT // config.patch_size[1], W_LAT // config.patch_size[2])
    seq_len = grid[0] * grid[1] * grid[2]

    dst = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
    os.makedirs(dst, exist_ok=True)

    for bits in (4, 8):
        print(f"Quantizing {bits}-bit (group_size={GROUP_SIZE})...")
        model = build_quantized(config, model_path, bits)

        context_emb = model.embed_text([context_raw])
        cross_kv = model.prepare_cross_kv(context_emb)
        rope_cs = model.prepare_rope([grid])
        out = model(
            [latent],
            t=mx.array([T_VAL]),
            context=context_emb,
            seq_len=seq_len,
            cross_kv_caches=cross_kv,
            rope_cos_sin=rope_cs,
        )[0]
        mx.eval(out)

        golden = {
            "latent": latent,
            "context_raw": context_raw,
            "t": mx.array([T_VAL]),
            "output": out.astype(mx.float32),
        }
        path = os.path.join(dst, f"q{bits}_dit_golden.safetensors")
        mx.save_safetensors(path, golden)
        print(f"  wrote {os.path.abspath(path)}  output_shape={list(out.shape)}")
        del model
        import gc

        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    main()
