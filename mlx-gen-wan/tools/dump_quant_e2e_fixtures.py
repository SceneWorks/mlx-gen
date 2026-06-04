#!/usr/bin/env python3
"""Dump the **end-to-end** Q4/Q8 MoE parity fixtures (sc-2682) — the quantized twin of
`dump_s6_real_fixtures.py`. Small-but-real dual-expert T2V run of the `mlx_video` reference on the
*actual converted* Wan2.2-T2V-A14B weights, with **both experts quantized independently** (the story's
per-expert requirement) via the reference predicate.

Each expert is loaded bf16 then quantized in memory the way `convert_wan.py::_quantize_saved_model`
does — `nn.quantize` with `_quantize_predicate` (attn `q/k/v/o` + `ffn.fc1/fc2`, group 64). T5 + VAE
stay f32 (the reference's quant scope). Only the init noise is injected; everything else is the real
chain, so the Rust gate (`tests/quant_e2e_parity.rs`) re-checks real-weight T5 parity too.

Run with the SceneWorks venv, pointing at the converted model dir + a fixture base path:

    WAN_A14B_MODEL_DIR=~/.cache/mlx-gen-models/wan2_2_t2v_a14b_mlx_bf16 \
    WAN_A14B_QUANT_FIXTURE=/tmp/wan_a14b_quant \
    WAN_QUANT_BITS=4,8 \
    "$HOME/Library/Application Support/SceneWorks/python/venv/bin/python" \
        mlx-gen-wan/tools/dump_quant_e2e_fixtures.py

Writes (NOT committed — tied to the 54 GB weights), one per bits:
  ${WAN_A14B_QUANT_FIXTURE}_q{bits}.safetensors  (noise + context + golden latents + golden video)
  ${WAN_A14B_QUANT_FIXTURE}_q{bits}.json         (metadata: prompt, geometry, routing, bits)
"""
import gc
import json
import math
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.utils

from mlx_video.convert_wan import _quantize_predicate
from mlx_video.models.wan.config import WanModelConfig
from mlx_video.models.wan.loading import (
    encode_text,
    load_t5_encoder,
    load_vae_decoder,
    load_wan_model,
)
from mlx_video.models.wan.scheduler import FlowUniPCScheduler

PROMPT = "a red fox trotting across a snowy meadow at sunrise, cinematic"
FRAMES, HEIGHT, WIDTH = 5, 128, 128
STEPS = 6
GROUP_SIZE = 64


def quantize_expert(model, bits):
    """In-memory `nn.quantize` of one loaded expert, matching the reference predicate."""
    nn.quantize(
        model,
        group_size=GROUP_SIZE,
        bits=bits,
        class_predicate=lambda path, m: _quantize_predicate(path, m),
    )
    mx.eval(model.parameters())
    n_q = sum(1 for k, _ in mlx.utils.tree_flatten(model.parameters()) if ".scales" in k)
    return n_q


def main():
    model_dir = Path(os.path.expanduser(os.environ["WAN_A14B_MODEL_DIR"]))
    fixture_base = Path(os.path.expanduser(os.environ["WAN_A14B_QUANT_FIXTURE"]))
    bits_list = [int(b) for b in os.environ.get("WAN_QUANT_BITS", "4,8").split(",")]

    with open(model_dir / "config.json") as f:
        cfg_json = json.load(f)
    fields = WanModelConfig.__dataclass_fields__
    cdict = {k: v for k, v in cfg_json.items() if k in fields}
    for key in ("patch_size", "vae_stride", "window_size", "sample_guide_scale"):
        if key in cdict and isinstance(cdict[key], list):
            cdict[key] = tuple(cdict[key])
    config = WanModelConfig(**cdict)
    assert config.dual_model, "expected the dual-expert A14B config"

    shift = config.sample_shift
    guide_low, guide_high = config.sample_guide_scale
    neg_prompt = config.sample_neg_prompt
    boundary = config.boundary * config.num_train_timesteps

    vae_stride, patch, z_dim = config.vae_stride, config.patch_size, config.vae_z_dim
    t_lat = (FRAMES - 1) // vae_stride[0] + 1
    h_lat, w_lat = HEIGHT // vae_stride[1], WIDTH // vae_stride[2]
    seq_len = math.ceil((h_lat * w_lat) / (patch[1] * patch[2]) * t_lat)
    f_grid, h_grid, w_grid = t_lat // patch[0], h_lat // patch[1], w_lat // patch[2]

    # --- Real UMT5 encode (once; reused for every bits) ---
    print("Loading T5 + tokenizer, encoding prompt...")
    from transformers import AutoTokenizer

    t5 = load_t5_encoder(model_dir / "t5_encoder.safetensors", config)
    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    context = encode_text(t5, tokenizer, PROMPT, config.text_len)
    context_null = encode_text(t5, tokenizer, neg_prompt, config.text_len)
    mx.eval(context, context_null)
    del t5
    gc.collect()
    mx.clear_cache()

    # --- Injected init noise (shared across bits so the goldens are comparable) ---
    mx.random.seed(1234)
    noise = mx.random.normal((z_dim, t_lat, h_lat, w_lat)).astype(mx.float32)
    mx.eval(noise)

    for bits in bits_list:
        print(f"\n=== {bits}-bit (group_size={GROUP_SIZE}) ===")
        print("Loading + quantizing low/high experts...")
        low_model = load_wan_model(model_dir / "low_noise_model.safetensors", config)
        n_lo = quantize_expert(low_model, bits)
        high_model = load_wan_model(model_dir / "high_noise_model.safetensors", config)
        n_hi = quantize_expert(high_model, bits)
        print(f"  quantized low={n_lo} high={n_hi} layers")

        def prep(model):
            emb = model.embed_text([context, context_null])
            ctx = mx.concatenate([emb[0:1], emb[1:2]], axis=0)
            kv = model.prepare_cross_kv(ctx)
            rope = model.prepare_rope([(f_grid, h_grid, w_grid), (f_grid, h_grid, w_grid)])
            return ctx, kv, rope

        ctx_low, kv_low, rope_low = prep(low_model)
        ctx_high, kv_high, rope_high = prep(high_model)

        sched = FlowUniPCScheduler(num_train_timesteps=config.num_train_timesteps)
        sched.set_timesteps(STEPS, shift=shift)

        latents = noise
        routing = []
        print(f"Denoising {STEPS} steps (boundary={boundary})...")
        for t in sched.timesteps.tolist():
            if t >= boundary:
                model, ctx, kv, rope, gs = high_model, ctx_high, kv_high, rope_high, guide_high
                routing.append(["high", t])
            else:
                model, ctx, kv, rope, gs = low_model, ctx_low, kv_low, rope_low, guide_low
                routing.append(["low", t])
            preds = model(
                [latents, latents],
                t=mx.array([t, t]),
                context=ctx,
                seq_len=seq_len,
                cross_kv_caches=kv,
                rope_cos_sin=rope,
            )
            noise_pred = preds[1] + gs * (preds[0] - preds[1])
            latents = sched.step(noise_pred[None], t, latents[None]).squeeze(0)
            mx.eval(latents)
        final_latents = latents
        assert any(r[0] == "high" for r in routing) and any(r[0] == "low" for r in routing), (
            f"fixture must exercise BOTH experts across boundary {boundary}; routing={routing}"
        )
        del low_model, high_model
        gc.collect()
        mx.clear_cache()

        print("Decoding with the z16 VAE...")
        vae = load_vae_decoder(model_dir / "vae.safetensors", config)
        video = vae.decode(final_latents[None])
        mx.eval(video)
        del vae
        gc.collect()
        mx.clear_cache()

        fixture_path = fixture_base.with_name(fixture_base.name + f"_q{bits}.safetensors")
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(
            str(fixture_path),
            {
                "noise": noise,
                "context": context,
                "context_null": context_null,
                "final_latents": final_latents.astype(mx.float32),
                "video": video.astype(mx.float32),
            },
        )
        meta = {
            "prompt": PROMPT,
            "neg_prompt": neg_prompt,
            "bits": bits,
            "group_size": GROUP_SIZE,
            "frames": FRAMES,
            "height": HEIGHT,
            "width": WIDTH,
            "steps": STEPS,
            "shift": shift,
            "guide_low": guide_low,
            "guide_high": guide_high,
            "boundary": config.boundary,
            "boundary_timestep": boundary,
            "num_train_timesteps": config.num_train_timesteps,
            "seq_len": seq_len,
            "routing": routing,
            "final_latents_shape": list(final_latents.shape),
            "video_shape": list(video.shape),
        }
        with open(fixture_path.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  routing={routing}")
        print(f"  wrote {fixture_path}")


if __name__ == "__main__":
    main()
