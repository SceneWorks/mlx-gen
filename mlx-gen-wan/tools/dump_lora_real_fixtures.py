#!/usr/bin/env python3
"""Dump a **real-weight LoRA** parity fixture (sc-2683): a small dual-expert MoE T2V run of the
`mlx_video` reference on the *actual converted* Wan2.2-T2V-A14B weights with a real MoE high/low
LoRA pair **merged** onto the experts (the production `--lora-high`/`--lora-low` path), for the Rust
`merge_wan_adapters` + `denoise_moe` chain to gate against end-to-end.

This is the definitive "parity vs a reference-merged golden" gate. The reference merges via
`load_wan_model(path, config, loras=[(lora_path, strength)])` → `load_and_apply_loras` →
`apply_loras_to_weights` (`W += (B·A·alpha/rank·strength).astype(bf16)`), exactly the path
`generate_wan.py` runs for `--lora-high`/`--lora-low`. The same injected noise drives a **bare**
(no-LoRA) run too, so the Rust side can also assert the LoRA visibly moves the output.

Run with the SceneWorks venv, pointing at the converted model dir + the lauren high/low pair:

    WAN_A14B_MODEL_DIR=~/.cache/mlx-gen-models/wan2_2_t2v_a14b_mlx_bf16 \
    WAN_LORA_HIGH="~/Library/Application Support/SceneWorks/data/loras/lauren_high/lauren_wan22_high_epoch_95.safetensors" \
    WAN_LORA_LOW="~/Library/Application Support/SceneWorks/data/loras/lauren_low/lauren_wan22_low_epoch_30.safetensors" \
    WAN_LORA_FIXTURE=/tmp/wan_a14b_lora.safetensors \
    "$HOME/Library/Application Support/SceneWorks/python/venv/bin/python" \
        mlx-gen-wan/tools/dump_lora_real_fixtures.py

Writes (NOT committed — tied to the 54 GB converted weights):
  - $WAN_LORA_FIXTURE                     (noise + context + LoRA-merged + bare golden latents/video)
  - ${WAN_LORA_FIXTURE%.safetensors}.json (metadata + thresholds)

The Rust side (`tests/lora_real_parity.rs`, #[ignore]) reads the env vars + this fixture.
"""
import json
import math
import os
from pathlib import Path

import mlx.core as mx

from mlx_video.models.wan.config import WanModelConfig
from mlx_video.models.wan.loading import encode_text, load_t5_encoder, load_vae_decoder, load_wan_model
from mlx_video.models.wan.scheduler import FlowUniPCScheduler

# Match tools/dump_s6_real_fixtures.py geometry (so the regime equals the validated base gate).
PROMPT = "a red fox trotting across a snowy meadow at sunrise, cinematic"
FRAMES, HEIGHT, WIDTH = 5, 128, 128
STEPS = 6
LORA_STRENGTH = 1.0


def main():
    model_dir = Path(os.path.expanduser(os.environ["WAN_A14B_MODEL_DIR"]))
    fixture_path = Path(os.path.expanduser(os.environ["WAN_LORA_FIXTURE"]))
    lora_high = os.path.expanduser(os.environ["WAN_LORA_HIGH"])
    lora_low = os.path.expanduser(os.environ["WAN_LORA_LOW"])

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

    print("Loading T5 + tokenizer, encoding prompt...")
    from transformers import AutoTokenizer

    t5 = load_t5_encoder(model_dir / "t5_encoder.safetensors", config)
    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    context = encode_text(t5, tokenizer, PROMPT, config.text_len)
    context_null = encode_text(t5, tokenizer, neg_prompt, config.text_len)
    mx.eval(context, context_null)
    del t5

    mx.random.seed(1234)
    noise = mx.random.normal((z_dim, t_lat, h_lat, w_lat)).astype(mx.float32)
    mx.eval(noise)

    def denoise(low_model, high_model):
        def prep(model):
            emb = model.embed_text([context, context_null])
            ctx = mx.concatenate([emb[0:1], emb[1:2]], axis=0)
            return ctx, model.prepare_cross_kv(ctx), model.prepare_rope(
                [(f_grid, h_grid, w_grid), (f_grid, h_grid, w_grid)]
            )

        ctx_low, kv_low, rope_low = prep(low_model)
        ctx_high, kv_high, rope_high = prep(high_model)
        sched = FlowUniPCScheduler(num_train_timesteps=config.num_train_timesteps)
        sched.set_timesteps(STEPS, shift=shift)
        latents = noise
        routing = []
        for t in sched.timesteps.tolist():
            if t >= boundary:
                model, ctx, kv, rope, gs = high_model, ctx_high, kv_high, rope_high, guide_high
                routing.append(["high", t])
            else:
                model, ctx, kv, rope, gs = low_model, ctx_low, kv_low, rope_low, guide_low
                routing.append(["low", t])
            preds = model([latents, latents], t=mx.array([t, t]), context=ctx, seq_len=seq_len,
                          cross_kv_caches=kv, rope_cos_sin=rope)
            noise_pred = preds[1] + gs * (preds[0] - preds[1])
            latents = sched.step(noise_pred[None], t, latents[None]).squeeze(0)
            mx.eval(latents)
        assert any(r[0] == "high" for r in routing) and any(r[0] == "low" for r in routing), routing
        return latents, routing

    # --- LoRA-merged run (production path: per-expert --lora-high/--lora-low) ---
    print("Loading LoRA-merged experts (high/low) ...")
    low_lora = load_wan_model(model_dir / "low_noise_model.safetensors", config,
                              loras=[(lora_low, LORA_STRENGTH)])
    high_lora = load_wan_model(model_dir / "high_noise_model.safetensors", config,
                               loras=[(lora_high, LORA_STRENGTH)])
    lora_latents, routing = denoise(low_lora, high_lora)
    mx.eval(lora_latents)
    del low_lora, high_lora

    # --- Bare run (no LoRA) on the SAME noise — the visible-effect baseline ---
    print("Loading bare experts (no LoRA) ...")
    low_bare = load_wan_model(model_dir / "low_noise_model.safetensors", config)
    high_bare = load_wan_model(model_dir / "high_noise_model.safetensors", config)
    bare_latents, _ = denoise(low_bare, high_bare)
    mx.eval(bare_latents)
    del low_bare, high_bare

    # --- VAE decode the LoRA video ---
    print("Decoding LoRA latents with the z16 VAE...")
    vae = load_vae_decoder(model_dir / "vae.safetensors", config)
    lora_video = vae.decode(lora_latents[None])
    mx.eval(lora_video)

    # Visible-effect magnitude (reference-side): how far the LoRA moved the latents.
    eff = float((mx.sum(mx.abs(lora_latents - bare_latents)) / mx.sum(mx.abs(bare_latents))).item())
    print(f"LoRA visible effect (reference) mean_rel vs bare = {eff:.4f}")

    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(fixture_path),
        {
            "noise": noise,
            "context": context,
            "context_null": context_null,
            "lora_latents": lora_latents.astype(mx.float32),
            "bare_latents": bare_latents.astype(mx.float32),
            "lora_video": lora_video.astype(mx.float32),
        },
    )
    meta = {
        "prompt": PROMPT, "neg_prompt": neg_prompt,
        "frames": FRAMES, "height": HEIGHT, "width": WIDTH, "steps": STEPS,
        "shift": shift, "guide_low": guide_low, "guide_high": guide_high,
        "boundary": config.boundary, "boundary_timestep": boundary,
        "num_train_timesteps": config.num_train_timesteps, "seq_len": seq_len,
        "lora_high": lora_high, "lora_low": lora_low, "lora_strength": LORA_STRENGTH,
        "routing": routing, "reference_visible_effect": eff,
        "lora_latents_shape": list(lora_latents.shape), "lora_video_shape": list(lora_video.shape),
    }
    with open(fixture_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"routing={routing}")
    print(f"lora_latents {tuple(lora_latents.shape)}  lora_video {tuple(lora_video.shape)}")
    print(f"wrote {fixture_path}")


if __name__ == "__main__":
    main()
