#!/usr/bin/env python3
"""Dump an I2V-14B **real-weight** parity fixture (sc-2681): a small dual-expert MoE *image→video*
run of the `mlx_video` reference on the *actual converted* Wan2.2-I2V-A14B weights, for the Rust
`Wan14b::generate` I2V path (`build_i2v_y` + `denoise_moe` with `y` + the z16 VAE) to gate against
end-to-end.

Like the T2V `dump_s6_real_fixtures.py`, this loads the **real** converted checkpoint and runs the
genuine 40-layer / dim-5120 experts + the real UMT5-XXL text encoder + the real Wan2.1 z16 VAE
(decoder *and* encoder). It additionally runs the I2V channel-concat conditioning: a real input image
is cover-fit + center-cropped (PIL LANCZOS), VAE-encoded as a first-frame video, and concatenated
under a temporal mask → `y` `[20, T_lat, h, w]`. Only the init **noise** is injected (seeded RNG is
not portable across the mlx-python / mlx-rs split); everything else is the real chain. The Rust test
re-encodes the same prompt (real-weight T5 parity), rebuilds + preprocesses the same image
(`build_i2v_y` parity), injects this same noise, and compares the latents + decoded frames.

Run with the SceneWorks venv, pointing at the converted I2V model dir + an input image:

    WAN_I2V_MODEL_DIR=~/.cache/mlx-gen-models/wan2_2_i2v_a14b_mlx_bf16 \
    WAN_I2V_IMAGE=~/Pictures/fox.png \
    WAN_I2V_FIXTURE=/tmp/wan_i2v.safetensors \
    "$HOME/Library/Application Support/SceneWorks/python/venv/bin/python" \
        mlx-gen-wan/tools/dump_i2v_real_fixtures.py

Writes (NOT committed — tied to the 54 GB converted weights, which live outside the repo):
  - $WAN_I2V_FIXTURE                     (raw image + preprocessed + y + noise + context + golden)
  - ${WAN_I2V_FIXTURE%.safetensors}.json (metadata: prompt, geometry, routing, thresholds)

The Rust side (`tests/i2v_real_parity.rs`, #[ignore]) reads WAN_I2V_MODEL_DIR + WAN_I2V_FIXTURE.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

import mlx.core as mx

from mlx_video.models.wan.config import WanModelConfig
from mlx_video.models.wan.loading import (
    encode_text,
    load_t5_encoder,
    load_vae_decoder,
    load_vae_encoder,
    load_wan_model,
)
from mlx_video.models.wan.scheduler import FlowUniPCScheduler

# --- Small-but-real generation knobs (keep both experts firing; within max_area, 16-aligned) ---
PROMPT = "a red fox trotting across a snowy meadow at sunrise, cinematic"
FRAMES, HEIGHT, WIDTH = 5, 128, 128
STEPS = 6
SCHEDULER = "unipc"


def main():
    model_dir = Path(os.path.expanduser(os.environ["WAN_I2V_MODEL_DIR"]))
    image_path = Path(os.path.expanduser(os.environ["WAN_I2V_IMAGE"]))
    fixture_path = Path(os.path.expanduser(os.environ["WAN_I2V_FIXTURE"]))

    with open(model_dir / "config.json") as f:
        cfg_json = json.load(f)
    fields = WanModelConfig.__dataclass_fields__
    cdict = {k: v for k, v in cfg_json.items() if k in fields}
    for key in ("patch_size", "vae_stride", "window_size", "sample_guide_scale"):
        if key in cdict and isinstance(cdict[key], list):
            cdict[key] = tuple(cdict[key])
    config = WanModelConfig(**cdict)
    assert config.dual_model, "expected the dual-expert A14B config"
    assert config.model_type == "i2v", "expected the channel-concat I2V config (model_type=i2v)"

    shift = config.sample_shift
    guide = config.sample_guide_scale
    guide_low, guide_high = (guide, guide) if isinstance(guide, (int, float)) else guide
    neg_prompt = config.sample_neg_prompt
    boundary = config.boundary * config.num_train_timesteps

    vae_stride, patch, z_dim = config.vae_stride, config.patch_size, config.vae_z_dim
    t_lat = (FRAMES - 1) // vae_stride[0] + 1
    h_lat, w_lat = HEIGHT // vae_stride[1], WIDTH // vae_stride[2]
    seq_len = math.ceil((h_lat * w_lat) / (patch[1] * patch[2]) * t_lat)
    f_grid, h_grid, w_grid = t_lat // patch[0], h_lat // patch[1], w_lat // patch[2]

    # --- Real UMT5 encode ---
    print("Loading T5 + tokenizer, encoding prompt...")
    from transformers import AutoTokenizer

    t5 = load_t5_encoder(model_dir / "t5_encoder.safetensors", config)
    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    context = encode_text(t5, tokenizer, PROMPT, config.text_len)
    context_null = encode_text(t5, tokenizer, neg_prompt, config.text_len)
    mx.eval(context, context_null)
    del t5

    # --- I2V conditioning y (reference's is_i2v_channel_concat, verbatim) ---
    print("Building I2V channel-concat conditioning y...")
    img = PILImage.open(image_path).convert("RGB")
    img_uint8 = np.array(img, dtype=np.uint8)  # [ih, iw, 3] raw (dumped for the Rust preprocess)
    scale = max(WIDTH / img.width, HEIGHT / img.height)
    img_r = img.resize((round(img.width * scale), round(img.height * scale)), PILImage.LANCZOS)
    x1, y1 = (img_r.width - WIDTH) // 2, (img_r.height - HEIGHT) // 2
    img_r = img_r.crop((x1, y1, x1 + WIDTH, y1 + HEIGHT))
    img_chw = mx.array(np.array(img_r, dtype=np.float32) / 255.0 * 2.0 - 1.0).transpose(2, 0, 1)

    vae_enc = load_vae_encoder(model_dir / "vae.safetensors", config)
    video_in = mx.concatenate(
        [img_chw[:, None, :, :], mx.zeros((3, FRAMES - 1, HEIGHT, WIDTH))], axis=1
    )
    z_video = vae_enc.encode(video_in[None])[0]  # [16, T_lat, h_lat, w_lat]
    msk = mx.ones((1, FRAMES, h_lat, w_lat))
    msk = mx.concatenate([msk[:, :1], mx.zeros((1, FRAMES - 1, h_lat, w_lat))], axis=1)
    msk = mx.concatenate([mx.repeat(msk[:, :1], 4, axis=1), msk[:, 1:]], axis=1)
    msk = msk.reshape(1, msk.shape[1] // 4, 4, h_lat, w_lat).transpose(0, 2, 1, 3, 4)[0]
    y_i2v = mx.concatenate([msk, z_video], axis=0)  # [20, T_lat, h_lat, w_lat]
    mx.eval(y_i2v, img_chw)
    del vae_enc

    # --- Injected init noise (pure noise; I2V conditions via y) ---
    mx.random.seed(1234)
    noise = mx.random.normal((z_dim, t_lat, h_lat, w_lat)).astype(mx.float32)
    mx.eval(noise)

    # --- Both real experts; embed per expert; dual-expert loop with y ---
    print("Loading low/high experts (27 GB each)...")
    low_model = load_wan_model(model_dir / "low_noise_model.safetensors", config)
    high_model = load_wan_model(model_dir / "high_noise_model.safetensors", config)

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
            y=[y_i2v, y_i2v],
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

    # --- Real z16 VAE decode ---
    print("Decoding with the z16 VAE...")
    vae = load_vae_decoder(model_dir / "vae.safetensors", config)
    video = vae.decode(final_latents[None])
    mx.eval(video)

    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(fixture_path),
        {
            "img_uint8": mx.array(img_uint8),
            "img_chw": img_chw.astype(mx.float32),
            "y": y_i2v.astype(mx.float32),
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
        "image": str(image_path),
        "frames": FRAMES,
        "height": HEIGHT,
        "width": WIDTH,
        "img_h": int(img_uint8.shape[0]),
        "img_w": int(img_uint8.shape[1]),
        "steps": STEPS,
        "scheduler": SCHEDULER,
        "shift": shift,
        "guide_low": guide_low,
        "guide_high": guide_high,
        "boundary": config.boundary,
        "boundary_timestep": boundary,
        "num_train_timesteps": config.num_train_timesteps,
        "vae_stride": list(config.vae_stride),
        "seq_len": seq_len,
        "routing": routing,
        "y_shape": list(y_i2v.shape),
        "final_latents_shape": list(final_latents.shape),
        "video_shape": list(video.shape),
    }
    with open(fixture_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"routing={routing}")
    print(f"img {img_uint8.shape[1]}x{img_uint8.shape[0]} -> y {tuple(y_i2v.shape)}")
    print(f"final_latents {tuple(final_latents.shape)}  video {tuple(video.shape)}")
    print(f"wrote {fixture_path}\n      {fixture_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
