#!/usr/bin/env python3
"""Dump I2V-14B parity fixtures (sc-2681): a full **channel-concat image→video** denoise + decode run
of the `mlx_video` reference, on two tiny seeded dual-expert models + a tiny z16 `WanVAE` (with
encoder), for the Rust `pipeline::{preprocess_i2v_image, build_i2v_y}` + `denoise_moe` (with `y`) to
gate against.

This is the S5 dual-expert MoE fixture extended to the I2V channel-concat path: the two tiny DiTs use
**in_dim = 36** (16 noise + 20 conditioning `y`), and the run reproduces `generate_wan.py`'s
`is_i2v_channel_concat` setup exactly — a synthetic RGB image is cover-fit + center-cropped (PIL
LANCZOS), made into a first-frame-only video, encoded by the 2.1 WanVAE → `z_video`, concatenated
under a 4-channel temporal mask → `y = [mask(4), z_video(16)]` → `[20, T_lat, h, w]`, then `y` is
channel-concatenated onto the noise latent inside the dual-expert boundary-switched loop.

Run with the SceneWorks venv (has `mlx_video` + `mlx` + `PIL`):
    "$HOME/Library/Application Support/SceneWorks/python/venv/bin/python" \
        mlx-gen-wan/tools/dump_i2v_fixtures.py

Writes (committed; small):
  - mlx-gen-wan/tests/fixtures/i2v.json
  - mlx-gen-wan/tests/fixtures/i2v_low.safetensors   (low DiT + VAE + image/y/io + golden)
  - mlx-gen-wan/tests/fixtures/i2v_high.safetensors  (high DiT)
"""
import dataclasses
import json
import math
import os

import numpy as np
from PIL import Image as PILImage

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from mlx_video.models.wan.config import WanModelConfig
from mlx_video.models.wan.model import WanModel
from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler
from mlx_video.models.wan.vae import CausalConv3d, Decoder3d, Encoder3d, WanVAE

# Tiny dual-expert I2V config: like the S5 base (boundary 0.9 for I2V), but in_dim = 36 (16 noise +
# 20 conditioning) and out_dim = 16.
CFG = dataclasses.replace(
    WanModelConfig.wan22_t2v_14b(),
    model_type="i2v",
    dim=128,
    num_heads=1,
    num_layers=2,
    ffn_dim=256,
    freq_dim=256,
    text_dim=32,
    text_len=8,
    in_dim=36,
    out_dim=16,
    vae_z_dim=16,
    boundary=0.9,
)
VAE_DIM = 4
Z_DIM = 16
STEPS = 4
SHIFT = 5.0
GUIDE_LOW = 3.5
GUIDE_HIGH = 3.5
FRAMES, HEIGHT, WIDTH = 5, 16, 16
IMG_H, IMG_W = 40, 48  # synthetic image dims (≠ target → exercises cover-fit + center-crop)
CTX_TOKENS = 4
KEEP = {"mean", "std", "inv_std"}
RANDN = lambda *s: (mx.random.normal(s)).astype(mx.float32)  # noqa: E731


def seeded_dit(seed: int) -> WanModel:
    mx.random.seed(seed)
    m = WanModel(CFG)
    flat = tree_flatten(m.parameters())
    m.update(tree_unflatten([(k, (mx.random.normal(v.shape) * 0.1)) for k, v in flat]))
    mx.eval(m.parameters())
    return m


def seeded_vae() -> WanVAE:
    """Tiny WanVAE *with encoder* (I2V needs encode), seeded — mirrors dump_s2's build_tiny."""
    mx.random.seed(7)
    vae = WanVAE(z_dim=Z_DIM, encoder=True)
    vae.decoder = Decoder3d(dim=VAE_DIM, z_dim=Z_DIM)
    vae.encoder = Encoder3d(dim=VAE_DIM, z_dim=Z_DIM * 2)
    vae.conv1 = CausalConv3d(Z_DIM * 2, Z_DIM * 2, 1)
    vae.conv2 = CausalConv3d(Z_DIM, Z_DIM, 1)
    new = []
    for k, v in tree_flatten(vae.parameters()):
        leaf = k.rsplit(".", 1)[-1]
        if k in KEEP or leaf in KEEP:
            new.append((k, v))
        else:
            new.append((k, (mx.random.normal(v.shape) * 0.5).astype(mx.float32)))
    vae.update(tree_unflatten(new))
    mx.eval(vae.parameters())
    return vae


def build_y(vae: WanVAE, img_chw: mx.array) -> mx.array:
    """Port of generate_wan.py's is_i2v_channel_concat y-build (the reference, verbatim)."""
    h_lat, w_lat = HEIGHT // CFG.vae_stride[1], WIDTH // CFG.vae_stride[2]
    # Conditioning video: first frame = image, rest zeros -> [3, F, H, W].
    video = mx.concatenate(
        [img_chw[:, None, :, :], mx.zeros((3, FRAMES - 1, HEIGHT, WIDTH))], axis=1
    )
    z_video = vae.encode(video[None])  # [1, 16, T_lat, h_lat, w_lat]
    mx.eval(z_video)
    z_video = z_video[0]
    # Temporal mask: 1 for first frame, 0 for rest -> [4, T_lat, h_lat, w_lat].
    msk = mx.ones((1, FRAMES, h_lat, w_lat))
    msk = mx.concatenate([msk[:, :1], mx.zeros((1, FRAMES - 1, h_lat, w_lat))], axis=1)
    msk = mx.concatenate([mx.repeat(msk[:, :1], 4, axis=1), msk[:, 1:]], axis=1)
    msk = msk.reshape(1, msk.shape[1] // 4, 4, h_lat, w_lat)
    msk = msk.transpose(0, 2, 1, 3, 4)[0]  # [4, T_lat, h_lat, w_lat]
    y = mx.concatenate([msk, z_video], axis=0)  # [20, T_lat, h_lat, w_lat]
    mx.eval(y)
    return y


def main():
    low_model = seeded_dit(10)
    high_model = seeded_dit(20)
    vae = seeded_vae()

    vae_stride, patch = CFG.vae_stride, CFG.patch_size
    t_lat = (FRAMES - 1) // vae_stride[0] + 1
    h_lat, w_lat = HEIGHT // vae_stride[1], WIDTH // vae_stride[2]
    seq_len = math.ceil((h_lat * w_lat) / (patch[1] * patch[2]) * t_lat)
    grid = (t_lat // patch[0], h_lat // patch[1], w_lat // patch[2])
    boundary = CFG.boundary * CFG.num_train_timesteps

    # Synthetic input image (deterministic uint8 [IMG_H, IMG_W, 3]); preprocess EXACTLY as the
    # reference (cover-fit LANCZOS resize + center-crop + normalize to [-1, 1], CHW).
    rng = np.random.default_rng(123)
    img_uint8 = rng.integers(0, 256, size=(IMG_H, IMG_W, 3), dtype=np.uint8)
    pil = PILImage.fromarray(img_uint8)
    scale = max(WIDTH / pil.width, HEIGHT / pil.height)
    pil = pil.resize((round(pil.width * scale), round(pil.height * scale)), PILImage.LANCZOS)
    x1, y1 = (pil.width - WIDTH) // 2, (pil.height - HEIGHT) // 2
    pil = pil.crop((x1, y1, x1 + WIDTH, y1 + HEIGHT))
    img_arr = mx.array(np.array(pil, dtype=np.float32) / 255.0 * 2.0 - 1.0)  # [H, W, 3]
    img_chw = img_arr.transpose(2, 0, 1)  # [3, H, W]
    mx.eval(img_chw)

    y_i2v = build_y(vae, img_chw)
    assert tuple(y_i2v.shape) == (20, t_lat, h_lat, w_lat), y_i2v.shape

    mx.random.seed(2)
    ctx_cond, ctx_uncond = RANDN(CTX_TOKENS, CFG.text_dim), RANDN(CTX_TOKENS, CFG.text_dim)
    mx.random.seed(3)
    init_noise = RANDN(Z_DIM, t_lat, h_lat, w_lat)  # pure noise (I2V starts from noise)
    mx.eval(ctx_cond, ctx_uncond, init_noise)

    def prep(model):
        emb = model.embed_text([ctx_cond, ctx_uncond])
        ccfg = mx.concatenate([emb[0:1], emb[1:2]], axis=0)
        return ccfg, model.prepare_cross_kv(ccfg), model.prepare_rope([grid, grid])

    ctx_low, kv_low, rope_low = prep(low_model)
    ctx_high, kv_high, rope_high = prep(high_model)

    sched = FlowMatchEulerScheduler(num_train_timesteps=CFG.num_train_timesteps)
    sched.set_timesteps(STEPS, shift=SHIFT)

    latents = init_noise
    routing = []
    for t in sched.timesteps.tolist():
        if t >= boundary:
            model, ctx, kv, rope, gs = high_model, ctx_high, kv_high, rope_high, GUIDE_HIGH
            routing.append(("high", t))
        else:
            model, ctx, kv, rope, gs = low_model, ctx_low, kv_low, rope_low, GUIDE_LOW
            routing.append(("low", t))
        preds = model(
            [latents, latents],
            t=mx.array([t, t]),
            context=ctx,
            seq_len=seq_len,
            cross_kv_caches=kv,
            y=[y_i2v, y_i2v],  # I2V channel-concat conditioning
            rope_cos_sin=rope,
        )
        noise_pred = preds[1] + gs * (preds[0] - preds[1])
        latents = sched.step(noise_pred[None], t, latents[None]).squeeze(0)
        mx.eval(latents)
    final_latents = latents
    video = vae.decode(final_latents[None])
    mx.eval(video)

    print(f"boundary={boundary}  routing={routing}")
    assert any(r[0] == "high" for r in routing) and any(r[0] == "low" for r in routing), (
        "fixture must exercise BOTH experts across the boundary"
    )

    dst = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
    os.makedirs(dst, exist_ok=True)

    low_save = {k: v.astype(mx.bfloat16) for k, v in tree_flatten(low_model.parameters())}
    for k, v in tree_flatten(vae.parameters()):
        low_save[k] = v.astype(mx.float32)
    low_save["img_uint8"] = mx.array(img_uint8)  # [IMG_H, IMG_W, 3] uint8 (raw input)
    low_save["img_chw"] = img_chw.astype(mx.float32)  # [3, H, W] golden preprocessed
    low_save["y"] = y_i2v.astype(mx.float32)  # [20, T_lat, h, w] golden conditioning
    low_save["ctx_cond"] = ctx_cond
    low_save["ctx_uncond"] = ctx_uncond
    low_save["init_noise"] = init_noise
    low_save["final_latents"] = final_latents.astype(mx.float32)
    low_save["video"] = video.astype(mx.float32)
    mx.save_safetensors(os.path.join(dst, "i2v_low.safetensors"), low_save)

    high_save = {k: v.astype(mx.bfloat16) for k, v in tree_flatten(high_model.parameters())}
    mx.save_safetensors(os.path.join(dst, "i2v_high.safetensors"), high_save)

    meta = {
        "steps": STEPS,
        "shift": SHIFT,
        "guide_low": GUIDE_LOW,
        "guide_high": GUIDE_HIGH,
        "boundary": CFG.boundary,
        "boundary_timestep": boundary,
        "num_train_timesteps": CFG.num_train_timesteps,
        "in_dim": CFG.in_dim,
        "frames": FRAMES,
        "height": HEIGHT,
        "width": WIDTH,
        "img_h": IMG_H,
        "img_w": IMG_W,
        "seq_len": seq_len,
        "grid": list(grid),
        "vae_stride": list(CFG.vae_stride),
        "routing": [[r[0], r[1]] for r in routing],
        "y_shape": list(y_i2v.shape),
        "final_latents_shape": list(final_latents.shape),
        "video_shape": list(video.shape),
    }
    with open(os.path.join(dst, "i2v.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"img {IMG_W}x{IMG_H} -> y {tuple(y_i2v.shape)}")
    print(f"final_latents {tuple(final_latents.shape)}  video {tuple(video.shape)}")
    print(f"wrote i2v_low/high.safetensors + i2v.json to {os.path.abspath(dst)}")


if __name__ == "__main__":
    main()
