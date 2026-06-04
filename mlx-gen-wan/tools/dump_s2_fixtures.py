#!/usr/bin/env python3
"""Dump S2 parity fixtures from the `mlx_video` Wan reference: the 2.1 `WanVAE` (z16, stride
4×8×8) decode + chunked encode, for the Rust port to gate against.

Unlike S1/S3 (which load the real ~11 GB 5B weights), the z16 WanVAE's production weights are not
on disk (only the 5B's z48 vae22 is). The architecture is dimension-parametric, so we gate against
a **self-contained tiny instance**: a `dim=4`, `z_dim=16` WanVAE with deterministically-seeded
random weights. Tiny `dim` keeps the committed fixture ~1 MB while exercising **every** code path
(causal 3-D conv, channel-L2 norm, per-frame spatial attention, temporal up/down `time_conv`,
the chunked-encode `feat_cache`, mean/std denorm, and the full decoder/encoder key hierarchy). The
real per-channel `VAE_MEAN`/`VAE_STD` (16 entries) are kept (not randomized) — they are constants
the Rust port hardcodes, so this also gates those values.

Run with the SceneWorks venv that has `mlx_video` + `mlx` installed:

    "$HOME/Library/Application Support/SceneWorks/python/venv/bin/python" \
        mlx-gen-wan/tools/dump_s2_fixtures.py

Writes (committed; small):
  - mlx-gen-wan/tests/fixtures/s2.json                (dim, z_dim, io shapes, VAE_MEAN/STD)
  - mlx-gen-wan/tests/fixtures/s2_vae.safetensors     (random weights + decode/encode in+out, f32)
"""
import json
import os

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from mlx_video.models.wan.vae import (
    VAE_MEAN,
    VAE_STD,
    CausalConv3d,
    Decoder3d,
    Encoder3d,
    WanVAE,
)

DIM = 4        # tiny (production is 96) — keeps the fixture small; architecture is dim-parametric
Z_DIM = 16     # the 2.1 WanVAE z_dim (fixed: VAE_MEAN/STD are 16-long)
KEEP = {"mean", "std", "inv_std"}  # real constants, not randomized


def build_tiny() -> WanVAE:
    """A WanVAE whose sub-nets use dim=DIM (vs the hardcoded 96), with seeded random weights."""
    mx.random.seed(0)
    vae = WanVAE(z_dim=Z_DIM, encoder=True)
    # Swap the hardcoded dim=96 sub-nets for tiny ones (same architecture, smaller channels).
    vae.decoder = Decoder3d(dim=DIM, z_dim=Z_DIM)
    vae.encoder = Encoder3d(dim=DIM, z_dim=Z_DIM * 2)
    vae.conv1 = CausalConv3d(Z_DIM * 2, Z_DIM * 2, 1)
    vae.conv2 = CausalConv3d(Z_DIM, Z_DIM, 1)

    # Randomize every learnable param (conv weight/bias, norm gamma); keep mean/std/inv_std real.
    flat = tree_flatten(vae.parameters())
    new = []
    for k, v in flat:
        leaf = k.rsplit(".", 1)[-1]
        if k in KEEP or leaf in KEEP:
            new.append((k, v))
        else:
            new.append((k, (mx.random.normal(v.shape) * 0.5).astype(mx.float32)))
    vae.update(tree_unflatten(new))
    mx.eval(vae.parameters())
    return vae


def main():
    vae = build_tiny()

    flat = dict(tree_flatten(vae.parameters()))
    print(f"=== {len(flat)} weight tensors (dim={DIM}, z_dim={Z_DIM}) ===")
    for k in sorted(flat):
        print(f"  {k}\t{tuple(flat[k].shape)}")

    # Decode: a small normalized latent [B, z_dim, T, H, W] → video [B, 3, 4T, 8H, 8W].
    mx.random.seed(1)
    dec_in = (mx.random.normal((1, Z_DIM, 2, 4, 4)) * 0.5).astype(mx.float32)
    dec_out = vae.decode(dec_in)
    mx.eval(dec_out)

    # Encode: a small video [B, 3, T, H, W] (T = 1+4k) → normalized latent. Chunked feat_cache path.
    mx.random.seed(2)
    enc_in = mx.random.normal((1, 3, 9, 32, 32)).astype(mx.float32)
    enc_in = mx.clip(enc_in, -1.0, 1.0)
    enc_out = vae.encode(enc_in)
    mx.eval(enc_out)

    print(f"dec_in {tuple(dec_in.shape)} -> dec_out {tuple(dec_out.shape)}")
    print(f"enc_in {tuple(enc_in.shape)} -> enc_out {tuple(enc_out.shape)}")

    save = {k: v.astype(mx.float32) for k, v in flat.items()}
    save["dec_in"] = dec_in
    save["dec_out"] = dec_out
    save["enc_in"] = enc_in
    save["enc_out"] = enc_out

    dst = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
    os.makedirs(dst, exist_ok=True)
    st_path = os.path.join(dst, "s2_vae.safetensors")
    mx.save_safetensors(st_path, save)

    meta = {
        "dim": DIM,
        "z_dim": Z_DIM,
        "dec_in_shape": list(dec_in.shape),
        "dec_out_shape": list(dec_out.shape),
        "enc_in_shape": list(enc_in.shape),
        "enc_out_shape": list(enc_out.shape),
        "vae_mean": list(VAE_MEAN),
        "vae_std": list(VAE_STD),
        "num_weight_tensors": len(flat),
    }
    with open(os.path.join(dst, "s2.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {os.path.abspath(st_path)} ({os.path.getsize(st_path) / 1e6:.2f} MB)")
    print(f"wrote {os.path.abspath(os.path.join(dst, 's2.json'))}")


if __name__ == "__main__":
    main()
