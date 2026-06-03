#!/usr/bin/env python3
"""Dump S0 parity fixtures from the `mlx_video` Wan reference (sigmas, integer timesteps, 3-axis
RoPE cos/sin, 3-D patchify/unpatchify reordering) for the Rust port to gate against.

Run with the SceneWorks venv that has `mlx_video` installed:

    "$HOME/Library/Application Support/SceneWorks/python/venv/bin/python" \
        mlx-gen-wan/tools/dump_s0_fixtures.py

Writes mlx-gen-wan/tests/fixtures/s0.json (committed; small).
"""
import json
import os

import numpy as np
import mlx.core as mx

from mlx_video.models.wan.scheduler import _compute_sigmas
from mlx_video.models.wan.rope import rope_params, rope_precompute_cos_sin


def sigmas_and_timesteps(num_steps, shift, num_train=1000):
    sig = _compute_sigmas(num_steps, shift, num_train)  # np.float32, len num_steps+1
    ts = (sig[:-1] * num_train).astype(np.int64).astype(np.float32)
    return sig.tolist(), ts.tolist()


def build_freqs(head_dim):
    """WanModel.freqs: three rope_params tables concatenated along the frequency axis."""
    d = head_dim
    return mx.concatenate(
        [
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ],
        axis=1,
    )


def rope_cos_sin(head_dim, grid):
    freqs = build_freqs(head_dim)
    cos, sin = rope_precompute_cos_sin([grid], freqs, dtype=mx.float32)
    # cos/sin shape [seq_len, 1, half_d]
    return (
        list(cos.shape),
        np.array(cos, dtype=np.float32).reshape(-1).tolist(),
        np.array(sin, dtype=np.float32).reshape(-1).tolist(),
    )


def patchify_reorder(x, patch_size):
    """WanModel._patchify reshape/transpose (no patch-embedding Linear), via mlx ops."""
    c, f, h, w = x.shape
    pt, ph, pw = patch_size
    fo, ho, wo = f // pt, h // ph, w // pw
    y = x.reshape(c, fo, pt, ho, ph, wo, pw)
    y = y.transpose(1, 3, 5, 0, 2, 4, 6)
    y = y.reshape(fo * ho * wo, -1)
    return (fo, ho, wo), y


def unpatchify_reorder(u, grid, out_dim, patch_size):
    """WanModel.unpatchify reshape/transpose, via mlx ops."""
    f, h, w = grid
    pt, ph, pw = patch_size
    c = out_dim
    y = u.reshape(f, h, w, pt, ph, pw, c)
    y = y.transpose(6, 0, 3, 1, 4, 2, 5)
    y = y.reshape(c, f * pt, h * ph, w * pw)
    return y


def main():
    out = {}

    # --- Sigmas + integer timesteps (5B = shift 5.0 / 40 steps; plus a couple alternates) ---
    out["sigmas"] = {}
    for name, (n, shift) in {
        "ti2v5b_40_shift5": (40, 5.0),
        "t2v14b_40_shift12": (40, 12.0),
        "euler_8_shift5": (8, 5.0),
    }.items():
        sig, ts = sigmas_and_timesteps(n, shift)
        out["sigmas"][name] = {"num_steps": n, "shift": shift, "sigmas": sig, "timesteps": ts}

    # --- 3-axis RoPE cos/sin (head_dim 128, small grid) ---
    shape, cos, sin = rope_cos_sin(128, (2, 3, 4))
    out["rope"] = {"head_dim": 128, "grid": [2, 3, 4], "shape": shape, "cos": cos, "sin": sin}

    # --- 3-D patchify reordering (C=2, F=2, H=4, W=4, patch (1,2,2)) ---
    n = 2 * 2 * 4 * 4
    x = mx.array(np.arange(n, dtype=np.float32).reshape(2, 2, 4, 4))
    grid, tokens = patchify_reorder(x, (1, 2, 2))
    out["patchify"] = {
        "in_shape": [2, 2, 4, 4],
        "patch_size": [1, 2, 2],
        "grid": list(grid),
        "tokens_shape": list(tokens.shape),
        "tokens": np.array(tokens, dtype=np.float32).reshape(-1).tolist(),
    }

    # --- 3-D unpatchify reordering (inverse layout; head tokens → video) ---
    # tokens_in shape [L, out_dim*pt*ph*pw]; use grid (1,2,2), out_dim 2, patch (1,2,2) → L=4, ch=8.
    lout = 1 * 2 * 2
    ch = 2 * 1 * 2 * 2
    u = mx.array(np.arange(lout * ch, dtype=np.float32).reshape(lout, ch))
    vid = unpatchify_reorder(u, (1, 2, 2), 2, (1, 2, 2))
    out["unpatchify"] = {
        "tokens_shape": [lout, ch],
        "grid": [1, 2, 2],
        "out_dim": 2,
        "patch_size": [1, 2, 2],
        "video_shape": list(vid.shape),
        "video": np.array(vid, dtype=np.float32).reshape(-1).tolist(),
    }

    dst = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
    os.makedirs(dst, exist_ok=True)
    path = os.path.join(dst, "s0.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"wrote {os.path.abspath(path)}")
    print("  sigmas:", list(out["sigmas"].keys()))
    print("  rope shape:", out["rope"]["shape"])
    print("  patchify tokens_shape:", out["patchify"]["tokens_shape"])
    print("  unpatchify video_shape:", out["unpatchify"]["video_shape"])


if __name__ == "__main__":
    main()
