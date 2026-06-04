#!/usr/bin/env python3
"""Dump the sc-2808 tiling golden: the `mlx_video` reference `WanVAE.decode_tiled` (z16, non-causal
`T→4T`, spatial ×8) run on the same tiny `dim=4` seeded VAE as the S2 fixture, for the Rust
`WanVae::decode_tiled` to gate against **exactly** (same overlap + trapezoidal blend, so bit-for-bit
up to the conv float-ordering gap — like the S2 decode gate, not the inherently-divergent
tiled-vs-untiled comparison a random-weight VAE would show).

Self-contained `s2_tiling.safetensors` = the tiny VAE weights + the normalized tiled latent
(`tiled_in`) + the reference tiled video (`tiled_out`). The Rust test rebuilds the VAE from these
weights and decodes `tiled_in` with the matching `TilingConfig`.

Run with the SceneWorks venv (has `mlx_video`):

    "$HOME/Library/Application Support/SceneWorks/python/venv/bin/python" \
        mlx-gen-wan/tools/dump_s2_tiling_fixtures.py

Writes (committed; small):
  - mlx-gen-wan/tests/fixtures/s2_tiling.safetensors
  - mlx-gen-wan/tests/fixtures/s2_tiling.json   (the tiling config + io shapes)
"""
import json
import os

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from mlx_video.models.wan.tiling import (
    SpatialTilingConfig,
    TemporalTilingConfig,
    TilingConfig,
)
from mlx_video.models.wan.vae import (
    CausalConv3d,
    Decoder3d,
    Encoder3d,
    WanVAE,
)

DIM = 4
Z_DIM = 16
KEEP = {"mean", "std", "inv_std"}

# Reference-valid tiling: tile_px ≥ 64 (÷32), tile_frames ≥ 16 (÷8). With Wan scales (spatial 8,
# temporal 4): spatial latent tile 64/8=8 / overlap 4; temporal latent tile 16/4=4 / overlap 2.
SPATIAL_TILE_PX, SPATIAL_OVERLAP_PX = 64, 32
TEMPORAL_TILE_FRAMES, TEMPORAL_OVERLAP_FRAMES = 16, 8
# A latent larger than the tile on every axis → 2×2×2 = 8 tiles (h=w=12 > 8, f=6 > 4).
F_LAT, H_LAT, W_LAT = 6, 12, 12


def build_tiny() -> WanVAE:
    """Identical construction to dump_s2_fixtures.py (seed 0, dim=4) — same weights."""
    mx.random.seed(0)
    vae = WanVAE(z_dim=Z_DIM, encoder=True)
    vae.decoder = Decoder3d(dim=DIM, z_dim=Z_DIM)
    vae.encoder = Encoder3d(dim=DIM, z_dim=Z_DIM * 2)
    vae.conv1 = CausalConv3d(Z_DIM * 2, Z_DIM * 2, 1)
    vae.conv2 = CausalConv3d(Z_DIM, Z_DIM, 1)
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

    cfg = TilingConfig(
        spatial_config=SpatialTilingConfig(
            tile_size_in_pixels=SPATIAL_TILE_PX,
            tile_overlap_in_pixels=SPATIAL_OVERLAP_PX,
        ),
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=TEMPORAL_TILE_FRAMES,
            tile_overlap_in_frames=TEMPORAL_OVERLAP_FRAMES,
        ),
    )

    mx.random.seed(11)
    tiled_in = (mx.random.normal((1, Z_DIM, F_LAT, H_LAT, W_LAT)) * 0.5).astype(mx.float32)
    tiled_out = vae.decode_tiled(tiled_in, cfg)
    mx.eval(tiled_out)
    print(f"tiled_in {tuple(tiled_in.shape)} -> tiled_out {tuple(tiled_out.shape)}")

    flat = dict(tree_flatten(vae.parameters()))
    save = {k: v.astype(mx.float32) for k, v in flat.items()}
    save["tiled_in"] = tiled_in
    save["tiled_out"] = tiled_out

    dst = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
    os.makedirs(dst, exist_ok=True)
    st_path = os.path.join(dst, "s2_tiling.safetensors")
    mx.save_safetensors(st_path, save)

    meta = {
        "dim": DIM,
        "z_dim": Z_DIM,
        "spatial_tile_px": SPATIAL_TILE_PX,
        "spatial_overlap_px": SPATIAL_OVERLAP_PX,
        "temporal_tile_frames": TEMPORAL_TILE_FRAMES,
        "temporal_overlap_frames": TEMPORAL_OVERLAP_FRAMES,
        "tiled_in_shape": list(tiled_in.shape),
        "tiled_out_shape": list(tiled_out.shape),
    }
    with open(os.path.join(dst, "s2_tiling.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {os.path.abspath(st_path)} ({os.path.getsize(st_path) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
