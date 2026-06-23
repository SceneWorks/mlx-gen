"""Real-weight Krea 2 VAE decode parity golden (sc-7570, the `#[ignore]` gate).

Krea 2's VAE is the **Qwen-Image** `AutoencoderKLQwenImage` (the published `vae/config.json` declares
`_name_or_path = "Qwen/Qwen-Image"`); the Rust side reuses `mlx-gen-qwen-image`'s `QwenVae`. This dump
loads the Krea snapshot's *own* `vae/` weights through diffusers and decodes a fixed-seed scaled latent
exactly as the reference `autoencoder.py::QwenAutoencoder.decode` does — manual per-channel de-norm
(`latent · std + mean`) then `ae.decode` — so the Rust `QwenVae::decode(latent)` is compared against
the model Krea actually ships.

All-f32 (the Krea `vae/` shards are F32), so this isolates VAE *math* parity (matmul reduced-precision
only, no dtype skew).

    KREA_TURBO_DIR=~/.cache/huggingface/hub/models--krea--Krea-2-Turbo/snapshots/<rev> \
      ~/Repos/mflux/.venv/bin/python tools/dump_krea_vae_golden.py
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from diffusers import AutoencoderKLQwenImage
from safetensors.torch import save_file

from _paths import fixture

# A small grid keeps the golden tiny but exercises all 3 spatial-upsample stages: latent 16×16 → 128².
LATENT_H = 16
LATENT_W = 16
SEED = 0


@torch.no_grad()
def main():
    root = Path(os.environ["KREA_TURBO_DIR"])
    ae = AutoencoderKLQwenImage.from_pretrained(str(root), subfolder="vae").to(
        device="cpu", dtype=torch.float32
    ).eval()

    mean = torch.tensor(ae.config.latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)
    std = torch.tensor(ae.config.latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1)

    gen = torch.Generator(device="cpu").manual_seed(SEED)
    # The scaled 16-ch latent (the DiT output space) the Rust `QwenVae::decode` consumes.
    latent = torch.randn(1, 16, LATENT_H, LATENT_W, generator=gen, dtype=torch.float32)

    # Reference `QwenAutoencoder.decode`: NCHW → NCTHW, manual de-norm, decode, drop the T axis.
    # NB: diffusers' `.decode().sample` is **internally clamped to [-1,1]** (verified: `.sample ==
    # clip(.sample)`), whereas the fork-faithful Rust `QwenVae::decode` returns the raw decoder output
    # and the Krea pipeline clamps *after* decode (`sampling.py`: `img.clamp(-1,1)`). The Rust
    # real-weight test therefore clamps its decode before comparing to this (already-clamped) golden.
    x = latent.view(1, 16, 1, LATENT_H, LATENT_W)
    x = x * std + mean
    image = ae.decode(x).sample  # [1, 3, 1, H, W], clamped to [-1, 1]

    tensors = {
        "in.latent": latent.contiguous(),  # [1, 16, h, w] (NCHW, pre-denorm)
        "out.image": image.to(torch.float32).contiguous(),  # [1, 3, 1, H, W] (NCTHW, T=1)
    }
    path = fixture("tools/golden/krea_vae_real.safetensors")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path)
    print(f"wrote {path}  (latent {tuple(latent.shape)} → image {tuple(image.shape)})")


if __name__ == "__main__":
    main()
