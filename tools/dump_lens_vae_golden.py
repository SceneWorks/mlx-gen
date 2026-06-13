#!/usr/bin/env python
"""Dump a VAE-decode golden for the Lens decode shim (mlx-gen sc-3169).

Replicates `LensPipeline._decode` with the real `AutoencoderKLFlux2` (the cached `microsoft/Lens-Turbo`
`vae/`) in **float32** on a synthetic DiT-output latent, and records the input latent + decoded image
so the Rust `mlx_gen_lens::vae::decode` (reusing `mlx_gen_flux2::Flux2Vae::decode_packed_latents`) can
be checked near-bit. Self-contained: no pipeline instance, no transformer/encoder.

Golden contents:
  - `dit_out`  [1, h·w, 128] — synthetic packed transformer output;
  - `image`    [1, 3, H, W]  — `_decode(...).sample` in [-1, 1] (NCHW; the Rust side is NHWC);
  - metadata: latent_h, latent_w, H, W.

Run (from repo root):
  ~/Repos/mflux/.venv/bin/python tools/dump_lens_vae_golden.py
Writes `tools/golden/lens_vae_golden.safetensors` (gitignored real-weights golden).
"""

from __future__ import annotations

import glob
import os

import torch
from diffusers import AutoencoderKLFlux2
from safetensors.torch import save_file

HOME = os.path.expanduser("~")
VAE_GLOB = f"{HOME}/.cache/huggingface/hub/models--microsoft--Lens-Turbo/snapshots/*/vae"
OUT = os.path.join(os.path.dirname(__file__), "golden", "lens_vae_golden.safetensors")

LATENT_H, LATENT_W = 8, 8  # → image 128×128


def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(b, c * 4, h // 2, w // 2)


def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c // 4, 2, 2, h, w)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(b, c // 4, h * 2, w * 2)


def main() -> None:
    matches = sorted(glob.glob(VAE_GLOB))
    if not matches:
        raise SystemExit(f"no Lens-Turbo vae snapshot at {VAE_GLOB}")
    vae = AutoencoderKLFlux2.from_pretrained(matches[-1], torch_dtype=torch.float32).eval()

    torch.manual_seed(0)
    dit_out = torch.randn(1, LATENT_H * LATENT_W, 128, dtype=torch.float32)

    with torch.no_grad():
        # `_decode`: rearrange "b (h w) (c p1 p2) -> b c (h p1) (w p2)" (p1=p2=2) — done by reshape.
        b = dit_out.shape[0]
        latents = (
            dit_out.view(b, LATENT_H, LATENT_W, 32, 2, 2)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(b, 32, LATENT_H * 2, LATENT_W * 2)
        )
        bn = vae.bn
        mean = bn.running_mean.view(1, -1, 1, 1)
        std = torch.sqrt(bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
        shift = -mean
        scale = 1.0 / std
        x = patchify_latents(latents)
        x = x / scale - shift
        x = unpatchify_latents(x)
        image = vae.decode(x).sample  # [1, 3, H, W] in [-1, 1]

    h_img, w_img = image.shape[2], image.shape[3]
    tensors = {"dit_out": dit_out.contiguous(), "image": image.contiguous()}
    meta = {
        "latent_h": str(LATENT_H),
        "latent_w": str(LATENT_W),
        "H": str(h_img),
        "W": str(w_img),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(tensors, OUT, metadata=meta)
    print(f"wrote {OUT}  (dit_out=[1,{LATENT_H*LATENT_W},128], image=[1,3,{h_img},{w_img}])")


if __name__ == "__main__":
    main()
