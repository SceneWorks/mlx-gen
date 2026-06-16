"""sc-5987 — golden dump for the Ideogram 4 VAE decode parity test.

Ideogram 4's VAE is `AutoencoderKLFlux2` (same as FLUX.2). This loads the converted bf16 VAE into
the diffusers reference (f32), decodes a seeded latent, and saves it (NHWC) alongside the input —
the Rust test reuses `mlx-gen-flux2::Flux2Vae` (the proven port) and must match. `decode` is
post_quant_conv + decoder (no bn / scaling_factor=1.0); bn de-normalization is the pipeline's job.

Run:
  ~/mlx-flux-venv/bin/python tools/dump_ideogram4_vae_golden.py \
      --converted ~/.cache/ideogram4-mlx-convert --out tools/golden/ideogram4_vae.safetensors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import torch
from safetensors.torch import load_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--converted", type=Path, default=Path.home() / ".cache/ideogram4-mlx-convert")
    ap.add_argument("--out", type=Path, default=Path("tools/golden/ideogram4_vae.safetensors"))
    args = ap.parse_args()

    vae_dir = args.converted / "vae"
    if not (vae_dir / "model.safetensors").exists():
        sys.exit(f"converted VAE not found: {vae_dir}")

    from diffusers import AutoencoderKLFlux2

    print(f"loading AutoencoderKLFlux2 (f32) from {vae_dir} …")
    # Build from config + load the converted state dict directly (the converter writes
    # `model.safetensors`, not the `diffusion_pytorch_model.*` name `from_pretrained` expects).
    cfg = json.loads((vae_dir / "config.json").read_text())
    vae = AutoencoderKLFlux2.from_config(cfg)  # f32 params; ignores `_class_name` etc.
    missing, unexpected = vae.load_state_dict(load_file(str(vae_dir / "model.safetensors")), strict=False)
    if missing or unexpected:
        sys.exit(f"VAE state_dict mismatch  missing={missing}  unexpected={unexpected}")
    vae = vae.eval()

    torch.manual_seed(0)
    latent_channels = vae.config.latent_channels  # 32
    z_nchw = torch.randn(1, latent_channels, 4, 4)
    with torch.no_grad():
        img_nchw = vae.decode(z_nchw).sample  # [1, 3, 32, 32]
    print(f"decoded: {tuple(img_nchw.shape)} (expect [1,3,32,32])")

    # Store NHWC to match the Rust `Flux2Vae` I/O (avoids layout-order skew in the cosine).
    z_nhwc = z_nchw.permute(0, 2, 3, 1).contiguous()       # [1,4,4,32]
    img_nhwc = img_nchw.permute(0, 2, 3, 1).contiguous()   # [1,32,32,3]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(args.out),
        {"z": mx.array(z_nhwc.numpy()), "golden": mx.array(img_nhwc.numpy())},
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
