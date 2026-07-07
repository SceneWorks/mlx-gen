#!/usr/bin/env python
"""Dump a diffusers DC-AE **encode** golden for the mlx-gen-sana img2img encoder port (sc-10190).

Runs the real `AutoencoderDC` (`dc-ae-f32c32-sana-1.0`, the SANA-1.6B VAE) encoder on a FIXED,
seeded input image and saves, to a committed golden, the two artifacts that pin the encode path:

  * `image`  — the input `[1, 3, H, W]` (NCHW, `[-1, 1]`, the pipeline's preprocess output range).
  * `latent` — the RAW encoder output `[1, 32, H/32, W/32]` (NCHW), BEFORE the `scaling_factor`
               multiply (diffusers `AutoencoderDC.encode(x).latent`; the Rust port multiplies by
               `scaling_factor` in `encode_init_latents`, so the golden compares the raw encode).

The `mlx-gen-sana/tests/encode_parity.rs::encode_matches_diffusers` test (gated behind
`SANA_DCAE_WEIGHTS`) loads the SAME weights + this golden, runs `DcAeEncoder::encode` on the golden
`image`, and asserts `mean_rel`/`peak_rel` faithfulness (~5e-3 over Metal's reduced-precision matmul —
the same convention as `decode_parity.rs`). A large gap = a real port bug (a wrong pixel-unshuffle
packing, a missing out-shortcut, a stride-2-vs-unshuffle downsample mixup, etc.).

Pinned recipe (must match the Rust `encode_parity` test):
  seed = 1234   size = 256x256   input = tanh(randn) ∈ (-1, 1)

Run (from a venv with diffusers + torch; the vendored `_vendor/pid/.venv-pid` has both):
  SANA_VAE_DIR=~/.cache/huggingface/hub/models--Efficient-Large-Model--Sana_1600M_1024px_diffusers/snapshots/<hash>/vae \
  python tools/dump_dcae_encode_golden.py
"""

import os

import torch
from diffusers import AutoencoderDC
from safetensors.torch import save_file

SEED = 1234
SIZE = 256
DEFAULT_VAE_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Efficient-Large-Model--Sana_1600M_1024px_diffusers/"
    "snapshots/d1b54936033cd7d45410ecadd692c5c502a19a38/vae"
)
OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "mlx-gen-sana",
    "tests",
    "fixtures",
    "dcae_encode_golden.safetensors",
)


def main() -> None:
    vae_dir = os.environ.get("SANA_VAE_DIR", DEFAULT_VAE_DIR)
    vae = AutoencoderDC.from_pretrained(vae_dir, torch_dtype=torch.float32)
    vae.eval()

    # Fixed input image in (-1, 1) — the pipeline preprocess range. `tanh(randn)` keeps it strictly
    # inside the open interval (no saturation artifacts) while staying deterministic on SEED.
    g = torch.Generator().manual_seed(SEED)
    image = torch.tanh(torch.randn(1, 3, SIZE, SIZE, generator=g, dtype=torch.float32))

    with torch.no_grad():
        latent = vae.encode(image, return_dict=False)[0]  # RAW encoder latent, pre-scaling_factor

    print(f"image  {tuple(image.shape)}  latent {tuple(latent.shape)}")
    print(f"latent mean {latent.mean().item():.5f}  std {latent.std().item():.5f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(
        {"image": image.contiguous(), "latent": latent.contiguous()},
        OUT,
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
