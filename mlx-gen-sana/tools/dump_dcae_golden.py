#!/usr/bin/env python3
"""Dump a DC-AE decoder golden (latent + reference decode) for the sc-8486 parity gate.

Loads diffusers `AutoencoderDC` (dc-ae-f32c32-sana-1.0) in f32, decodes a fixed-seed latent through
the RAW decoder module (no scaling/tiling — matching the Rust `DcAeDecoder::decode`), and saves both
the input latent and the reference image to a safetensors the Rust test reads back.

Usage: python dump_dcae_golden.py MODEL_DIR OUT.safetensors
"""
import sys
import torch
from diffusers import AutoencoderDC
from safetensors.torch import save_file

model_dir, out = sys.argv[1], sys.argv[2]
model = AutoencoderDC.from_pretrained(model_dir, torch_dtype=torch.float32).eval()

torch.manual_seed(0)
latent = torch.randn(1, 32, 32, 32, dtype=torch.float32)
with torch.no_grad():
    image = model.decoder(latent)  # raw decoder forward → [1, 3, 1024, 1024]

print("latent", tuple(latent.shape), "-> image", tuple(image.shape),
      "min", float(image.min()), "max", float(image.max()))
save_file({"latent": latent.contiguous(), "image": image.contiguous()}, out)
print("wrote", out)
