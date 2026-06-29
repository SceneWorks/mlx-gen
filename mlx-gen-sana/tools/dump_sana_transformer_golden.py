#!/usr/bin/env python3
"""Dump a SANA Linear-DiT **trunk** golden for the sc-8487 parity gate.

Mirrors `dump_dcae_golden.py`: runs the diffusers `SanaTransformer2DModel` for a fixed seed and
saves the inputs (latent, timestep, caption embedding) + the reference noise prediction AND every
weight tensor into one safetensors that the Rust parity test reads back.

Two modes:

  * **tiny** (default, no args) — builds a SMALL random-init `SanaTransformer2DModel` (a faithful
    instance of the real architecture: ReLU linear self-attn, cross-attn, GLUMBConv Mix-FFN,
    adaLN-single timestep modulation, NoPE) with a reduced dim/depth so the fixture is tiny enough
    to commit. This keeps the parity gate reproducible in CI without the ~1.6B-param real weights.

  * **real MODEL_DIR** — loads the real `Sana_1600M_1024px_diffusers` transformer in fp16 and dumps a
    single-step forward golden (large; gitignored). Used to characterise full-model parity locally.

Usage:
    python dump_sana_transformer_golden.py                      # tiny committed fixture
    python dump_sana_transformer_golden.py OUT.safetensors      # tiny → custom path
    python dump_sana_transformer_golden.py --real MODEL_DIR OUT.safetensors
"""

import os
import sys

import torch
from diffusers.models.transformers.sana_transformer import SanaTransformer2DModel
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "..", "tests", "fixtures", "sana_transformer_golden.safetensors")


def dump(model: SanaTransformer2DModel, out: str, *, in_channels: int, sample: int, seq: int,
         caption_channels: int, dtype: torch.dtype):
    model = model.to(dtype).eval()
    torch.manual_seed(0)
    latent = torch.randn(1, in_channels, sample, sample, dtype=dtype)
    # diffusers expects the caption embedding shaped [B, 1, seq, caption_channels] (PixArt convention,
    # squeezed to [B, seq, C] inside caption_projection.view); pass [B, seq, C] which the model's
    # `.view(B, -1, inner)` handles identically.
    caption = torch.randn(1, seq, caption_channels, dtype=dtype)
    timestep = torch.tensor([500.0], dtype=dtype)

    with torch.no_grad():
        out_sample = model(
            hidden_states=latent,
            encoder_hidden_states=caption,
            timestep=timestep,
            return_dict=True,
        ).sample  # [1, out_channels, sample, sample]

    tensors = {
        "input.latent": latent.contiguous(),
        "input.caption": caption.contiguous(),
        "input.timestep": timestep.contiguous(),
        "output.sample": out_sample.float().contiguous(),
    }
    for k, v in model.state_dict().items():
        tensors[f"w.{k}"] = v.contiguous()

    print("latent", tuple(latent.shape), "caption", tuple(caption.shape),
          "timestep", tuple(timestep.shape), "-> output", tuple(out_sample.shape),
          "min", float(out_sample.min()), "max", float(out_sample.max()))
    save_file(tensors, out)
    print("wrote", out, "with", len(tensors), "tensors")


def main():
    args = sys.argv[1:]
    if args and args[0] == "--real":
        model_dir, out = args[1], args[2]
        model = SanaTransformer2DModel.from_pretrained(model_dir)
        cfg = model.config
        dump(model, out, in_channels=cfg.in_channels, sample=cfg.sample_size,
             seq=300, caption_channels=cfg.caption_channels, dtype=torch.float16)
        return

    out = args[0] if args else DEFAULT_OUT
    # Tiny, faithful instance of the real arch. inner_dim = num_attention_heads * attention_head_dim.
    model = SanaTransformer2DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=8,           # inner_dim = 16
        num_layers=2,
        num_cross_attention_heads=2,
        cross_attention_head_dim=8,     # cross inner = 16
        cross_attention_dim=16,
        caption_channels=24,
        mlp_ratio=2.5,
        attention_bias=False,
        sample_size=4,
        patch_size=1,
        norm_elementwise_affine=False,
        norm_eps=1e-6,
        interpolation_scale=None,       # NoPE
        qk_norm=None,
    )
    dump(model, out, in_channels=4, sample=4, seq=5, caption_channels=24, dtype=torch.float32)


if __name__ == "__main__":
    main()
