#!/usr/bin/env python
"""Dump SANA-Sprint goldens for the mlx-gen-sana Sprint parity tests (sc-8490, Phase A).

Two goldens (mirroring `dump_sana_transformer_golden.py` / `dump_sana_pipeline_golden.py`):

  1. A SMALL, committed **guidance-embed trunk** golden (`--tiny`, default): a faithful random-init
     `SanaTransformer2DModel` with `guidance_embeds=True` + `qk_norm="rms_norm_across_heads"` at a
     reduced dim/depth, plus its inputs (latent, caption, the SCM conditioning timestep, the embedded
     guidance scalar) and the reference noise prediction. The Rust `SanaTransformer::from_weights(...,
     sana_sprint config).forward_with_guidance(...)` must reproduce it. Saved to
     `tests/fixtures/sana_sprint_trunk_golden.safetensors` (committed, ~tens of KB, runs in CI).

  2. (`--real`) A large fp16 single-step golden from the real `Sana_Sprint_1.6B_1024px_diffusers`
     transformer, for the `#[ignore]`d `SANA_SPRINT_WEIGHTS` parity characterisation.

The SCM *scheduler* step math (trigflow x0-pred + renoise) is parity-checked directly in Rust against
hand-computed diffusers references (`tests/sprint_scm_parity.rs`), since it is pure host math with no
weights — no golden needed there.

Run (from a venv with diffusers + torch):
  python tools/dump_sana_sprint_golden.py            # tiny committed golden
  SANA_SPRINT_WEIGHTS=/path python tools/dump_sana_sprint_golden.py --real
"""

import os
import sys

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def tiny() -> None:
    from diffusers.models.transformers.sana_transformer import SanaTransformer2DModel

    torch.manual_seed(0)
    cfg = dict(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=8,  # inner = 16
        num_layers=2,
        num_cross_attention_heads=2,
        cross_attention_head_dim=8,
        cross_attention_dim=16,  # = inner_dim (caption is projected to inner before attn2)
        caption_channels=24,
        mlp_ratio=2.5,
        patch_size=1,
        sample_size=4,
        norm_eps=1e-6,
        interpolation_scale=None,
        guidance_embeds=True,
        guidance_embeds_scale=0.1,
        qk_norm="rms_norm_across_heads",
    )
    model = SanaTransformer2DModel(**cfg).eval()

    latent = torch.randn(1, 4, 4, 4)
    caption = torch.randn(1, 5, 24)
    # SCM conditioning timestep (sin(t)/(cos(t)+sin(t)) for some angle t) and the embedded guidance
    # scalar (guidance_scale * guidance_embeds_scale). Both are [1].
    scm_timestep = torch.tensor([0.6])
    guidance = torch.tensor([4.5 * 0.1])

    with torch.no_grad():
        out = model(
            latent,
            encoder_hidden_states=caption,
            timestep=scm_timestep,
            guidance=guidance,
            return_dict=False,
        )[0]

    tensors = {f"w.{k}": v.contiguous() for k, v in model.state_dict().items()}
    tensors["input.latent"] = latent.contiguous()
    tensors["input.caption"] = caption.contiguous()
    tensors["input.timestep"] = scm_timestep.contiguous()
    tensors["input.guidance"] = guidance.contiguous()
    tensors["output.sample"] = out.contiguous()

    out_path = os.path.join(ROOT, "mlx-gen-sana/tests/fixtures/sana_sprint_trunk_golden.safetensors")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_file(tensors, out_path)
    print(f"wrote {out_path}  output {tuple(out.shape)}")


def real() -> None:
    from diffusers import SanaTransformer2DModel

    weights = os.environ["SANA_SPRINT_WEIGHTS"]
    model = SanaTransformer2DModel.from_pretrained(
        os.path.join(weights, "transformer"), torch_dtype=torch.float16
    ).eval()

    latent = torch.randn(1, 32, 32, 32, dtype=torch.float16)
    caption = torch.randn(1, 300, 2304, dtype=torch.float16)
    scm_timestep = torch.tensor([0.6], dtype=torch.float16)
    guidance = torch.tensor([4.5 * 0.1], dtype=torch.float16)

    with torch.no_grad():
        out = model(
            latent,
            encoder_hidden_states=caption,
            timestep=scm_timestep,
            guidance=guidance,
            return_dict=False,
        )[0]

    out_path = os.path.join(HERE, "golden/sana/sprint_trunk_real.safetensors")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_file(
        {
            "input.latent": latent.contiguous(),
            "input.caption": caption.contiguous(),
            "input.timestep": scm_timestep.contiguous(),
            "input.guidance": guidance.contiguous(),
            "output.sample": out.contiguous(),
        },
        out_path,
    )
    print(f"wrote {out_path}  output {tuple(out.shape)}")


if __name__ == "__main__":
    if "--real" in sys.argv:
        real()
    else:
        tiny()
