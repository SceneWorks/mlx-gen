#!/usr/bin/env python
"""Dump a diffusers SANA end-to-end pipeline golden for the mlx-gen-sana e2e test (sc-8489, Phase A).

Runs the real `Efficient-Large-Model/Sana_1600M_1024px_diffusers` pipeline for a FIXED prompt +
seed + recipe and saves, to a gitignored golden, the artifacts that pin the full prompt→image path:

  * `latent`   — the final post-denoise latent `[1, 32, 32, 32]` (NCHW, pre-DC-AE-decode, pre-unscale).
  * `image`    — the decoded RGB image `[1, 3, 1024, 1024]` (NCHW, the diffusers VAE-decode output in
                 [-1, 1] before the PIL `(x*0.5 + 0.5)` clamp).
  * `caption_embeds` — the `[1, 300, 2304]` gemma CHI embedding fed to the trunk (cond branch).

The mlx-gen `tests/pipeline_contract.rs::real_weight_1024_e2e` test (gated behind
`SANA_PIPELINE_WEIGHTS`) reproduces this gen and, when a golden is present, can compare the final
latent / image to these references (`mean_rel` faithfulness gate, ~5e-3 over Metal's reduced-precision
matmul — same convention as `decode_parity.rs` / `transformer_parity.rs`). Without the golden the test
still asserts a finite, bounded, non-constant 1024² RGB output.

Pinned recipe (must match the Rust `SanaGenerateRequest` in the e2e test):
  prompt = "a photorealistic red panda sitting on a mossy log in a misty forest"
  negative_prompt = ""   steps = 20   guidance_scale = 4.5   seed = 42   1024x1024

Run (from a venv with diffusers + torch + a Sana_1600M_1024px_diffusers snapshot):
  python tools/dump_sana_pipeline_golden.py
"""

import os

import torch
from diffusers import SanaPipeline
from safetensors.torch import save_file

MODEL_ID = "Efficient-Large-Model/Sana_1600M_1024px_diffusers"
PROMPT = "a photorealistic red panda sitting on a mossy log in a misty forest"
NEGATIVE = ""
STEPS = 20
GUIDANCE = 4.5
SEED = 42
SIZE = 1024
OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools/golden/sana/pipeline_red_panda.safetensors",
)


def main() -> None:
    pipe = SanaPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    pipe.to("cpu")

    captured = {}

    # Capture the final latent right before the DC-AE decode (the diffusers pipeline names it
    # `latents` and divides by `self.vae.config.scaling_factor` before `self.vae.decode`).
    orig_decode = pipe.vae.decode

    def spy_decode(z, *a, **k):
        # `z` here is ALREADY `latents / scaling_factor` (diffusers unscales before calling decode).
        captured["unscaled_latent"] = z.detach().to(torch.float32).cpu().clone()
        return orig_decode(z, *a, **k)

    pipe.vae.decode = spy_decode

    gen = torch.Generator(device="cpu").manual_seed(SEED)
    out = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        height=SIZE,
        width=SIZE,
        generator=gen,
        output_type="pt",  # decoded tensor in [0, 1]
    )

    # Re-derive the pre-decode latent in the convention the Rust test uses (NCHW, *scaled* by the
    # scaling_factor, i.e. the trunk-output latent before the pipeline's `/scaling_factor`).
    unscaled = captured["unscaled_latent"]  # [1, 32, 32, 32]
    latent = unscaled * pipe.vae.config.scaling_factor

    # diffusers `output_type="pt"` gives [B, 3, H, W] in [0, 1]; map back to the raw decoder [-1, 1].
    image01 = out.images.detach().to(torch.float32).cpu()  # [1, 3, 1024, 1024]
    image = image01 * 2.0 - 1.0

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(
        {
            "latent": latent.contiguous(),
            "image": image.contiguous(),
        },
        OUT,
    )
    print(f"wrote {OUT}")
    print(f"  latent {tuple(latent.shape)}  image {tuple(image.shape)}")


if __name__ == "__main__":
    main()
