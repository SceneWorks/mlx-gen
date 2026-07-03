"""Diffusers reference golden for the SD3.5-Large **default empty-negative** e2e parity test
(sc-9311, follow-up to F-004 / sc-9090), consumed by the `#[ignore]`d Rust test
`mlx-gen-sd3/tests/e2e_empty_negative_real_weights.rs`.

Why this exists: the mlx-gen SD3 default (unset) negative-prompt path encodes the EMPTY string ""
for the true-CFG uncond branch. This dump renders the diffusers `StableDiffusion3Pipeline` with an
explicit **empty** negative (`negative_prompt=""`) so the Rust `negative_prompt = None` render can be
A/B'd against it — proving the F-004 empty-CLIP uncond path (BOS-preserving `clip_ids("")`) is
end-to-end correct on real weights.

Torch/diffusers reference (NOT the mflux native path — the frozen fork has no native SD3): everything
is forced to float32 (the Rust VAE-decode runs f32), rendered small (256²/20-step, guidance 3.5) so
the f32 run is feasible. The captured `decoded` is the RAW VAE-decode output in [-1, 1] NCHW (BEFORE
diffusers' `image_processor.postprocess`), so the Rust `decoded_to_image` (which does x*0.5+0.5, clip,
NCHW->NHWC, *255) reproduces the same RGB8. Also captured: the empty-negative uncond pooled/context
conditioning for an optional tighter chaos-free A/B.

Gitignored output (derives from licensed weights + needs a torch env). Run from the mflux fork venv:
    cd ~/repos/mflux && .venv-0312/bin/python ~/repos/mlx-gen/tools/dump_sd3_empty_negative_e2e_golden.py

Requires: `pip install diffusers transformers torch` + the licensed
`stabilityai/stable-diffusion-3.5-large` snapshot in the HF cache (or set MODEL_ID below to a local
path). Honors HF_HUB_CACHE / HF_HOME via the standard huggingface resolution.
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline

from _paths import fixture

MODEL_ID = "stabilityai/stable-diffusion-3.5-large"
PROMPT = "a photograph of a red fox sitting in a green meadow, sharp focus, daylight"
# The whole point of sc-9311: the uncond branch conditions on the EMPTY negative.
NEGATIVE = ""
SEED, STEPS, SIZE, GUIDANCE = 7, 20, 256, 3.5
OUT = fixture("tools/golden/sd3_5_large_empty_negative_e2e.safetensors")

pipe = StableDiffusion3Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
# f32 reference. `cpu` is the safe default; set SD3_DUMP_DEVICE=mps (or cuda) for speed — the math
# stays f32 (MPS/CUDA matmul is f32), well within the e2e test's chaos-limited px>8 tolerance.
DEVICE = os.environ.get("SD3_DUMP_DEVICE", "cpu")
pipe.to(DEVICE)

# Capture the empty-negative uncond conditioning (pooled + context) for the optional tighter A/B.
with torch.no_grad():
    (
        prompt_embeds,
        neg_embeds,
        pooled_embeds,
        neg_pooled_embeds,
    ) = pipe.encode_prompt(
        prompt=PROMPT,
        prompt_2=PROMPT,
        prompt_3=PROMPT,
        negative_prompt=NEGATIVE,
        negative_prompt_2=NEGATIVE,
        negative_prompt_3=NEGATIVE,
        do_classifier_free_guidance=True,
        device=DEVICE,
        num_images_per_prompt=1,
    )

generator = torch.Generator(device="cpu").manual_seed(SEED)
with torch.no_grad():
    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        height=SIZE,
        width=SIZE,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=generator,
        output_type="latent",  # keep the raw latent; decode ourselves for the [-1,1] NCHW tensor
    )
    latents = result.images  # [1, 16, H/8, W/8]

    # Undo the pipeline's latent scale/shift, then VAE-decode to the raw [-1,1] NCHW sample.
    vae = pipe.vae
    latents_dn = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    decoded = vae.decode(latents_dn, return_dict=False)[0]  # [1, 3, H, W], ~[-1, 1]


def to_mx(t: torch.Tensor) -> mx.array:
    return mx.array(t.detach().to(torch.float32).cpu().numpy().astype(np.float32))


tensors = {
    "decoded": to_mx(decoded),  # raw VAE decode in [-1,1] NCHW; decoded_to_image consumes this
    "neg_pooled": to_mx(neg_pooled_embeds),  # empty-negative uncond pooled [1, 2048]
    "neg_context": to_mx(neg_embeds),  # empty-negative uncond context [1, 333, 4096]
}
mx.save_safetensors(OUT, tensors)
print(f"wrote {OUT}")
print(f"  decoded {tuple(decoded.shape)}  neg_pooled {tuple(neg_pooled_embeds.shape)} "
      f"neg_context {tuple(neg_embeds.shape)}")
