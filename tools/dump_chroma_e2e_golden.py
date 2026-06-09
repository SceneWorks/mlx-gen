"""Real-weight e2e parity fixture for Chroma (epic 3531, sc-3839).

Runs the torch `diffusers` ChromaPipeline (HD) in **f32** (the bf16 checkpoint upcast — bf16-valued
weights in f32 containers, matching the Rust port's bf16-weights/f32-activations path) and dumps:
  - `prompt_embeds` / `neg_embeds` + their transformer masks  → validates the masked T5 encode (sc-3838)
  - a single real-weight transformer `noise_pred` on fixed inputs → tight DiT gate (sc-3837 at scale)
  - the final packed latents + decoded image from a fixed initial latent → e2e coherence (sc-3839)

Small (256x256, 4 steps) to keep the f32 run tractable. Run from the SceneWorks torch venv:
    "/Users/michael/Library/Application Support/SceneWorks/python/venv/bin/python" \
        tools/dump_chroma_e2e_golden.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from safetensors.torch import save_file

from diffusers import ChromaPipeline

from _paths import fixture, hf_hub_cache

# argv: [variant=hd|base|flash] [guidance] [steps]
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "hd"
REPO = {"hd": "Chroma1-HD", "base": "Chroma1-Base", "flash": "Chroma1-Flash"}[VARIANT]
PROMPT = "a photograph of an astronaut riding a horse"
NEG = ""
H = W = 256
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
GUIDANCE = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
MAX_SEQ = 512

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32


def snapshot() -> str:
    base = hf_hub_cache() / f"models--lodestones--{REPO}" / "snapshots"
    return str(next(p for p in base.iterdir() if p.is_dir()))


@torch.no_grad()
def main() -> None:
    pipe = ChromaPipeline.from_pretrained(snapshot(), torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)

    # --- text: masked T5 encode (sc-3838 numeric parity) ---
    (prompt_embeds, text_ids, prompt_mask, neg_embeds, neg_text_ids, neg_mask) = pipe.encode_prompt(
        prompt=PROMPT, negative_prompt=NEG, do_classifier_free_guidance=True,
        device=device, num_images_per_prompt=1, max_sequence_length=MAX_SEQ,
    )

    # --- fixed packed initial latent [1, Si, 64] ---
    ch = pipe.transformer.config.in_channels // 4  # 16
    si = (H // 16) * (W // 16)
    gen = torch.Generator(device="cpu").manual_seed(0)
    init = torch.randn(1, si, ch * 4, generator=gen, dtype=torch.float32).to(device)

    img_ids = pipe._prepare_latent_image_ids(1, H // 16, W // 16, device, dtype)

    # --- single real-weight transformer forward (tight DiT gate) ---
    sigmas = np.linspace(1.0, 1.0 / STEPS, STEPS)
    from diffusers.pipelines.chroma.pipeline_chroma import calculate_shift
    mu = calculate_shift(si, 256, 4096, 0.5, 1.15)
    pipe.scheduler.set_timesteps(sigmas=sigmas, mu=mu, device=device)
    t0 = pipe.scheduler.timesteps[0]
    full_mask = pipe._prepare_attention_mask(1, si, dtype, prompt_mask)
    noise_pred = pipe.transformer(
        hidden_states=init,
        timestep=(t0.expand(1) / 1000).to(dtype),
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=img_ids,
        attention_mask=full_mask,
        return_dict=False,
    )[0]

    # --- full e2e: final packed latents + decoded image from the same init ---
    latents = pipe(
        prompt=PROMPT, negative_prompt=NEG, height=H, width=W, num_inference_steps=STEPS,
        guidance_scale=GUIDANCE, latents=init.clone(), max_sequence_length=MAX_SEQ,
        output_type="latent",
    ).images
    unpacked = pipe._unpack_latents(latents, H, W, pipe.vae_scale_factor)
    unpacked = (unpacked / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(unpacked, return_dict=False)[0]  # [1,3,H,W] in [-1,1]

    # The model path is identical across variants (validated comprehensively on HD); base/flash only
    # differ in the sigma schedule, so their fixtures are minimal (init + final + image) to stay small.
    out = {
        "init_latents": init.cpu().float(),                # [1, Si, 64]
        "final_latents": latents.cpu().float(),            # packed, post-denoise
        "image": image.cpu().float(),                      # [1,3,H,W], [-1,1]
    }
    if VARIANT == "hd":
        out.update({
            # embeds stored bf16 to keep the committed fixture small; the parity gates (5%) absorb it.
            "prompt_embeds": prompt_embeds.cpu().bfloat16(),
            "prompt_mask": prompt_mask.cpu().float(),      # transformer text mask [1, L]
            "neg_embeds": neg_embeds.cpu().bfloat16(),
            "neg_mask": neg_mask.cpu().float(),
            "img_ids": img_ids.cpu().float(),
            "timestep": (t0.expand(1) / 1000).cpu().float(),  # the sigma fed to the transformer
            "noise_pred": noise_pred.cpu().float(),        # single-forward DiT output
        })
    save_file(out, fixture(f"mlx-gen-chroma/tests/fixtures/chroma_e2e_{VARIANT}.safetensors"))
    print(f"variant={VARIANT} device={device} dtype={dtype} si={si} t0={float(t0):.3f} g={GUIDANCE} steps={STEPS}")
    print("shapes:", {k: tuple(v.shape) for k, v in out.items()})
    print("wrote chroma_e2e.safetensors")


if __name__ == "__main__":
    main()
