"""Hyper-FLUX few-step golden — the reference for the mlx-gen FLUX acceleration gate (sc-2908).

FLUX few-step acceleration (Hyper-FLUX) is NOT a new scheduler: it is the standard flow-match
`FlowMatchEulerDiscreteScheduler` at a reduced step count + a distilled LoRA (ByteDance Hyper-SD).
The reference here is **diffusers** (torch), NOT the mflux fork — the fork's `FluxLoRAMapping` omits
the top-level global projections (`x_embedder`, `context_embedder`, `proj_out`, `norm_out.linear`,
`time_text_embed.*`) that the Hyper-FLUX PEFT LoRA trains, so the fork cannot load this file. The Rust
crate (sc-2908) extends its mapping to cover those globals.

This renders FLUX.1-dev + `Hyper-FLUX.1-dev-8steps-lora.safetensors` (the documented recipe:
`fuse_lora(0.125)`, 8 steps, guidance 3.5) and dumps everything the Rust gate
(`mlx-gen-flux/tests/hyper_flux_real_weights.rs`) needs:

  - `image_u8`        : the final decoded image as uint8 RGB pixels [H*W*3] (direct px>8 compare).
  - `final_latents`   : the packed [1, seq, 64] latents (output_type="latent") — the injected
                        denoise-loop comparison target (isolates transformer+LoRA+sampler from the VAE).
  - `init`            : the packed init latents the loop starts from (same seed) — injected in.
  - `prompt_embeds`   : T5 [1, 512, 4096]; `pooled_prompt_embeds`: CLIP [1, 768] — injected in so the
                        comparison isolates the denoise from the (torch vs MLX) text encoders.
  - `sigmas`          : the scheduler's flow-match sigmas (len steps+1, trailing 0) — injected in.

NOTE: this is a CROSS-BACKEND reference (torch ↔ our MLX build). Unlike the mflux goldens it is NOT
bit-exact — the injected latent comparison lands in the single-digit-% range (8 chaotic few-step
iterations amplify the ~1e-3 backend delta + the fused-vs-residual LoRA application). The Rust gate
treats it as a coarse/structural bound + a visible-effect check, plus a saved PNG for visual parity.

Gitignored output. Run from the diffusers venv (torch + diffusers):
    cd ~/Repos/mflux && .venv/bin/python ~/Repos/mlx-gen/tools/dump_hyper_flux_golden.py

Env-overridable: FLUX_DEV (snapshot dir or HF id), HYPER_LORA (path), FLUX_SEED, FLUX_STEPS, FLUX_W,
FLUX_H, FLUX_GUIDANCE, HYPER_LORA_SCALE, FLUX_PROMPT.
"""

import os

import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift
from safetensors.numpy import save_file

_GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(_GOLDEN_DIR, exist_ok=True)

BASE = os.environ.get(
    "FLUX_DEV",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/"
        "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
    ),
)
LORA = os.environ.get(
    "HYPER_LORA",
    os.path.expanduser("~/repos/test-files/Hyper-FLUX.1-dev-8steps-lora.safetensors"),
)
PROMPT = os.environ.get("FLUX_PROMPT", "a photo of a red fox in a snowy forest, golden hour")
SEED = int(os.environ.get("FLUX_SEED", "7"))
W = int(os.environ.get("FLUX_W", "512"))
H = int(os.environ.get("FLUX_H", "512"))
STEPS = int(os.environ.get("FLUX_STEPS", "8"))
GUIDANCE = float(os.environ.get("FLUX_GUIDANCE", "3.5"))
LORA_SCALE = float(os.environ.get("HYPER_LORA_SCALE", "0.125"))

OUT = os.path.join(_GOLDEN_DIR, "flux1_dev_hyper_golden.safetensors")
PNG = os.path.join(_GOLDEN_DIR, "diffusers_hyper_flux.png")

print(f"loading FLUX.1-dev from {BASE} (bf16) …")
pipe = FluxPipeline.from_pretrained(BASE, torch_dtype=torch.bfloat16)
pipe.to("mps" if torch.backends.mps.is_available() else "cpu")

# --- the exact flow-match sigmas the pipeline will use (for injection) ---------------------------
# Mirrors FluxPipeline.__call__: sigmas = linspace(1, 1/n, n); mu = calculate_shift(seq_len); the
# scheduler builds sigmas with a trailing 0. seq_len = (H/16)*(W/16) (packed token count).
seq_len = (H // 16) * (W // 16)
base_sigmas = np.linspace(1.0, 1.0 / STEPS, STEPS)
mu = calculate_shift(
    seq_len,
    pipe.scheduler.config.get("base_image_seq_len", 256),
    pipe.scheduler.config.get("max_image_seq_len", 4096),
    pipe.scheduler.config.get("base_shift", 0.5),
    pipe.scheduler.config.get("max_shift", 1.15),
)
pipe.scheduler.set_timesteps(sigmas=base_sigmas, mu=mu, device="cpu")
sigmas = pipe.scheduler.sigmas.detach().cpu().float().numpy().copy()  # len STEPS+1, trailing 0
print(f"sigmas (len {len(sigmas)}): {sigmas}")

# --- prompt embeds (T5 + CLIP), injected so the gate isolates the denoise from the text encoders --
prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
    prompt=PROMPT, prompt_2=PROMPT, device=pipe._execution_device, num_images_per_prompt=1, max_sequence_length=512
)

# --- init latents (packed), same seed the full render uses ---------------------------------------
gen_init = torch.Generator(device="cpu").manual_seed(SEED)
init_latents, _ = pipe.prepare_latents(
    1, pipe.transformer.config.in_channels // 4, H, W,
    prompt_embeds.dtype, pipe._execution_device, gen_init, None,
)

# --- render the SAME injected (prompt_embeds, seed, steps, sigmas) two ways: base FLUX.1-dev (no LoRA,
#     the cross-backend floor reference) and Hyper (LoRA fused @ 0.125). Same init both ways (the LoRA
#     does not touch encode_prompt / prepare_latents), so the Rust gate can assert that the LoRA adds
#     ~0 net divergence over the (torch ↔ MLX) backend floor. -----------------------------------------
def render(output_type):
    g = torch.Generator(device="cpu").manual_seed(SEED)
    return pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        height=H, width=W, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
        generator=g, output_type=output_type,
    ).images


print("rendering BASE FLUX.1-dev (no LoRA) 8-step — the cross-backend floor reference …")
base_latents = render("latent")
base_image = render("pil")[0]
base_image.save(os.path.join(_GOLDEN_DIR, "diffusers_base_flux_8step.png"))

print(f"loading Hyper-FLUX LoRA {LORA} and fusing at scale {LORA_SCALE} …")
pipe.load_lora_weights(LORA)
pipe.fuse_lora(lora_scale=LORA_SCALE)

print("rendering HYPER FLUX.1-dev (LoRA fused) 8-step …")
hyper_latents = render("latent")
hyper_image = render("pil")[0]
hyper_image.save(PNG)
print(f"saved diffusers Hyper-FLUX render → {PNG}")

tensors = {
    "image_u8": np.asarray(hyper_image, dtype=np.uint8).reshape(-1),
    "base_image_u8": np.asarray(base_image, dtype=np.uint8).reshape(-1),
    "final_latents": hyper_latents.detach().cpu().float().numpy().copy(),
    "base_final_latents": base_latents.detach().cpu().float().numpy().copy(),
    "init": init_latents.detach().cpu().float().numpy().copy(),
    "prompt_embeds": prompt_embeds.detach().cpu().float().numpy().copy(),
    "pooled_prompt_embeds": pooled_prompt_embeds.detach().cpu().float().numpy().copy(),
    "sigmas": sigmas.astype(np.float32),
}
meta = {
    "prompt": PROMPT,
    "seed": str(SEED),
    "steps": str(STEPS),
    "width": str(W),
    "height": str(H),
    "guidance": str(GUIDANCE),
    "lora_scale": str(LORA_SCALE),
}
save_file(tensors, OUT, metadata=meta)
print(f"wrote {OUT}")
print({k: v.shape for k, v in tensors.items()})
