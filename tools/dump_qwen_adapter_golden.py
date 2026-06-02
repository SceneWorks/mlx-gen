"""Real-weights Qwen-Image LoRA + LoKr golden — the reference for the mlx-gen adapter gate (sc-2528).

Run from the fork venv (loads the full ~54 GB model):
    ~/Repos/mflux/.venv/bin/python ~/Repos/mlx-gen/.../tools/dump_qwen_adapter_golden.py

Builds a deterministic synthetic adapter (LoRA, then LoKr) targeting the joint-attention projections
across a few blocks — the trained case — saves each in the on-disk format BOTH engines parse (peft
`diffusion_model.`-prefixed `lora_A/B.weight`, NO `.alpha` key so the fork's bare-only alpha
patterns and the Rust loader agree; bare `lokr_w1/w2` + `networkType=lokr` metadata for LoKr),
applies it through the fork's real `QwenImage(lora_paths=…, lora_scales=[1.0])`, runs the fixed
(prompt, seed, steps, size) CFG render, and dumps the decoded image. The Rust gate
(`tests/adapter_real_weights.rs`) loads the SAME adapter file via `LoadSpec.adapters`.
"""

import os

import mlx.core as mx
import numpy as np
from mflux.models.common.config.config import Config
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.model.qwen_text_encoder.qwen_prompt_encoder import QwenPromptEncoder
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

_GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(_GOLDEN_DIR, exist_ok=True)

SEED = 42
PROMPT = "a fox sitting in a forest, photorealistic"
STEPS = 4
H = int(os.environ.get("QWEN_H", "256"))
W = int(os.environ.get("QWEN_W", "256"))
GUIDANCE = 4.0
LORA_STD = float(os.environ.get("QWEN_LORA_STD", "0.02"))
LOKR_STD = float(os.environ.get("QWEN_LOKR_STD", "0.05"))
RANK = 8
BLOCKS = [0, 30, 59]
PROJS = ["to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]
# The fork's LoKrLoader navigates RAW file paths (no mapping translation), so it cannot reach the
# image output projection — internally renamed `attn.attn_to_out.0`, addressed in the file as
# `attn.to_out.0` — and silently skips it (the fork's LoRA loader applies it via the mapping). The
# Rust host resolves `to_out.0` for both (more correct), but to gate LoKr on the set where the two
# engines AGREE we exclude it from the LoKr golden. (LoRA keeps the full set.)
LOKR_PROJS = [p for p in PROJS if p != "to_out.0"]
DIM = 3072  # all attention projections are [3072, 3072]


def build_lora(path):
    # peft `lora_A/B` under the `diffusion_model.` prefix, but a BARE `alpha` (the fork's Qwen
    # mapping has bare-only alpha patterns). alpha = 2*RANK so alpha/rank = 2 has a visible,
    # non-trivial effect — this exercises the loader's bare-alpha-under-a-prefix fold (sc-2528
    # adversarial review); the fork applies the same scaling via its bare alpha pattern.
    rng = np.random.default_rng(20260602)
    t = {}
    for b in BLOCKS:
        for proj in PROJS:
            base = f"diffusion_model.transformer_blocks.{b}.attn.{proj}"
            bare = f"transformer_blocks.{b}.attn.{proj}"
            a = rng.normal(0.0, LORA_STD, size=(RANK, DIM)).astype(np.float32)
            bb = rng.normal(0.0, LORA_STD, size=(DIM, RANK)).astype(np.float32)
            t[f"{base}.lora_A.weight"] = mx.array(a)
            t[f"{base}.lora_B.weight"] = mx.array(bb)
            t[f"{bare}.alpha"] = mx.array(np.array([float(2 * RANK)], dtype=np.float32))
    mx.save_safetensors(path, t)
    return path


def build_lokr(path):
    rng = np.random.default_rng(20260603)
    t = {}
    for b in BLOCKS:
        for proj in LOKR_PROJS:
            base = f"transformer_blocks.{b}.attn.{proj}"
            w1 = rng.normal(0.0, LOKR_STD, size=(48, 48)).astype(np.float32)
            w2 = rng.normal(0.0, LOKR_STD, size=(64, 64)).astype(np.float32)
            t[f"{base}.lokr_w1"] = mx.array(w1)
            t[f"{base}.lokr_w2"] = mx.array(w2)
    mx.save_safetensors(path, t, {"networkType": "lokr", "alpha": "1.0", "rank": "1"})
    return path


def render(adapter_path):
    model = QwenImage(quantize=None, lora_paths=[adapter_path], lora_scales=[1.0])
    config = Config(
        model_config=model.model_config,
        num_inference_steps=STEPS,
        height=H,
        width=W,
        guidance=GUIDANCE,
        scheduler="linear",
    )
    noise = QwenLatentCreator.create_noise(SEED, H, W)
    pe, pm, ne, nm = QwenPromptEncoder.encode_prompt(
        prompt=PROMPT,
        negative_prompt="",
        prompt_cache={},
        qwen_tokenizer=model.tokenizers["qwen"],
        qwen_text_encoder=model.text_encoder,
    )
    latents = noise
    for t in config.time_steps:
        n_pos = model.transformer(t=t, config=config, hidden_states=latents, encoder_hidden_states=pe, encoder_hidden_states_mask=pm)  # fmt: off
        n_neg = model.transformer(t=t, config=config, hidden_states=latents, encoder_hidden_states=ne, encoder_hidden_states_mask=nm)  # fmt: off
        guided = QwenImage.compute_guided_noise(n_pos, n_neg, config.guidance)
        latents = config.scheduler.step(noise=guided, timestep=t, latents=latents)
        mx.eval(latents)
    unpacked = QwenLatentCreator.unpack_latents(latents=latents, height=H, width=W)
    decoded = model.vae.decode(unpacked)
    mx.eval(decoded)
    return decoded


for kind, builder in [("lora", build_lora), ("lokr", build_lokr)]:
    adapter_path = os.path.join(_GOLDEN_DIR, f"qwen_{kind}_adapter.safetensors")
    builder(adapter_path)
    decoded = render(adapter_path)
    out = os.path.join(_GOLDEN_DIR, f"qwen_{kind}_golden.safetensors")
    mx.save_safetensors(
        out,
        {"decoded": decoded.astype(mx.float32)},
        {
            "prompt": PROMPT, "seed": str(SEED), "steps": str(STEPS), "width": str(W),
            "height": str(H), "guidance": str(GUIDANCE), "kind": kind,
        },
    )
    print(f"wrote {out} + {adapter_path}; decoded {tuple(decoded.shape)}")
