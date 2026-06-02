"""Real-weights Qwen-Image **T2I img2img** golden — the reference for the mlx-gen img2img port
(sc-2530), mirroring `QwenImage.generate_image` on the img2img branch (an init `image_path` +
`image_strength`).

Run from the mflux fork venv (loads the full ~54 GB model):
    cd ~/repos/mflux && uv run python ~/repos/mlx-gen/tools/dump_qwen_image_img2img_golden.py
Set `QUANTIZE=8` to dump the Q8 transformer golden instead (transformer-only, the fork's scope):
    cd ~/repos/mflux && QUANTIZE=8 uv run python ~/repos/mlx-gen/tools/dump_qwen_image_img2img_golden.py

Faithful to the fork: `LatentCreator.create_for_txt2img_or_img2img(..., img2img=Img2Img(...))` seeds
the latents (VAE-encode the init image, pack, blend `(1-σ)·clean + σ·noise` at
`σ = config.scheduler.sigmas[init_time_step]`), then the denoise loop runs
`range(init_time_step, steps)` with true-CFG — exactly `variants/txt2img/qwen_image.py`. Dumps each
stage so the Rust port can be validated piece by piece:

  - init_image_u8  : the synthetic RGB init image (int32 HWC) — the Rust test reads these exact
                     bytes so both sides start from an identical image (no PIL-load drift).
  - image_nchw     : the fork's `ImageUtil.to_array(scale_to_dimensions(LANCZOS))` ([-1,1], NCHW).
  - clean_encoded  : `VAEUtil.encode` of that image ([1,16,1,H/8,W/8]).
  - clean          : `QwenLatentCreator.pack_latents(clean_encoded)` ([1,(h/16)·(w/16),64]).
  - init_latents   : the blended init `(1-σ)·clean + σ·noise` at σ = sigmas[init_time_step].
  - noise          : the seeded packed pure noise (f32).
  - final_latents  : after the denoise loop `range(init_time_step, steps)`.
  - decoded        : VAE-decoded image tensor.
  - sigmas         : the linear flow-match schedule (len steps+1).

Env-overridable (QWEN_*): PROMPT, NEGATIVE, SEED, STEPS, W, H, STRENGTH, and the init-image size
IW/IH. Output (gitignored): tools/golden/qwen_image_img2img_golden.safetensors (or `…_q8…`).
"""

import os

import mlx.core as mx
import numpy as np
from mflux.models.common.config.config import Config
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.model.qwen_text_encoder.qwen_prompt_encoder import QwenPromptEncoder
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.utils.image_util import ImageUtil
from PIL import Image

# --- fixed generation config (mirror these constants in the Rust test) ---
SEED = int(os.environ.get("QWEN_SEED", "42"))
PROMPT = os.environ.get("QWEN_PROMPT", "a fox sitting in a forest, photorealistic")
NEGATIVE = os.environ.get("QWEN_NEGATIVE", "")  # -> the fork's single-space fallback
STEPS = int(os.environ.get("QWEN_STEPS", "4"))
H = int(os.environ.get("QWEN_H", "256"))
W = int(os.environ.get("QWEN_W", "256"))
GUIDANCE = 4.0
STRENGTH = float(os.environ.get("QWEN_STRENGTH", "0.6"))
QUANTIZE = int(os.environ["QUANTIZE"]) if os.environ.get("QUANTIZE") else None
# Init-image size — deliberately non-square and not a multiple of the target so the LANCZOS
# scale_to_dimensions path is exercised (a no-op resize would hide resampler bugs).
IW = int(os.environ.get("QWEN_IW", "384"))
IH = int(os.environ.get("QWEN_IH", "320"))

_GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(_GOLDEN_DIR, exist_ok=True)
PNG_IN = os.path.join(_GOLDEN_DIR, "qwen_image_img2img_init.png")
PNG_OUT = os.path.join(_GOLDEN_DIR, "qwen_image_img2img_out.png")

# Synthetic init image: smooth diagonal gradients with per-channel phase — deterministic, and
# bit-reproducible in Rust from the dumped bytes.
yy, xx = np.mgrid[0:IH, 0:IW]
r = ((xx * 255) // max(IW - 1, 1)).astype(np.uint8)
g = ((yy * 255) // max(IH - 1, 1)).astype(np.uint8)
b = (((xx + yy) * 255) // max(IW + IH - 2, 1)).astype(np.uint8)
init_u8 = np.stack([r, g, b], axis=-1).astype(np.uint8)  # HWC
Image.fromarray(init_u8, mode="RGB").save(PNG_IN)

model = QwenImage(quantize=QUANTIZE)
config = Config(
    model_config=model.model_config,
    num_inference_steps=STEPS,
    height=H,
    width=W,
    guidance=GUIDANCE,
    scheduler="linear",
    image_path=PNG_IN,
    image_strength=STRENGTH,
)
init_step = config.init_time_step
sigmas = config.scheduler.sigmas
print(f"init_time_step={init_step}  strength={STRENGTH}  steps={STEPS}  W={W} H={H}")
print(f"sigmas={[round(float(s), 5) for s in sigmas]}")

# 1a. Preprocessed image (LANCZOS scale → [-1,1] NCHW) — isolates resize+normalize parity.
scaled_user = ImageUtil.scale_to_dimensions(
    image=ImageUtil.load_image(PNG_IN).convert("RGB"), target_width=config.width, target_height=config.height
)
image_nchw = ImageUtil.to_array(scaled_user)

# 1b. Clean latents (encode + pack) — isolates the VAE encoder.
clean_encoded = LatentCreator.encode_image(
    vae=model.vae, image_path=config.image_path, height=config.height, width=config.width
)
clean = QwenLatentCreator.pack_latents(clean_encoded, config.height, config.width)

# 1c. Blended init latents = exactly what create_for_txt2img_or_img2img returns (img2img branch).
init_latents = LatentCreator.create_for_txt2img_or_img2img(
    seed=SEED,
    width=config.width,
    height=config.height,
    img2img=Img2Img(
        vae=model.vae,
        latent_creator=QwenLatentCreator,
        sigmas=sigmas,
        init_time_step=init_step,
        image_path=config.image_path,
        tiling_config=model.tiling_config,
    ),
)
# The seeded pure noise, for the Rust blend gate (f32, packed).
noise = QwenLatentCreator.create_noise(SEED, H, W)

# 2. Encode positive + negative prompts (drop-34, bf16).
prompt_embeds, prompt_mask, neg_embeds, neg_mask = QwenPromptEncoder.encode_prompt(
    prompt=PROMPT,
    negative_prompt=NEGATIVE,
    prompt_cache={},
    qwen_tokenizer=model.tokenizers["qwen"],
    qwen_text_encoder=model.text_encoder,
)

# 3. Denoise loop over range(init_time_step, steps) with CFG (faithful to qwen_image.py).
latents = init_latents
for t in config.time_steps:
    latents = config.scheduler.scale_model_input(latents, t)
    n_pos = model.transformer(t=t, config=config, hidden_states=latents, encoder_hidden_states=prompt_embeds, encoder_hidden_states_mask=prompt_mask)  # fmt: off
    n_neg = model.transformer(t=t, config=config, hidden_states=latents, encoder_hidden_states=neg_embeds, encoder_hidden_states_mask=neg_mask)  # fmt: off
    guided = QwenImage.compute_guided_noise(n_pos, n_neg, config.guidance)
    latents = config.scheduler.step(noise=guided, timestep=t, latents=latents)
    mx.eval(latents)

# 4. Unpack + VAE decode.
unpacked = QwenLatentCreator.unpack_latents(latents=latents, height=H, width=W)
decoded = model.vae.decode(unpacked)
mx.eval(decoded)
ImageUtil._numpy_to_pil(ImageUtil._to_numpy(ImageUtil._denormalize(decoded))).save(PNG_OUT)

out = {
    "init_image_u8": mx.array(init_u8.astype(np.int32)),
    "image_nchw": image_nchw.astype(mx.float32),
    "clean_encoded": clean_encoded.astype(mx.float32),
    "clean": clean.astype(mx.float32),
    "init_latents": init_latents.astype(mx.float32),
    "noise": noise.astype(mx.float32),
    "final_latents": latents.astype(mx.float32),
    "decoded": decoded.astype(mx.float32),
    "sigmas": mx.array(sigmas).astype(mx.float32),
}
meta = {
    "prompt": PROMPT,
    "seed": str(SEED),
    "steps": str(STEPS),
    "w": str(W),
    "h": str(H),
    "guidance": str(GUIDANCE),
    "strength": str(STRENGTH),
    "init_time_step": str(int(init_step)),
    "iw": str(IW),
    "ih": str(IH),
    "quantize": str(QUANTIZE),
}
suffix = f"_q{QUANTIZE}" if QUANTIZE else ""
OUT = os.path.join(_GOLDEN_DIR, f"qwen_image_img2img{suffix}_golden.safetensors")
mx.save_safetensors(OUT, out, metadata=meta)
print(f"\nwrote {OUT} ({len(out)} tensors)")
print(f"  init {IW}x{IH} -> target {W}x{H}; clean {tuple(clean.shape)}; decoded {tuple(decoded.shape)}")
print(f"  + {PNG_IN} (init) and {PNG_OUT} (result)")
