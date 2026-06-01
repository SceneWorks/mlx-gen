"""Real-weights Z-Image golden run — the reference for the mlx-gen end-to-end (sc-2352).

Run from the fork:  cd ~/repos/mflux && uv run python /Users/michael/repos/mlx-gen/tools/dump_z_image_golden.py

Loads the real Tongyi-MAI/Z-Image-Turbo models and runs a fixed (prompt, seed, steps, size)
generation by hand (mirroring z_image.py), dumping EVERY intermediate so the Rust port can be
validated stage-by-stage: the chat-template string, input_ids/attention_mask, cap_feats, the
seeded init noise, the final latents, and the decoded image. Real bf16 path (matches production).

Set `QUANTIZE=8` (or `4`) to dump the Q8/Q4 golden instead (sc-2532) — `ZImage(quantize=N)` runs
the fork's real quantized path (`nn.quantize` over transformer + text encoder + VAE). The dumped
`cap_feats` is then the fork's quantized-text-encoder output; the Rust Q8/Q4 parity test feeds it
in, so the gate isolates the transformer's (and VAE-decode's) quantization parity — the same
methodology as `dump_qwen_image_golden.py`. Output suffix: `_q8` / `_q4`.
"""

import mlx.core as mx
import numpy as np
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import (
    FlowMatchEulerDiscreteScheduler as S,
)
from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator
from mflux.models.z_image.model.z_image_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.z_image.z_image_initializer import ZImageInitializer
from mflux.utils.image_util import ImageUtil

import os

# Golden lives next to this script (tools/golden/), gitignored — and is where the Rust e2e test's
# `CARGO_MANIFEST_DIR/../tools/golden` resolves when run from this checkout/worktree.
_GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(_GOLDEN_DIR, exist_ok=True)
# Env-overridable so the same golden can be regenerated at any size/prompt for parity checks.
# Defaults match the fast 256^2 stage-test baseline; set ZIMAGE_* to render at, e.g., 1024^2.
PROMPT = os.environ.get("ZIMAGE_PROMPT", "a fox")
SEED = int(os.environ.get("ZIMAGE_SEED", "42"))
STEPS = int(os.environ.get("ZIMAGE_STEPS", "4"))
W = int(os.environ.get("ZIMAGE_W", "256"))
H = int(os.environ.get("ZIMAGE_H", "256"))
QUANTIZE = int(os.environ["QUANTIZE"]) if os.environ.get("QUANTIZE") else None

_SUFFIX = f"_q{QUANTIZE}" if QUANTIZE else ""
OUT = os.path.join(_GOLDEN_DIR, f"z_image{_SUFFIX}_golden.safetensors")
PNG = os.path.join(_GOLDEN_DIR, f"z_image{_SUFFIX}_golden.png")


class Holder:
    pass


model = Holder()
ZImageInitializer.init(model, model_config=ModelConfig.z_image_turbo(), quantize=QUANTIZE)

tok = model.tokenizers["z_image"]
chat = tok.tokenizer.apply_chat_template(
    [{"role": "user", "content": PROMPT}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
print("=== chat-template string ===")
print(repr(chat))

tout = tok.tokenize(PROMPT)
input_ids, attn = tout.input_ids, tout.attention_mask
num_valid = int(mx.sum(attn[0]).item())
print(f"num_valid tokens: {num_valid}; first ids: {np.array(input_ids[0, :num_valid]).tolist()}")

cap_feats = PromptEncoder.encode_prompt(PROMPT, tok, model.text_encoder)  # [num_valid, 2560]

# Schedule + seeded init noise. Z-Image-Turbo pins a STATIC time-shift (its
# scheduler/scheduler_config.json: FlowMatchEulerDiscreteScheduler, shift=3.0,
# use_dynamic_shifting=false) — NOT the empirical per-step mu (that is the *full* Z-Image
# model's scheduler). The exponential time-shift with mu=ln(shift) == diffusers' static shift
# (sc-2536). The Rust port uses FlowMatchEuler::for_static_shift(steps, 3.0) to match.
import math  # noqa: E402

mu = math.log(3.0)
sigmas = mx.linspace(1.0, 1.0 / STEPS, STEPS)
sigmas = S._time_shift_exponential_array(mu, 1.0, sigmas)
sigmas = mx.concatenate([sigmas, mx.zeros((1,), dtype=sigmas.dtype)], axis=0)

init = ZImageLatentCreator.create_noise(SEED, H, W)
latents = init
v0 = None
for t in range(STEPS):
    ts = mx.array(1.0 - float(sigmas[t]), dtype=mx.float32)
    v = model.transformer(x=latents, timestep=ts, sigmas=sigmas, cap_feats=cap_feats)
    if t == 0:
        v0 = v
    latents = latents + (sigmas[t + 1] - sigmas[t]) * v
    mx.eval(latents)

unpacked = ZImageLatentCreator.unpack_latents(latents, H, W)
decoded = model.vae.decode(unpacked)

# Save the PNG (fork ImageUtil) + raw decoded for byte compare.
img = ImageUtil._numpy_to_pil(ImageUtil._to_numpy(ImageUtil._denormalize(decoded)))
img.save(PNG)

tensors = {
    "input_ids": input_ids.astype(mx.int32),
    "attention_mask": attn.astype(mx.int32),
    "cap_feats": cap_feats.astype(mx.float32),
    "init": init.astype(mx.float32),
    "v0": v0.astype(mx.float32),
    "final_latents": latents.astype(mx.float32),
    "decoded": decoded.astype(mx.float32),
    "sigmas": sigmas.astype(mx.float32),
}
meta = {"prompt": PROMPT, "seed": str(SEED), "steps": str(STEPS), "w": str(W), "h": str(H),
        "num_valid": str(num_valid), "chat": chat, "quantize": str(QUANTIZE)}
mx.save_safetensors(OUT, tensors, meta)
print(f"\nwrote {OUT} + {PNG}; final_latents {tuple(latents.shape)}, decoded {tuple(decoded.shape)}")
