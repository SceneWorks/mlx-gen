"""Real-weights Z-Image **ControlNet** golden run — the reference for the mlx-gen control e2e
(sc-2349 / sc-2257).

Run from the mflux fork (the main fork now carries the ZImageControl variant; use the 0.31.2 venv
for the sc-2782 golden):
    cd ~/repos/mflux && QUANTIZE=8 .venv-0312/bin/python /path/to/mlx-gen/tools/dump_z_image_control_golden.py

Loads the real `Tongyi-MAI/Z-Image-Turbo` base + the `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-
Union-2.1` control overlay (the fork's `ZImageControl`), builds the 33ch control context from a
fixed synthetic skeleton image, and runs a fixed (prompt, seed, steps, scale, size) generation by
hand — same method as `dump_z_image_golden.py` — with the **static shift=3.0** schedule the Rust
port uses (so both sides hold the schedule fixed). Dumps every intermediate so the Rust port is
validated stage-by-stage: cap_feats, the control_context, the seeded init noise, the first velocity
v0, the final latents, the decoded image, and the control image's exact u8 bytes (so the Rust full
pipeline re-encodes byte-identical pixels).

Set `QUANTIZE=8` (or `4`) to dump the Q8/Q4 control golden (the fork's `ZImageControl(quantize=N)`:
base + control applied dense, then the whole transformer quantized — the 132-wide control patch
embedder stays dense). Output suffix `_q8` / `_q4`.
"""

import glob
import math
import os

import mlx.core as mx
import numpy as np
from PIL import Image as PILImage

from _paths import hf_hub_cache
from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import (
    FlowMatchEulerDiscreteScheduler as S,
)
from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator
from mflux.models.z_image.model.z_image_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.z_image.variants.z_image_control import ZImageControl
from mflux.utils.image_util import ImageUtil

_GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(_GOLDEN_DIR, exist_ok=True)

PROMPT = os.environ.get("ZIMAGE_PROMPT", "a person standing in a park, photorealistic, detailed")
SEED = int(os.environ.get("ZIMAGE_SEED", "42"))
STEPS = int(os.environ.get("ZIMAGE_STEPS", "8"))  # the Fun-Controlnet-Union checkpoint is 8-step
W = int(os.environ.get("ZIMAGE_W", "1024"))
H = int(os.environ.get("ZIMAGE_H", "1024"))
SCALE = float(os.environ.get("CONTROL_SCALE", "1.0"))
QUANTIZE = int(os.environ["QUANTIZE"]) if os.environ.get("QUANTIZE") else None


def _find_control_weights() -> str:
    if os.environ.get("CONTROL_WEIGHTS"):
        return os.environ["CONTROL_WEIGHTS"]
    pat = str(
        hf_hub_cache()
        / "models--alibaba-pai--Z-Image-Turbo-Fun-Controlnet-Union-2.1"
        / "snapshots/*/*.safetensors"
    )
    hits = glob.glob(pat)
    if not hits:
        raise SystemExit(f"control weights not found (set CONTROL_WEIGHTS); looked at {pat}")
    return hits[0]


CONTROL_PATH = _find_control_weights()
_SUFFIX = f"_q{QUANTIZE}" if QUANTIZE else ""
OUT = os.path.join(_GOLDEN_DIR, f"z_image_control{_SUFFIX}_golden.safetensors")
PNG = os.path.join(_GOLDEN_DIR, f"z_image_control{_SUFFIX}_golden.png")
CONTROL_PNG = os.path.join(_GOLDEN_DIR, "control_input.png")

# Fixed synthetic control image (W×H RGB8): deterministic gradients + a couple of solid blocks so
# the VAE-encoded control context is structured (not noise). Content is irrelevant to parity — only
# that the Rust side re-encodes the *same* bytes (saved below as control_image_u8).
yy, xx = np.mgrid[0:H, 0:W]
r = ((xx * 255) // W).astype(np.uint8)
g = ((yy * 255) // H).astype(np.uint8)
b = (((xx + yy) * 255) // (W + H)).astype(np.uint8)
control_img = np.stack([r, g, b], axis=-1).astype(np.uint8)  # HWC
control_img[H // 4 : H // 2, W // 4 : W // 2] = 255
control_img[H // 2 : 3 * H // 4, W // 3 : 2 * W // 3] = 0
PILImage.fromarray(control_img, "RGB").save(CONTROL_PNG)

model = ZImageControl(control_weights_path=CONTROL_PATH, quantize=QUANTIZE)

tok = model.tokenizers["z_image"]
tout = tok.tokenize(PROMPT)
input_ids, attn = tout.input_ids, tout.attention_mask
num_valid = int(mx.sum(attn[0]).item())

cap_feats = PromptEncoder.encode_prompt(PROMPT, tok, model.text_encoder)  # [num_valid, 2560]
control_context = model._encode_control_context(CONTROL_PNG, W, H)  # (33,1,H/8,W/8)

# Static shift=3.0 schedule (the model's scheduler_config; the Rust port's FlowMatchEuler::
# for_static_shift(steps, 3.0)). Held identical on both sides — see dump_z_image_golden.py.
mu = math.log(3.0)
sigmas = mx.linspace(1.0, 1.0 / STEPS, STEPS)
sigmas = S._time_shift_exponential_array(mu, 1.0, sigmas)
sigmas = mx.concatenate([sigmas, mx.zeros((1,), dtype=sigmas.dtype)], axis=0)

init = ZImageLatentCreator.create_noise(SEED, H, W)
latents = init
v0 = None
for t in range(STEPS):
    ts = mx.array(1.0 - float(sigmas[t]), dtype=mx.float32)
    v = model.transformer(
        x=latents,
        timestep=ts,
        sigmas=sigmas,
        cap_feats=cap_feats,
        control_context=control_context,
        control_context_scale=SCALE,
    )
    if t == 0:
        v0 = v
    latents = latents + (sigmas[t + 1] - sigmas[t]) * v
    mx.eval(latents)

unpacked = ZImageLatentCreator.unpack_latents(latents, H, W)
decoded = model.vae.decode(unpacked)
img = ImageUtil._numpy_to_pil(ImageUtil._to_numpy(ImageUtil._denormalize(decoded)))
img.save(PNG)

tensors = {
    "input_ids": input_ids.astype(mx.int32),
    "attention_mask": attn.astype(mx.int32),
    "cap_feats": cap_feats.astype(mx.float32),
    "control_context": control_context.astype(mx.float32),
    "control_image_u8": mx.array(control_img.astype(np.int32)),  # [H, W, 3]
    "init": init.astype(mx.float32),
    "v0": v0.astype(mx.float32),
    "final_latents": latents.astype(mx.float32),
    "decoded": decoded.astype(mx.float32),
    "sigmas": sigmas.astype(mx.float32),
}
meta = {
    "prompt": PROMPT, "seed": str(SEED), "steps": str(STEPS), "w": str(W), "h": str(H),
    "num_valid": str(num_valid), "control_scale": str(SCALE), "quantize": str(QUANTIZE),
    "control_weights": CONTROL_PATH,
}  # fmt: off
mx.save_safetensors(OUT, tensors, meta)
print(f"\nwrote {OUT} + {PNG}")
print(f"  control_context {tuple(control_context.shape)}, final_latents {tuple(latents.shape)}, decoded {tuple(decoded.shape)}")
print(f"  scale={SCALE} steps={STEPS} {W}x{H} quantize={QUANTIZE} control={os.path.basename(CONTROL_PATH)}")
