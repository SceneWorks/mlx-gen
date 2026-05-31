"""Real-weights Z-Image golden run — the reference for the mlx-gen end-to-end (sc-2352).

Run from the fork:  cd ~/repos/mflux && uv run python /Users/michael/repos/mlx-gen/tools/dump_z_image_golden.py

Loads the real Tongyi-MAI/Z-Image-Turbo models and runs a fixed (prompt, seed, steps, size)
generation by hand (mirroring z_image.py), dumping EVERY intermediate so the Rust port can be
validated stage-by-stage: the chat-template string, input_ids/attention_mask, cap_feats, the
seeded init noise, the final latents, and the decoded image. Real bf16 path (matches production).
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

OUT = "/Users/michael/repos/mlx-gen/tools/golden/z_image_golden.safetensors"
PNG = "/Users/michael/repos/mlx-gen/tools/golden/z_image_golden.png"
PROMPT, SEED, STEPS, W, H = "a fox", 42, 4, 256, 256


class Holder:
    pass


model = Holder()
ZImageInitializer.init(model, model_config=ModelConfig.z_image_turbo(), quantize=None)

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

# Schedule (resolution-dependent mu) + seeded init noise.
seq_len = (H // 16) * (W // 16)
mu = S._compute_empirical_mu(seq_len, STEPS)
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
        "num_valid": str(num_valid), "chat": chat}
mx.save_safetensors(OUT, tensors, meta)
print(f"\nwrote {OUT} + {PNG}; final_latents {tuple(latents.shape)}, decoded {tuple(decoded.shape)}")
