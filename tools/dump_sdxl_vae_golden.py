"""SDXL VAE golden — reference for mlx-gen-sdxl S4/S6 (sc-2400).

Runs the EXACT vendored Apple `Autoencoder` (`_vendor/mlx_sd/vae.py`, always f32) for a decode
(random latents → image) and an encode (random image → latent mean), so the Rust VAE port can be
validated to tight tolerance. SDXL `scaling_factor` is 0.13025.

Run from the mflux venv:
  /Users/michael/Repos/mflux/.venv/bin/python3 tools/dump_sdxl_vae_golden.py
"""

import os
import sys

import mlx.core as mx

os.environ.setdefault("HF_HUB_OFFLINE", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_DIR = os.path.join(_HERE, "golden")
os.makedirs(_GOLDEN_DIR, exist_ok=True)

VENDOR_PARENT = os.environ.get(
    "SDXL_VENDOR_PARENT",
    "/Users/michael/Repos/SceneWorks/apps/worker/scene_worker/_vendor",
)
sys.path.insert(0, VENDOR_PARENT)
from mlx_sd import StableDiffusionXL  # noqa: E402

REPO = "stabilityai/stable-diffusion-xl-base-1.0"
sd = StableDiffusionXL(REPO, float16=False)
sd.ensure_models_are_loaded()

# Decode: random latents [1, 64, 64, 4] -> image [1, 512, 512, 3].
mx.random.seed(1)
latents = mx.random.normal((1, 64, 64, 4)).astype(mx.float32)
decoded = sd.autoencoder.decode(latents)

# Encode: random image in [-1, 1], NHWC [1, 256, 256, 3] -> latent mean [1, 32, 32, 4].
mx.random.seed(2)
image = (mx.random.uniform(shape=(1, 256, 256, 3)) * 2 - 1).astype(mx.float32)
mean, logvar = sd.autoencoder.encode(image)

mx.eval(decoded, mean, logvar)

tensors = {
    "latents": latents.astype(mx.float32),
    "decoded": decoded.astype(mx.float32),
    "image": image.astype(mx.float32),
    "enc_mean": mean.astype(mx.float32),
}
out = os.path.join(_GOLDEN_DIR, "sdxl_vae_golden.safetensors")
mx.save_safetensors(out, tensors, {"scaling_factor": "0.13025"})
print(f"wrote {out}")
print(f"  decode {tuple(latents.shape)} -> {tuple(decoded.shape)}")
print(f"  encode {tuple(image.shape)} -> mean {tuple(mean.shape)}")
