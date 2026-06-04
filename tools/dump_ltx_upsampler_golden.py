"""LTX-2.3 spatial latent-upsampler golden — reference `upsample_latents` I/O (sc-2679 S4).

Loads the **real** `upsampler.safetensors` (the `ltx-2-spatial-upscaler-x2` checkpoint) from the
on-disk `ltx_2_3_base_q8` snapshot via the reference `load_upsampler` (which handles the PyTorch→MLX
conv-weight transposes and the `upsampler.0.→upsampler.conv.` rename), pulls the VAE
`per_channel_statistics.{mean,std}` (= `vae_decoder.latents_mean/std`), and runs the reference
`upsample_latents` (un-normalize → 2× spatial upsample → re-normalize) over a deterministic synthetic
latent. Everything stays **bf16** — exactly the production path (`generate.py` loads the upsampler
bf16, feeds bf16 stage-1 latents, and `upsample_latents` keeps `mean`/`std` bf16). The Rust
`LatentUpsampler`/`upsample_latents` (mlx-gen-ltx/tests/upsampler_parity.rs) loads the SAME bf16
weights and must reproduce the output.

The upsampler is pure dense (conv + group-norm, **no quantized ops**), and dense ops are bit-identical
across mlx 0.31.0/0.31.2 — so either venv works. Uses the mflux venv for convenience; the Rust build
is 0.31.2.

Small shapes keep the fixture tiny while exercising every path: 2 latent frames (Conv3d temporal
kernel) × 8×8 spatial → 16×16 (2× pixel-shuffle), 1024 mid-channels (32-group norm).

Run (mflux venv + mlx_video source):
    MLX_VIDEO_SRC=~/.cache/uv/archive-v0/DtG1XO51ABFxUGHg \
      ~/Repos/mflux/.venv/bin/python tools/dump_ltx_upsampler_golden.py
Output (committed): mlx-gen-ltx/tests/fixtures/ltx_upsampler_golden.safetensors
"""

import glob
import os
import sys
from pathlib import Path

from _paths import fixture


def _find_mlx_video_src() -> str:
    if env := os.environ.get("MLX_VIDEO_SRC"):
        return str(Path(env).expanduser())
    for cand in sorted(glob.glob(str(Path.home() / ".cache/uv/archive-v0/*/mlx_video"))):
        return str(Path(cand).parent)
    raise SystemExit("Set MLX_VIDEO_SRC to the dir containing `mlx_video/`.")


sys.path.insert(0, _find_mlx_video_src())

import mlx.core as mx  # noqa: E402

from mlx_video.models.ltx.upsampler import load_upsampler, upsample_latents  # noqa: E402

MODEL = Path.home() / "Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8"
LF, LH, LW = 2, 8, 8  # latent frames / height / width → upsampled to 2×16×16

upsampler = load_upsampler(str(MODEL / "upsampler.safetensors"))
mx.eval(upsampler.parameters())

# latents_mean/std = the VAE `per_channel_statistics` (bf16), as `vae_decoder.latents_{mean,std}`.
vae = mx.load(str(MODEL / "vae_decoder.safetensors"))
latent_mean = vae["per_channel_statistics.mean"]  # (128,) bf16
latent_std = vae["per_channel_statistics.std"]  # (128,) bf16
print(f"latents stats: mean{latent_mean.shape} std{latent_std.shape} dtype={latent_mean.dtype}")

# Deterministic synthetic latent — channels-first NCFHW, **bf16** (the production stage-1 dtype).
mx.random.seed(7)
latent = (mx.random.normal((1, 128, LF, LH, LW)) * 0.5).astype(mx.bfloat16)

out = upsample_latents(latent, upsampler, latent_mean, latent_std)
mx.eval(out)
print(f"upsampler: latent{latent.shape} -> {out.shape} dtype={out.dtype}")

tensors = {
    "latent": latent,
    "latent_mean": latent_mean,
    "latent_std": latent_std,
    "output": out,
}
out_path = fixture("mlx-gen-ltx/tests/fixtures/ltx_upsampler_golden.safetensors")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(out_path, tensors)
print(f"wrote {out_path}")
