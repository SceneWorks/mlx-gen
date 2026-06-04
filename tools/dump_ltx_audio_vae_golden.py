"""LTX-2.3 audio VAE decoder golden — reference `AudioDecoder` mel output (sc-2684 S3).

Constructs the reference `AudioDecoder` (ch 128, ch_mult (1,2,4), z 8, out 2, PIXEL norm, causal-on-
height) with **`mid_block_add_attention=False`** — the shipped `embedded_config.json` value (the
checkpoint ships no `mid.attn_1` weights). NB: the reference `generate_av.load_audio_decoder` HARDCODES
the constructor and so defaults this to True, building a *randomly-initialized* mid attention → its
audio decode is non-deterministic across processes (~4% of the signal). We honor the config (the
model's intended decode), so this golden is built with attention OFF and is reproducible.

Loads `audio_vae.safetensors`, upcasts to f32 (the Rust `AudioDecoder` runs f32 — a post-sampling
quality island), and decodes a deterministic synthetic latent `(1, 8, T, 16)` → mel `(1, 2, 4T-3, 64)`.

Run (mlx 0.31.2 env + mlx_video source):
    MLX_VIDEO_SRC=~/.cache/uv/archive-v0/DtG1XO51ABFxUGHg \
      /tmp/mlx312/bin/python tools/dump_ltx_audio_vae_golden.py
Output (committed): mlx-gen-ltx/tests/fixtures/ltx_audio_vae_golden.safetensors
"""

import glob
import os
import sys
import types
from pathlib import Path

from _paths import fixture


def _find_mlx_video_src() -> str:
    if env := os.environ.get("MLX_VIDEO_SRC"):
        return str(Path(env).expanduser())
    for cand in sorted(glob.glob(str(Path.home() / ".cache/uv/archive-v0/*/mlx_video"))):
        return str(Path(cand).parent)
    raise SystemExit("Set MLX_VIDEO_SRC to the dir containing `mlx_video/`.")


sys.path.insert(0, _find_mlx_video_src())
for _name in ("mlx_vlm", "mlx_vlm.models", "mlx_vlm.models.gemma3"):
    sys.modules.setdefault(_name, types.ModuleType(_name))

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_map  # noqa: E402

from mlx_video.models.ltx.audio_vae import (  # noqa: E402
    AudioDecoder,
    CausalityAxis,
    NormType,
)

MODEL = Path.home() / "Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8"
T = 6  # audio latent frames → mel time = 4T-3 = 21

dec = AudioDecoder(
    ch=128, out_ch=2, ch_mult=(1, 2, 4), num_res_blocks=2, attn_resolutions={8, 16, 32},
    resolution=256, z_channels=8, norm_type=NormType.PIXEL, causality_axis=CausalityAxis.HEIGHT,
    mel_bins=64, mid_block_add_attention=False,
)
raw = mx.load(str(MODEL / "audio_vae.safetensors"))
dec.load_weights([(k, v) for k, v in raw.items()], strict=False)
dec.per_channel_statistics._mean_of_means = raw["per_channel_statistics._mean_of_means"]
dec.per_channel_statistics._std_of_means = raw["per_channel_statistics._std_of_means"]
# f32 decode (the Rust path): upcast every param.
dec.update(tree_map(lambda p: p.astype(mx.float32), dec.parameters()))
mx.eval(dec.parameters())

mx.random.seed(11)
latent = (mx.random.normal((1, 8, T, 16)) * 0.7).astype(mx.float32)
mel = dec(latent)
mx.eval(mel)
print(f"audio vae: latent {latent.shape} -> mel {mel.shape} dtype={mel.dtype}")

tensors = {"latent": latent, "mel": mel.astype(mx.float32)}
out = fixture("mlx-gen-ltx/tests/fixtures/ltx_audio_vae_golden.safetensors")
Path(out).parent.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(out, tensors, metadata={"T": str(T)})
print(f"wrote {out}")
