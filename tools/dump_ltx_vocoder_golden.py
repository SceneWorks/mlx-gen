"""LTX-2.3 vocoder golden — reference `VocoderWithBWE` waveform (sc-2684 S4).

Builds the reference vocoder via `generate_av.load_vocoder` (the production loader: reads the
`embedded_config.json` `vocoder` block → BigVGAN core + BWE generator = `VocoderWithBWE`, 48 kHz,
applies the ConvTranspose1d layout fixup, loads `vocoder.safetensors`), upcasts to f32 (the Rust
`LtxVocoder` runs f32), and runs it on a deterministic synthetic mel `(1, 2, T, 64)`. The Rust
`LtxVocoder` loads the SAME weights and must reproduce the waveform.

Run (mlx 0.31.2 env + mlx_video source):
    MLX_VIDEO_SRC=~/.cache/uv/archive-v0/DtG1XO51ABFxUGHg \
      /tmp/mlx312/bin/python tools/dump_ltx_vocoder_golden.py
Output (committed): mlx-gen-ltx/tests/fixtures/ltx_vocoder_golden.safetensors
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

from mlx_video.generate_av import load_vocoder  # noqa: E402

MODEL = Path.home() / "Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8"
T = 21  # mel time frames (= audio VAE output 4·6-3)

voc = load_vocoder(MODEL, use_unified=True)
voc.update(tree_map(lambda p: p.astype(mx.float32), voc.parameters()))
mx.eval(voc.parameters())
sr = int(getattr(voc, "output_sampling_rate", getattr(voc, "output_sample_rate", 24000)))
print(f"vocoder = {type(voc).__name__}, output_sample_rate = {sr}")

mx.random.seed(13)
mel = (mx.random.normal((1, 2, T, 64)) * 3.0).astype(mx.float32)
# Intermediate taps for stage bisection: core-BigVGAN output, its mel, BWE residual, skip.
low = voc.vocoder(mel)
mel_from_low = voc._compute_mel(low)
mel_for_bwe = mx.transpose(mel_from_low, (0, 1, 3, 2))
residual = voc.bwe_generator(mel_for_bwe)
skip = voc._upsample_skip(low)
mx.eval(low, mel_from_low, residual, skip)
wav = voc(mel)
mx.eval(wav)
print(f"vocoder: mel {mel.shape} -> waveform {wav.shape} dtype={wav.dtype}")
print(f"  low {low.shape}  mel_from_low {mel_from_low.shape}  residual {residual.shape}  skip {skip.shape}")

# Core-BigVGAN stage taps (NLC), replicating BigVGANVocoder.__call__ to bisect `low`.
core = voc.vocoder
xx = mx.transpose(mel, (0, 1, 3, 2))
b, s, c, t = xx.shape
xx = mx.transpose(xx.reshape(b, s * c, t), (0, 2, 1))
xx = core.conv_pre(xx)
core_after_conv_pre = xx
for i in range(core.num_upsamples):
    xx = core.ups[i](xx)
    start = i * core.num_kernels
    bo = [core.resblocks[idx](xx) for idx in range(start, start + core.num_kernels)]
    xx = mx.mean(mx.stack(bo, axis=0), axis=0)
core_after_up = xx
core_after_act = core.act_post(xx)
# Isolation taps: ups[0] alone on conv_pre output; act_post alone on the (diverged) up output.
up0_only = core.ups[0](core_after_conv_pre)
act_on_up = core.act_post(core_after_up)
rb0_on_up0 = core.resblocks[0](up0_only)  # single AMPBlock1 (k=3, dil [1,3,5]) on golden input
mx.eval(core_after_conv_pre, core_after_up, core_after_act, up0_only, act_on_up, rb0_on_up0)

tensors = {
    "mel": mel,
    "waveform": wav.astype(mx.float32),
    "low": low.astype(mx.float32),
    "mel_from_low": mel_from_low.astype(mx.float32),
    "residual": residual.astype(mx.float32),
    "skip": skip.astype(mx.float32),
    "core_after_conv_pre": core_after_conv_pre.astype(mx.float32),
    "core_after_up": core_after_up.astype(mx.float32),
    "core_after_act": core_after_act.astype(mx.float32),
    "up0_only": up0_only.astype(mx.float32),
    "act_on_up": act_on_up.astype(mx.float32),
    "rb0_on_up0": rb0_on_up0.astype(mx.float32),
}
out = fixture("mlx-gen-ltx/tests/fixtures/ltx_vocoder_golden.safetensors")
Path(out).parent.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(out, tensors, metadata={"T": str(T), "sample_rate": str(sr)})
print(f"wrote {out}")
