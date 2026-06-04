"""LTX-2.3 AudioVideo DiT golden — reference joint `LTXModel(video, audio)` velocities (sc-2684 S2).

Builds the reference **AudioVideo** `LTXModel` (the dual-modality DiT: 48 layers, video dim 4096 +
audio dim 2048 + bidirectional cross-modal attention), loads the real `ltx_2_3_base_q8`
`transformer.safetensors` (the WHOLE model — video + audio + `av_ca`/`a2v`/`v2a` cross modules),
`nn.quantize`-ing every Q8 Linear (group 64 / 8-bit), and runs ONE joint forward over deterministic
synthetic video+audio inputs. The Rust `AvDiT` (mlx-gen-ltx/tests/av_dit_parity.rs, `Precision::F32Q8`)
loads the SAME Q8 weights and must reproduce BOTH velocities.

f32 activations are the quality target (`Precision::F32Q8`); `LTX_BF16=1` emits the reference's native
bf16+Q8 forward (`Precision::Bf16Q8`). Like the video-only gate, the distilled stage-1 sampler is
chaos-sensitive, so each per-forward must be bit-exact — hence the dedicated gate.

**The golden MUST be generated with mlx 0.31.2** (matching the Rust build): `quantized_matmul` changed
0.31.0→0.31.2.

Run (mlx 0.31.2 env + mlx_video source):
    MLX_VIDEO_SRC=~/.cache/uv/archive-v0/DtG1XO51ABFxUGHg \
      /tmp/mlx312/bin/python tools/dump_ltx_av_dit_golden.py
    LTX_BF16=1 MLX_VIDEO_SRC=... /tmp/mlx312/bin/python tools/dump_ltx_av_dit_golden.py
Output (committed): mlx-gen-ltx/tests/fixtures/ltx_av_dit_golden{,_bf16}.safetensors
"""

import glob
import os
import sys
import types
from pathlib import Path

import numpy as np

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
_lang = types.ModuleType("mlx_vlm.models.gemma3.language")
_lang.Gemma3Model = object
sys.modules["mlx_vlm.models.gemma3.language"] = _lang
_cfg = types.ModuleType("mlx_vlm.models.gemma3.config")
_cfg.TextConfig = object
sys.modules["mlx_vlm.models.gemma3.config"] = _cfg

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from mlx_video.generate import create_position_grid  # noqa: E402
from mlx_video.models.ltx.config import (  # noqa: E402
    LTXModelConfig,
    LTXModelType,
    LTXRopeType,
)
from mlx_video.models.ltx.ltx import LTXModel  # noqa: E402
from mlx_video.models.ltx.transformer import Modality  # noqa: E402

MODEL = Path.home() / "Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8"
DIM, ADIM, CTX, ACTX = 4096, 2048, 16, 12
LF, LH, LW = 2, 4, 4  # video: S_v = 32 tokens
AT = 6  # audio latent frames (S_a)

BF16 = os.environ.get("LTX_BF16") == "1"
ACT = mx.bfloat16 if BF16 else mx.float32


def create_audio_position_grid(batch_size, audio_frames, sample_rate=16000, hop_length=160,
                               downsample_factor=4, is_causal=True):
    """Inlined copy of generate_av.create_audio_position_grid (avoids importing generate_av)."""
    def t(s, e):
        lf = np.arange(s, e, dtype=np.float32)
        mel = lf * downsample_factor
        if is_causal:
            mel = np.clip(mel + 1 - downsample_factor, 0, None)
        return mel * hop_length / sample_rate
    start = t(0, audio_frames)
    end = t(1, audio_frames + 1)
    pos = np.stack([start, end], axis=-1)[np.newaxis, np.newaxis, :, :]
    pos = np.tile(pos, (batch_size, 1, 1, 1))
    return mx.array(pos, dtype=mx.float32)


config = LTXModelConfig(
    model_type=LTXModelType.AudioVideo,
    num_attention_heads=32,
    attention_head_dim=128,
    in_channels=128,
    out_channels=128,
    num_layers=48,
    cross_attention_dim=4096,
    caption_channels=4096,
    caption_projection_first_linear=False,
    caption_projection_second_linear=False,
    adaln_embedding_coefficient=9,
    apply_gated_attention=True,
    audio_num_attention_heads=32,
    audio_attention_head_dim=64,
    audio_in_channels=128,
    audio_out_channels=128,
    audio_cross_attention_dim=2048,
    audio_caption_channels=2048,
    rope_type=LTXRopeType.SPLIT,
    double_precision_rope=True,
    positional_embedding_theta=10000.0,
    positional_embedding_max_pos=[20, 2048, 2048],
    audio_positional_embedding_max_pos=[20],
    use_middle_indices_grid=True,
    timestep_scale_multiplier=1000,
)
model = LTXModel(config)

# Load the WHOLE transformer (video + audio + cross), quantize the Q8 Linears, load.
raw = mx.load(str(MODEL / "transformer.safetensors"))
quantized_paths = {k.rsplit(".", 1)[0] for k in raw if k.endswith(".scales")}
print(f"keys {len(raw)}, quantized Linears {len(quantized_paths)}")


def _should_quantize(path, module):
    return isinstance(module, nn.Linear) and path in quantized_paths


nn.quantize(model, group_size=64, bits=8, class_predicate=_should_quantize)
model.load_weights(list(raw.items()), strict=False)
if not BF16:
    from mlx.utils import tree_map  # noqa: E402

    model.update(
        tree_map(
            lambda p: p.astype(mx.float32) if p.dtype != mx.uint32 else p,
            model.parameters(),
        )
    )
mx.eval(model.parameters())

sigma = 0.909375 if BF16 else 0.5
mx.random.seed(7)
video_latent = (mx.random.normal((1, LF * LH * LW, 128)) * 0.5).astype(ACT)
audio_latent = (mx.random.normal((1, AT, 128)) * 0.5).astype(ACT)
video_context = (mx.random.normal((1, CTX, DIM)) * 0.5).astype(ACT)
audio_context = (mx.random.normal((1, ACTX, ADIM)) * 0.5).astype(ACT)
video_timestep = mx.full((1, LF * LH * LW), sigma, dtype=ACT)
audio_timestep = mx.full((1, AT), sigma, dtype=ACT)
video_positions = create_position_grid(1, LF, LH, LW)  # (1, 3, 32, 2) f32
audio_positions = create_audio_position_grid(1, AT)  # (1, 1, AT, 2) f32

video_modality = Modality(
    latent=video_latent, timesteps=video_timestep, positions=video_positions,
    context=video_context, context_mask=None, enabled=True,
)
audio_modality = Modality(
    latent=audio_latent, timesteps=audio_timestep, positions=audio_positions,
    context=audio_context, context_mask=None, enabled=True,
)

vx, ax = model(video=video_modality, audio=audio_modality)
mx.eval(vx, ax)
print(f"av dit: video {vx.shape} dtype={vx.dtype}  audio {ax.shape} dtype={ax.dtype}")

tensors = {
    "video_latent": video_latent,
    "audio_latent": audio_latent,
    "video_context": video_context,
    "audio_context": audio_context,
    "video_timestep": video_timestep,
    "audio_timestep": audio_timestep,
    "video_positions": video_positions.astype(mx.float32),
    "audio_positions": audio_positions.astype(mx.float32),
    "video_velocity": vx.astype(ACT),
    "audio_velocity": ax.astype(ACT),
}
name = "ltx_av_dit_golden_bf16.safetensors" if BF16 else "ltx_av_dit_golden.safetensors"
out_path = fixture(f"mlx-gen-ltx/tests/fixtures/{name}")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(out_path, tensors, metadata={"sv": str(LF * LH * LW), "sa": str(AT)})
print(f"wrote {out_path}")
