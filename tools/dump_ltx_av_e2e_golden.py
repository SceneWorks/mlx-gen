"""LTX-2.3 AudioVideo e2e pipeline golden — reference joint 2-stage denoise + decode (sc-2684 S5).

Composes the verified stages end-to-end through the REFERENCE: builds the AV `LTXModel` (Q8) +
upsampler + video VAE decoder + audio VAE decoder (attn OFF, the config-correct decode) + vocoder
(VocoderWithBWE), injects deterministic synthetic video+audio embeddings + per-stage noise, and runs
the reference `denoise_av` 2-stage orchestration (joint stage-1 → 2× upsample VIDEO + re-noise both →
joint stage-2) → video uint8 frames + audio waveform. The Rust `generate_av_latents` +
`decode_to_frames` + `decode_audio_track` (tests/av_e2e_parity.rs) inject the SAME conditioning and
must reproduce the frames (px>8) + waveform.

Everything runs **f32** (the Rust F32Q8 path). Golden MUST be mlx 0.31.2.

Run (mlx 0.31.2 env + mlx_video source):
    MLX_VIDEO_SRC=~/.cache/uv/archive-v0/DtG1XO51ABFxUGHg \
      /tmp/mlx312/bin/python tools/dump_ltx_av_e2e_golden.py
Output (committed): mlx-gen-ltx/tests/fixtures/ltx_av_e2e_golden.safetensors
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
_lang = types.ModuleType("mlx_vlm.models.gemma3.language")
_lang.Gemma3Model = object
sys.modules["mlx_vlm.models.gemma3.language"] = _lang
_cfg = types.ModuleType("mlx_vlm.models.gemma3.config")
_cfg.TextConfig = object
sys.modules["mlx_vlm.models.gemma3.config"] = _cfg

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402
from mlx.utils import tree_map  # noqa: E402

from mlx_video.generate_av import (  # noqa: E402
    DEFAULT_STAGE_1_SIGMAS,
    DEFAULT_STAGE_2_SIGMAS,
    create_audio_position_grid,
    create_video_position_grid,
    denoise_av,
)
from mlx_video.models.ltx.config import LTXModelConfig, LTXModelType, LTXRopeType  # noqa: E402
from mlx_video.models.ltx.ltx import LTXModel  # noqa: E402
from mlx_video.models.ltx.upsampler import load_upsampler, upsample_latents  # noqa: E402
from mlx_video.models.ltx.video_vae.decoder import load_vae_decoder  # noqa: E402
from mlx_video.models.ltx.audio_vae import AudioDecoder, CausalityAxis, NormType  # noqa: E402
from mlx_video.generate_av import load_vocoder  # noqa: E402

MODEL = Path.home() / "Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8"
DIM, ADIM, CTX, ACTX = 4096, 2048, 24, 16
LF, LH1, LW1, LH2, LW2 = 2, 4, 4, 8, 8  # 256×256, 9 frames
AF = 9  # audio frames (compute_audio_frames(9, 24) = 9)


def f32(model):
    model.update(tree_map(lambda p: p.astype(mx.float32) if p.dtype != mx.uint32 else p, model.parameters()))
    mx.eval(model.parameters())


config = LTXModelConfig(
    model_type=LTXModelType.AudioVideo, num_attention_heads=32, attention_head_dim=128,
    in_channels=128, out_channels=128, num_layers=48, cross_attention_dim=4096,
    caption_channels=4096, caption_projection_first_linear=False,
    caption_projection_second_linear=False, adaln_embedding_coefficient=9,
    apply_gated_attention=True, audio_num_attention_heads=32, audio_attention_head_dim=64,
    audio_in_channels=128, audio_out_channels=128, audio_cross_attention_dim=2048,
    audio_caption_channels=2048, rope_type=LTXRopeType.SPLIT, double_precision_rope=True,
    positional_embedding_theta=10000.0, positional_embedding_max_pos=[20, 2048, 2048],
    audio_positional_embedding_max_pos=[20], use_middle_indices_grid=True,
    timestep_scale_multiplier=1000,
)
transformer = LTXModel(config)
raw = mx.load(str(MODEL / "transformer.safetensors"))
quantized_paths = {k.rsplit(".", 1)[0] for k in raw if k.endswith(".scales")}
nn.quantize(transformer, group_size=64, bits=8,
            class_predicate=lambda p, m: isinstance(m, nn.Linear) and p in quantized_paths)
transformer.load_weights(list(raw.items()), strict=False)
f32(transformer)

upsampler = load_upsampler(str(MODEL), use_unified=True)
f32(upsampler)
vae_decoder = load_vae_decoder(str(MODEL), timestep_conditioning=None, use_unified=True)
mx.eval(vae_decoder.parameters())  # decode stays f32-internally; leave as loaded

audio_decoder = AudioDecoder(ch=128, out_ch=2, ch_mult=(1, 2, 4), num_res_blocks=2,
                             attn_resolutions={8, 16, 32}, resolution=256, z_channels=8,
                             norm_type=NormType.PIXEL, causality_axis=CausalityAxis.HEIGHT,
                             mel_bins=64, mid_block_add_attention=False)
araw = mx.load(str(MODEL / "audio_vae.safetensors"))
audio_decoder.load_weights([(k, v) for k, v in araw.items()], strict=False)
audio_decoder.per_channel_statistics._mean_of_means = araw["per_channel_statistics._mean_of_means"]
audio_decoder.per_channel_statistics._std_of_means = araw["per_channel_statistics._std_of_means"]
f32(audio_decoder)
vocoder = load_vocoder(MODEL, use_unified=True)
f32(vocoder)
audio_sr = int(getattr(vocoder, "output_sampling_rate", getattr(vocoder, "output_sample_rate", 24000)))

# Deterministic synthetic conditioning + noise (f32).
mx.random.seed(20)
video_ctx = (mx.random.normal((1, CTX, DIM)) * 0.5).astype(mx.float32)
audio_ctx = (mx.random.normal((1, ACTX, ADIM)) * 0.5).astype(mx.float32)
video_s1 = mx.random.normal((1, 128, LF, LH1, LW1)).astype(mx.float32)
video_s2 = mx.random.normal((1, 128, LF, LH2, LW2)).astype(mx.float32)
audio_s1 = mx.random.normal((1, 8, AF, 16)).astype(mx.float32)
audio_s2 = mx.random.normal((1, 8, AF, 16)).astype(mx.float32)
mx.eval(video_ctx, audio_ctx, video_s1, video_s2, audio_s1, audio_s2)

vpos1 = create_video_position_grid(1, LF, LH1, LW1)
vpos2 = create_video_position_grid(1, LF, LH2, LW2)
apos = create_audio_position_grid(1, AF)

S1, S2 = list(DEFAULT_STAGE_1_SIGMAS), list(DEFAULT_STAGE_2_SIGMAS)
# Stage 1 (joint).
vlat, alat = denoise_av(video_s1, audio_s1, vpos1, apos, video_ctx, audio_ctx, None, None,
                        transformer, S1, verbose=False, stage=1, cfg_scale=1.0, use_legacy_euler=True)
# Upsample video; re-noise both.
vlat = upsample_latents(vlat, upsampler, vae_decoder.latents_mean, vae_decoder.latents_std)
ns = mx.array(S2[0], dtype=mx.float32)
vlat = video_s2 * ns + vlat * (mx.array(1.0, dtype=mx.float32) - ns)
alat = audio_s2 * ns + alat * (mx.array(1.0, dtype=mx.float32) - ns)
# Stage 2 (joint).
vlat, alat = denoise_av(vlat, alat, vpos2, apos, video_ctx, audio_ctx, None, None,
                        transformer, S2, verbose=False, stage=2, cfg_scale=1.0, use_legacy_euler=True)
mx.eval(vlat, alat)

# Decode video → uint8 frames (F, H, W, 3).
video = vae_decoder(vlat)
video = mx.squeeze(video, axis=0)
video = mx.transpose(video, (1, 2, 3, 0))
video = (mx.clip((video + 1.0) / 2.0, 0.0, 1.0) * 255).astype(mx.uint8)
# Decode audio → waveform (raw vocoder output, pre-normalize).
mel = audio_decoder(alat)
wav = vocoder(mel)
mx.eval(video, wav)
print(f"av e2e: video_latents {vlat.shape} frames {video.shape} | audio_latents {alat.shape} wav {wav.shape} sr {audio_sr}")

tensors = {
    "video_ctx": video_ctx, "audio_ctx": audio_ctx,
    "video_s1": video_s1, "video_s2": video_s2, "audio_s1": audio_s1, "audio_s2": audio_s2,
    "video_latents": vlat.astype(mx.float32), "audio_latents": alat.astype(mx.float32),
    "frames": video, "waveform": wav.astype(mx.float32),
}
out = fixture("mlx-gen-ltx/tests/fixtures/ltx_av_e2e_golden.safetensors")
Path(out).parent.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(out, tensors, metadata={"sr": str(audio_sr), "lf": str(LF), "af": str(AF)})
print(f"wrote {out}")
