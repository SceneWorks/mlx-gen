"""LTX-2.3 LoRA-in-generate golden (sc-2687).

Applies a real LTX-2.3 LoRA to the shipped `ltx_2_3_base_q8` transformer via the reference's
**forward-time residual** (`mlx_video/lora/apply.py::LoRALinear` — `out + scale·strength·(x·Aᵀ·Bᵀ)`,
NOT the merge path), then runs the **native bf16 + Q8** 2-stage distilled denoise → frames. This is
the exact strategy the Rust port uses (`Precision::Bf16Q8` + a residual stack), so the golden gates
the Rust LoRA path byte-for-byte (the chaos-sensitive stage-1 demands a bit-exact per-forward).

Inputs (`video_embeddings` + the two stage noises + position grids) are reused verbatim from the
committed bf16 e2e golden, so this is fully deterministic and needs no text encoder.

Default LoRA = `LTX2.3_Crisp_Enhance` (attn + ff + gate, 576 video targets, no audio/adaLN — a clean
1:1 with the video-only port). Override with `LTX_LORA=/path/to/lora.safetensors` and
`LTX_LORA_STRENGTH=1.0`.

**MUST run with mlx 0.31.2** (the Q8 `quantized_matmul` kernel). Example:
    MLX_VIDEO_SRC=~/.cache/uv/archive-v0/DtG1XO51ABFxUGHg /tmp/mlx312/bin/python tools/dump_ltx_lora_golden.py
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

from mlx_video.generate_av import create_video_position_grid  # noqa: E402
from mlx_video.lora.apply import LoRALinear, _normalize_lora_key  # noqa: E402
from mlx_video.lora.loader import load_lora_weights  # noqa: E402
from mlx_video.models.ltx.config import LTXModelConfig, LTXModelType, LTXRopeType  # noqa: E402
from mlx_video.models.ltx.ltx import LTXModel  # noqa: E402
from mlx_video.models.ltx.transformer import Modality  # noqa: E402
from mlx_video.models.ltx.upsampler import load_upsampler, upsample_latents  # noqa: E402
from mlx_video.models.ltx.video_vae.decoder import load_vae_decoder  # noqa: E402
from mlx_video.utils import to_denoised  # noqa: E402

MODEL = Path.home() / "Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8"
LORA = Path(
    os.environ.get(
        "LTX_LORA",
        str(
            Path.home()
            / "Library/Application Support/SceneWorks/data/loras/crisp_enhance/LTX2.3_Crisp_Enhance.safetensors"
        ),
    )
)
STRENGTH = float(os.environ.get("LTX_LORA_STRENGTH", "1.0"))
STAGE1_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
STAGE2_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]
LF, H1, W1, H2, W2 = 2, 4, 4, 8, 8

# Reuse the committed bf16 e2e golden's injected inputs (deterministic, no text encoder).
E2E = mx.load(fixture("mlx-gen-ltx/tests/fixtures/ltx_e2e_golden_bf16.safetensors"))
context = E2E["video_embeddings"]  # (1, 128, 4096) bf16
stage1_noise = E2E["stage1_noise"]
stage2_noise = E2E["stage2_noise"]


config = LTXModelConfig(
    model_type=LTXModelType.AudioVideo,
    num_attention_heads=32, attention_head_dim=128, in_channels=128, out_channels=128,
    num_layers=48, cross_attention_dim=4096, caption_channels=4096,
    caption_projection_first_linear=False, caption_projection_second_linear=False,
    adaln_embedding_coefficient=9, apply_gated_attention=True,
    audio_num_attention_heads=32, audio_attention_head_dim=64, audio_in_channels=128,
    audio_out_channels=128, audio_cross_attention_dim=2048, audio_caption_channels=2048,
    rope_type=LTXRopeType.SPLIT, double_precision_rope=True,
    positional_embedding_theta=10000.0, positional_embedding_max_pos=[20, 2048, 2048],
    audio_positional_embedding_max_pos=[20], use_middle_indices_grid=True,
    timestep_scale_multiplier=1000,
)
model = LTXModel(config)
raw = mx.load(str(MODEL / "transformer.safetensors"))
video = {k: v for k, v in raw.items() if "audio" not in k and "av_ca" not in k and "a2v" not in k}
qpaths = {k.rsplit(".", 1)[0] for k in video if k.endswith(".scales")}
nn.quantize(model, group_size=64, bits=8,
            class_predicate=lambda p, m: isinstance(m, nn.Linear) and p in qpaths)
model.load_weights(list(video.items()), strict=False)


def wrap_loras(root, lora_weights, strength):
    """Wrap each resolved target Linear/QuantizedLinear in a residual `LoRALinear` (the reference's
    forward-time path), counting applied vs skipped — mirrors `apply_loras_to_model`'s traversal but
    wraps instead of merging, so the base stays Q8 (what the Rust residual does)."""
    module_paths = set()
    for name, _ in root.named_modules():
        module_paths.add(name)
        module_paths.add(f"{name}.weight")
    applied, skipped = 0, []
    for lora_key, weights in lora_weights.items():
        norm = _normalize_lora_key(lora_key, module_paths)
        if norm.endswith(".weight"):
            norm = norm[: -len(".weight")]
        parts = norm.split(".")
        parent = root
        try:
            for part in parts[:-1]:
                parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
            leaf = parts[-1]
            target = getattr(parent, leaf) if not leaf.isdigit() else parent[int(leaf)]
        except (AttributeError, IndexError, TypeError):
            skipped.append(lora_key)
            continue
        if isinstance(target, (nn.Linear, nn.QuantizedLinear)):
            wrapped = LoRALinear(target, [(weights, strength)])
            if leaf.isdigit():
                parent[int(leaf)] = wrapped
            else:
                setattr(parent, leaf, wrapped)
            applied += 1
        else:
            skipped.append(lora_key)
    return applied, skipped


lora_weights = load_lora_weights(LORA)
applied, skipped = wrap_loras(model, lora_weights, STRENGTH)
print(f"LoRA {LORA.name}: applied={applied} skipped={len(skipped)} strength={STRENGTH}")
mx.eval(model.parameters())


def forward_velocity(video_flat, timesteps, positions):
    modality = Modality(latent=video_flat, timesteps=timesteps, positions=positions,
                        context=context, context_mask=None, enabled=True)
    args = model.video_args_preprocessor.prepare(modality, None)
    emb_ts = args.embedded_timestep
    v = args
    for block in model.transformer_blocks.values():
        v, _ = block(video=v, audio=None)
    return model._process_output(model.scale_shift_table, model.norm_out, model.proj_out, v.x, emb_ts)


def denoise(latents, positions, sigmas):
    dtype = latents.dtype
    lat = latents
    for i in range(len(sigmas) - 1):
        sigma, sn = sigmas[i], sigmas[i + 1]
        b, c, f, h, w = lat.shape
        flat = mx.transpose(mx.reshape(lat, (b, c, -1)), (0, 2, 1))
        ts = mx.full((b, f * h * w), sigma, dtype=dtype)
        vel = forward_velocity(flat, ts, positions)
        vel = mx.reshape(mx.transpose(vel, (0, 2, 1)), (b, c, f, h, w))
        den = to_denoised(lat, vel, sigma)
        if sn > 0:
            lat = den + mx.array(sn, dtype=dtype) * (lat - den) / mx.array(sigma, dtype=dtype)
        else:
            lat = den
        mx.eval(lat)
    return lat


upsampler = load_upsampler(str(MODEL / "upsampler.safetensors"))
vae = load_vae_decoder(str(MODEL), timestep_conditioning=None, use_unified=True)
mx.eval(upsampler.parameters(), vae.parameters())

pos1 = create_video_position_grid(1, LF, H1, W1)
pos2 = create_video_position_grid(1, LF, H2, W2)

s1 = denoise(stage1_noise, pos1, STAGE1_SIGMAS)
ups = upsample_latents(s1, upsampler, vae.latents_mean, vae.latents_std)
ns = mx.array(STAGE2_SIGMAS[0], dtype=ups.dtype)
renoised = stage2_noise * ns + ups * (mx.array(1.0, dtype=ups.dtype) - ns)
final_latents = denoise(renoised, pos2, STAGE2_SIGMAS)

vid = vae(final_latents)
vid = mx.transpose(mx.squeeze(vid, axis=0), (1, 2, 3, 0))
frames = (mx.clip((vid + 1.0) / 2.0, 0.0, 1.0) * 255).astype(mx.uint8)
mx.eval(final_latents, frames)
print(f"lora e2e: context{context.shape} -> final{final_latents.shape} -> frames{frames.shape}")

tensors = {
    "video_embeddings": context.astype(mx.bfloat16),
    "stage1_noise": stage1_noise,
    "stage2_noise": stage2_noise,
    "stage1_positions": pos1.astype(mx.float32),
    "stage2_positions": pos2.astype(mx.float32),
    "final_latents": final_latents.astype(mx.bfloat16),
    "frames": frames,
}
out = fixture("mlx-gen-ltx/tests/fixtures/ltx_lora_golden.safetensors")
Path(out).parent.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(
    out, tensors,
    metadata={"lora": LORA.name, "strength": str(STRENGTH), "applied": str(applied),
              "skipped": str(len(skipped)), "res": "256x256", "frames": "9", "prec": "bf16"},
)
print(f"wrote {out}")
