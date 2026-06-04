"""LTX-2.3 full text-encoder golden — reference video_embeddings (sc-2679 S1).

Reuses the Gemma golden's 49 hidden states (from dump_ltx_gemma_golden.py) so it does NOT reload
the 24 GB Gemma. Runs the reference feature extractor + connector on those hiddens:
  norm_and_concat_per_token_rms → rescale_norm → video_aggregate_embed → Embeddings1DConnector,
dumping the intermediate video_features and the final video_embeddings. The Rust
`LtxTextEncoder::encode` (tests/te_parity.rs) must reproduce these (it runs the real Gemma forward
itself, so this is the end-to-end S1 gate).

Weights come from a converted BASE model's connector.safetensors (NOT eros). Run:
    MLX_VIDEO_SRC=~/.cache/uv/archive-v0/DtG1XO51ABFxUGHg \
      ~/Repos/mflux/.venv/bin/python tools/dump_ltx_te_golden.py
Output (gitignored): tools/golden/ltx_te_golden.safetensors
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
# Stub mlx_vlm so text_encoder.py imports without the mlx_lm tree (we don't use the Gemma class).
for _name in ("mlx_vlm", "mlx_vlm.models", "mlx_vlm.models.gemma3"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
_lang = types.ModuleType("mlx_vlm.models.gemma3.language")
_lang.Gemma3Model = object
sys.modules["mlx_vlm.models.gemma3.language"] = _lang
_cfg = types.ModuleType("mlx_vlm.models.gemma3.config")
_cfg.TextConfig = object
sys.modules["mlx_vlm.models.gemma3.config"] = _cfg

import mlx.core as mx  # noqa: E402

from mlx_video.models.ltx.text_encoder import (  # noqa: E402
    Embeddings1DConnector,
    norm_and_concat_per_token_rms,
    rescale_norm,
)

BASE = Path.home() / "Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8"
HIDDEN, OUT_DIM = 3840, 4096
# eros/base connector config.
DIM, HEADS, HEAD_DIM, LAYERS, REGISTERS, MAX_POS = 4096, 32, 128, 8, 128, [4096]
# Audio connector config (sc-2684): dim 2048 = 32 heads × 64 head_dim, same 8 layers / 128 regs.
AUDIO_OUT_DIM, AUDIO_DIM, AUDIO_HEADS, AUDIO_HEAD_DIM = 2048, 2048, 32, 64

gemma_golden = fixture("tools/golden/ltx_gemma_golden.safetensors")
g = mx.load(gemma_golden)
num = sum(1 for k in g if k.startswith("h_"))
all_hidden = [g[f"h_{i:02d}"].astype(mx.bfloat16) for i in range(num)]  # back to bf16 (lossless)
attention_mask = g["attention_mask"]

# Feature extractor (v2 / per-token-RMS path).
normed = norm_and_concat_per_token_rms(all_hidden, attention_mask).astype(mx.bfloat16)
rescaled = rescale_norm(normed, OUT_DIM, HIDDEN)

raw = mx.load(str(BASE / "connector.safetensors"))
agg_w = raw["text_embedding_projection.video_aggregate_embed.weight"].astype(mx.bfloat16)
agg_b = raw["text_embedding_projection.video_aggregate_embed.bias"].astype(mx.bfloat16)
video_features = rescaled @ agg_w.T + agg_b  # Linear(188160 -> 4096)

# Connector (8-layer gated) — load video_embeddings_connector.* with the reference key remap.
conn = Embeddings1DConnector(
    dim=DIM, num_heads=HEADS, head_dim=HEAD_DIM, num_layers=LAYERS,
    num_learnable_registers=REGISTERS, positional_embedding_max_pos=MAX_POS,
    apply_gated_attention=True,
)
prefix = "video_embeddings_connector."
mapped, registers = {}, None
for k, v in raw.items():
    if not k.startswith(prefix):
        continue
    sub = k[len(prefix):]
    v = v.astype(mx.bfloat16)
    if sub == "learnable_registers":
        registers = v
        continue
    sub = sub.replace(".ff.net.0.proj.", ".ff.proj_in.").replace(".ff.net.2.", ".ff.proj_out.")
    sub = sub.replace(".to_out.0.", ".to_out.")
    mapped[sub] = v
conn.load_weights(list(mapped.items()), strict=False)
if registers is not None:
    conn.learnable_registers = registers
mx.eval(conn.parameters())

additive = (attention_mask.astype(mx.bfloat16) - 1.0).reshape(attention_mask.shape[0], 1, 1, -1) * 1e9
video_embeddings, _ = conn(video_features, additive)
mx.eval(video_embeddings)

# --- Audio half (sc-2684): separate aggregate_embed + connector off the SAME normed_hidden. ---
audio_rescaled = rescale_norm(normed, AUDIO_OUT_DIM, HIDDEN)
audio_agg_w = raw["text_embedding_projection.audio_aggregate_embed.weight"].astype(mx.bfloat16)
audio_agg_b = raw["text_embedding_projection.audio_aggregate_embed.bias"].astype(mx.bfloat16)
audio_features = audio_rescaled @ audio_agg_w.T + audio_agg_b  # Linear(188160 -> 2048)

audio_conn = Embeddings1DConnector(
    dim=AUDIO_DIM, num_heads=AUDIO_HEADS, head_dim=AUDIO_HEAD_DIM, num_layers=LAYERS,
    num_learnable_registers=REGISTERS, positional_embedding_max_pos=MAX_POS,
    apply_gated_attention=True,
)
aprefix = "audio_embeddings_connector."
amapped, aregisters = {}, None
for k, v in raw.items():
    if not k.startswith(aprefix):
        continue
    sub = k[len(aprefix):]
    v = v.astype(mx.bfloat16)
    if sub == "learnable_registers":
        aregisters = v
        continue
    sub = sub.replace(".ff.net.0.proj.", ".ff.proj_in.").replace(".ff.net.2.", ".ff.proj_out.")
    sub = sub.replace(".to_out.0.", ".to_out.")
    amapped[sub] = v
audio_conn.load_weights(list(amapped.items()), strict=False)
if aregisters is not None:
    audio_conn.learnable_registers = aregisters
mx.eval(audio_conn.parameters())
audio_embeddings, _ = audio_conn(audio_features, additive)
mx.eval(audio_embeddings)

tensors = {
    "input_ids": g["input_ids"],
    "attention_mask": attention_mask,
    "video_features": video_features.astype(mx.float32),
    "video_embeddings": video_embeddings.astype(mx.float32),
    "audio_features": audio_features.astype(mx.float32),
    "audio_embeddings": audio_embeddings.astype(mx.float32),
}
out = fixture("tools/golden/ltx_te_golden.safetensors")
Path(out).parent.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(out, tensors, metadata={"hidden": str(HIDDEN), "out_dim": str(OUT_DIM)})
print(f"wrote {out}")
print(f"  video_features {video_features.shape}  video_embeddings {video_embeddings.shape}")
print(f"  audio_features {audio_features.shape}  audio_embeddings {audio_embeddings.shape}")
