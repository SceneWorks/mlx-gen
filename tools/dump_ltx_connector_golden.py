"""LTX-2.3 connector golden — reference Embeddings1DConnector I/O (sc-2679 S1).

Weight-grounded but Gemma-free: loads `video_embeddings_connector.*` from the eros
`connector.safetensors`, builds the reference `Embeddings1DConnector` with the eros config
(dim 4096, 32 heads × 128, 8 layers, gated, 128 registers, max_pos [4096]), runs an **f32**
forward over a deterministic left-padded random feature input, and dumps input / mask / output.
The Rust `Connector` (mlx-gen-ltx/tests/connector_parity.rs) loads the SAME connector.safetensors
weights and must reproduce the output.

f32 reference: the module weights + input are cast to f32 so the gate isolates connector
*correctness* from bf16 rounding (the Rust path runs f32 too; the connector's dense GEMMs have
K=4096 > 512, outside the pmetal bf16-GEMM bug regime, but f32 is the quality target regardless).

Run (mflux venv + mlx_video source):
    MLX_VIDEO_SRC=~/.cache/uv/archive-v0/DtG1XO51ABFxUGHg \
      ~/Repos/mflux/.venv/bin/python tools/dump_ltx_connector_golden.py
Output (committed): mlx-gen-ltx/tests/fixtures/ltx_connector_golden.safetensors
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

# text_encoder.py imports `mlx_vlm.models.gemma3.{language,config}` at module load (pulling in the
# whole mlx_lm/mlx_vlm tree). We only need `Embeddings1DConnector`, which never touches the Gemma
# class, so stub those names rather than installing the dependency tree.
import types  # noqa: E402

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

from mlx_video.models.ltx.text_encoder import Embeddings1DConnector  # noqa: E402

EROS = Path.home() / "Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_eros"

# eros connector config (from embedded_config.json).
DIM, HEADS, HEAD_DIM, LAYERS, REGISTERS, MAX_POS = 4096, 32, 128, 8, 128, [4096]
SEQ, NUM_VALID = 256, 40  # left-padded: 216 pad + 40 valid; SEQ % REGISTERS == 0.

mx.random.seed(0)

# Build the reference connector with the eros config + gated attention.
conn = Embeddings1DConnector(
    dim=DIM,
    num_heads=HEADS,
    head_dim=HEAD_DIM,
    num_layers=LAYERS,
    num_learnable_registers=REGISTERS,
    positional_embedding_max_pos=MAX_POS,
    apply_gated_attention=True,
)

# Load video connector weights from connector.safetensors with the reference key remapping.
raw = mx.load(str(EROS / "connector.safetensors"))
prefix = "video_embeddings_connector."
mapped = {}
registers = None
for k, v in raw.items():
    if not k.startswith(prefix):
        continue
    sub = k[len(prefix):]
    v = v.astype(mx.float32)  # f32 reference
    if sub == "learnable_registers":
        registers = v
        continue
    sub = sub.replace(".ff.net.0.proj.", ".ff.proj_in.")
    sub = sub.replace(".ff.net.2.", ".ff.proj_out.")
    sub = sub.replace(".to_out.0.", ".to_out.")
    mapped[sub] = v
conn.load_weights(list(mapped.items()), strict=False)
if registers is not None:
    conn.learnable_registers = registers
mx.eval(conn.parameters())

# Deterministic left-padded input + additive mask.
features = mx.random.normal((1, SEQ, DIM)).astype(mx.float32)
mask01 = mx.concatenate(
    [mx.zeros((1, SEQ - NUM_VALID), dtype=mx.int32), mx.ones((1, NUM_VALID), dtype=mx.int32)],
    axis=1,
)
additive = (mask01.astype(mx.float32) - 1.0).reshape(1, 1, 1, SEQ) * 1e9

video_embeddings, _ = conn(features, additive)
mx.eval(video_embeddings)

# --- Audio connector (sc-2684): dim 2048 = 32 heads × 64 head_dim, same 8 layers / 128 regs. ---
AUDIO_DIM, AUDIO_HEADS, AUDIO_HEAD_DIM = 2048, 32, 64
audio_conn = Embeddings1DConnector(
    dim=AUDIO_DIM,
    num_heads=AUDIO_HEADS,
    head_dim=AUDIO_HEAD_DIM,
    num_layers=LAYERS,
    num_learnable_registers=REGISTERS,
    positional_embedding_max_pos=MAX_POS,
    apply_gated_attention=True,
)
aprefix = "audio_embeddings_connector."
amapped, aregisters = {}, None
for k, v in raw.items():
    if not k.startswith(aprefix):
        continue
    sub = k[len(aprefix):]
    v = v.astype(mx.float32)
    if sub == "learnable_registers":
        aregisters = v
        continue
    sub = sub.replace(".ff.net.0.proj.", ".ff.proj_in.")
    sub = sub.replace(".ff.net.2.", ".ff.proj_out.")
    sub = sub.replace(".to_out.0.", ".to_out.")
    amapped[sub] = v
audio_conn.load_weights(list(amapped.items()), strict=False)
if aregisters is not None:
    audio_conn.learnable_registers = aregisters
mx.eval(audio_conn.parameters())

audio_features = mx.random.normal((1, SEQ, AUDIO_DIM)).astype(mx.float32)
audio_embeddings, _ = audio_conn(audio_features, additive)
mx.eval(audio_embeddings)

tensors = {
    "features": features,
    "mask01": mask01.astype(mx.int32),
    "video_embeddings": video_embeddings.astype(mx.float32),
    "audio_features": audio_features,
    "audio_embeddings": audio_embeddings.astype(mx.float32),
}
meta = {"seq": str(SEQ), "num_valid": str(NUM_VALID), "dim": str(DIM), "audio_dim": str(AUDIO_DIM)}
out = fixture("mlx-gen-ltx/tests/fixtures/ltx_connector_golden.safetensors")
Path(out).parent.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(out, tensors, metadata=meta)
print(f"wrote {out}")
print(f"  features {features.shape}  video_embeddings {video_embeddings.shape}")
print(f"  audio_features {audio_features.shape}  audio_embeddings {audio_embeddings.shape}")
