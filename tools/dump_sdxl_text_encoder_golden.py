"""SDXL dual-CLIP text-encoder golden — reference for mlx-gen-sdxl S2 (sc-2400).

Runs the EXACT vendored Apple `CLIPTextModel` (`_vendor/mlx_sd/clip.py`) for both SDXL encoders in
**f32** (so the Rust f32 port can be validated to tight tolerance — the production fp16 path's
rounding is absorbed into the e2e px>8 gate later), and dumps the SDXL conditioning exactly as
`StableDiffusionXL._get_text_conditioning`:
  conditioning = concat([te1.hidden_states[-2], te2.hidden_states[-2]], axis=-1)   # [1, N, 2048]
  pooled       = te2.pooled_output                                                 # [1, 1280]

Run from a venv with mlx + regex (the mflux venv):
  /Users/michael/Repos/mflux/.venv/bin/python3 tools/dump_sdxl_text_encoder_golden.py
"""

import json
import os
import sys

import mlx.core as mx

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_DIR = os.path.join(_HERE, "golden")
os.makedirs(_GOLDEN_DIR, exist_ok=True)

# The vendored modules use package-relative imports (`from .config import ...`), so import them as
# the `mlx_sd` package — add its PARENT (`_vendor/`) to sys.path.
VENDOR_PARENT = os.environ.get(
    "SDXL_VENDOR_PARENT",
    "/Users/michael/Repos/SceneWorks/apps/worker/scene_worker/_vendor",
)
sys.path.insert(0, VENDOR_PARENT)
from mlx_sd.clip import CLIPTextModel  # noqa: E402
from mlx_sd.config import CLIPTextModelConfig  # noqa: E402
from mlx_sd.model_io import (  # noqa: E402
    map_clip_text_encoder_weights,
    _load_safetensor_weights,
)
from mlx_sd.tokenizer import Tokenizer  # noqa: E402

PROMPT = os.environ.get("SDXL_PROMPT", "a red fox in a forest")


def _find_snapshot() -> str:
    if os.environ.get("SDXL_SNAPSHOT"):
        return os.environ["SDXL_SNAPSHOT"]
    base = os.path.expanduser(
        "~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots"
    )
    for d in sorted(os.listdir(base)):
        p = os.path.join(base, d)
        if os.path.isdir(p):
            return p
    raise SystemExit("SDXL snapshot not found; set SDXL_SNAPSHOT")


def _load_tokenizer(tok_dir: str) -> Tokenizer:
    with open(os.path.join(tok_dir, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    with open(os.path.join(tok_dir, "merges.txt"), encoding="utf-8") as f:
        merges = f.read().strip().split("\n")[1 : 49152 - 256 - 2 + 1]
    ranks = dict(map(reversed, enumerate([tuple(m.split()) for m in merges])))
    return Tokenizer(ranks, vocab)


def _load_clip(subdir: str, cfg: CLIPTextModelConfig) -> CLIPTextModel:
    model = CLIPTextModel(cfg)
    wf = os.path.join(snap, subdir, "model.safetensors")
    _load_safetensor_weights(map_clip_text_encoder_weights, model, wf, float16=False)
    mx.eval(model.parameters())
    return model


snap = _find_snapshot()
tok = _load_tokenizer(os.path.join(snap, "tokenizer"))

# CFG off (no negative) -> tokens = [[bos, ..., eos]] padded to itself -> [1, N].
tokens = mx.array([tok.tokenize(PROMPT)])

te1 = _load_clip("text_encoder", CLIPTextModelConfig(num_layers=12, model_dims=768, num_heads=12, max_length=77, vocab_size=49408, projection_dim=None, hidden_act="quick_gelu"))
te2 = _load_clip("text_encoder_2", CLIPTextModelConfig(num_layers=32, model_dims=1280, num_heads=20, max_length=77, vocab_size=49408, projection_dim=1280, hidden_act="gelu"))

c1 = te1(tokens)
c2 = te2(tokens)
conditioning = mx.concatenate([c1.hidden_states[-2], c2.hidden_states[-2]], axis=-1)
pooled = c2.pooled_output
mx.eval(conditioning, pooled, c1.hidden_states[-2], c2.hidden_states[-2])

tensors = {
    "input_ids": tokens.astype(mx.int32),
    "te1_hidden_m2": c1.hidden_states[-2].astype(mx.float32),
    "te2_hidden_m2": c2.hidden_states[-2].astype(mx.float32),
    "conditioning": conditioning.astype(mx.float32),
    "pooled": pooled.astype(mx.float32),
}
meta = {"prompt": PROMPT, "seq_len": str(tokens.shape[1])}
out = os.path.join(_GOLDEN_DIR, "sdxl_text_encoder_golden.safetensors")
mx.save_safetensors(out, tensors, meta)
print(f"wrote {out}")
print(f"  prompt {PROMPT!r}, N={tokens.shape[1]}")
print(f"  conditioning {tuple(conditioning.shape)}, pooled {tuple(pooled.shape)}")
