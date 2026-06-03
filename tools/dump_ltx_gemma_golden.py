"""LTX-2.3 Gemma-3-12B text-encoder golden — reference 49 hidden states (sc-2679 S1).

Runs the EXACT reference path: the LTX `LanguageModel` (text_encoder.py) wrapping the mlx_vlm
`Gemma3Model`, loaded from `mlx-community/gemma-3-12b-it-bf16`, with its custom causal+padding mask
and `output_hidden_states=True` → the 49-element hidden-states list that feeds the feature
extractor. Dumps input_ids / attention_mask / all 49 hidden states for a fixed left-padded input.

The Rust `mlx_gen_ltx::gemma` port (tests/gemma_parity.rs) loads the same gemma-3-12b-it-bf16
shards and must reproduce these hidden states.

Loads ~24 GB (the bf16 Gemma) — slow; run backgrounded:
    ~/Repos/mflux/.venv/bin/python tools/dump_ltx_gemma_golden.py
Output (gitignored, real-weights): tools/golden/ltx_gemma_golden.safetensors
"""

import glob
import os
import sys
import types
from pathlib import Path

from _paths import fixture

ARC = str(Path.home() / ".cache/uv/archive-v0/DtG1XO51ABFxUGHg")  # mlx_video
VLM = str(Path.home() / ".cache/uv/archive-v0/69kyKiVsISWokLQN")  # mlx_vlm
LM = str(Path.home() / ".cache/uv/archive-v0/tKxBd9P9nMT7vnfO")  # mlx_lm
for p in (ARC, VLM, LM):
    sys.path.insert(0, p)

# __path__ stubs so importing mlx_vlm.models.gemma3.* skips the heavy mlx_vlm/__init__ chain.
_vlm = Path(VLM) / "mlx_vlm"
for name, d in [
    ("mlx_vlm", _vlm),
    ("mlx_vlm.models", _vlm / "models"),
    ("mlx_vlm.models.gemma3", _vlm / "models" / "gemma3"),
]:
    m = types.ModuleType(name)
    m.__path__ = [str(d)]
    sys.modules[name] = m

import mlx.core as mx  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from mlx_video.models.ltx.text_encoder import LanguageModel  # noqa: E402


def gemma_path() -> str:
    base = Path.home() / ".cache/huggingface/hub/models--mlx-community--gemma-3-12b-it-bf16/snapshots"
    snaps = sorted(glob.glob(str(base / "*")))
    if not snaps:
        raise SystemExit("gemma-3-12b-it-bf16 snapshot not found in HF cache")
    return snaps[-1]


MAX_LEN = 64
PROMPT = "A cat playing a grand piano on a city rooftop at sunset."

gp = gemma_path()
lm = LanguageModel.from_pretrained(gp)
mx.eval(lm.parameters())

tok = AutoTokenizer.from_pretrained(gp, trust_remote_code=True)
tok.padding_side = "left"
enc = tok(PROMPT, return_tensors="np", max_length=MAX_LEN, truncation=True, padding="max_length")
input_ids = mx.array(enc["input_ids"])
attention_mask = mx.array(enc["attention_mask"])

_, all_hidden = lm(
    inputs=input_ids,
    input_embeddings=None,
    attention_mask=attention_mask,
    output_hidden_states=True,
)
print(f"num hidden states: {len(all_hidden)}  each {all_hidden[0].shape} {all_hidden[0].dtype}")

tensors = {
    "input_ids": input_ids.astype(mx.int32),
    "attention_mask": attention_mask.astype(mx.int32),
}
for i, h in enumerate(all_hidden):
    tensors[f"h_{i:02d}"] = h.astype(mx.float32)
mx.eval(list(tensors.values()))

meta = {"max_len": str(MAX_LEN), "num_hidden": str(len(all_hidden)), "prompt": PROMPT}
out = fixture("tools/golden/ltx_gemma_golden.safetensors")
Path(out).parent.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(out, tensors, metadata=meta)
print(f"wrote {out}  ({len(tensors)} tensors)")
