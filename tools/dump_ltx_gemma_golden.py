"""LTX-2.3 Gemma-3-12B text-encoder golden — reference 49 hidden states (sc-2679 S1).

Runs the EXACT reference path: the LTX `LanguageModel` (text_encoder.py) wrapping the mlx_vlm
`Gemma3Model`, loaded from `mlx-community/gemma-3-12b-it-bf16`, with its custom causal+padding mask
and `output_hidden_states=True` → the 49-element hidden-states list that feeds the feature
extractor. Dumps input_ids / attention_mask / all 49 hidden states for a fixed left-padded input.

The Rust `mlx_gen_ltx::gemma` port (tests/gemma_parity.rs) loads the same gemma-3-12b-it-bf16
shards and must reproduce these hidden states.

`LTX_GEMMA_DIR` selects the snapshot (default = the bf16 in the HF cache). Pointing it at a
**quantized** Gemma snapshot (e.g. `mlx-community/gemma-3-12b-it-8bit`) exercises the reference
`utils.apply_quantization` path — the golden then gates the Rust **TE-quant** consumption (sc-2686);
the output is suffixed `_q{bits}` (read from the snapshot's `config.json` `quantization` block).

Loads ~13–24 GB (the Gemma) — slow; run backgrounded:
    ~/Repos/mflux/.venv/bin/python tools/dump_ltx_gemma_golden.py                       # bf16
    LTX_GEMMA_DIR=…/gemma-3-12b-it-8bit ~/Repos/mflux/.venv/bin/python tools/dump_ltx_gemma_golden.py
Output (gitignored, real-weights): tools/golden/ltx_gemma_golden{,_q8,_q4}.safetensors
"""

import glob
import json
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

# __path__ stubs so importing mlx_vlm.models.gemma3.* / mlx_lm.models.base skips the heavy
# mlx_vlm/__init__ + mlx_lm/__init__ chains (the latter pulls `transformers`, absent on the 0.31.2
# env — but `mlx_lm/models/base.py` itself needs only mlx, so a namespace stub imports it directly).
_vlm = Path(VLM) / "mlx_vlm"
_lm = Path(LM) / "mlx_lm"
for name, d in [
    ("mlx_vlm", _vlm),
    ("mlx_vlm.models", _vlm / "models"),
    ("mlx_vlm.models.gemma3", _vlm / "models" / "gemma3"),
    ("mlx_lm", _lm),
    ("mlx_lm.models", _lm / "models"),
]:
    m = types.ModuleType(name)
    m.__path__ = [str(d)]
    sys.modules[name] = m

import mlx.core as mx  # noqa: E402

from mlx_video.models.ltx.text_encoder import LanguageModel  # noqa: E402


def gemma_path() -> str:
    if env := os.environ.get("LTX_GEMMA_DIR"):
        return str(Path(env).expanduser())
    base = Path.home() / ".cache/huggingface/hub/models--mlx-community--gemma-3-12b-it-bf16/snapshots"
    snaps = sorted(glob.glob(str(base / "*")))
    if not snaps:
        raise SystemExit("gemma-3-12b-it-bf16 snapshot not found in HF cache")
    return snaps[-1]


def quant_suffix(gp: str) -> str:
    """`_q{bits}` for a quantized snapshot (config.json `quantization`), else "" (bf16)."""
    cfg = Path(gp) / "config.json"
    if cfg.exists():
        q = json.loads(cfg.read_text()).get("quantization")
        if q and "bits" in q:
            return f"_q{int(q['bits'])}"
    return ""


# Multiple of the connector's 128 learnable registers (the connector tiles registers over seq_len).
MAX_LEN = 128
PROMPT = "A cat playing a grand piano on a city rooftop at sunset."

gp = gemma_path()
lm = LanguageModel.from_pretrained(gp)
mx.eval(lm.parameters())

# `LTX_REUSE_IDS` reuses the (tokenizer-independent) input_ids/attention_mask from a prior golden so
# the dump needs no `transformers` — letting the **quantized** golden run on the mlx-0.31.2 env (the
# Rust build), whose `quantized_matmul` differs from 0.31.0. The bf16 default path still tokenizes.
if reuse := os.environ.get("LTX_REUSE_IDS"):
    prior = mx.load(str(Path(reuse).expanduser()))
    input_ids, attention_mask = prior["input_ids"], prior["attention_mask"]
else:
    from transformers import AutoTokenizer

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

suffix = quant_suffix(gp)
qcfg = {}
_cfg = Path(gp) / "config.json"
if _cfg.exists():
    qcfg = json.loads(_cfg.read_text()).get("quantization") or {}
meta = {
    "max_len": str(MAX_LEN),
    "num_hidden": str(len(all_hidden)),
    "prompt": PROMPT,
    "quant": suffix or "bf16",
    # quant geometry carried so the Rust gate is self-contained (matches the snapshot config.json).
    "bits": str(int(qcfg.get("bits", 0))),
    "group": str(int(qcfg.get("group_size", 0))),
}
out = fixture(f"tools/golden/ltx_gemma_golden{suffix}.safetensors")
Path(out).parent.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(out, tensors, metadata=meta)
print(f"wrote {out}  ({len(tensors)} tensors)")
