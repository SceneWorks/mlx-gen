"""SDXL CLIP-tokenizer golden — the reference token ids for the mlx-gen-sdxl tokenizer (sc-2400 S1).

Runs the EXACT vendored Apple tokenizer (`_vendor/mlx_sd/tokenizer.py`) over a fixed prompt set so
the Rust `ClipBpeTokenizer` port can be validated to byte-identical ids. The vendored tokenizer is a
char-level CLIP BPE with dynamic batch-max padding (no 77-token cap) — see the Rust module docstring.

Run from a venv that has `regex` + `mlx` (e.g. the mflux venv):
  /Users/michael/Repos/mflux/.venv/bin/python3 tools/dump_sdxl_tokenizer_golden.py

Env overrides: SDXL_SNAPSHOT (snapshot dir), SDXL_VENDOR (vendored mlx_sd dir).
"""

import json
import os
import sys

import mlx.core as mx

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_DIR = os.path.join(_HERE, "golden")
os.makedirs(_GOLDEN_DIR, exist_ok=True)

VENDOR = os.environ.get(
    "SDXL_VENDOR",
    "/Users/michael/Repos/SceneWorks/apps/worker/scene_worker/_vendor/mlx_sd",
)
sys.path.insert(0, VENDOR)
from tokenizer import Tokenizer  # noqa: E402  (vendored module, regex-only)


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
    # Mirror model_io.load_tokenizer (without hf_hub_download): vocab.json + merges.txt[1:48895].
    with open(os.path.join(tok_dir, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    with open(os.path.join(tok_dir, "merges.txt"), encoding="utf-8") as f:
        bpe_merges = f.read().strip().split("\n")[1 : 49152 - 256 - 2 + 1]
    bpe_merges = [tuple(m.split()) for m in bpe_merges]
    bpe_ranks = dict(map(reversed, enumerate(bpe_merges)))
    return Tokenizer(bpe_ranks, vocab)


snap = _find_snapshot()
tok = _load_tokenizer(os.path.join(snap, "tokenizer"))

PROMPTS = [
    "a fox",
    "a red fox, 1024 pixels!",
    "A photo of an astronaut riding a horse on the moon.",
    "cinematic, 8k,   ultra-detailed",
]
NEGATIVE = "blurry, low quality"

tensors = {}
meta = {"n_prompts": str(len(PROMPTS)), "negative": NEGATIVE}
for i, p in enumerate(PROMPTS):
    ids = tok.tokenize(p)  # prepend_bos + append_eos defaults
    tensors[f"ids_{i}"] = mx.array(ids, dtype=mx.int32)
    meta[f"prompt_{i}"] = p
    print(f"[{i}] {p!r} -> {len(ids)} ids: {ids[:12]}{'...' if len(ids) > 12 else ''}")

# The CFG batch case (vendored StableDiffusion._tokenize): [prompt, negative] padded with 0 to max.
batch = [tok.tokenize(PROMPTS[0]), tok.tokenize(NEGATIVE)]
N = max(len(t) for t in batch)
batch = [t + [0] * (N - len(t)) for t in batch]
tensors["batch_prompt_neg"] = mx.array(batch, dtype=mx.int32)
meta["batch_prompt"] = PROMPTS[0]

out = os.path.join(_GOLDEN_DIR, "sdxl_tokenizer_golden.safetensors")
mx.save_safetensors(out, tensors, meta)
print(f"\nwrote {out}: {len(PROMPTS)} prompts + batch {tuple(tensors['batch_prompt_neg'].shape)}")
