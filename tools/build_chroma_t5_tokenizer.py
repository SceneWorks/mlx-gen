"""Materialize the Chroma T5 tokenizer.json (epic 3531).

Chroma's `tokenizer/` ships only the sentencepiece `spiece.model` (+ configs) — no `tokenizer.json`
that the Rust core `TextTokenizer` (HF `tokenizers`) can load directly. This is the same google
t5-v1.1-xxl tokenizer FLUX uses. We convert it once with `transformers` and vendor the result as
`mlx-gen-chroma/assets/t5_tokenizer.json` (compiled into the crate, like flux's clip_tokenizer.json
and the sensenova tokenizer), so loading needs no network and no slow-tokenizer conversion at runtime.

Run from the SceneWorks torch venv:
    "/Users/michael/Library/Application Support/SceneWorks/python/venv/bin/python" \
        tools/build_chroma_t5_tokenizer.py
"""

from __future__ import annotations

import os

from transformers import T5TokenizerFast

from _paths import fixture, hf_hub_cache


def chroma_tokenizer_dir() -> str:
    base = hf_hub_cache() / "models--lodestones--Chroma1-HD" / "snapshots"
    snap = next(p for p in base.iterdir() if p.is_dir())
    return str(snap / "tokenizer")


def main() -> None:
    tok = T5TokenizerFast.from_pretrained(chroma_tokenizer_dir())
    out = fixture("mlx-gen-chroma/assets/t5_tokenizer.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tok.backend_tokenizer.save(out)
    print(f"class={type(tok).__name__} vocab={tok.vocab_size} pad={tok.pad_token_id} "
          f"eos={tok.eos_token_id}")
    print(f"wrote {out} ({os.path.getsize(out)} bytes)")


if __name__ == "__main__":
    main()
