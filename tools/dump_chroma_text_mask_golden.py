"""Parity fixture for the Chroma text-mask construction (epic 3531, sc-3838).

Reproduces `ChromaPipeline._get_t5_prompt_embeds`'s mask arithmetic with the **tokenizer only** (no
T5 weights needed): the standard tokenizer padding mask, and the transformer `attention_mask` with
Chroma's keep-one-extra-pad quirk `(arange(L) <= seq_lengths)`. Also dumps `input_ids` so the Rust
port can confirm its vendored tokenizer encodes identically. The T5 *numeric* masked-encode parity
is validated in the e2e (sc-3839), where the real T5 weights load.

Run from the SceneWorks torch venv:
    "/Users/michael/Library/Application Support/SceneWorks/python/venv/bin/python" \
        tools/dump_chroma_text_mask_golden.py
"""

from __future__ import annotations

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import T5TokenizerFast

from _paths import fixture, hf_hub_cache

MAX_LEN = 64  # small fixture; the mask logic is length-agnostic (production uses 512)
PROMPTS = ["a photograph of an astronaut riding a horse", "a cat"]


def tokenizer_dir() -> str:
    base = hf_hub_cache() / "models--lodestones--Chroma1-HD" / "snapshots"
    snap = next(p for p in base.iterdir() if p.is_dir())
    return str(snap / "tokenizer")


def main() -> None:
    tok = T5TokenizerFast.from_pretrained(tokenizer_dir())
    out = {}
    for i, prompt in enumerate(PROMPTS):
        ti = tok(prompt, padding="max_length", max_length=MAX_LEN, truncation=True,
                 add_special_tokens=True, return_tensors="pt")
        input_ids = ti.input_ids
        tokenizer_mask = ti.attention_mask  # standard 0/1 padding mask

        # ChromaPipeline._get_t5_prompt_embeds transformer mask (keep-one-extra-pad):
        seq_lengths = tokenizer_mask.sum(dim=1)
        seq_len = tokenizer_mask.shape[1]
        mask_indices = torch.arange(seq_len).unsqueeze(0).expand(1, -1)
        attention_mask = (mask_indices <= seq_lengths.unsqueeze(1)).to(torch.float32)

        out[f"input_ids_{i}"] = input_ids.to(torch.int32)
        out[f"tokenizer_mask_{i}"] = tokenizer_mask.to(torch.int32)
        out[f"attention_mask_{i}"] = attention_mask
        print(f"[{i}] {prompt!r}: real={int(seq_lengths.item())} "
              f"transformer_ones={int(attention_mask.sum().item())} (keep-one-extra)")

    save_file(out, fixture("mlx-gen-chroma/tests/fixtures/chroma_text_mask.safetensors"))
    print("wrote chroma_text_mask.safetensors", {k: tuple(v.shape) for k, v in out.items()})


if __name__ == "__main__":
    main()
