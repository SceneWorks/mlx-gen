#!/usr/bin/env python
"""Dump a tokenizer golden for the Lens text path (mlx-gen sc-3167).

Reproduces `LensPipeline._build_chat_inputs`: wrap each prompt in the gpt-oss **harmony** chat
template (fixed system `_CHAT_SYSTEM` + user prompt + assistant `analysis` thinking), split at
`<|return|>`, and tokenize with `add_special_tokens=True`. Writes the reference `input_ids` per
prompt so the Rust `LensTokenizer` can be checked byte-for-byte.

The 97-token preamble carries `Current date: {today}` (dynamic), so the date used is recorded in
metadata; the Rust side takes the date as a parameter and reproduces the full sequence. The
DiT-relevant conditioning is `input_ids[97:]` (the `txt_offset`), which is date-independent.

Run (from repo root):
  ~/Repos/mflux/.venv/bin/python tools/dump_lens_tokenizer_golden.py
Writes `tools/golden/lens_tokenizer_golden.safetensors` (gitignored).
"""

from __future__ import annotations

import datetime
import glob
import os

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

HOME = os.path.expanduser("~")
TOK_GLOB = f"{HOME}/.cache/huggingface/hub/models--microsoft--Lens-Turbo/snapshots/*/tokenizer"
OUT = os.path.join(os.path.dirname(__file__), "golden", "lens_tokenizer_golden.safetensors")

_CHAT_SYSTEM = (
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background."
)
_CHAT_ASSISTANT_THINKING = "Need to generate one image according to the description."
TXT_OFFSET = 97

PROMPTS = [
    "a red cube on a wooden table",
    "X",
    "A photorealistic portrait of an astronaut riding a horse on Mars, golden hour lighting.",
    "猫が窓辺で眠っている",  # non-ASCII (byte-level BPE coverage)
]


def main() -> None:
    matches = sorted(glob.glob(TOK_GLOB))
    if not matches:
        raise SystemExit(f"no Lens-Turbo tokenizer snapshot at {TOK_GLOB}")
    tok = AutoTokenizer.from_pretrained(matches[-1])

    def render(prompt: str) -> str:
        conversation = [
            {"role": "system", "content": _CHAT_SYSTEM, "thinking": None},
            {"role": "user", "content": prompt, "thinking": None},
            {"role": "assistant", "thinking": _CHAT_ASSISTANT_THINKING, "content": ""},
        ]
        text = tok.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        return text.split("<|return|>")[0]

    tensors: dict[str, torch.Tensor] = {}
    meta = {"n_prompts": str(len(PROMPTS)), "txt_offset": str(TXT_OFFSET)}
    # The current date the harmony preamble embeds (mirrors the template's strftime).
    meta["current_date"] = datetime.date.today().isoformat()

    for i, prompt in enumerate(PROMPTS):
        ids = tok(render(prompt), add_special_tokens=True)["input_ids"]
        tensors[f"ids_{i}"] = torch.tensor(ids, dtype=torch.int32)
        meta[f"prompt_{i}"] = prompt
        print(f"prompt {i}: {len(ids)} tokens (offset-relative {len(ids) - TXT_OFFSET})")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(tensors, OUT, metadata=meta)
    print(f"wrote {OUT}  (current_date={meta['current_date']})")


if __name__ == "__main__":
    main()
