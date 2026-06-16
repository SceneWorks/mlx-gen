"""sc-5988 — dump chat-templated prompt token ids for the Ideogram 4 end-to-end smoke.

Ideogram 4 was trained EXCLUSIVELY on structured JSON captions; a plain-text prompt is
out-of-distribution and yields a coherent but prompt-agnostic image. So the smoke uses a JSON
caption (the model's native format), tokenized with the Qwen3-VL chat template (the pipeline's
`_tokenize`). Native Rust tokenization of the chat template is a follow-up; this isolates the
end-to-end run from tokenizer wiring.

Run:
  ~/mlx-flux-venv/bin/python tools/dump_ideogram4_prompt_ids.py \
      --tokenizer ~/.cache/ideogram4-mlx-convert/tokenizer \
      --out tools/golden/ideogram4_prompt_ids.safetensors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import torch
from transformers import AutoTokenizer

# Keep in lockstep with `CAPTION_JSON` in mlx-gen-ideogram/tests/common/mod.rs: `json.dumps(CAPTION)`
# must be byte-identical to that const (the `tokenizer_parity` test tokenizes the const and asserts
# it reproduces the `input_ids` this script dumps). If you edit the caption, update both.
CAPTION = {
    "high_level_description": "A photograph of a red fox sitting in a snowy forest at golden hour.",
    "style_description": {
        "aesthetics": "serene, warm, naturalistic",
        "lighting": "golden hour, soft warm backlight, long shadows",
        "photo": "telephoto, shallow depth of field, sharp focus, eye-level",
        "medium": "photograph",
    },
    "compositional_deconstruction": {
        "background": "A snowy forest of tall pine trees, soft golden sunlight filtering through the branches, snow on the ground.",
        "elements": [
            {
                "type": "obj",
                "bbox": [250, 320, 950, 760],
                "desc": "A red fox with vivid orange fur, white chest and a thick bushy tail, sitting upright in the snow and facing the camera.",
            }
        ],
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=Path, default=Path.home() / ".cache/ideogram4-mlx-convert/tokenizer")
    ap.add_argument("--out", type=Path, default=Path("tools/golden/ideogram4_prompt_ids.safetensors"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(str(args.tokenizer))
    caption = json.dumps(CAPTION)  # the JSON caption is fed to the model as a string
    messages = [{"role": "user", "content": [{"type": "text", "text": caption}]}]
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    print(f"JSON caption → {len(ids)} tokens")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.out), {"input_ids": mx.array(ids.to(torch.int32).numpy())})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
