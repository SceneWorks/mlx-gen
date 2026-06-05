"""Dump a Qwen-Image-Edit VL-tokenization parity golden for the Rust port (sc-2465, slice 6b-2).

Image-free + weight-free: take the fork's edit chat template (`use_picture_prefix=False`), expand
the single `<|image_pad|>` to a fixed count exactly as `QwenVisionLanguageProcessor` does, tokenize
with the snapshot's HF tokenizer, and dump the `input_ids`. The Rust test reconstructs the same
formatted text (`vl_tokenizer::build_edit_text`) and tokenizes via the materialized `tokenizer.json`,
asserting byte-exact `input_ids` — verifying the template string + the special-token mapping.

Prereq: the Edit snapshot ships only vocab.json + merges.txt, so materialize the fast tokenizer once:
    QWEN_IMAGE_SNAPSHOT=<edit-snapshot-dir> uv run python tools/build_qwen_tokenizer.py
Run (fork venv):
    cd ~/repos/mflux && uv run python ~/Repos/mlx-gen/.claude/worktrees/musing-mclaren-676094/tools/dump_qwen_vl_tokenize_golden.py
Output (gitignored): tools/golden/qwen_vl_tokenize_golden.safetensors
"""

import glob
import os

import mlx.core as mx
from transformers import AutoTokenizer

from mflux.models.qwen.tokenizer.qwen_vision_language_tokenizer import QwenVisionLanguageTokenizer

# Fixed test inputs — MUST match the Rust test constants.
PROMPT = "make the sky purple at sunset"
N_IMAGE_TOKENS = 36  # prod(grid)//4 for a (1,12,12) grid

snap = sorted(
    p
    for p in glob.glob(
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/*")
    )
    if os.path.isdir(p)
)[0]
tok_dir = os.path.join(snap, "tokenizer")

# Fork's edit template (use_picture_prefix=False); processor expands the single <|image_pad|>.
vlt = QwenVisionLanguageTokenizer(processor=None, use_picture_prefix=False)
formatted = vlt.edit_template.format(PROMPT)
expanded = formatted.replace("<|image_pad|>", "<|placeholder|>" * N_IMAGE_TOKENS, 1).replace(
    "<|placeholder|>", "<|image_pad|>"
)

tok = AutoTokenizer.from_pretrained(tok_dir)
ids = tok(expanded)["input_ids"]
input_ids = mx.array([ids], dtype=mx.int32)

golden_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(golden_dir, exist_ok=True)
path_out = os.path.join(golden_dir, "qwen_vl_tokenize_golden.safetensors")
mx.save_safetensors(path_out, {"input_ids": input_ids})
print(f"prompt={PROMPT!r} n_image={N_IMAGE_TOKENS} -> input_ids {input_ids.shape}")
print(f"wrote {path_out}")
