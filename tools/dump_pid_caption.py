#!/usr/bin/env python
"""Dump reference PiD caption embeddings for the mlx-gen-pid caption glue (sc-7843 component 5).

Replicates `pixeldit_model._encode_text_raw` standalone (gemma-2-2b-it decoder + the real Chi-prompt,
in f32 for a tight gate): tokenize `chi_prompt + caption`, run the gemma decoder, select
`[0]+range(-299,0)` → `[1,300,2304]`. Dumps the padded input_ids (discrete tokenizer gate) + the
caption_embs (numeric gate) + num_chi_tokens to a gitignored golden.

Run: /Users/michael/Repos/mlx-gen/_vendor/pid/.venv-pid/bin/python tools/dump_pid_caption.py
"""

import os
import sys

import torch
from safetensors.torch import save_file

PID_ROOT = "/Users/michael/Repos/mlx-gen/_vendor/pid"
sys.path.insert(0, PID_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from pid._src.configs.pid.experiment.shared_config import _CHI_PROMPT  # noqa: E402

MODEL_ID = "Efficient-Large-Model/gemma-2-2b-it"
MODEL_MAX_LENGTH = 300
CAPTION = "a mountain valley landscape at golden hour with a winding river and pine forest"
OUT = "/Users/michael/Repos/mlx-gen/.claude/worktrees/dazzling-gauss-61cef9/tools/golden/pid/caption_landscape.safetensors"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "right"
    # bf16 — the actual PiD inference dtype (the student runs the gemma decoder in bf16), so this is
    # the right reference for the bf16 MLX path. torch bf16 GEMM accumulates in fp32, as does the
    # pmetal NAX bf16 GEMM, so cross-backend agreement is the bf16 rounding floor, not f32-vs-bf16.
    enc = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).get_decoder().eval()

    chi = "\n".join(_CHI_PROMPT)
    num_chi = len(tok.encode(chi))
    max_len = num_chi + MODEL_MAX_LENGTH - 2
    batch = tok([chi + CAPTION], max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        embs = enc(batch.input_ids, batch.attention_mask)[0]  # [1, max_len, 2304]
    sel = [0] + list(range(-MODEL_MAX_LENGTH + 1, 0))
    embs = embs[:, sel]  # [1, 300, 2304]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(
        {
            "caption_embs": embs.contiguous().float(),
            "input_ids": batch.input_ids.to(torch.int32).contiguous(),
        },
        OUT,
        metadata={"num_chi_tokens": str(num_chi), "max_len": str(max_len), "caption": CAPTION},
    )
    print(f"wrote {OUT}")
    print(f"  num_chi_tokens={num_chi}  max_len={max_len}  embs {tuple(embs.shape)} "
          f"mean={embs.mean():.5f} std={embs.std():.5f}")
    print(f"  first 12 input_ids: {batch.input_ids[0, :12].tolist()}")


if __name__ == "__main__":
    main()
