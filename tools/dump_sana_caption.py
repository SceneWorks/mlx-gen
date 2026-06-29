#!/usr/bin/env python
"""Dump reference SANA caption embeddings for the mlx-gen-sana text-encoder reuse (sc-8488).

Replicates diffusers `SanaPipeline._get_gemma_prompt_embeds` + `encode_prompt` standalone (the
gemma-2-2b-it decoder + SANA's `complex_human_instruction` CHI prompt): prepend the joined CHI
prompt, tokenize `add_special_tokens=True` + right-pad/truncate to `num_chi + 300 - 2`, run the
gemma decoder, take `last_hidden_state` (`prompt_embeds[0]`), then gather
`select_index = [0] + range(-299, 0)` → `[1, 300, 2304]`. Dumps the padded input_ids (discrete
tokenizer gate) + the caption_embs (numeric gate) + num_chi_tokens to a gitignored golden.

This is byte-for-byte PiD's `_encode_text_raw` EXCEPT the CHI prompt text: SANA's
`complex_human_instruction` wraps `Enhanced prompt` in single-quotes (PiD uses double-quotes), so the
tokenization differs by the quote tokens. The golden therefore differs from the PiD golden.

Run: /Users/michael/Repos/mlx-gen/_vendor/pid/.venv-pid/bin/python tools/dump_sana_caption.py
  (any venv with transformers + a gemma-2-2b-it snapshot works; SANA's CHI list is inlined below so
   no diffusers checkout is required.)
"""

import os

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Efficient-Large-Model/gemma-2-2b-it"
MAX_SEQUENCE_LENGTH = 300
CAPTION = "a mountain valley landscape at golden hour with a winding river and pine forest"
OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools/golden/sana/caption_landscape.safetensors",
)

# SANA's `complex_human_instruction` default list (diffusers pipeline_sana.py / NVlabs Sana).
SANA_CHI = [
    "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "right"
    # bf16 — the SANA inference dtype for the gemma TE; torch bf16 GEMM accumulates in fp32 as the
    # pmetal NAX bf16 GEMM does, so cross-backend agreement is the bf16 rounding floor.
    enc = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    ).get_decoder().eval()

    chi = "\n".join(SANA_CHI)
    num_chi = len(tok.encode(chi))
    max_len = num_chi + MAX_SEQUENCE_LENGTH - 2
    batch = tok(
        [chi + CAPTION],
        max_length=max_len,
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        embs = enc(batch.input_ids, batch.attention_mask)[0]  # last_hidden [1, max_len, 2304]
    sel = [0] + list(range(-MAX_SEQUENCE_LENGTH + 1, 0))
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
    print(
        f"  num_chi_tokens={num_chi}  max_len={max_len}  embs {tuple(embs.shape)} "
        f"mean={embs.mean():.5f} std={embs.std():.5f}"
    )
    print(f"  first 12 input_ids: {batch.input_ids[0, :12].tolist()}")


if __name__ == "__main__":
    main()
