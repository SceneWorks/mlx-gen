"""Real-weights golden for the FLUX.2-**dev** Mistral text encoder (sc-5915), for the #[ignore]d
Rust smoke. Loads the real dev `text_encoder` (a 24B-class `Mistral3ForConditionalGeneration`,
bf16 — the production precision), tokenizes a sample prompt EXACTLY as the dev pipeline does (its
`format_input` + system message + `apply_chat_template`, padded to 512), and dumps
input_ids/attention_mask + the `[10,20,30]` hidden-state concat (`prompt_embeds`, width 15360).

Heavy (~48 GB bf16 to load the language tower); gitignored, regenerable output. Run on the Mac:
    ~/mlx-flux-venv/bin/python ~/Repos/mlx-gen/tools/dump_flux2_te_dev_real_golden.py

Set FLUX2_TE_DEV_F32=1 to cast the encoder to f32 (the Rust port's precision) for a tight gate —
but that ~doubles memory; default is the native bf16 with a generous mean-relative bound.
"""

import os

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoProcessor, Mistral3ForConditionalGeneration

# The dev pipeline's exact prompt formatting (system message + chat template).
from diffusers.pipelines.flux2.pipeline_flux2 import SYSTEM_MESSAGE, format_input

from _paths import fixture, hf_hub_cache

SNAP = str(
    next((hf_hub_cache() / "models--black-forest-labs--FLUX.2-dev" / "snapshots").glob("*"))
)
PROMPT = "a red fox resting in fresh snow under soft winter light"
HIDDEN_LAYERS = (10, 20, 30)
MAX_LEN = 512

f32 = os.environ.get("FLUX2_TE_DEV_F32") == "1"
dtype = torch.float32 if f32 else torch.bfloat16

# Tokenize exactly like `_get_mistral_3_small_prompt_embeds`: format_input(prompt, SYSTEM_MESSAGE)
# → apply_chat_template(add_generation_prompt=False, padding=max_length, max_length=512).
proc = AutoProcessor.from_pretrained(f"{SNAP}/tokenizer")
messages = format_input(prompts=[PROMPT], system_message=SYSTEM_MESSAGE)
inputs = proc.apply_chat_template(
    messages,
    add_generation_prompt=False,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN,
)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

te = Mistral3ForConditionalGeneration.from_pretrained(
    f"{SNAP}/text_encoder", dtype=dtype, low_cpu_mem_usage=True
).eval()

with torch.no_grad():
    out = te(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    stacked = torch.stack([out.hidden_states[k] for k in HIDDEN_LAYERS], dim=1)
    b, n, s, h = stacked.shape
    prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(b, s, n * h)  # (1, 512, 15360)

tens = {
    "input_ids": input_ids.cpu().numpy().astype(np.int32),
    "attention_mask": attention_mask.cpu().numpy().astype(np.int32),
    "prompt_embeds": prompt_embeds.float().cpu().numpy().astype(np.float32),
}
suffix = "_f32" if f32 else ""
path = fixture(f"tools/golden/flux2_te_dev_real{suffix}.safetensors")
save_file(tens, path)
print(f"wrote {path}")
print(f"  prompt_embeds: {tuple(prompt_embeds.shape)}  mask_sum={int(attention_mask.sum())}")
print(f"  first 16 ids: {input_ids[0, :16].tolist()}")
