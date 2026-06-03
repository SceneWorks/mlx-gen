"""Real-weights golden for the FLUX.2 Qwen3 text encoder (sc-2346 S1), for the #[ignore]d Rust
smoke test. Loads the real 9b `text_encoder` shards (bf16, the fork's production precision),
tokenizes a sample prompt with FLUX.2's chat template, and dumps input_ids + prompt_embeds.

Gitignored output (large, regenerable). Run from the mflux fork venv:
    cd ~/repos/mflux && .venv/bin/python ~/repos/mlx-gen/tools/dump_flux2_te_real_golden.py
"""

import glob

import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten
from transformers import AutoTokenizer

from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder

from _paths import fixture, hf_hub_cache

SNAP = str(
    next(
        (hf_hub_cache() / "models--black-forest-labs--FLUX.2-klein-9b" / "snapshots").glob("*")
    )
)
PROMPT = "a red fox resting in fresh snow under soft winter light"

# Tokenize with the FLUX.2 chat template (enable_thinking=False), padded to 512.
tok = AutoTokenizer.from_pretrained(f"{SNAP}/tokenizer")
text = tok.apply_chat_template(
    [{"role": "user", "content": PROMPT}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
enc = tok(
    [text],
    padding="max_length",
    max_length=512,
    truncation=True,
    add_special_tokens=True,
    return_tensors="np",
)
input_ids = mx.array(enc["input_ids"].astype(np.int32))
attention_mask = mx.array(enc["attention_mask"].astype(np.int32))

# Load the real TE (bf16) — strip the `model.` prefix, leave the rotary inv_freq buffer.
te = Qwen3TextEncoder(
    vocab_size=151936,
    hidden_size=4096,
    num_hidden_layers=36,
    num_attention_heads=32,
    num_key_value_heads=8,
    intermediate_size=12288,
    head_dim=128,
    rope_theta=1_000_000.0,
    rms_norm_eps=1e-6,
    attention_bias=False,
)
# FLUX2_TE_F32=1 casts the encoder to f32 (the Rust port's precision) to isolate port
# correctness from the fork's production bf16; default is the fork's native bf16.
import os

f32 = os.environ.get("FLUX2_TE_F32") == "1"
params = {}
for f in sorted(glob.glob(f"{SNAP}/text_encoder/*.safetensors")):
    for k, v in mx.load(f).items():
        if k.startswith("model."):
            params[k[len("model.") :]] = v.astype(mx.float32) if f32 else v
te.update(tree_unflatten(list(params.items())))

emb = te.get_prompt_embeds(input_ids, attention_mask, hidden_state_layers=(9, 18, 27))
mx.eval(emb)

out = {
    "input_ids": input_ids.astype(mx.int32),
    "attention_mask": attention_mask.astype(mx.int32),
    "prompt_embeds": emb.astype(mx.float32),
}
suffix = "_f32" if f32 else ""
path = fixture(f"tools/golden/flux2_te_real{suffix}.safetensors")
mx.save_safetensors(path, out)
print(f"wrote {path}")
print(f"  prompt_embeds: {tuple(emb.shape)}  mask_sum={int(attention_mask.sum())}")
print(f"  first 16 ids: {input_ids[0, :16].tolist()}")
