#!/usr/bin/env python3
"""Dump S1 parity fixtures from the `mlx_video` Wan reference: UMT5-XXL prompt embeddings, plus the
cleaned text + token ids, for the Rust port to gate against.

Run with the SceneWorks venv that has `mlx_video` + torch + ftfy + transformers installed:

    "$HOME/Library/Application Support/SceneWorks/python/venv/bin/python" \
        mlx-gen-wan/tools/dump_s1_fixtures.py

Side effects (idempotent): builds the converted 5B snapshot dir (default
`~/Library/Application Support/SceneWorks/data/models/mlx/wan_2_2_ti2v_5b`, override `WAN_5B_DIR`)
containing `t5_encoder.safetensors` (bf16, from the original `.pth`), `config.json`, and a copy of
the `google/umt5-xxl` `tokenizer.json` — the same files the Rust heavy parity test loads.

Writes (committed; small):
  - mlx-gen-wan/tests/fixtures/s1.json               (prompts, cleaned text, ids, seq_len)
  - mlx-gen-wan/tests/fixtures/s1_t5_golden.safetensors   (reference embeds per prompt, f32)
"""
import glob
import json
import os
import shutil

import mlx.core as mx
from transformers import AutoTokenizer

from mlx_video.convert_wan import load_torch_weights, sanitize_wan_t5_weights
from mlx_video.models.wan.config import WanModelConfig
from mlx_video.models.wan.loading import _clean_text, encode_text, load_t5_encoder

HOME = os.path.expanduser("~")
HF = os.path.join(HOME, ".cache/huggingface/hub")
OUT_DIR = os.environ.get(
    "WAN_5B_DIR",
    os.path.join(HOME, "Library/Application Support/SceneWorks/data/models/mlx/wan_2_2_ti2v_5b"),
)

# Prompts: an English prompt, a second English prompt, and the full Chinese negative prompt — the
# load-bearing case for `_clean_text` (fullwidth commas → ASCII) + a long (non-trivial) sequence.
PROMPTS = {
    "english": "a cat playing the piano",
    "scene": "a serene mountain lake at sunrise, photorealistic, 8k",
    "neg": WanModelConfig.wan22_ti2v_5b().sample_neg_prompt,
}


def _resolve(pattern: str) -> str:
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[-1]


def ensure_snapshot(config) -> None:
    """Convert the T5 `.pth` → `t5_encoder.safetensors` and populate the snapshot dir (idempotent)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    t5_out = os.path.join(OUT_DIR, "t5_encoder.safetensors")
    if not os.path.exists(t5_out):
        pth = _resolve(
            os.path.join(HF, "models--Wan-AI--Wan2.2-TI2V-5B/snapshots/*/models_t5_umt5-xxl-enc-bf16.pth")
        )
        print(f"Converting T5 encoder from {pth} ...")
        weights = sanitize_wan_t5_weights(load_torch_weights(pth))
        weights = {k: v.astype(mx.bfloat16) for k, v in weights.items()}
        mx.save_safetensors(t5_out, weights)
        print(f"  wrote {len(weights)} tensors → {t5_out}")
    # config.json (the 5B serialized config) + tokenizer.json (google/umt5-xxl).
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    tok_dst = os.path.join(OUT_DIR, "tokenizer.json")
    if not os.path.exists(tok_dst):
        tok_src = _resolve(os.path.join(HF, "models--google--umt5-xxl/snapshots/*/tokenizer.json"))
        shutil.copyfile(tok_src, tok_dst)
        print(f"  copied tokenizer.json from {tok_src}")


def main():
    config = WanModelConfig.wan22_ti2v_5b()
    ensure_snapshot(config)

    encoder = load_t5_encoder(os.path.join(OUT_DIR, "t5_encoder.safetensors"), config)
    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")

    meta = {"text_len": config.text_len, "prompts": {}}
    golden = {}
    for name, prompt in PROMPTS.items():
        cleaned = _clean_text(prompt)
        tok = tokenizer(
            cleaned, max_length=config.text_len, padding="max_length",
            truncation=True, return_tensors="np",
        )
        mask = tok["attention_mask"][0]
        seq_len = int(mask.sum())
        ids = tok["input_ids"][0][:seq_len].tolist()
        embeds = encode_text(encoder, tokenizer, prompt, config.text_len)  # [seq_len, dim] f32
        mx.eval(embeds)
        golden[f"embeds_{name}"] = embeds.astype(mx.float32)

        # Block-0 output (one prompt) — a durable "per-op math is bit-exact" gate. The small
        # end-to-end gap vs the reference is cross-build f32 accumulation over 24 layers (MLX 0.31.2
        # wheel vs pmetal-0.31.1 source); block-0 must match to GEMM noise, isolating a real math
        # regression from harmless accumulation drift.
        if name == "english":
            ids_full = mx.array(tok["input_ids"])
            mask_full = mx.array(tok["attention_mask"])
            h = encoder.token_embedding(ids_full)
            h = encoder.blocks[0](h, mask=mask_full, pos_bias=None)
            mx.eval(h)
            golden["block0_english"] = h[0, :seq_len].astype(mx.float32)
        meta["prompts"][name] = {
            "prompt": prompt,
            "cleaned": cleaned,
            "ids": ids,
            "seq_len": seq_len,
            "embed_shape": list(embeds.shape),
        }
        print(f"  [{name}] seq_len={seq_len} embed_shape={list(embeds.shape)}")

    dst = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
    os.makedirs(dst, exist_ok=True)
    mx.save_safetensors(os.path.join(dst, "s1_t5_golden.safetensors"), golden)
    with open(os.path.join(dst, "s1.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"wrote {os.path.abspath(os.path.join(dst, 's1.json'))}")
    print(f"wrote {os.path.abspath(os.path.join(dst, 's1_t5_golden.safetensors'))}")


if __name__ == "__main__":
    main()
