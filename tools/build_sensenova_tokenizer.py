"""sc-3186: materialize the SenseNova-U1 fast `tokenizer.json` into the HF snapshot + dump golden
tokenizations and `neo1_0` conversation strings for the Rust parity test.

The snapshot ships only vocab.json + merges.txt + added_tokens.json (no fast `tokenizer.json`), so —
mirroring `tools/build_qwen_tokenizer.py` — this writes `tokenizer.json` next to them via
`AutoTokenizer`. It also dumps golden `(string -> input_ids)` pairs and the reference `neo1_0` T2I
query strings (built via the vendored `conversation.py`) so the Rust template + tokenizer pipeline
can be validated.

Run: cd _vendor/sensenova_u1 && PYTHONPATH=src .venv/bin/python ../../tools/build_sensenova_tokenizer.py
"""

from __future__ import annotations

import json
import os
import sys

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from sensenova_u1.models.neo_unify.conversation import get_conv_template
from sensenova_u1.models.neo_unify.utils import SYSTEM_MESSAGE_FOR_GEN


def hf_snapshot() -> str:
    cache = os.environ.get(
        "SENSENOVA_U1_SNAPSHOT",
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--sensenova--SenseNova-U1-8B-MoT/snapshots"
        ),
    )
    if os.path.isdir(cache) and "snapshots" in cache:
        snaps = [os.path.join(cache, d) for d in os.listdir(cache)]
        cache = next(d for d in snaps if os.path.isdir(d))
    return cache


def neo1_0_query(prompt: str, system_message: str) -> str:
    conv = get_conv_template("neo1_0")
    conv.system_message = system_message
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def main() -> None:
    snap = hf_snapshot()
    tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
    # Materialize the fast tokenizer.json into the snapshot (idempotent).
    out_json = os.path.join(snap, "tokenizer.json")
    tok.backend_tokenizer.save(out_json)
    print(f"wrote {out_json}")

    prompt = "a red fox sitting in a snowy forest, photorealistic"
    q_gen = neo1_0_query(prompt, SYSTEM_MESSAGE_FOR_GEN)
    q_empty = neo1_0_query(prompt, "")
    strings = {
        "plain": prompt,
        "query_gen": q_gen,
        "query_empty": q_empty,
        "specials": "<|im_start|>user\n<img></img><think></think><|im_end|>\n",
    }

    tensors = {}
    meta = {}
    for name, s in strings.items():
        ids = tok(s, return_tensors=None)["input_ids"]
        tensors[f"ids.{name}"] = torch.tensor(ids, dtype=torch.int32)
        meta[f"str.{name}"] = s

    dst = os.path.join(
        REPO_ROOT, "mlx-gen-sensenova", "tests", "fixtures", "tokenizer_golden.safetensors"
    )
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    save_file(tensors, dst, metadata=meta)
    print(f"wrote {dst}")
    for name in strings:
        print(f"  ids.{name}: {len(tensors[f'ids.{name}'])} tokens")
    # Emit the query strings so the Rust template can be checked against them verbatim.
    print("query_gen repr:")
    print(json.dumps(q_gen))


if __name__ == "__main__":
    sys.exit(main())
