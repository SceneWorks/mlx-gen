"""Kolors ChatGLM3-6B text-encoder golden — reference hidden states (sc-3091).

Runs the EXACT diffusers reference path: `diffusers.pipelines.kolors.text_encoder.ChatGLMModel`
(the `KolorsPipeline` text encoder) loaded from the `Kwai-Kolors/Kolors-diffusers` HF snapshot's
`text_encoder/` (fp16 shards), with `output_hidden_states=True` — i.e. exactly what
`KolorsPipeline.encode_prompt` consumes (`hidden_states[-2]` = context, `hidden_states[-1]` last
token = pooled).

For two fixed inputs — `packed` (no padding → pure causal) and `padded` (right-padded → the
causal+padding `get_masks` path) — dumps input_ids / attention_mask / all 29 hidden states
(permuted `[S,B,H]→[B,S,H]` to match the Rust layout) / context / pooled, for BOTH **f32** (near-bit
parity) and **fp16** (the production-dtype floor, prefixed `f16_`).

The Rust `mlx_gen_kolors::chatglm3::ChatGlmModel` (tests/chatglm_parity.rs) loads the SAME fp16
shards and must reproduce these.

Loads ~12.5 GB fp16 (cast to f32 ≈ 25 GB) — run backgrounded:
    ~/repos/mflux/.venv-0312/bin/python tools/dump_kolors_chatglm_golden.py
Output (gitignored, real-weights): tools/golden/kolors_chatglm_golden.safetensors
"""

import glob
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from safetensors.torch import load_file

from _paths import fixture, hf_hub_cache

from diffusers.pipelines.kolors.text_encoder import ChatGLMConfig, ChatGLMModel


def te_dir() -> Path:
    base = hf_hub_cache() / "models--Kwai-Kolors--Kolors-diffusers" / "snapshots"
    snaps = sorted(glob.glob(str(base / "*")))
    if not snaps:
        raise SystemExit("Kolors-diffusers snapshot not found in HF cache")
    return Path(snaps[-1]) / "text_encoder"


def load_model(te: Path) -> ChatGLMModel:
    cfg = ChatGLMConfig(**json.loads((te / "config.json").read_text()))
    model = ChatGLMModel(cfg)
    state = {}
    for shard in sorted(glob.glob(str(te / "*.safetensors"))):
        state.update(load_file(shard))
    missing, unexpected = model.load_state_dict(state, strict=False)
    # output_layer (LM head) is unused by the encoder path; everything else must be present.
    assert not missing, f"missing weights: {missing[:8]}"
    model.eval()
    return model


# Fixed, deterministic valid token ids (in [0, 65024)); 24 "real" tokens. No tokenizer needed —
# the encoder gate validates the forward given (ids, mask); tokenizer parity is sc-3092/3094.
REAL_IDS = [
    64790, 64792, 790, 30951, 517, 30910, 30939, 30996, 13, 280,
    260, 312, 1773, 750, 30910, 30943, 31010, 280, 260, 1336,
    295, 30994, 13, 30910,
]
PAD_ID = 0


@torch.no_grad()
def run_case(model: ChatGLMModel, input_ids, attention_mask):
    ids = torch.tensor([input_ids], dtype=torch.long)
    mask = torch.tensor([attention_mask], dtype=torch.long)
    out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True, return_dict=True)
    # all_hidden_states: tuple of [S, B, H] → permute to [B, S, H].
    hiddens = [h.permute(1, 0, 2).contiguous() for h in out.hidden_states]
    context = out.hidden_states[-2].permute(1, 0, 2).contiguous()  # [B, S, H]
    pooled = out.hidden_states[-1][-1, :, :].contiguous()  # last seq position → [B, H]
    return hiddens, context, pooled


def to_mx(t):
    return mx.array(t.float().cpu().numpy().astype(np.float32))


def main():
    te = te_dir()
    model = load_model(te)

    cases = {
        "packed": (REAL_IDS, [1] * len(REAL_IDS)),
        "padded": (REAL_IDS + [PAD_ID] * 8, [1] * len(REAL_IDS) + [0] * 8),
    }

    tensors = {}
    num_hidden = None
    for dtype_name, cast in [("", lambda m: m.float()), ("f16_", lambda m: m.half())]:
        cast(model)
        for case, (ids, mask) in cases.items():
            hiddens, context, pooled = run_case(model, ids, mask)
            num_hidden = len(hiddens)
            p = f"{dtype_name}{case}_"
            tensors[f"{p}input_ids"] = mx.array(np.array([ids], dtype=np.int32))
            tensors[f"{p}attention_mask"] = mx.array(np.array([mask], dtype=np.int32))
            for i, h in enumerate(hiddens):
                tensors[f"{p}h_{i:02d}"] = to_mx(h)
            tensors[f"{p}context"] = to_mx(context)
            tensors[f"{p}pooled"] = to_mx(pooled)
            print(f"{p}: {num_hidden} hidden, context {tuple(context.shape)} pooled {tuple(pooled.shape)}")

    mx.eval(list(tensors.values()))
    meta = {"num_hidden": str(num_hidden), "cases": "packed,padded", "dtypes": "f32,f16"}
    out = fixture("tools/golden/kolors_chatglm_golden.safetensors")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(out, tensors, metadata=meta)
    print(f"wrote {out}  ({len(tensors)} tensors)")


if __name__ == "__main__":
    main()
