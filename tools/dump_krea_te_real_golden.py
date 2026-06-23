"""Real-weight Krea 2 text-encoder parity golden (sc-7569, the `#[ignore]` gate).

Loads the published `krea/Krea-2-Turbo` `text_encoder/` (Qwen3-VL-4B) text tower into the transformers
`Qwen3VLTextModel`, renders the Krea prompt template (prefix + user + assistant cue) with the snapshot
tokenizer, runs text-only, and dumps the stacked select-layer conditioning `[1, L-prefix, 12, 2560]`
plus the `input_ids` the Rust `#[ignore]` test feeds verbatim (and checks its own tokenizer against).

Text-only → standard RoPE (the interleaved MRoPE sections all index the same sequential text position
with no image tokens), so the standalone text tower reproduces the full model's text conditioning.

    KREA_TURBO_DIR=~/.cache/huggingface/hub/models--krea--Krea-2-Turbo/snapshots/<rev> \
    KREA_DEVICE=cpu KREA_DTYPE=f32 \
      ~/Repos/mflux/.venv/bin/python tools/dump_krea_te_real_golden.py
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

from _paths import fixture

# Must match `mlx-gen-krea/src/text_encoder/tokenizer.rs`.
PREFIX = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PREFIX_TOKENS = 34
SELECT = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
PROMPT = "A medium-shot photograph of a red fox sitting in a snowy forest at golden hour."


def load_text_tower_state(te_dir: Path) -> dict:
    """Read the `text_encoder/` shards, keep the `language_model.*` text tower, strip the prefix."""
    from safetensors.torch import load_file

    idx = glob.glob(str(te_dir / "*.index.json"))
    if idx:
        shards = sorted(set(json.loads(Path(idx[0]).read_text())["weight_map"].values()))
    else:
        shards = [os.path.basename(p) for p in glob.glob(str(te_dir / "*.safetensors"))]
    sd = {}
    for shard in shards:
        sd.update(load_file(str(te_dir / shard)))
    return {
        k[len("language_model."):]: v
        for k, v in sd.items()
        if k.startswith("language_model.")
    }


@torch.no_grad()
def main():
    root = Path(os.environ["KREA_TURBO_DIR"])
    device = os.environ.get("KREA_DEVICE", "cpu")
    dtype = {"bf16": torch.bfloat16, "f32": torch.float32}[os.environ.get("KREA_DTYPE", "f32")]
    te_dir = root / "text_encoder"

    cfg = AutoConfig.from_pretrained(str(te_dir)).text_config
    # Build on the real device (NOT meta) so non-persistent buffers — the rotary `inv_freq` — are
    # materialized by __init__ (they aren't in the state dict), then copy the weights in.
    model = Qwen3VLTextModel(cfg)
    model.load_state_dict(load_text_tower_state(te_dir), strict=True)
    model = model.to(device=device, dtype=dtype).eval()

    tok = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
    text = PREFIX + PROMPT + SUFFIX
    enc = tok(text, add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = torch.ones_like(input_ids)
    prefix_len = len(tok(PREFIX, add_special_tokens=False)["input_ids"])
    print(f"seq_len={input_ids.shape[1]}  prefix_len={prefix_len} (expected {PREFIX_TOKENS})")

    out = model(
        input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
    )
    hiddens = torch.stack([out.hidden_states[i] for i in SELECT], dim=2)  # [1, L, 12, 2560]
    hiddens = hiddens[:, PREFIX_TOKENS:]

    from safetensors.torch import save_file

    tensors = {
        "in.input_ids": input_ids.to(torch.int32).cpu(),
        "in.attention_mask": attention_mask.to(torch.int32).cpu(),
        "out.hiddens": hiddens.to(torch.float32).cpu().contiguous(),
    }
    path = fixture("tools/golden/krea_te_real.safetensors")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path)
    print(f"wrote {path}  (device={device} dtype={dtype}, hiddens {tuple(hiddens.shape)})")


if __name__ == "__main__":
    main()
