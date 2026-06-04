#!/usr/bin/env python3
"""Dump the Wan LoRA **key-normalization** parity fixture (sc-2683): the reference
`_normalize_wan_lora_key` applied to every real Wan2.2 MoE LoRA module stem plus synthetic global /
alternate-prefix spellings, resolved against the actual converted A14B weight-key set. The Rust
`normalize_wan_key` (mlx-gen-wan/src/adapters.rs) must reproduce each mapping exactly — this is the
cheap CI gate (just strings; no weights at test time) that the Wan key→module map matches the
reference's, the load-bearing piece of the merge.

The committed fixture `tests/fixtures/wan_lora_keys.json` is `{ "<raw stem>": "<normalized path>" }`.

Run with the SceneWorks venv (which has `mlx_video`), pointing at the converted T2V-A14B dir + a real
Wan MoE LoRA (for the realistic block/attn/ffn stems):

    WAN_A14B_MODEL_DIR=~/.cache/mlx-gen-models/wan2_2_t2v_a14b_mlx_bf16 \
    WAN_LORA_FILE="~/Library/Application Support/SceneWorks/data/loras/lauren_high/lauren_wan22_high_epoch_95.safetensors" \
    "$HOME/Library/Application Support/SceneWorks/python/venv/bin/python" \
        mlx-gen-wan/tools/dump_lora_fixtures.py

Only the safetensors *headers* are read (`safe_open(...).keys()`), so the 27 GB expert is never
materialized — the model-key set is recovered from the header alone.
"""
import json
import os
import re
from pathlib import Path

from safetensors import safe_open

from mlx_video.lora.apply import _normalize_wan_lora_key

HERE = Path(__file__).resolve().parent
FIXTURE = HERE.parent / "tests" / "fixtures" / "wan_lora_keys.json"

# Synthetic global / alternate-prefix stems exercising every rename branch (the real LoRA only
# touches block attn/ffn, so the global renames need explicit coverage).
SYNTHETIC = [
    "diffusion_model.text_embedding.0",
    "diffusion_model.text_embedding.2",
    "diffusion_model.time_embedding.0",
    "diffusion_model.time_embedding.2",
    "diffusion_model.time_projection.1",
    "diffusion_model.patch_embedding",
    "model.diffusion_model.blocks.0.self_attn.q",
    "base_model.model.blocks.3.ffn.0",
    "model.blocks.7.cross_attn.o",
    "blocks.11.ffn.2",
]


def main():
    model_dir = Path(os.path.expanduser(os.environ["WAN_A14B_MODEL_DIR"]))
    lora_file = Path(os.path.expanduser(os.environ["WAN_LORA_FILE"]))

    # Model weight-key set (headers only — no tensor data loaded).
    with safe_open(model_dir / "low_noise_model.safetensors", framework="numpy") as f:
        model_keys = set(f.keys())
    print(f"model keys: {len(model_keys)}")

    # Real LoRA module stems (drop the .lora_A/B/.alpha suffix).
    stems = set()
    with safe_open(lora_file, framework="numpy") as f:
        for k in f.keys():
            m = re.match(r"(.+)\.(lora_A|lora_B|lora_down|lora_up|alpha)(\.weight)?$", k)
            if m:
                stems.add(m.group(1))
    print(f"real LoRA stems: {len(stems)}")

    stems = sorted(stems) + SYNTHETIC
    mapping = {stem: _normalize_wan_lora_key(stem, model_keys) for stem in stems}

    # Sanity: every real-LoRA stem must normalize to a key present in the model (else the merge would
    # skip it — the reference would too, but the real files target only existing modules).
    missing = [
        s for s in stems if not s.startswith(("diffusion_model.time", "diffusion_model.text"))
        and f"{mapping[s]}.weight" not in model_keys and "patch_embedding" not in s
    ]
    if missing:
        print(f"WARNING: {len(missing)} stems normalized to a non-model key, e.g. {missing[:3]}")

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with open(FIXTURE, "w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    print(f"wrote {FIXTURE}  ({len(mapping)} entries)")


if __name__ == "__main__":
    main()
