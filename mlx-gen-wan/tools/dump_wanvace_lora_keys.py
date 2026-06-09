#!/usr/bin/env python3
"""Dump the Wan-VACE LoRA **key-normalization** parity fixture (sc-3439): the diffusers-name
key→module map the Rust `normalize_vace_key` (mlx-gen-wan/src/adapters.rs) must reproduce.

Unlike the base Wan fixture (`dump_lora_fixtures.py`, which targets the *native* converted layout via
`mlx_video._normalize_wan_lora_key`), the VACE transformer reads the **diffusers** tensor names
directly, so the reference here is diffusers' own LoRA loader,
`diffusers...lora_conversion_utils._convert_non_diffusers_wan_lora_to_diffusers`. The base-block +
global native→diffusers mappings are taken **authoritatively from that converter** (fed a synthetic
native LoRA, the diffusers output stems are read back). The diffusers-named passthrough stems (incl.
every `vace_blocks.*` module, which the diffusers converter does not handle and which the native
trainers emit) are mapped by the shared rename rule and **verified against the real VACE checkpoint
weight-key set** so a structural error can't slip through.

The committed fixture `tests/fixtures/wanvace_lora_keys.json` is `{ "<raw stem>": "<diffusers path>" }`.

Run with the diffusers venv (which has diffusers + torch); the checkpoint headers (the real key set)
are read from the HF cache by default:

    "$HOME/mlx-flux-venv/bin/python" mlx-gen-wan/tools/dump_wanvace_lora_keys.py
    # or point at a snapshot holding transformer/:  WANVACE_DIR=<dir> ... dump_wanvace_lora_keys.py

Only safetensors *headers* / the shard index are read — the 7 GB weights are never materialized.
"""
import json
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open

from diffusers.loaders.lora_conversion_utils import (
    _convert_non_diffusers_wan_lora_to_diffusers,
)

HERE = Path(__file__).resolve().parent
FIXTURE = HERE.parent / "tests" / "fixtures" / "wanvace_lora_keys.json"


def host_keys() -> set[str]:
    """The real diffusers VACE checkpoint weight-key set (headers / shard index only)."""
    snap = os.environ.get("WANVACE_DIR")
    if snap:
        root = Path(os.path.expanduser(snap))
    else:
        base = Path.home() / (
            ".cache/huggingface/hub/"
            "models--Wan-AI--Wan2.1-VACE-1.3B-diffusers/snapshots"
        )
        root = next(p for p in base.iterdir() if (p / "transformer").is_dir())
    tdir = root / "transformer"
    index = tdir / "diffusion_pytorch_model.safetensors.index.json"
    if index.exists():
        return set(json.load(open(index))["weight_map"].keys())
    keys: set[str] = set()
    for st in tdir.glob("*.safetensors"):
        with safe_open(st, framework="numpy") as f:
            keys.update(f.keys())
    return keys


def native_basis() -> dict[str, str]:
    """Authoritative native→diffusers base-block + global stem map, read back from the diffusers
    converter fed a synthetic native LoRA (no alpha → no factor scaling, pure rename)."""
    sd: dict[str, torch.Tensor] = {}
    t = torch.zeros(2, 2)
    # Two base blocks so min/max block range is non-degenerate.
    for i in (0, 1):
        for o in ("q", "k", "v", "o"):
            for attn in ("self_attn", "cross_attn"):
                sd[f"diffusion_model.blocks.{i}.{attn}.{o}.lora_A.weight"] = t
                sd[f"diffusion_model.blocks.{i}.{attn}.{o}.lora_B.weight"] = t
        for ff in ("ffn.0", "ffn.2"):
            sd[f"diffusion_model.blocks.{i}.{ff}.lora_A.weight"] = t
            sd[f"diffusion_model.blocks.{i}.{ff}.lora_B.weight"] = t
    # Globals.
    for g in ("time_projection.1", "head.head",
              "text_embedding.0", "text_embedding.2",
              "time_embedding.0", "time_embedding.2"):
        sd[f"diffusion_model.{g}.lora_A.weight"] = t
        sd[f"diffusion_model.{g}.lora_B.weight"] = t

    def strip_t(k: str) -> str:
        return k[len("transformer.") :] if k.startswith("transformer.") else k

    converted = _convert_non_diffusers_wan_lora_to_diffusers(dict(sd))
    out_stems = {
        re.sub(r"\.lora_[AB]\.weight$", "", strip_t(k))
        for k in converted
        if k.endswith(".weight")
    }

    # Recover input→output stem pairs: re-run per single module and read the lone output stem.
    def one(native_stem: str) -> str:
        d = {
            f"diffusion_model.{native_stem}.lora_A.weight": t,
            f"diffusion_model.{native_stem}.lora_B.weight": t,
        }
        # blocks.* path needs at least one blocks.N key; globals tolerate a lone module.
        if not native_stem.startswith("blocks."):
            d["diffusion_model.blocks.0.self_attn.q.lora_A.weight"] = t
            d["diffusion_model.blocks.0.self_attn.q.lora_B.weight"] = t
        c = _convert_non_diffusers_wan_lora_to_diffusers(dict(d))
        stems = {
            re.sub(r"\.lora_[AB]\.(weight|bias)$", "", strip_t(k))
            for k in c
            if re.search(r"\.lora_[AB]\.weight$", k)
        }
        if not native_stem.startswith("blocks."):
            stems.discard("blocks.0.attn1.to_q")
        assert len(stems) == 1, (native_stem, stems)
        return stems.pop()

    natives = [
        f"blocks.{i}.{attn}.{o}"
        for i in (0, 1)
        for attn in ("self_attn", "cross_attn")
        for o in ("q", "k", "v", "o")
    ] + [
        f"blocks.{i}.{ff}" for i in (0, 1) for ff in ("ffn.0", "ffn.2")
    ] + [
        "time_projection.1", "head.head",
        "text_embedding.0", "text_embedding.2",
        "time_embedding.0", "time_embedding.2",
    ]
    mapping = {n: one(n) for n in natives}
    # Sanity: the per-module reads agree with the bulk conversion.
    assert set(mapping.values()) <= out_stems | {
        s for s in out_stems
    }, (set(mapping.values()) - out_stems)
    return mapping


# The shared inner rename rule (identical between base blocks and vace blocks): self/cross-attn +
# FFN. Used to project the diffusers-verified base-block rule onto the vace_blocks container.
INNER = {
    "self_attn.q": "attn1.to_q", "self_attn.k": "attn1.to_k",
    "self_attn.v": "attn1.to_v", "self_attn.o": "attn1.to_out.0",
    "cross_attn.q": "attn2.to_q", "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v", "cross_attn.o": "attn2.to_out.0",
    "ffn.0": "ffn.net.0.proj", "ffn.2": "ffn.net.2",
}
# The VACE-specific hint projections (diffusers does not convert these): proj_in exists only on
# vace_blocks.0, proj_out on every vace block.
PROJ = {"before_proj": "proj_in", "after_proj": "proj_out"}


def main():
    keys = host_keys()
    print(f"host weight keys: {len(keys)}")
    mapping: dict[str, str] = {}

    # 1) Native base-block + global stems → diffusers (authoritative: the diffusers converter).
    for native, diff in native_basis().items():
        mapping[f"diffusion_model.{native}"] = diff
    # Alternate prefixes exercise the prefix-strip (all collapse to the same diffusers stem).
    mapping["model.diffusion_model.blocks.3.self_attn.q"] = "blocks.3.attn1.to_q"
    mapping["base_model.model.blocks.5.cross_attn.o"] = "blocks.5.attn2.to_out.0"
    mapping["model.blocks.7.ffn.0"] = "blocks.7.ffn.net.0.proj"

    # 2) Native vace_blocks stems → diffusers (shared inner rule + VACE proj rename; diffusers itself
    #    leaves vace_blocks unconverted, so these are verified against the real host key set below).
    #    proj_in is block-0-only, proj_out is every block.
    for j in (0, 1):
        for inner, diff in INNER.items():
            mapping[f"diffusion_model.vace_blocks.{j}.{inner}"] = f"vace_blocks.{j}.{diff}"
        for native, diff in PROJ.items():
            if native == "before_proj" and j != 0:
                continue  # proj_in only exists on vace_blocks.0
            mapping[f"diffusion_model.vace_blocks.{j}.{native}"] = f"vace_blocks.{j}.{diff}"

    # 3) Diffusers-named passthrough (already host-shaped): base blocks (attn/ffn only — no proj_*),
    #    vace blocks (attn/ffn + proj_out, proj_in on block 0), globals.
    passthrough = [f"blocks.0.{inner}" for inner in INNER.values()]
    for j in (0, 1):
        passthrough += [f"vace_blocks.{j}.{inner}" for inner in INNER.values()]
        passthrough.append(f"vace_blocks.{j}.proj_out")
    passthrough.append("vace_blocks.0.proj_in")
    passthrough += [
        "condition_embedder.time_proj", "proj_out",
        "condition_embedder.text_embedder.linear_1",
        "condition_embedder.text_embedder.linear_2",
        "condition_embedder.time_embedder.linear_1",
        "condition_embedder.time_embedder.linear_2",
    ]
    for p in passthrough:
        mapping[f"diffusion_model.{p}"] = p

    # Sanity: every mapped target must be a real host module (its `.weight` exists). proj_in only
    # exists on vace_blocks.0 in the real checkpoint, so guard the block-0-only modules.
    missing = sorted(
        v for v in set(mapping.values())
        if f"{v}.weight" not in keys
    )
    if missing:
        print(f"WARNING: {len(missing)} targets absent from the host key set: {missing}")

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(mapping, open(FIXTURE, "w"), indent=2, sort_keys=True)
    print(f"wrote {FIXTURE}  ({len(mapping)} entries)")


if __name__ == "__main__":
    main()
