#!/usr/bin/env python
"""sc-3642 parity reference generator — third-party (non-peft / lycoris) LoKr.

Builds REAL lycoris LoKr adapters over toy Linear/Conv2d modules and emits, per case:

  <case>.safetensors           the lycoris state_dict exactly as a third-party file ships it
                               (per-module `<PREFIX>_<flattened.path>.lokr_*` + `.alpha`,
                               NO global `networkType`/`rank`/`alpha` metadata).
  <case>.expected.safetensors  ground-truth per-module delta `ΔW` keyed by the DOTTED
                               diffusers module path, = `LokrModule.get_weight(shape)` at
                               multiplier=scalar=1 — what the engine's reconstruction must match.
                               Metadata records the derived (prefix, rank, alpha, scale) per module
                               so the Rust port can cross-check its own derivation.

The Rust test (src/adapters/loader.rs) loads both and asserts the reconstructed delta matches
within tolerance — the on-device A/B that replaces a torch-on-Linux golden (torch lives in
`~/mlx-flux-venv`). Run: `~/mlx-flux-venv/bin/python tools/sc3642_lokr_reference.py`.

Cases cover the four reconstruction shapes a real file hits:
  - linear_w1full_w2lr  : full lokr_w1 + decomposed lokr_w2_a@lokr_w2_b   (common SDXL-attn LoKr)
  - linear_bothlr       : decomposed lokr_w1_a@lokr_w1_b + lokr_w2_a@lokr_w2_b (decompose_both)
  - linear_bothfull     : full lokr_w1 + full lokr_w2                       (scale forced to 1)
  - conv_tucker         : conv with lokr_t2 tucker factor (use_cp / use_tucker)
"""

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

from lycoris import LycorisNetwork, create_lycoris

from _paths import fixture

OUT = Path(fixture("tests/fixtures/sc3642_lokr"))


def build(module: nn.Module, *, dim: int, alpha: float, factor: int, tucker: bool, decompose_both: bool):
    """Wrap `module` (a Sequential with named children) in a lycoris LoKr network and apply it."""
    LycorisNetwork.apply_preset({"target_module": ["Sequential"]})
    net = create_lycoris(
        module,
        1.0,
        linear_dim=dim,
        linear_alpha=alpha,
        conv_dim=dim,
        conv_alpha=alpha,
        algo="lokr",
        factor=factor,
        use_tucker=tucker,
        decompose_both=decompose_both,
    )
    net.apply_to()
    return net


def reconstruct(net) -> dict:
    """Per-LoKr-module ground-truth delta + derived (rank/alpha/scale), keyed by dotted path."""
    deltas, meta = {}, {}
    for lora in net.loras:
        # lora.lora_name is the lycoris prefix + flattened path, e.g. "lycoris_blocks_0_proj".
        delta = lora.get_weight(lora.shape).detach().float().contiguous()
        # Recover the DOTTED diffusers path the engine resolves against. lycoris stores the
        # underscore-flattened name; our toy module names use no underscores so a single split works.
        dotted = lora.lora_name.split("_", 1)[1].replace("_", ".")
        deltas[dotted] = delta
        rank = int(lora.lora_dim)
        alpha = float(lora.alpha.item())
        meta[dotted] = {
            "prefix": lora.lora_name.rsplit(dotted.replace(".", "_"), 1)[0].rstrip("_"),
            "rank": rank,
            "alpha": alpha,
            "scale": float(lora.scale),
            "use_w1": bool(lora.use_w1),
            "use_w2": bool(lora.use_w2),
            "tucker": bool(lora.tucker),
            "shape": list(lora.shape),
        }
    return deltas, meta


def emit(name: str, net):
    OUT.mkdir(parents=True, exist_ok=True)
    # The third-party file: raw lycoris state_dict, float32, NO global metadata.
    sd = {k: v.detach().float().contiguous() for k, v in net.state_dict().items()}
    save_file(sd, str(OUT / f"{name}.safetensors"))
    deltas, meta = reconstruct(net)
    save_file(deltas, str(OUT / f"{name}.expected.safetensors"), metadata={"derived": json.dumps(meta)})
    print(f"  {name}: {len(sd)} tensors -> {sorted(sd.keys())}")
    for path, m in meta.items():
        print(f"     {path}: rank={m['rank']} alpha={m['alpha']} scale={m['scale']:.6f} "
              f"use_w1={m['use_w1']} use_w2={m['use_w2']} tucker={m['tucker']} shape={m['shape']}")


def main():
    torch.manual_seed(0)
    print("sc-3642 LoKr parity fixtures ->", OUT)

    # linear_w1full_w2lr — out/in large enough that w2 decomposes but w1 stays full (typical attn LoKr).
    m = nn.Sequential(); m.add_module("proj", nn.Linear(128, 128))
    emit("linear_w1full_w2lr", build(m, dim=4, alpha=4, factor=-1, tucker=False, decompose_both=False))

    # linear_bothlr — decompose_both so w1 also low-rank (144=12x12 → small factors decompose at dim 4).
    m = nn.Sequential(); m.add_module("proj", nn.Linear(144, 144))
    emit("linear_bothlr", build(m, dim=4, alpha=4, factor=-1, tucker=False, decompose_both=True))

    # linear_bothfull — tiny dims force both factors full (scale forced to 1).
    m = nn.Sequential(); m.add_module("proj", nn.Linear(16, 24))
    emit("linear_bothfull", build(m, dim=4, alpha=2, factor=4, tucker=False, decompose_both=False))

    # conv_tucker — 3x3 conv with tucker (lokr_t2).
    m = nn.Sequential(); m.add_module("conv", nn.Conv2d(64, 96, kernel_size=3, padding=1))
    emit("conv_tucker", build(m, dim=4, alpha=4, factor=-1, tucker=True, decompose_both=False))


if __name__ == "__main__":
    main()
