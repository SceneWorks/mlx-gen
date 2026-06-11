#!/usr/bin/env python
"""sc-3643 parity reference generator — third-party (non-peft / lycoris) LoHa.

Sibling of `sc3642_lokr_reference.py`. Builds REAL lycoris LoHa adapters over toy Linear/Conv2d
modules and emits, per case, the third-party `<case>.safetensors` (per-module
`<PREFIX>_<flattened.path>.hada_*` + `.alpha`, no global metadata) and `<case>.expected.safetensors`
(ground-truth ΔW = `LohaModule.get_weight(shape)` keyed by dotted module path). The Rust test
asserts the reconstruction matches — on-device A/B via `~/mlx-flux-venv`.

Cases: linear (`hada_w1_a@hada_w1_b ⊙ hada_w2_a@hada_w2_b`), conv-non-tucker (k folded into dim 1),
and conv-tucker (`hada_t1`/`hada_t2`). Run: `~/mlx-flux-venv/bin/python tools/sc3643_loha_reference.py`.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

from lycoris import LycorisNetwork, create_lycoris

from _paths import fixture

OUT = Path(fixture("tests/fixtures/sc3643_loha"))


def build(module, *, dim, alpha, tucker):
    LycorisNetwork.apply_preset({"target_module": ["Sequential"]})
    net = create_lycoris(
        module, 1.0, linear_dim=dim, linear_alpha=alpha, conv_dim=dim, conv_alpha=alpha,
        algo="loha", use_tucker=tucker,
    )
    net.apply_to()
    return net


def emit(name, net):
    OUT.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().float().contiguous() for k, v in net.state_dict().items()}
    save_file(sd, str(OUT / f"{name}.safetensors"))
    deltas, meta = {}, {}
    for lora in net.loras:
        dotted = lora.lora_name.split("_", 1)[1].replace("_", ".")
        deltas[dotted] = lora.get_weight(lora.shape).detach().float().contiguous()
        meta[dotted] = {
            "rank": int(lora.lora_dim),
            "alpha": float(lora.alpha.item()),
            "scale": float(lora.scale),
            "tucker": bool(lora.tucker),
            "shape": list(lora.shape),
        }
    save_file(deltas, str(OUT / f"{name}.expected.safetensors"), metadata={"derived": json.dumps(meta)})
    print(f"  {name}: {sorted(sd.keys())}")
    for path, m in meta.items():
        print(f"     {path}: rank={m['rank']} alpha={m['alpha']} scale={m['scale']:.6f} "
              f"tucker={m['tucker']} shape={m['shape']}")


def main():
    torch.manual_seed(0)
    print("sc-3643 LoHa parity fixtures ->", OUT)

    m = nn.Sequential(); m.add_module("proj", nn.Linear(128, 96))
    emit("linear", build(m, dim=4, alpha=2, tucker=False))

    m = nn.Sequential(); m.add_module("conv", nn.Conv2d(48, 64, kernel_size=3, padding=1))
    emit("conv_notucker", build(m, dim=4, alpha=4, tucker=False))

    m = nn.Sequential(); m.add_module("conv", nn.Conv2d(48, 64, kernel_size=3, padding=1))
    emit("conv_tucker", build(m, dim=4, alpha=4, tucker=True))


if __name__ == "__main__":
    main()
