"""Real-weight LoRA golden for the Wan-VACE diffusers-name adapter merge (sc-3439).

The definitive end-to-end gate for `merge_vace_adapters`: it folds a synthetic LoRA onto the **real**
`Wan-AI/Wan2.1-VACE-1.3B-diffusers` transformer two ways that must agree —
  * base-transformer-block factors are emitted in **native Wan naming** (`blocks.N.self_attn.q`, …,
    `blocks.N.ffn.0/2`) and routed to the diffusers host through diffusers' own
    `_convert_non_diffusers_wan_lora_to_diffusers` (the authoritative reference for the native→diffusers
    rename the Rust `normalize_vace_key` reproduces), and
  * VACE-block factors are emitted in **diffusers naming** (`vace_blocks.N.attn1.to_q`, `proj_in`,
    `proj_out`, `ffn.net.*`) — the diffusers converter does not touch `vace_blocks`, so these are
    folded directly.
The same on-disk LoRA file is later read by the Rust `wanvace_lora_real_parity` test, which merges it
via `merge_vace_adapters` and must reproduce the `out.lora` forward to the cross-backend f32 matmul
floor, while diverging from `out.bare` (a real, non-trivial effect).

Run: WANVACE_DIR=~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-VACE-1.3B-diffusers/snapshots/<hash> \
     "$HOME/mlx-flux-venv/bin/python" tools/dump_wanvace_lora_real_golden.py
Writes the committed LoRA `mlx-gen-wan/tests/fixtures/wanvace_real_lora.safetensors` (a few MB) and the
golden `mlx-gen-wan/tests/fixtures/wanvace_lora_real_io.safetensors` (bare + lora outputs).
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from diffusers.models.transformers.transformer_wan_vace import WanVACETransformer3DModel
from diffusers.loaders.lora_conversion_utils import (
    _convert_non_diffusers_wan_lora_to_diffusers,
)

from _paths import fixture


def transformer_dir() -> str:
    d = os.environ.get("WANVACE_DIR")
    if d and Path(d, "transformer").is_dir():
        return str(Path(d, "transformer"))
    if d and Path(d, "config.json").is_file():
        return d
    hits = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-VACE-1.3B-diffusers/snapshots/*/transformer"
        )
    )
    if not hits:
        raise SystemExit("set WANVACE_DIR (or download the Wan2.1-VACE-1.3B-diffusers transformer/)")
    return hits[0]


RANK = 4
torch.manual_seed(3439)

tdir = transformer_dir()
print("loading", tdir)
model = WanVACETransformer3DModel.from_pretrained(tdir, torch_dtype=torch.float32).eval()
sd = model.state_dict()
n_vace = len(model.config.vace_layers)
print("vace_layers:", model.config.vace_layers)


def factors(out_dim: int, in_dim: int):
    """A small-rank LoRA pair (A=[rank,in], B=[out,rank]) with a visible but stable delta."""
    a = torch.randn(RANK, in_dim) * 0.04
    b = torch.randn(out_dim, RANK) * 0.04
    return a, b


def host_shape(path: str) -> tuple[int, int]:
    w = sd[f"{path}.weight"]
    return w.shape[0], w.shape[1]


# Build the LoRA: native-named base blocks + diffusers-named vace blocks. Stored `diffusion_model.`-
# prefixed (the real-file convention). We track the diffusers host path each factor targets so the
# Python golden can fold it; the Rust side rediscovers the same path via normalize_vace_key.
lora: dict[str, torch.Tensor] = {}
targets: dict[str, str] = {}  # host param path → "native" | "vace"

BASE_BLOCKS = [0, 1, 29]
NATIVE_INNER = {
    "self_attn.q": "attn1.to_q", "self_attn.k": "attn1.to_k",
    "self_attn.v": "attn1.to_v", "self_attn.o": "attn1.to_out.0",
    "cross_attn.q": "attn2.to_q", "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v", "cross_attn.o": "attn2.to_out.0",
    "ffn.0": "ffn.net.0.proj", "ffn.2": "ffn.net.2",
}
for i in BASE_BLOCKS:
    for native, diff in NATIVE_INNER.items():
        host = f"blocks.{i}.{diff}"
        a, b = factors(*host_shape(host))
        lora[f"diffusion_model.blocks.{i}.{native}.lora_A.weight"] = a
        lora[f"diffusion_model.blocks.{i}.{native}.lora_B.weight"] = b
        targets[host] = "native"

VACE_BLOCKS = [0, 1]
VACE_DIFF = list(NATIVE_INNER.values())  # attn/ffn, already diffusers
for j in VACE_BLOCKS:
    mods = list(VACE_DIFF) + (["proj_in", "proj_out"] if j == 0 else ["proj_out"])
    for diff in mods:
        host = f"vace_blocks.{j}.{diff}"
        a, b = factors(*host_shape(host))
        lora[f"diffusion_model.vace_blocks.{j}.{diff}.lora_A.weight"] = a
        lora[f"diffusion_model.vace_blocks.{j}.{diff}.lora_B.weight"] = b
        targets[host] = "vace"

lora_path = fixture("mlx-gen-wan/tests/fixtures/wanvace_real_lora.safetensors")
Path(lora_path).parent.mkdir(parents=True, exist_ok=True)
save_file({k: v.contiguous() for k, v in lora.items()}, lora_path)
print(f"wrote {lora_path}  ({len(lora)} tensors, {len(targets)} modules)")

# --- Inputs: reuse the committed real-IO fixture (same grid as dump_wanvace_real_golden.py) ---
io = load_file(fixture("mlx-gen-wan/tests/fixtures/wanvace_real_io.safetensors"))
fwd = dict(
    hidden_states=io["in.hidden_states"],
    timestep=io["in.timestep"],
    encoder_hidden_states=io["in.encoder_hidden_states"],
    control_hidden_states=io["in.control_hidden_states"],
    control_hidden_states_scale=io["in.control_hidden_states_scale"],
    return_dict=False,
)

with torch.no_grad():
    bare = model(**fwd)[0].contiguous()

# --- Fold the LoRA: base via the diffusers converter (native→diffusers), vace directly ---
base_sd = {k: v for k, v in lora.items() if ".vace_blocks." not in k}
converted = _convert_non_diffusers_wan_lora_to_diffusers(dict(base_sd))


def strip_t(k: str) -> str:
    return k[len("transformer.") :] if k.startswith("transformer.") else k


# Group converted (diffusers-named) + vace (diffusers-named) factors by host path.
groups: dict[str, dict[str, torch.Tensor]] = {}
for k, v in converted.items():
    m = re.match(r"(.+)\.(lora_A|lora_B)\.weight$", strip_t(k))
    if m:
        groups.setdefault(m.group(1), {})[m.group(2)] = v
for k, v in lora.items():
    if ".vace_blocks." not in k:
        continue
    m = re.match(r"diffusion_model\.(.+)\.(lora_A|lora_B)\.weight$", k)
    if m:
        groups.setdefault(m.group(1), {})[m.group(2)] = v

folded = dict(sd)
for path, fac in groups.items():
    delta = fac["lora_B"] @ fac["lora_A"]  # [out,in]
    folded[f"{path}.weight"] = sd[f"{path}.weight"] + delta
assert set(groups) == set(targets), (set(groups) ^ set(targets))
model.load_state_dict(folded)

with torch.no_grad():
    lora_out = model(**fwd)[0].contiguous()

eff = (lora_out - bare).abs().mean() / bare.abs().mean()
print(f"folded {len(groups)} modules  |Δ| effect (lora vs bare) mean_rel={float(eff):.4e}")

out_path = fixture("mlx-gen-wan/tests/fixtures/wanvace_lora_real_io.safetensors")
save_file({"out.bare": bare, "out.lora": lora_out}, out_path)
print(f"wrote {out_path}")
print("  bare mean/std:", float(bare.mean()), float(bare.std()))
print("  lora mean/std:", float(lora_out.mean()), float(lora_out.std()))
