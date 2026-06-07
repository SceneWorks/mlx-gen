"""Real-weight golden for the Wan-VACE transformer parity (epic 3040 / sc-3388, S1 upgrade).

Loads the **real** diffusers `Wan-AI/Wan2.1-VACE-1.3B-diffusers` transformer (dim 1536, 30 layers,
vace_layers=[0,2,...,28]) in f32 and runs one forward on seeded injected inputs (no text encoder / no
VAE — the context is injected directly, like the base-Wan S3 parity). Dumps **only the I/O tensors**
(a few MB) so the Rust port loads the same 7 GB safetensors shards directly and compares its
`forward_vace` against this output.

Run: WANVACE_DIR=~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-VACE-1.3B-diffusers/snapshots/<hash> \
     /Users/michael/repos/mflux/.venv-0312/bin/python tools/dump_wanvace_real_golden.py
Writes `mlx-gen-wan/tests/fixtures/wanvace_real_io.safetensors`.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import torch
from safetensors.torch import save_file
from diffusers.models.transformers.transformer_wan_vace import WanVACETransformer3DModel

from _paths import fixture


def transformer_dir() -> str:
    d = os.environ.get("WANVACE_DIR")
    if d and Path(d, "transformer").is_dir():
        return str(Path(d, "transformer"))
    if d and Path(d, "config.json").is_file():
        return d
    # Default: the HF cache snapshot.
    hits = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-VACE-1.3B-diffusers/snapshots/*/transformer"
        )
    )
    if not hits:
        raise SystemExit("set WANVACE_DIR (or download Wan-AI/Wan2.1-VACE-1.3B-diffusers transformer/)")
    return hits[0]


torch.manual_seed(3388)

tdir = transformer_dir()
print("loading", tdir)
model = WanVACETransformer3DModel.from_pretrained(tdir, torch_dtype=torch.float32).eval()
n_vace = len(model.config.vace_layers)
print("vace_layers:", model.config.vace_layers)

# Small grid keeps the CPU forward fast on real 1536-dim/30-layer weights: latent [1,16,4,16,16] →
# patchify (1,2,2) → grid (4,8,8) → L=256 tokens. 96-ch control at the same grid. Text [1,12,4096].
T, H, W = 4, 16, 16
hidden_states = torch.randn(1, 16, T, H, W)
control_hidden_states = torch.randn(1, 96, T, H, W)
timestep = torch.tensor([3.0])
encoder_hidden_states = torch.randn(1, 12, 4096)
# Non-trivial per-vace-layer scales (monotone) so the gate catches a mis-applied / reversed scale.
control_scale = torch.tensor([1.0 - 0.5 * i / (n_vace - 1) for i in range(n_vace)])

with torch.no_grad():
    out = model(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        control_hidden_states=control_hidden_states,
        control_hidden_states_scale=control_scale,
        return_dict=False,
    )[0]

tensors = {
    "in.hidden_states": hidden_states.contiguous(),
    "in.control_hidden_states": control_hidden_states.contiguous(),
    "in.timestep": timestep.contiguous(),
    "in.encoder_hidden_states": encoder_hidden_states.contiguous(),
    "in.control_hidden_states_scale": control_scale.contiguous(),
    "out.sample": out.contiguous(),
}
out_path = fixture("mlx-gen-wan/tests/fixtures/wanvace_real_io.safetensors")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, out_path)
print(f"wrote {out_path}")
print("  output:", tuple(out.shape), " mean/std:", float(out.mean()), float(out.std()))
