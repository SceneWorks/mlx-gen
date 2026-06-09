"""sc-3184: synthetic-fixture golden for the SenseNova-U1 flow-matching head, the timestep/
noise-scale embedders, and the FM sampler math.

For the 8B-MoT checkpoint (`use_pixel_head=false`, `fm_head_layers=2`) the FM head is a plain
`Sequential(Linear, GELU, Linear)` (no AdaLN/ResBlocks). `TimestepEmbedder` (GLIDE sinusoidal → MLP)
backs both `timestep_embedder` and `noise_scale_embedder`. The sampler is the reference's
`_apply_time_schedule` (standard + dynamic-μ), `_euler_step`, the velocity formula, and
`patchify`/`unpatchify`. This dumps each, calling the real reference methods via a config shim.

Run: cd _vendor/sensenova_u1 && PYTHONPATH=src .venv/bin/python ../../tools/dump_sensenova_fm_golden.py
Fixture -> mlx-gen-sensenova/tests/fixtures/fm_golden.safetensors
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
from safetensors.torch import save_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from sensenova_u1.models.neo_unify.modeling_fm_modules import TimestepEmbedder
from sensenova_u1.models.neo_unify.modeling_neo_chat import NEOChatModel


class Shim:
    """A bare object carrying just the config attrs the pure sampler methods read."""

    base_image_seq_len = 64
    max_image_seq_len = 4096
    base_shift = 0.5
    max_shift = 1.15
    time_shift_type = "exponential"
    time_schedule = "standard"


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    h = 32            # llm hidden (small)
    inter = 64        # fm_head intermediate (the real model hardcodes 4096; the port is dim-agnostic)
    out_dim = 48      # 3 * (patch*merge)^2 with patch=2, merge=2 -> 3*16

    tensors = {}

    # ---- fm_head: Sequential(Linear, GELU, Linear) ----
    fm_head = nn.Sequential(
        nn.Linear(h, inter, bias=True), nn.GELU(), nn.Linear(inter, out_dim, bias=True)
    ).to(torch.float32).eval()
    fm_in = torch.randn(2, 3, h, dtype=torch.float32)  # [B, L, H]
    fm_out = fm_head(fm_in)
    for k, v in fm_head.state_dict().items():
        tensors[f"fm_modules.fm_head.{k}"] = v.contiguous().to(torch.float32)
    tensors["fm.in"] = fm_in
    tensors["fm.out"] = fm_out

    # ---- TimestepEmbedder (used for timestep_embedder and noise_scale_embedder) ----
    te = TimestepEmbedder(h).to(torch.float32).eval()
    ts_in = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=torch.float32)
    ts_out = te(ts_in)
    for k, v in te.state_dict().items():
        tensors[f"fm_modules.timestep_embedder.{k}"] = v.contiguous().to(torch.float32)
    tensors["ts.in"] = ts_in
    tensors["ts.out"] = ts_out

    # ---- sampler math (pure reference methods via the shim) ----
    # NOTE: `_apply_time_schedule` sets `self.time_schedule = "standard"` unconditionally on entry,
    # so the "dynamic"/`_calculate_dynamic_mu` branch is DEAD code — the effective schedule is always
    # the standard one: sigma=1-t; sigma = shift*sigma/(1+(shift-1)*sigma); return 1-sigma.
    shim = Shim()
    t_vec = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32)
    tensors["sched.t"] = t_vec
    tensors["sched.standard_shift1"] = NEOChatModel._apply_time_schedule(shim, t_vec.clone(), 1024, 1.0)
    tensors["sched.standard_shift3"] = NEOChatModel._apply_time_schedule(shim, t_vec.clone(), 1024, 3.0)
    tensors["sched.standard_shift05"] = NEOChatModel._apply_time_schedule(shim, t_vec.clone(), 1024, 0.5)

    # euler + velocity
    z = torch.randn(2, 4, out_dim, dtype=torch.float32)
    x_pred = torch.randn(2, 4, out_dim, dtype=torch.float32)
    t_scalar = torch.tensor(0.3, dtype=torch.float32)
    t_next = torch.tensor(0.55, dtype=torch.float32)
    t_eps = 0.05
    v_pred = (x_pred - z) / (1 - t_scalar).clamp_min(t_eps)
    z_next = NEOChatModel._euler_step(shim, v_pred, z, t_scalar, t_next)
    tensors["euler.z"] = z
    tensors["euler.x_pred"] = x_pred
    tensors["euler.v_pred"] = v_pred
    tensors["euler.z_next"] = z_next

    # patchify / unpatchify (patch_size 2 over an 8x8 image)
    img = torch.randn(1, 3, 8, 8, dtype=torch.float32)
    patches = NEOChatModel.patchify(shim, img, 2)            # [1, 16, 12]
    recon = NEOChatModel.unpatchify(shim, patches, 2)         # [1, 3, 8, 8]
    tensors["patch.img"] = img
    tensors["patch.patches"] = patches
    tensors["patch.recon"] = recon

    dst = os.path.join(REPO_ROOT, "mlx-gen-sensenova", "tests", "fixtures", "fm_golden.safetensors")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    meta = {"t_eps": "0.05", "t_scalar": "0.3", "t_next": "0.55"}
    save_file(tensors, dst, metadata=meta)
    print(f"wrote {dst}")
    print(f"  fm.out {tuple(fm_out.shape)}  ts.out {tuple(ts_out.shape)}  patches {tuple(patches.shape)}")
    print(f"  tensors: {len(tensors)}")


if __name__ == "__main__":
    sys.exit(main())
