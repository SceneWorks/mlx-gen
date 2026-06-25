#!/usr/bin/env python
"""Dump a tiny PidNet (backbone + sigma-aware LQ adapter) parity fixture for sc-7843 component 2.

Exercises the latent-only LQ path the catalog uses: `lq_in_channels=0`, `z_to_patch_ratio=2`
(nearest-upsample), `lq_interval=2` over a 4-block patch stream → 2 output heads + 2 gates at blocks
0 and 2. Captures the LQ-projection feature sets, an isolated sigma-gate I/O, and the full PidNet
forward (gate-injected) so the Rust port can parity-check each seam.

Run: /Users/michael/Repos/mlx-gen/_vendor/pid/.venv-pid/bin/python tools/dump_pid_lq.py
"""

import os
import sys

import torch
from safetensors.torch import save_file

PID_ROOT = "/Users/michael/Repos/mlx-gen/_vendor/pid"
sys.path.insert(0, PID_ROOT)

from pid._src.networks.pid_net import PidNet  # noqa: E402

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "mlx-gen-pid", "tests", "fixtures", "pidnet_tiny.safetensors",
)

CFG = dict(
    # backbone (same shape family as pixdit_tiny, but 4 patch blocks for the interval=2 mapping)
    in_channels=3, num_groups=2, hidden_size=32, pixel_hidden_size=8,
    pixel_attn_hidden_size=16, pixel_num_groups=2, patch_depth=4, pixel_depth=2,
    patch_size=2, txt_embed_dim=12, txt_max_length=5, use_text_rope=True,
    text_rope_theta=10000.0, rope_mode="ntk_aware", rope_ref_h=16, rope_ref_w=16,
    repa_encoder_index=-1, enable_ed=False,
    # LQ adapter (latent-only, the catalog path)
    lq_inject_mode="controlnet", lq_in_channels=0, lq_latent_channels=4, lq_hidden_dim=8,
    lq_num_res_blocks=2, lq_gate_type="sigma_aware_per_token_per_dim", lq_interval=2,
    zero_init_lq=True, sr_scale=2, latent_spatial_down_factor=2, pit_lq_inject=False,
)
H, W = 8, 12  # Hs=4, Ws=6, L=24; z_to_patch_ratio=(2*2)/2=2 -> latent zH=2, zW=3


def main():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(4321)
    model = PidNet(**CFG)
    model.eval()

    with torch.no_grad():
        for name, p in model.named_parameters():
            r = torch.randn(p.shape, generator=g) * 0.2
            if "norm" in name and p.dim() == 1:
                r = r + 1.0
            p.copy_(r)

    patch = CFG["patch_size"]
    pH, pW = H // patch, W // patch
    Ltxt = CFG["txt_max_length"]
    hidden = CFG["hidden_size"]
    L = pH * pW

    x = torch.randn(1, 3, H, W, generator=g)
    t = torch.tensor([3.7], dtype=torch.float32)
    y = torch.randn(1, Ltxt, CFG["txt_embed_dim"], generator=g)
    # latent half the patch grid (ratio 2): [B, z_dim=4, zH=2, zW=3]
    lq_latent = torch.randn(1, CFG["lq_latent_channels"], pH // 2, pW // 2, generator=g)
    sigma = torch.tensor([0.3], dtype=torch.float32)

    tensors = {}
    for k, v in model.state_dict().items():
        tensors[k] = v.detach().contiguous().float()

    with torch.no_grad():
        # 1) LQ projection feature sets (conv stack + output heads) at the target patch grid
        lq_feats = model.lq_proj(lq_latent=lq_latent, target_pH=pH, target_pW=pW)
        # 2) isolated sigma-gate I/O (controlled x, lq, sigma) on gate 0
        xg = torch.randn(1, L, hidden, generator=g)
        lqg = torch.randn(1, L, hidden, generator=g)
        gate_out = model.lq_proj.gate(xg, lqg, sigma=sigma, out_idx=0)
        # 3) full gate-injected PidNet forward
        out = model(x, t, y, lq_latent=lq_latent, degrade_sigma=sigma)

    tensors["__io__.x"] = x.contiguous().float()
    tensors["__io__.t"] = t.contiguous().float()
    tensors["__io__.y"] = y.contiguous().float()
    tensors["__io__.lq_latent"] = lq_latent.contiguous().float()
    tensors["__io__.sigma"] = sigma.contiguous().float()
    tensors["__io__.output"] = out.contiguous().float()
    for i, f in enumerate(lq_feats):
        tensors[f"__io__.lq_feat_{i}"] = f.contiguous().float()
    tensors["__io__.gate_xg"] = xg.contiguous().float()
    tensors["__io__.gate_lqg"] = lqg.contiguous().float()
    tensors["__io__.gate_out"] = gate_out.contiguous().float()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    meta = {k: str(v) for k, v in CFG.items()}
    meta.update({"img_h": str(H), "img_w": str(W)})
    save_file(tensors, OUT, metadata=meta)

    print(f"wrote {OUT}")
    print(f"  num lq_proj outputs: {len(lq_feats)} (each {tuple(lq_feats[0].shape)})")
    print(f"  output {tuple(out.shape)} mean={out.mean():.5f} std={out.std():.5f}")
    print(f"  gate_out {tuple(gate_out.shape)}")
    print("  lq_proj.* keys:")
    for k in sorted(tensors):
        if k.startswith("lq_proj"):
            print(f"    {k}  {tuple(tensors[k].shape)}")


if __name__ == "__main__":
    main()
