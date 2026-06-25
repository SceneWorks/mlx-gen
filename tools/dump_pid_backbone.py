#!/usr/bin/env python
"""Dump a tiny PixDiT_T2I parity fixture for the mlx-gen-pid backbone port (sc-7843).

Builds a *small* PixDiT_T2I (the base T2I backbone PidNet inherits — no LQ adapter), gives it
deterministic non-degenerate weights (the real `initialize_weights` zeroes the final layer, which
would make the whole fixture all-zeros), runs one forward pass, and writes the bare-key state_dict +
inputs + every intermediate seam to a committed safetensors fixture. The Rust parity test loads it,
rebuilds the modules `from_weights`, and asserts each seam agrees.

Config is deliberately tiny and *non-square* (H != W) with an NTK ref grid != sampled grid, so the
fixture exercises: the (x,y) meshgrid order, per-axis NTK theta scaling, unfold/fold patchify, the
pixel-stream compress/expand, and the dual-stream joint attention — all the places a port goes wrong.

Run from the reference's isolated env (torch 2.9.1):
  /Users/michael/Repos/mlx-gen/_vendor/pid/.venv-pid/bin/python tools/dump_pid_backbone.py
"""

import os
import sys

import torch
from safetensors.torch import save_file

PID_ROOT = "/Users/michael/Repos/mlx-gen/_vendor/pid"
sys.path.insert(0, PID_ROOT)

from pid._src.networks.pixeldit_official import PixDiT_T2I  # noqa: E402

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "mlx-gen-pid",
    "tests",
    "fixtures",
    "pixdit_tiny.safetensors",
)

# --- tiny config (mirrors PID_SR4X structurally; head_dims stay divisible by 4 for 2D RoPE) ---
CFG = dict(
    in_channels=3,
    num_groups=2,             # head_dim = 32/2 = 16
    hidden_size=32,
    pixel_hidden_size=8,
    pixel_attn_hidden_size=16,  # pixel_head_dim = 16/2 = 8
    pixel_num_groups=2,
    patch_depth=2,
    pixel_depth=2,
    patch_size=2,
    txt_embed_dim=12,
    txt_max_length=5,
    use_text_rope=True,
    text_rope_theta=10000.0,
    rope_mode="ntk_aware",
    rope_ref_h=16,            # ref grid 8x8 vs sampled 4x6 -> non-identity NTK on both axes
    rope_ref_w=16,
    repa_encoder_index=-1,
    enable_ed=False,
)
H, W = 8, 12  # -> Hs=4, Ws=6, L=24 (non-square)
B = 1


def main():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1234)
    model = PixDiT_T2I(**CFG)
    model.eval()

    # Replace every parameter with deterministic non-degenerate values. Norm weights centre on 1
    # (so qk-norm / RMSNorm don't annihilate the signal); everything else is small N(0, 0.2).
    with torch.no_grad():
        for name, p in model.named_parameters():
            r = torch.randn(p.shape, generator=g) * 0.2
            if "norm" in name and p.dim() == 1:
                r = r + 1.0
            p.copy_(r)

    Hs, Ws = H // CFG["patch_size"], W // CFG["patch_size"]
    Ltxt = CFG["txt_max_length"]

    x = torch.randn(B, 3, H, W, generator=g)
    t = torch.tensor([3.7], dtype=torch.float32)
    y = torch.randn(B, Ltxt, CFG["txt_embed_dim"], generator=g)

    tensors = {}
    # bare-key state_dict (the converter strips the real ckpt's `net.` prefix to exactly these)
    for k, v in model.state_dict().items():
        tensors[k] = v.detach().contiguous().float()

    # capture every seam via forward hooks
    caps = {}

    def cap(name):
        def hook(_m, _inp, out):
            if isinstance(out, tuple):
                for i, o in enumerate(out):
                    caps[f"{name}.{i}"] = o.detach().contiguous().float()
            else:
                caps[name] = out.detach().contiguous().float()
        return hook

    model.s_embedder.register_forward_hook(cap("s_embedder"))
    model.t_embedder.register_forward_hook(cap("t_embedder"))
    model.y_embedder.register_forward_hook(cap("y_embedder"))
    model.pixel_embedder.register_forward_hook(cap("pixel_embedder"))
    for i, blk in enumerate(model.patch_blocks):
        blk.register_forward_hook(cap(f"patch_block_{i}"))
    for i, blk in enumerate(model.pixel_blocks):
        blk.register_forward_hook(cap(f"pixel_block_{i}"))
    model.final_layer.register_forward_hook(cap("final_layer"))

    with torch.no_grad():
        out = model(x, t, y)

    # positional tables (host-deterministic; unit-test the Rust ports directly)
    rope_img = model.fetch_pos(Hs, Ws, x.device)          # [L, head_dim/2, 2]
    rope_txt = model.fetch_pos_text(Ltxt, x.device)        # [Ltxt, head_dim/2, 2]
    pixel_pos = model.pixel_embedder._fetch_pixel_pos_image(H, W, x.device, torch.float32)  # [H*W, D]

    tensors["__io__.x"] = x.detach().contiguous().float()
    tensors["__io__.t"] = t.detach().contiguous().float()
    tensors["__io__.y"] = y.detach().contiguous().float()
    tensors["__io__.output"] = out.detach().contiguous().float()
    tensors["__io__.rope_img"] = rope_img.detach().contiguous().float()
    tensors["__io__.rope_txt"] = rope_txt.detach().contiguous().float()
    tensors["__io__.pixel_pos"] = pixel_pos.detach().contiguous().float()
    for k, v in caps.items():
        tensors[f"__io__.{k}"] = v

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # store the tiny config as metadata so the Rust test reads the exact shapes
    meta = {k: str(v) for k, v in CFG.items()}
    meta.update({"img_h": str(H), "img_w": str(W), "batch": str(B)})
    save_file(tensors, OUT, metadata=meta)

    print(f"wrote {OUT}")
    print(f"  state_dict tensors: {len(model.state_dict())}")
    print(f"  output shape: {tuple(out.shape)}  mean={out.mean().item():.6f}  std={out.std().item():.6f}")
    print(f"  rope_img {tuple(rope_img.shape)}  rope_txt {tuple(rope_txt.shape)}  pixel_pos {tuple(pixel_pos.shape)}")
    # print a few key names so the Rust loader uses the exact strings
    sample = [k for k in tensors if not k.startswith("__io__")][:18]
    print("  sample keys:")
    for k in sample:
        print(f"    {k}  {tuple(tensors[k].shape)}")


if __name__ == "__main__":
    main()
