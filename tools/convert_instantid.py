#!/usr/bin/env python
"""InstantID weight conversion to mlx-gen format (sc-3112).

InstantID ships two checkpoints (from `InstantX/InstantID`):

  1. ``ip-adapter.bin`` — a torch **pickle** dict ``{"image_proj": {...}, "ip_adapter": {...}}``.
     mlx-gen's `Weights::from_file` reads **safetensors**, not pickle, so this must be re-serialized.
     The two sub-dicts map onto mlx-gen exactly (no key rewrite — the names already match):
       - ``image_proj.*`` → the face Resampler (`Resampler::from_weights(.., "image_proj", ..)`,
         `ResamplerConfig::instantid_face()`); 51 tensors.
       - ``ip_adapter.{n}.to_k_ip/to_v_ip.weight`` → the decoupled-cross-attn K/V pairs
         (`load_ip_kv_pairs`, which strips the ``ip_adapter.`` prefix); 140 tensors = 70 SDXL
         cross-attn layers × (k, v). The indices are the diffusers attn-processor walk order
         (1, 3, 5, …, 139) — already what `load_ip_kv_pairs` returns sorted.
     Output mirrors the h94 IP-Adapter convention (both namespaces in ONE safetensors).

  2. ``ControlNetModel/diffusion_pytorch_model.safetensors`` — the **IdentityNet**. This is a stock
     diffusers SDXL ``ControlNetModel`` (config: cross_attention_dim=2048, addition_time_embed_dim=256,
     projection_class_embeddings_input_dim=2816, block_out_channels=[320,640,1280], heads=[5,10,20]) —
     i.e. exactly the layout `ControlNet::from_weights(.., &UNetConfig::sdxl_base())` (sc-3058) already
     consumes. **No conversion is required**; it loads directly. This script only verifies its
     presence + config so the bundle is self-documenting.

The `image_proj`/`ip_adapter` tensors are preserved at their **source dtype** (f32 in the published
checkpoint); the mlx-gen loader casts to the model dtype (fp16) at load time. Output is gitignored
(see tools/golden/README.md) — multi-GB, regenerable, licensed weights.

Run from a torch venv (has torch + safetensors):
    ~/repos/mflux/.venv-0312/bin/python ~/Repos/mlx-gen/tools/convert_instantid.py
    # optional: --src <snapshot dir>  --out <output dir>
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

HUB = Path.home() / ".cache/huggingface/hub"
DEFAULT_SRC = (
    HUB
    / "models--InstantX--InstantID/snapshots/57b32dfee076092ad2930c71fd6d439c2c3b1820"
)
DEFAULT_OUT = Path(__file__).resolve().parent / "golden" / "instantid"

# IdentityNet must match UNetConfig::sdxl_base() for the existing ControlNet loader to consume it.
EXPECTED_CN_CONFIG = {
    "cross_attention_dim": 2048,
    "addition_time_embed_dim": 256,
    "projection_class_embeddings_input_dim": 2816,
    "block_out_channels": [320, 640, 1280],
}


def convert_ip_adapter(src: Path, out: Path) -> Path:
    """Split `ip-adapter.bin` (pickle) → one safetensors with `image_proj.*` + `ip_adapter.*`."""
    # weights_only=True: ip-adapter.bin is a plain tensor state dict, so refuse to execute arbitrary
    # pickle opcodes from a third-party artifact (unsafe default on torch < 2.6 — F-152).
    state = torch.load(str(src / "ip-adapter.bin"), map_location="cpu", weights_only=True)
    assert set(state) >= {"image_proj", "ip_adapter"}, f"unexpected top keys: {list(state)}"

    tensors = {}
    for k, v in state["image_proj"].items():
        tensors[f"image_proj.{k}"] = v.contiguous()
    for k, v in state["ip_adapter"].items():
        tensors[f"ip_adapter.{k}"] = v.contiguous()

    n_img = sum(1 for k in tensors if k.startswith("image_proj."))
    n_ip = sum(1 for k in tensors if k.startswith("ip_adapter."))
    kv_pairs = sum(1 for k in tensors if k.startswith("ip_adapter.") and k.endswith(".to_k_ip.weight"))
    assert n_img == 51, f"expected 51 image_proj tensors, got {n_img}"
    assert kv_pairs == 70, f"expected 70 ip_adapter K/V pairs, got {kv_pairs}"

    dst = out / "ip-adapter.safetensors"
    save_file(tensors, str(dst), metadata={"format": "pt", "source": "InstantX/InstantID ip-adapter.bin"})
    print(f"wrote {dst}")
    print(f"  image_proj: {n_img} tensors   ip_adapter: {n_ip} tensors ({kv_pairs} K/V pairs)")
    print(f"  image_proj.proj_in.weight {tuple(tensors['image_proj.proj_in.weight'].shape)}  "
          f"ip_adapter.1.to_k_ip.weight {tuple(tensors['ip_adapter.1.to_k_ip.weight'].shape)}")
    return dst


def verify_identitynet(src: Path) -> None:
    """IdentityNet is a stock diffusers SDXL ControlNetModel — confirm it loads directly (no rewrite)."""
    cn_dir = src / "ControlNetModel"
    st = cn_dir / "diffusion_pytorch_model.safetensors"
    cfg = json.loads((cn_dir / "config.json").read_text())
    mismatch = {k: (cfg.get(k), v) for k, v in EXPECTED_CN_CONFIG.items() if cfg.get(k) != v}
    assert not mismatch, f"IdentityNet config != UNetConfig::sdxl_base(): {mismatch}"
    with safe_open(str(st), framework="pt") as f:
        keys = list(f.keys())
    has = lambda p: any(k.startswith(p) for k in keys)  # noqa: E731
    for p in ("conv_in", "down_blocks.", "mid_block.", "controlnet_cond_embedding.",
              "controlnet_down_blocks.", "controlnet_mid_block"):
        assert has(p), f"IdentityNet missing expected diffusers key prefix {p!r}"
    print(f"IdentityNet OK (loads directly via ControlNet::from_weights + UNetConfig::sdxl_base())")
    print(f"  {st}")
    print(f"  {len(keys)} tensors; config matches SDXL ControlNet (cross_attention_dim=2048, "
          f"proj_class_emb=2816, channels={cfg['block_out_channels']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="InstantID snapshot dir")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output dir")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    convert_ip_adapter(args.src, args.out)
    verify_identitynet(args.src)
    print(f"\nInstantID weights ready. Consumers:")
    print(f"  Resampler  : {args.out / 'ip-adapter.safetensors'} (image_proj.*)")
    print(f"  IP K/V     : {args.out / 'ip-adapter.safetensors'} (ip_adapter.*)")
    print(f"  IdentityNet: {args.src / 'ControlNetModel/diffusion_pytorch_model.safetensors'} (direct)")


if __name__ == "__main__":
    main()
