#!/usr/bin/env python3
"""sc-3070 — EVA02-CLIP-L-14-336 *visual* tower converter (torch .pt -> mlx-gen safetensors).

PuLID-FLUX consumes only the `.visual` submodule of EVA02-CLIP-L-14-336 (the face-identity ViT).
The upstream tag `EVA02-CLIP-L-14-336` + pretrained `eva_clip` resolves to
`QuanSun/EVA-CLIP/EVA02_CLIP_L_336_psz14_s6B.pt` (see eva_clip/pretrained.py). The checkpoint ships
fp16 with PyTorch `visual.*` key naming; we keep the names 1:1 (the Rust port mirrors them) and only
  * transpose the Conv2d patch-embed weight OIHW -> OHWI (MLX channels-last conv), and
  * drop the deterministic RoPE buffers (`rope.freqs_cos/sin`, recomputed in Rust + gate-checked).

`remap_visual` is shared with tools/dump_eva_clip_golden.py so the production weights and the parity
golden go through the *exact* same key/transpose logic.

The checkpoint is fp16-native; numpy (hence safetensors.numpy) has no bf16, so we ship f16 (the
native dtype) or f32. The Rust loader casts to bf16 at load if the generate path wants it.

Usage:
    /private/tmp/pulidenv/bin/python tools/convert_eva_clip.py [--out PATH] [--dtype f16|f32]
"""
import argparse
import hashlib
import os

import numpy as np

EVA_CKPT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--QuanSun--EVA-CLIP/snapshots/"
    "11afd202f2ae80869d6cef18b1ec775e79bd8d12/EVA02_CLIP_L_336_psz14_s6B.pt"
)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remap_visual(visual_sd: dict) -> dict:
    """torch `visual.*` state-dict (already stripped of the `visual.` prefix) -> mlx-gen numpy dict.

    Returns float32 numpy arrays under mlx-gen key names. The caller casts to the target dtype.
    Keys are kept identical to the torch module tree except:
      * `patch_embed.proj.weight`  [O,I,kH,kW] -> [O,kH,kW,I]  (MLX NHWC Conv2d).
      * `rope.*` and per-block `*.attn.rope.*` buffers are DROPPED (recomputed in Rust).
    """
    out = {}
    for k, v in visual_sd.items():
        if ".rope.freqs_" in k or k.startswith("rope.freqs_"):
            continue  # deterministic buffer, rebuilt in Rust (gate-checked vs this)
        a = v.detach().cpu().float().numpy()
        if k == "patch_embed.proj.weight":
            a = np.transpose(a, (0, 2, 3, 1))  # OIHW -> OHWI
        out[k] = np.ascontiguousarray(a)
    return out


def load_visual_state_dict() -> dict:
    """Load the raw checkpoint and return the `visual.`-stripped state dict (torch tensors).

    Loads with `weights_only=True` so a tampered third-party `.pt` cannot execute arbitrary pickle
    opcodes on the dev machine (F-152) — EVA02-CLIP ships a plain tensor state dict, so this is the
    normal path. If a checkpoint genuinely needs full unpickling, opt in explicitly by setting
    `EVA_CLIP_SHA256` to the file's verified SHA-256: the file is hashed and checked before the unsafe
    `weights_only=False` load, which refuses on mismatch (and refuses outright if no hash is set).
    """
    import torch

    try:
        sd = torch.load(EVA_CKPT, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001 — fall back only behind an explicit, hash-verified opt-in
        expected = os.environ.get("EVA_CLIP_SHA256")
        if not expected:
            raise SystemExit(
                f"convert_eva_clip: {EVA_CKPT} could not be loaded with weights_only=True ({e}).\n"
                "Refusing to unpickle a third-party checkpoint unsafely. If you trust this exact "
                "file, verify it and re-run with EVA_CLIP_SHA256=<sha256> to allow weights_only=False."
            ) from e
        actual = _sha256(EVA_CKPT)
        if actual.lower() != expected.lower():
            raise SystemExit(
                f"convert_eva_clip: SHA-256 mismatch for {EVA_CKPT}\n"
                f"  expected {expected}\n  actual   {actual}"
            )
        sd = torch.load(EVA_CKPT, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return {k[len("visual."):]: v for k, v in sd.items() if k.startswith("visual.")}


def main():
    from safetensors.numpy import save_file

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "eva02_clip_l336.safetensors"))
    ap.add_argument("--dtype", default="f16", choices=["f16", "f32"])
    args = ap.parse_args()

    np_cast = {"f16": np.float16, "f32": np.float32}[args.dtype]
    remapped = remap_visual(load_visual_state_dict())
    tensors = {k: v.astype(np_cast) for k, v in remapped.items()}
    save_file(tensors, args.out)
    print(f"wrote {len(tensors)} tensors ({args.dtype}) -> {args.out}")


if __name__ == "__main__":
    main()
