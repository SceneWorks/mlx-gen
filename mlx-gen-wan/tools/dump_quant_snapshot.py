#!/usr/bin/env python3
"""Produce a **pre-quantized** Wan snapshot dir from a converted bf16 one (sc-2682 consume path).

Quantizes the transformer(s) — `_quantize_predicate`: attn `q/k/v/o` + `ffn.fc1/fc2`, the same
predicate the Rust `WanTransformer::quantize` uses — packs them to disk with `.scales`/`.biases`, and
writes `config.json` with a `quantization: {bits, group_size}` block. The shared f32 components
(`t5_encoder.safetensors`, `vae.safetensors`, `tokenizer.json`) are **symlinked** into the new dir
(not copied) so six quantized snapshots don't duplicate ~66 GB of T5. The Rust loader
(`WanTransformer::from_weights` reading `cfg.quantization` + `.scales`) consumes the result directly.

Reuses the reference `convert_wan._quantize_saved_model` (the exact `nn.quantize` path), so the packed
weights are byte-identical to what `loading.py` would load — and to what the Rust load-time quant
produces (the bf16 checkpoint is quantized by the same MLX op either way).

Usage:
    WAN_SRC=~/.cache/mlx-gen-models/wan2_2_t2v_a14b_mlx_bf16 \
    WAN_DST=~/.cache/mlx-gen-models/wan2_2_t2v_a14b_mlx_q4 \
    WAN_BITS=4 WAN_GROUP=64 \
    "$HOME/Library/Application Support/SceneWorks/python/venv/bin/python" \
        mlx-gen-wan/tools/dump_quant_snapshot.py
"""
import json
import os
import shutil
from pathlib import Path

from mlx_video.convert_wan import _quantize_saved_model
from mlx_video.models.wan.config import WanModelConfig

AUX_FILES = ("t5_encoder.safetensors", "vae.safetensors", "tokenizer.json")
TRANSFORMER_FILES = (
    "low_noise_model.safetensors",
    "high_noise_model.safetensors",
    "model.safetensors",
)


def build_config(cfg_json):
    fields = WanModelConfig.__dataclass_fields__
    cdict = {k: v for k, v in cfg_json.items() if k in fields}
    for key in ("patch_size", "vae_stride", "window_size", "sample_guide_scale"):
        if key in cdict and isinstance(cdict[key], list):
            cdict[key] = tuple(cdict[key])
    return WanModelConfig(**cdict)


def main():
    src = Path(os.path.expanduser(os.environ["WAN_SRC"]))
    dst = Path(os.path.expanduser(os.environ["WAN_DST"]))
    bits = int(os.environ.get("WAN_BITS", "4"))
    group = int(os.environ.get("WAN_GROUP", "64"))

    with open(src / "config.json") as f:
        cfg_json = json.load(f)
    if cfg_json.get("quantization"):
        raise SystemExit(f"{src} is already quantized ({cfg_json['quantization']}); use a bf16 source")
    config = build_config(cfg_json)
    is_dual = (src / "low_noise_model.safetensors").exists()

    dst.mkdir(parents=True, exist_ok=True)
    # Symlink the shared f32 components (avoid duplicating ~11 GB T5 per snapshot).
    for name in AUX_FILES:
        s = src / name
        if not s.exists():
            continue
        link = dst / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(s.resolve())
        print(f"  symlinked {name} -> {s.resolve()}")
    # config.json must exist in dst before _quantize_saved_model adds the quantization block.
    shutil.copy2(src / "config.json", dst / "config.json")

    print(f"Quantizing transformer(s) in {src.name} → {dst} ({bits}-bit, group {group}, dual={is_dual})...")
    _quantize_saved_model(dst, config, is_dual, bits, group, source_dir=src)

    with open(dst / "config.json") as f:
        out_cfg = json.load(f)
    assert out_cfg.get("quantization") == {"group_size": group, "bits": bits}, out_cfg.get("quantization")
    print(f"Done: {dst}  (quantization={out_cfg['quantization']})")


if __name__ == "__main__":
    main()
