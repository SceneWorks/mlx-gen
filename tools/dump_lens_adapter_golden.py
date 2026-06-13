#!/usr/bin/env python
"""Dump LoRA + LoKr adapter goldens for the Lens DiT (mlx-gen sc-3174).

Builds **deterministic synthetic** PEFT adapters (LoRA, then LoKr) on the authoritative vendor
`LensTransformer2DModel`, targeting the exact trainer modules
(`lens_train_runner.DEFAULT_LORA_TARGET_MODULES` = `img_qkv` / `txt_qkv` / `to_out.0` / `to_add_out`),
saves each in the **on-disk format the trainer ships** (diffusers `save_lora_adapter` for LoRA;
`get_peft_model_state_dict` + `networkType=lokr` metadata for LoKr — the same code paths as
`lens_train_runner`), and dumps the base + adapter-applied DiT outputs over fixed synthetic inputs.

The Rust gate (`tests/adapter_parity.rs`) loads the SAME adapter files via `apply_lens_adapters` and
asserts the applied DiT matches these outputs (f32, tight — LoRA/LoKr is a linear-merge delta), and
that a scale-0 apply is a bit-exact no-op.

Run from the reference venv (peft 0.19.1; loads the ~16 GB f32 transformer):
  ~/Repos/mflux/.venv/bin/python tools/dump_lens_adapter_golden.py
Writes (gitignored) under tools/golden/:
  lens_lora_adapter.safetensors, lens_lokr_adapter.safetensors, lens_adapter_golden.safetensors
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os

import peft
import torch
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file

HOME = os.path.expanduser("~")
SNAP_GLOB = f"{HOME}/.cache/huggingface/hub/models--microsoft--Lens-Turbo/snapshots/*/transformer"
VENDOR_T = os.path.expanduser(
    "~/Repos/SceneWorks/apps/worker/scene_worker/_vendor/lens/transformer.py"
)
GOLD = os.path.join(os.path.dirname(__file__), "golden")

FRAME, H_LAT, W_LAT = 1, 16, 16
TXT_LEN = 120
TARGET_MODULES = ["img_qkv", "txt_qkv", "to_out.0", "to_add_out"]
RANK = 8
ALPHA = 8  # alpha/rank = 1 → internal scaling 1.0 (the Rust apply uses scale 1.0 to match)
DECOMPOSE_FACTOR = -1


def load_model_cls():
    spec = importlib.util.spec_from_file_location("lens_transformer", VENDOR_T)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LensTransformer2DModel


def randomize_adapter(model, seed: int, std: float) -> None:
    """Overwrite the freshly-attached adapter params with seeded gaussians so the delta is non-zero
    (PEFT inits LoRA-B / one LoKr factor to zero → a no-op delta otherwise)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_" in name or "lokr_" in name or "hada_" in name:
                p.copy_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * std)


def main() -> None:
    matches = sorted(glob.glob(SNAP_GLOB))
    if not matches:
        raise SystemExit(f"no Lens-Turbo transformer snapshot at {SNAP_GLOB}")
    tdir = matches[-1]
    os.makedirs(GOLD, exist_ok=True)

    LensTransformer2DModel = load_model_cls()
    print("loading transformer (f32, CPU)…", flush=True)
    model = (
        LensTransformer2DModel.from_pretrained(tdir, torch_dtype=torch.float32).to("cpu").eval()
    )

    img_len = FRAME * H_LAT * W_LAT
    n_text = len(model.config.selected_layer_index)
    enc_dim = model.config.enc_hidden_dim

    torch.manual_seed(0)
    hidden_states = torch.randn(1, img_len, model.config.in_channels, dtype=torch.float32)
    feats = [torch.randn(1, TXT_LEN, enc_dim, dtype=torch.float32) for _ in range(n_text)]
    timestep = torch.rand(1, dtype=torch.float32)
    text_mask = torch.ones(1, TXT_LEN, dtype=torch.bool)
    img_shapes = [(FRAME, H_LAT, W_LAT)]

    def forward():
        with torch.no_grad():
            return model(hidden_states, feats, text_mask, timestep, img_shapes)

    tensors = {
        "hidden_states": hidden_states.contiguous(),
        "timestep": timestep.contiguous(),
        "base_out": forward().contiguous(),
    }
    for i, f in enumerate(feats):
        tensors[f"feat_{i}"] = f.contiguous()
    print("base forward done", flush=True)

    # --- LoRA: attach (gaussian), randomize B, save via diffusers, forward ---
    lora_cfg = peft.LoraConfig(
        r=RANK, lora_alpha=ALPHA, init_lora_weights="gaussian", target_modules=TARGET_MODULES
    )
    model.add_adapter(lora_cfg)
    randomize_adapter(model, seed=20260613, std=0.02)
    model.save_lora_adapter(GOLD, weight_name="lens_lora_adapter.safetensors", safe_serialization=True)
    tensors["lora_out"] = forward().contiguous()
    model.delete_adapters(model.active_adapters())
    print("LoRA done", flush=True)

    # --- LoKr: attach, randomize, save via get_peft_model_state_dict + metadata, forward ---
    lokr_cfg = peft.LoKrConfig(
        r=RANK,
        alpha=ALPHA,
        decompose_factor=DECOMPOSE_FACTOR,
        init_weights=True,
        target_modules=TARGET_MODULES,
    )
    model.add_adapter(lokr_cfg)
    randomize_adapter(model, seed=20260614, std=0.05)
    lokr_state = {k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(model).items()}
    save_file(
        lokr_state,
        os.path.join(GOLD, "lens_lokr_adapter.safetensors"),
        metadata={
            "format": "pt",
            "networkType": "lokr",
            "rank": str(RANK),
            "alpha": str(ALPHA),
            "decomposeFactor": str(DECOMPOSE_FACTOR),
            "targetModules": json.dumps(TARGET_MODULES),
        },
    )
    tensors["lokr_out"] = forward().contiguous()
    print("LoKr done", flush=True)

    meta = {
        "frame": str(FRAME), "h_lat": str(H_LAT), "w_lat": str(W_LAT),
        "txt_len": str(TXT_LEN), "img_len": str(img_len), "n_text": str(n_text),
        "rank": str(RANK), "alpha": str(ALPHA),
    }
    out = os.path.join(GOLD, "lens_adapter_golden.safetensors")
    save_file(tensors, out, metadata=meta)
    print(f"wrote {out} + lens_lora_adapter.safetensors + lens_lokr_adapter.safetensors")


if __name__ == "__main__":
    main()
