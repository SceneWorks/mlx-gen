#!/usr/bin/env python
"""Dump the qwenimage `clean_latent` (QwenImage_VAE_2d encode) for the mlx-gen-pid e2e decode.

Loads ONLY the small Qwen-Image 2D VAE (`QwenImageVAE2d`, ~100 MB Wan-2D conv VAE) — not the 1.36 B
net or the gemma encoder — on CPU in f32, and encodes each runA `from_clean` input at a downscaled
(512→64²) and the native (1024→128²) resolution. The VAE is the one piece of the from_clean path not
yet in Rust (a different 2D-conv VAE than mlx-gen-qwen-image's 3D QwenVae); MLX consumes these latents
for the decode. One safetensors per sample → `clean_latent_<name>.safetensors`.

Run: /Users/michael/Repos/mlx-gen/_vendor/pid/.venv-pid/bin/python tools/dump_pid_clean_latent.py
"""

import os
import sys

import torch
import torch.nn.functional as F

# torch.load: the .pth is CUDA-saved — force CPU.
_torch_load = torch.load
def _patched_load(*a, **k):
    k["map_location"] = "cpu"
    k.setdefault("weights_only", False)
    return _torch_load(*a, **k)
torch.load = _patched_load
# imaginaire utils poke torch.cuda even on CPU runs.
for _n in ("empty_cache", "synchronize", "set_device"):
    setattr(torch.cuda, _n, lambda *a, **k: None)
torch.cuda.is_available = lambda: False

PID_ROOT = "/Users/michael/Repos/mlx-gen/_vendor/pid"
sys.path.insert(0, PID_ROOT)
os.chdir(PID_ROOT)

from safetensors.torch import save_file  # noqa: E402
from pid._src.tokenizers.qwenimage_vae import QwenImageVAE2d  # noqa: E402
from pid._src.inference.inference_utils import load_input_image  # noqa: E402

SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--PiD/snapshots/b4887b3c8fc65277a9b7a084292bf9fc0778c5ac")
VAE_PTH = f"{SNAP}/checkpoints/QwenImage_VAE_2d.pth"
RUNA = os.path.expanduser("~/pid-validate-samples/01_runA_from_clean")
OUT_DIR = "/Users/michael/Repos/mlx-gen/.claude/worktrees/dazzling-gauss-61cef9/tools/golden/pid"

# The three runA `from_clean` samples (each `<name>__input__1024.png`).
SAMPLES = ["landscape", "portrait", "text_storefront"]


def main():
    print("building Qwen-Image 2D VAE (cpu, f32)...", flush=True)
    vae = QwenImageVAE2d(z_dim=16, vae_pth=VAE_PTH, dtype=torch.float32, device="cpu", is_amp=False)
    os.makedirs(OUT_DIR, exist_ok=True)

    for name in SAMPLES:
        inp = f"{RUNA}/{name}__input__1024.png"
        img = load_input_image(inp).to(dtype=torch.float32, device="cpu")  # [1,3,1024,1024] in [-1,1]
        tensors = {}
        with torch.no_grad():
            # 512 -> latent 64 -> PiD 2048 (low end of the 2k->4k regime; tractable for a quick check).
            img_small = F.interpolate(img, size=(512, 512), mode="bicubic", align_corners=False).clamp(-1, 1)
            tensors["clean_latent_small"] = vae.encode(img_small).float().contiguous()  # [1,16,64,64]
            tensors["clean_latent_native"] = vae.encode(img).float().contiguous()       # [1,16,128,128]
        out = f"{OUT_DIR}/clean_latent_{name}.safetensors"
        save_file(tensors, out)
        zn = tensors["clean_latent_native"]
        print(f"{name}: native {tuple(zn.shape)} mean={zn.mean():.4f} std={zn.std():.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
