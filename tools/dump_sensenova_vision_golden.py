"""sc-3183: synthetic-fixture golden for the SenseNova-U1 NEO vision embedder.

The 8B-MoT "vision tower" is an embedder, not a transformer: a full-kernel `patch_embedding`
(Conv2d 3->hidden, kernel=stride=patch_size) + GELU, interleaved 2D RoPE over the patch grid, then a
2x2-strided `dense_embedding` (Conv2d hidden->llm_hidden) patch-merge. The same module backs both the
understanding-path `vision_model` and the generation-path `fm_modules.vision_model_mot_gen`.

Builds a tiny `NEOVisionConfig`, runs the reference `NEOVisionModel` on a small patch grid, and dumps
weights + input + output for the Rust parity test. float32.

Run:
  cd _vendor/sensenova_u1 && PYTHONPATH=src .venv/bin/python ../../tools/dump_sensenova_vision_golden.py
Fixture -> mlx-gen-sensenova/tests/fixtures/vision_golden.safetensors
"""

from __future__ import annotations

import os
import sys

import torch
from safetensors.torch import save_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from sensenova_u1.models.neo_unify.configuration_neo_vit import NEOVisionConfig
from sensenova_u1.models.neo_unify.modeling_neo_vit import NEOVisionModel


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    patch = 4
    cfg = NEOVisionConfig(
        num_channels=3,
        patch_size=patch,
        hidden_size=64,
        llm_hidden_size=128,
        downsample_ratio=0.5,  # -> downsample_factor 2 (2x2 patch merge)
        rope_theta_vision=10000.0,
        max_position_embeddings_vision=4096,
    )
    model = NEOVisionModel(cfg).to(torch.float32).eval()

    # One image: a 4x4 patch grid (16 patches) -> dense merge -> 2x2 = 4 tokens.
    grid_hw = torch.tensor([[4, 4]], dtype=torch.long)
    n_patches = int((grid_hw[:, 0] * grid_hw[:, 1]).sum())
    pixel_values = torch.randn(n_patches, 3 * patch * patch, dtype=torch.float32)

    out = model(pixel_values=pixel_values, grid_hw=grid_hw).last_hidden_state

    tensors = {}
    for k, v in model.state_dict().items():
        # Skip the non-persistent rope buffers (registered persistent=False, but be defensive).
        if "cached" in k:
            continue
        tensors[f"vision_model.{k}"] = v.contiguous().to(torch.float32)
    tensors["input.pixel_values"] = pixel_values
    tensors["input.grid_hw"] = grid_hw.to(torch.int32)
    tensors["vis.embeds"] = out.to(torch.float32)

    dst = os.path.join(
        REPO_ROOT, "mlx-gen-sensenova", "tests", "fixtures", "vision_golden.safetensors"
    )
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    meta = {
        "hidden_size": str(cfg.hidden_size),
        "llm_hidden_size": str(cfg.llm_hidden_size[0]),
        "num_channels": str(cfg.num_channels),
        "patch_size": str(cfg.patch_size),
        "downsample_ratio": repr(cfg.downsample_ratio[0]),
        "rope_theta_vision": repr(cfg.rope_theta_vision),
    }
    save_file(tensors, dst, metadata=meta)
    print(f"wrote {dst}")
    print(f"  pixel_values {tuple(pixel_values.shape)}  grid {grid_hw.tolist()}  embeds {tuple(out.shape)}")
    print(f"  tensors: {len(tensors)}")


if __name__ == "__main__":
    sys.exit(main())
