#!/usr/bin/env python3
"""sc-3072 — PerceiverAttentionCA parity golden (torch f32 reference).

Loads the 20 `pulid_ca.{0..19}.*` modules from `pulid_flux_v0.9.1.safetensors` into the reference
`pulid.encoders_transformer.PerceiverAttentionCA`, runs a representative subset on fixed inputs
(`id_embedding` [1,32,2048], `img` [1,S,3072]), and dumps the f32 weights + inputs + per-module
outputs. The Rust test drives these through the `PulidCa` injector so the SAME case validates both
the CA math AND the double→single ca_idx schedule (double block i→ca[i/2]; single block i→ca[10+i/4]):

    after_double(0)  == ca[0]   after_double(18) == ca[9]
    after_single(0)  == ca[10]  after_single(36) == ca[19]

Run:
    cd /Users/michael/Repos/SceneWorks/apps/worker/scene_worker/_vendor/pulid_flux && \
      PYTHONPATH=. /private/tmp/pulidenv/bin/python \
      /path/to/mlx-gen/tools/dump_pulid_ca_golden.py
Output: tools/golden/pulid_ca_golden.safetensors
"""
import glob
import os

import numpy as np
import torch

OUT_DIR = os.path.join(os.path.dirname(__file__), "golden")
PULID_CKPT = glob.glob(
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--guozinan--PuLID/snapshots/*/pulid_flux_v0.9.1.safetensors"
    )
)[0]
# ca module indices exercised (cover both ends of the double + single schedules)
CA_INDICES = [0, 9, 10, 19]
SEQ = 64  # small image-token count for a fast golden


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from safetensors.numpy import save_file
    from safetensors.torch import load_file

    from pulid.encoders_transformer import PerceiverAttentionCA

    state = load_file(PULID_CKPT)
    out = {}
    # f32 weights for ALL 20 modules (the Rust PulidCa loads the full set)
    for k, v in state.items():
        if k.startswith("pulid_ca."):
            out[k] = v.float().numpy().astype(np.float32)

    g = torch.Generator().manual_seed(7)
    id_embedding = torch.randn(1, 32, 2048, generator=g, dtype=torch.float32)
    img = torch.randn(1, SEQ, 3072, generator=g, dtype=torch.float32)
    out["id_embedding"] = id_embedding.numpy().astype(np.float32)
    out["img"] = img.numpy().astype(np.float32)

    for i in CA_INDICES:
        mod = PerceiverAttentionCA().float().eval()
        sd = {
            k[len(f"pulid_ca.{i}."):]: v.float()
            for k, v in state.items()
            if k.startswith(f"pulid_ca.{i}.")
        }
        miss, unexp = mod.load_state_dict(sd, strict=True)
        assert not miss and not unexp, (i, miss, unexp)
        with torch.no_grad():
            res = mod(id_embedding, img)  # (x=id_embedding, latents=img)
        out[f"ca_out_{i}"] = res.numpy().astype(np.float32)

    path = os.path.join(OUT_DIR, "pulid_ca_golden.safetensors")
    save_file(out, path)
    n_w = sum(1 for k in out if k.startswith("pulid_ca."))
    print(f"wrote {len(out)} tensors ({n_w} weights) -> {path}")
    print("ca_out shape", out[f"ca_out_{CA_INDICES[0]}"].shape)


if __name__ == "__main__":
    main()
