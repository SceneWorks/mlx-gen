#!/usr/bin/env python3
"""sc-3071 — IDFormer perceiver-resampler parity golden (torch f32 reference).

Loads the `pulid_encoder.*` weights from `guozinan/PuLID/pulid_flux_v0.9.1.safetensors` into the
reference `pulid.encoders_transformer.IDFormer`, runs it in float32 on deterministic inputs
(`id_cond` [1,1280], 5 EVA hidden states [1,577,1024]), and dumps the f32 weights + inputs +
`id_embedding` [1,32,2048] for the Rust parity test.

Run:
    cd /Users/michael/Repos/SceneWorks/apps/worker/scene_worker/_vendor/pulid_flux && \
      PYTHONPATH=. /private/tmp/pulidenv/bin/python \
      /Users/michael/Repos/mlx-gen/.claude/worktrees/objective-snyder-2a2b40/tools/dump_idformer_golden.py
Output: tools/golden/idformer_golden.safetensors
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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from safetensors.numpy import save_file
    from safetensors.torch import load_file

    from pulid.encoders_transformer import IDFormer

    model = IDFormer().float().eval()
    state = load_file(PULID_CKPT)
    enc = {k[len("pulid_encoder."):]: v.float() for k, v in state.items() if k.startswith("pulid_encoder.")}
    missing, unexpected = model.load_state_dict(enc, strict=True)
    assert not missing and not unexpected, (missing, unexpected)

    g = torch.Generator().manual_seed(2025)
    id_cond = torch.randn(1, 1280, generator=g, dtype=torch.float32)
    hidden = [torch.randn(1, 577, 1024, generator=g, dtype=torch.float32) for _ in range(5)]
    with torch.no_grad():
        id_embedding = model(id_cond, hidden)
    assert tuple(id_embedding.shape) == (1, 32, 2048), id_embedding.shape

    out = {}
    for k, v in model.state_dict().items():
        out[f"pulid_encoder.{k}"] = v.float().numpy().astype(np.float32)
    out["id_cond"] = id_cond.numpy().astype(np.float32)
    for i, h in enumerate(hidden):
        out[f"hidden_{i}"] = h.numpy().astype(np.float32)
    out["id_embedding"] = id_embedding.numpy().astype(np.float32)

    path = os.path.join(OUT_DIR, "idformer_golden.safetensors")
    save_file(out, path)
    n_w = sum(1 for k in out if k.startswith("pulid_encoder."))
    print(f"wrote {len(out)} tensors ({n_w} weights) -> {path}")
    print("id_embedding", out["id_embedding"].shape)


if __name__ == "__main__":
    main()
