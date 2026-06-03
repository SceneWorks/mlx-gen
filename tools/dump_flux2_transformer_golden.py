"""Parity fixture for the FLUX.2 MMDiT transformer (sc-2346 S3) — a TINY synthetic config so it
commits cheaply and CI stays fast, while exercising every path: the double (joint img+txt)
block, the single (fused parallel attention+SwiGLU) block, shared per-stream modulation, the
4-axis interleaved RoPE, the sinusoidal time embedding, and the AdaLayerNormContinuous output.
Random weights, f32 throughout (`ModelConfig.precision = float32` so the fork's internal casts
stay f32 — the Rust port runs f32 activations).

Run from the mflux fork venv:
    cd ~/repos/mflux && .venv/bin/python ~/repos/mlx-gen/tools/dump_flux2_transformer_golden.py
"""

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.common.config.model_config import ModelConfig

ModelConfig.precision = mx.float32  # isolate port correctness from the fork's production bf16

from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer  # noqa: E402

from _paths import fixture  # noqa: E402

mx.random.seed(0)

IN_CH, HEADS, HEAD_DIM, JOINT = 4, 2, 8, 12  # inner = 16
t = Flux2Transformer(
    patch_size=1,
    in_channels=IN_CH,
    num_layers=1,
    num_single_layers=1,
    attention_head_dim=HEAD_DIM,
    num_attention_heads=HEADS,
    joint_attention_dim=JOINT,
    timestep_guidance_channels=16,
    mlp_ratio=3.0,
    axes_dims_rope=(2, 2, 2, 2),
    rope_theta=2000,
    guidance_embeds=False,
)

flat = tree_flatten(t.parameters())
new = []
for k, v in flat:
    if "norm_q" in k or "norm_k" in k or "norm_added" in k:
        new.append((k, (1.0 + 0.1 * mx.random.normal(v.shape)).astype(mx.float32)))
    else:
        new.append((k, (0.1 * mx.random.normal(v.shape)).astype(mx.float32)))
t.update(tree_unflatten(new))

# seq_img = 4 (2x2 grid), seq_txt = 3.
hidden = mx.random.normal((1, 4, IN_CH)).astype(mx.float32)
encoder = mx.random.normal((1, 3, JOINT)).astype(mx.float32)
img_ids = mx.array(
    np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0]], dtype=np.int32)
)
txt_ids = mx.array(np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]], dtype=np.int32))
timestep = 500.0

out = t(
    hidden_states=hidden,
    encoder_hidden_states=encoder,
    timestep=timestep,
    img_ids=img_ids,
    txt_ids=txt_ids,
    guidance=None,
)
mx.eval(out)

dump = {k: v.astype(mx.float32) for k, v in tree_flatten(t.parameters())}
dump["hidden"] = hidden
dump["encoder"] = encoder
dump["img_ids"] = img_ids.astype(mx.int32)
dump["txt_ids"] = txt_ids.astype(mx.int32)
dump["out"] = out.astype(mx.float32)

path = fixture("mlx-gen-flux2/tests/fixtures/transformer_golden.safetensors")
mx.save_safetensors(path, dump)
print(f"wrote {path} ({len(dump)} tensors)")
print(f"  out: {tuple(out.shape)}")
