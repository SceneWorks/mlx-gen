"""Tiny end-to-end denoise-loop parity fixture (sc-2344).

Run from the fork:  cd ~/repos/mflux && uv run python /Users/michael/repos/mlx-gen/tools/dump_denoise_loop.py

Reuses the tiny ZImageTransformer config from dump_z_transformer.py, then runs the flow-match
Euler loop by hand (mirroring z_image.py): for each step, predict velocity with the DiT and take
an Euler step. Validates the Rust loop ORCHESTRATION (timestep = 1 - sigma, scheduler stepping,
velocity sign) — the transformer/scheduler are independently parity-tested.
"""

import mlx.core as mx
from mlx.utils import tree_flatten
from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import (
    FlowMatchEulerDiscreteScheduler as S,
)
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer

OUT = "/Users/michael/repos/mlx-gen/mlx-gen-z-image/tests/fixtures/denoise_loop.safetensors"

mx.random.seed(0)
CFG = dict(
    patch_size=2, f_patch_size=1, in_channels=4, dim=96, n_layers=2, n_refiner_layers=1,
    n_heads=4, norm_eps=1e-5, qk_norm=True, cap_feat_dim=32, rope_theta=256.0, t_scale=1000.0,
    axes_dims=[8, 8, 8], axes_lens=[64, 64, 64],
)
model = ZImageTransformer(**CFG)
model.cap_embedder[0].weight = 1.0 + 0.1 * mx.random.normal(model.cap_embedder[0].weight.shape)

init = mx.random.normal((4, 1, 4, 4)).astype(mx.float32)  # (C=in_channels, F, H, W)
cap_feats = mx.random.normal((5, 32)).astype(mx.float32)

STEPS, MU = 3, 2.0
sigmas = mx.linspace(1.0, 1.0 / STEPS, STEPS)
sigmas = S._time_shift_exponential_array(MU, 1.0, sigmas)
sigmas = mx.concatenate([sigmas, mx.zeros((1,), dtype=sigmas.dtype)], axis=0).astype(mx.float32)

latents = init
for t in range(STEPS):
    ts = mx.array(1.0 - float(sigmas[t]), dtype=mx.float32)
    v = model(x=latents, timestep=ts, sigmas=sigmas, cap_feats=cap_feats)
    latents = latents + (sigmas[t + 1] - sigmas[t]) * v
mx.eval(latents)

out = {f"w.{k}": v.astype(mx.float32) for k, v in tree_flatten(model.parameters())}
out["init"] = init
out["cap_feats"] = cap_feats
out["sigmas"] = sigmas
out["final_latents"] = latents.astype(mx.float32)
meta = {"steps": str(STEPS)}

mx.save_safetensors(OUT, out, meta)
print(f"wrote {OUT}: steps={STEPS} final_shape={tuple(latents.shape)}")
