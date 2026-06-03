"""Parity fixtures for the FLUX.2-klein S0 scaffold (sc-2346):
the flow-match schedule, 2×2 latent pack/unpack/patchify, the 4-axis RoPE table, and the
latent/text id builders. All pure math — no model weights — so the Rust port is parity-checked
tight (1e-4 for the trig RoPE, exact for the integer id builders, 1e-5 for the schedule).

Run from the mflux fork venv:
    cd ~/repos/mflux && .venv/bin/python ~/repos/mlx-gen/tools/dump_flux2_s0_golden.py
"""

import mlx.core as mx
import numpy as np

from mflux.models.flux2.model.flux2_transformer.pos_embed import Flux2PosEmbed
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder
from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import (
    FlowMatchEulerDiscreteScheduler,
)

from _paths import fixture

mx.random.seed(0)
out = {}

# --- Flow-match schedule (empirical-mu shift, no terminal stretch) ---
# The Config wiring for requires_sigma_shift models is exactly get_timesteps_and_sigmas.
SCHED_CONFIGS = [(256, 256, 4), (1024, 1024, 4), (1024, 560, 4), (512, 512, 20)]
for (w, h, steps) in SCHED_CONFIGS:
    seq = (h // 16) * (w // 16)
    timesteps, sigmas = FlowMatchEulerDiscreteScheduler.get_timesteps_and_sigmas(
        image_seq_len=seq, num_inference_steps=steps
    )
    tag = f"{w}x{h}x{steps}"
    out[f"sched.{tag}.sigmas"] = sigmas.astype(mx.float32)  # len steps+1, trailing 0
    out[f"sched.{tag}.timesteps"] = timesteps.astype(mx.float32)  # len steps, sigma*1000

# --- 4-axis RoPE (theta=2000, axes=(32,32,32,32)) over a mixed joint sequence ---
# 3 text tokens [0,0,0,k] + 6 latent tokens [0,h,w,0] (2x3 grid) + 2 edit-ref tokens [10,..].
rope_rows = []
for k in range(3):
    rope_rows.append([0, 0, 0, k])
for hh in range(2):
    for ww in range(3):
        rope_rows.append([0, hh, ww, 0])
rope_rows.append([10, 0, 0, 0])
rope_rows.append([10, 0, 1, 0])
ids_np = np.array([rope_rows], dtype=np.int32)  # [1, 11, 4]
ids_mx = mx.array(ids_np)
pos = Flux2PosEmbed(theta=2000, axes_dim=(32, 32, 32, 32))
cos, sin = pos(ids_mx)
out["rope.ids"] = ids_mx.astype(mx.int32)
out["rope.cos"] = cos.astype(mx.float32)  # [1, 11, 64]
out["rope.sin"] = sin.astype(mx.float32)

# --- 2×2 patchify: [1, C, H, W] -> [1, C*4, H/2, W/2] ---
patch_in = mx.random.normal((1, 2, 4, 6))
out["patch.in"] = patch_in.astype(mx.float32)
out["patch.out"] = Flux2LatentCreator.patchify_latents(patch_in).astype(mx.float32)  # [1,8,2,3]

# --- pack: [1, C, H, W] -> [1, H*W, C] ---
pack_in = mx.random.normal((1, 5, 3, 2))
out["pack.in"] = pack_in.astype(mx.float32)
out["pack.out"] = Flux2LatentCreator.pack_latents(pack_in).astype(mx.float32)  # [1,6,5]

# --- unpack: [1, seq, C] -> [1, C, lat_h, lat_w] (height=48, width=32 -> lat 3x2) ---
unpack_in = mx.random.normal((1, 6, 5))
out["unpack.in"] = unpack_in.astype(mx.float32)
out["unpack.out"] = Flux2LatentCreator.unpack_latents(
    unpack_in, height=48, width=32, vae_scale_factor=8
).astype(mx.float32)  # [1,5,3,2]

# --- latent grid ids: prepare_grid_ids over a [1,C,lat_h,lat_w]=[1,4,3,2] latent, t_coord=10 ---
grid_latents = mx.zeros((1, 4, 3, 2))
out["gridids.out"] = Flux2LatentCreator.prepare_grid_ids(grid_latents, t_coord=10).astype(mx.int32)  # [1,6,4]

# --- text ids: prepare_text_ids over a [1,seq,_]=[1,5,16] embedding ---
text_x = mx.zeros((1, 5, 16))
out["textids.out"] = Flux2PromptEncoder.prepare_text_ids(text_x).astype(mx.int32)  # [1,5,4]

path = fixture("mlx-gen-flux2/tests/fixtures/s0_golden.safetensors")
mx.save_safetensors(path, out)
print(f"wrote {path} ({len(out)} tensors)")
for k in sorted(out):
    print(f"  {k}: {tuple(out[k].shape)} {out[k].dtype}")
