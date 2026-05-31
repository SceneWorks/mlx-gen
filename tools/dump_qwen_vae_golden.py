"""Dump a Qwen-Image VAE parity golden for the Rust port (sc-2348, slice 1).

Loads the fork's `QwenVAE` with real weights, casts it to **float32**, and runs `encode` + `decode`
on fixed seeded inputs. Dumps the f32 weights (keyed by the fork's internal module tree, already in
MLX conv layout) + the inputs + the fork outputs to a single safetensors. The Rust test
(`mlx-gen-qwen-image/tests/vae_real_weights.rs`) loads it, runs its own f32 VAE, and compares —
isolating VAE *math* parity from the bf16/disk-loader concerns (the on-disk key remapping lands with
the full-model assembly later).

Run from the mflux fork venv:
    cd ~/repos/mflux && uv run python ~/repos/mlx-gen/tools/dump_qwen_vae_golden.py

Output (gitignored): ~/repos/mlx-gen/tools/golden/qwen_vae_golden.safetensors
"""

import os

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

from mflux.models.common.config import ModelConfig
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition

cfg = ModelConfig.qwen_image()
path = cfg.model_name

# Lazy-load all component weights; only the VAE mapping is applied, so transformer / text-encoder
# tensors stay unmaterialized (no big memory hit).
weights = WeightLoader.load(weight_definition=QwenWeightDefinition, model_path=path)
vae = QwenVAE()
WeightApplier.apply_and_quantize(
    weights=weights,
    quantize_arg=None,
    weight_definition=QwenWeightDefinition,
    models={"vae": vae},
)

# Force f32 so the Rust f32 path compares without bf16 noise.
vae.update(tree_map(lambda a: a.astype(mx.float32), vae.parameters()))

mx.random.seed(0)
# Small fixed inputs: 16×16 latent -> 128×128 image (decode); 64×64 image -> 8×8 latent (encode).
dec_in = mx.random.normal((1, 16, 16, 16)).astype(mx.float32)
enc_in = mx.random.normal((1, 3, 64, 64)).astype(mx.float32)

dec_out = vae.decode(dec_in)
enc_out = vae.encode(enc_in)
mx.eval(dec_out, enc_out)

# Key params by the fork's internal module tree (decoder.conv_in.conv3d.weight, …) — matches the
# Rust QwenVae::from_weights prefixes directly. Conv weights are already MLX-layout; norm gammas 1-D.
out = {k: v.astype(mx.float32) for k, v in tree_flatten(vae.parameters())}
out["dec_in"] = dec_in
out["dec_out"] = dec_out
out["enc_in"] = enc_in
out["enc_out"] = enc_out

golden_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(golden_dir, exist_ok=True)
path_out = os.path.join(golden_dir, "qwen_vae_golden.safetensors")
mx.save_safetensors(path_out, out)
print(f"dec_in={dec_in.shape} dec_out={dec_out.shape} enc_in={enc_in.shape} enc_out={enc_out.shape}")
print(f"wrote {path_out} ({len(out)} tensors)")
