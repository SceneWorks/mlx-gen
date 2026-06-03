"""Real-weights golden for the FLUX.2 VAE (sc-2346 S2), for the #[ignore]d Rust parity test.
Loads the real `vae/` shards (cast to f32 — the Rust port's precision), then dumps:
  - decode_packed_latents: a seeded packed latent [1,128,4,4] (NCHW) → image [1,3,64,64],
  - encode: a seeded image [1,3,64,64] → latent mean [1,32,8,8].
Tensors are NCHW (fork-native); the Rust test transposes to NHWC.

Gitignored output. Run from the mflux fork venv:
    cd ~/repos/mflux && .venv/bin/python ~/repos/mlx-gen/tools/dump_flux2_vae_golden.py
"""

import glob

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE

from _paths import fixture, hf_hub_cache

SNAP = str(
    next((hf_hub_cache() / "models--black-forest-labs--FLUX.2-klein-9b" / "snapshots").glob("*"))
)

# Build the VAE and load the raw checkpoint into it (transpose rank-4 conv weights to mlx
# [O,H,W,I], rename the Sequential to_out.0 → to_out), cast everything to f32.
vae = Flux2VAE()
module_keys = {k for k, _ in tree_flatten(vae.parameters())}
params = {}
for f in glob.glob(f"{SNAP}/vae/*.safetensors"):
    for k, v in mx.load(f).items():
        key = k.replace(".to_out.0.", ".to_out.")
        if v.ndim == 4:  # conv weight [O,I,H,W] -> [O,H,W,I]
            v = mx.transpose(v, (0, 2, 3, 1))
        if key in module_keys:
            params[key] = v.astype(mx.float32)
vae.update(tree_unflatten(list(params.items())))

mx.random.seed(0)
packed_in = mx.random.normal((1, 128, 4, 4)).astype(mx.float32)
image_in = mx.random.normal((1, 3, 64, 64)).astype(mx.float32)

decoded = vae.decode_packed_latents(packed_in)
encoded = vae.encode(image_in)
mx.eval(decoded, encoded)

out = {
    "packed_in": packed_in,  # NCHW [1,128,4,4]
    "decode_out": decoded.astype(mx.float32),  # NCHW [1,3,64,64]
    "image_in": image_in,  # NCHW [1,3,64,64]
    "encode_out": encoded.astype(mx.float32),  # NCHW [1,32,8,8]
}
path = fixture("tools/golden/flux2_vae.safetensors")
mx.save_safetensors(path, out)
print(f"wrote {path}")
print(f"  decode_out: {tuple(decoded.shape)}  encode_out: {tuple(encoded.shape)}")
print(f"  loaded {len(params)}/{len(module_keys)} module params")
