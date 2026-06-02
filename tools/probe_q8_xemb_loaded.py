"""sc-2604 decisive probe: separate the base-z_image Q8 per-op residual into KERNEL vs QUANTIZE vs
WRAPPER/GRAPH vs STRIDING — first entirely *within the fork build*, then dumping everything the Rust
diagnostic needs for the cross-build comparison.

Loads the real base `ZImage(quantize=8)` and, for the x-embedder (K=64, M=4096, the stage the sc-2349
bisection found diverging ~0.3%) and the t-embedder (M=1 gemv, never isolated), captures:
  - the loaded QuantizedLinear bytes (wq/scales/biases/bias)
  - a FRESH mx.quantize of the same raw weight (does loaded == fresh?)
  - the in-model linear forward (QuantizedLinear.__call__)
  - the bare mx.quantized_matmul on the SAME inputs (does in-model == bare?)
  - bare qmm on a forced-CONTIGUOUS copy of the (transpose-derived) embedder input (does striding
    change the kernel result within one build?)

Within-fork conclusions are printed; the f32/uint32 tensors are dumped so a Rust diag can repeat the
exact comparison on the source-built MLX and localize the Rust-vs-fork gap.

Run: cd ~/Repos/mflux-sc2257 && uv run python ~/Repos/mlx-gen/tools/probe_q8_xemb_loaded.py
"""

import glob
import os

import mlx.core as mx
import numpy as np

from _paths import hf_hub_cache
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
from mflux.models.z_image.z_image_initializer import ZImageInitializer

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(D, exist_ok=True)
SEED, H, W = 42, 1024, 1024


def rel(a, b):
    a = np.asarray(a.astype(mx.float32)).ravel()
    b = np.asarray(b.astype(mx.float32)).ravel()
    peak = max(float(np.max(np.abs(b))), 1e-12)
    mabs = max(float(np.mean(np.abs(b))), 1e-12)
    d = np.abs(a - b)
    return float(np.max(d)) / peak, float(np.mean(d)) / mabs


class Holder:
    pass


model = Holder()
ZImageInitializer.init(model, model_config=ModelConfig.z_image_turbo(), quantize=8)
t = model.transformer
key = f"{t.patch_size}-{t.f_patch_size}"
emb = t.all_x_embedder[key]
gs, bits = emb.group_size, emb.bits
print(f"x-embedder key={key} group_size={gs} bits={bits}")

# ---- the embedder input: patchify the seeded init exactly like ZImageTransformer._patchify ----
init = ZImageLatentCreator.create_noise(SEED, H, W).astype(mx.float32)  # [16,1,128,128]
C, F, Hh, Ww = init.shape
pf, ph, pw = t.f_patch_size, t.patch_size, t.patch_size
Ft, Ht, Wt = F // pf, Hh // ph, Ww // pw
x_tokens = (
    init.reshape(C, Ft, pf, Ht, ph, Wt, pw)
    .transpose(1, 3, 5, 2, 4, 6, 0)
    .reshape(Ft * Ht * Wt, pf * ph * pw * C)
)  # [4096, 64], transpose-derived (strided)
mx.eval(x_tokens)
x_contig = mx.array(np.array(x_tokens))  # forced contiguous copy, identical values
print(f"x_tokens {tuple(x_tokens.shape)} dtype={x_tokens.dtype}")

# ---- loaded quantized bytes ----
wq, scales, biases, bias = emb.weight, emb.scales, emb.biases, emb.bias

# ---- FRESH quantize of the same raw weight (from the snapshot) ----
tdir = glob.glob(str(hf_hub_cache() / "models--Tongyi-MAI--Z-Image-Turbo" / "snapshots/*/transformer"))[0]
w_raw = None
for f in glob.glob(f"{tdir}/*.safetensors"):
    d = mx.load(f)
    k = f"all_x_embedder.{key}.weight"
    if k in d:
        w_raw = d[k].astype(mx.bfloat16)
        break
assert w_raw is not None, "raw x-embedder weight not found"
fwq, fscales, fbiases = mx.quantize(w_raw, group_size=gs, bits=bits)
mx.eval(fwq, fscales, fbiases)

# ---- forwards ----
y_inmodel = emb(x_tokens)  # QuantizedLinear.__call__
y_bare = mx.quantized_matmul(x_tokens, wq, scales, biases, transpose=True, group_size=gs, bits=bits) + bias
y_bare_nobias = mx.quantized_matmul(x_tokens, wq, scales, biases, transpose=True, group_size=gs, bits=bits)
y_bare_contig = mx.quantized_matmul(x_contig, wq, scales, biases, transpose=True, group_size=gs, bits=bits) + bias
mx.eval(y_inmodel, y_bare, y_bare_nobias, y_bare_contig)

print("\n=== WITHIN-FORK (single build) ===")
wq_same = bool(mx.all(wq == fwq).item())
print(f"loaded wq == fresh mx.quantize wq : {wq_same}")
print(f"loaded vs fresh scales  peak/mean_rel : {rel(scales, fscales)}")
print(f"loaded vs fresh biases  peak/mean_rel : {rel(biases, fbiases)}")
print(f"in-model emb(x) vs bare qmm+bias      : {rel(y_inmodel, y_bare)}   (0 ⇒ __call__ is just qmm+bias)")
print(f"bare strided vs bare contiguous       : {rel(y_bare, y_bare_contig)} (0 ⇒ striding irrelevant within build)")

# ---- t-embedder (M=1 gemv) ----
te = t.t_embedder
ts = mx.array([1.0 - 0.0], dtype=mx.float32) * 1000.0  # representative timestep*t_scale
# fork TimestepEmbedder.__call__ path: timestep_embedding -> linear1 -> silu -> linear2
t_freq = te.timestep_embedding(ts, te.frequency_embedding_size) if hasattr(te, "timestep_embedding") else None
t_emb_inmodel = te(ts)
mx.eval(t_emb_inmodel)
if t_freq is not None:
    mx.eval(t_freq)

out = {
    "init": init,
    "x_tokens": x_tokens.astype(mx.float32),
    "w_raw": w_raw.astype(mx.float32),
    "xe_wq": wq,
    "xe_scales": scales.astype(mx.float32),
    "xe_biases": biases.astype(mx.float32),
    "xe_bias": bias.astype(mx.float32),
    "xe_inmodel": y_inmodel.astype(mx.float32),
    "xe_bare": y_bare.astype(mx.float32),
    "xe_bare_nobias": y_bare_nobias.astype(mx.float32),
    "ts": ts.astype(mx.float32),
    "t_emb_inmodel": t_emb_inmodel.astype(mx.float32),
}
if t_freq is not None:
    out["t_freq"] = t_freq.astype(mx.float32)
path = os.path.join(D, "q8_xemb_probe.safetensors")
mx.save_safetensors(path, out, {"group_size": str(gs), "bits": str(bits), "seed": str(SEED), "h": str(H), "w": str(W)})
print(f"\nwrote {path} ({len(out)} tensors)")
