"""sc-2532 packing-parity golden: prove the Rust port's Q8 quantization is byte-identical to the
fork on a REAL bf16 model weight (the committed `quant_parity.rs` fixture covers a synthetic f32
weight; the model quantizes bf16). Dumps `layers.0.attention.to_q`'s dense bf16 weight, a fixed
bf16 input, and the fork's `mx.quantize`/`mx.quantized_matmul` reference — consumed by
`mlx-gen-z-image/tests/e2e_real_weights.rs::q8_packing_byte_identical_to_fork`.

Run from the mflux fork venv (single dense load):
    cd ~/repos/mflux && uv run python /path/to/mlx-gen/tools/dump_z_image_q8_pack_probe.py
Output (gitignored): tools/golden/zq8_pack_probe.safetensors
"""

import os
import mlx.core as mx
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.z_image.z_image_initializer import ZImageInitializer

_GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(_GOLDEN_DIR, exist_ok=True)
OUT = os.path.join(_GOLDEN_DIR, "zq8_pack_probe.safetensors")


class Holder:
    pass


model = Holder()
ZImageInitializer.init(model, model_config=ModelConfig.z_image_turbo(), quantize=None)
w = model.transformer.layers[0].attention.to_q.weight  # bf16 [out, in]

mx.random.seed(0)
x = mx.random.normal((4, w.shape[1])).astype(mx.bfloat16)

wq, scales, biases = mx.quantize(w, group_size=64, bits=8)
qmm = mx.quantized_matmul(x, wq, scales, biases, transpose=True, group_size=64, bits=8)

mx.save_safetensors(
    OUT,
    {
        "w": w.astype(mx.float32),  # bf16 values widened to f32 (exact); Rust casts back to bf16
        "x": x.astype(mx.float32),
        "wq": wq,
        "scales": scales.astype(mx.float32),
        "biases": biases.astype(mx.float32),
        "qmm": qmm.astype(mx.float32),
    },
)
print(f"wrote {OUT}: w{tuple(w.shape)}/{w.dtype} wq{tuple(wq.shape)}/{wq.dtype}")
