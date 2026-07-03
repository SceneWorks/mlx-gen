//! Qwen-Image **60-layer dual-stream MMDiT** transformer. Port of the fork's `QwenTransformer`.
//!
//! Dual-stream: image and text tokens carry separate AdaLN modulation (from the timestep
//! embedding) and feed-forward, but attend **jointly** over the concatenated `[txt, img]` sequence.
//! Uses interleaved-complex 3D RoPE ([`rope`]), per-head q/k RMSNorm, affine-less LayerNorms, and
//! `AdaLayerNormContinuous` at the output. NCS (batch, seq, dim) tensors throughout.

pub mod attention;
pub mod block;
pub mod feed_forward;
pub mod norm_out;
pub mod rope;
pub mod time_text_embed;
pub mod timesteps;
#[allow(clippy::module_inception)]
pub mod transformer;

pub use attention::QwenJointAttention;
pub use block::QwenTransformerBlock;
pub use feed_forward::FeedForward;
pub use norm_out::AdaLayerNormContinuous;
pub use rope::QwenRope3d;
pub use transformer::{QwenTransformer, QwenTransformerConfig};

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

// sc-2963 compiled-glue toggle (rollout of the Wan sc-2957 template): when on, the MMDiT's fusable
// elementwise *glue* — adaLN affine (`x·(1+scale)+shift`), gated residual (`x+gate·y`), the tanh-GELU
// FFN activation, and the complex RoPE rotation — runs through `mx.compile` so MLX fuses each chain
// into a single Metal kernel. The big GEMMs / SDPA / `mx.fast` norms stay eager. **Bit-exact** to the
// eager form (`tests/compile_parity.rs` gates `max|Δ|=0`). **Enabled by the production denoise loops**
// (T2I + Edit, [`crate::pipeline`]); left **off by default** so the reference-parity gates run eager.
// The dtype flow (bf16 weights, f32 latents) is preserved unchanged.
//
// The toggle + its RAII [`CompileGlueGuard`] are hoisted into core (F-104); re-export core's so the
// process-global is shared with the FLUX family rather than each crate hand-rolling its own `AtomicBool`.
pub(crate) use mlx_gen::nn::compile_glue;
pub use mlx_gen::nn::{set_compile_glue, CompileGlueGuard};

/// Load a Linear at `{prefix}.weight` (+ `{prefix}.bias` when `has_bias`) into an
/// [`AdaptableLinear`] — the dense-or-quantizable base every transformer Linear uses, so the whole
/// model can be Q8-quantized in place without touching the forward. Routes through
/// [`crate::quant::lin`], which **auto-detects** a pre-quantized (packed) snapshot via
/// `{prefix}.scales` and loads it packed with no dense transient (sc-8670); a dense snapshot loads
/// dense exactly as before. The post-load `transformer.quantize(bits)` is a no-op on an
/// already-packed module, so the same path serves both pre-quantized tiers and dense control loads.
pub(crate) fn linear_from(w: &Weights, prefix: &str, has_bias: bool) -> Result<AdaptableLinear> {
    crate::quant::lin(w, prefix, has_bias)
}

/// Join a module prefix with a leaf name, tolerating an empty prefix.
pub(crate) fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

/// sc-2963 shared helpers for the per-module compiled-glue bit-exactness tests (each submodule's
/// private compiled chain — `modulate`/`gated` in [`block`], `gelu_ffn` in [`feed_forward`],
/// `rope_rotate` in [`attention`] — is gated `max|Δ|=0` compiled-vs-eager at its real dtypes).
#[cfg(test)]
pub(crate) mod compile_test_util {
    use mlx_rs::{random, Array, Dtype};

    pub(crate) fn rnd(shape: &[i32], dt: Dtype) -> Array {
        let k = random::key(0).unwrap();
        let x = random::normal::<f32>(shape, None, None, Some(&k)).unwrap();
        let x = if dt == Dtype::Float32 {
            x
        } else {
            x.as_dtype(dt).unwrap()
        };
        mlx_rs::transforms::eval([&x]).unwrap();
        x
    }

    pub(crate) fn max_abs(a: &Array, b: &Array) -> f32 {
        let d = mlx_rs::ops::abs(mlx_rs::ops::subtract(a, b).unwrap()).unwrap();
        mlx_rs::ops::max(&d, None)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap()
            .item::<f32>()
    }
}
