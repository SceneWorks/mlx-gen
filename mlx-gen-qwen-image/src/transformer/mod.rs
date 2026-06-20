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

use std::sync::atomic::{AtomicBool, Ordering};

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// sc-2963 (rollout of the Wan sc-2957 template): when on, the MMDiT's fusable elementwise *glue* —
/// adaLN affine (`x·(1+scale)+shift`), gated residual (`x+gate·y`), the tanh-GELU FFN activation, and
/// the complex RoPE rotation — runs through `mx.compile` so MLX fuses each chain into a single Metal
/// kernel (vs one kernel per primitive op when eager). The big GEMMs / SDPA / `mx.fast` norms stay
/// eager. **Bit-exact** to the eager form (`tests/compile_parity.rs` gates `max|Δ|=0`). **Enabled by
/// the production denoise loops** (T2I + Edit, [`crate::pipeline`]); left **off by default** so the
/// reference-parity gates run eager and `compile_parity` can A/B both. The dtype flow (bf16 weights,
/// f32 latents) is preserved — the compiled closures cast nothing the eager form didn't.
///
/// **Concurrency (F-087):** correct under the single-threaded MLX-device model — one generate runs on
/// the device at a time — so `Relaxed` suffices (no cross-thread ordering to establish). The flag is
/// process-global; the production denoise scopes it with [`CompileGlueGuard`] (F-006/F-007) so it is
/// not left stuck `true` after a generate. A future concurrent caller would still need `SeqCst` +
/// strict per-call scoping — revisit before adding one.
static COMPILE_GLUE: AtomicBool = AtomicBool::new(false);

/// Enable/disable compiled elementwise glue (sc-2963). Process-global; prefer the scoped
/// [`CompileGlueGuard`] in production (the raw setter is for the A/B `compile_parity`/`perf` gates).
pub fn set_compile_glue(on: bool) {
    COMPILE_GLUE.store(on, Ordering::Relaxed);
}

pub(crate) fn compile_glue() -> bool {
    COMPILE_GLUE.load(Ordering::Relaxed)
}

/// RAII guard (F-006/F-007, mirroring core `mlx_gen::nn::CompileGlueGuard` and z-image's) that enables
/// this crate's compiled elementwise glue for its lifetime and **restores the prior [`COMPILE_GLUE`]
/// value on drop** — even on an early `?`. The production denoise binds one across the render so the
/// toggle is scoped, not left stuck `true` process-wide (F-006), and same-process eager code (the
/// `compile_parity`/`perf` gates) sees the restored value.
#[must_use = "dropping the guard restores the prior compile-glue setting; bind it for the render's lifetime"]
pub(crate) struct CompileGlueGuard {
    prev: bool,
}

impl CompileGlueGuard {
    /// Turn compiled glue on, remembering the prior value to restore on drop.
    pub(crate) fn enable() -> Self {
        Self {
            prev: COMPILE_GLUE.swap(true, Ordering::Relaxed),
        }
    }
}

impl Drop for CompileGlueGuard {
    fn drop(&mut self) {
        COMPILE_GLUE.store(self.prev, Ordering::Relaxed);
    }
}

/// Load a Linear at `{prefix}.weight` (+ `{prefix}.bias` when `has_bias`) into an
/// [`AdaptableLinear`] — the dense-or-quantizable base every transformer Linear uses, so the whole
/// model can be Q8-quantized in place without touching the forward.
pub(crate) fn linear_from(w: &Weights, prefix: &str, has_bias: bool) -> Result<AdaptableLinear> {
    let weight = w.require(&format!("{prefix}.weight"))?.clone();
    let bias = if has_bias {
        Some(w.require(&format!("{prefix}.bias"))?.clone())
    } else {
        None
    };
    Ok(AdaptableLinear::dense(weight, bias))
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
