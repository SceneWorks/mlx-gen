//! Packed (pre-quantized) weight loading — the consume side of [`crate::convert`].
//!
//! A pre-quantized Q4/Q8 snapshot stores each quantized Linear as the packed triple
//! `{base}.weight` (u32 codes) + `{base}.scales` + `{base}.biases`. The [`lin`] loader
//! **auto-detects** it by the presence of `{base}.scales` and builds the quantized module directly —
//! so a published Q4 snapshot loads packed with no dense bf16 transient. A dense snapshot (no
//! `.scales`) loads dense exactly as before, so the same loader serves both.
//!
//! FLUX.1 quantizes all three components — the DiT transformer, the T5 + CLIP text encoders, and the
//! (shared Z-Image) VAE's mid-block attention — so every quantizable Linear across the transformer +
//! text-encoder `*_from_weights` paths routes through here. The VAE is the shared
//! `mlx_gen_z_image::vae::Vae`, already packed-aware via its own `crate::quant::lin` (sc-8670). The
//! text encoders' token/position/relative-bias embeddings use the crate-local `TokenEmbedding` enum,
//! which grows its own packed-detect `from_weights` (it is not the shared `mlx_gen::nn` type). This
//! is the Group-B per-crate template (sc-8669); a thin wrapper over `mlx_gen::quant::lin`.

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// Group size the converter writes — the codebase-wide `mlx_gen::quant::DEFAULT_GROUP_SIZE` (64).
pub(crate) const GROUP_SIZE: i32 = 64;

/// Load `{base}` as an [`AdaptableLinear`] at FLUX's [`GROUP_SIZE`] — packed when `{base}.scales`
/// is present (a pre-quantized snapshot), else dense. The shared [`mlx_gen::quant::lin`].
pub(crate) fn lin(w: &Weights, base: &str, bias: bool) -> Result<AdaptableLinear> {
    mlx_gen::quant::lin(w, base, bias, GROUP_SIZE)
}
