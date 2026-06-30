//! Packed (pre-quantized) weight loading — the consume side of [`crate::convert`].
//!
//! A pre-quantized Q4/Q8 snapshot stores each quantized Linear / embedding as the packed triple
//! `{base}.weight` (u32 codes) + `{base}.scales` + `{base}.biases`. The [`lin`] / [`embedding`]
//! loaders **auto-detect** it by the presence of `{base}.scales` (no `quantization` manifest to
//! read) and build the quantized module directly — so a published Q4 snapshot loads packed with no
//! dense bf16 transient and is ~¼ the on-disk size. A dense snapshot (no `.scales`) loads dense
//! exactly as before, so the same loaders serve both.
//!
//! Z-Image quantizes all three components — DiT transformer, Qwen3 text encoder, and the VAE's
//! mid-block attention — so every quantizable Linear / the token embedding across those three
//! `*_from_weights` paths routes through here (sc-8670). This is the Group-B per-crate template
//! (sc-8669): a 30-line wrapper over the shared `mlx_gen::quant::{lin, embedding}`.

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::TokenEmbedding;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// Group size the converter writes — the codebase-wide `mlx_gen::quant::DEFAULT_GROUP_SIZE` (64).
pub(crate) const GROUP_SIZE: i32 = 64;

/// Load `{base}` as an [`AdaptableLinear`] at Z-Image's [`GROUP_SIZE`] — packed when `{base}.scales`
/// is present (a pre-quantized snapshot), else dense. The shared [`mlx_gen::quant::lin`]; `bias` loads
/// the dense `{base}.bias` (the quantization's own `{base}.biases` is distinct, always loaded packed).
pub(crate) fn lin(w: &Weights, base: &str, bias: bool) -> Result<AdaptableLinear> {
    mlx_gen::quant::lin(w, base, bias, GROUP_SIZE)
}

/// Load `{base}` as a [`TokenEmbedding`] at Z-Image's [`GROUP_SIZE`] — packed when `{base}.scales`
/// is present, else dense ([`mlx_gen::quant::embedding`]).
pub(crate) fn embedding(w: &Weights, base: &str) -> Result<TokenEmbedding> {
    mlx_gen::quant::embedding(w, base, GROUP_SIZE)
}
