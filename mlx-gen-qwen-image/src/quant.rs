//! Packed (pre-quantized) weight loading — the consume side of [`crate::convert`].
//!
//! A pre-quantized Q4/Q8 transformer stores each quantized Linear as the packed triple
//! `{base}.weight` (u32 codes) + `{base}.scales` + `{base}.biases`. The [`lin`] loader
//! **auto-detects** it by the presence of `{base}.scales` (no `quantization` manifest to read) and
//! builds the quantized module directly — so a published Q4 snapshot loads packed with no dense bf16
//! transient and is ~¼ the on-disk size. A dense snapshot (no `.scales`) loads dense exactly as
//! before, so the same loader serves both.
//!
//! Qwen-Image quantizes the **transformer only** — the Qwen2.5-VL text encoder is
//! `skip_quantization` (semantic degradation) and the VAE is all-conv (no quantizable leaves), so
//! both stay dense bf16 in every tier (see [`crate::model::load`]). Every transformer Linear is
//! built through the crate's `transformer::linear_from` helper, which routes here, so the single
//! edit packs the whole transformer (sc-8669/sc-8670 Group-B template; mirrors `mlx-gen-z-image`).

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// Group size the converter writes — the codebase-wide `mlx_gen::quant::DEFAULT_GROUP_SIZE` (64).
pub(crate) const GROUP_SIZE: i32 = 64;

/// Load `{base}` as an [`AdaptableLinear`] at Qwen-Image's [`GROUP_SIZE`] — packed when
/// `{base}.scales` is present (a pre-quantized snapshot), else dense. The shared
/// [`mlx_gen::quant::lin`]; `bias` loads the dense `{base}.bias` (the quantization's own
/// `{base}.biases` is distinct, always loaded packed).
pub(crate) fn lin(w: &Weights, base: &str, bias: bool) -> Result<AdaptableLinear> {
    mlx_gen::quant::lin(w, base, bias, GROUP_SIZE)
}
