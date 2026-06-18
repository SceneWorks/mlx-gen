//! Packed (pre-quantized) weight loading helpers — auto-detect a Q4/Q8 snapshot by the presence of
//! `{base}.scales` and build the quantized module directly (no dense bf16 transient), else load
//! dense. The same loaders serve a dense bf16 snapshot and a pre-quantized one (E8). Mirrors the
//! ideogram crate's `quant` helpers.

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::TokenEmbedding;
use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_rs::Array;

/// Group size the converter writes — the codebase-wide `mlx_gen::quant::DEFAULT_GROUP_SIZE` (64).
pub(crate) const GROUP_SIZE: i32 = 64;

/// Derive the quant bit-width from the packed shapes: `scales` is `[out, in/gs]` and the u32-packed
/// `weight` is `[out, in·bits/32]`, so `bits = wq.cols·32 / (scales.cols·gs)`.
fn packed_bits(wq: &Array, scales: &Array) -> i32 {
    let in_dim = scales.shape()[1] * GROUP_SIZE;
    wq.shape()[1] * 32 / in_dim
}

/// Load `{base}` as an [`AdaptableLinear`] — packed when `{base}.scales` is present, else dense.
/// `bias` additionally loads the dense `{base}.bias` (distinct from the quant's `{base}.biases`).
pub(crate) fn lin(w: &Weights, base: &str, bias: bool) -> Result<AdaptableLinear> {
    let bias = if bias {
        Some(w.require(&format!("{base}.bias"))?.clone())
    } else {
        None
    };
    if let Some(scales) = w.get(&format!("{base}.scales")) {
        let wq = w.require(&format!("{base}.weight"))?;
        let bits = packed_bits(wq, scales);
        return Ok(AdaptableLinear::from_quantized_parts(
            wq.clone(),
            scales.clone(),
            w.require(&format!("{base}.biases"))?.clone(),
            bias,
            GROUP_SIZE,
            bits,
        ));
    }
    Ok(AdaptableLinear::dense(
        w.require(&format!("{base}.weight"))?.clone(),
        bias,
    ))
}

/// Load `{base}` as a [`TokenEmbedding`] — packed when `{base}.scales` is present, else dense.
pub(crate) fn embedding(w: &Weights, base: &str) -> Result<TokenEmbedding> {
    if let Some(scales) = w.get(&format!("{base}.scales")) {
        let wq = w.require(&format!("{base}.weight"))?;
        let bits = packed_bits(wq, scales);
        return Ok(TokenEmbedding::from_quantized_parts(
            wq.clone(),
            scales.clone(),
            w.require(&format!("{base}.biases"))?.clone(),
            GROUP_SIZE,
            bits,
        ));
    }
    Ok(TokenEmbedding::Dense(
        w.require(&format!("{base}.weight"))?.clone(),
    ))
}
