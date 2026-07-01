//! Packed (pre-quantized) weight loading — the consume side of [`crate::convert`].
//!
//! A pre-quantized Q4/Q8 snapshot stores each quantized Linear as the packed triple
//! `{base}.weight` (u32 codes) + `{base}.scales` + `{base}.biases` (plus the dense `{base}.bias`).
//! The shared [`mlx_gen::quant::lin`] **auto-detects** it by the presence of `{base}.scales` and
//! builds the quantized module directly — so a published Q4 turnkey loads packed with no dense
//! bf16/f32 transient. A dense snapshot (no `.scales`) loads dense exactly as before, so the same
//! [`crate::transformer::Lin::load`] serves both.
//!
//! Chroma quantizes a **single** component: the DiT transformer's matmul-heavy block Linears (the
//! double blocks' attention + FFN and the single blocks' attention + `proj_mlp`/`proj_out`). The
//! small/precision-sensitive modules — `x_embedder`/`context_embedder`/`proj_out` and the
//! distilled-guidance Approximator (which drives all per-block modulation) — stay dense in every
//! tier, and so do the shared T5 text encoder and FLUX.1 VAE (never quantized here; their quant is a
//! measurably-0% memory-only win and not wired). This matches
//! [`crate::transformer::ChromaTransformer::quantize`] exactly. Because every Chroma `Lin::load`
//! routes through [`lin`] below, a packed tier loads its block Linears as already-quantized bases and
//! the in-app `.quantize()` becomes a no-op (`AdaptableLinear::quantize` no-ops on a quantized base);
//! the dense embedders/Approximator load dense (no `.scales`), matching the load-time scope.
//!
//! Group-B per-crate template (sc-8669 / sc-8777), a thin wrapper over the shared
//! `mlx_gen::quant::{lin, DEFAULT_GROUP_SIZE}`.

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// Group size the converter writes — the codebase-wide `mlx_gen::quant::DEFAULT_GROUP_SIZE` (64),
/// matching the load-time `.quantize` seam (`AdaptableLinear::quantize` defaults to 64).
pub(crate) const GROUP_SIZE: i32 = mlx_gen::quant::DEFAULT_GROUP_SIZE;

/// Load `{base}` as an [`AdaptableLinear`] at Chroma's [`GROUP_SIZE`] — packed when `{base}.scales`
/// is present (a pre-quantized turnkey), else dense. `bias` additionally loads the dense `{base}.bias`
/// (every Chroma Linear carries a bias). The shared [`mlx_gen::quant::lin`]. Used by
/// [`crate::transformer::Lin::load`] for every Linear (block, embedder, and Approximator) — the
/// packed-detect keeps a dense embedder/Approximator dense while loading the packed block Linears
/// directly.
pub(crate) fn lin(w: &Weights, base: &str, bias: bool) -> Result<AdaptableLinear> {
    mlx_gen::quant::lin(w, base, bias, GROUP_SIZE)
}
