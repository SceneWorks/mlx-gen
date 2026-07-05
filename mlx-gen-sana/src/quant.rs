//! Packed (pre-quantized) weight loading — the consume side of [`crate::convert`].
//!
//! A pre-quantized Q4/Q8 snapshot stores each quantized Linear as the packed triple
//! `{base}.weight` (u32 codes) + `{base}.scales` + `{base}.biases` (plus the dense `{base}.bias`).
//! The shared [`mlx_gen::quant::lin`] **auto-detects** it by the presence of `{base}.scales` and
//! builds the quantized [`AdaptableLinear`] directly — so a published Q4 turnkey loads packed with no
//! dense bf16/f32 transient. A dense snapshot (no `.scales`) loads dense exactly as before, so the
//! same [`crate::transformer`] `Lin` construction serves both tiers.
//!
//! SANA quantizes **two** components — the Linear-DiT trunk's matmul-heavy Linears (attention
//! projections + timestep/caption/`proj_out` MLPs) and the shared **Gemma-2 CHI text encoder** (the
//! biggest component, so its quant is where the low-RAM win is). The DC-AE decoder stays dense in
//! every tier (it is all-conv — `conv2d` weights are not a quant target and its Linear-attention is a
//! measurably-0% memory win), and so do the trunk's small conv modules (`patch_embed`, GLUMBConv).
//! The Gemma-2 side is routed through the shared [`mlx_gen::quant::lin`] in `mlx_gen_pid::gemma2`, so
//! a dense PiD Gemma-2 (no `.scales`) is unaffected.
//!
//! Group-B per-crate template (sc-8669), a thin wrapper over the shared
//! `mlx_gen::quant::{lin, DEFAULT_GROUP_SIZE}`.

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// Group size the converter writes — the codebase-wide `mlx_gen::quant::DEFAULT_GROUP_SIZE` (64),
/// matching the load-time `.quantize` seam (`AdaptableLinear::quantize` defaults to 64).
pub(crate) const GROUP_SIZE: i32 = mlx_gen::quant::DEFAULT_GROUP_SIZE;

/// Load `{base}` as an [`AdaptableLinear`] at SANA's [`GROUP_SIZE`] — packed when `{base}.scales` is
/// present (a pre-quantized turnkey), else dense. `bias` additionally loads the dense `{base}.bias`.
/// Every SANA trunk Linear (attention `to_q/k/v/out`, the timestep/guidance/caption MLPs, and
/// `proj_out`) routes through this, so a packed tier loads them already-quantized and a dense tier
/// (bf16) loads them dense — identical numerics to the pre-refactor bespoke `Linear`.
pub(crate) fn lin(w: &Weights, base: &str, bias: bool) -> Result<AdaptableLinear> {
    mlx_gen::quant::lin(w, base, bias, GROUP_SIZE)
}
