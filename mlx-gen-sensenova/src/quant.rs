//! Packed (pre-quantized) weight loading — the consume side of [`crate::convert`].
//!
//! A pre-quantized Q4/Q8 snapshot stores each quantized Linear as the packed triple
//! `{base}.weight` (u32 codes) + `{base}.scales` + `{base}.biases`. The shared
//! [`mlx_gen::quant::lin`] **auto-detects** it by the presence of `{base}.scales` and builds the
//! quantized module directly — so a published Q4 turnkey loads packed with no dense bf16/f32
//! transient. A dense snapshot (no `.scales`) loads dense exactly as before, so the same loader
//! serves both.
//!
//! SenseNova-U1 is a **unified** MoT model: there is no separate VAE or text encoder. What the T2I
//! generation path quantizes ([`crate::model::load`] → [`crate::t2i::T2iModel::quantize`] →
//! [`crate::qwen3::Qwen3Backbone::quantize`]) is exactly the **decoder-stack Linears**: every layer's
//! four attention projections (`{q,k,v,o}_proj`) and three SwiGLU Linears (`gate/up/down_proj`) on
//! **both** the understanding (`""`) and generation (`_mot_gen`) paths — `7 · 2 · layers` bias-less
//! `[out, in]` Linears, the bulk of the 8B params. Everything else stays dense in every tier: the
//! token embedding, `lm_head`, all RMSNorms + QK-norms, the two Conv vision embedders, the flow-
//! matching `fm_head`, and the timestep/noise-scale embedders (small / precision-sensitive / not
//! plain `[out,in]` Linears). Those dense projections are loaded through [`crate::qwen3::linear`],
//! which now routes through [`lin`] below — packed when `.scales` is present, dense otherwise.
//!
//! "MoT" here is Mixture of **Transformers** (two parallel *dense* stacks with distinct
//! `_mot_gen`-suffixed keys), **not** Mixture of Experts — there are no stacked `[E, …]` expert
//! tensors and no fused proj tensors to slice, so the pack is the plain per-Linear triple with no
//! special handling.
//!
//! Group-B per-crate template (sc-8669 / sc-8771), a thin wrapper over the shared
//! `mlx_gen::quant::{lin, DEFAULT_GROUP_SIZE}`.

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// Group size the converter writes — the codebase-wide `mlx_gen::quant::DEFAULT_GROUP_SIZE` (64),
/// matching the load-time `.quantize` seam (`AdaptableLinear::quantize` defaults to 64).
pub(crate) const GROUP_SIZE: i32 = mlx_gen::quant::DEFAULT_GROUP_SIZE;

/// Load `{base}` as an [`AdaptableLinear`] at SenseNova's [`GROUP_SIZE`] — packed when `{base}.scales`
/// is present (a pre-quantized turnkey), else dense. The shared [`mlx_gen::quant::lin`]. Used by the
/// backbone loader ([`crate::qwen3`]) for every quantizable projection.
pub(crate) fn lin(w: &Weights, base: &str, bias: bool, group_size: i32) -> Result<AdaptableLinear> {
    mlx_gen::quant::lin(w, base, bias, group_size)
}
