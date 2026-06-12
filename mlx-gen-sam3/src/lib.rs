//! `mlx-gen-sam3` — native-MLX SAM3 (Segment Anything 3) concept segmenter for mlx-gen (epic 4910).
//!
//! SAM3 adds open-vocabulary **Promptable Concept Segmentation** (PCS): segment *all* instances of
//! a text concept ("person") with no geometric prompt, plus the **PVS** box/point prompt path and a
//! memory-based video tracker. This crate ports the model directly from the public Apache-2.0
//! `transformers` reference (no MLX reference port exists): the PE ViT backbone + FPN neck
//! ([`vision`]), CLIP text encoder ([`text`]), DETR detector + presence + scoring ([`detr`]), the
//! mask head ([`mask`]), the geometry/exemplar encoder ([`geometry`]), and the SAM2-style tracker +
//! multi-object video pipeline ([`tracker`], [`video`]).
//!
//! ## Public API (a plain utility segmenter — not a generation-registry provider)
//! * [`Sam3ImageSegmenter`] — text-concept PCS ([`Sam3ImageSegmenter::segment`]) and box-prompt PVS
//!   ([`Sam3ImageSegmenter::segment_with_boxes`]) on a still image.
//! * [`Sam3VideoModel`] — multi-object text-concept tracking across a clip
//!   ([`Sam3VideoModel::propagate`]).
//! * [`Sam3Tracker`] — single-frame box-prompt tracking.
//!
//! Each loads dense via `from_weights` and can be affine-quantized in place to Q8 (near-lossless) or
//! Q4 (coherent) with `quantize(bits)` — the attention/FFN/projection linears are quantized at the
//! MLX default group size, while convs, GroupNorms, embeddings, and the few small/odd projections
//! stay dense (sc-4925; see [`quantize_linear`]).

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

pub mod config;
pub mod detr;
pub mod geometry;
pub mod mask;
pub mod model;
pub mod text;
pub mod tracker;
pub mod video;
pub mod vision;

/// Load a dense quant-aware [`AdaptableLinear`] from `{name}.weight` (+ optional `{name}.bias`).
/// The shared linear constructor across the SAM3 modules — dense by default (its forward is the
/// same fused `addmm` as the old `nn::linear`, so loading stays parity-preserving) and made
/// quantizable in place via the per-module `quantize` cascades (Q8/Q4 affine, sc-4925).
pub(crate) fn load_linear(w: &Weights, name: &str) -> Result<AdaptableLinear> {
    let weight = w.require(&format!("{name}.weight"))?.clone();
    let bias = w.get(&format!("{name}.bias")).cloned();
    Ok(AdaptableLinear::dense(weight, bias))
}

/// Affine-quantize a linear in place to `bits` (Q8/Q4) at the default group size **iff** its
/// in-features divide the group size; otherwise leave it dense. MLX `quantize` requires
/// `in_features % group_size == 0`, so the handful of tiny/odd projections (the BoxRPB `2→256`
/// embedders, the geometry `4→256` box projection and its `258→256` pos projection) stay dense —
/// their parameter mass is negligible, so this is lossless for the Q8-near-bf16 target (sc-4925).
pub(crate) fn quantize_linear(l: &mut AdaptableLinear, bits: i32) -> Result<()> {
    if let Some((weight, _)) = l.dense_weight() {
        let shape = weight.shape();
        let in_features = shape[shape.len() - 1];
        if in_features % mlx_gen::quant::DEFAULT_GROUP_SIZE == 0 {
            l.quantize(bits, None)?;
        }
    }
    Ok(())
}

pub use config::{Sam3DetrConfig, Sam3GeometryConfig, Sam3TextConfig, Sam3VisionConfig};
pub use detr::{DetectorOutput, Sam3Detector};
pub use geometry::Sam3GeometryEncoder;
pub use mask::{post_process_instances, Instance, Sam3MaskHead};
pub use model::{Sam3ImageSegmenter, SegmentationOutput};
pub use text::{Sam3TextEncoder, Sam3Tokenizer};
pub use tracker::{MemoryFeatures, Sam3Tracker, TrackerFrameOutput, TrackerMask};
pub use video::{Sam3VideoModel, VideoFrameOutput};
pub use vision::Sam3VisionEncoder;
