//! # mlx-gen-face
//!
//! Native MLX face-analysis stack (epic 3079) — shared infrastructure for the PuLID-FLUX
//! (epic 3069) and InstantID (epic 3061) identity ports, replacing the torch/onnx
//! preprocessing with a Rust/MLX path (the "zero Python inference on Mac" north star).
//!
//! Sub-models (per the sc-3080 spike, all Tier-B native):
//! - **ArcFace iresnet100** ([`iresnet`]) — the fidelity-critical 512-d recognition embedding,
//!   a faithful port of antelopev2 `glintr100` (sc-3081).
//! - SCRFD detector (sc-3082), 5-pt alignment / norm_crop (sc-3083), BiSeNet parsing (sc-3084),
//!   and the unified `FaceAnalysis` API (sc-3085) land alongside.

pub mod iresnet;
pub mod scrfd;

pub use iresnet::ArcFace;
pub use scrfd::{Detection, Scrfd};
