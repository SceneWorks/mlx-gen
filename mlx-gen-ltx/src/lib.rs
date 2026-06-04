//! # mlx-gen-ltx
//!
//! LTX-2.3 **video** (text-to-video) provider crate for [`mlx-gen`]. Port of the
//! `mlx-video-with-audio` package's LTX video path (`generate_av.py`, `models/ltx/*`,
//! `models/ltx/video_vae/*`) onto Rust + `mlx-rs`.
//!
//! **Scope:** the bf16/f32 **video-only** T2V core (sc-2679). The audio half (`generate_av.py`'s
//! AudioVideo path), I2V, Q4/Q8, LoRA, and LoKr are sibling stories.
//!
//! This crate self-registers `ltx_2_3` into the `mlx-gen` model registry; load it with
//! `mlx_gen::load("ltx_2_3", spec)`.
//!
//! ## Status (S0)
//! Foundation slice: registry + config (`embedded_config.json`-driven) + SPLIT 3-D RoPE
//! (double-precision) + f32 position grid + distilled sigma schedules + legacy Euler step. The
//! denoise pipeline lands across S1–S5; `Generator::generate` errors until then.

pub mod config;
pub mod connector;
pub mod gemma;
pub mod model;
pub mod positions;
pub mod rope;
pub mod schedule;
pub mod text_encoder;
pub mod tiling;
pub mod transformer;
pub mod upsampler;
pub mod vae;

pub use config::{LtxConfig, LtxVaeConfig, RopeType, VaeBlock};
pub use connector::Connector;
pub use model::{descriptor, load, Ltx, MODEL_ID};
pub use text_encoder::LtxTextEncoder;
pub use tiling::TilingConfig;
pub use transformer::{to_denoised, LtxDiT, Precision, VideoBlock};
pub use upsampler::{upsample_latents, LatentUpsampler};
pub use vae::LtxVideoVae;
