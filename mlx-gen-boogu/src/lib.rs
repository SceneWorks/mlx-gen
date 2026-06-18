//! # mlx-gen-boogu
//!
//! The **Boogu-Image-0.1** provider crate for [`mlx-gen`](mlx_gen). Boogu is a
//! **Lumina-Image-2.0 / OmniGen2-lineage** flow-matching image model:
//! - a mixed-stream DiT (`BooguImageTransformer2DModel`): context-refiner + noise-refiner
//!   (+ ref-image-refiner for edit) → `num_double_stream_layers` dual-stream blocks →
//!   `num_layers - num_double_stream_layers` single-stream blocks → continuous-AdaLN out,
//!   with a 3-axis (t,h,w) OmniGen2 unified RoPE,
//! - a **Qwen3-VL-8B-Instruct** condition encoder (per-token hidden states, 1 layer, dim 4096),
//! - the **FLUX.1** 16-channel `AutoencoderKL` (identical config to `mlx-gen-flux`).
//!
//! Two inference paths share the same weights/architecture: **Base** (flow-match Euler +
//! time-shift, true-CFG) and **Turbo** (DMD student few-step, no CFG) — Turbo is just a
//! different sampler over the (separately-distilled) DiT weights, not a separate engine.
//!
//! Status: E1 (this commit) lands the config parse + transformer architecture validation
//! (the converter is an *identity* key map — the diffusers keys match the module tree 1:1,
//! so [`mlx_gen::weights::Weights::from_dir`] loads them directly). E2+ add the encoder, DiT
//! forward, VAE wiring, and the `Generator` registration.

pub mod config;
pub mod convert;
pub mod loader;
pub mod quant;
pub mod text_encoder;

pub use config::BooguConfig;
pub use loader::load_text_encoder;
pub use text_encoder::{BooguTextEncoder, BooguTextEncoderConfig};
