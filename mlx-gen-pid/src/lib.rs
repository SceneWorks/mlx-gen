//! # mlx-gen-pid — NVIDIA PiD (Pixel Diffusion Decoder)
//!
//! An optional, super-resolving replacement for an engine's VAE decode step (epic 7840). PiD denoises
//! directly in high-resolution pixel space, **decoding and upsampling in one 4-step pass**. It is tied
//! to a *latent space*, not a model: one `PixDiT_T2I` student topology serves the whole image catalog,
//! parameterized per latent space by a checkpoint + channel count + latent norm (see [`registry`]).
//!
//! This crate implements the core [`mlx_gen::LatentDecoder`] trait (the seam from sc-7844), so a
//! PiD-eligible engine can swap `vae.decode(latent)` for `pid.decode(latent)` at its decode call site
//! when the per-generation toggle is set — without N bespoke per-engine ports.
//!
//! ## Status (sc-7843, in progress)
//! Parity-verified against torch fixtures to the mlx-Metal-f32 matmul floor (see `tests/`): the
//! [`config`] + [`registry`] tables, the [`backbone`] `PixDiT_T2I` forward, the [`lq`] sigma-aware LQ
//! adapter + gate-injected [`lq::PidNet`], and the [`sampler`] 4-step SDE distill loop. The
//! `.pth → safetensors` converter (`tools/convert_pid.py`) is written + transform-verified.
//!
//! Remaining (the real-weight tail gated on the `nvidia/PiD` qwenimage checkpoint + the
//! `gemma-2-2b-it` weights + the sc-7931 CUDA reference samples): the Gemma-2-2B caption encoder
//! (a port, not an LTX reuse — LTX's Gemma is the 3-12B generation), the `LatentDecoder` decode
//! entry (caption embeds + target resolution + latent handoff), and the end-to-end real-weight smoke.
//!
//! ## License
//! PiD weights are NVIDIA NSCLv1 (non-commercial). The NC restriction flows to PiD-decoded output —
//! it must be surfaced/labeled as research/evaluation-only at the worker/web layer (Phase 3).

pub mod backbone;
pub mod caption;
pub mod config;
pub mod decoder;
pub mod gemma2;
pub mod lq;
pub mod registry;
pub mod sampler;

pub use backbone::PixDiT;
pub use caption::CaptionEncoder;
pub use config::{CaptionConfig, PidConfig, RopeMode, SampleType, SamplerConfig};
pub use decoder::PidDecoder;
pub use gemma2::{Gemma2, Gemma2Config};
pub use lq::{LqAdapter, PidNet};
pub use registry::{lookup, BackboneSpec, CkptType, LatentNorm};
pub use sampler::Sampler;
