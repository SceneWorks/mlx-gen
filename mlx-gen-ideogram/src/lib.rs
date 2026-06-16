//! # mlx-gen-ideogram
//!
//! The **Ideogram 4.0** provider crate for [`mlx-gen`](mlx_gen) (epic 4725). Ideogram 4 is a
//! flow-matching text-to-image model whose useful prompt contract is a structured **JSON
//! caption** (handled SceneWorks-side); the engine consumes that caption as a plain string.
//!
//! Architecture (from the `ideogram-ai/ideogram-4-fp8` checkpoint, sc-5984):
//! * **Text encoder** — `Qwen3-VL-8B-Instruct` (text path), hidden states from 13 layers
//!   (`config::EXTRACTED_LAYERS`) concatenated to 53248 features. Reuses the `mlx-gen-flux2`
//!   Qwen3 blocks + a multi-layer capture hook.
//! * **Transformer** — single-stream 34-layer `Ideogram4Transformer2DModel` (AdaLN-modulated
//!   SwiGLU, fused QKV + per-head QK-norm, 3D MRoPE), instantiated **twice**
//!   (conditional + unconditional) for asymmetric CFG.
//! * **VAE** — `AutoencoderKLFlux2` (the FLUX.2 VAE) → reuse `mlx-gen-flux2::Flux2Vae`.
//! * **Scheduler** — `FlowMatchEulerDiscreteScheduler` → reuse the core flow-match schedule.
//!
//! Weights are provisioned offline by `tools/convert_ideogram4_to_mlx.py` (fp8 weight-only →
//! bf16 MLX safetensors). Runtime is pure Rust/MLX.
//!
//! Slice status: **S0** — scaffold + the dimension-parametric [`config`]. The text encoder
//! (sc-5985), transformer (sc-5986), VAE (sc-5987), and `load()`/`generate()` pipeline
//! (sc-5988) land next.

pub mod config;
pub mod loader;
pub mod text_encoder;
pub mod transformer;

pub use config::{
    Ideogram4DitConfig, Ideogram4TextEncoderConfig, DEFAULT_GUIDANCE, DEFAULT_HEIGHT,
    DEFAULT_STEPS, DEFAULT_WIDTH, EXTRACTED_LAYERS, IDEOGRAM_4_FP8_REPO, IDEOGRAM_4_ID,
};
pub use loader::{load_text_encoder, load_transformer, load_unconditional_transformer};
pub use text_encoder::Ideogram4TextEncoder;
pub use transformer::Ideogram4Transformer;
