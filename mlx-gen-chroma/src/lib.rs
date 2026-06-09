//! # mlx-gen-chroma
//!
//! Chroma provider crate for [`mlx-gen`](mlx_gen) (epic 3531). Chroma (`chroma1_hd` / `chroma1_base`
//! / `chroma1_flash`, family `chroma`) is a FLUX.1-schnell-derived DiT: the FLUX MMDiT skeleton with
//! a distilled-guidance **Approximator** replacing the FLUX modulation stack, **T5-XXL-only**
//! conditioning (no CLIP / no pooled), MMDiT attention masking, and **true CFG**.
//!
//! Reuses `mlx-gen-flux` for the T5-XXL encoder, the AutoencoderKL VAE loader, and the
//! pack/unpack/sigma helpers; the Chroma DiT is ported fresh.

pub mod adapters;
pub mod beta;
pub mod config;
pub mod loader;
pub mod model;
pub mod text;
pub mod transformer;

pub use adapters::apply_chroma_adapters;
pub use config::{
    ChromaTransformerConfig, ChromaVariant, CHROMA1_BASE_ID, CHROMA1_FLASH_ID, CHROMA1_HD_ID,
    DEFAULT_SAMPLER, MAX_SEQUENCE_LENGTH,
};
pub use model::{
    descriptor_base, descriptor_flash, descriptor_hd, load_base, load_chroma, load_flash, load_hd,
    Chroma,
};
pub use text::{encode_prompt, t5_key_mask, transformer_text_mask};
pub use transformer::ChromaTransformer;
