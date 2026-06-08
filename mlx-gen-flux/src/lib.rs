//! # mlx-gen-flux
//!
//! FLUX.1 provider crate for [`mlx-gen`](mlx_gen). This crate establishes the provider
//! boundary for FLUX.1-schnell and FLUX.1-dev: registry ids, fork-derived variant
//! configuration, tokenizer loading contracts, text encoders, MMDiT transformer, VAE loading, and
//! the base txt2img generation path.

pub mod adapters;
pub mod config;
pub mod image_encoder;
pub mod ip_adapter;
pub mod loader;
pub mod model;
pub mod pipeline;
pub mod text_encoder;
pub mod transformer;

pub use adapters::apply_flux_adapters;
pub use config::{
    FluxTokenizerKind, FluxVariant, DEFAULT_GUIDANCE, DEFAULT_HEIGHT, DEFAULT_WIDTH, FLUX1_DEV_ID,
    FLUX1_SCHNELL_ID,
};
pub use image_encoder::FluxIpImageEncoder;
pub use ip_adapter::{FluxIpAdapter, FluxIpInjector};
pub use loader::{
    load_clip_encoder, load_clip_tokenizer, load_t5_encoder, load_t5_tokenizer, load_transformer,
    load_vae,
};
pub use model::{
    descriptor_dev, descriptor_for, descriptor_schnell, load_dev, load_schnell, Flux1,
};
pub use pipeline::{
    build_linear_sigmas, create_noise, image_seq_len, pack_latents, unpack_latents,
};
pub use text_encoder::{ClipTextEncoder, FluxTextEncoders, T5TextEncoder};
pub use transformer::{FluxTransformer, FluxTransformerConfig};
