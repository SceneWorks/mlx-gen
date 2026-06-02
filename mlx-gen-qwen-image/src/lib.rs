//! # mlx-gen-qwen-image
//!
//! The **Qwen-Image** (+ Qwen-Image-Edit) provider crate for [`mlx-gen`](mlx_gen). Depends only on
//! the `mlx-gen` core (nn primitives, adapters, weights, quant, the `Generator` contract, the
//! registry) and — once the model lands — self-registers via `inventory` so that
//! `mlx_gen::load("qwen_image", …)` resolves. See `docs/MODEL_ARCHITECTURE.md`.
//!
//! Ported from the frozen Python mflux fork (`~/repos/mflux/src/mflux/models/qwen/`) and
//! parity-proven against it on real bf16 weights. Shipped: **Qwen-Image T2I** (`qwen_image`,
//! sc-2348) and **Qwen-Image-Edit** (`qwen_image_edit`, sc-2465) — the causal-Conv3d VAE, the
//! Qwen2.5-VL text encoder, the 60-layer dual-stream MMDiT, the Qwen2-VL image processor +
//! Qwen2.5-VL vision transformer + reference-latent conditioning (Edit), and transformer-only
//! Q4/Q8 quantization (sc-2565; the fork keeps the text encoder + VAE dense). LoRA/LoKr and
//! multi-image Edit is not yet wired (sc-2529). LoRA/LoKr is wired (sc-2528).

pub mod adapters;
pub mod image_processor;
pub mod loader;
pub mod model;
pub mod model_edit;
pub mod pipeline;
pub mod text_encoder;
pub mod transformer;
pub mod vae;
pub mod vl_tokenizer;

pub use adapters::apply_qwen_adapters;
pub use image_processor::{ImageInput, ProcessedImage, QwenImageProcessor};
pub use loader::{
    load_text_encoder, load_tokenizer, load_transformer, load_vae, load_vision_encoder,
    load_vision_language_encoder,
};
pub use model::{descriptor, load, QwenImage, MODEL_ID};
pub use model_edit::QwenImageEdit;
pub use pipeline::{
    add_noise_by_interpolation, compute_guided_noise, create_noise, decoded_to_image,
    denoise_edit_with_progress, denoise_with_progress, encode_init_latents, init_time_step,
    pack_latents, preprocess_init_image, qwen_scheduler, unpack_latents,
};
pub use text_encoder::{QwenTextEncoder, QwenTextEncoderConfig};
pub use transformer::{QwenTransformer, QwenTransformerConfig};
pub use vae::QwenVae;
pub use vl_tokenizer::{
    encode_reference_latents, preprocess_edit_image, tokenize_edit, tokenize_edit_text, EditImage,
    EditInputs,
};
