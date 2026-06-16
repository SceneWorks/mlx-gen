//! FLUX.2-dev's **Pixtral vision tower** + **Mistral3 multimodal projector** (sc-5918).
//!
//! dev's text encoder is a `Mistral3ForConditionalGeneration`: the language tower (sc-5915, the
//! T2I path) plus — for **edit / reference** conditioning — a Pixtral ViT that encodes reference
//! images and a projector that maps the image features into the Mistral token-embedding space.
//! The projected image tokens are scattered into the Mistral input embeddings where
//! `input_ids == image_token_index (10)`; the language tower then produces the 15360-wide
//! multimodal `prompt_embeds`. This module ports the **tower + projector**; splicing the projected
//! tokens into the edit `generate()` path is sc-5919.
//!
//! Ports the transformers reference (`modeling_pixtral.py` `PixtralVisionModel` +
//! `modeling_mistral3.py` `Mistral3MultiModalProjector`), parity-gated against a golden
//! (`tools/dump_flux2_dev_pixtral_vision_golden.py` → `tests/vision_parity.rs`). The tower maps
//! almost 1:1 onto `mlx-gen-qwen-image`'s vision attention (block-diagonal SDPA via `cu_seqlens` +
//! `rotate_half` 2-D RoPE in f32); the deltas are **split** q/k/v/o (not a fused QKV), RMSNorm, and
//! a SwiGLU FFN — all **bias-free**.
//!
//! Like the Qwen2.5-VL vision tower, the Pixtral tower stays full precision (only the MMDiT + the
//! Mistral language tower quantize) and runs **f32 activations**, matching the rest of the FLUX.2
//! generate path.

pub mod attention;
pub mod block;
pub mod mlp;
pub mod patch_embed;
pub mod projector;
pub mod rope_2d;
pub mod transformer;

pub use attention::PixtralAttention;
pub use block::PixtralBlock;
pub use mlp::PixtralMlp;
pub use patch_embed::PatchConv;
pub use projector::Mistral3Projector;
pub use rope_2d::{cu_seqlens, rope_2d};
pub use transformer::PixtralVisionTower;

/// Pixtral vision-tower dimensions (dev `text_encoder/config.json` → `vision_config`). Kept
/// parametric so the parity fixture can drive a tiny synthetic tower; [`PixtralVisionConfig::dev`]
/// is the real model.
#[derive(Clone, Copy, Debug)]
pub struct PixtralVisionConfig {
    pub hidden_size: i32,
    pub num_layers: usize,
    pub num_heads: i32,
    pub head_dim: i32,
    pub intermediate_size: i32,
    /// Conv patch (= stride). dev: 14.
    pub patch_size: i32,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub num_channels: i32,
}

impl PixtralVisionConfig {
    /// FLUX.2-dev Pixtral tower: 24 layers, hidden 1024, 16 heads × head_dim 64, intermediate 4096,
    /// patch 14, RoPE θ=10000, RMSNorm eps 1e-5, 3 channels.
    pub fn dev() -> Self {
        Self {
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            head_dim: 64,
            intermediate_size: 4096,
            patch_size: 14,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            num_channels: 3,
        }
    }
}
