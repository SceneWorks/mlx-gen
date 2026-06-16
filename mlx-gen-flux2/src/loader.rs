//! Real-checkpoint loading for FLUX.2-klein from a `black-forest-labs/FLUX.2-klein-9B` snapshot
//! directory (standard diffusers multi-component tree):
//! ```text
//!   <root>/tokenizer/tokenizer.json
//!   <root>/text_encoder/*.safetensors   (Qwen3, `model.*` keys)
//!   <root>/transformer/*.safetensors    (S3)
//!   <root>/vae/*.safetensors            (S2)
//! ```
//! The Qwen3 `text_encoder` layout maps directly onto the encoder under the `"model"` prefix
//! (the fork's mapping only strips `model.`), so it needs no remap.

use std::path::Path;

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::Flux2Config;
use crate::text_encoder::{Qwen3TextEncoder, Qwen3TextEncoderConfig};
use crate::transformer::Flux2Transformer;
use crate::vae::Flux2Vae;

/// Qwen2 pad token id (`<|endoftext|>`).
pub const PAD_TOKEN_ID: i32 = 151643;
/// The fork's `LanguageTokenizer` max_length for the FLUX.2 `qwen3` tokenizer.
pub const MAX_LENGTH: usize = 512;

/// Load the Qwen2 tokenizer with FLUX.2's chat template (`enable_thinking=False`) and the fork's
/// padding policy (`padding="max_length"` → every prompt padded to 512).
pub fn load_tokenizer(root: &Path) -> Result<TextTokenizer> {
    let path = root.join("tokenizer/tokenizer.json");
    TextTokenizer::from_file(
        path,
        TokenizerConfig {
            max_length: MAX_LENGTH,
            pad_token_id: PAD_TOKEN_ID,
            chat_template: ChatTemplate::QwenInstructNoThink,
            pad_to_max_length: true,
        },
    )
    .map_err(Into::into)
}

/// Load the Qwen3 text encoder. The on-disk `model.*` keys map directly onto the encoder tree
/// under the `"model"` prefix — no remap needed.
pub fn load_text_encoder(root: &Path) -> Result<Qwen3TextEncoder> {
    let w = Weights::from_dir(root.join("text_encoder"))?;
    Qwen3TextEncoder::from_weights(&w, "model", &Qwen3TextEncoderConfig::klein_9b())
}

/// `<pad>` token id for the FLUX.2-dev Mistral tokenizer (vs klein's Qwen2 `<|endoftext|>` 151643).
pub const DEV_PAD_TOKEN_ID: i32 = 11;

/// Load the FLUX.2-dev tokenizer: the Mistral/Pixtral `tokenizer.json` with the dev chat template
/// (`[SYSTEM_PROMPT]…[/SYSTEM_PROMPT][INST]…[/INST]`, BOS auto-prepended by the post-processor) and
/// the fork's `padding="max_length"` (every prompt padded to 512 with `<pad>`). The `PixtralProcessor`
/// image path is not part of the T2I tokenization (sc-5918).
pub fn load_tokenizer_dev(root: &Path) -> Result<TextTokenizer> {
    let path = root.join("tokenizer/tokenizer.json");
    TextTokenizer::from_file(
        path,
        TokenizerConfig {
            max_length: MAX_LENGTH,
            pad_token_id: DEV_PAD_TOKEN_ID,
            chat_template: ChatTemplate::Flux2DevMistral,
            pad_to_max_length: true,
        },
    )
    .map_err(Into::into)
}

/// Load the **FLUX.2-dev Mistral** text encoder (sc-5915). The dev `text_encoder` is a
/// `Mistral3ForConditionalGeneration`; the T2I path consumes only its language tower, whose weights
/// live under the `language_model.model.*` prefix (the vision tower + projector are unused here,
/// sc-5918). Same decoder-LM graph as klein's Qwen3 minus the per-head q/k-norm (`qk_norm: false`).
pub fn load_text_encoder_dev(root: &Path) -> Result<Qwen3TextEncoder> {
    let w = Weights::from_dir(root.join("text_encoder"))?;
    Qwen3TextEncoder::from_weights(
        &w,
        "language_model.model",
        &Qwen3TextEncoderConfig::mistral_dev(),
    )
}

/// Load the FLUX.2 VAE. The on-disk diffusers keys (`encoder.*`/`decoder.*`/`quant_conv.*`/
/// `bn.*`) map directly onto the module; conv weights are transposed `[O,I,H,W]→[O,H,W,I]` at
/// construction.
pub fn load_vae(root: &Path) -> Result<Flux2Vae> {
    let w = Weights::from_dir(root.join("vae"))?;
    Flux2Vae::from_weights(&w)
}

/// Load the MMDiT transformer for `cfg`, applying the diffusers→internal renames (the fork's
/// `Flux2WeightMapping`): the time embedding `time_guidance_embed.timestep_embedder.linear_{1,2}`
/// → `time_guidance_embed.linear_{1,2}`, and each double block's Sequential
/// `transformer_blocks.{i}.attn.to_out.0` → `to_out`. Everything else matches 1:1. The renames are
/// arch-general (klein and dev are the same `Flux2Transformer2DModel`); only `cfg` differs.
fn load_transformer_with(root: &Path, cfg: &Flux2Config) -> Result<Flux2Transformer> {
    let mut w = Weights::from_dir(root.join("transformer"))?;
    for n in ["linear_1", "linear_2"] {
        w.alias(
            &format!("time_guidance_embed.timestep_embedder.{n}.weight"),
            &format!("time_guidance_embed.{n}.weight"),
        );
    }
    for i in 0..cfg.num_double_layers {
        w.alias(
            &format!("transformer_blocks.{i}.attn.to_out.0.weight"),
            &format!("transformer_blocks.{i}.attn.to_out.weight"),
        );
    }
    Flux2Transformer::from_weights(&w, cfg)
}

/// Load the FLUX.2-klein MMDiT transformer.
pub fn load_transformer(root: &Path) -> Result<Flux2Transformer> {
    load_transformer_with(root, &Flux2Config::klein_9b())
}

/// Load the FLUX.2-dev MMDiT transformer (sc-5916): the same parametric module tree as klein at the
/// dev dims (48 single blocks / 48 heads / joint 15360), via `Flux2Config::dev()`.
pub fn load_transformer_dev(root: &Path) -> Result<Flux2Transformer> {
    load_transformer_with(root, &Flux2Config::dev())
}
