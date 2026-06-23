//! Real-checkpoint loading from a Krea 2 snapshot (standard diffusers multi-component tree):
//! `text_encoder/` (Qwen3-VL-4B condition encoder), `transformer/` (single-stream DiT), `vae/`
//! (Qwen-Image `AutoencoderKLQwenImage`, loaded via [`crate::vae::load_vae`]). The transformer +
//! text-encoder checkpoints are identity-keyed (diffusers names = the module tree), so
//! [`Weights::from_dir`] drops straight in; the VAE remap lives in `mlx-gen-qwen-image`.

use std::path::Path;

use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::Krea2Config;
use crate::text_encoder::{KreaTeConfig, KreaTextEncoder};
use crate::transformer::Krea2Transformer;

/// Load the Qwen3-VL-4B condition encoder from a snapshot's `text_encoder/` dir. The text tower lives
/// under `language_model.*`; the visual tower (`visual.*`, unused for text-to-image) is not assembled.
pub fn load_text_encoder(root: impl AsRef<Path>) -> Result<KreaTextEncoder> {
    let root = root.as_ref();
    let cfg = KreaTeConfig::from_snapshot(root)?;
    let w = Weights::from_dir(root.join("text_encoder"))?;
    KreaTextEncoder::from_weights(&w, "language_model", &cfg)
}

/// Load the single-stream DiT from a snapshot's `transformer/` dir: parse + validate the config, load
/// the (identity-keyed diffusers) weights, validate the architecture against the config, then assemble
/// the model. A pre-quantized snapshot loads through the same path (`quant::lin` auto-detects packed
/// keys); a dense bf16 build is quantized later via [`crate::pipeline::KreaPipeline::quantize`].
pub fn load_transformer(root: impl AsRef<Path>) -> Result<Krea2Transformer> {
    let root = root.as_ref();
    let cfg = Krea2Config::from_snapshot(root)?;
    let w = Weights::from_dir(root.join("transformer"))?;
    crate::convert::validate_transformer(&w, &cfg)?;
    Krea2Transformer::from_weights(&w, &cfg)
}
