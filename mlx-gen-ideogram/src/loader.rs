//! Real-weight loading from a converted Ideogram 4 MLX snapshot (produced by
//! `tools/convert_ideogram4_to_mlx.py`):
//! ```text
//!   <root>/text_encoder/model.safetensors   (Qwen3-VL, `language_model.*` + unused `visual.*`)
//!   <root>/transformer/model.safetensors    (E3)
//!   <root>/unconditional_transformer/...     (E3)
//!   <root>/vae/model.safetensors             (E4)
//! ```
//! The converted `text_encoder` keys map directly onto the encoder under the `"language_model"`
//! prefix — no remap. The `visual.*` vision-tower tensors are present but unused for T2I.

use std::path::Path;

use mlx_gen::weights::Weights;
use mlx_gen::Result;

use mlx_gen_flux2::Flux2Vae;

use crate::config::{Ideogram4DitConfig, Ideogram4TextEncoderConfig};
use crate::text_encoder::Ideogram4TextEncoder;
use crate::transformer::Ideogram4Transformer;

/// Load the Qwen3-VL text encoder from the converted `text_encoder` component.
pub fn load_text_encoder(root: &Path) -> Result<Ideogram4TextEncoder> {
    let w = Weights::from_dir(root.join("text_encoder"))?;
    Ideogram4TextEncoder::from_weights(
        &w,
        "language_model",
        &Ideogram4TextEncoderConfig::qwen3_vl_8b(),
    )
}

/// Load the conditional DiT (`transformer` component). Keys are top-level (empty prefix).
pub fn load_transformer(root: &Path) -> Result<Ideogram4Transformer> {
    let w = Weights::from_dir(root.join("transformer"))?;
    Ideogram4Transformer::from_weights(&w, "", &Ideogram4DitConfig::v4())
}

/// Load the unconditional DiT (`unconditional_transformer` component) — the asymmetric-CFG
/// negative branch. Same architecture, separately trained weights.
pub fn load_unconditional_transformer(root: &Path) -> Result<Ideogram4Transformer> {
    let w = Weights::from_dir(root.join("unconditional_transformer"))?;
    Ideogram4Transformer::from_weights(&w, "", &Ideogram4DitConfig::v4())
}

/// Load the VAE (`vae` component) as a `Flux2Vae` — Ideogram's `AutoencoderKLFlux2` weights map
/// directly onto the FLUX.2 VAE (same architecture; `encoder.*`/`decoder.*`/`quant_conv`/`bn.*`,
/// conv weights transposed `[O,I,H,W]→[O,H,W,I]` at construction).
pub fn load_vae(root: &Path) -> Result<Flux2Vae> {
    Flux2Vae::from_weights(&Weights::from_dir(root.join("vae"))?)
}
