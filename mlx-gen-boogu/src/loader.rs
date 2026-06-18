//! Real-checkpoint loading from a Boogu-Image-0.1 snapshot (standard diffusers multi-component
//! tree): `mllm/` (Qwen3-VL condition encoder), `transformer/` (DiT), `vae/` (FLUX.1 AutoencoderKL).

use std::path::Path;

use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::text_encoder::{BooguTextEncoder, BooguTextEncoderConfig};

/// Load the Qwen3-VL-8B condition encoder from a snapshot's `mllm/` dir. The text tower lives under
/// `model.language_model.*`; the visual tower + `lm_head` are loaded but unused for text-to-image.
pub fn load_text_encoder(root: impl AsRef<Path>) -> Result<BooguTextEncoder> {
    let w = Weights::from_dir(root.as_ref().join("mllm"))?;
    BooguTextEncoder::from_weights(
        &w,
        "model.language_model",
        &BooguTextEncoderConfig::qwen3_vl_8b(),
    )
}
