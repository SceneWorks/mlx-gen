//! Real-checkpoint loading for SDXL: assemble the components from a
//! `stabilityai/stable-diffusion-xl-base-1.0` snapshot directory (the diffusers multi-component
//! tree). Grows component-by-component as the slices land (tokenizers → text encoders → U-Net →
//! VAE).
//!
//! Snapshot layout:
//! ```text
//!   <root>/tokenizer/{vocab.json,merges.txt}      (+ tokenizer_2/ — byte-identical)
//!   <root>/text_encoder/model.safetensors          CLIP-L (f32)
//!   <root>/text_encoder_2/model.safetensors        OpenCLIP-bigG (f32)
//!   <root>/unet/diffusion_pytorch_model.safetensors
//!   <root>/vae/diffusion_pytorch_model.safetensors
//! ```

use std::path::Path;

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::{ClipTextConfig, UNetConfig, VaeConfig};
use crate::text_encoder::ClipTextEncoder;
use crate::tokenizer::ClipBpeTokenizer;
use crate::unet::UNet2DConditionModel;
use crate::vae::Autoencoder;

/// Load the SDXL CLIP-BPE tokenizer (one instance serves both encoders — `tokenizer/` and
/// `tokenizer_2/` ship byte-identical vocab+merges).
pub fn load_tokenizer(root: &Path) -> Result<ClipBpeTokenizer> {
    ClipBpeTokenizer::from_dir(root.join("tokenizer"))
}

/// Load one CLIP text encoder from a component subdir (`text_encoder` or `text_encoder_2`). Reads
/// the f32 `model.safetensors` (SDXL ships f32 + a parallel fp16 file; we take f32 and run f32).
fn load_clip(root: &Path, subdir: &str, cfg: &ClipTextConfig) -> Result<ClipTextEncoder> {
    let file = root.join(subdir).join("model.safetensors");
    if !file.exists() {
        return Err(Error::Msg(format!(
            "sdxl: missing {}/model.safetensors",
            subdir
        )));
    }
    let w = Weights::from_file(&file)?;
    ClipTextEncoder::from_weights(&w, "text_model", cfg)
}

/// Load CLIP-L (`text_encoder`) — the 768-wide encoder, no projection.
pub fn load_text_encoder_1(root: &Path) -> Result<ClipTextEncoder> {
    load_clip(root, "text_encoder", &ClipTextConfig::sdxl_te1())
}

/// Load OpenCLIP-bigG (`text_encoder_2`) — the 1280-wide encoder with the pooled projection.
pub fn load_text_encoder_2(root: &Path) -> Result<ClipTextEncoder> {
    load_clip(root, "text_encoder_2", &ClipTextConfig::sdxl_te2())
}

/// Load the SDXL U-Net from `unet/diffusion_pytorch_model.safetensors`.
pub fn load_unet(root: &Path) -> Result<UNet2DConditionModel> {
    let file = root.join("unet/diffusion_pytorch_model.safetensors");
    if !file.exists() {
        return Err(Error::Msg(
            "sdxl: missing unet/diffusion_pytorch_model.safetensors".into(),
        ));
    }
    let w = Weights::from_file(&file)?;
    UNet2DConditionModel::from_weights(&w, &UNetConfig::sdxl_base())
}

/// Load the SDXL VAE (encoder + decoder) from `vae/diffusion_pytorch_model.safetensors`.
pub fn load_vae(root: &Path) -> Result<Autoencoder> {
    let file = root.join("vae/diffusion_pytorch_model.safetensors");
    if !file.exists() {
        return Err(Error::Msg(
            "sdxl: missing vae/diffusion_pytorch_model.safetensors".into(),
        ));
    }
    let w = Weights::from_file(&file)?;
    Autoencoder::from_weights(&w, &VaeConfig::sdxl_base())
}
