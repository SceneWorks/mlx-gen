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

use std::path::{Path, PathBuf};

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::Dtype;

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

/// `true` when a loaded component is a **pre-quantized** (packed Q4/Q8) snapshot — detected by any
/// `{base}.scales` key (sc-8746). A packed component must NOT be `cast_all`-ed: its `.weight` are u32
/// codes and its `.scales`/`.biases` carry the quantization at a fixed dtype, so a blanket
/// `astype(f16)` would corrupt the codes/scales. The `crate::quant::lin` packed-detect then builds the
/// quantized module directly (no post-load `.quantize`, which no-ops on an already-quantized base).
fn is_packed(w: &Weights) -> bool {
    w.keys().any(|k| k.ends_with(".scales"))
}

/// Resolve a component's weight file inside `subdir`, picking the variant that best matches `dtype`.
/// diffusers snapshots ship the f32 master (`<stem>.safetensors`) and/or an fp16 variant
/// (`<stem>.fp16.safetensors`); the fp16 file is exactly `astype(f16)` of the f32 master, so for an
/// f16 load the two are equivalent. We prefer the variant matching `dtype` (fp16 file for f16, the
/// f32 file otherwise) and fall back to the other when only one is cached — the caller casts to
/// `dtype` regardless, so the result is identical when both exist.
fn resolve_weight_file(root: &Path, subdir: &str, stem: &str, dtype: Dtype) -> Result<PathBuf> {
    let plain = root.join(subdir).join(format!("{stem}.safetensors"));
    let fp16 = root.join(subdir).join(format!("{stem}.fp16.safetensors"));
    let (first, second) = if dtype == Dtype::Float16 {
        (&fp16, &plain)
    } else {
        (&plain, &fp16)
    };
    if first.exists() {
        Ok(first.clone())
    } else if second.exists() {
        Ok(second.clone())
    } else {
        Err(Error::Msg(format!(
            "sdxl: missing {subdir}/{stem}.safetensors (and no .fp16 variant)"
        )))
    }
}

/// Load one CLIP text encoder from a component subdir (`text_encoder` or `text_encoder_2`) at a
/// given compute dtype. Reads the best-matching `model{,.fp16}.safetensors` and casts every tensor to
/// `dtype` — the vendored reference loads the f32 master and applies `v.astype(dtype)`, so f16 here
/// byte-matches the production `StableDiffusionXL(float16=True)` text encoder.
fn load_clip_dtype(
    root: &Path,
    subdir: &str,
    cfg: &ClipTextConfig,
    dtype: Dtype,
) -> Result<ClipTextEncoder> {
    let file = resolve_weight_file(root, subdir, "model", dtype)?;
    let mut w = Weights::from_file(&file)?;
    // A packed (pre-quantized) snapshot keeps its on-disk dtypes; only a dense snapshot downcasts.
    if !is_packed(&w) {
        w.cast_all(dtype)?;
    }
    ClipTextEncoder::from_weights(&w, "text_model", cfg)
}

/// Load CLIP-L (`text_encoder`) — the 768-wide encoder, no projection — at `dtype`.
pub fn load_text_encoder_1_dtype(root: &Path, dtype: Dtype) -> Result<ClipTextEncoder> {
    load_clip_dtype(root, "text_encoder", &ClipTextConfig::sdxl_te1(), dtype)
}

/// Load OpenCLIP-bigG (`text_encoder_2`) — the 1280-wide encoder with the pooled projection — at
/// `dtype`.
pub fn load_text_encoder_2_dtype(root: &Path, dtype: Dtype) -> Result<ClipTextEncoder> {
    load_clip_dtype(root, "text_encoder_2", &ClipTextConfig::sdxl_te2(), dtype)
}

/// f32 CLIP-L — the tight-stage-gate path (validated against the `float16=False` golden).
pub fn load_text_encoder_1(root: &Path) -> Result<ClipTextEncoder> {
    load_text_encoder_1_dtype(root, Dtype::Float32)
}

/// f32 OpenCLIP-bigG — the tight-stage-gate path.
pub fn load_text_encoder_2(root: &Path) -> Result<ClipTextEncoder> {
    load_text_encoder_2_dtype(root, Dtype::Float32)
}

/// Load the SDXL U-Net at `dtype` from `unet/diffusion_pytorch_model{,.fp16}.safetensors`. The chosen
/// file is cast to `dtype` (f16 byte-matches the production `float16=True` U-Net).
pub fn load_unet_dtype(root: &Path, dtype: Dtype) -> Result<UNet2DConditionModel> {
    load_unet_with_config(root, dtype, &UNetConfig::sdxl_base())
}

/// Load the U-Net at `dtype` with an explicit [`UNetConfig`] — the shared body of
/// [`load_unet_dtype`] (SDXL) and the Kolors loader. The `encoder_hid_proj` (Kolors) is auto-detected
/// from the weights, so the same file-resolution + cast path serves both.
pub fn load_unet_with_config(
    root: &Path,
    dtype: Dtype,
    cfg: &UNetConfig,
) -> Result<UNet2DConditionModel> {
    let file = resolve_weight_file(root, "unet", "diffusion_pytorch_model", dtype)?;
    let mut w = Weights::from_file(&file)?;
    // A packed (pre-quantized) snapshot keeps its on-disk dtypes; only a dense snapshot downcasts.
    if !is_packed(&w) {
        w.cast_all(dtype)?;
    }
    UNet2DConditionModel::from_weights(&w, cfg)
}

/// f32 U-Net — the tight-stage-gate path (validated against the `float16=False` golden).
pub fn load_unet(root: &Path) -> Result<UNet2DConditionModel> {
    load_unet_dtype(root, Dtype::Float32)
}

/// Load the **Kolors** U-Net (epic 3090) at `dtype` — [`UNetConfig::kolors`] + the auto-detected
/// `encoder_hid_proj`. `root` is the `Kwai-Kolors/Kolors-diffusers` snapshot.
pub fn load_unet_kolors_dtype(root: &Path, dtype: Dtype) -> Result<UNet2DConditionModel> {
    load_unet_with_config(root, dtype, &UNetConfig::kolors())
}

/// Load an SDXL **ControlNet** branch (sc-3058) from a diffusers `ControlNetModel` checkpoint — a
/// single `.safetensors` file or a directory containing `diffusion_pytorch_model.safetensors`. Cast
/// to `dtype` (fp16 in production, matching the U-Net it injects into).
pub fn load_controlnet(
    src: &mlx_gen::WeightsSource,
    dtype: Dtype,
) -> Result<crate::unet::ControlNet> {
    let mut w = match src {
        mlx_gen::WeightsSource::File(p) => Weights::from_file(p)?,
        mlx_gen::WeightsSource::Dir(p) => Weights::from_dir(p)?,
    };
    w.cast_all(dtype)?;
    crate::unet::ControlNet::from_weights(&w, &UNetConfig::sdxl_base())
}

/// Load the **IP-Adapter** (sc-3059) from an `h94/IP-Adapter`-layout snapshot directory: the ViT-H
/// image encoder at `models/image_encoder/model.safetensors` and the IP weights (Resampler +
/// decoupled-attn K/V pairs) at `sdxl_models/ip-adapter-plus[-face]_sdxl_vit-h.safetensors`
/// (plus-face preferred, plus as fallback — they share the Resampler architecture). Returns the
/// image-token encoder + the per-cross-attn K/V pairs to install into the U-Net. Cast to `dtype`.
pub fn load_ip_adapter(
    dir: &Path,
    dtype: Dtype,
) -> Result<(
    crate::ip_adapter::IpImageEncoder,
    Vec<(mlx_rs::Array, mlx_rs::Array)>,
)> {
    use crate::ip_adapter::{load_ip_kv_pairs, IpImageEncoder, Resampler, ResamplerConfig};
    use crate::vision_encoder::{ClipVisionEncoder, VisionConfig};

    let mut enc_w = Weights::from_file(dir.join("models/image_encoder/model.safetensors"))?;
    enc_w.cast_all(dtype)?;
    let encoder = ClipVisionEncoder::from_weights(&enc_w, &VisionConfig::vit_h_14())?;

    let ip_file = [
        "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors",
        "sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",
    ]
    .iter()
    .map(|f| dir.join(f))
    .find(|p| p.exists())
    .ok_or_else(|| {
        Error::Msg(format!(
            "ip-adapter: no plus/plus-face sdxl_vit-h weights under {}/sdxl_models",
            dir.display()
        ))
    })?;
    let mut ip_w = Weights::from_file(&ip_file)?;
    ip_w.cast_all(dtype)?;
    let resampler =
        Resampler::from_weights(&ip_w, "image_proj", &ResamplerConfig::plus_sdxl_vit_h())?;
    let pairs = load_ip_kv_pairs(&ip_w)?;

    Ok((IpImageEncoder::new(encoder, resampler), pairs))
}

/// Load the SDXL VAE (encoder + decoder). The VAE always runs **f32**, even when the U-Net/TEs are
/// fp16 — the vendored `StableDiffusion.__init__` loads `load_autoencoder(model, float16=False)`
/// unconditionally (the SDXL VAE is fp16-unstable). Prefers the f32 master; if only the fp16 variant
/// is cached it is upcast to f32 (fp16-precision weights — note: not bit-identical to the true f32
/// VAE; fetch `vae/diffusion_pytorch_model.safetensors` for an exact decode).
pub fn load_vae(root: &Path) -> Result<Autoencoder> {
    let file = resolve_weight_file(root, "vae", "diffusion_pytorch_model", Dtype::Float32)?;
    let mut w = Weights::from_file(&file)?;
    w.cast_all(Dtype::Float32)?;
    Autoencoder::from_weights(&w, &VaeConfig::sdxl_base())
}
