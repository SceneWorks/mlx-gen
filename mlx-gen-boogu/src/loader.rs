//! Real-checkpoint loading from a Boogu-Image-0.1 snapshot (standard diffusers multi-component
//! tree): `mllm/` (Qwen3-VL condition encoder), `transformer/` (DiT), `vae/` (FLUX.1 AutoencoderKL).

use std::path::Path;

use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_gen_z_image::loader::{remap_vae_decoder, remap_vae_encoder};
use mlx_gen_z_image::vae::{Vae, VaeDecoderConfig, VaeEncoderConfig};

use crate::config::BooguConfig;
use crate::text_encoder::{BooguTextEncoder, BooguTextEncoderConfig};
use crate::transformer::BooguTransformer;
use crate::vision::{VisionConfig, VisionTower};

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

/// Load the Qwen3-VL **vision tower** from a snapshot's `mllm/` dir (`model.visual.*` keys) — the
/// image-conditioned edit path (E7b). The text tower + DiT load separately.
///
/// The tower runs in **f32**: it is small (~600 M params, run once per edit, not per denoise step)
/// and a bf16 path drifts ~0.3% over its 27 layers (cross-framework GEMM rounding amplified by the
/// merger's outlier channels — see the E7b-1 parity finding). f32 is parity-grade (image-embeds
/// cosine 0.9998 vs the reference) for negligible cost; the 10 B DiT stays bf16.
pub fn load_vision_tower(root: impl AsRef<Path>) -> Result<VisionTower> {
    let mut w = Weights::from_dir(root.as_ref().join("mllm"))?;
    let keys: Vec<String> = w
        .keys()
        .filter(|k| k.starts_with("model.visual."))
        .map(String::from)
        .collect();
    for k in keys {
        let t = w.require(&k)?.as_dtype(mlx_rs::Dtype::Float32)?;
        w.insert(k, t);
    }
    VisionTower::from_weights(&w, VisionConfig::qwen3_vl(), "model.visual")
}

/// Load the DiT from a snapshot's `transformer/` dir: parse the config, load the (identity-keyed)
/// weights, validate the architecture against the config, then assemble the model.
pub fn load_transformer(root: impl AsRef<Path>) -> Result<BooguTransformer> {
    let root = root.as_ref();
    let cfg = BooguConfig::from_snapshot(root)?;
    let w = Weights::from_dir(root.join("transformer"))?;
    crate::convert::validate_transformer(&w, &cfg)?;
    BooguTransformer::from_weights(&w, &cfg)
}

/// Load the VAE from a snapshot's `vae/` dir. Boogu ships the **FLUX.1 16-channel `AutoencoderKL`**
/// (verified from `vae/config.json`: `latent_channels 16`, `block_out_channels [128,256,512,512]`,
/// `scaling_factor 0.3611`, `shift_factor 0.1159`, no quant convs) — byte-identical to the VAE
/// `mlx-gen-flux` reuses — so we reuse [`mlx_gen_z_image::vae::Vae`] with the same `default_z_image`
/// config (its `decode` bakes in that exact scale/shift). The diffusers `decoder.*`/`encoder.*` keys
/// are remapped to the z-image module layout (conv weights NCHW→NHWC) exactly as the flux loader does.
pub fn load_vae(root: impl AsRef<Path>) -> Result<Vae> {
    let mut w = Weights::from_dir(root.as_ref().join("vae"))?;
    remap_vae_decoder(&mut w)?;
    remap_vae_encoder(&mut w)?;
    Vae::from_weights(&w, "", &VaeDecoderConfig::default_z_image())?.with_encoder(
        &w,
        "encoder",
        &VaeEncoderConfig::default_z_image(),
    )
}

// F-086: the diffusers→NHWC decoder/encoder key remaps were line-for-line copies of the FLUX.1
// loader's; both now call the shared `mlx_gen_z_image::vae::{remap_vae_decoder, remap_vae_encoder}`.
