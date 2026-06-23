//! Krea 2's VAE — the **Qwen-Image** `AutoencoderKLQwenImage` (f8, 16 latent channels), reused
//! wholesale from [`mlx_gen_qwen_image`].
//!
//! The published `krea/Krea-2-Turbo` `vae/config.json` declares `_class_name =
//! "AutoencoderKLQwenImage"` and `_name_or_path = "Qwen/Qwen-Image"`, and the reference
//! `autoencoder.py` literally loads `AutoencoderKLQwenImage.from_pretrained("Qwen/Qwen-Image",
//! subfolder="vae")` — so the Krea snapshot's `vae/` weights are byte-identical to Qwen-Image's and
//! load through the same diffusers→internal key remap ([`mlx_gen_qwen_image::load_vae`]). This is the
//! provider→provider VAE-reuse precedent (boogu→z-image, kolors→sdxl); the alternative — re-porting
//! the causal-Conv3d VAE — would duplicate already-parity-proven code for zero benefit.
//!
//! De-normalization is **per-channel** `latents_mean`/`latents_std` (a 16-vector, NOT a scalar
//! scale/shift), already baked into [`QwenVae::decode`] (`(latent · std) + mean → post_quant_conv →
//! decoder`) — matching the reference `QwenAutoencoder.decode` exactly. The Krea `vae/config.json`
//! `latents_mean`/`latents_std` are identical to the constants `QwenVae` carries.

use std::path::Path;

use mlx_gen::Result;

pub use mlx_gen_qwen_image::QwenVae;

/// VAE spatial compression factor (`ae.compression`) — 3 spatial-downsample stages = 8×. With
/// `patch_size = 2` this gives the pipeline's W/H alignment `compression · patch = 16`.
pub const VAE_COMPRESSION: u32 = 8;
/// VAE latent channel count (`ae.channels` = the DiT's `z_dim`).
pub const VAE_CHANNELS: u32 = 16;

/// Load the Qwen-Image VAE from a Krea snapshot's `vae/` dir, applying the diffusers→internal key
/// remap. `root` is the **snapshot root** (the `vae/` subdir is joined internally), matching
/// [`crate::config::Krea2Config::from_snapshot`].
pub fn load_vae(root: impl AsRef<Path>) -> Result<QwenVae> {
    mlx_gen_qwen_image::load_vae(root.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vae_constants_match_qwen_image() {
        // Sanity: the Qwen-Image VAE is f8 / 16-channel; `compression · patch(2) = 16` is the
        // pipeline alignment the model-layer `RES_MULTIPLE` enforces.
        assert_eq!(VAE_COMPRESSION, 8);
        assert_eq!(VAE_CHANNELS, 16);
        assert_eq!(VAE_COMPRESSION * 2, 16);
    }

    #[test]
    fn load_vae_errors_cleanly_on_missing_dir() {
        // `QwenVae` isn't `Debug`, so match rather than `unwrap_err`.
        match load_vae("/nonexistent-krea-snapshot") {
            Err(e) => assert!(!e.to_string().is_empty()),
            Ok(_) => panic!("expected a load error for a missing snapshot dir"),
        }
    }
}
