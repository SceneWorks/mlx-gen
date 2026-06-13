//! sc-5140: `get_vae_features` — encode source media through the Wan z16 VAE into the planner's
//! source-conditioning latents. Port of `get_vae_features` (`data_utils.py`) + the `.mode()` / `.sample()`
//! split in `bernini_process.py`.
//!
//! The reference serializes the VAE `latent_dist.parameters` (mean+logvar) and later resolves them per
//! source type: **images** take `DiagonalGaussianDistribution.mode()` (the Gaussian mean), **videos**
//! take `.sample()` (`mean + exp(0.5·logvar)·ε`); both are normalized by the z16 latent stats
//! (`vae.config.latents_mean/std`, identical to [`mlx_gen_wan`]'s `VAE_MEAN`/`VAE_STD`). Those two
//! paths are [`mlx_gen_wan::WanVae::encode`] and [`mlx_gen_wan::WanVae::encode_sample`] respectively, so
//! this module is the thin Bernini-facing contract: the input reshape + the image/video dispatch.

use mlx_rs::Array;

use mlx_gen::Result;
use mlx_gen_wan::WanVae;

/// Reshape a source-media tensor to the VAE's `[1, 3, T, H, W]` (`get_vae_features`): a `[3, H, W]`
/// image gains a length-1 temporal axis (reference `x.ndim==3 -> x.unsqueeze(1)`), a `[3, T, H, W]`
/// video keeps its frames; both gain the leading batch axis (`x.unsqueeze(0)`).
pub fn to_vae_input(x: &Array) -> Result<Array> {
    let chw_t = match x.ndim() {
        3 => x.expand_dims(1)?, // [3, H, W] -> [3, 1, H, W]
        4 => x.clone(),         // [3, T, H, W]
        n => {
            return Err(mlx_gen::Error::Msg(format!(
                "get_vae_features: expected a 3-D image or 4-D video tensor, got {n}-D"
            )))
        }
    };
    Ok(chw_t.expand_dims(0)?) // -> [1, 3, T, H, W]
}

/// **Image** source → normalized VAE latent via `.mode()` (the Gaussian mean). `image` is `[3, H, W]`
/// (or already `[3, 1, H, W]`) in `[-1, 1]`.
pub fn image_vae_latent(vae: &WanVae, image: &Array) -> Result<Array> {
    vae.encode(&to_vae_input(image)?)
}

/// **Video** source → normalized VAE latent via `.sample()` (`mean + exp(0.5·logvar)·ε`). `video` is
/// `[3, T, H, W]` (T = 1 + 4·k) in `[-1, 1]`; `eps` is standard-normal noise of the latent shape
/// `[1, z, T_lat, H/8, W/8]`, supplied by the caller so the encode is deterministic.
pub fn video_vae_latent(vae: &WanVae, video: &Array, eps: &Array) -> Result<Array> {
    vae.encode_sample(&to_vae_input(video)?, eps)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reshape contract: a 3-D image gains a unit temporal axis; a 4-D video keeps its frames;
    /// both gain the batch axis. (Weight-free — the encode itself is gated in `render_real`/sc-5145.)
    #[test]
    fn to_vae_input_shapes() {
        let img = Array::zeros::<f32>(&[3, 4, 4]).unwrap();
        assert_eq!(to_vae_input(&img).unwrap().shape(), &[1, 3, 1, 4, 4]);
        let vid = Array::zeros::<f32>(&[3, 9, 8, 8]).unwrap();
        assert_eq!(to_vae_input(&vid).unwrap().shape(), &[1, 3, 9, 8, 8]);
        assert!(to_vae_input(&Array::zeros::<f32>(&[8, 8]).unwrap()).is_err());
    }
}
