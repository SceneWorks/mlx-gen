//! The latent→pixel decode seam (epic 7840, sc-7844).
//!
//! Every image engine ends sampling with `vae.decode(latent)` called inline. To let a single
//! generation optionally route that final step through NVIDIA **PiD** — a pixel-diffusion decoder
//! that decodes *and* super-resolves in one pass — instead of the native VAE, without N bespoke
//! per-engine swaps, the decode step is expressed against this one trait. The native VAE implements
//! it (the behavior-preserving default); `mlx-gen-pid` implements it for PiD once that engine lands
//! (sc-7843), and the per-generation toggle selects which implementor a request decodes through
//! (Phase 3, sc-7849).

use crate::Result;
use mlx_rs::Array;

/// Decodes a model's final **unpacked** latent into a decoded image tensor — the input that
/// [`crate::image::decoded_to_image`] turns into an [`crate::media::Image`].
///
/// Contract:
/// - The input is the engine's unpacked latent in its latent space's native layout (e.g. Qwen/FLUX
///   16-ch `[1, C, H/8, W/8]`, SDXL 4-ch). Each implementor is tied to one latent space.
/// - The output is an `f32` tensor ready for [`crate::image::decoded_to_image`].
/// - The output's spatial size **may exceed** the VAE-native size: PiD decodes and super-resolves
///   in a single pass. Callers must read dimensions from the returned tensor, never assume
///   `latent · spatial_scale`.
pub trait LatentDecoder {
    /// Decode `latents` to a decoded image tensor.
    fn decode(&self, latents: &Array) -> Result<Array>;
}
