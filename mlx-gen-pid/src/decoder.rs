//! The PiD [`LatentDecoder`] — the per-generation decoder that swaps an engine's `vae.decode(latent)`
//! for a super-resolving PiD pixel-diffusion decode (the sc-7844 seam). It carries this generation's
//! caption embeddings (+ degrade σ + SR scale), so `decode(latents)` stays the unchanged trait method
//! (the engine already holds the conditioning). Faithful to `from_clean.py`: PiD consumes the
//! **normalized** VAE latent directly; the output resolution is `latent_grid · vae_compression · scale`.

use mlx_rs::Array;

use mlx_gen::decoder::LatentDecoder;
use mlx_gen::Result;

use crate::lq::PidNet;
use crate::sampler::Sampler;

/// A PiD decoder bound to one generation's caption + σ + scale.
pub struct PidDecoder {
    net: PidNet,
    sampler: Sampler,
    /// `[1, L, txt_embed_dim]` caption embeddings for this generation (from [`crate::caption`]).
    caption_embs: Array,
    /// Degrade σ fed to the LQ gate (0 for a clean-latent decode).
    sigma: f32,
    /// Spatial SR factor (4× for the released students).
    scale: i32,
    /// VAE spatial compression (latent grid → pixel grid; 8 for the catalog VAEs).
    vae_compression: i32,
    /// Per-decode RNG seed for the sampler's noise + per-step ε.
    seed: u64,
}

impl PidDecoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        net: PidNet,
        sampler: Sampler,
        caption_embs: Array,
        sigma: f32,
        scale: i32,
        vae_compression: i32,
        seed: u64,
    ) -> Self {
        Self {
            net,
            sampler,
            caption_embs,
            sigma,
            scale,
            vae_compression,
            seed,
        }
    }

    /// The output pixel resolution for a latent grid `[.., .., zH, zW]`.
    pub fn target_hw(&self, latents: &Array) -> (i32, i32) {
        let sh = latents.shape();
        let f = self.vae_compression * self.scale;
        (sh[2] * f, sh[3] * f)
    }
}

impl LatentDecoder for PidDecoder {
    /// `latents`: the normalized VAE latent `[B, C, zH, zW]`. Returns super-resolved pixels
    /// `[B, 3, zH·vae_compression·scale, zW·vae_compression·scale]` in `[-1, 1]`.
    fn decode(&self, latents: &Array) -> Result<Array> {
        let b = latents.shape()[0];
        let (th, tw) = self.target_hw(latents);
        let sigma = Array::from_slice(&vec![self.sigma; b as usize], &[b]);
        self.sampler.sample(
            &self.net,
            &self.caption_embs,
            latents,
            &sigma,
            b,
            th,
            tw,
            self.seed,
        )
    }
}
