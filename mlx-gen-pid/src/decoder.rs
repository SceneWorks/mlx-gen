//! The PiD [`LatentDecoder`] — the per-generation decoder that swaps an engine's `vae.decode(latent)`
//! for a super-resolving PiD pixel-diffusion decode (the sc-7844 seam). It carries this generation's
//! caption embeddings (+ degrade σ + SR scale), so `decode(latents)` stays the unchanged trait method
//! (the engine already holds the conditioning). Faithful to `from_clean.py`: PiD consumes the
//! **normalized** VAE latent directly; the output resolution is `latent_grid · vae_compression · scale`.

use mlx_rs::{Array, Dtype};

use mlx_gen::decoder::LatentDecoder;
use mlx_gen::{CancelFlag, Error, Result};

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
    /// Cooperative cancellation for the ~100 s 4-step decode (F-006). Bound at decoder-mint time
    /// from `req.cancel` so [`LatentDecoder::decode`] — whose trait signature carries no flag — can
    /// still honor a cancel per sampler step. `None` ⇒ uncancellable (direct struct-API construction).
    cancel: Option<CancelFlag>,
    /// Spatial-tiling policy `(tile_edge_px, overlap_px)` for large outputs (sc-10087). Bound at
    /// mint time from the budget/watchdog plan ([`crate::budget::plan_tile_edge`]); `decode` tiles when
    /// the output exceeds `tile_edge` on either axis, else takes the exact whole-image path. `None` ⇒
    /// always whole-image (small outputs, or a direct struct-API caller that hasn't opted in).
    tile: Option<(i32, i32)>,
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
            cancel: None,
            tile: None,
        }
    }

    /// Bind a cooperative cancellation handle (F-006). Callers that go through
    /// [`crate::resolve_pid_decoder`] get this wired from `req.cancel` automatically; direct
    /// struct-API callers (e.g. InstantID) can opt in with their request's flag.
    pub fn with_cancel(mut self, cancel: CancelFlag) -> Self {
        self.cancel = Some(cancel);
        self
    }

    /// Enable spatial tiling for large outputs (sc-10087): [`LatentDecoder::decode`] tiles the per-step
    /// velocity forward with `tile_edge`-px tiles (feather `overlap`) whenever the output exceeds
    /// `tile_edge` on either axis, and takes the exact whole-image path otherwise. Callers through
    /// [`crate::resolve_pid_decoder`] get this wired from [`crate::budget::plan_tile_edge`] automatically.
    pub fn with_tiling(mut self, tile_edge: i32, overlap: i32) -> Self {
        self.tile = Some((tile_edge, overlap));
        self
    }

    /// The output pixel resolution for a latent grid `[.., .., zH, zW]`.
    pub fn target_hw(&self, latents: &Array) -> (i32, i32) {
        let sh = latents.shape();
        let f = self.vae_compression * self.scale;
        (sh[2] * f, sh[3] * f)
    }

    /// Validate the `[B, z, zH, zW]` LQ-latent contract (F-100) and prepare the shared decode inputs:
    /// the bf16-cast latent, batch `b`, output `(th, tw)`, and the per-sample degrade σ.
    fn prep(&self, latents: &Array) -> Result<(Array, i32, i32, i32, Array)> {
        let sh = latents.shape();
        if sh.len() != 4 {
            return Err(Error::Msg(format!(
                "pid decode: expected a rank-4 [B, z, zH, zW] LQ latent, got shape {sh:?}"
            )));
        }
        let expected_z = self.net.lq_latent_channels();
        if sh[1] != expected_z {
            return Err(Error::Msg(format!(
                "pid decode: LQ latent has {} channels, expected {expected_z}",
                sh[1]
            )));
        }
        let b = sh[0];
        let (th, tw) = self.target_hw(latents);
        // PiD runs the released bf16 inference dtype, and the LQ-adapter convs require their input in
        // that dtype. An engine may hand us an f32 sampler latent (Qwen/Krea keep latents f32 through
        // the denoise loop), so cast here — a no-op when the latent is already bf16. Matches the
        // validated `from_clean` path (sc-7843), which cast the VAE latent to bf16 before decode.
        let latents = latents.as_dtype(Dtype::Bfloat16)?;
        let sigma = Array::from_slice(&vec![self.sigma; b as usize], &[b]);
        Ok((latents, b, th, tw, sigma))
    }

    /// Repro capture (sc-10087): when `PID_CAPTURE_LATENT` names a `.safetensors` path, dump the exact
    /// LQ latent + this generation's caption embeddings (plus σ/scale/vae_compression/seed as metadata)
    /// the *first* time `decode` runs, so a real production latent can be replayed offline (e.g. the
    /// tiled-vs-whole A/B) without re-running the full engine. Best-effort: a failure logs but never
    /// breaks the decode. No-op (one env read) when unset.
    fn maybe_capture(&self, latents: &Array) {
        let Ok(path) = std::env::var("PID_CAPTURE_LATENT") else {
            return;
        };
        if std::path::Path::new(&path).exists() {
            return; // capture once
        }
        // Best-effort (F-154): the dtype casts can fail (an exotic input dtype), so fold them into the
        // same fallible log path as the save — a capture failure must never `unwrap()`-panic the
        // production decode this env var is merely observing.
        match self.try_capture(latents, &path) {
            Ok(()) => eprintln!("[pid] captured LQ latent + caption to {path}"),
            Err(e) => eprintln!("[pid] PID_CAPTURE_LATENT capture failed: {e}"),
        }
    }

    /// The fallible body of [`Self::maybe_capture`] — casts + save, all through `?` so any failure is
    /// surfaced to the best-effort log path rather than panicking mid-decode.
    fn try_capture(&self, latents: &Array, path: &str) -> Result<()> {
        let meta = std::collections::HashMap::from([
            ("sigma".to_string(), self.sigma.to_string()),
            ("scale".to_string(), self.scale.to_string()),
            (
                "vae_compression".to_string(),
                self.vae_compression.to_string(),
            ),
            ("seed".to_string(), self.seed.to_string()),
        ]);
        let arrays = [
            ("latent", latents.as_dtype(Dtype::Float32)?),
            ("caption", self.caption_embs.as_dtype(Dtype::Float32)?),
        ];
        Array::save_safetensors(arrays.iter().map(|(k, v)| (*k, v)), &meta, path)?;
        Ok(())
    }

    /// Spatially-tiled decode (sc-10087): same result geometry + seeded noise as [`LatentDecoder::decode`],
    /// but the per-step velocity forward runs on overlapping `tile`-px pixel windows (feather-blended) so
    /// the whole-image `PidNet::forward` peak / long Metal command buffer never materializes. `tile` /
    /// `overlap` are output-pixel units. Used above a size threshold at the decode seam, and by the A/B
    /// harness to compare tiled vs whole-image on an identical latent.
    pub fn decode_tiled(&self, latents: &Array, tile: i32, overlap: i32) -> Result<Array> {
        // Capture the LQ latent here too (F-154): the public tiled entry (used by the A/B harness) must
        // honor `PID_CAPTURE_LATENT` symmetrically with [`LatentDecoder::decode`], not silently skip it.
        self.maybe_capture(latents);
        let (latents, b, th, tw, sigma) = self.prep(latents)?;
        self.sampler.sample_tiled(
            &self.net,
            &self.caption_embs,
            &latents,
            &sigma,
            b,
            th,
            tw,
            self.seed,
            tile,
            overlap,
            self.cancel.as_ref(),
        )
    }
}

impl LatentDecoder for PidDecoder {
    /// `latents`: the normalized VAE latent `[B, C, zH, zW]`. Returns super-resolved pixels
    /// `[B, 3, zH·vae_compression·scale, zW·vae_compression·scale]` in `[-1, 1]`.
    fn decode(&self, latents: &Array) -> Result<Array> {
        self.maybe_capture(latents);
        let (latents, b, th, tw, sigma) = self.prep(latents)?;
        // Tile when a policy is set and the output exceeds one tile on either axis (sc-10087); otherwise
        // the exact whole-image path — byte-identical to the pre-tiling decode for small outputs.
        if let Some((edge, overlap)) = self.tile {
            if th > edge || tw > edge {
                return self.sampler.sample_tiled(
                    &self.net,
                    &self.caption_embs,
                    &latents,
                    &sigma,
                    b,
                    th,
                    tw,
                    self.seed,
                    edge,
                    overlap,
                    self.cancel.as_ref(),
                );
            }
        }
        self.sampler.sample(
            &self.net,
            &self.caption_embs,
            &latents,
            &sigma,
            b,
            th,
            tw,
            self.seed,
            self.cancel.as_ref(),
        )
    }
}
