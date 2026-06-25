//! [`PidEngine`] — the load-once, decode-many entry point a PiD-eligible provider holds (epic 7840,
//! sc-7845). It owns the heavy weights (the `PixDiT` student checkpoint + the Gemma-2 caption encoder)
//! and the per-latent-space [`PidConfig`], and mints a per-generation [`PidDecoder`] bound to that
//! generation's caption + degrade σ + seed via [`PidEngine::decoder`].
//!
//! A PiD decoder is tied to a *latent space*, not a model, so the engine is parameterized by a
//! backbone tag (`"qwenimage"`, `"flux"`, …) resolved against the [`crate::registry`]. The released
//! students all share the `sr4x` `PixDiT` topology; only the LQ latent-channel count differs per
//! space. This is the shared home the Phase-2 wiring stories (qwen/krea sc-7845, flux sc-7846,
//! flux2 sc-7847, sdxl sc-7848) construct PiD through.

use std::path::{Path, PathBuf};

use mlx_rs::Dtype;

use mlx_gen::weights::Weights;
use mlx_gen::{Error, GenerationRequest, PidWeights, Result, WeightsSource};

use crate::caption::CaptionEncoder;
use crate::config::{PidConfig, SamplerConfig};
use crate::decoder::PidDecoder;
use crate::gemma2::{Gemma2, Gemma2Config};
use crate::lq::PidNet;
use crate::registry::lookup;
use crate::sampler::Sampler;

/// Filename of the merged Gemma-2-2b-it checkpoint inside the gemma snapshot dir; falls back to
/// loading every `*.safetensors` shard in the dir when absent.
const GEMMA_MERGED_FILE: &str = "gemma-2-2b-it.safetensors";

/// A loaded PiD decoder engine for one latent space — built once, reused across generations.
pub struct PidEngine {
    /// The converted student checkpoint, retained so [`Self::decoder`] can rebuild a [`PidNet`] per
    /// generation (cheap vs the ~100 s decode — `Array` handles are refcounted).
    weights: Weights,
    /// Per-latent-space backbone config (`sr4x` topology + the space's LQ latent-channel count).
    cfg: PidConfig,
    /// The released 4-step SDE distill sampler config.
    sampler_cfg: SamplerConfig,
    /// The Gemma-2-2b caption encoder (loaded once; the projection runs per caption).
    caption: CaptionEncoder,
    /// Key prefix for [`PidNet::from_weights`] — `""` for the converted checkpoint (the EMA export
    /// pre-strips the `net.` nesting).
    ckpt_prefix: &'static str,
}

impl PidEngine {
    /// Build from explicit paths: the converted PiD checkpoint (a single `.safetensors`), the
    /// `gemma-2-2b-it` snapshot dir (weights + `tokenizer.json`), and the backbone latent-space tag
    /// (e.g. `"qwenimage"`). Errors on an unknown/out-of-scope backbone tag.
    pub fn load(checkpoint: &Path, gemma_dir: &Path, backbone: &str) -> Result<Self> {
        let spec = lookup(backbone).ok_or_else(|| {
            Error::Msg(format!(
                "pid: unknown/out-of-scope backbone {backbone:?} (no PiD latent-space mapping)"
            ))
        })?;
        // The released students share the sr4x PixDiT topology; only the LQ latent-channel count and
        // the latent grid's spatial compression differ per latent space: 16-ch / 8× for qwen/flux/sd3,
        // 4-ch / 8× for sdxl, and **128-ch / 16×** for flux2 (the packed BN latent — see the registry
        // `FLUX2` note, sc-7847). Both fields drive the LQ adapter geometry + `PidDecoder` output size.
        let mut cfg = PidConfig::sr4x();
        cfg.lq_latent_channels = spec.latent_channels;
        cfg.latent_spatial_down_factor = spec.latent_spatial_down_factor;

        let weights = Weights::from_file(checkpoint)?;

        // Gemma: prefer the merged single-file checkpoint, else load the snapshot dir's shards.
        let merged = gemma_dir.join(GEMMA_MERGED_FILE);
        let gw = if merged.is_file() {
            Weights::from_file(&merged)?
        } else {
            Weights::from_dir(gemma_dir)?
        };
        let gemma = Gemma2::from_weights(&gw, "model.", &Gemma2Config::gemma_2_2b())?;
        let caption = CaptionEncoder::new(gemma, gemma_dir.join("tokenizer.json"))?;

        Ok(Self {
            weights,
            cfg,
            sampler_cfg: SamplerConfig::distill_4step(),
            caption,
            ckpt_prefix: "",
        })
    }

    /// Build from a [`PidWeights`] load-spec component (the gen-core seam) for the given backbone tag.
    /// `checkpoint` must be a [`WeightsSource::File`] (the converted `.safetensors`); `gemma` must be a
    /// [`WeightsSource::Dir`] (the snapshot dir).
    pub fn from_spec(pid: &PidWeights, backbone: &str) -> Result<Self> {
        let checkpoint = file_path(&pid.checkpoint, "pid checkpoint")?;
        let gemma_dir = dir_path(&pid.gemma, "pid gemma encoder")?;
        Self::load(&checkpoint, &gemma_dir, backbone)
    }

    /// Spatial SR factor baked into the student (4× for every released backbone).
    pub fn scale(&self) -> i32 {
        self.cfg.sr_scale
    }

    /// VAE spatial compression (latent grid → pixel grid; 8 for the catalog VAEs).
    pub fn vae_compression(&self) -> i32 {
        self.cfg.latent_spatial_down_factor
    }

    /// Mint a per-generation [`PidDecoder`] bound to one caption. `sigma` is the LQ degrade level
    /// (0 for a clean-latent decode of a fully-denoised latent); `seed` drives the sampler's noise +
    /// per-step ε. Rebuilds the [`PidNet`] from the retained weights (cheap relative to decode) and
    /// encodes the caption to bf16 embeddings (the released inference dtype).
    pub fn decoder(&self, caption: &str, sigma: f32, seed: u64) -> Result<PidDecoder> {
        let net = PidNet::from_weights(&self.weights, self.ckpt_prefix, &self.cfg)?;
        let caption_embs = self.caption.encode(caption)?.as_dtype(Dtype::Bfloat16)?;
        Ok(PidDecoder::new(
            net,
            Sampler::new(&self.sampler_cfg),
            caption_embs,
            sigma,
            self.cfg.sr_scale,
            self.cfg.latent_spatial_down_factor,
            seed,
        ))
    }
}

/// Resolve the decode seam for one generation (epic 7840) — the shared entry point every PiD-eligible
/// provider calls (Qwen/Krea sc-7845; FLUX.1/Boogu/Chroma/Z-Image sc-7846; flux2/sdxl to follow). It
/// lives here in `mlx-gen-pid` rather than in a provider crate because the providers don't share a
/// dependency edge (Z-Image depends on neither Qwen-Image nor FLUX), but they all depend on this one.
///
/// When `req.use_pid` is set, mint a per-generation [`PidDecoder`] bound to the prompt — a **clean σ=0
/// decode of the fully-denoised latent**, seeded from `base_seed`; the caller passes it (as a
/// `&dyn LatentDecoder`) to its decode call site in place of the native VAE. Errors (rather than
/// silently falling back) if PiD was requested but the model was not loaded with `LoadSpec::pid`. When
/// the flag is unset, returns `None` and the caller uses the native VAE — the byte-exact default path.
///
/// `model_id` only labels the error. The returned decoder owns its caption embeddings + a freshly built
/// `PidNet`, so it lives as long as the borrow passed to the decode site; all `count` images in a
/// request share this one decoder (same prompt → same caption). The `from_ldm` early-stop x_t-capture
/// (σ>0, decoding a partially-denoised latent) is a separate follow-on — this path always decodes the
/// fully-denoised latent at σ=0.
pub fn resolve_pid_decoder(
    pid: Option<&PidEngine>,
    req: &GenerationRequest,
    base_seed: u64,
    model_id: &str,
) -> Result<Option<PidDecoder>> {
    if !req.use_pid {
        return Ok(None);
    }
    let engine = pid.ok_or_else(|| {
        Error::Msg(format!(
            "{model_id}: use_pid was requested but no PiD decoder is loaded (load with LoadSpec::pid)"
        ))
    })?;
    Ok(Some(engine.decoder(&req.prompt, 0.0, base_seed)?))
}

/// Extract the single-file path from a [`WeightsSource`], rejecting a directory.
fn file_path(src: &WeightsSource, what: &str) -> Result<PathBuf> {
    match src {
        WeightsSource::File(p) => Ok(p.clone()),
        WeightsSource::Dir(_) => Err(Error::Msg(format!(
            "{what}: expected the converted .safetensors file, got a directory"
        ))),
    }
}

/// Extract the directory path from a [`WeightsSource`], rejecting a single file.
fn dir_path(src: &WeightsSource, what: &str) -> Result<PathBuf> {
    match src {
        WeightsSource::Dir(p) => Ok(p.clone()),
        WeightsSource::File(_) => Err(Error::Msg(format!(
            "{what}: expected a snapshot directory, got a single file"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `PidEngine` is not `Debug` (it owns `Weights`/`CaptionEncoder`), so match rather than
    // `.expect_err()` (which would require `Debug` on the `Ok` payload).
    fn err_string<T>(r: Result<T>) -> String {
        match r {
            Ok(_) => panic!("expected an error"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn unknown_backbone_errors() {
        let err = err_string(PidEngine::load(
            Path::new("/nonexistent/ckpt.safetensors"),
            Path::new("/nonexistent/gemma"),
            "dinov2", // out-of-scope (vision-encoder latent, not a VAE latent)
        ));
        assert!(err.contains("out-of-scope backbone"), "got: {err}");
    }

    #[test]
    fn from_spec_rejects_swapped_sources() {
        // checkpoint must be a File, gemma must be a Dir — a swap is rejected before any load.
        let swapped = PidWeights {
            checkpoint: WeightsSource::Dir("/nonexistent/ckpt".into()),
            gemma: WeightsSource::Dir("/nonexistent/gemma".into()),
        };
        let err = err_string(PidEngine::from_spec(&swapped, "qwenimage"));
        assert!(err.contains("converted .safetensors file"), "got: {err}");
    }

    #[test]
    fn resolve_pid_decoder_off_is_none() {
        // use_pid unset → None (the native VAE path), even with no engine loaded.
        let req = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        assert!(resolve_pid_decoder(None, &req, 0, "some_model")
            .unwrap()
            .is_none());
    }

    #[test]
    fn resolve_pid_decoder_requested_without_engine_errors() {
        // use_pid set but no PiD loaded → a clear error, not a silent VAE fallback. `PidDecoder` is
        // not `Debug`, so match rather than `.expect_err()`.
        let req = GenerationRequest {
            prompt: "a fox".into(),
            use_pid: true,
            ..Default::default()
        };
        let err = err_string(resolve_pid_decoder(None, &req, 0, "some_model"));
        assert!(err.contains("no PiD decoder is loaded"), "got: {err}");
    }
}
