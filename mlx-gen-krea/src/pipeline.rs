//! Krea 2 **Turbo** text-to-image pipeline (sc-7571) — the vertical that makes Krea 2 runnable:
//! tokenize → Qwen3-VL-4B condition-encode (the 12-layer select stack) → DiT (text_fusion aggregator +
//! single-stream denoise) → Qwen-Image VAE decode. Port of the reference `sampling.py::sample` Turbo
//! path.
//!
//! **CFG-free.** The TDM distillation baked the guided velocity into the weights, so there is no
//! unconditional branch (`guidance == 0` in the reference) — one DiT forward per step. Per-sample
//! `B = 1`: one prompt → no padding → `mask = None` (the DiT runs the full valid context).
//!
//! **Rectified-flow v-param Euler.** The DiT consumes the raw sigma as its timestep
//! ([`TimestepConvention::Sigma`]; it scales ×1000 internally) and predicts the flow velocity
//! directly, so the core [`run_flow_sampler`] Euler step `x + v·(σ_{i+1} − σ_i)` is exactly the
//! reference `img += (tprev − tcurr)·v`. The native exponential-mu schedule ([`turbo_sigmas`]) is the
//! byte-exact default; a per-generation curated sampler/scheduler (epic 7114) reshapes over the same
//! mu. The `clamp(-1,1)` + denormalize the reference applies after decode lives in `decoded_to_image`
//! (`clip(x·0.5 + 0.5, 0, 1)`, the algebraic equal).

use mlx_rs::{random, Array, Dtype};

use mlx_gen::image::{decoded_to_image, validate_multiple_of_16};
use mlx_gen::media::Image;
use mlx_gen::{
    resolve_flow_schedule, run_flow_sampler, CancelFlag, Progress, Result, TimestepConvention,
};

use std::path::Path;

use crate::loader::{load_text_encoder, load_transformer};
use crate::schedule::{turbo_sigmas, TURBO_MU};
use crate::text_encoder::{KreaTextEncoder, KreaTokenizer};
use crate::transformer::Krea2Transformer;
use crate::vae::{load_vae, QwenVae};

/// Turbo text-to-image knobs, resolved from the [`crate::model`] request. Dimensions are validated at
/// the Generator layer (multiple-of-16, in the resolution range) before the pipeline runs.
pub struct TurboOptions {
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    pub seed: u64,
    /// Curated sampler override (epic 7114). `None` = the native byte-exact rectified-flow Euler.
    pub sampler: Option<String>,
    /// Curated scheduler override. `None` = the native exponential-mu schedule.
    pub scheduler: Option<String>,
}

/// The assembled Krea 2 Turbo pipeline: tokenizer + Qwen3-VL-4B condition encoder + single-stream DiT
/// + Qwen-Image VAE.
pub struct KreaPipeline {
    tok: KreaTokenizer,
    te: KreaTextEncoder,
    dit: Krea2Transformer,
    vae: QwenVae,
}

impl KreaPipeline {
    /// Load all Turbo components from a Krea 2 snapshot (`tokenizer/ text_encoder/ transformer/ vae/`).
    pub fn from_snapshot(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        Ok(Self {
            tok: KreaTokenizer::from_snapshot(root)?,
            te: load_text_encoder(root)?,
            dit: load_transformer(root)?,
            vae: load_vae(root)?,
        })
    }

    /// Quantize the DiT + text-encoder Linears in place (group-wise affine Q4/Q8); the VAE stays dense
    /// (the published `vae/` is f32), matching the converter's quant-target set. A no-op on an
    /// already-packed snapshot (`AdaptableLinear::quantize` skips quantized bases).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.te.quantize(bits)?;
        self.dit.quantize(bits)?;
        Ok(())
    }

    /// Generate one RGB image from a text prompt. Convenience wrapper over
    /// [`Self::generate_turbo_with_progress`] with no cancellation and a no-op progress sink.
    pub fn generate_turbo(&self, prompt: &str, opts: &TurboOptions) -> Result<Image> {
        self.generate_turbo_with_progress(prompt, opts, &CancelFlag::new(), &mut |_| {})
    }

    /// Generate one RGB image, streaming [`Progress`] and honoring `cancel` at each denoise step. A
    /// pre/mid-flight cancellation returns [`mlx_gen::Error::Canceled`]; the per-step `eval` (inside
    /// [`run_flow_sampler`]) bounds the lazy MLX graph so the cancel check can interrupt mid-render.
    pub fn generate_turbo_with_progress(
        &self,
        prompt: &str,
        opts: &TurboOptions,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        validate_multiple_of_16(opts.width, opts.height, "krea_2_turbo")?;

        // Condition encoding: the 12 selected Qwen3-VL hidden layers, stacked + prefix-dropped → the
        // DiT's text_fusion context [1, n_tok, 12, 2560]. CFG-free, B=1 → mask = None.
        let (ids, attn) = self.tok.encode_prompt(prompt)?;
        let context = self.te.forward(&ids, &attn)?;

        // Initial latent noise [1, 16, H/8, W/8] (f32; the DiT casts to its compute dtype).
        let noise = init_noise(opts.height, opts.width, opts.seed)?;

        // Native exponential-mu Turbo sigmas are the byte-exact default; a curated scheduler reshapes
        // over the same mu. Raw sigma → DiT timestep, raw velocity → Euler `x + v·(σ_{i+1} − σ_i)`.
        let native = turbo_sigmas(opts.steps);
        let sigmas = resolve_flow_schedule(
            opts.scheduler.as_deref(),
            TURBO_MU as f32,
            opts.steps,
            &native,
        );
        let lat = run_flow_sampler(
            opts.sampler.as_deref(),
            TimestepConvention::Sigma,
            &sigmas,
            noise,
            opts.seed,
            cancel,
            on_progress,
            |x, timestep| {
                let t = Array::from_slice(&[timestep], &[1]);
                let v = self.dit.forward(x, &t, &context, None)?;
                Ok(v.as_dtype(Dtype::Float32)?)
            },
        )?;

        on_progress(Progress::Decoding);
        self.decode_latents(&lat)
    }

    /// VAE-decode a latent to an RGB image. `decoded_to_image` applies `clip(x·0.5 + 0.5, 0, 1)` — the
    /// algebraic equal of the reference `img.clamp(-1,1)·0.5 + 0.5` — and drops the singleton temporal
    /// axis (`QwenVae::decode` is NCTHW with T=1).
    fn decode_latents(&self, lat: &Array) -> Result<Image> {
        let decoded = self.vae.decode(lat)?.as_dtype(Dtype::Float32)?; // [1, 3, 1, H, W]
        decoded_to_image(&decoded)
    }
}

/// Seeded initial Gaussian latent noise `[1, 16, H/8, W/8]` (f32; the VAE's 8× spatial compression).
/// The model layer offsets `seed` per image in a batch, mirroring the reference `seed + i`.
fn init_noise(height: u32, width: u32, seed: u64) -> Result<Array> {
    let (hl, wl) = ((height / 8) as i32, (width / 8) as i32);
    let key = random::key(seed)?;
    Ok(random::normal::<f32>(
        &[1, 16, hl, wl],
        None,
        None,
        Some(&key),
    )?)
}
