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

use mlx_gen::adapters::loader::apply_adapters_strict;
use mlx_gen::image::{decoded_to_image, validate_multiple_of_16};
use mlx_gen::media::Image;
use mlx_gen::runtime::AdapterSpec;
use mlx_gen::{
    resolve_flow_schedule, run_flow_sampler, CancelFlag, LatentDecoder, Progress, Result,
    TimestepConvention,
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

    /// Install Raw-trained LoRA/LoKr adapters onto the single-stream DiT (sc-7911). The shared
    /// [`apply_adapters_strict`] seam parses PEFT/diffusers/kohya/LoKr files, folds alpha/rank, and
    /// pushes a residual onto each matched `AdaptableLinear` — erroring (never silently dropping) on an
    /// adapter target that matches no module. The `Krea2Transformer` adapter host routes the trained
    /// `transformer_blocks.{i}.attn.{to_q,to_k,to_v,to_out.0}` paths (+ `text_fusion` + globals); the
    /// residual stacks over the (possibly already-quantized) base, so it composes with the Q8/Q4
    /// turnkey. Multiple + mixed LoRA/LoKr adapters stack by construction.
    pub fn apply_adapters(&mut self, specs: &[AdapterSpec]) -> Result<()> {
        apply_adapters_strict(&mut self.dit, specs, "krea_2")?;
        Ok(())
    }

    /// Generate one RGB image from a text prompt. Convenience wrapper over
    /// [`Self::generate_turbo_with_progress`] with no cancellation, a no-op progress sink, and the
    /// native VAE decode (no PiD).
    pub fn generate_turbo(&self, prompt: &str, opts: &TurboOptions) -> Result<Image> {
        // `keep = usize::MAX` → the full schedule (clean σ=0 decode; no from_ldm early-stop).
        self.generate_turbo_with_progress(
            prompt,
            opts,
            None,
            usize::MAX,
            &CancelFlag::new(),
            &mut |_| {},
        )
    }

    /// Generate one RGB image, streaming [`Progress`] and honoring `cancel` at each denoise step. A
    /// pre/mid-flight cancellation returns [`mlx_gen::Error::Canceled`]; the per-step `eval` (inside
    /// [`run_flow_sampler`]) bounds the lazy MLX graph so the cancel check can interrupt mid-render.
    /// `decoder` (epic 7840, sc-7845): the latent→pixel decode seam — `None` uses the native
    /// [`QwenVae`] (the byte-exact default), `Some` routes through a PiD super-resolving decoder
    /// (built per-generation from the prompt by the caller). The caller owns the PiD decoder so it can
    /// be reused across a batch (same prompt → same caption); PiD output is 4× the native resolution.
    ///
    /// `keep` (epic 7840, sc-7993) is the PiD `from_ldm` early-stop truncation: run only the first
    /// `keep` schedule entries so the denoise exits at a partially-denoised `x_k`, then hand that latent
    /// to the PiD `decoder` bound to the matching degrade σ. `usize::MAX` (the clean default) runs the
    /// full schedule (σ=0). The caller resolves `keep` + σ together from [`turbo_schedule`] via
    /// `mlx_gen_pid::flow_capture_for_request`, so the truncation and the decoder's σ always agree.
    pub fn generate_turbo_with_progress(
        &self,
        prompt: &str,
        opts: &TurboOptions,
        decoder: Option<&dyn LatentDecoder>,
        keep: usize,
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
        // `from_ldm` early-stop (sc-7993): truncate to `keep` entries (σ=0 clean path runs them all).
        let full = turbo_schedule(opts.steps, opts.scheduler.as_deref());
        let sigmas = &full[..keep.min(full.len())];
        let lat = run_flow_sampler(
            opts.sampler.as_deref(),
            TimestepConvention::Sigma,
            sigmas,
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
        self.decode_latents(&lat, decoder)
    }

    /// Decode a latent to an RGB image through the seam. `decoded_to_image` applies
    /// `clip(x·0.5 + 0.5, 0, 1)` — the algebraic equal of the reference `img.clamp(-1,1)·0.5 + 0.5` —
    /// and drops the singleton temporal axis when present (`QwenVae::decode` is NCTHW with T=1; PiD
    /// returns NCHW at 4× resolution). `decoder` is the native VAE when `None`.
    fn decode_latents(&self, lat: &Array, decoder: Option<&dyn LatentDecoder>) -> Result<Image> {
        let dec: &dyn LatentDecoder = decoder.unwrap_or(&self.vae);
        let decoded = dec.decode(lat)?.as_dtype(Dtype::Float32)?;
        decoded_to_image(&decoded)
    }
}

/// The Turbo flow-match sigma schedule for `steps` (native exponential-mu by default, or a curated
/// scheduler over the same mu). Length `steps + 1`, strictly descending with a trailing `0.0`. Exposed
/// so the caller can resolve a PiD `from_ldm` early-stop capture (sc-7993, via
/// `mlx_gen_pid::flow_capture_for_request`) before building the decoder — the same schedule
/// [`KreaPipeline::generate_turbo_with_progress`] then runs (the build is pure host math).
pub fn turbo_schedule(steps: usize, scheduler: Option<&str>) -> Vec<f32> {
    let native = turbo_sigmas(steps);
    resolve_flow_schedule(scheduler, TURBO_MU as f32, steps, &native)
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
