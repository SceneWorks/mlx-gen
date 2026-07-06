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

use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::{random, Array, Dtype};

use mlx_gen::adapters::loader::apply_adapters_strict;
use mlx_gen::array::scalar;
use mlx_gen::image::{decoded_to_image, validate_multiple_of_16};
use mlx_gen::img2img::{add_noise_by_interpolation, init_time_step, preprocess_init_image};
use mlx_gen::media::Image;
use mlx_gen::runtime::AdapterSpec;
use mlx_gen::{
    resolve_flow_schedule, run_flow_sampler, CancelFlag, LatentDecoder, Progress, Result,
    TimestepConvention,
};

use std::path::Path;

use crate::loader::{load_text_encoder, load_transformer};
use crate::schedule::{dynamic_mu, krea_sigmas, turbo_sigmas, TURBO_MU};
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
        // Hoist the step-invariant text fusion + joint RoPE out of the per-step closure (F-079): the
        // context (and the latent geometry) are fixed across the denoise, so `prepare` runs ONCE and
        // every step reuses `prep` via `forward_prepared` (bit-identical to the per-step `forward`).
        let prep = self.dit.prepare(&context, None, &noise)?;
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
                let v = self.dit.forward_prepared(x, &t, &prep)?;
                Ok(v.as_dtype(Dtype::Float32)?)
            },
        )?;

        on_progress(Progress::Decoding);
        self.decode_latents(&lat, decoder)
    }

    /// Generate one RGB image seeded from a reference `init` image at the given `strength` (img2img,
    /// slice A / sc-8590). Convenience wrapper over [`Self::generate_turbo_img2img_with_progress`] with
    /// no cancellation, a no-op progress sink, and the native VAE decode (no PiD).
    pub fn generate_turbo_img2img(
        &self,
        prompt: &str,
        init: &Image,
        strength: f32,
        opts: &TurboOptions,
    ) -> Result<Image> {
        self.generate_turbo_img2img_with_progress(
            prompt,
            init,
            strength,
            opts,
            None,
            &CancelFlag::new(),
            &mut |_| {},
        )
    }

    /// **img2img latent-init on Turbo** (epic 8588 slice A; sc-8589 validated the strength window,
    /// sc-8590 productionized this). Generate one RGB image seeded from a reference `init` at the given
    /// `strength` — reference fidelity in the fork's [`init_time_step`] convention: higher strength →
    /// later start step → fewer denoise steps → the output stays closer to the reference; `strength ≤ 0`
    /// degenerates to a full txt2img (start step 0, identical to [`Self::generate_turbo_with_progress`]).
    ///
    /// Reuses the proven shared [`mlx_gen::img2img`] leaves (the Qwen-Image / Z-Image img2img path) with
    /// Krea's **unpacked** `[1, 16, H/8, W/8]` latent (the DiT patchifies internally, so — unlike Qwen —
    /// the clean latent is NOT pre-packed): VAE-encode the reference into the same normalized latent
    /// space as [`init_noise`] (`QwenVae::encode` returns `(e − mean)/std`), blend
    /// `(1 − σ_k)·clean + σ_k·noise` at the start sigma `σ_k = sigmas[k]`, and run the rectified-flow
    /// Euler sampler over `sigmas[k..]`. Distilled Turbo is CFG-free → one DiT forward per step (no
    /// guidance branch), and the step-invariant text-fusion + joint RoPE are prepared ONCE (the F-079
    /// hoist), exactly like [`Self::generate_turbo_with_progress`].
    ///
    /// `decoder` is the same latent→pixel seam as the t2i path (`None` = native [`QwenVae`], `Some` = a
    /// PiD super-resolving decoder over the final clean latent). The PiD `from_ldm` early-stop (`keep`)
    /// is intentionally NOT plumbed here: combining a partial-denoise capture with the img2img start
    /// step needs the caller to resolve the capture against the *sliced* schedule — tracked separately
    /// so it is not silently dropped. Cancellation + progress stream through [`run_flow_sampler`].
    #[allow(clippy::too_many_arguments)]
    pub fn generate_turbo_img2img_with_progress(
        &self,
        prompt: &str,
        init: &Image,
        strength: f32,
        opts: &TurboOptions,
        decoder: Option<&dyn LatentDecoder>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        validate_multiple_of_16(opts.width, opts.height, "krea_2_turbo")?;

        let (ids, attn) = self.tok.encode_prompt(prompt)?;
        let context = self.te.forward(&ids, &attn)?;

        // Reference → clean latent [1, 16, H/8, W/8]. `QwenVae::encode` already returns the normalized
        // `(e − mean)/std` latent (the same space as `init_noise`); drop the singleton temporal axis.
        let image_nchw = preprocess_init_image(init, opts.width, opts.height)?;
        let clean = self.vae.encode(&image_nchw)?.squeeze_axes(&[2])?;

        let noise = init_noise(opts.height, opts.width, opts.seed)?;

        // Start step from strength; blend the clean latent with noise at σ_k, then denoise sigmas[k..].
        let full = turbo_schedule(opts.steps, opts.scheduler.as_deref());
        let start = init_time_step(opts.steps, Some(strength)).min(full.len().saturating_sub(1));
        let sigmas = &full[start..];
        let x_start = add_noise_by_interpolation(&clean, &noise, full[start])?;

        let prep = self.dit.prepare(&context, None, &x_start)?;
        let lat = run_flow_sampler(
            opts.sampler.as_deref(),
            TimestepConvention::Sigma,
            sigmas,
            x_start,
            opts.seed,
            cancel,
            on_progress,
            |x, timestep| {
                let t = Array::from_slice(&[timestep], &[1]);
                let v = self.dit.forward_prepared(x, &t, &prep)?;
                Ok(v.as_dtype(Dtype::Float32)?)
            },
        )?;

        on_progress(Progress::Decoding);
        self.decode_latents(&lat, decoder)
    }

    /// Generate one RGB image through the **Raw** classifier-free-guidance path with no cancellation, a
    /// no-op progress sink, and the native VAE decode (no PiD). Convenience wrapper over
    /// [`Self::generate_base_with_progress`].
    pub fn generate_base(
        &self,
        prompt: &str,
        negative_prompt: &str,
        guidance: f32,
        opts: &TurboOptions,
    ) -> Result<Image> {
        self.generate_base_with_progress(
            prompt,
            negative_prompt,
            guidance,
            opts,
            None,
            usize::MAX,
            &CancelFlag::new(),
            &mut |_| {},
        )
    }

    /// Generate one RGB image through the **Raw** classifier-free-guidance path (`krea_2_raw`): the
    /// undistilled 12B DiT with a real guidance scale + optional user negative prompt (reference
    /// `sampling.py::sample` with `guidance > 0`). Two DiT forwards per step — the conditional (positive
    /// prompt) and the unconditional (the negative prompt, or `""` when none) — combined by the
    /// **reference** formula `v = cond + guidance·(cond − uncond)` (NOT the standard
    /// `uncond + g·(cond − uncond)`: Krea's guidance is offset by one). `guidance ≤ 0` collapses to a
    /// single conditional forward (the uncond context is never encoded), matching the reference
    /// `cfg = guidance > 0` short-circuit.
    ///
    /// The Raw schedule is resolution-**dynamic** ([`base_schedule`] / [`dynamic_mu`]), unlike the
    /// distilled Turbo's fixed `mu = 1.15`. `decoder` / `keep` are the same PiD decode + `from_ldm`
    /// early-stop seam as [`Self::generate_turbo_with_progress`] (Krea reuses the Qwen-Image latent
    /// space); the caller resolves them from [`base_schedule`] so the truncation and the decoder's σ
    /// agree. Both the positive and unconditional contexts are prepared ONCE (the F-079 hoist), so each
    /// step reuses the two `JointPrep`s.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_base_with_progress(
        &self,
        prompt: &str,
        negative_prompt: &str,
        guidance: f32,
        opts: &TurboOptions,
        decoder: Option<&dyn LatentDecoder>,
        keep: usize,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        validate_multiple_of_16(opts.width, opts.height, "krea_2_raw")?;

        // Positive (conditional) condition encoding → step-invariant prep (the Turbo F-079 hoist).
        let (ids, attn) = self.tok.encode_prompt(prompt)?;
        let context = self.te.forward(&ids, &attn)?;

        // Initial latent noise [1, 16, H/8, W/8] — shared by both prep branches (same geometry).
        let noise = init_noise(opts.height, opts.width, opts.seed)?;
        let prep_pos = self.dit.prepare(&context, None, &noise)?;

        // Unconditional branch, encoded + prepared ONLY when CFG is active (reference
        // `cfg = guidance > 0`). The negative prompt defaults to `""` at the caller (reference
        // `negative_prompts = [""] * n`).
        let prep_neg = if guidance > 0.0 {
            let (nids, nattn) = self.tok.encode_prompt(negative_prompt)?;
            let ncontext = self.te.forward(&nids, &nattn)?;
            Some(self.dit.prepare(&ncontext, None, &noise)?)
        } else {
            None
        };

        // Resolution-dynamic Raw schedule (mu from image-token count), truncated to `keep` for a PiD
        // `from_ldm` early-stop (σ=0 clean path runs them all).
        let full = base_schedule(
            opts.steps,
            opts.width,
            opts.height,
            opts.scheduler.as_deref(),
        );
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
                let cond = self.dit.forward_prepared(x, &t, &prep_pos)?;
                let v = match &prep_neg {
                    Some(neg) => {
                        let uncond = self.dit.forward_prepared(x, &t, neg)?;
                        krea_cfg_combine(&cond, &uncond, guidance)?
                    }
                    None => cond,
                };
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

/// The **Raw** flow-match sigma schedule for `steps` at a given resolution: the exponential-mu shift
/// with a resolution-**dynamic** `mu` interpolated in image-token count ([`dynamic_mu`]), unlike the
/// Turbo fixed `mu = 1.15`. Length `steps + 1`, descending with a trailing `0.0`; a curated scheduler
/// (epic 7114) reshapes over the same dynamic mu. Exposed so the caller can resolve a PiD `from_ldm`
/// early-stop capture from the same schedule [`KreaPipeline::generate_base_with_progress`] runs.
pub fn base_schedule(steps: usize, width: u32, height: u32, scheduler: Option<&str>) -> Vec<f32> {
    // Image token count = (W/16)·(H/16) (latent /8 then patch /2) — the reference `x.shape[1]`.
    let seq_len = (width as f64 / 16.0) * (height as f64 / 16.0);
    let mu = dynamic_mu(seq_len);
    let native = krea_sigmas(steps, mu);
    resolve_flow_schedule(scheduler, mu as f32, steps, &native)
}

/// Krea's classifier-free-guidance velocity combine — the reference `sampling.py:129`
/// `v = v_cond + guidance·(v_cond − v_uncond)`, **NOT** the standard `v_uncond + g·Δ`. Krea's guidance
/// is offset by one: the standard form applies one full step LESS guidance, and at `guidance = 1.0`
/// collapses to exactly `v_cond` (zero effective CFG). Single source of truth so the Raw inference path
/// and the trainer preview (`training::render_sample`) can never drift again (sc-10009). The caller runs
/// this only for `guidance > 0` (a single conditional forward otherwise, matching `cfg = guidance > 0`).
pub(crate) fn krea_cfg_combine(v_cond: &Array, v_uncond: &Array, guidance: f32) -> Result<Array> {
    Ok(add(
        v_cond,
        &multiply(&subtract(v_cond, v_uncond)?, scalar(guidance))?,
    )?)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The Krea CFG combine is the reference `cond + g·(cond − uncond)`, not the standard
    /// `uncond + g·Δ`. With cond = 2, uncond = 1 (Δ = 1): g = 1 → 3 (the standard form would give 2 —
    /// exactly `cond` — which is why the default `sample_guidance_scale = 1.0` washed previews out); a
    /// larger g pushes further from cond, away from uncond.
    #[test]
    fn cfg_combine_is_reference_offset_by_one() {
        let cond = Array::from_slice(&[2.0f32], &[1]);
        let uncond = Array::from_slice(&[1.0f32], &[1]);
        for (g, want) in [(1.0f32, 3.0f32), (3.5, 5.5), (0.0, 2.0)] {
            let v = krea_cfg_combine(&cond, &uncond, g).unwrap();
            assert!(
                (v.item::<f32>() - want).abs() < 1e-5,
                "g={g}: got {}, want {want}",
                v.item::<f32>()
            );
        }
    }
}
