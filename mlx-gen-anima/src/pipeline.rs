//! The end-to-end Anima txt2img pipeline: prompt → (Qwen3 encode → mask-multiply → conditioner) →
//! DiT denoise (flow-match) → VAE decode → image. Transcribed from the diffusers Anima modular
//! pipeline (`encoders.py` / `before_denoise.py` / `denoise.py` / `decoders.py`).

use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::{random, Array, Dtype};

use mlx_gen::adapters::loader::ApplyReport;
use mlx_gen::image::decoded_to_image;
use mlx_gen::media::Image;
use mlx_gen::runtime::{AdapterSpec, CancelFlag};
use mlx_gen::{
    resolve_flow_schedule, run_flow_sampler, Error, Progress, Result, TimestepConvention,
    WeightsSource,
};

use crate::conditioner::AnimaTextConditioner;
use crate::config::{Variant, SIGMA_SHIFT, VAE_CHANNELS, VAE_COMPRESSION};
use crate::loader::AnimaComponents;
use crate::text_encoder::AnimaQwen3;
use crate::transformer::CosmosDiT;
use crate::vae::QwenVae;

/// Anima's recommended default sampler (the ER-SDE-3 solver added for this epic, sc-10519). A request
/// `sampler` overrides it; any curated flow solver (euler, dpmpp_2m, …) is valid.
pub const DEFAULT_SAMPLER: &str = "er_sde";

/// The Anima sigma schedule: `linspace(1.0, 1/N, N)` (**NOT** the diffusers default) time-shifted by
/// the static `shift=3.0` (`3σ / (1 + 2σ)`), with the trailing terminal `0.0` the flow sampler
/// integrates to. Length `N + 1`, descending.
pub fn anima_sigmas(steps: usize) -> Vec<f32> {
    let n = steps.max(1);
    let shift = SIGMA_SHIFT as f64;
    let mut sigmas: Vec<f32> = (0..n)
        .map(|i| {
            // linspace(1.0, 1/n, n)
            let s = if n == 1 {
                1.0
            } else {
                1.0 + (i as f64) * (1.0 / n as f64 - 1.0) / ((n - 1) as f64)
            };
            // static time-shift: shift·s / (1 + (shift−1)·s)
            (shift * s / (1.0 + (shift - 1.0) * s)) as f32
        })
        .collect();
    sigmas.push(0.0);
    sigmas
}

/// Resolve the Anima σ schedule, honoring a per-generation curated `scheduler` name (epic 7114
/// scheduler axis, sc-11123 / F-115). The engine advertises the full curated scheduler menu, so this
/// threads `req.scheduler` into the σ-schedule construction instead of ignoring it:
///
/// - `None`, an unknown, or a native-aliased name (`linear` / `flow_match_euler`) returns
///   [`anima_sigmas`] **byte-for-byte** — the N1 default-parity guarantee (never hard-fail a
///   generation over a scheduling knob; the shared floor already rejects any non-curated name).
/// - A curated name (`karras` / `simple` / `beta` / `beta57` / …) re-shapes σ over the SAME static
///   `shift=3.0` (`mu = ln(3)`), so a schedule stays consistent with Anima's time-shift rather than
///   degrading to a linear σ ramp. The scheduler picks WHERE the steps land; `req.sampler` still picks
///   the integrator.
pub fn anima_schedule(scheduler: Option<&str>, steps: usize) -> Vec<f32> {
    let native = anima_sigmas(steps);
    resolve_flow_schedule(scheduler, SIGMA_SHIFT.ln(), steps, &native)
}

/// Seeded initial latent noise `[1, 16, 1, H/8, W/8]` (f32 standard normal), the 5-D Cosmos latent.
fn create_noise(seed: u64, width: u32, height: u32) -> Result<Array> {
    let key = random::key(seed)?;
    let shape = [
        1,
        VAE_CHANNELS as i32,
        1,
        (height / VAE_COMPRESSION) as i32,
        (width / VAE_COMPRESSION) as i32,
    ];
    Ok(random::normal::<f32>(&shape[..], None, None, Some(&key))?)
}

/// Per-generation options.
pub struct GenOptions {
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    pub guidance: f32,
    pub seed: u64,
    /// Curated sampler name; `None` ⇒ [`DEFAULT_SAMPLER`].
    pub sampler: Option<String>,
    /// Curated scheduler name (epic 7114 scheduler axis); `None` ⇒ the native [`anima_sigmas`]
    /// schedule. Resolved via [`anima_schedule`].
    pub scheduler: Option<String>,
}

/// The assembled Anima pipeline.
pub struct AnimaPipeline {
    components: AnimaComponents,
}

impl AnimaPipeline {
    pub fn from_source(source: &WeightsSource, variant: Variant) -> Result<Self> {
        Ok(Self {
            components: AnimaComponents::load(source, variant)?,
        })
    }

    pub fn components(&self) -> &AnimaComponents {
        &self.components
    }

    /// Bake LoRA/LoKr adapters onto the DiT **and** the bundled `AnimaTextConditioner` at load time
    /// (sc-10521). Stacked + mixed LoRA/LoKr are supported by construction; an unmatched target is a
    /// hard error (strict). No-op for an empty spec list. Returns the [`ApplyReport`] (its `applied`
    /// count is 508 for the turbo LoRA — 448 DiT + 60 conditioner — and 448 for the DiT-only style
    /// LoRA). Applied on the still-mutable model during `load`, mirroring the Z-Image/Qwen seam.
    pub fn apply_adapters(&mut self, specs: &[AdapterSpec]) -> Result<ApplyReport> {
        crate::adapters::apply_anima_adapters(
            &mut self.components.dit,
            &mut self.components.conditioner,
            specs,
        )
    }

    /// Encode a prompt to the DiT's `encoder_hidden_states` `[1, 512, 1024]` (bf16): Qwen3
    /// `last_hidden_state` → **mask-multiply** (VERIFIED trap) → `AnimaTextConditioner`.
    ///
    /// ComfyUI-style `(text:weight)` emphasis (sc-10566) applies to the **T5 query-token path only**:
    /// the Qwen tower is tokenized on the de-weighted text (its token weights are forced to `1.0`),
    /// while the parsed per-token weights scale the conditioner output. See [`crate::prompt_weight`].
    pub fn encode_prompt(&self, prompt: &str) -> Result<Array> {
        self.encode_prompt_with(
            prompt,
            &self.components.text_encoder,
            &self.components.conditioner,
        )
    }

    /// [`encode_prompt`](Self::encode_prompt) with an EXPLICIT text-encoder + conditioner pair, so a
    /// caller can encode with the resident **bf16** modules or an **fp32-upcast reference** pair from
    /// [`crate::loader::load_conditioning_at_dtype`] (sc-10577). The conditioning dtype is whatever `te`
    /// produces (its `compute_dtype`), threaded into the conditioner — so `te` and `cond` MUST share a
    /// dtype (both bf16, the shipped default, or both fp32). With the resident bf16 modules this is
    /// byte-identical to the pre-sc-10577 `encode_prompt`.
    pub fn encode_prompt_with(
        &self,
        prompt: &str,
        te: &AnimaQwen3,
        cond: &AnimaTextConditioner,
    ) -> Result<Array> {
        let c = &self.components;
        // Qwen is weight-blind: strip the emphasis syntax to plain text before tokenizing (a no-op for
        // an unweighted prompt). This mirrors ComfyUI forcing the Qwen token weights to 1.0.
        let qwen_text = crate::prompt_weight::strip_prompt_weights(prompt);
        let (qwen_ids, qwen_mask) = c.tokenizers.encode_qwen(&qwen_text)?;
        let source = te.forward(&qwen_ids, &qwen_mask)?; // [1, S, 1024] in te.compute_dtype
                                                         // Multiply the Qwen states by the attention mask BEFORE the conditioner (zeros padded/uncond
                                                         // tokens) — the flagged trap. Batch-1 real prompts have an all-ones mask (no-op); the empty
                                                         // uncond prompt's single token (mask 0) is zeroed so the conditioner cross-attn contributes 0.
        let mask = qwen_mask.as_dtype(source.dtype())?.expand_dims(2)?; // [1, S, 1]
        let source = multiply(&source, &mask)?;
        // T5 carries the per-token weights (all 1.0 ⇒ strict no-op equal to the unweighted path).
        let (t5_ids, t5_weights) = c.tokenizers.encode_t5_weighted(prompt)?; // [1, St], len St
        cond.forward_weighted(&source, &t5_ids, Some(&t5_weights), source.dtype())
    }

    /// Generate one image. `negative` is used only when `variant.uses_cfg()`.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        prompt: &str,
        negative: &str,
        variant: Variant,
        opts: &GenOptions,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let noise = create_noise(opts.seed, opts.width, opts.height)?;
        let sampler = opts.sampler.as_deref().unwrap_or(DEFAULT_SAMPLER);
        // VAE decode (applies the baked latents_mean/std de-norm) → [1, 3, 1, H, W] f32 in [-1, 1].
        let latent = self.denoise(
            &noise,
            prompt,
            negative,
            variant,
            opts.steps,
            opts.guidance,
            sampler,
            opts.scheduler.as_deref(),
            opts.seed,
            Dtype::Bfloat16,
            cancel,
            on_progress,
        )?;
        let decoded = self.components.vae.decode(&latent)?;
        decoded_to_image(&decoded)
    }

    /// The flow denoise loop shared by [`generate`](Self::generate) and the stage-7 parity hook. Encodes
    /// the prompt (+ negative for CFG variants), then runs `sampler` over [`anima_sigmas`] from the given
    /// `init` latent, evaluating the DiT in `dtype`. Returns the final latent `[1, 16, 1, H/8, W/8]`
    /// (f32, pre-VAE). The DiT is a **standard flow denoiser**: it predicts `v ≈ ε − x0` and embeds the
    /// **raw σ** as its timestep, so the sampler (`TimestepConvention::Sigma`, `x + (σ_next − σ)·v`)
    /// consumes it directly — no negation, no `1 − σ` timestep.
    #[allow(clippy::too_many_arguments)]
    fn denoise(
        &self,
        init: &Array,
        prompt: &str,
        negative: &str,
        variant: Variant,
        steps: usize,
        guidance: f32,
        sampler: &str,
        scheduler: Option<&str>,
        seed: u64,
        dtype: Dtype,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        let cond = self.encode_prompt(prompt)?;
        let uncond = if variant.uses_cfg() {
            Some(self.encode_prompt(negative)?)
        } else {
            None
        };
        self.denoise_loop(
            init,
            &cond,
            uncond.as_ref(),
            steps,
            guidance,
            sampler,
            scheduler,
            seed,
            dtype,
            cancel,
            on_progress,
        )
    }

    /// The core flow-denoise loop given ALREADY-COMPUTED conditioning (sc-10577 decoupling): run
    /// `sampler` over [`anima_sigmas`] from `init`, evaluating the DiT in `dit_dtype`, with CFG when
    /// `uncond` is `Some`. Split out of [`denoise`](Self::denoise) so a caller can inject bf16 or fp32
    /// conditioning into the IDENTICAL DiT trajectory — the isolation measurement for the
    /// bf16-conditioning parity offset.
    #[allow(clippy::too_many_arguments)]
    fn denoise_loop(
        &self,
        init: &Array,
        cond: &Array,
        uncond: Option<&Array>,
        steps: usize,
        guidance: f32,
        sampler: &str,
        scheduler: Option<&str>,
        seed: u64,
        dit_dtype: Dtype,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        let sigmas = anima_schedule(scheduler, steps);
        let guidance = Array::from_slice(&[guidance], &[1]);
        let dit = &self.components.dit;
        let predict = |x: &Array, sigma: f32| -> Result<Array> {
            let s = Array::from_slice(&[sigma], &[1]);
            let v_cond = dit.forward(x, &s, cond, dit_dtype)?;
            let v = match uncond {
                // CFG: v = v_uncond + guidance·(v_cond − v_uncond).
                Some(u) => {
                    let v_u = dit.forward(x, &s, u, dit_dtype)?;
                    add(&v_u, &multiply(&subtract(&v_cond, &v_u)?, &guidance)?)?
                }
                None => v_cond,
            };
            // Integrate in f32 (the reference keeps latents f32).
            Ok(v.as_dtype(Dtype::Float32)?)
        };
        run_flow_sampler(
            Some(sampler),
            TimestepConvention::Sigma,
            &sigmas,
            init.clone(),
            seed,
            cancel,
            on_progress,
            predict,
        )
    }

    /// Test hook (sc-10577 isolation measurement): the deterministic stage-7 denoise from an **injected**
    /// latent using ALREADY-COMPUTED conditioning `cond` (+ `uncond` for CFG variants), so the caller
    /// feeds either the resident bf16 conditioning or an fp32-upcast reference through the identical
    /// injected-init + Euler + schedule trajectory. The DiT runs in `dit_dtype` (fp32 for the stage-7
    /// golden). The delta between the two runs' final latents IS the bf16-conditioning contribution.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_from_latent_with_conditioning(
        &self,
        init: &Array,
        cond: &Array,
        uncond: Option<&Array>,
        steps: usize,
        guidance: f32,
        sampler: &str,
        dit_dtype: Dtype,
    ) -> Result<Array> {
        let cancel = CancelFlag::default();
        let mut prog = |_p: Progress| {};
        // Parity hook: native schedule (scheduler = None) so the injected-init trajectory is byte-exact.
        self.denoise_loop(
            init, cond, uncond, steps, guidance, sampler, None, 0, dit_dtype, &cancel, &mut prog,
        )
    }

    /// Test hook (sc-10524 stage-7 golden): run the flow denoise from an **injected** initial latent
    /// (instead of sampling noise) with an explicit `sampler`/`dtype`, returning the final latent
    /// `[1, 16, 1, H/8, W/8]` (pre-VAE). Lets an MLX-vs-diffusers end-to-end comparison feed BOTH sides
    /// the identical starting point + a deterministic solver, so residual drift is float error, not the
    /// chaos of two independently-sampled noises.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_from_latent(
        &self,
        init: &Array,
        prompt: &str,
        negative: &str,
        variant: Variant,
        steps: usize,
        guidance: f32,
        sampler: &str,
        dtype: Dtype,
    ) -> Result<Array> {
        let cancel = CancelFlag::default();
        let mut prog = |_p: Progress| {};
        // Parity hook: native schedule (scheduler = None) so the injected-init trajectory is byte-exact.
        self.denoise(
            init, prompt, negative, variant, steps, guidance, sampler, None, 0, dtype, &cancel,
            &mut prog,
        )
    }

    /// Test hook (sc-10524 stage-7 golden, intermediate-latent capture): run the same deterministic
    /// denoise as [`denoise_from_latent`], additionally snapshotting the latent AFTER each step count in
    /// `capture_after` (`x_k` = the state after `k` Euler steps). Returns `(final_latent, [(k, x_k)])`.
    ///
    /// `x_k` is exactly the input the `(k+1)`-th DiT call sees (the sampler calls `predict(x_k, σ_k)` at
    /// step `k`), so it matches the Python generator's `caps[k]` bit-for-bit in definition. Comparing the
    /// per-step deltas lets the parity test distinguish a systematic BIAS (the MLX bf16-conditioning lock —
    /// present from step 1) from diffuse float ACCUMULATION (grows with step count).
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_from_latent_capture(
        &self,
        init: &Array,
        prompt: &str,
        negative: &str,
        variant: Variant,
        steps: usize,
        guidance: f32,
        sampler: &str,
        dtype: Dtype,
        capture_after: &[usize],
    ) -> Result<(Array, Vec<(usize, Array)>)> {
        let cancel = CancelFlag::default();
        let mut prog = |_p: Progress| {};
        let cond = self.encode_prompt(prompt)?;
        let uncond = if variant.uses_cfg() {
            Some(self.encode_prompt(negative)?)
        } else {
            None
        };
        let sigmas = anima_sigmas(steps);
        let guidance = Array::from_slice(&[guidance], &[1]);
        let dit = &self.components.dit;
        let want: std::collections::HashSet<usize> = capture_after.iter().copied().collect();
        let call = std::cell::Cell::new(0usize);
        let captures: std::cell::RefCell<Vec<(usize, Array)>> = std::cell::RefCell::new(Vec::new());
        let predict = |x: &Array, sigma: f32| -> Result<Array> {
            // The sampler calls predict(x_k, σ_k) at step k, so `x` here is the state after k steps.
            let k = call.get();
            if want.contains(&k) {
                let snap = x.as_dtype(Dtype::Float32)?;
                mlx_rs::transforms::eval([&snap])?;
                captures.borrow_mut().push((k, snap));
            }
            call.set(k + 1);
            let s = Array::from_slice(&[sigma], &[1]);
            let v_cond = dit.forward(x, &s, &cond, dtype)?;
            let v = match &uncond {
                Some(u) => {
                    let v_u = dit.forward(x, &s, u, dtype)?;
                    add(&v_u, &multiply(&subtract(&v_cond, &v_u)?, &guidance)?)?
                }
                None => v_cond,
            };
            Ok(v.as_dtype(Dtype::Float32)?)
        };
        let final_latent = run_flow_sampler(
            Some(sampler),
            TimestepConvention::Sigma,
            &sigmas,
            init.clone(),
            0,
            &cancel,
            &mut prog,
            predict,
        )?;
        Ok((final_latent, captures.into_inner()))
    }
}

// ==================================================================================================
// In-training preview sampling (sc-10641)
// ==================================================================================================
//
// The trainer renders a periodic preview from the IN-PROGRESS adapter so the user can watch the LoRA
// converge (the sc-5637 `TrainingProgress::Sample` contract). Unlike z-image (one DiT host), an Anima
// preview MUST run the conditioner (`llm_adapter`) forward through the LIVE graph: its 60 adapter
// targets are trained, so reusing a cached conditioner OUTPUT would render a model whose conditioner
// adapters are silently inert — the exact sc-10522 trap. The trainer therefore caches the conditioner
// INPUTS (masked Qwen3 states + T5 ids) and calls [`render_preview`], which re-runs the conditioner
// (with the current adapters installed) every preview. These helpers are forward-only (no gradients)
// and mirror the inference denoise ([`AnimaPipeline::denoise`]): [`anima_sigmas`] + [`run_flow_sampler`]
// over [`TimestepConvention::Sigma`], CFG-combining a live uncond conditioner forward on CFG variants.

/// Render one preview image from the in-training adapter (sc-10641): run the (live) conditioner, denoise
/// the DiT from seeded noise, VAE-decode → RGB. `source`/`t5_ids` are the cached conditioner inputs for
/// the positive prompt; `uncond` is the cached empty-prompt inputs (CFG variants only). Forward-only.
///
/// The conditioner forward here is the sc-10522 correctness point: the DiT + conditioner adapters must
/// already be installed on `dit`/`conditioner` (the trainer re-installs the current factors before each
/// cadence), so this reflects the conditioner adapters' training — it is NOT a cached output.
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_preview(
    dit: &CosmosDiT,
    conditioner: &AnimaTextConditioner,
    vae: &QwenVae,
    source: &Array,
    t5_ids: &Array,
    uncond: Option<&(Array, Array)>,
    steps: usize,
    guidance: f32,
    edge: u32,
    seed: u64,
    dtype: Dtype,
    cancel: &CancelFlag,
) -> Result<Image> {
    let init = create_noise(seed, edge, edge)?;
    let latent = render_preview_latent(
        dit,
        conditioner,
        source,
        t5_ids,
        uncond,
        &init,
        steps,
        guidance,
        seed,
        dtype,
        cancel,
    )?;
    // F-117: honor a cancel that arrived during the preview denoise before the (lazy) VAE decode.
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    let decoded = vae.decode(&latent)?; // [1, 3, 1, H, W] f32 in [-1, 1]
    decoded_to_image(&decoded)
}

/// The VAE-free core of [`render_preview`]: run the LIVE conditioner (positive + optional uncond) then
/// denoise the DiT from `init`, returning the final latent `[1, 16, 1, H/8, W/8]`. Split out so a
/// weights-free (no-VAE) test can drive the exact production path and prove the conditioner is live —
/// if this ever regressed to a cached conditioner output, the latent would stop responding to the
/// conditioner adapters (the sc-10522 trap) and the guard test reddens.
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_preview_latent(
    dit: &CosmosDiT,
    conditioner: &AnimaTextConditioner,
    source: &Array,
    t5_ids: &Array,
    uncond: Option<&(Array, Array)>,
    init: &Array,
    steps: usize,
    guidance: f32,
    seed: u64,
    dtype: Dtype,
    cancel: &CancelFlag,
) -> Result<Array> {
    // LIVE conditioner forward — reflects the in-training `llm_adapter` adapters (sc-10522).
    let cond = conditioner.forward(source, t5_ids, dtype)?;
    // CFG (base/aesthetic): a live uncond conditioner forward too. Skipped at guidance 1.0 (the
    // combination collapses to the positive forward) and for guidance-free variants (uncond == None).
    let uncond_enc = match uncond {
        Some((s, ids)) if guidance != 1.0 => Some(conditioner.forward(s, ids, dtype)?),
        _ => None,
    };
    render_latent_with_enc(
        dit,
        &cond,
        uncond_enc.as_ref(),
        init,
        steps,
        guidance,
        seed,
        dtype,
        cancel,
    )
}

/// Denoise the DiT from `init` over [`anima_sigmas`] using a PRE-COMPUTED conditioner output `cond`
/// (+ optional `uncond_enc` for CFG). The sampling half of [`render_preview_latent`], factored so the
/// live path and the guard test share one integrator and differ ONLY in whether `cond` was produced
/// live or (the trap) reused stale.
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_latent_with_enc(
    dit: &CosmosDiT,
    cond: &Array,
    uncond_enc: Option<&Array>,
    init: &Array,
    steps: usize,
    guidance: f32,
    seed: u64,
    dtype: Dtype,
    cancel: &CancelFlag,
) -> Result<Array> {
    let sigmas = anima_sigmas(steps.max(1));
    let guidance = Array::from_slice(&[guidance], &[1]);
    let mut prog = |_p: Progress| {};
    let predict = |x: &Array, sigma: f32| -> Result<Array> {
        let s = Array::from_slice(&[sigma], &[1]);
        let v_cond = dit.forward(x, &s, cond, dtype)?;
        let v = match uncond_enc {
            // CFG: v = v_uncond + guidance·(v_cond − v_uncond).
            Some(u) => {
                let v_u = dit.forward(x, &s, u, dtype)?;
                add(&v_u, &multiply(&subtract(&v_cond, &v_u)?, &guidance)?)?
            }
            None => v_cond,
        };
        Ok(v.as_dtype(Dtype::Float32)?)
    };
    run_flow_sampler(
        Some(DEFAULT_SAMPLER),
        TimestepConvention::Sigma,
        &sigmas,
        init.clone(),
        seed,
        cancel,
        &mut prog,
        predict,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigma_schedule_linspace_shift3() {
        // N=10 ⇒ linspace(1.0, 0.1, 10) time-shifted by 3.0, trailing 0.0. Length 11.
        let s = anima_sigmas(10);
        assert_eq!(s.len(), 11);
        // shift(σ) = 3σ/(1+2σ): shift(1.0)=1.0, shift(0.9)=2.7/2.8, shift(0.1)=0.3/1.2=0.25.
        assert!((s[0] - 1.0).abs() < 1e-6, "s0={}", s[0]);
        assert!(
            (s[1] - (2.7 / 2.8)) < 1e-5 && (s[1] - (2.7 / 2.8)).abs() < 1e-5,
            "s1={}",
            s[1]
        );
        assert!((s[9] - 0.25).abs() < 1e-5, "s9={}", s[9]);
        assert_eq!(s[10], 0.0);
        // strictly descending (a valid flow schedule).
        for w in s.windows(2) {
            assert!(w[0] > w[1], "not descending: {} !> {}", w[0], w[1]);
        }
    }

    #[test]
    fn sigma_schedule_turbo_10_and_base_30_lengths() {
        assert_eq!(anima_sigmas(10).len(), 11);
        assert_eq!(anima_sigmas(30).len(), 31);
        assert_eq!(anima_sigmas(1), vec![1.0, 0.0]); // shift(1.0)=1.0
    }

    // ----- sc-11123 / F-115: req.scheduler is wired into the σ schedule (was advertised-but-inert) -----

    /// The default path (`scheduler == None`) MUST return the native [`anima_sigmas`] schedule
    /// **byte-for-byte** — the epic-7114 N1 default-parity guarantee. Any drift here would silently
    /// change every existing (scheduler-less) Anima generation.
    #[test]
    fn schedule_default_none_is_native_byte_for_byte() {
        for steps in [1usize, 4, 10, 30] {
            assert_eq!(
                anima_schedule(None, steps),
                anima_sigmas(steps),
                "scheduler=None must equal the native schedule byte-for-byte (steps={steps})"
            );
        }
    }

    /// An unknown name or a native alias falls back to the native schedule (N3 — never hard-fail a
    /// generation over a scheduling knob; the shared capability floor already rejects non-curated names
    /// before we get here, so this is defense-in-depth for aliases).
    #[test]
    fn schedule_unknown_and_native_aliases_fall_back_to_native() {
        let native = anima_sigmas(10);
        for name in ["", "not_a_scheduler", "linear", "flow_match_euler"] {
            assert_eq!(
                anima_schedule(Some(name), 10),
                native,
                "unknown/native-aliased scheduler {name:?} must fall back to native"
            );
        }
    }

    /// EVERY advertised curated scheduler must produce a valid descending σ schedule terminating in
    /// `0.0` — proving `req.scheduler` is actually consumed, not dropped. This is the exact
    /// `curated_scheduler_names()` menu the descriptor advertises (F-115: advertised == honored).
    #[test]
    fn schedule_every_advertised_scheduler_is_valid() {
        for name in mlx_gen::curated_scheduler_names() {
            let s = anima_schedule(Some(name), 10);
            assert!(s.len() >= 2, "{name}: schedule too short ({})", s.len());
            assert_eq!(*s.last().unwrap(), 0.0, "{name}: must terminate at 0.0");
            // Strictly descending (a valid flow schedule).
            for w in s.windows(2) {
                assert!(w[0] > w[1], "{name}: not descending: {} !> {}", w[0], w[1]);
            }
        }
    }

    /// The structurally-distinct curated schedulers (they re-distribute where the steps land) MUST
    /// change the schedule vs the native default — the load-bearing proof that the wiring is live and
    /// not a silent fallback. `karras`/`exponential`/`beta`/`beta57`/`ddim_uniform` are the ones that
    /// visibly diverge from Anima's `linspace(1,1/N,N)`-through-shift ramp; `normal`/`sgm_uniform`/
    /// `simple` can nearly coincide with it, so they are not asserted to differ (only to be valid,
    /// above).
    #[test]
    fn schedule_distinct_schedulers_differ_from_native() {
        let native = anima_sigmas(10);
        for name in ["karras", "exponential", "beta", "beta57", "ddim_uniform"] {
            let s = anima_schedule(Some(name), 10);
            let differs =
                s.len() != native.len() || s.iter().zip(&native).any(|(a, b)| (a - b).abs() > 1e-4);
            assert!(
                differs,
                "{name}: must re-shape the schedule vs native (req.scheduler ignored?)"
            );
        }
    }
}
