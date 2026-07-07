//! SANA text-to-image sampling pipeline (epic 8485, story sc-8489 — **Phase A: the mlx-gen side**).
//!
//! Composes the three already-merged native SANA components into one end-to-end prompt→image path:
//!
//! ```text
//!  prompt ─▶ SanaTextEncoder (sc-8488: CHI → gemma-2-2b-it last-hidden) ─▶ [1, 300, 2304]
//!         ─▶ SanaTransformer  (sc-8487: Linear-DiT trunk, velocity prediction) ─▶ [1, 32, h, w]
//!         ─▶ DcAeDecoder      (sc-8486: DC-AE f32c32 decode)                   ─▶ [1, 1024, 1024, 3]
//! ```
//!
//! driven by the **unified flow-matching scheduler** (epic 7114): the schedule is built by
//! [`mlx_gen::FlowMatchEuler`] and integrated by [`mlx_gen::run_flow_sampler`] — the SAME machinery
//! the sibling flow-match families use (`mlx-gen-sd3`, `mlx-gen-z-image`). No bespoke scheduler.
//!
//! ## Sampler / shift / timestep convention
//!
//! * **Flow-match Euler, static shift 3.0.** `Sana_1600M_1024px_diffusers` ships a
//!   `FlowMatchEulerDiscreteScheduler` with `shift = 3.0` and `use_dynamic_shifting = false`, so the
//!   native schedule is [`FlowMatchEuler::for_static_shift(steps, 3.0)`] (resolution-independent,
//!   `exp(mu) = shift`). An unset `scheduler` keeps that byte-exact; a curated epic-7114 name re-shapes
//!   σ over the same `mu = ln(3)` via [`mlx_gen::resolve_flow_schedule`].
//! * **Timestep convention.** The unified sampler hands the predict closure `ms.timestep(σ) = σ`
//!   ([`TimestepConvention::Sigma`]); the SANA trunk embeds the diffusers-scale timestep `σ · 1000`
//!   (`num_train_timesteps`), so the closure scales it before the forward (identical to SD3's MMDiT).
//!   The Euler update itself stays in σ-space (`x += (σ_{t+1} − σ_t) · v`).
//!
//! ## CFG
//!
//! Base SANA is a **true-CFG** model (the Sprint CFG-free distilled variant is the LATER story
//! sc-8490). Each step runs the trunk TWICE — cond (prompt) + uncond (negative/empty prompt) — and
//! combines `pred = uncond + scale · (cond − uncond)` (diffusers `SanaPipeline.__call__` default
//! `guidance_scale = 4.5`). When `guidance_scale <= 1.0` the uncond forward is skipped (CFG off),
//! matching diffusers' `do_classifier_free_guidance = guidance_scale > 1.0`.
//!
//! ## DC-AE latent scaling
//!
//! diffusers `SanaPipeline` decodes `latents / vae.config.scaling_factor` (the DC-AE
//! `scaling_factor = 0.41407`, [`DcAeConfig::scaling_factor`]); [`DcAeDecoder::decode`] expects the
//! **already-unscaled** latent, so the division is applied here before decode. The decoder emits NHWC
//! `[1, H, W, 3]`; [`mlx_gen::image::decoded_to_image`] expects NCHW, so the output is transposed back
//! to NCHW before the `clip(x·0.5 + 0.5)` → RGB8 conversion.

use mlx_gen::image::decoded_to_image;
use mlx_gen::img2img::{add_noise_by_interpolation, init_time_step, preprocess_init_image};
use mlx_gen::{
    run_flow_sampler, CancelFlag, Error, FlowMatchEuler, Image, Progress, Result,
    TimestepConvention,
};
use mlx_rs::ops::{add, divide, multiply, subtract};
use mlx_rs::{random, Array};

use crate::config::DcAeConfig;
use crate::dc_ae::{DcAeDecoder, DcAeEncoder};
use crate::scm::ScmScheduler;
use crate::text_encoder::SanaTextEncoder;
use crate::transformer::SanaTransformer;

/// DC-AE f32c32 latent channel count (the SANA trunk's `out_channels`).
pub const LATENT_CHANNELS: i32 = 32;
/// DC-AE deep-compression spatial downsample (latent edge is image/32).
pub const SPATIAL_SCALE: u32 = 32;
/// diffusers `num_train_timesteps` — the SANA trunk embeds `sigma * 1000`.
pub const NUM_TRAIN_TIMESTEPS: f32 = 1000.0;
/// SANA-1.6B static flow-match shift (`scheduler_config.json` `shift = 3.0`, no dynamic shifting).
pub const SCHEDULE_SHIFT: f32 = 3.0;
/// diffusers `SanaPipeline` default `num_inference_steps`.
pub const DEFAULT_STEPS: usize = 20;
/// diffusers `SanaPipeline` default `guidance_scale`.
pub const DEFAULT_GUIDANCE: f32 = 4.5;

/// Seeded txt2img latent noise — shape `[1, 32, height/32, width/32]`, f32. diffusers
/// `randn_tensor([B, 32, H/32, W/32])`; we draw f32 via `mx.random.normal` keyed on `seed`.
/// (`init_noise_sigma = 1.0` for flow-match, so the latent is the raw normal draw.)
pub fn create_noise(seed: u64, width: u32, height: u32) -> Result<Array> {
    let key = random::key(seed)?;
    let shape = [
        1,
        LATENT_CHANNELS,
        (height / SPATIAL_SCALE) as i32,
        (width / SPATIAL_SCALE) as i32,
    ];
    Ok(random::normal::<f32>(&shape[..], None, None, Some(&key))?)
}

/// One flow-match Euler denoise with **true CFG** + progress + cooperative cancellation. Each step
/// runs the SANA trunk twice (cond + uncond) and combines `uncond + scale·(cond − uncond)`; the Euler
/// step then advances the latents in σ-space. The trunk timestep is `σ·1000`. When `guidance_scale`
/// is `<= 1.0` the uncond branch is skipped (CFG off, one forward per step; diffusers parity).
#[allow(clippy::too_many_arguments)]
pub fn denoise_cfg(
    transformer: &SanaTransformer,
    scheduler: &FlowMatchEuler,
    sampler_name: Option<&str>,
    start_step: usize,
    seed: u64,
    latents: Array,
    cond: &Array,
    cond_mask: Option<&Array>,
    uncond: Option<&Array>,
    uncond_mask: Option<&Array>,
    guidance_scale: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    let predict = |x: &Array, timestep: f32| -> Result<Array> {
        // The unified flow sampler hands `timestep = σ`; the SANA trunk embeds `σ·1000`.
        let t = Array::from_slice(&[timestep * NUM_TRAIN_TIMESTEPS], &[1]);
        let pred_cond = transformer.forward_with_guidance(x, cond, &t, None, cond_mask)?;
        match uncond {
            Some(uc) if guidance_scale > 1.0 => {
                let pred_uncond =
                    transformer.forward_with_guidance(x, uc, &t, None, uncond_mask)?;
                // pred = uncond + scale·(cond − uncond).
                let delta = subtract(&pred_cond, &pred_uncond)?;
                Ok(add(
                    &pred_uncond,
                    &multiply(&delta, Array::from_slice(&[guidance_scale], &[1]))?,
                )?)
            }
            _ => Ok(pred_cond),
        }
    };
    // img2img runs the tail of the schedule (`sigmas[start_step..]`); txt2img passes `start_step = 0`
    // → the full schedule, byte-identical to the pre-img2img path. The pre-noised init latent (blended
    // at `sigmas[start_step]` by the caller) is the loop's starting point.
    run_flow_sampler(
        sampler_name,
        TimestepConvention::Sigma,
        &scheduler.sigmas[start_step.min(scheduler.sigmas.len().saturating_sub(1))..],
        latents,
        seed,
        cancel,
        on_progress,
        predict,
    )
}

/// DC-AE-decode the final `[1, 32, H/32, W/32]` latent → an RGB8 [`Image`]. diffusers
/// `SanaPipeline` divides by `vae.config.scaling_factor` before decode; the decoder emits NHWC and
/// [`decoded_to_image`] expects NCHW, so the result is transposed back before the RGB8 conversion.
pub fn decode_to_image(decoder: &DcAeDecoder, cfg: &DcAeConfig, latents: &Array) -> Result<Image> {
    let scale = Array::from_slice(&[cfg.scaling_factor], &[1]);
    let unscaled = divide(latents, &scale)?; // diffusers: latents / scaling_factor
    let decoded_nhwc = decoder.decode(&unscaled)?; // [1, H, W, 3] NHWC, f32
    let decoded_nchw = decoded_nhwc.transpose_axes(&[0, 3, 1, 2])?; // → NCHW for decoded_to_image
    decoded_to_image(&decoded_nchw)
}

// =================================================================================================
// SANA-Sprint: continuous-time-consistency (SCM/TrigFlow), CFG-free, 1–4 step (sc-8490).
// =================================================================================================

/// diffusers `SanaSprintPipeline` default `num_inference_steps`.
pub const SPRINT_DEFAULT_STEPS: usize = 2;
/// diffusers `SanaSprintPipeline` default `guidance_scale` (embedded, NOT classifier-free).
pub const SPRINT_DEFAULT_GUIDANCE: f32 = 4.5;

fn arr1(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// One SCM (TrigFlow continuous-time consistency) denoise — the **CFG-free, few-step** SANA-Sprint
/// loop. A faithful port of the diffusers `SanaSprintPipeline` denoise + `SCMScheduler.step`:
///
/// 1. seed the latent and pre-scale by `sigma_data` (the diffusers `latents = latents * sigma_data`);
/// 2. per step `i` over the angle schedule `t = scheduler.timesteps[i]`:
///    * `scm_t = sin(t)/(cos(t)+sin(t))`; model input = `(latents / sigma_data) · sqrt(scm_t² + (1−scm_t)²)`;
///    * ONE trunk forward with the **embedded guidance scalar** (`guidance · guidance_embeds_scale`)
///      and `timestep = scm_t` (no uncond branch — Sprint is CFG-free);
///    * recombine the raw output trigonometrically, `· sigma_data`;
///    * `SCMScheduler.step`: `x0 = cos(s)·x − sin(s)·output`; renoise `x = cos(t')·x0 + sin(t')·noise·sigma_data`
///      (skipped on the final step / single-step schedule);
/// 3. return `denoised / sigma_data` (the diffusers `latents = denoised / sigma_data`).
///
/// The per-step `eval` boundary + cooperative cancel + monotone progress mirror the unified
/// [`mlx_gen::run_flow_sampler`] run-loop contract (the epic-7114 seam SCM reuses; its trigflow step
/// is the consistency parameterization the flow-match `Solver` menu cannot represent).
#[allow(clippy::too_many_arguments)]
pub fn denoise_sprint(
    transformer: &SanaTransformer,
    scheduler: &ScmScheduler,
    seed: u64,
    latents: Array,
    cond: &Array,
    cond_mask: Option<&Array>,
    guidance_scale: f32,
    guidance_embeds_scale: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    // txt2img: seed the SCM prior (`latents * sigma_data`) and run the whole angle schedule (start 0).
    let latents = multiply(&latents, arr1(scheduler.sigma_data))?;
    denoise_sprint_from(
        transformer,
        scheduler,
        0,
        seed,
        latents,
        cond,
        cond_mask,
        guidance_scale,
        guidance_embeds_scale,
        cancel,
        on_progress,
    )
}

/// The SCM (TrigFlow) few-step denoise loop starting at angle index `start_step`, over an **already
/// `sigma_data`-scaled** `latents` (the caller seeds it: txt2img = `noise · σ_data`; img2img =
/// `cos(t)·x0 + sin(t)·noise·σ_data` renoised to `t = timesteps[start_step]`). `start_step = 0` runs
/// the full schedule (the txt2img path, via [`denoise_sprint`]). Progress is reported over the steps
/// actually run (`n - start_step`).
#[allow(clippy::too_many_arguments)]
pub fn denoise_sprint_from(
    transformer: &SanaTransformer,
    scheduler: &ScmScheduler,
    start_step: usize,
    seed: u64,
    latents: Array,
    cond: &Array,
    cond_mask: Option<&Array>,
    guidance_scale: f32,
    guidance_embeds_scale: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    use mlx_rs::transforms::eval;

    let sd = scheduler.sigma_data;
    let mut latents = latents;

    // The embedded guidance scalar (CFG-free): guidance_scale * guidance_embeds_scale, a [1] tensor
    // fed to the trunk's guidance embedder. Constant across steps.
    let guidance = arr1(guidance_scale * guidance_embeds_scale);

    let n = scheduler.num_steps();
    let start = start_step.min(n);
    let total = (n - start).max(1) as u32;
    let mut denoised = latents.clone();
    // Per-step renoise key — a distinct subkey per step so the between-step noise is decorrelated and
    // deterministic for a given request seed (mirrors the unified sampler's `StepRng` derivation).
    let step_key = |step: usize| -> Result<Array> {
        let sub = seed.wrapping_add(0x9E37_79B9_7F4A_7C15_u64.wrapping_mul(step as u64 + 1));
        Ok(random::key(sub)?)
    };

    for i in start..n {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        // Per-eval compute boundary (MLX is lazy): force the prior step's graph so cancel/progress are
        // responsive rather than deferred to decode.
        eval([&latents])?;
        on_progress(Progress::Step {
            current: (i - start) as u32 + 1,
            total,
        });

        let s = scheduler.timesteps[i];
        let t_next = scheduler.timesteps[i + 1];
        let scm_t = scheduler.scm_timestep(i);
        let in_scale = scheduler.input_scale(i);

        // model input = (latents / sigma_data) * sqrt(scm_t² + (1-scm_t)²).
        let lat_in = multiply(&divide(&latents, arr1(sd))?, arr1(in_scale))?;
        let scm_t_arr = arr1(scm_t);
        let raw = transformer.forward_with_guidance(
            &lat_in,
            cond,
            &scm_t_arr,
            Some(&guidance),
            cond_mask,
        )?;

        // diffusers trigflow recombination of the raw output (uses `latent_model_input` = the SCALED
        // `lat_in`, NOT the un-scaled latent):
        //   noise_pred = ((1-2·scm_t)·lat_in + (1-2·scm_t+2·scm_t²)·raw) / sqrt(scm_t²+(1-scm_t)²)
        //   noise_pred = noise_pred * sigma_data
        let a = 1.0 - 2.0 * scm_t;
        let b = 1.0 - 2.0 * scm_t + 2.0 * scm_t * scm_t;
        let model_output = multiply(
            &divide(
                &add(&multiply(&lat_in, arr1(a))?, &multiply(&raw, arr1(b))?)?,
                arr1(in_scale),
            )?,
            arr1(sd),
        )?;

        // SCMScheduler.step (trigflow x0-pred + renoise). `s` = current angle, `t_next` = next angle.
        // pred_x0 = cos(s)·latents − sin(s)·model_output.
        let pred_x0 = subtract(
            &multiply(&latents, arr1(s.cos()))?,
            &multiply(&model_output, arr1(s.sin()))?,
        )?;
        denoised = pred_x0.clone();
        // Renoise to the next angle (skipped on the final / single-step transition, matching diffusers
        // `if len(self.timesteps) > 1`). On the last step `t_next == 0` ⇒ `cos(0)=1`, `sin(0)=0`, so the
        // renoise reduces to exactly `pred_x0` — gate the noise DRAW on a non-terminal step (`i+1 < n`)
        // so the final step doesn't burn a wasted `random::normal` + key derivation (F-092; bit-exact,
        // the drawn noise was multiplied by `sin(0)=0` anyway).
        latents = if scheduler.is_single_step() || i + 1 >= n {
            pred_x0
        } else {
            let noise = multiply(
                &random::normal::<f32>(latents.shape(), None, None, Some(&step_key(i)?))?,
                arr1(sd),
            )?;
            add(
                &multiply(&pred_x0, arr1(t_next.cos()))?,
                &multiply(&noise, arr1(t_next.sin()))?,
            )?
        };
    }

    // diffusers: latents = denoised / sigma_data (the decode input).
    let out = divide(&denoised, arr1(sd))?;
    eval([&out])?;
    Ok(out)
}

/// The composed SANA text-to-image pipeline: text encoder + trunk + DC-AE decoder, with the DC-AE
/// config (for the latent `scaling_factor`). A clean `generate` entrypoint mirroring the sibling
/// flow-match pipelines (`mlx-gen-sd3`).
///
/// `sprint` selects the variant: `false` = base SANA-1.6B (true-CFG flow-match Euler); `true` =
/// SANA-Sprint (CFG-free SCM/TrigFlow few-step, sc-8490). The trunk must be loaded with the matching
/// config (`SanaTransformerConfig::sana_sprint_1600m()` for Sprint — its guidance embedder +
/// rms-norm-across-heads are config-gated).
pub struct SanaPipeline {
    text_encoder: SanaTextEncoder,
    transformer: SanaTransformer,
    /// DC-AE **encoder** — the img2img reference→latent path (sc-10190). Loaded from the SAME
    /// `vae/` snapshot as the decoder (the checkpoint ships both `encoder.*` and `decoder.*` keys).
    encoder: DcAeEncoder,
    decoder: DcAeDecoder,
    dc_ae_cfg: DcAeConfig,
    sprint: bool,
    guidance_embeds_scale: f32,
}

/// One text-to-image request for [`SanaPipeline::generate`]. `None` fields fall back to the diffusers
/// `SanaPipeline` defaults (`steps = 20`, `guidance = 4.5`, `seed = 0`, empty negative prompt).
#[derive(Clone, Debug)]
pub struct SanaGenerateRequest<'a> {
    pub prompt: &'a str,
    pub negative_prompt: Option<&'a str>,
    pub height: u32,
    pub width: u32,
    pub steps: Option<usize>,
    pub guidance_scale: Option<f32>,
    pub seed: Option<u64>,
    /// Optional curated epic-7114 sampler name (e.g. `"euler"`, `"dpmpp_2m"`); `None` = native Euler.
    pub sampler: Option<&'a str>,
    /// Optional curated epic-7114 scheduler name re-shaping σ over the same `mu = ln(shift)`.
    pub scheduler: Option<&'a str>,
    /// **img2img** reference image (sc-10190): when present with a positive [`Self::strength`], the
    /// DC-AE-encoded init latent seeds the denoise instead of pure noise. `None` = plain txt2img.
    pub init_image: Option<&'a Image>,
    /// img2img strength ∈ `(0, 1]` (the fork's `init_time_step` convention: higher → start later →
    /// output stays closer to the init image). `None` (or with no `init_image`) = txt2img.
    pub strength: Option<f32>,
}

impl<'a> SanaGenerateRequest<'a> {
    /// A 1024px request for `prompt` with all diffusers defaults.
    pub fn new(prompt: &'a str) -> Self {
        Self {
            prompt,
            negative_prompt: None,
            height: 1024,
            width: 1024,
            steps: None,
            guidance_scale: None,
            seed: None,
            sampler: None,
            scheduler: None,
            init_image: None,
            strength: None,
        }
    }
}

/// VAE-encode an img2img reference image into a **denoise-space** DC-AE latent
/// `[1, latent_channels, H/32, W/32]` (sc-10190): LANCZOS-resize + `[-1,1]` NCHW preprocess →
/// [`DcAeEncoder::encode`] → multiply by the DC-AE `scaling_factor`. The `scaling_factor` places the
/// latent in the same space the denoise loop + [`decode_to_image`] (which divides it back) operate in.
pub fn encode_init_latents(
    encoder: &DcAeEncoder,
    cfg: &DcAeConfig,
    image: &Image,
    width: u32,
    height: u32,
) -> Result<Array> {
    let image_nchw = preprocess_init_image(image, width, height)?;
    let raw = encoder.encode(&image_nchw)?; // [1, 32, H/32, W/32], raw (pre-scale)
    Ok(multiply(&raw, arr1(cfg.scaling_factor))?)
}

impl SanaPipeline {
    /// Compose the **base SANA-1.6B** pipeline (true-CFG flow-match) from its three already-constructed
    /// components plus the DC-AE config (used for the latent `scaling_factor`).
    pub fn new(
        text_encoder: SanaTextEncoder,
        transformer: SanaTransformer,
        encoder: DcAeEncoder,
        decoder: DcAeDecoder,
        dc_ae_cfg: DcAeConfig,
    ) -> Self {
        Self {
            text_encoder,
            transformer,
            encoder,
            decoder,
            dc_ae_cfg,
            sprint: false,
            guidance_embeds_scale: 0.0,
        }
    }

    /// Compose the **SANA-Sprint** pipeline (CFG-free SCM/TrigFlow few-step, sc-8490). The
    /// `transformer` MUST be loaded with [`crate::SanaTransformerConfig::sana_sprint_1600m`] (its
    /// guidance embedder + rms-norm-across-heads are required for the embedded-guidance forward).
    /// `guidance_embeds_scale` is the trunk config's `guidance_embeds_scale` (`0.1`), pre-multiplied
    /// into the guidance scalar before the embedder.
    pub fn new_sprint(
        text_encoder: SanaTextEncoder,
        transformer: SanaTransformer,
        encoder: DcAeEncoder,
        decoder: DcAeDecoder,
        dc_ae_cfg: DcAeConfig,
        guidance_embeds_scale: f32,
    ) -> Self {
        Self {
            text_encoder,
            transformer,
            encoder,
            decoder,
            dc_ae_cfg,
            sprint: true,
            guidance_embeds_scale,
        }
    }

    /// Whether this is a SANA-Sprint (CFG-free few-step) pipeline.
    pub fn is_sprint(&self) -> bool {
        self.sprint
    }

    /// Run the full prompt→image pipeline. Encodes the prompt (and the negative prompt when CFG is
    /// active) ONCE, seeds the DC-AE latent, runs the flow-match Euler denoise over the SANA trunk
    /// with true CFG, then DC-AE-decodes to an RGB8 [`Image`].
    pub fn generate(&self, req: &SanaGenerateRequest<'_>) -> Result<Image> {
        let cancel = CancelFlag::default();
        let mut noop = |_: Progress| {};
        self.generate_with(req, &cancel, &mut noop)
    }

    /// [`SanaPipeline::generate`] with caller-supplied cancellation + progress (the seam Phase B's
    /// worker `Generator` adapter wires into the gen-core contract).
    pub fn generate_with(
        &self,
        req: &SanaGenerateRequest<'_>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        // F-091: `create_noise` derives the latent grid via `dim / SPATIAL_SCALE` integer division, so
        // a width/height not a multiple of 32 silently truncates the latent (and the output image) to
        // the floor multiple instead of honoring the request. Reject it up front (both the CFG and the
        // Sprint path funnel through here) rather than returning a quietly-smaller image.
        if !req.width.is_multiple_of(SPATIAL_SCALE) || !req.height.is_multiple_of(SPATIAL_SCALE) {
            return Err(Error::Msg(format!(
                "sana: width and height must be multiples of {SPATIAL_SCALE}, got {}x{}",
                req.width, req.height
            )));
        }
        if self.sprint {
            return self.generate_sprint(req, cancel, on_progress);
        }
        let steps = req.steps.unwrap_or(DEFAULT_STEPS);
        let guidance = req.guidance_scale.unwrap_or(DEFAULT_GUIDANCE);
        let seed = req.seed.unwrap_or(0);

        // Conditioning is seed-independent — encode once. Cond = the prompt; uncond = the negative
        // prompt (empty string when unset), used only when CFG is active. diffusers gates CFG on
        // `do_classifier_free_guidance = guidance_scale > 1.0`.
        // Encode WITH the caption padding mask — SANA's attn2 cross-attention masks PAD keys (diffusers
        // `encoder_attention_mask`); dropping it lets padding swamp short-prompt conditioning.
        let (cond, cond_mask) = self.text_encoder.encode_with_mask(req.prompt)?;
        let cfg_on = guidance > 1.0;
        let (uncond, uncond_mask) = if cfg_on {
            let neg = req.negative_prompt.unwrap_or("");
            let (u, um) = self.text_encoder.encode_with_mask(neg)?;
            (Some(u), Some(um))
        } else {
            (None, None)
        };

        // Static shift=3.0 schedule (scheduler_config.json), resolution-independent — build once. An
        // unset scheduler keeps it byte-exact; a curated name re-shapes σ over the same mu=ln(3).
        let native = FlowMatchEuler::for_static_shift(steps, SCHEDULE_SHIFT);
        let scheduler = FlowMatchEuler::from_sigmas(mlx_gen::resolve_flow_schedule(
            req.scheduler,
            SCHEDULE_SHIFT.ln(),
            steps,
            &native.sigmas,
        ))?;

        // img2img (sc-10190): a reference image + positive strength starts the denoise at
        // `sigmas[start_step]` over the DC-AE-encoded init latent blended with noise; else start 0
        // (pure-noise txt2img). `init_time_step` returns 0 when strength is None/≤0 (→ txt2img).
        let start_step = match req.init_image {
            Some(_) => init_time_step(steps, req.strength),
            None => 0,
        };
        let clean = if start_step > 0 {
            let image = req
                .init_image
                .expect("start_step > 0 implies an init image");
            Some(encode_init_latents(
                &self.encoder,
                &self.dc_ae_cfg,
                image,
                req.width,
                req.height,
            )?)
        } else {
            None
        };

        let noise = create_noise(seed, req.width, req.height)?;
        let latents = match &clean {
            // Blend the pre-encoded clean latents with the noise at `sigma = sigmas[start_step]`.
            Some(clean) => {
                let sigma = *scheduler.sigmas.get(start_step).ok_or_else(|| {
                    Error::Msg(format!(
                        "sana img2img: start step {start_step} out of range for {}-element schedule",
                        scheduler.sigmas.len()
                    ))
                })?;
                add_noise_by_interpolation(clean, &noise, sigma)?
            }
            None => noise,
        };
        let latents = denoise_cfg(
            &self.transformer,
            &scheduler,
            req.sampler,
            start_step,
            seed,
            latents,
            &cond,
            Some(&cond_mask),
            uncond.as_ref(),
            uncond_mask.as_ref(),
            guidance,
            cancel,
            on_progress,
        )?;
        on_progress(Progress::Decoding);
        decode_to_image(&self.decoder, &self.dc_ae_cfg, &latents)
    }

    /// The **SANA-Sprint** (CFG-free SCM/TrigFlow few-step) generate path (sc-8490). Encodes the
    /// prompt ONCE (no uncond — Sprint is CFG-free), seeds the latent, runs the SCM loop over an
    /// [`ScmScheduler`] (default 2 steps, embedded guidance 4.5), then DC-AE-decodes. The negative
    /// prompt / curated sampler+scheduler knobs are inapplicable to the SCM loop and ignored.
    ///
    /// **img2img (sc-10190):** a reference image + positive strength starts the SCM loop at angle
    /// index `start = init_time_step(n, strength)`, seeding `latents` by TrigFlow-renoising the
    /// DC-AE-encoded init to that angle: `x_t = cos(t)·x0 + sin(t)·noise·σ_data` with `x0 =
    /// encode·scaling_factor·σ_data` and `t = timesteps[start]`. Distilled/consistency, so the strength
    /// window is narrow — validate the band on-device. `start = 0` is the byte-identical txt2img path.
    fn generate_sprint(
        &self,
        req: &SanaGenerateRequest<'_>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let steps = req.steps.unwrap_or(SPRINT_DEFAULT_STEPS);
        let guidance = req.guidance_scale.unwrap_or(SPRINT_DEFAULT_GUIDANCE);
        let seed = req.seed.unwrap_or(0);

        let (cond, cond_mask) = self.text_encoder.encode_with_mask(req.prompt)?;
        let scheduler = ScmScheduler::new(steps);
        let n = scheduler.num_steps();
        let sd = scheduler.sigma_data;

        let start_step = match req.init_image {
            Some(_) => init_time_step(n, req.strength),
            None => 0,
        };
        let noise = create_noise(seed, req.width, req.height)?;
        let latents = if start_step > 0 {
            // img2img: renoise the encoded init to the start angle `timesteps[start_step]`.
            let image = req
                .init_image
                .expect("start_step > 0 implies an init image");
            let clean =
                encode_init_latents(&self.encoder, &self.dc_ae_cfg, image, req.width, req.height)?;
            // x0 in the SCM prior space (σ_data-scaled); noise likewise. TrigFlow renoise to angle t.
            let x0 = multiply(&clean, arr1(sd))?;
            let noise_sd = multiply(&noise, arr1(sd))?;
            let t = scheduler.timesteps[start_step];
            add(
                &multiply(&x0, arr1(t.cos()))?,
                &multiply(&noise_sd, arr1(t.sin()))?,
            )?
        } else {
            // txt2img: the SCM prior is `noise · σ_data`.
            multiply(&noise, arr1(sd))?
        };
        let latents = denoise_sprint_from(
            &self.transformer,
            &scheduler,
            start_step,
            seed,
            latents,
            &cond,
            Some(&cond_mask),
            guidance,
            self.guidance_embeds_scale,
            cancel,
            on_progress,
        )?;
        on_progress(Progress::Decoding);
        decode_to_image(&self.decoder, &self.dc_ae_cfg, &latents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::transforms::eval;

    #[test]
    fn noise_shape_is_batch1_32ch() {
        let n = create_noise(0, 1024, 1024).unwrap();
        assert_eq!(n.shape(), &[1, 32, 32, 32]);
        let n = create_noise(0, 512, 1024).unwrap();
        assert_eq!(n.shape(), &[1, 32, 32, 16]);
    }

    #[test]
    fn noise_is_seed_deterministic() {
        let a = create_noise(7, 256, 256).unwrap();
        let b = create_noise(7, 256, 256).unwrap();
        let c = create_noise(8, 256, 256).unwrap();
        eval([&a, &b, &c]).unwrap();
        assert_eq!(
            a.as_slice::<f32>(),
            b.as_slice::<f32>(),
            "same seed reproduces"
        );
        assert_ne!(
            a.as_slice::<f32>(),
            c.as_slice::<f32>(),
            "diff seed differs"
        );
    }

    #[test]
    fn static_shift_schedule_matches_diffusers() {
        // SANA-1.6B: FlowMatchEulerDiscreteScheduler shift=3.0, no dynamic shifting.
        let s = FlowMatchEuler::for_static_shift(4, SCHEDULE_SHIFT);
        let expected = [1.0_f32, 0.9, 0.75, 0.5, 0.0];
        assert_eq!(s.sigmas.len(), 5);
        for (got, want) in s.sigmas.iter().zip(expected) {
            assert!((got - want).abs() < 1e-5, "got {got} want {want}");
        }
    }
}
