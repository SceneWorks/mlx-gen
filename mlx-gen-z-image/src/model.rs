//! `ZImageTurbo` — the Z-Image-turbo implementation of [`mlx_gen::Generator`], plus its
//! [`descriptor`]/[`load`] entry points and the `inventory` registration that wires it into
//! `mlx_gen`'s registry.
//!
//! [`load`] assembles the full model from a `Tongyi-MAI/Z-Image-Turbo` snapshot directory (see
//! [`crate::loader`]) — tokenizer, Qwen text encoder, DiT transformer, VAE decoder — and
//! [`ZImageTurbo::generate`] runs the complete prompt→image pipeline: tokenize → encode →
//! seeded noise → flow-match Euler denoise over the DiT → VAE decode → RGB8. The chain is
//! parity-proven against the frozen Python fork on real bf16 weights (sc-2352).

use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, default_seed, resolve_flow_schedule,
    Capabilities, ConditioningKind, Error, FlowMatchEuler, GenerationOutput, GenerationRequest,
    Generator, LatentDecoder, LoadSpec, Modality, ModelDescriptor, OffloadPolicy, Precision,
    Progress, Quant, Residency, Result, WeightsSource,
};
use mlx_gen_pid::{flow_capture_for_request, resolve_pid_decoder_at_sigma, PidEngine};
use mlx_rs::Dtype;
use std::path::Path;

use crate::loader;
use crate::pipeline::{self, denoise_with_progress, encode_init_latents, init_time_step};
use crate::text_encoder::TextEncoder;
use crate::transformer::ZImageTransformer;
use crate::vae::Vae;

/// Z-Image-turbo is guidance-distilled to a fixed 4-step schedule; used when a request omits
/// `steps`. (`pub(crate)` so the ControlNet variant shares the same default.)
pub(crate) const DEFAULT_STEPS: u32 = 4;

/// Flow-match time-shift for Z-Image-Turbo: the model's own published schedule from
/// `scheduler/scheduler_config.json` (`FlowMatchEulerDiscreteScheduler`, `shift=3.0`,
/// `use_dynamic_shifting=false`) — static, resolution-independent.
///
/// **Deliberate choice (sc-2536; Michael, 2026-06-01) — do NOT "restore" `linear`.** The mflux
/// MLX path this port replaces (`MlxZImageAdapter` → `generate_image`'s default `linear`
/// scheduler) actually uses a *dynamic*, resolution-dependent shift (≈3.16 @1024², 1.88 @512²,
/// 25 @2048²). We use the model's static 3.0 instead: A/B renders (`tools/compare_z_image_
/// schedulers.py`) are visually identical at 1024² and only differ at lower resolutions, where
/// 3.0 reads slightly crisper — the preferred look. So 3.0 is an intentional, model-config-backed
/// deviation from the MLX path, not drift.
///
/// (The *original* port's bug — replaced by sc-2536 — was using `FlowMatchEuler::for_image`'s
/// empirical per-step `mu`, the *full* Z-Image model's scheduler, ≈shift 10. That was wrong;
/// `linear` and 3.0 are both reasonable, empirical-`mu` was not.)
///
/// `pub(crate)` so the ControlNet variant ([`crate::model_control`]) uses the identical schedule —
/// it is the same base turbo model, and the parity golden holds the schedule fixed on both sides.
pub(crate) const SCHEDULE_SHIFT: f32 = 3.0;

/// Registry id for Z-Image-turbo (matches the SceneWorks worker's `payload.model`).
pub const MODEL_ID: &str = "z_image_turbo";

/// PiD backbone (latent-space) tag for Z-Image (epic 7840, sc-7846). Z-Image ships Flux1-dev's 16-ch
/// VAE, so it reuses the `flux` PiD student — the `mlx_gen_pid::registry` `zimage-turbo` alias resolves
/// to exactly that checkpoint/space. Used only at load time to build the [`PidEngine`].
pub const PID_BACKBONE: &str = "zimage-turbo";

/// Z-Image-turbo's identity + capabilities — constructible without loading weights (registry
/// introspection). Values are conservative-but-real; sampler/scheduler lists fill in with the
/// scheduler port.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "z-image",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            supported_quants: &[Quant::Q4, Quant::Q8],
            // Turbo is guidance-distilled: no CFG, no negative prompt.
            supports_negative_prompt: false,
            supports_guidance: false,
            supports_true_cfg: false,
            // img2img reference; ControlNet is a separate variant (sc-2349).
            conditioning: vec![ConditioningKind::Reference],
            supports_lora: true,
            supports_lokr: true,
            // Curated unified-framework integrator menu (epic 7114 P3). Turbo is guidance-distilled to
            // ~4 steps; an unset `req.sampler` is the curated Euler over the static-shift schedule.
            samplers: curated_sampler_names(),
            // Scheduler axis (epic 7114): the static-shift schedule is the byte-exact default (an unset
            // `req.scheduler`); a curated name re-shapes the σ schedule over the same `shift=3.0`.
            schedulers: curated_scheduler_names(),
            supported_guidance_methods: vec![],
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: true,
            supports_kv_cache: false,
            requires_sigma_shift: false,
            // Wired onto the shared `Residency` seam; honors Sequential offload (F-176).
            supports_sequential_offload: true,
        },
    }
}

/// A loaded Z-Image-turbo generator: the four model components assembled from a snapshot
/// directory, plus the cached descriptor.
pub struct ZImageTurbo {
    descriptor: ModelDescriptor,
    tokenizer: TextTokenizer,
    /// Component-residency strategy (epic 10834 Phase 1, sc-10839; hoisted to the shared seam in
    /// sc-11125), selected from [`LoadSpec::offload_policy`]. `Resident` (default) holds the Qwen
    /// text encoder + DiT + VAE warm for the whole job and across jobs; `Sequential` holds only the
    /// per-phase loader closures and re-loads per generation in phase order (encode → **drop the text
    /// encoder** → denoise/decode), bounding peak unified memory to `max(text-encoder, DiT+VAE)`
    /// instead of the sum — the big win on Z-Image, whose Qwen encoder is comparable to the DiT. The
    /// [`Residency`] seam owns the eval/drop/clear discipline, the stage-boundary cancel checks, and
    /// the error-safe cache flush once for all providers.
    residency: Residency<TextEncoder, ZImageHeavyOwned>,
}

/// The heavy render-phase components (the DiT transformer, the VAE, and the optional PiD decoder) —
/// everything but the text encoder. Owned by the `Resident` components or by a `Sequential` generate.
/// `pub(crate)` so the **base** (non-distilled) `z_image` sibling ([`crate::model_base`]) shares the
/// identical heavy bundle on the same shared [`load_residency`] seam (sc-11124, F-172).
pub(crate) struct ZImageHeavyOwned {
    pub(crate) transformer: ZImageTransformer,
    pub(crate) vae: Vae,
    /// Optional PiD super-resolving decoder overlay (epic 7840, sc-7846). `Some` only when the
    /// `LoadSpec` carried `pid`; selected per-generation by `req.use_pid`. Z-Image shares Flux1's VAE
    /// latent space, so it reuses the `flux` PiD student via the `zimage-turbo` registry alias.
    pub(crate) pid: Option<PidEngine>,
}

/// A borrow of the heavy render-phase components, so the denoise/decode body runs identically
/// whether they are held resident or were just loaded by the `Sequential` path (candle's `DitRef`).
pub(crate) struct ZImageHeavy<'a> {
    pub(crate) transformer: &'a ZImageTransformer,
    pub(crate) vae: &'a Vae,
    pub(crate) pid: Option<&'a PidEngine>,
}

impl ZImageHeavyOwned {
    pub(crate) fn as_ref(&self) -> ZImageHeavy<'_> {
        ZImageHeavy {
            transformer: &self.transformer,
            vae: &self.vae,
            pid: self.pid.as_ref(),
        }
    }
}

/// Construct a [`ZImageTurbo`] from a [`LoadSpec`].
///
/// `spec.weights` must be a [`WeightsSource::Dir`] pointing at a `Tongyi-MAI/Z-Image-Turbo`
/// snapshot (the diffusers multi-component tree — `tokenizer/`, `text_encoder/`, `transformer/`,
/// `vae/`). Weights load dense at their on-disk dtype (bf16); the text encoder promotes to f32
/// internally. `spec.quantize` (Q4/Q8) quantizes the **whole model** — transformer, text encoder,
/// and VAE (group_size 64) — after the dense load, matching the mflux fork's `nn.quantize` over
/// every quantizable Linear (plus the text encoder's token Embedding) so a Q4/Q8 consumer gets the
/// full memory saving and fork-matching output (sc-2532). An fp32 precision override is not wired
/// (the validated dense path is bf16) and is rejected rather than silently ignored.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let (tokenizer, residency) = load_residency(spec, PRECISION_MSG, FILE_MSG)?;
    Ok(Box::new(ZImageTurbo {
        descriptor: descriptor(),
        tokenizer,
        residency,
    }))
}

/// Build the tokenizer + [`Residency`] seam for a non-control Z-Image generator from a [`LoadSpec`],
/// honoring [`LoadSpec::offload_policy`] (epic 10834 Phase 1, sc-10839; hoisted to the shared seam in
/// sc-11125). `Resident` (default) builds every heavy component now via [`build_residency`] and holds
/// it warm; `Sequential` keeps only the spec and re-loads per generate in phase order (encode → drop
/// the text encoder → denoise/decode) to bound peak memory to `max(text-encoder, DiT+VAE)`. Both use
/// the same per-phase loaders, so the components are byte-identical.
///
/// `pub(crate)` and parameterized by the two per-id error strings (precision override / single-file
/// rejection) so the **base** `z_image` sibling ([`crate::model_base`]) shares the identical policy
/// routing rather than re-deriving it (sc-11124, F-172 — before which the base always loaded
/// `Resident`, silently OOMing a fit-gated Sequential request).
pub(crate) fn load_residency(
    spec: &LoadSpec,
    precision_msg: &'static str,
    file_msg: &'static str,
) -> Result<(TextTokenizer, Residency<TextEncoder, ZImageHeavyOwned>)> {
    // Precision + snapshot-dir guard up front for BOTH policies (fail fast), then the always-warm
    // tokenizer; the heavy component dispatch is the shared [`build_residency`] seam below.
    let root = resolve_precision_and_root(spec, precision_msg, file_msg)?;
    // F-181: a `Sequential` + load-time (re)quant over a *dense* snapshot re-quantizes the whole model
    // on every generate. An already-packed turnkey loads packed (no re-quant); `Resident` quantizes
    // once. So warn only for the Sequential-over-dense combination that actually pays the repeated cost.
    if let Some(q) = spec.quantize {
        if matches!(spec.offload_policy, OffloadPolicy::Sequential)
            && loader::needs_load_time_quant(root, q.bits(), MODEL_ID)?
        {
            mlx_gen::residency::warn_sequential_requantize(MODEL_ID, q.bits());
        }
    }
    let tokenizer = loader::load_tokenizer(root)?;
    Ok((tokenizer, build_residency(spec, precision_msg, file_msg)?))
}

/// The policy→[`Residency`] dispatch every Z-Image variant shares (sc-11126, F-180), routed through
/// the single [`Residency::from_policy`] seam so no variant re-derives the `match offload_policy`
/// (the divergence that let a sibling silently ignore `offload_policy` before sc-11124). `Resident`
/// eager-loads the text encoder + heavy bundle (byte-identical to the pre-seam composition — the same
/// per-phase loaders over independent weight files); `Sequential` captures the two loader closures and
/// loads nothing now, deferring each to [`Residency::run`]. The deferral is weight-free-testable: under
/// `Sequential` this touches no files, so a dispatch that ignored the policy would eager-load and fail
/// the residency unit test's "Sequential defers" assertion.
pub(crate) fn build_residency(
    spec: &LoadSpec,
    precision_msg: &'static str,
    file_msg: &'static str,
) -> Result<Residency<TextEncoder, ZImageHeavyOwned>> {
    let spec_text = spec.clone();
    let spec_heavy = spec.clone();
    Residency::from_policy(
        spec.offload_policy,
        move || {
            let root = resolve_precision_and_root(&spec_text, precision_msg, file_msg)?;
            load_text_encoder_only(root, spec_text.quantize)
        },
        move |use_pid| {
            let root = resolve_precision_and_root(&spec_heavy, precision_msg, file_msg)?;
            load_heavy(&spec_heavy, root, use_pid)
        },
    )
}

/// The `z_image_turbo` precision-override / single-file rejection messages, shared by the
/// [`build_residency`] dispatch and the `Sequential` [`resolve_precision_and_root`] guard.
const PRECISION_MSG: &str = "z_image_turbo: only dense bf16 is wired in the Rust port; the text \
     encoder already runs f32 internally (drop the precision override)";
const FILE_MSG: &str = "z_image_turbo expects a snapshot directory (tokenizer/ text_encoder/ \
     transformer/ vae/), not a single .safetensors file";

/// Precision guard + snapshot-dir resolution (rejecting a single-file source), shared by
/// [`build_residency`]'s per-phase loaders (sc-10839, sc-11126).
fn resolve_precision_and_root<'a>(
    spec: &'a LoadSpec,
    precision_msg: &str,
    file_msg: &str,
) -> Result<&'a Path> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(precision_msg.into()));
    }
    match &spec.weights {
        WeightsSource::Dir(p) => Ok(p),
        WeightsSource::File(_) => Err(Error::Msg(file_msg.into())),
    }
}

/// Load the Qwen text encoder (+ optional whole-model Q4/Q8), the phase-A component dropped first
/// under `Sequential`. Q4/Q8 is `group_size 64` over every quantizable Linear + the token Embedding
/// (sc-2532); factored out so the `Resident` and `Sequential` paths build byte-identical encoders.
fn load_text_encoder_only(root: &Path, quant: Option<Quant>) -> Result<TextEncoder> {
    let mut text_encoder = loader::load_text_encoder(root)?;
    if let Some(q) = quant {
        text_encoder.quantize(q.bits())?;
    }
    Ok(text_encoder)
}

/// Load the heavy render-phase components — DiT transformer (+ Q4/Q8 + LoRA/LoKr residuals), VAE
/// (+ Q4/Q8), and the optional PiD overlay — everything but the text encoder. Factored so the
/// `Sequential` path loads these AFTER the encoder is dropped (bounding peak to
/// `max(text-encoder, DiT+VAE)`). Quantize-then-adapters order matches the pre-sc-10839
/// resident composition; the components are independent of the text encoder (separate weight files,
/// deterministic RNG-free quant), so the `Resident` composition below is byte-identical.
fn load_heavy(spec: &LoadSpec, root: &Path, load_pid: bool) -> Result<ZImageHeavyOwned> {
    let mut transformer = loader::load_transformer(root)?;
    let mut vae = loader::load_vae(root)?;
    if let Some(q) = spec.quantize {
        let bits = q.bits();
        transformer.quantize(bits)?;
        vae.quantize(bits)?;
    }
    // LoRA/LoKr (sc-2602): applied after quantization, as forward-time residuals over the
    // (possibly quantized) base — fork-faithful (the fork applies adapters in its initializer over
    // the quantized model). No-op when `spec.adapters` is empty.
    if !spec.adapters.is_empty() {
        crate::adapters::apply_z_image_adapters(&mut transformer, &spec.adapters)?;
    }
    // Optional PiD decoder overlay (epic 7840, sc-7846): Z-Image is the Flux1 latent space, so it
    // reuses the `flux` PiD student (the `zimage-turbo` registry alias). Loaded only when the spec
    // carries `pid` AND this generate uses it (`load_pid`, F-177) — the Resident path passes `true`
    // (loaded once, reused), the Sequential path passes `req.use_pid` so a non-PiD generate skips
    // the student + its Gemma caption encoder entirely. The native VAE decode path is untouched.
    let pid = if load_pid {
        spec.pid
            .as_ref()
            .map(|p| PidEngine::from_spec(p, PID_BACKBONE))
            .transpose()?
    } else {
        None
    };
    Ok(ZImageHeavyOwned {
        transformer,
        vae,
        pid,
    })
}

mlx_gen::impl_generator!(ZImageTurbo {
    validate: |s, req| validate_request(s.descriptor.id, &s.descriptor.capabilities, req),
    generate: generate_impl,
});

impl ZImageTurbo {
    /// The rich-`Result` body behind [`Generator::generate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the family
    /// helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    ///
    /// The staged residency lifecycle (encode → drop the text encoder under `Sequential` → load the
    /// DiT/VAE/PiD → denoise/decode → free the heavy bundle) is driven by the shared
    /// [`Residency::run`] seam (sc-11125), which owns the eval/drop/clear discipline, the
    /// stage-boundary cancel checks, and the error-safe cache flush.
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;

        let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
        let base_seed = req.seed.unwrap_or_else(default_seed);

        // img2img: a single `Reference` image, with a per-reference strength overriding `req.strength`.
        let reference = pipeline::resolve_reference(req, MODEL_ID)?;
        let start_step = match reference {
            Some((_, strength)) => init_time_step(steps, strength),
            None => 0,
        };
        let is_img2img = start_step > 0;

        let images = self.residency.run(
            &req.cancel,
            req.use_pid,
            on_progress,
            // ── Phase A: prompt → cap_feats. txt2img runs the DiT in bf16 (the parity-proven path);
            // img2img matches the fork's f32 init latents, so keep cap f32. PARITY-BF16 (sc-2609):
            // round the text embeddings to bf16 to match the fork's golden; f32 is sharper — flip to
            // f32 once parity is not the goal.
            |text_encoder: &TextEncoder| {
                let cap =
                    pipeline::encode_prompt(&self.tokenizer, text_encoder, &req.prompt, MODEL_ID)?;
                if is_img2img {
                    Ok(cap)
                } else {
                    Ok(cap.as_dtype(Dtype::Bfloat16)?)
                }
            },
            // Materialize the final `cap` while the encoder is still alive (Sequential only) — MLX is
            // lazy, so an un-evaluated `cap` keeps the encoder referenced through the graph and the
            // drop would free nothing. The bf16 cast is applied above so this barrier materializes the
            // post-cast `cap`, not the pre-cast f32 graph.
            |cap| Ok(mlx_rs::transforms::eval([cap])?),
            // ── Phase B: denoise/decode from the heavy bundle. Runs identically for both residencies.
            |heavy_owned, cap, on_progress| {
                let heavy = heavy_owned.as_ref();

                // Static shift=3.0 schedule (the model's scheduler_config.json), resolution- and
                // seed-independent. See SCHEDULE_SHIFT. An unset `req.scheduler` keeps this native
                // schedule byte-exact (epic 7114 N1); a curated name re-shapes the σ schedule over the
                // same `shift=3.0` (`mu = ln(3)`).
                let native = FlowMatchEuler::for_static_shift(steps, SCHEDULE_SHIFT);
                let resolved_sigmas = resolve_flow_schedule(
                    req.scheduler.as_deref(),
                    SCHEDULE_SHIFT.ln(),
                    steps,
                    &native.sigmas,
                );
                // PiD decode overlay (sc-7846) + `from_ldm` early-stop (sc-8048): mint a decoder when
                // `use_pid` is set AND a PiD overlay was loaded, else `None` → the native VAE. Errors
                // loudly if `use_pid` was requested without a loaded overlay (no silent VAE fallback).
                // Z-Image is flow-match (`vp_frame=false`), so the schedule σ *is* the degrade σ: when
                // `pid_capture_sigma` asks for an early exit, `flow_capture_for_request` folds the σ
                // ceiling + schedule into `(capture_sigma, keep)` — mint the decoder at `capture_sigma`
                // and build the scheduler over the *truncated* `resolved_sigmas[..keep]` so the denoise
                // stops at the achieved-σ step (the img2img blend still reads `sigmas[start_step]`,
                // valid since `keep > start_step`). The clean path yields `(0.0, len())` → full
                // schedule, σ=0, byte-identical. `start_step` is the img2img noise-blend offset.
                let (capture_sigma, keep) =
                    flow_capture_for_request(req, &resolved_sigmas, start_step);
                let pid_decoder = resolve_pid_decoder_at_sigma(
                    heavy.pid,
                    req,
                    base_seed,
                    MODEL_ID,
                    capture_sigma,
                )?;
                let scheduler = FlowMatchEuler::from_sigmas(resolved_sigmas[..keep].to_vec())?;

                // VAE-encode the init image once: the clean latents depend only on the init image +
                // target dims, not the per-image seed, so they're constant across the count loop
                // (F-034). Only the noise (and its blend) vary per image.
                let clean = if is_img2img {
                    let (image, _) = reference.expect("is_img2img implies a reference");
                    Some(encode_init_latents(
                        heavy.vae, image, req.width, req.height,
                    )?)
                } else {
                    None
                };

                // Per-image batch render shared with the control variant (F-035); the base branch's
                // only difference is the plain `denoise_with_progress` step.
                let sampler_name = req.sampler.as_deref();
                let images = pipeline::render_batch(
                    heavy.vae,
                    pid_decoder.as_ref().map(|d| d as &dyn LatentDecoder),
                    &scheduler,
                    clean.as_ref(),
                    start_step,
                    base_seed,
                    req,
                    on_progress,
                    |latents, seed, op| {
                        denoise_with_progress(
                            heavy.transformer,
                            &scheduler,
                            sampler_name,
                            seed,
                            latents,
                            &cap,
                            start_step,
                            &req.cancel,
                            op,
                        )
                    },
                )?;
                Ok(GenerationOutput::Images(images))
            },
        )?;
        Ok(images)
    }
}

/// Capability-driven request validation, factored out so it can be unit-tested without loaded
/// weights. Rejects unsupported guidance / negative prompt / conditioning / size / count.
/// Required divisor for requested image dims: VAE downsample (8) × transformer patch (2).
const SIZE_MULTIPLE: u32 = 16;

pub(crate) fn validate_request(
    id: &str,
    caps: &Capabilities,
    req: &GenerationRequest,
) -> Result<()> {
    // Shared capability floor (F-030): count/steps range, size range, negative_prompt/guidance/
    // true_cfg support gating + finiteness, sampler/scheduler/guidance_method membership, and accepted
    // conditioning kinds. The hand-rolled copy validated NONE of sampler/scheduler/true_cfg — a typo'd
    // sampler silently fell back to Euler in `run_flow_sampler`. Delegating to core (like Kolors,
    // F-132) rejects it. `id` is threaded from each of the four registered variants so the error
    // strings name the actual model instead of a hardcoded `z_image_turbo:` (F-089a). The `?` keeps
    // the typed `Error::Unsupported` for capability gaps.
    caps.validate_request(id, req)?;

    // Z-Image-specific checks layered on top of the shared floor:
    // F-096: reject a whitespace-only prompt too (`"   "` renders an effectively unconditioned image),
    // not just the empty string — the boogu F-146 `trim()` fix that didn't travel back to the crate the
    // empty-prompt class originated from.
    if req.prompt.trim().is_empty() {
        return Err(mlx_gen::Error::Msg(format!(
            "{id}: prompt must not be empty"
        )));
    }
    // The pipeline needs dims divisible by VAE downsample (8) × patch (2) = 16. A non-multiple either
    // blows up deep in `patchify`'s reshape with a cryptic mlx error, or truncates in `create_noise`
    // and silently returns a smaller image than requested (F-033) — reject it clearly at the boundary.
    if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
        return Err(mlx_gen::Error::Msg(format!(
            "{id}: {}x{} must be a multiple of {SIZE_MULTIPLE} (VAE 8 × patch 2)",
            req.width, req.height
        )));
    }
    Ok(())
}

// Link-time registration (epic 3720): the macro emits the `inventory::submit!` and bridges the
// crate's rich `Result` into the registry's backend-neutral `gen_core::Result`.
mlx_gen::register_generators! { descriptor => load }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_is_z_image_turbo() {
        let d = descriptor();
        assert_eq!(d.id, "z_image_turbo");
        assert_eq!(d.family, "z-image");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.supports_lora && d.capabilities.supports_lokr);
        assert!(!d.capabilities.supports_guidance);
    }

    #[test]
    fn validate_rejects_empty_prompt() {
        // An empty prompt must surface as a typed error (it would otherwise panic in encode via
        // `as_slice` on the size-0 token array — F-001).
        let caps = descriptor().capabilities;
        let req = GenerationRequest::default(); // default prompt is empty
        let err = validate_request(MODEL_ID, &caps, &req)
            .unwrap_err()
            .to_string();
        assert!(err.contains("empty"), "got: {err}");

        // F-096: a whitespace-only prompt is also rejected (it would otherwise render an effectively
        // unconditioned image) — the boogu `trim()` fix travelling back to z-image.
        let ws = GenerationRequest {
            prompt: "   \t\n".into(),
            ..Default::default()
        };
        let err = validate_request(MODEL_ID, &caps, &ws)
            .unwrap_err()
            .to_string();
        assert!(err.contains("empty"), "whitespace-only prompt got: {err}");
    }

    #[test]
    fn validate_rejects_guidance_and_bad_size() {
        let caps = descriptor().capabilities;
        // guidance on a distilled model (non-empty prompt so the empty-prompt guard doesn't mask it).
        let mut req = GenerationRequest {
            prompt: "a fox".into(),
            guidance: Some(4.0),
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID, &caps, &req).is_err());
        // out-of-range size.
        req = GenerationRequest {
            prompt: "a fox".into(),
            width: 64,
            height: 64,
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID, &caps, &req).is_err());
        // a plain valid request passes.
        req = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID, &caps, &req).is_ok());
    }

    #[test]
    fn validate_rejects_zero_steps() {
        // F-032: an explicit `steps = 0` builds a degenerate 1-element schedule; img2img then indexes
        // `sigmas[init_time_step]` (>= 1) out of bounds (process abort) and txt2img silently decodes
        // pure noise. Reject it at the boundary; `None` (default) and any positive count still pass.
        let caps = descriptor().capabilities;
        let req = GenerationRequest {
            prompt: "a fox".into(),
            steps: Some(0),
            ..Default::default()
        };
        let err = validate_request(MODEL_ID, &caps, &req)
            .unwrap_err()
            .to_string();
        assert!(err.contains("steps must be >= 1"), "got: {err}");

        // `None` falls back to DEFAULT_STEPS and a positive count both pass.
        for steps in [None, Some(1), Some(20)] {
            let ok = GenerationRequest {
                prompt: "a fox".into(),
                steps,
                ..Default::default()
            };
            assert!(
                validate_request(MODEL_ID, &caps, &ok).is_ok(),
                "steps={steps:?}"
            );
        }
    }

    #[test]
    fn validate_rejects_non_multiple_of_16_size() {
        // F-033: in-range dims that aren't a multiple of 16 must be rejected at the boundary, not
        // crash in patchify (1000×1000) or silently truncate (257×257).
        let caps = descriptor().capabilities;
        for (w, h) in [(1000, 1000), (257, 257), (512, 520)] {
            let req = GenerationRequest {
                prompt: "a fox".into(),
                width: w,
                height: h,
                ..Default::default()
            };
            let err = validate_request(MODEL_ID, &caps, &req)
                .unwrap_err()
                .to_string();
            assert!(err.contains("multiple of 16"), "{w}x{h} got: {err}");
        }
        // A multiple-of-16 in-range request still passes.
        let ok = GenerationRequest {
            prompt: "a fox".into(),
            width: 512,
            height: 768,
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID, &caps, &ok).is_ok());
    }

    #[test]
    fn validate_rejects_typoed_sampler_and_scheduler() {
        // F-030: the hand-rolled copy never validated sampler/scheduler, so a typo'd sampler silently
        // fell back to Euler in `run_flow_sampler`. Delegating to the floor now rejects an
        // un-advertised name as a typed Unsupported gap. The error string threads the actual model id.
        let caps = descriptor().capabilities;
        let bad_sampler = GenerationRequest {
            prompt: "a fox".into(),
            sampler: Some("eular".into()), // typo of "euler"
            ..Default::default()
        };
        let err = validate_request(MODEL_ID, &caps, &bad_sampler).unwrap_err();
        assert!(
            matches!(err, mlx_gen::Error::Unsupported(_)),
            "typo'd sampler must be a typed Unsupported gap, got {err:?}"
        );
        assert!(
            err.to_string().contains(MODEL_ID),
            "error must name the model id, got: {err}"
        );
        let bad_scheduler = GenerationRequest {
            prompt: "a fox".into(),
            scheduler: Some("not_a_scheduler".into()),
            ..Default::default()
        };
        assert!(matches!(
            validate_request(MODEL_ID, &caps, &bad_scheduler).unwrap_err(),
            mlx_gen::Error::Unsupported(_)
        ));
        // An advertised curated sampler passes.
        let ok = GenerationRequest {
            prompt: "a fox".into(),
            sampler: Some("euler".into()),
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID, &caps, &ok).is_ok());
    }

    #[test]
    fn validate_rejects_unsupported_conditioning() {
        let caps = descriptor().capabilities;
        let req = GenerationRequest {
            prompt: "a fox".into(),
            conditioning: vec![mlx_gen::Conditioning::Depth {
                image: mlx_gen::Image::default(),
            }],
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID, &caps, &req).is_err());
    }

    #[test]
    fn load_rejects_single_file_source() {
        // Z-Image is a multi-component snapshot, not a single safetensors file.
        let spec = LoadSpec::new(WeightsSource::File("/tmp/z.safetensors".into()));
        // `Box<dyn Generator>` isn't Debug, so use `.err()` rather than `unwrap_err()`.
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }

    #[test]
    fn load_accepts_quantization_spec() {
        // Q4/Q8 is wired (whole model: transformer + text encoder + VAE); a quant spec must get
        // past the load entry point and fail later on the missing snapshot, not on quantization
        // being unsupported.
        for q in [mlx_gen::Quant::Q4, mlx_gen::Quant::Q8] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_quant(q);
            let err = load(&spec).err().expect("expected an error").to_string();
            assert!(!err.contains("quantization"), "got: {err}");
        }
    }

    // ── F-180 (sc-11126): weight-free, default-run proof that Z-Image Turbo's dispatch HONORS
    // `offload_policy`. `build_residency` points at a non-existent snapshot *directory* (so the
    // up-front precision/single-file guard passes) and the discriminator is deferral:
    //   * `Sequential` captures the two per-phase loaders, touches NO weights → `Ok` + `is_sequential`.
    //   * `Resident` eager-loads the text encoder from the missing dir → `Err`.
    // A dispatch that ignored `offload_policy` (always `Resident` — the F-172 regression this whole
    // seam exists to prevent) would eager-load under a `Sequential` request and fail the first
    // assertion. The existing `sequential_residency_real_weights.rs` A/B is `#[ignore]`d; this runs
    // by default.
    fn missing_snapshot_spec(policy: OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(
            "/nonexistent/z-image-residency-test-snapshot".into(),
        ))
        .with_offload_policy(policy)
    }

    #[test]
    fn build_residency_sequential_defers_all_component_loads() {
        let res = build_residency(
            &missing_snapshot_spec(OffloadPolicy::Sequential),
            PRECISION_MSG,
            FILE_MSG,
        )
        .expect("Sequential must defer loads and not touch the (missing) snapshot dir");
        assert!(
            res.is_sequential(),
            "Sequential policy must build a Sequential (deferred) residency"
        );
    }

    #[test]
    fn build_residency_resident_eager_loads_and_fails_on_missing_snapshot() {
        let err = build_residency(
            &missing_snapshot_spec(OffloadPolicy::Resident),
            PRECISION_MSG,
            FILE_MSG,
        )
        .err()
        .expect("Resident must eager-load and fail on a missing snapshot dir");
        let msg = err.to_string();
        assert!(
            !msg.contains("single .safetensors file") && !msg.contains("precision override"),
            "expected an eager-load failure, not the up-front guard: {msg}"
        );
    }
}
