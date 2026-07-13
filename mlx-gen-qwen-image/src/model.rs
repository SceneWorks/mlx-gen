//! `QwenImage` — the Qwen-Image T2I implementation of [`mlx_gen::Generator`], plus its
//! [`descriptor`]/[`load`] entry points and the `inventory` registration that wires it into
//! `mlx_gen`'s registry.
//!
//! [`load`] assembles the model from a `Qwen/Qwen-Image` snapshot directory (see [`crate::loader`])
//! — tokenizer, Qwen2.5-VL text encoder, 60-layer MMDiT, causal-Conv3d VAE — and
//! [`QwenImage::generate`] runs the prompt→image pipeline: tokenize (+ system template) → encode
//! (drop the 34 template tokens) → seeded packed noise → flow-match Euler denoise with classifier-
//! free guidance (two forwards/step) → unpack → VAE decode → RGB8. The component math is parity-
//! proven against the frozen Python fork (slices 1–3); the e2e bf16 path is gated by
//! `tests/e2e_real_weights.rs`.

use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput, GenerationRequest,
    Generator, Image, LatentDecoder, LoadSpec, Modality, ModelDescriptor, OffloadPolicy, Precision,
    Progress, Quant, Residency, Result, WeightsSource,
};
use mlx_gen_pid::{flow_capture_for_request, resolve_pid_decoder_at_sigma, PidEngine};
use std::path::Path;

use crate::loader;
use crate::pipeline::{
    add_noise_by_interpolation, create_noise, decode_and_collect, denoise_with_progress,
    encode_init_latents, encode_prompt, init_time_step, negative_or_fallback, qwen_samplers,
    qwen_schedulers, resolve_run_params, LIGHTNING_SAMPLER, PID_BACKBONE,
};
use crate::text_encoder::QwenTextEncoder;
use crate::transformer::QwenTransformer;
use crate::vae::QwenVae;

/// Registry id for Qwen-Image (matches the SceneWorks worker's `payload.model`).
pub const MODEL_ID: &str = "qwen_image";

/// Qwen-Image's identity + capabilities — constructible without loading weights (registry
/// introspection). This is the **T2I** variant (`qwen_image`), which also accepts a single init
/// `Reference` image for **img2img** (sc-2530); Qwen-Image-Edit ships as a separate `qwen_image_edit`
/// model (sc-2465). LoRA/LoKr is wired (sc-2528). Few-step **Lightning** acceleration is exposed as
/// the `lightning` sampler (sc-2909); an unset sampler is the production flow-match path.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "qwen-image",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            // True CFG with a negative prompt + guidance (not distilled).
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: true,
            // img2img: a single init `Reference` image (+ `image_strength`) seeds the latents via
            // the noise blend (sc-2530, the fork's `Img2Img` path). Reference *conditioning* for
            // editing is the separate `qwen_image_edit` variant (sc-2465).
            conditioning: vec![ConditioningKind::Reference],
            // LoRA/LoKr wired (sc-2528): the fork's `QwenLoRAMapping` targets routed onto the
            // transformer's `AdaptableHost`; stacked + mixed via the core seam.
            supports_lora: true,
            supports_lokr: true,
            // The curated unified-framework integrator menu (epic 7114 P3) + the `lightning` few-step
            // acceleration profile (sc-2909). An unset `req.sampler` is the production flow-match
            // Euler path; any name outside the menu is rejected in `validate_request`.
            samplers: qwen_samplers(),
            schedulers: qwen_schedulers(),
            supported_guidance_methods: vec![],
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: true,
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: false,
            // Flow-match schedule uses the resolution-dependent sigma shift.
            requires_sigma_shift: true,
            // Wired onto the shared `Residency` seam; honors Sequential offload (F-176).
            supports_sequential_offload: true,
        },
    }
}

/// A loaded Qwen-Image generator: the cached descriptor, the (tiny, always-warm) tokenizer, and the
/// heavy-component residency strategy.
pub struct QwenImage {
    descriptor: ModelDescriptor,
    tokenizer: TextTokenizer,
    /// Component-residency strategy (epic 10834 Phase 1, sc-11000; hoisted to the shared seam in
    /// sc-11125), selected from [`LoadSpec::offload_policy`]. `Resident` (default) holds the
    /// Qwen2.5-VL text encoder + DiT + VAE warm for the whole job and across jobs; `Sequential` holds
    /// only the per-phase loader closures and re-loads per generation in phase order (encode → **drop
    /// the text encoder** → denoise/decode), bounding peak unified memory to `max(text-encoder,
    /// DiT+VAE)` instead of the sum — the biggest image-lane win (the Qwen2.5-VL encoder ≈15 GB is
    /// comparable to the 20 GB DiT: 36→20 GB). The [`Residency`] seam owns the eval/drop/clear
    /// discipline, the stage-boundary cancel checks, and the error-safe cache flush.
    residency: Residency<QwenTextEncoder, QwenHeavyOwned>,
}

/// The heavy render-phase components (the MMDiT transformer, the VAE, and the optional PiD decoder) —
/// everything but the text encoder. Owned by the `Resident` components or by a `Sequential` generate.
pub(crate) struct QwenHeavyOwned {
    transformer: QwenTransformer,
    vae: QwenVae,
    /// Optional PiD super-resolving decoder (epic 7840, sc-7845), loaded when `spec.pid` is set;
    /// `req.use_pid` then routes decode through it instead of the VAE. `None` for the plain VAE path.
    pid: Option<PidEngine>,
}

/// A borrow of the heavy render-phase components, so the denoise/decode body runs identically
/// whether they are held resident or were just loaded by the `Sequential` path (candle's `DitRef`).
struct QwenHeavy<'a> {
    transformer: &'a QwenTransformer,
    vae: &'a QwenVae,
    pid: Option<&'a PidEngine>,
}

impl QwenHeavyOwned {
    fn as_ref(&self) -> QwenHeavy<'_> {
        QwenHeavy {
            transformer: &self.transformer,
            vae: &self.vae,
            pid: self.pid.as_ref(),
        }
    }
}

/// Construct a [`QwenImage`] from a [`LoadSpec`].
///
/// `spec.weights` must be a [`WeightsSource::Dir`] pointing at a `Qwen/Qwen-Image` snapshot (the
/// diffusers multi-component tree). Weights load dense at their on-disk dtype (bf16); the text
/// encoder promotes to f32 internally. `spec.quantize` (Q4/Q8) quantizes the transformer only
/// (group_size 64) — the fork's full `quantize=N` scope (sc-2565; see the inline note in
/// [`load_heavy`]). An fp32 precision override is not wired (the validated dense path is bf16) and
/// is rejected rather than silently ignored.
///
/// Component residency (epic 10834 Phase 1, sc-11000; hoisted to the shared [`Residency::from_policy`]
/// seam in sc-11126): `Resident` (default) builds every heavy component now via [`build_residency`]
/// and holds it warm; `Sequential` keeps only the spec and re-loads per generate in phase order
/// (encode → drop the text encoder → denoise/decode) to bound peak memory to `max(text-encoder,
/// DiT+VAE)`. Both use the same per-phase loaders, so the components are byte-identical.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    // Resolve the snapshot dir up front — fail-fast for BOTH policies — then the always-warm
    // tokenizer, then the shared [`build_residency`] dispatch.
    let root = resolve_root(spec)?;
    // F-181: Sequential + a load-time quant over a dense snapshot re-quantizes every generate. An
    // already-packed turnkey loads packed (no re-quant); `Resident` quantizes once. So warn only for
    // the Sequential-over-dense combination that actually pays the repeated cost.
    if let Some(q) = spec.quantize {
        if matches!(spec.offload_policy, OffloadPolicy::Sequential)
            && loader::needs_load_time_quant(root, q.bits(), MODEL_ID)?
        {
            mlx_gen::residency::warn_sequential_requantize(MODEL_ID, q.bits());
        }
    }
    let tokenizer = loader::load_tokenizer(root)?;
    Ok(Box::new(QwenImage {
        descriptor: descriptor(),
        tokenizer,
        residency: build_residency(spec)?,
    }))
}

/// The policy→[`Residency`] dispatch, routed through the single [`Residency::from_policy`] seam
/// (sc-11000; hoisted to the shared seam in sc-11126, F-180) so the `match offload_policy` lives in
/// one place. `Resident` eager-loads the text encoder + heavy bundle now (the heavy loader with
/// `use_pid = true`, loading any PiD overlay once and reusing it); `Sequential` captures the two
/// per-phase loaders and loads nothing now, deferring each to [`Residency::run`]. Both use the same
/// [`load_text_encoder_only`] / [`load_heavy`], so the `Resident` composition is byte-identical to the
/// pre-seam one. The deferral is weight-free-testable: under `Sequential` this touches no component
/// weights, so a dispatch that ignored `offload_policy` would eager-load and fail the "Sequential
/// defers" unit test.
pub(crate) fn build_residency(
    spec: &LoadSpec,
) -> Result<Residency<QwenTextEncoder, QwenHeavyOwned>> {
    let spec_text = spec.clone();
    let spec_heavy = spec.clone();
    Residency::from_policy(
        spec.offload_policy,
        move || load_text_encoder_only(resolve_root(&spec_text)?),
        move |use_pid| load_heavy(&spec_heavy, resolve_root(&spec_heavy)?, use_pid),
    )
}

/// Precision guard (only dense bf16 is wired) + snapshot-dir resolution (rejecting a single-file
/// source), shared by [`build_residency`]'s per-phase loaders (sc-11000, sc-11126).
fn resolve_root(spec: &LoadSpec) -> Result<&Path> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "qwen_image: only dense bf16 is wired in the Rust port (drop the precision override)"
                .into(),
        ));
    }
    match &spec.weights {
        WeightsSource::Dir(p) => Ok(p),
        WeightsSource::File(_) => Err(Error::Msg(
            "qwen_image expects a snapshot directory (tokenizer/ text_encoder/ transformer/ \
             vae/), not a single .safetensors file"
                .into(),
        )),
    }
}

/// Load the Qwen2.5-VL text encoder — the phase-A component dropped first under `Sequential`.
/// Qwen-Image quantizes the **transformer only** (the fork marks the `text_encoder` component
/// `skip_quantization=True` — "Quantization causes significant semantic degradation"), so unlike
/// Z-Image the encoder is never quantized; the `Resident` and `Sequential` paths build byte-identical
/// encoders.
fn load_text_encoder_only(root: &Path) -> Result<QwenTextEncoder> {
    loader::load_text_encoder(root)
}

/// Load the heavy render-phase components — MMDiT transformer (+ Q4/Q8 + LoRA/LoKr residuals), VAE,
/// and the optional PiD overlay — everything but the text encoder. Factored so the `Sequential`
/// path loads these AFTER the encoder is dropped (bounding peak to `max(text-encoder, DiT+VAE)`).
/// Quantize-then-adapters order matches the pre-sc-11000 `load`; the components are independent of
/// the text encoder (separate weight files, deterministic RNG-free quant), so the `Resident`
/// composition is byte-identical.
fn load_heavy(spec: &LoadSpec, root: &Path, load_pid: bool) -> Result<QwenHeavyOwned> {
    // Q4/Q8 quantizes the **transformer only** (group_size 64) after the dense bf16 load. This is
    // the fork's full `quantize=N` scope, not a descope (sc-2565): `QwenWeightDefinition` marks the
    // `text_encoder` component `skip_quantization=True`, and the VAE is all-conv (`nn.Conv2d`/
    // `Conv3d` lack `to_quantized`), so the fork's `nn.quantize(vae)` is a no-op. The transformer is
    // the only component with quantizable leaves. (Z-Image differs — its fork *does* quantize the
    // TE+VAE, hence sc-2532; do not generalize that here.)
    let mut transformer = loader::load_transformer(root)?;
    if let Some(q) = spec.quantize {
        // F-076: reject a requested-vs-packed quant-tier mismatch instead of silently serving the
        // snapshot's tier; skip the no-op quantize when the turnkey is already packed at the
        // requested bits (see `loader::needs_load_time_quant`).
        if loader::needs_load_time_quant(root, q.bits(), MODEL_ID)? {
            transformer.quantize(q.bits())?;
        }
    }
    // LoRA/LoKr (sc-2528): applied after quantization, as forward-time residuals over the
    // (possibly quantized) transformer — fork-faithful. No-op when `spec.adapters` is empty.
    if !spec.adapters.is_empty() {
        crate::adapters::apply_qwen_adapters(&mut transformer, &spec.adapters)?;
    }
    // Optional PiD decoder overlay (sc-7845): load the qwenimage student + Gemma-2 caption encoder
    // once when `spec.pid` is set AND this generate uses it (`load_pid`, F-177) — Resident passes
    // `true` (loaded once, reused), Sequential passes `req.use_pid` so a non-PiD generate skips the
    // student + its Gemma-2 caption encoder entirely.
    let pid = if load_pid {
        spec.pid
            .as_ref()
            .map(|p| PidEngine::from_spec(p, PID_BACKBONE))
            .transpose()?
    } else {
        None
    };
    let vae = loader::load_vae(root)?;
    Ok(QwenHeavyOwned {
        transformer,
        vae,
        pid,
    })
}

impl QwenImage {
    /// Extract the single img2img init image + its strength from the request's conditioning. The
    /// per-reference strength wins over `req.strength`. Qwen-Image T2I img2img conditions on exactly
    /// one init image, so more than one `Reference` is an error (the multi-image edit path is
    /// `qwen_image_edit` + `MultiReference`, sc-2529). Returns `None` for pure txt2img.
    fn resolve_reference<'a>(
        &self,
        req: &'a GenerationRequest,
    ) -> Result<Option<(&'a Image, Option<f32>)>> {
        let mut reference = None;
        for c in &req.conditioning {
            if let Conditioning::Reference { image, strength } = c {
                if reference.is_some() {
                    return Err(Error::Msg(
                        "qwen_image: multiple reference images are not supported (single img2img \
                         init only)"
                            .into(),
                    ));
                }
                reference = Some((image, strength.or(req.strength)));
            }
        }
        Ok(reference)
    }
}

mlx_gen::impl_generator!(QwenImage {
    validate: |s, req| validate_request(s.descriptor.id, &s.descriptor.capabilities, req),
    generate: generate_impl,
});

impl QwenImage {
    /// The rich-`Result` body behind [`Generator::generate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the family
    /// helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    ///
    /// The staged residency lifecycle (encode pos+neg → drop the Qwen2.5-VL encoder under `Sequential`
    /// → load the DiT/VAE/PiD → denoise/decode → free the heavy bundle) is driven by the shared
    /// [`Residency::run`] seam (sc-11125), which owns the eval/drop/clear discipline, the
    /// stage-boundary cancel checks, and the error-safe cache flush.
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;

        // Shared step/sampler/guidance/seed resolution (F-117); `req.sampler == "lightning"` selects
        // the few-step recipe, else the production resolution-dependent schedule.
        let params = resolve_run_params(req, req.width, req.height);

        // img2img: a single `Reference` image, with a per-reference strength overriding `req.strength`.
        // `start_step = 0` for pure txt2img (the fork's `Config.init_time_step`).
        let reference = self.resolve_reference(req)?;
        let start_step = match reference {
            Some((_, strength)) => init_time_step(params.steps, strength),
            None => 0,
        };
        let is_img2img = start_step > 0;
        // Lightning is the CFG-distilled few-step *txt2img* recipe; an init image (img2img) is out of
        // scope (its blend seeds a different trajectory than the distillation targets).
        if params.is_lightning && is_img2img {
            return Err(Error::Msg(
                "qwen_image: the lightning sampler is txt2img-only (no img2img init image)".into(),
            ));
        }

        // Phase A: prompt → embeds (epic 10834 Phase 1, sc-11000; sc-11125). Under `Sequential` the
        // shared seam loads the Qwen2.5-VL encoder, encodes pos+neg, materializes, then DROPS it +
        // `clear_cache()` so its ~15 GB frees before the DiT/VAE load below — the peak-bounding win.
        // Under `Resident` it borrows the warm encoder. Positive conditioning (bf16) always; the
        // negative branch is built only for true CFG (`neg` is `None` under Lightning, CFG-distilled).
        self.residency.run(
            &req.cancel,
            req.use_pid,
            on_progress,
            |te: &QwenTextEncoder| {
                let pos = encode_prompt(&self.tokenizer, te, &req.prompt, MODEL_ID)?;
                let neg = if params.is_lightning {
                    None
                } else {
                    Some(encode_prompt(
                        &self.tokenizer,
                        te,
                        negative_or_fallback(req),
                        MODEL_ID,
                    )?)
                };
                Ok((pos, neg))
            },
            // Materialize pos (+neg) while the encoder is still alive (Sequential only) — MLX is lazy,
            // so an un-evaluated output keeps the encoder referenced and the drop would free nothing.
            |(pos, neg)| {
                match neg {
                    Some(neg) => mlx_rs::transforms::eval([pos, neg])?,
                    None => mlx_rs::transforms::eval([pos])?,
                }
                Ok(())
            },
            // ── Establish the heavy render components (DiT + VAE + PiD) and run the denoise/decode
            // body once against the `heavy` borrow — identical for both residencies.
            |heavy_owned, enc, on_progress| {
                let heavy = heavy_owned.as_ref();
                let (pos, neg) = enc;

                // VAE-encode the init image to packed clean latents (f32) ONCE — it's seed-independent
                // (LANCZOS resize + a full VAE encode), so doing it inside the per-image loop ran the encoder
                // `count` times for identical output (F-118). Under `Sequential` this runs after the text
                // encoder was dropped and the VAE loaded — independent of `pos`/`neg`, so byte-identical.
                let clean = if is_img2img {
                    let (image, _) = reference.expect("is_img2img implies a reference");
                    Some(encode_init_latents(
                        heavy.vae, image, req.width, req.height,
                    )?)
                } else {
                    None
                };

                // Decode seam (sc-7845) + `from_ldm` early-stop (sc-7993): when `req.use_pid` and
                // `req.pid_capture_sigma` ask for an early exit on this flow-match schedule, decode the
                // partially-denoised x_k at the achieved degrade σ and truncate the denoise to the matching
                // step; otherwise the clean σ=0 full-denoise path (`capture_sigma = 0`, full schedule).
                let (capture_sigma, keep) =
                    flow_capture_for_request(req, &params.sigmas, start_step);
                let pid_decoder = resolve_pid_decoder_at_sigma(
                    heavy.pid,
                    req,
                    params.base_seed,
                    MODEL_ID,
                    capture_sigma,
                )?;
                let decoder: &dyn LatentDecoder = match &pid_decoder {
                    Some(d) => d,
                    None => heavy.vae,
                };
                let denoise_sigmas = &params.sigmas[..keep];
                let images = decode_and_collect(
                    decoder,
                    req.count,
                    params.base_seed,
                    req.width,
                    req.height,
                    on_progress,
                    |seed, progress| {
                        // Latents stay f32 through the loop: the fork keeps txt2img/img2img noise f32, and MLX
                        // promotes the bf16 transformer weights to f32 per-op (only `prompt_embeds` is bf16).
                        let noise = create_noise(seed, req.width, req.height)?;
                        let latents = match &clean {
                            // Blend the (hoisted) clean latents with this image's noise at
                            // `sigma = sigmas[init_time_step]` (fork `create_for_txt2img_or_img2img`).
                            Some(clean) => {
                                let sigma = params.sigmas[start_step];
                                add_noise_by_interpolation(clean, &noise, sigma)?
                            }
                            None => noise,
                        };
                        denoise_with_progress(
                            heavy.transformer,
                            params.sampler_name.as_deref(),
                            denoise_sigmas,
                            seed,
                            latents,
                            &pos,
                            neg.as_ref(),
                            params.guidance,
                            req.width,
                            req.height,
                            start_step,
                            &req.cancel,
                            progress,
                        )
                    },
                )?;
                Ok(GenerationOutput::Images(images))
            },
        )
    }
}

/// Capability-driven request validation, factored out for unit testing without loaded weights.
pub(crate) fn validate_request(
    id: &str,
    caps: &Capabilities,
    req: &GenerationRequest,
) -> Result<()> {
    // The shared capability floor (F-101): count, steps==0 + the F-004 ceiling, size range,
    // negative/guidance/true_cfg support gating + the F-053/F-001 finiteness guard, and — the checks
    // qwen's hand-rolled validator missed — sampler/scheduler/guidance_method membership (an unknown
    // scheduler silently fell back to the native schedule; guidance_method was silently ignored) and
    // non-finite guidance/true_cfg (NaN rendered garbage). `?` keeps the typed `Unsupported` for gaps.
    caps.validate_request(id, req)?;
    // Qwen-Image latents pack 2×2; sizes must be a multiple of 16 per side (VAE/8 then patch/2).
    if !req.width.is_multiple_of(16) || !req.height.is_multiple_of(16) {
        return Err(Error::Msg(format!(
            "{}x{} must be a multiple of 16 per side",
            req.width, req.height
        )));
    }
    // The production flow-match schedule needs >= 2 steps: at 1 step `qwen_sigmas`' terminal-sigma
    // rescale divides by zero (`scale == 0`) and produces a `[NaN, 0.0]` schedule that silently
    // renders garbage (F-113). The floor already rejects an explicit `0`; this catches the production
    // `1` (lightning's distilled few-step recipe is unaffected, so only guard the production path; an
    // unset `steps` uses the safe `DEFAULT_STEPS`).
    let is_lightning = req.sampler.as_deref() == Some(LIGHTNING_SAMPLER);
    if !is_lightning {
        if let Some(steps) = req.steps {
            if steps < 2 {
                return Err(Error::Msg(format!(
                    "qwen_image: steps must be >= 2 for the production sampler (got {steps})"
                )));
            }
        }
    }
    Ok(())
}

// Link-time registration (epic 3720): the macro emits the `inventory::submit!` and bridges the
// crate's rich `Result` into the registry's backend-neutral `gen_core::Result`.
/// Per-component on-disk footprint (sc-10894) for the MLX fit-gate's staged-residency split — the
/// Qwen2.5-VL text/vision encoder (`text_encoder/`; the edit variant reads `visual.*` from the same
/// subdir), the DiT (`transformer/`), and the VAE (`vae/`), summed from the subdirs [`crate::loader`]
/// loads. Shared by every qwen-image id (base/edit/control); the control checkpoint is folded by the
/// worker.
pub(crate) fn component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    mlx_gen::PerComponentBytes::from_spec_subdirs(
        spec,
        &["text_encoder"],
        &["transformer"],
        &["vae"],
    )
}

mlx_gen::register_generators! { descriptor => load ; footprint = component_footprint }

#[cfg(test)]
mod tests {
    use super::*;

    /// Documents + guards the PiD `from_ldm` capture-index policy (sc-7993) on the **production
    /// flow-match** schedule — the 50-step trajectory the sc-7843 runB validation captured. A σ ceiling
    /// is schedule-agnostic, so the worker/web can expose one knob and it maps to the right step on any
    /// trajectory. Prints the table with `--nocapture`; the assertions pin the runB anchor
    /// (`x_t@step44, σ≈0.199`) so a schedule/mu regression that moved the capture index would fail here.
    #[test]
    fn pid_capture_indices_production_50step() {
        let sigmas = crate::pipeline::qwen_scheduler(50, 1024, 1024).sigmas; // 51 entries, trailing 0
        for ceiling in [0.5_f32, 0.3, 0.2, 0.1] {
            if let Some(p) = mlx_gen::flow_capture_plan(&sigmas, Some(ceiling)) {
                eprintln!(
                    "[qwen 50-step] ceiling σ≤{ceiling:.2} -> stop after {} of 50 steps, decode at σ={:.4} (saves {} steps)",
                    p.keep - 1,
                    p.sigma,
                    50 - (p.keep - 1),
                );
            }
        }
        // runB anchor: ceiling 0.2 lands at the same x_t@44 (σ≈0.199) the sc-7843 runB harness decoded.
        let p = mlx_gen::flow_capture_plan(&sigmas, Some(0.2)).expect("a sub-0.2 capture exists");
        assert_eq!(
            p.keep, 45,
            "ceiling 0.2 → keep first 45 (stop after step 44)"
        );
        assert!(
            (0.18..=0.20).contains(&p.sigma),
            "achieved σ at step 44 ≈ 0.199, got {}",
            p.sigma
        );
        assert_eq!(p.sigma, sigmas[44]);
    }

    /// The same policy over Krea's 8-step Turbo trajectory and the production path's few-step Lightning
    /// regime: a *coarse* schedule means a σ ceiling resolves to a much earlier fractional stop (fewer
    /// steps total), which is exactly why the index differs per trajectory (the story's "8-step ≠
    /// 50-step"). Prints the table; asserts the early-stop is a genuine truncation with residual noise.
    #[test]
    fn pid_capture_indices_fewstep_trajectories() {
        // Krea Turbo (8-step) shares the exponential-mu flow-match shape; reuse the production builder
        // at 8 steps as a faithful stand-in for the coarse trajectory (same family of sigmas).
        let sigmas8 = crate::pipeline::qwen_scheduler(8, 1024, 1024).sigmas; // 9 entries
        for ceiling in [0.5_f32, 0.3, 0.2] {
            if let Some(p) = mlx_gen::flow_capture_plan(&sigmas8, Some(ceiling)) {
                eprintln!(
                    "[8-step] ceiling σ≤{ceiling:.2} -> stop after {} of 8 steps, decode at σ={:.4}",
                    p.keep - 1,
                    p.sigma,
                );
                assert!(p.sigma <= ceiling && p.sigma > 0.0);
                assert!(p.keep < sigmas8.len(), "a real early stop");
            }
        }
    }

    #[test]
    fn descriptor_is_qwen_image() {
        let d = descriptor();
        assert_eq!(d.id, "qwen_image");
        assert_eq!(d.family, "qwen-image");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.supports_true_cfg);
        assert!(d.capabilities.requires_sigma_shift);
        // Lightning acceleration is advertised (sc-2909).
        assert!(d.capabilities.samplers.contains(&LIGHTNING_SAMPLER));
    }

    #[test]
    fn validate_sampler_selection() {
        let caps = descriptor().capabilities;
        // Unset sampler → the production flow-match path.
        let req = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID, &caps, &req).is_ok());
        // The advertised `lightning` sampler is accepted.
        let req = GenerationRequest {
            prompt: "a fox".into(),
            sampler: Some(LIGHTNING_SAMPLER.into()),
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID, &caps, &req).is_ok());
        // A curated sampler (epic 7114) is now accepted — `lcm`/`dpmpp_2m`/… select that integrator.
        let req = GenerationRequest {
            prompt: "a fox".into(),
            sampler: Some("dpmpp_2m".into()),
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID, &caps, &req).is_ok());
        // A name outside the menu is still rejected, not silently downgraded.
        let req = GenerationRequest {
            prompt: "a fox".into(),
            sampler: Some("nonsense".into()),
            ..Default::default()
        };
        let err = validate_request(MODEL_ID, &caps, &req)
            .expect_err("expected an error")
            .to_string();
        assert!(err.contains("unsupported sampler"), "got: {err}");
    }

    #[test]
    fn validate_rejects_production_steps_below_two() {
        // F-113: 0/1 production steps make qwen_sigmas' terminal rescale divide by zero → NaN
        // schedule. Reject them; Lightning few-step and the default (unset) path stay valid.
        let caps = descriptor().capabilities;
        let prod = |steps| GenerationRequest {
            prompt: "a fox".into(),
            steps,
            ..Default::default()
        };
        // An explicit `0` is rejected by the shared floor ("steps must be >= 1"); the production `1`
        // is caught by qwen's own >= 2 guard. Both are rejected — that is the contract.
        let err0 = validate_request(MODEL_ID, &caps, &prod(Some(0)))
            .expect_err("steps 0 must be rejected")
            .to_string();
        assert!(err0.contains("steps must be >= 1"), "steps 0 got: {err0}");
        let err1 = validate_request(MODEL_ID, &caps, &prod(Some(1)))
            .expect_err("production steps 1 must be rejected")
            .to_string();
        assert!(err1.contains("steps must be >= 2"), "steps 1 got: {err1}");
        assert!(validate_request(MODEL_ID, &caps, &prod(Some(2))).is_ok());
        assert!(validate_request(MODEL_ID, &caps, &prod(None)).is_ok());
        // Lightning at 1 step is fine (distilled few-step recipe).
        let lightning = GenerationRequest {
            prompt: "a fox".into(),
            steps: Some(1),
            sampler: Some(LIGHTNING_SAMPLER.into()),
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID, &caps, &lightning).is_ok());
    }

    #[test]
    fn validate_rejects_bad_size_and_conditioning() {
        let caps = descriptor().capabilities;
        // out-of-range size.
        let req = GenerationRequest {
            width: 64,
            height: 64,
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID, &caps, &req).is_err());
        // non-multiple-of-16 size.
        let req = GenerationRequest {
            width: 1000,
            height: 1024,
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID, &caps, &req).is_err());
        // T2I accepts no conditioning.
        let req = GenerationRequest {
            conditioning: vec![Conditioning::Depth {
                image: mlx_gen::Image::default(),
            }],
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID, &caps, &req).is_err());
        // guidance + negative prompt + valid size passes.
        let req = GenerationRequest {
            prompt: "a fox".into(),
            negative_prompt: Some("blurry".into()),
            guidance: Some(4.0),
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID, &caps, &req).is_ok());
    }

    #[test]
    fn load_rejects_single_file() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/q.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }

    #[test]
    fn load_accepts_q8_spec() {
        // Q8 is wired (transformer-only); a Q8 spec must get past the quant gate and fail later on
        // the missing snapshot, not on quantization being unsupported.
        let spec =
            LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_quant(mlx_gen::Quant::Q8);
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(!err.contains("quantization"), "got: {err}");
    }

    // ── F-180 (sc-11126): weight-free, default-run proof that Qwen-Image's dispatch HONORS
    // `offload_policy`. `build_residency` points at a non-existent snapshot *directory* (so the
    // up-front precision/single-file guard in `resolve_root` passes) and the discriminator is deferral:
    //   * `Sequential` captures the two per-phase loaders, touches NO weights → `Ok` + `is_sequential`.
    //   * `Resident` eager-loads the Qwen2.5-VL text encoder from the missing dir → `Err`.
    // A dispatch that ignored `offload_policy` (always `Resident`) would eager-load under a `Sequential`
    // request and fail the first assertion. The A/B real-weight test is `#[ignore]`d; this runs by
    // default.
    fn missing_snapshot_spec(policy: OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(
            "/nonexistent/qwen-image-residency-test-snapshot".into(),
        ))
        .with_offload_policy(policy)
    }

    #[test]
    fn build_residency_sequential_defers_all_component_loads() {
        let res = build_residency(&missing_snapshot_spec(OffloadPolicy::Sequential))
            .expect("Sequential must defer loads and not touch the (missing) snapshot dir");
        assert!(
            res.is_sequential(),
            "Sequential policy must build a Sequential (deferred) residency"
        );
    }

    #[test]
    fn build_residency_resident_eager_loads_and_fails_on_missing_snapshot() {
        let err = build_residency(&missing_snapshot_spec(OffloadPolicy::Resident))
            .err()
            .expect("Resident must eager-load and fail on a missing snapshot dir");
        let msg = err.to_string();
        assert!(
            !msg.contains("single .safetensors file") && !msg.contains("precision override"),
            "expected an eager-load failure, not the up-front guard: {msg}"
        );
    }
}
