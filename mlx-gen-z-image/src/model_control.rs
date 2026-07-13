//! `ZImageTurboControl` — the Z-Image-turbo **Fun-Controlnet-Union** variant (sc-2349 / sc-2257):
//! strict pose (VACE-style) conditioning via `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`,
//! registered as its own `Generator` (`z_image_turbo_control`).
//!
//! Identical to [`crate::model::ZImageTurbo`] except the transformer is a
//! [`ZImageControlTransformer`] (base DiT + control branch) and `generate` threads a VAE-encoded
//! control context through it. [`load`] needs the base snapshot (`spec.weights`) **and** the
//! control checkpoint (`spec.control`); it applies both dense, then quantizes the whole transformer
//! together (the fork's `d32454c` ordering — quantizing before the overlay would leave the control
//! Linears unable to accept their real weights). The control patch embedder stays dense (its
//! in-features is not divisible by the quant group size).
//!
//! Parity-proven against the frozen Python fork (sc-2257): the control branch is bit-identical to
//! the base transformer at `control_context_scale = 0`, and the full control render matches the
//! fork's control golden — see `tests/z_control_transformer.rs` and `tests/control_real_weights.rs`.

use mlx_gen::gen_core;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, default_seed, require_base_dir,
    require_control, resolve_flow_schedule, AcceptedControlKinds, Capabilities, ConditioningKind,
    ControlBranch, Error, FlowMatchEuler, GenerationOutput, GenerationRequest, Generator, LoadSpec,
    Modality, ModelDescriptor, OffloadPolicy, Precision, Progress, Quant, Residency, Result,
    WeightsSource,
};
use mlx_rs::Dtype;
use std::path::Path;

use crate::control_transformer::ZImageControlTransformer;
use crate::loader;
use crate::model::{validate_request, DEFAULT_STEPS, SCHEDULE_SHIFT};
use crate::pipeline::{
    self, denoise_control_with_progress, encode_control_context, encode_init_latents,
    init_time_step,
};
use crate::text_encoder::TextEncoder;
use crate::vae::Vae;

/// Registry id for the Z-Image-turbo Fun-Controlnet-Union variant.
pub const MODEL_ID: &str = "z_image_turbo_control";

/// The control variant's identity + capabilities. Same distilled turbo base (no CFG / negative
/// prompt) as `z_image_turbo`, plus `Control` conditioning (the required pose/union skeleton) and
/// `Reference` (an optional img2img init — the fork's `generate_image` accepts both).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "z-image",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_negative_prompt: false,
            supports_guidance: false,
            supports_true_cfg: false,
            // Control (required) + an optional img2img Reference init.
            conditioning: vec![ConditioningKind::Control, ConditioningKind::Reference],
            supports_lora: true,
            supports_lokr: true,
            // Curated unified-framework integrator menu (epic 7114 P3), as the base turbo variant.
            samplers: curated_sampler_names(),
            // Curated scheduler menu (epic 7114), as the base turbo variant — static-shift default.
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

/// A loaded control generator: the cached descriptor, the (tiny, always-warm) tokenizer, and the
/// component-residency strategy (base text encoder + control transformer + VAE), driven through the
/// shared [`Residency`] seam so the control variant honors [`LoadSpec::offload_policy`] family-wide
/// (sc-11124, F-172) — `Sequential` drops the text encoder after the encode phase, bounding peak
/// unified memory to `max(text-encoder, control-DiT+VAE)`.
pub struct ZImageTurboControl {
    descriptor: ModelDescriptor,
    tokenizer: TextTokenizer,
    /// Component-residency strategy (sc-11124), selected from [`LoadSpec::offload_policy`] via the
    /// shared [`load_control_residency`] builder.
    residency: Residency<TextEncoder, ZImageControlHeavyOwned>,
}

/// The heavy render-phase components for both Z-Image ControlNet variants (the composed base+control
/// transformer and the VAE) — everything but the text encoder. There is no PiD overlay on the control
/// path (sc-7846 is base-`z_image_turbo`-only), so the seam's `use_pid` loader flag is ignored. Owned
/// by the `Resident` components or by a `Sequential` generate. `pub(crate)` so the **base** control
/// sibling ([`crate::model_base_control`]) shares the identical bundle + seam (sc-11124).
pub(crate) struct ZImageControlHeavyOwned {
    pub(crate) transformer: ZImageControlTransformer,
    pub(crate) vae: Vae,
}

/// A borrow of the heavy control components, so the denoise/decode body runs identically whether they
/// are held resident or were just loaded by the `Sequential` path.
pub(crate) struct ZImageControlHeavy<'a> {
    pub(crate) transformer: &'a ZImageControlTransformer,
    pub(crate) vae: &'a Vae,
}

impl ZImageControlHeavyOwned {
    pub(crate) fn as_ref(&self) -> ZImageControlHeavy<'_> {
        ZImageControlHeavy {
            transformer: &self.transformer,
            vae: &self.vae,
        }
    }
}

/// Precision guard (only dense bf16 is wired) + base-snapshot-dir resolution + the **required**
/// control-checkpoint resolution, shared by [`load_control_residency`]'s `Resident` composition and its
/// `Sequential` per-phase loaders (sc-11124). Preserves the original message order: a single-file base
/// is rejected first (via [`require_base_dir`]), then a missing control checkpoint (via
/// [`require_control`]). `precision_msg` is the per-id override-rejection text (turbo vs base control).
pub(crate) fn resolve_control_base_and_control<'a>(
    spec: &'a LoadSpec,
    model_id: &str,
    precision_msg: &str,
) -> Result<(&'a Path, &'a WeightsSource)> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(precision_msg.into()));
    }
    let root = require_base_dir(spec, model_id, "a base snapshot directory")?;
    let control = require_control(spec, model_id, "Fun-Controlnet-Union")?;
    Ok((root, control))
}

/// Load the text encoder — the phase-A component dropped first under `Sequential`. Quantized with the
/// whole-model bits when `quant` is set (the Z-Image control quant scope covers the text encoder), so
/// the `Resident` and `Sequential` paths build byte-identical encoders.
pub(crate) fn load_control_text_encoder_only(
    root: &Path,
    quant: Option<Quant>,
) -> Result<TextEncoder> {
    let mut text_encoder = loader::load_text_encoder(root)?;
    if let Some(q) = quant {
        text_encoder.quantize(q.bits())?;
    }
    Ok(text_encoder)
}

/// Load the heavy control render components — the composed base+control transformer (+ Q4/Q8 + the
/// base's LoRA/LoKr residuals) and the VAE (+ Q4/Q8) — everything but the text encoder. The
/// overlay-then-quantize order (dense base + dense control, THEN quantize) matches the pre-sc-11124
/// hand-written `load`; the components are independent of the text encoder (separate weight files,
/// deterministic RNG-free quant), so the `Resident` composition is byte-identical. Shared by both
/// control variants (turbo + base) — they differ only in the generate-time schedule + CFG.
pub(crate) fn load_control_heavy(
    spec: &LoadSpec,
    root: &Path,
    control: &WeightsSource,
) -> Result<ZImageControlHeavyOwned> {
    // Base + control applied dense first, THEN quantize together (the fork's ordering): quantizing
    // before the overlay would replace the control Linears with QuantizedLinear that can't accept
    // the raw bf16 control weights.
    let mut transformer = loader::load_control_transformer(root, control)?;
    let mut vae = loader::load_vae(root)?;
    if let Some(q) = spec.quantize {
        let bits = q.bits();
        transformer.quantize(bits)?;
        vae.quantize(bits)?;
    }
    // LoRA/LoKr (sc-2602): install onto the composed base DiT (the control branch is not an adapter
    // target). Same load-time, post-quantize, residual-over-base path. No-op when `spec.adapters` is
    // empty.
    if !spec.adapters.is_empty() {
        crate::adapters::apply_z_image_adapters(&mut transformer, &spec.adapters)?;
    }
    Ok(ZImageControlHeavyOwned { transformer, vae })
}

/// Build the tokenizer + [`Residency`] seam for either Z-Image ControlNet variant, honoring
/// [`LoadSpec::offload_policy`] (sc-11124, F-172). `Resident` (default) builds every heavy component
/// now and holds it warm; `Sequential` keeps only the spec and re-loads per generate in phase order
/// (encode → drop the text encoder → denoise/decode). Both use the same per-phase loaders, so the
/// components are byte-identical. Parameterized by `model_id` + the per-id precision-override message so
/// the base control sibling shares it (before sc-11124 both control variants ignored `offload_policy`
/// and silently loaded full-`Resident`).
pub(crate) fn load_control_residency(
    spec: &LoadSpec,
    model_id: &'static str,
    precision_msg: &'static str,
) -> Result<(
    TextTokenizer,
    Residency<TextEncoder, ZImageControlHeavyOwned>,
)> {
    // Validate precision + base dir + the required control checkpoint up front (fail fast, same for
    // BOTH policies); then the always-warm tokenizer, then the shared [`build_control_residency`]
    // dispatch.
    let (root, _control) = resolve_control_base_and_control(spec, model_id, precision_msg)?;
    // F-181: Sequential + a load-time quant over a dense snapshot re-quantizes every generate; only
    // that combination pays the repeated cost, so gate the warning on it.
    if let Some(q) = spec.quantize {
        if matches!(spec.offload_policy, OffloadPolicy::Sequential)
            && loader::needs_load_time_quant(root, q.bits(), model_id)?
        {
            mlx_gen::residency::warn_sequential_requantize(model_id, q.bits());
        }
    }
    let tokenizer = loader::load_tokenizer(root)?;
    Ok((
        tokenizer,
        build_control_residency(spec, model_id, precision_msg)?,
    ))
}

/// The policy→[`Residency`] dispatch both Z-Image control variants share (turbo + base control),
/// routed through the single [`Residency::from_policy`] seam (sc-11126, F-180) so neither re-derives
/// the `match offload_policy` (before sc-11124 both control variants ignored `offload_policy` and
/// silently loaded full-`Resident`). `Resident` eager-loads the text encoder + heavy (base DiT +
/// control branch + VAE) now; `Sequential` captures the two per-phase loaders and loads nothing now.
/// The pose branch carries no PiD overlay, so the seam's `use_pid` arg is unused. Weight-free-testable:
/// under `Sequential` this touches no component weights, so a dispatch that ignored the policy would
/// eager-load and fail the "Sequential defers" unit test.
pub(crate) fn build_control_residency(
    spec: &LoadSpec,
    model_id: &'static str,
    precision_msg: &'static str,
) -> Result<Residency<TextEncoder, ZImageControlHeavyOwned>> {
    let spec_text = spec.clone();
    let spec_heavy = spec.clone();
    Residency::from_policy(
        spec.offload_policy,
        move || {
            let (root, _control) =
                resolve_control_base_and_control(&spec_text, model_id, precision_msg)?;
            load_control_text_encoder_only(root, spec_text.quantize)
        },
        move |_use_pid| {
            let (root, control) =
                resolve_control_base_and_control(&spec_heavy, model_id, precision_msg)?;
            load_control_heavy(&spec_heavy, root, control)
        },
    )
}

/// The per-id precision-override rejection message for the turbo control variant, shared by
/// [`load_control_residency`]'s eager guard and its `Sequential` per-phase loaders.
const PRECISION_MSG: &str =
    "z_image_turbo_control: only dense bf16 is wired (the text encoder runs \
     f32 internally); drop the precision override";

/// Construct a [`ZImageTurboControl`] from a [`LoadSpec`].
///
/// `spec.weights` must be a [`WeightsSource::Dir`] base `Tongyi-MAI/Z-Image-Turbo` snapshot, and
/// `spec.control` (required) the Fun-Controlnet-Union checkpoint (a single `.safetensors` `File`,
/// or a `Dir` of them). Weights load dense (bf16); `spec.quantize` (Q4/Q8) then quantizes the whole
/// transformer (base + control, group_size 64) plus the text encoder + VAE — the fork's whole-model
/// quant, with the control patch embedder left dense (its in-features is not a multiple of 64).
///
/// Component residency (sc-11124, F-172): `Resident` (default) holds every heavy component warm;
/// `Sequential` re-loads per generate in phase order to bound peak memory — routed through the shared
/// [`load_control_residency`] builder.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let (tokenizer, residency) = load_control_residency(spec, MODEL_ID, PRECISION_MSG)?;
    Ok(Box::new(ZImageTurboControl {
        descriptor: descriptor(),
        tokenizer,
        residency,
    }))
}

/// The Fun-Controlnet-Union is a *union* ControlNet (pose/canny/depth share one VAE-encoded control
/// path), so all the control boilerplate (resolve/validate-present + the load helpers above) comes
/// from the shared trait (sc-8241). F-089: this is the SAME union checkpoint as the base variant, so
/// it shares the base `accepted_kinds()` (`Only([Pose, Canny, Depth])`) — previously it fell back to
/// the trait default `AcceptedControlKinds::Any`, accepting `Other("scribble")` the base rejects.
impl ControlBranch for ZImageTurboControl {
    fn model_id(&self) -> &'static str {
        MODEL_ID
    }

    fn accepted_control_kinds(&self) -> AcceptedControlKinds {
        crate::model_base_control::accepted_kinds()
    }
}

impl ZImageTurboControl {
    /// The rich-`Result` body behind [`Generator::generate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the family
    /// helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;

        let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
        let base_seed = req.seed.unwrap_or_else(default_seed);

        // Required pose/union control + optional img2img init.
        let (control_image, control_scale) = self.resolve_control(req)?;
        let reference = pipeline::resolve_reference(req, MODEL_ID)?;
        let start_step = match reference {
            Some((_, strength)) => init_time_step(steps, strength),
            None => 0,
        };
        let is_img2img = start_step > 0;

        // The staged residency lifecycle (encode → drop the text encoder under `Sequential` → load the
        // control DiT/VAE → denoise/decode → free the heavy bundle) is driven by the shared
        // [`Residency::run`] seam (sc-11124), owning the eval/drop/clear discipline, the stage-boundary
        // cancel checks, and the error-safe cache flush — identically to the base `z_image_turbo`. The
        // control variant is guidance-distilled (no CFG / negative prompt), so the encode phase is a
        // single cond `cap`.
        let images = self.residency.run(
            &req.cancel,
            // No PiD overlay on the control path (sc-7846 is base-turbo-only); the heavy loader ignores
            // this flag, so `false` avoids loading a student that would never be used.
            false,
            on_progress,
            // ── Phase A: prompt → cap_feats. The fork's control path is **mixed precision**, NOT pure
            // bf16: it feeds the latents (`x`) and `cap_feats` as bf16 but `control_context` as **f32**
            // (sc-2720, verified against the fork). The f32 control branch then promotes the bf16
            // image/caption stream to f32 when its hints are added, and `latents += dt·velocity` makes
            // the latents f32 after step 0 — so most of the loop runs f32. We match that exactly: bf16
            // cap (txt2img) + f32 control_context below. (img2img keeps f32 cap, mirroring the base
            // img2img; the DiT promotes per-op either way.)
            |text_encoder: &TextEncoder| {
                let cap =
                    pipeline::encode_prompt(&self.tokenizer, text_encoder, &req.prompt, MODEL_ID)?;
                if is_img2img {
                    Ok(cap)
                } else {
                    // PARITY-BF16 (sc-2609): round the text embeddings to bf16 to match the fork's cap.
                    Ok(cap.as_dtype(Dtype::Bfloat16)?)
                }
            },
            // Materialize the post-cast `cap` while the encoder is still alive (Sequential only) — MLX
            // is lazy, so an un-evaluated `cap` keeps the encoder referenced through the graph and the
            // drop would free nothing.
            |cap| Ok(mlx_rs::transforms::eval([cap])?),
            // ── Phase B: denoise/decode from the heavy bundle. Runs identically for both residencies.
            |heavy_owned, cap, on_progress| {
                let heavy = heavy_owned.as_ref();

                // Static shift=3.0 schedule (shared with the base turbo, sc-2536) — build once. An
                // unset `req.scheduler` keeps it byte-exact (epic 7114 N1); a curated name re-shapes σ
                // over the shift.
                let native = FlowMatchEuler::for_static_shift(steps, SCHEDULE_SHIFT);
                let scheduler = FlowMatchEuler::from_sigmas(resolve_flow_schedule(
                    req.scheduler.as_deref(),
                    SCHEDULE_SHIFT.ln(),
                    steps,
                    &native.sigmas,
                ))?;

                // The 33ch control context is constant across steps + the batch — build once. It stays
                // **f32** (the fork feeds it f32, which promotes the whole control branch to f32).
                let control_context =
                    encode_control_context(heavy.vae, control_image, req.width, req.height)?;

                // VAE-encode the init image once too: like control_context, the clean img2img latents
                // depend only on the init image + dims, not the per-image seed, so they're constant
                // across the batch (F-034). Only the noise (and its blend) vary per image.
                let clean = if is_img2img {
                    let (image, _) = reference.expect("is_img2img implies a reference");
                    Some(encode_init_latents(
                        heavy.vae, image, req.width, req.height,
                    )?)
                } else {
                    None
                };

                // Per-image batch render shared with the base variant (F-035); the control branch's
                // only difference is the `denoise_control_with_progress` step threading the f32 control
                // context + scale (the mixed-precision dtype flow, sc-2720, is preserved in the closure).
                let sampler_name = req.sampler.as_deref();
                // The Fun-ControlNet variant is outside the PiD decode scope of sc-7846; pass `None` so
                // it keeps the native VAE decode unchanged.
                let images = pipeline::render_batch(
                    heavy.vae,
                    None,
                    &scheduler,
                    clean.as_ref(),
                    start_step,
                    base_seed,
                    req,
                    on_progress,
                    |latents, seed, op| {
                        denoise_control_with_progress(
                            heavy.transformer,
                            &scheduler,
                            sampler_name,
                            seed,
                            latents,
                            &cap,
                            &control_context,
                            control_scale,
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

impl Generator for ZImageTurboControl {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // Shared capability checks (size/count/guidance/negative/accepted conditioning), then the
        // shared control-present check (sc-8241's `ControlBranch::require_control_present`).
        validate_request(self.descriptor.id, &self.descriptor.capabilities, req)?;
        self.require_control_present(req)?;
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }
}

// Link-time registration (epic 3720): the macro emits the `inventory::submit!` and bridges the
// crate's rich `Result` into the registry's backend-neutral `gen_core::Result`. The `impl
// Generator` above stays hand-written because `validate` adds a control-specific check beyond the
// shared `validate_request`, so it is not the plain delegation `impl_generator!` expresses.
mlx_gen::register_generators! { descriptor => load ; footprint = crate::model::component_footprint }

#[cfg(test)]
mod tests {
    use super::*;
    // `WeightsSource` + `OffloadPolicy` come in via `super::*` (both used by `load`/its helpers).

    #[test]
    fn descriptor_is_z_image_turbo_control() {
        let d = descriptor();
        assert_eq!(d.id, "z_image_turbo_control");
        assert_eq!(d.family, "z-image");
        assert!(d.capabilities.accepts(ConditioningKind::Control));
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        assert!(!d.capabilities.supports_guidance);
    }

    #[test]
    fn load_rejects_missing_control_weights() {
        // Without `spec.control`, load must fail on the missing control weights (not on the
        // missing snapshot) — proving the control overlay is wired as a hard requirement.
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("Fun-Controlnet-Union"), "got: {err}");
    }

    #[test]
    fn load_rejects_single_file_base() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/z.safetensors".into()))
            .with_control(WeightsSource::File("/tmp/control.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("base snapshot directory"), "got: {err}");
    }

    #[test]
    fn load_honors_sequential_offload_policy() {
        // F-172 (sc-11124): before the fix the control variant ignored `offload_policy` and always
        // went `Resident`. Now `load` routes through the shared `load_control_residency` seam under
        // either policy — proven weight-free by the up-front single-file base rejection running on the
        // `Sequential` arm too, exactly as `Resident` rejects it.
        for policy in [OffloadPolicy::Resident, OffloadPolicy::Sequential] {
            let spec = LoadSpec::new(WeightsSource::File("/tmp/z.safetensors".into()))
                .with_control(WeightsSource::File("/tmp/control.safetensors".into()))
                .with_offload_policy(policy);
            let err = load(&spec).err().expect("expected an error").to_string();
            assert!(
                err.contains("base snapshot directory"),
                "policy {policy:?} must reach the shared base-dir validation, got: {err}"
            );
        }
    }

    // ── F-180 (sc-11126): the MEANINGFUL control-variant test the smoke test above cannot be. The
    // `load_honors_sequential_offload_policy` case only proves BOTH policies reach the same up-front
    // base-dir guard — a dispatch that ignored `offload_policy` would pass it too. This drives the
    // dispatch itself (`build_control_residency`) past that guard with a *valid-looking* base dir and
    // control (non-existent, so no weights load) and asserts the deferral discriminator:
    //   * `Sequential` captures the two per-phase loaders, touches NO weights → `Ok` + `is_sequential`.
    //   * `Resident` eager-loads the text encoder from the missing base dir → `Err`.
    // A `Sequential → Resident` regression (the F-172 bug this seam prevents) would eager-load under
    // the Sequential request and fail the first assertion. Covers the turbo control variant directly;
    // the base control variant (`model_base_control`) shares this exact `build_control_residency`.
    fn missing_control_spec(policy: OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(
            "/nonexistent/z-image-control-base".into(),
        ))
        .with_control(WeightsSource::File(
            "/nonexistent/z-image-control-overlay.safetensors".into(),
        ))
        .with_offload_policy(policy)
    }

    #[test]
    fn build_control_residency_sequential_defers_all_component_loads() {
        let res = build_control_residency(
            &missing_control_spec(OffloadPolicy::Sequential),
            MODEL_ID,
            PRECISION_MSG,
        )
        .expect("Sequential must defer loads and not touch the (missing) base/control weights");
        assert!(
            res.is_sequential(),
            "Sequential policy must build a Sequential (deferred) control residency"
        );
    }

    #[test]
    fn build_control_residency_resident_eager_loads_and_fails() {
        let err = build_control_residency(
            &missing_control_spec(OffloadPolicy::Resident),
            MODEL_ID,
            PRECISION_MSG,
        )
        .err()
        .expect("Resident must eager-load and fail on the missing base snapshot");
        let msg = err.to_string();
        assert!(
            !msg.contains("single .safetensors file") && !msg.contains("precision override"),
            "expected an eager-load failure, not the up-front guard: {msg}"
        );
    }
}
