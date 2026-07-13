//! `ZImageControl` — the **base** (non-distilled, full-CFG) Z-Image **Fun-Controlnet-Union** variant
//! (sc-8251): VACE-style structural conditioning (pose/canny/depth — input-agnostic) via the base
//! control checkpoint `alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1`, registered as its own
//! `Generator` (`z_image_control`).
//!
//! It is the [`crate::model_control::ZImageTurboControl`] variant re-pointed at the base `z_image`
//! model + the base control repo:
//!
//! * **Base DiT, base schedule, real CFG.** Same `ZImageControlTransformer` (base DiT + control
//!   branch) as the Turbo control variant, but assembled from a base `Tongyi-MAI/Z-Image` snapshot and
//!   driven with the base model's scheduler (`shift=6.0`, default 50 steps) and **classifier-free
//!   guidance** (the base is undistilled — guidance + a negative prompt, unlike the guidance-distilled
//!   Turbo). The control denoise threads the constant control context through **both** the cond and the
//!   uncond forward of the CFG combine — see [`pipeline::denoise_control_cfg_with_progress`].
//! * **Same control branch shape.** The base control checkpoint
//!   `Z-Image-Fun-Controlnet-Union-2.1.safetensors` is **byte-structurally identical** to the Turbo
//!   control checkpoint (verified vs the cached Turbo ckpt: 295 keys, identical `control_all_x_embedder`
//!   / `control_layers` / `control_noise_refiner` prefixes, zero shape/dtype mismatches), so the shared
//!   [`loader::load_control_transformer`] + [`ZImageControlTransformer::from_weights`] loader is reused
//!   unchanged — no remap, no loader adaptation.
//!
//! [`load`] needs the base snapshot (`spec.weights`) **and** the base control checkpoint
//! (`spec.control`); the SceneWorks catalog row + re-pin at the base control repo is a coordinator
//! follow-up. The `z_image`, `z_image_turbo`, and `z_image_turbo_control` variants are untouched.

use mlx_gen::gen_core;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, default_seed, resolve_flow_schedule,
    AcceptedControlKinds, Capabilities, ConditioningKind, ControlBranch, ControlKind,
    FlowMatchEuler, GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality,
    ModelDescriptor, Progress, Quant, Residency, Result,
};

use crate::model::validate_request;
use crate::model_base::{DEFAULT_GUIDANCE, DEFAULT_STEPS, SCHEDULE_SHIFT};
use crate::model_control::{load_control_residency, ZImageControlHeavyOwned};
use crate::pipeline::{
    self, denoise_control_cfg_with_progress, encode_control_context, encode_init_latents,
    init_time_step,
};
use crate::text_encoder::TextEncoder;

/// Registry id for the **base** (non-Turbo) Z-Image Fun-Controlnet-Union variant. Coexists with
/// `z_image`, `z_image_turbo`, and `z_image_turbo_control` — distinct id, separate `inventory`
/// registration, no clash.
pub const MODEL_ID: &str = "z_image_control";

/// The base control variant's identity + capabilities. Same undistilled base (real CFG + a negative
/// prompt) as `z_image`, plus `Control` conditioning (the required structural control image) and
/// `Reference` (an optional img2img init — the fork's `generate_image` accepts both).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "z-image",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            supported_quants: &[Quant::Q4, Quant::Q8],
            // Base is undistilled → full classifier-free guidance + negative prompting (mirrors the
            // base `z_image` descriptor), unlike the guidance-distilled Turbo control variant.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: true,
            // Control (required) + an optional img2img Reference init.
            conditioning: vec![ConditioningKind::Control, ConditioningKind::Reference],
            supports_lora: true,
            supports_lokr: true,
            // Curated unified-framework integrator menu (epic 7114 P3), as the base variant.
            samplers: curated_sampler_names(),
            // Curated scheduler menu (epic 7114), as the base variant — static-shift default.
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

/// A loaded base control generator: the cached descriptor, the (tiny, always-warm) tokenizer, and the
/// component-residency strategy (base text encoder + control transformer + VAE), driven through the
/// shared [`Residency`] seam so the base control variant honors [`LoadSpec::offload_policy`] family-wide
/// (sc-11124, F-172). Same component set + seam as [`crate::model_control::ZImageTurboControl`]; only the
/// generate-time schedule + CFG differ.
pub struct ZImageControl {
    descriptor: ModelDescriptor,
    tokenizer: TextTokenizer,
    /// Component-residency strategy (sc-11124), selected from [`LoadSpec::offload_policy`] via the
    /// shared [`load_control_residency`] builder (reused from the Turbo control variant).
    residency: Residency<TextEncoder, ZImageControlHeavyOwned>,
}

/// The per-id precision-override rejection message for the base control variant, threaded into the
/// shared [`load_control_residency`] builder.
const PRECISION_MSG: &str = "z_image_control: only dense bf16 is wired (the text encoder runs f32 \
     internally); drop the precision override";

/// Construct a [`ZImageControl`] from a [`LoadSpec`].
///
/// `spec.weights` must be a [`WeightsSource::Dir`] base `Tongyi-MAI/Z-Image` snapshot, and
/// `spec.control` (required) the base Fun-Controlnet-Union checkpoint
/// (`alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1` — a single `.safetensors` `File`, or a `Dir` of
/// them). Weights load dense (bf16); `spec.quantize` (Q4/Q8) then quantizes the whole transformer
/// (base + control, group_size 64) plus the text encoder + VAE — the fork's whole-model quant, with the
/// control patch embedder left dense (its in-features is not a multiple of 64). Byte-identical load path
/// to [`crate::model_control::load`] (the control branch shape is identical — see the module doc) — it
/// shares the same [`load_control_residency`] builder, so `offload_policy` is honored identically
/// (sc-11124, F-172); only the generate-time schedule + CFG differ.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let (tokenizer, residency) = load_control_residency(spec, MODEL_ID, PRECISION_MSG)?;
    Ok(Box::new(ZImageControl {
        descriptor: descriptor(),
        tokenizer,
        residency,
    }))
}

/// The base Fun-Controlnet-Union admits the three structural control signals — pose, canny, and depth
/// — differing only by the preprocessor-produced control image (no mode index, S0). Spelled out as
/// `Only([Pose, Canny, Depth])` so a free-form `ControlKind::Other` is rejected rather than silently
/// coerced into the union path. A free function so the policy is unit-testable without a loaded model.
/// `pub(crate)` so the Turbo variant (same Fun-Controlnet-Union checkpoint) shares it — F-089: the
/// turbo variant previously fell back to `AcceptedControlKinds::Any`, accepting `Other("scribble")`
/// the base rejects, an inconsistent contract on the same weights.
pub(crate) fn accepted_kinds() -> AcceptedControlKinds {
    AcceptedControlKinds::Only(vec![
        ControlKind::Pose,
        ControlKind::Canny,
        ControlKind::Depth,
    ])
}

/// The Fun-Controlnet-Union is a *union* ControlNet (pose/canny/depth share one VAE-encoded control
/// path — input-agnostic, no mode index, S0). The structural kinds {Pose, Canny, Depth} are all
/// accepted; the shared trait supplies the resolve/validate-present plumbing (sc-8241).
impl ControlBranch for ZImageControl {
    fn model_id(&self) -> &'static str {
        MODEL_ID
    }

    fn accepted_control_kinds(&self) -> AcceptedControlKinds {
        accepted_kinds()
    }

    /// Fun-Union accepts pose/canny/depth; only the catch-all `Other` reaches this rejection, so the
    /// default Qwen "pose control only" wording is replaced with the union family's actual surface.
    fn unsupported_kind_message(&self, kind: &ControlKind) -> String {
        format!("{MODEL_ID}: Fun-Controlnet-Union accepts pose/canny/depth control, got {kind:?}")
    }
}

impl ZImageControl {
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
        // Real CFG: the request's `guidance` is the classifier-free guidance scale; default 4.0 (the
        // base card). 1.0 collapses CFG to a single cond forward (Turbo-control-equivalent cost).
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
        let cfg_on = guidance != 1.0;

        // Required structural control + optional img2img init.
        let (control_image, control_scale) = self.resolve_control(req)?;
        let reference = pipeline::resolve_reference(req, MODEL_ID)?;
        let start_step = match reference {
            Some((_, strength)) => init_time_step(steps, strength),
            None => 0,
        };
        let is_img2img = start_step > 0;

        // The staged residency lifecycle (encode cond+uncond → drop the text encoder under `Sequential`
        // → load the control DiT/VAE → denoise/decode → free the heavy bundle) is driven by the shared
        // [`Residency::run`] seam (sc-11124). Base-control delta vs the Turbo control variant: real CFG
        // (a negative-prompt uncond branch) and no bf16 cap cast.
        let images = self.residency.run(
            &req.cancel,
            // No PiD overlay on the control path (sc-7846 is base-turbo-only); the heavy loader ignores
            // this flag.
            false,
            on_progress,
            // ── Phase A: prompt → (cap, neg_cap). The base is undistilled and runs real CFG; like the
            // base `z_image` (and unlike the Turbo bf16 seed-parity golden), keep the conditioning at
            // the text encoder's native precision and let the DiT promote per-op against the bf16
            // weights. The control branch's f32 mixed-precision flow (sc-2720) is preserved inside the
            // denoise closure regardless.
            |text_encoder: &TextEncoder| {
                let cap =
                    pipeline::encode_prompt(&self.tokenizer, text_encoder, &req.prompt, MODEL_ID)?;
                // Uncond conditioning = the negative prompt (empty string when unset), encoded only
                // when CFG is active. Empty prompt is valid for the negative branch.
                let neg_cap = if cfg_on {
                    let neg = req.negative_prompt.as_deref().unwrap_or("");
                    Some(pipeline::encode_uncond(&self.tokenizer, text_encoder, neg)?)
                } else {
                    None
                };
                Ok((cap, neg_cap))
            },
            // Materialize cap (+neg_cap) while the encoder is still alive (Sequential only).
            |(cap, neg_cap)| {
                match neg_cap {
                    Some(neg) => mlx_rs::transforms::eval([cap, neg])?,
                    None => mlx_rs::transforms::eval([cap])?,
                }
                Ok(())
            },
            // ── Phase B: denoise/decode from the heavy bundle. Runs identically for both residencies.
            |heavy_owned, (cap, neg_cap), on_progress| {
                let heavy = heavy_owned.as_ref();

                // Static shift=6.0 schedule (the base model's scheduler_config.json) — build once. An
                // unset `req.scheduler` keeps it byte-exact (epic 7114 N1); a curated name re-shapes σ
                // over shift=6.0.
                let native = FlowMatchEuler::for_static_shift(steps, SCHEDULE_SHIFT);
                let scheduler = FlowMatchEuler::from_sigmas(resolve_flow_schedule(
                    req.scheduler.as_deref(),
                    SCHEDULE_SHIFT.ln(),
                    steps,
                    &native.sigmas,
                ))?;

                // The 33ch control context is constant across steps + the batch + both CFG branches —
                // build once. It stays **f32** (the fork feeds it f32, promoting the control branch to
                // f32).
                let control_context =
                    encode_control_context(heavy.vae, control_image, req.width, req.height)?;

                // VAE-encode the init image once too (constant across the batch — only the noise
                // varies, F-034).
                let clean = if is_img2img {
                    let (image, _) = reference.expect("is_img2img implies a reference");
                    Some(encode_init_latents(
                        heavy.vae, image, req.width, req.height,
                    )?)
                } else {
                    None
                };

                // Per-image batch render shared with the base/turbo variants (F-035); the control+CFG
                // branch's only difference is the `denoise_control_cfg_with_progress` step threading the
                // f32 control context + scale through both the cond and uncond forward of the CFG
                // combine.
                let sampler_name = req.sampler.as_deref();
                let neg_cap_ref = neg_cap.as_ref();
                // The Fun-ControlNet variant is outside the PiD decode scope (sc-7846); pass `None` so
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
                        denoise_control_cfg_with_progress(
                            heavy.transformer,
                            &scheduler,
                            sampler_name,
                            seed,
                            latents,
                            &cap,
                            neg_cap_ref,
                            guidance,
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

impl Generator for ZImageControl {
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

// Link-time registration (epic 3720): emits the `inventory::submit!` and bridges the crate's rich
// `Result` into the registry's backend-neutral `gen_core::Result`. The `impl Generator` above stays
// hand-written because `validate` adds a control-specific check beyond the shared `validate_request`,
// so it is not the plain delegation `impl_generator!` expresses. A distinct id (`z_image_control`) →
// no clash with the base / turbo / turbo-control submissions in the same crate.
mlx_gen::register_generators! { descriptor => load ; footprint = crate::model::component_footprint }

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::{OffloadPolicy, WeightsSource};

    #[test]
    fn descriptor_is_z_image_control() {
        let d = descriptor();
        assert_eq!(d.id, "z_image_control");
        assert_eq!(d.family, "z-image");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.accepts(ConditioningKind::Control));
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        // Base control = undistilled → real CFG + a negative prompt (unlike the Turbo control variant).
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_true_cfg);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.supports_lora && d.capabilities.supports_lokr);
    }

    #[test]
    fn base_control_differs_from_turbo_control_only_in_cfg() {
        // The two control variants share family/backend/modality/size envelope + the Control+Reference
        // conditioning; the documented delta is CFG support (base undistilled vs guidance-distilled Turbo).
        let base = descriptor();
        let turbo = crate::model_control::descriptor();
        assert_eq!(base.family, turbo.family);
        assert_eq!(base.backend, turbo.backend);
        assert_eq!(base.modality, turbo.modality);
        assert_eq!(base.capabilities.min_size, turbo.capabilities.min_size);
        assert_eq!(base.capabilities.max_size, turbo.capabilities.max_size);
        assert_ne!(base.id, turbo.id);
        // Turbo control is guidance-distilled (CFG off); base control is full-CFG.
        assert!(!turbo.capabilities.supports_guidance);
        assert!(base.capabilities.supports_guidance);
    }

    #[test]
    fn accepts_pose_canny_depth_via_control_branch() {
        // The Fun-Union family is input-agnostic: pose, canny, and depth are all accepted (they differ
        // only by the preprocessor-produced control image). A free-form `Other` kind is rejected. This
        // is exactly the `accepted_control_kinds()` policy the `ControlBranch` impl returns.
        let accepted = accepted_kinds();
        assert!(accepted.accepts(&ControlKind::Pose));
        assert!(accepted.accepts(&ControlKind::Canny));
        assert!(accepted.accepts(&ControlKind::Depth));
        assert!(!accepted.accepts(&ControlKind::Other("scribble".into())));
    }

    #[test]
    fn load_rejects_missing_control_weights() {
        // Without `spec.control`, load must fail on the missing control weights (not the missing
        // snapshot) — proving the control overlay is wired as a hard requirement.
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

    // ── F-180 (sc-11126): weight-free, default-run proof that the Z-Image BASE-CONTROL dispatch
    // HONORS `offload_policy`. Upgraded from the sc-11124 smoke test (which pointed a *single File*
    // base at both arms, so both merely hit the shared up-front single-file rejection — an
    // always-`Resident` impl passed it). This drives the shared `build_control_residency` seam with the
    // base control variant's own `MODEL_ID`/`PRECISION_MSG` past that guard, using a non-existent base
    // dir + control (so no weights load), and asserts the deferral discriminator:
    //   * `Sequential` captures the two per-phase loaders, touches NO weights → `Ok` + `is_sequential`.
    //   * `Resident` eager-loads the text encoder from the missing base dir → `Err`.
    // A `Sequential → Resident` regression (the exact F-172 bug this seam fixed) would eager-load under
    // the Sequential request and fail the first assertion. The real-weight A/B in
    // `tests/sequential_residency_real_weights.rs` is `#[ignore]`d; this runs by default.
    fn missing_control_spec(policy: OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(
            "/nonexistent/z-image-base-control-residency-test-base".into(),
        ))
        .with_control(WeightsSource::File(
            "/nonexistent/z-image-base-control-residency-test-overlay.safetensors".into(),
        ))
        .with_offload_policy(policy)
    }

    #[test]
    fn build_control_residency_sequential_defers_all_component_loads() {
        let res = crate::model_control::build_control_residency(
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
        let err = crate::model_control::build_control_residency(
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
