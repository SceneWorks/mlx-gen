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
    curated_sampler_names, curated_scheduler_names, default_seed, require_base_dir,
    require_control, resolve_flow_schedule, AcceptedControlKinds, Capabilities, ConditioningKind,
    ControlBranch, ControlKind, Error, FlowMatchEuler, GenerationOutput, GenerationRequest,
    Generator, LoadSpec, Modality, ModelDescriptor, Precision, Progress, Quant, Result,
};

use crate::control_transformer::ZImageControlTransformer;
use crate::loader;
use crate::model::validate_request;
use crate::model_base::{DEFAULT_GUIDANCE, DEFAULT_STEPS, SCHEDULE_SHIFT};
use crate::pipeline::{
    self, denoise_control_cfg_with_progress, encode_control_context, encode_init_latents,
    init_time_step,
};
use crate::text_encoder::TextEncoder;
use crate::vae::Vae;

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
        },
    }
}

/// A loaded base control generator: base components + the control transformer assembled from the base
/// snapshot and the base Fun-Controlnet-Union overlay.
pub struct ZImageControl {
    descriptor: ModelDescriptor,
    tokenizer: TextTokenizer,
    text_encoder: TextEncoder,
    transformer: ZImageControlTransformer,
    vae: Vae,
}

/// Construct a [`ZImageControl`] from a [`LoadSpec`].
///
/// `spec.weights` must be a [`WeightsSource::Dir`] base `Tongyi-MAI/Z-Image` snapshot, and
/// `spec.control` (required) the base Fun-Controlnet-Union checkpoint
/// (`alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1` — a single `.safetensors` `File`, or a `Dir` of
/// them). Weights load dense (bf16); `spec.quantize` (Q4/Q8) then quantizes the whole transformer
/// (base + control, group_size 64) plus the text encoder + VAE — the fork's whole-model quant, with the
/// control patch embedder left dense (its in-features is not a multiple of 64). Byte-identical load path
/// to [`crate::model_control::load`] (the control branch shape is identical — see the module doc); only
/// the generate-time schedule + CFG differ.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "z_image_control: only dense bf16 is wired (the text encoder runs f32 \
             internally); drop the precision override"
                .into(),
        ));
    }
    // Shared load boilerplate (sc-8241): the base must be a snapshot dir, the control checkpoint is
    // required. The model id + labels keep the messages consistent with the Turbo control variant.
    let root = require_base_dir(spec, MODEL_ID, "a base snapshot directory")?;
    let control = require_control(spec, MODEL_ID, "Fun-Controlnet-Union")?;

    // Base + control applied dense first, THEN quantize together (the fork's ordering): quantizing
    // before the overlay would replace the control Linears with QuantizedLinear that can't accept
    // the raw bf16 control weights.
    let mut transformer = loader::load_control_transformer(root, control)?;
    let mut text_encoder = loader::load_text_encoder(root)?;
    let mut vae = loader::load_vae(root)?;
    if let Some(q) = spec.quantize {
        let bits = q.bits();
        transformer.quantize(bits)?;
        text_encoder.quantize(bits)?;
        vae.quantize(bits)?;
    }
    // LoRA/LoKr (sc-2602): install onto the composed base DiT (the control branch is not an adapter
    // target). Same load-time, post-quantize, residual-over-base path as the base/turbo variants. No-op
    // when `spec.adapters` is empty.
    if !spec.adapters.is_empty() {
        crate::adapters::apply_z_image_adapters(&mut transformer, &spec.adapters)?;
    }
    Ok(Box::new(ZImageControl {
        descriptor: descriptor(),
        tokenizer: loader::load_tokenizer(root)?,
        text_encoder,
        transformer,
        vae,
    }))
}

/// The base Fun-Controlnet-Union admits the three structural control signals — pose, canny, and depth
/// — differing only by the preprocessor-produced control image (no mode index, S0). Spelled out as
/// `Only([Pose, Canny, Depth])` so a free-form `ControlKind::Other` is rejected rather than silently
/// coerced into the union path. A free function so the policy is unit-testable without a loaded model.
fn accepted_kinds() -> AcceptedControlKinds {
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

        // Prompt → cap_feats. The base is undistilled and runs real CFG; like the base `z_image` (and
        // unlike the Turbo bf16 seed-parity golden), keep the conditioning at the text encoder's native
        // precision and let the DiT promote per-op against the bf16 weights. The control branch's f32
        // mixed-precision flow (sc-2720) is preserved inside the denoise closure regardless.
        let cap =
            pipeline::encode_prompt(&self.tokenizer, &self.text_encoder, &req.prompt, MODEL_ID)?;
        // Uncond conditioning = the negative prompt (empty string when unset), encoded only when CFG
        // is active. Empty prompt is valid for the negative branch (the unconditional embedding).
        let neg_cap = if cfg_on {
            let neg = req.negative_prompt.as_deref().unwrap_or("");
            Some(pipeline::encode_uncond(
                &self.tokenizer,
                &self.text_encoder,
                neg,
            )?)
        } else {
            None
        };

        // Static shift=6.0 schedule (the base model's scheduler_config.json) — build once. An unset
        // `req.scheduler` keeps it byte-exact (epic 7114 N1); a curated name re-shapes σ over shift=6.0.
        let native = FlowMatchEuler::for_static_shift(steps, SCHEDULE_SHIFT);
        let scheduler = FlowMatchEuler::from_sigmas(resolve_flow_schedule(
            req.scheduler.as_deref(),
            SCHEDULE_SHIFT.ln(),
            steps,
            &native.sigmas,
        ));

        // The 33ch control context is constant across steps + the batch + both CFG branches — build
        // once. It stays **f32** (the fork feeds it f32, which promotes the whole control branch to f32).
        let control_context =
            encode_control_context(&self.vae, control_image, req.width, req.height)?;

        // VAE-encode the init image once too (constant across the batch — only the noise varies, F-034).
        let clean = if is_img2img {
            let (image, _) = reference.expect("is_img2img implies a reference");
            Some(encode_init_latents(
                &self.vae, image, req.width, req.height,
            )?)
        } else {
            None
        };

        // Per-image batch render shared with the base/turbo variants (F-035); the control+CFG branch's
        // only difference is the `denoise_control_cfg_with_progress` step threading the f32 control
        // context + scale through both the cond and uncond forward of the CFG combine.
        let sampler_name = req.sampler.as_deref();
        let neg_cap_ref = neg_cap.as_ref();
        // The Fun-ControlNet variant is outside the PiD decode scope (sc-7846); pass `None` so it keeps
        // the native VAE decode unchanged (epic 7840 is a separate follow-on for the control path).
        let images = pipeline::render_batch(
            &self.vae,
            None,
            &scheduler,
            clean.as_ref(),
            start_step,
            base_seed,
            req,
            on_progress,
            |latents, seed, op| {
                denoise_control_cfg_with_progress(
                    &self.transformer,
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
    }
}

impl Generator for ZImageControl {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // Shared capability checks (size/count/guidance/negative/accepted conditioning), then the
        // shared control-present check (sc-8241's `ControlBranch::require_control_present`).
        validate_request(&self.descriptor.capabilities, req)?;
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
mlx_gen::register_generators! { descriptor => load }

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::WeightsSource;

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
}
