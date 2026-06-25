//! `Boogu` — the [`mlx_gen::Generator`] implementation for Boogu-Image-0.1, plus its
//! [`descriptor`]/[`load`] entry points and the `inventory` registration that wires the three
//! variants into `mlx_gen`'s registry under ids `"boogu_image"` (Base, true-CFG T2I),
//! `"boogu_image_turbo"` (DMD few-step, CFG-free), and `"boogu_image_edit"` (instruction
//! image-edit). Linking this crate is all the worker needs to resolve a model by id.
//!
//! All three variants share one architecture/loader ([`crate::pipeline::BooguPipeline`]); they
//! differ only in which snapshot they load (Base / Turbo / Edit checkpoint) and which sampler
//! [`Boogu::generate`] runs. `spec.quantize` (Q4/Q8) quantizes the dense base in place after the
//! load — a **no-op** when the snapshot is already a packed Q8/Q4 turnkey (the turnkey's default),
//! so pointing at a pre-quantized snapshot skips the dense transient. A precision override and LoRA
//! adapters are rejected rather than silently ignored.

use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, default_seed, Capabilities, Conditioning,
    ConditioningKind, Error, GenerationOutput, GenerationRequest, Generator, Image, LoadSpec,
    Modality, ModelDescriptor, Precision, Progress, Quant, Result, WeightsSource,
};

use crate::pipeline::{BooguPipeline, EditOptions, GenerateOptions, TurboOptions};

/// Registry id for the Base text-to-image variant (true-CFG). Matches the SceneWorks worker's
/// `payload.model`.
pub const BOOGU_IMAGE_ID: &str = "boogu_image";
/// Registry id for the Turbo variant (DMD few-step, CFG-free).
pub const BOOGU_IMAGE_TURBO_ID: &str = "boogu_image_turbo";
/// Registry id for the instruction image-edit variant.
pub const BOOGU_IMAGE_EDIT_ID: &str = "boogu_image_edit";

/// Max images per request (the image-model standard, shared with the other MLX families).
const MAX_COUNT: u32 = 8;
/// Resolution bounds (W/H), both multiples of 16. The catalog/worker gate the actual UI options
/// tighter; this is the engine validation ceiling.
const RES_MIN: u32 = 256;
const RES_MAX: u32 = 2048;
/// Patch(2)·ae_scale(8) = 16 — `patchify` requires dims divisible by this.
const RES_MULTIPLE: u32 = 16;

/// Max reference images the Edit checkpoint supports — the DiT's `image_index_embedding` carries 5
/// per-image index slots (`[5, hidden]`, OmniGen2 lineage), so `N ∈ [1, 5]` references can be packed.
const MAX_EDIT_REFS: usize = 5;

/// Base/Edit default steps + guidance (the reference `__call__`: 50-step true-CFG, guidance 4.0).
const DEFAULT_STEPS: u32 = 50;
const DEFAULT_GUIDANCE: f32 = 4.0;
/// Turbo default steps (DMD student few-step) + the lowest sigma in the DMD schedule.
const DEFAULT_TURBO_STEPS: u32 = 4;
const DEFAULT_TURBO_SIGMA: f32 = 0.001;

/// Boogu Base's identity + capabilities — constructible without loading weights (registry
/// introspection / capability advertisement). True-CFG text-to-image: `guidance` is offered, the
/// CFG-negative is the model's own fixed empty/drop instruction (not a user negative prompt), and
/// there is no img2img/control conditioning on the Base checkpoint.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: BOOGU_IMAGE_ID,
        family: "boogu",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            // The CFG-negative is a fixed empty/drop instruction, not a user negative prompt.
            supports_negative_prompt: false,
            supports_guidance: true,
            supports_true_cfg: false,
            // Base/Turbo are text-to-image only; the instruction-edit reference path is a capability
            // of the Edit checkpoint (`descriptor_edit`).
            conditioning: Vec::new(),
            supports_lora: false,
            supports_lokr: false,
            // Base/Edit are rectified-flow Euler over a static-shift (`mu = 1.15`) schedule, routed
            // through the unified curated-sampler framework (epic 7114). Turbo overrides these to empty
            // (its DMD distillation sampler is not an ODE — see `descriptor_turbo`).
            samplers: curated_sampler_names(),
            schedulers: curated_scheduler_names(),
            min_size: RES_MIN,
            max_size: RES_MAX,
            max_count: MAX_COUNT,
            mac_only: true,
            // The turnkey ships pre-packed Q8 (default) + bf16; load-time quantize (Q4/Q8) over the
            // dense bf16 build is a no-op on an already-packed snapshot. The DiT + Qwen3-VL text
            // tower are quantized; the FLUX.1 VAE + (edit-only) vision tower stay dense.
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// The curated samplers the Turbo DMD student stays coherent under (real-weight survey, sc-7491). The
/// student was distilled against a **stochastic** (re-noised) trajectory — predict the clean estimate,
/// then renoise to the next level with fresh noise — so the curated *stochastic* solvers match its
/// training regime and render at native quality. `lcm` is the closest match (it IS the consistency
/// predict→renoise loop, like ComfyUI's `lcm`/`sgm_uniform` combo), once `lcm` re-noises through the
/// FLOW `noise_scaling` convex blend rather than the VE additive form (the gen-core sc-7491 fix). The
/// deterministic ODE solvers (`euler`/`ddim`/`heun`/`dpmpp_2m`/`uni_pc`) feed the few-step student
/// out-of-regime latents (background artifacts), so they stay off the menu; the native DMD loop
/// (`req.sampler == None`) stays the byte-exact default.
const TURBO_SAMPLERS: &[&str] = &["lcm", "euler_ancestral", "dpmpp_sde"];

/// Boogu **Turbo** identity + capabilities. Same surface as [`descriptor`] except it is **CFG-free**
/// — the DMD student distilled the guided velocity into the weights, so `guidance` is not offered
/// (no unconditional branch). Few-step ([`DEFAULT_TURBO_STEPS`]).
pub fn descriptor_turbo() -> ModelDescriptor {
    let mut d = descriptor();
    d.id = BOOGU_IMAGE_TURBO_ID;
    d.capabilities.supports_guidance = false;
    // The Turbo student is a DMD distillation sampler (predict clean estimate → flow-renoise with fresh
    // noise). Its native loop (`generate_turbo`, `req.sampler == None`) stays the byte-exact default; a
    // real-weight survey (sc-7491) showed the curated *stochastic* solvers ([`TURBO_SAMPLERS`]) — `lcm`
    // most of all — match its re-noised regime and render at native quality over the curated σ schedules
    // (the ComfyUI `lcm`/`sgm_uniform` combo), so the sampler AND scheduler axes are both selectable. The
    // deterministic ODE solvers degrade on the few-step student (out-of-regime) and are not advertised.
    d.capabilities.samplers = TURBO_SAMPLERS.to_vec();
    d.capabilities.schedulers = curated_scheduler_names();
    d
}

/// Boogu **Edit** identity + capabilities. Same true-CFG surface as [`descriptor`] plus instruction-edit
/// source images: one [`ConditioningKind::Reference`] or up to [`MAX_EDIT_REFS`] via
/// [`ConditioningKind::MultiReference`]. Each source image is read by the Qwen3-VL vision tower
/// (semantic edit) and VAE-encoded into the DiT's spatial reference sequence (`image_index_embedding`
/// has 5 per-image slots, so 2–5 references compose into one edit, e.g. subject-from-A in scene-from-B).
pub fn descriptor_edit() -> ModelDescriptor {
    let mut d = descriptor();
    d.id = BOOGU_IMAGE_EDIT_ID;
    d.capabilities.conditioning = vec![
        ConditioningKind::Reference,
        ConditioningKind::MultiReference,
    ];
    d
}

/// A loaded Boogu generator: the assembled pipeline plus the cached descriptor (which selects the
/// sampler path in [`Boogu::generate`]).
pub struct Boogu {
    descriptor: ModelDescriptor,
    pipeline: BooguPipeline,
}

/// Load a Boogu generator from a [`LoadSpec`] under the given `descriptor`. `spec.weights` must be a
/// [`WeightsSource::Dir`] pointing at a Boogu snapshot (`mllm/ transformer/ vae/`). The loader
/// auto-detects a packed Q8/Q4 turnkey (the shipped default) vs a dense bf16 snapshot; `spec.quantize`
/// then quantizes the dense base in place (a no-op on an already-packed snapshot). A precision
/// override and LoRA/LoKr adapters are rejected rather than silently ignored.
fn load_with(spec: &LoadSpec, descriptor: ModelDescriptor) -> Result<Box<dyn Generator>> {
    let id = descriptor.id;
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{id}: only the default dense precision is wired (drop the precision override)"
        )));
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(format!(
            "{id}: LoRA/LoKr adapters are not supported"
        )));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(Error::Msg(format!(
                "{id} expects a snapshot directory (mllm/ transformer/ vae/), not a single \
                 .safetensors file"
            )))
        }
    };
    let mut pipeline = BooguPipeline::from_snapshot(root)?;
    // No-op when the snapshot is already packed (the turnkey default); quantizes the dense bf16
    // build otherwise (`AdaptableLinear::quantize` skips already-quantized bases).
    if let Some(q) = spec.quantize {
        pipeline.quantize(q.bits())?;
    }
    Ok(Box::new(Boogu {
        descriptor,
        pipeline,
    }))
}

/// Construct a Boogu **Base** generator (true-CFG text-to-image) from a [`LoadSpec`].
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_with(spec, descriptor())
}

/// Construct a Boogu **Turbo** generator (DMD few-step, CFG-free) from a [`LoadSpec`].
pub fn load_turbo(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_with(spec, descriptor_turbo())
}

/// Construct a Boogu **Edit** generator (instruction image-edit) from a [`LoadSpec`].
pub fn load_edit(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_with(spec, descriptor_edit())
}

mlx_gen::impl_generator!(Boogu {
    validate: |s, req| validate_request(&s.descriptor, req),
    generate: generate_impl,
});

impl Boogu {
    /// The rich-`Result` body behind [`Generator::generate`] — kept on the crate's own
    /// [`mlx_gen::Error`] so `?` lifts `mlx_rs` device exceptions transparently; the trait wrapper
    /// bridges the tail into [`gen_core::Error`].
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        validate_request(&self.descriptor, req)?;

        let base_seed = req.seed.unwrap_or_else(default_seed);
        let mut images = Vec::with_capacity(req.count as usize);

        if self.descriptor.id == BOOGU_IMAGE_TURBO_ID {
            let steps = req.steps.unwrap_or(DEFAULT_TURBO_STEPS) as usize;
            for n in 0..req.count {
                let opts = TurboOptions {
                    height: req.height,
                    width: req.width,
                    steps,
                    seed: base_seed.wrapping_add(n as u64),
                    conditioning_sigma: DEFAULT_TURBO_SIGMA,
                    sampler: req.sampler.clone(),
                    scheduler: req.scheduler.clone(),
                };
                let img = self.pipeline.generate_turbo_with_progress(
                    &req.prompt,
                    &opts,
                    &req.cancel,
                    on_progress,
                )?;
                images.push(img);
            }
        } else if self.descriptor.id == BOOGU_IMAGE_EDIT_ID {
            // Source images arrive as `Reference` / `MultiReference` (1..=MAX_EDIT_REFS); the prompt is
            // the edit instruction. Clone once into an owned slice for the pipeline (cheap next to the
            // multi-step DiT denoise).
            let references: Vec<Image> =
                resolve_edit_references(req)?.into_iter().cloned().collect();
            let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
            let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
            for n in 0..req.count {
                let opts = EditOptions {
                    height: req.height,
                    width: req.width,
                    steps,
                    text_guidance_scale: guidance,
                    seed: base_seed.wrapping_add(n as u64),
                    condition_on_image: true,
                    use_input_images_4_neg_instruct: false,
                    sampler: req.sampler.clone(),
                    scheduler: req.scheduler.clone(),
                };
                let img = self.pipeline.generate_edit_multi_with_progress(
                    &references,
                    &req.prompt,
                    &opts,
                    &req.cancel,
                    on_progress,
                )?;
                images.push(img);
            }
        } else {
            let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
            let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
            for n in 0..req.count {
                let opts = GenerateOptions {
                    height: req.height,
                    width: req.width,
                    steps,
                    text_guidance_scale: guidance,
                    seed: base_seed.wrapping_add(n as u64),
                    sampler: req.sampler.clone(),
                    scheduler: req.scheduler.clone(),
                };
                let img = self.pipeline.generate_with_progress(
                    &req.prompt,
                    &opts,
                    &req.cancel,
                    on_progress,
                )?;
                images.push(img);
            }
        }

        Ok(GenerationOutput::Images(images))
    }
}

/// The instruction-edit source images, in order: any mix of [`Conditioning::Reference`] (single) and
/// [`Conditioning::MultiReference`] (a list), flattened. The Edit path needs at least one and at most
/// [`MAX_EDIT_REFS`] (the DiT's `image_index_embedding` slot count); none, or more than the cap, is an
/// error.
fn resolve_edit_references(req: &GenerationRequest) -> Result<Vec<&Image>> {
    let mut refs: Vec<&Image> = Vec::new();
    for c in &req.conditioning {
        match c {
            Conditioning::Reference { image, .. } => refs.push(image),
            Conditioning::MultiReference { images } => refs.extend(images.iter()),
            _ => {}
        }
    }
    if refs.is_empty() {
        return Err(Error::Msg(
            "boogu_image_edit: an instruction edit requires at least one source reference image"
                .into(),
        ));
    }
    if refs.len() > MAX_EDIT_REFS {
        return Err(Error::Msg(format!(
            "boogu_image_edit: at most {MAX_EDIT_REFS} reference images are supported (got {})",
            refs.len()
        )));
    }
    Ok(refs)
}

/// Capability-driven request validation, factored out so it can be unit-tested without loaded
/// weights. Layers Boogu's model-specific constraints (non-empty prompt, size multiple-of-16, steps
/// ≥ 1, the Edit variant requires a reference) on top of the shared [`Capabilities::validate_request`]
/// floor (count/size range, negative/guidance/true_cfg flags, conditioning kinds).
pub(crate) fn validate_request(desc: &ModelDescriptor, req: &GenerationRequest) -> Result<()> {
    let id = desc.id;
    if req.prompt.is_empty() {
        return Err(Error::Msg(format!("{id}: prompt must not be empty")));
    }
    desc.capabilities.validate_request(id, req)?;
    if req.steps == Some(0) {
        return Err(Error::Msg(format!("{id}: steps must be >= 1")));
    }
    if !req.width.is_multiple_of(RES_MULTIPLE) || !req.height.is_multiple_of(RES_MULTIPLE) {
        return Err(Error::Msg(format!(
            "{id}: {}x{} must be a multiple of {RES_MULTIPLE}",
            req.width, req.height
        )));
    }
    // The Edit variant needs 1..=MAX_EDIT_REFS source references (Reference and/or MultiReference);
    // the floor already rejects any reference conditioning on Base/Turbo (their surface is empty).
    if id == BOOGU_IMAGE_EDIT_ID {
        let refs: usize = req
            .conditioning
            .iter()
            .map(|c| match c {
                Conditioning::Reference { .. } => 1,
                Conditioning::MultiReference { images } => images.len(),
                _ => 0,
            })
            .sum();
        if refs == 0 {
            return Err(Error::Msg(format!(
                "{id}: instruction edit requires at least one source reference image"
            )));
        }
        if refs > MAX_EDIT_REFS {
            return Err(Error::Msg(format!(
                "{id}: at most {MAX_EDIT_REFS} source reference images are supported (got {refs})"
            )));
        }
    }
    Ok(())
}

// Link-time registration (epic 3720): the macro emits each `inventory::submit!` and bridges the
// crate's rich `Result` into the registry's backend-neutral `gen_core::Result`.
mlx_gen::register_generators! {
    descriptor => load,
    descriptor_turbo => load_turbo,
    descriptor_edit => load_edit,
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core;

    fn req(w: u32, h: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "a red apple on a wooden table".into(),
            width: w,
            height: h,
            ..Default::default()
        }
    }

    fn img(w: u32, h: u32) -> Image {
        Image {
            width: w,
            height: h,
            pixels: vec![0u8; (w * h * 3) as usize],
        }
    }

    #[test]
    fn descriptor_is_boogu_image() {
        let d = descriptor();
        assert_eq!(d.id, "boogu_image");
        assert_eq!(d.family, "boogu");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_negative_prompt);
        // Base is text-to-image only — no conditioning surface.
        assert!(d.capabilities.conditioning.is_empty());
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert!(d.capabilities.mac_only);
    }

    #[test]
    fn descriptor_turbo_is_cfg_free_else_matches_base() {
        let (b, t) = (descriptor(), descriptor_turbo());
        assert_eq!(t.id, "boogu_image_turbo");
        assert_eq!(t.family, b.family);
        assert_eq!(t.modality, b.modality);
        assert!(b.capabilities.supports_guidance);
        assert!(!t.capabilities.supports_guidance);
        assert!(t.capabilities.conditioning.is_empty());
        assert_eq!(
            t.capabilities.supported_quants,
            b.capabilities.supported_quants
        );
    }

    #[test]
    fn descriptor_turbo_advertises_the_stochastic_sampler_subset_and_scheduler_axis() {
        let (b, t) = (descriptor(), descriptor_turbo());
        // Turbo exposes the DMD-compatible stochastic samplers (incl. flow-aware `lcm`), a strict subset
        // of the Base sampler menu, plus the full curated scheduler axis (the ComfyUI lcm/sgm_uniform).
        assert_eq!(
            t.capabilities.samplers,
            vec!["lcm", "euler_ancestral", "dpmpp_sde"]
        );
        assert_eq!(t.capabilities.schedulers, b.capabilities.schedulers);
        assert!(t.capabilities.schedulers.contains(&"sgm_uniform"));
        for s in &t.capabilities.samplers {
            assert!(
                b.capabilities.samplers.contains(s),
                "turbo sampler {s:?} must be a subset of the Base curated menu"
            );
        }
        // The deterministic ODE solvers degrade on the few-step student and are NOT advertised.
        for excluded in ["euler", "ddim", "heun", "dpmpp_2m", "uni_pc"] {
            assert!(!t.capabilities.samplers.contains(&excluded));
        }
    }

    #[test]
    fn turbo_validate_gates_to_the_advertised_sampler_subset() {
        let d = descriptor_turbo();
        // Advertised stochastic samplers are accepted, optionally with a curated scheduler
        // (the ComfyUI lcm/sgm_uniform combo).
        for s in ["lcm", "euler_ancestral", "dpmpp_sde"] {
            let r = GenerationRequest {
                sampler: Some(s.into()),
                scheduler: Some("sgm_uniform".into()),
                ..req(512, 512)
            };
            assert!(
                validate_request(&d, &r).is_ok(),
                "turbo should accept {s}+sgm_uniform"
            );
        }
        // An unadvertised sampler (degraded on the few-step student) is rejected before any work.
        let bad = GenerationRequest {
            sampler: Some("dpmpp_2m".into()),
            ..req(512, 512)
        };
        assert!(validate_request(&d, &bad).is_err());
    }

    #[test]
    fn descriptor_edit_adds_reference() {
        let d = descriptor_edit();
        assert_eq!(d.id, "boogu_image_edit");
        assert!(d.capabilities.supports_guidance);
        // Both single and multi reference are advertised (the DiT carries 5 image-index slots).
        assert!(d
            .capabilities
            .conditioning
            .contains(&ConditioningKind::Reference));
        assert!(d
            .capabilities
            .conditioning
            .contains(&ConditioningKind::MultiReference));
        assert!(!d
            .capabilities
            .conditioning
            .contains(&ConditioningKind::Mask));
    }

    #[test]
    fn validate_accepts_in_surface() {
        assert!(validate_request(&descriptor(), &req(1024, 1024)).is_ok());
        assert!(validate_request(
            &descriptor(),
            &GenerationRequest {
                guidance: Some(4.0),
                ..req(512, 512)
            }
        )
        .is_ok());
    }

    #[test]
    fn validate_rejects_empty_prompt_and_bad_size() {
        assert!(validate_request(&descriptor(), &GenerationRequest::default()).is_err());
        for (w, h) in [(1000, 1000), (257, 256)] {
            let e = validate_request(&descriptor(), &req(w, h))
                .unwrap_err()
                .to_string();
            assert!(e.contains("multiple of 16"), "{w}x{h} got: {e}");
        }
        assert!(validate_request(&descriptor(), &req(128, 128)).is_err()); // below min
        assert!(validate_request(&descriptor(), &req(2064, 256)).is_err()); // above max
    }

    #[test]
    fn validate_rejects_guidance_on_turbo_and_negative_prompt() {
        assert!(validate_request(
            &descriptor_turbo(),
            &GenerationRequest {
                guidance: Some(4.0),
                ..req(512, 512)
            }
        )
        .is_err());
        assert!(validate_request(
            &descriptor(),
            &GenerationRequest {
                negative_prompt: Some("x".into()),
                ..req(512, 512)
            }
        )
        .is_err());
    }

    #[test]
    fn base_rejects_reference_conditioning() {
        // Base has no conditioning surface, so the capability floor rejects a Reference.
        let r = GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: img(512, 512),
                strength: None,
            }],
            ..req(512, 512)
        };
        assert!(validate_request(&descriptor(), &r).is_err());
    }

    #[test]
    fn edit_accepts_one_to_five_references() {
        // No reference → error.
        assert!(validate_request(&descriptor_edit(), &req(512, 512)).is_err());
        assert!(resolve_edit_references(&req(512, 512)).is_err());

        // A request carrying `n` single `Reference` conditionings.
        let with_refs = |n: usize| GenerationRequest {
            conditioning: (0..n)
                .map(|_| Conditioning::Reference {
                    image: img(512, 512),
                    strength: None,
                })
                .collect(),
            ..req(512, 512)
        };

        // 1..=MAX_EDIT_REFS single references → ok (the DiT has 5 image-index slots), flattened in order.
        for n in 1..=MAX_EDIT_REFS {
            assert!(
                validate_request(&descriptor_edit(), &with_refs(n)).is_ok(),
                "{n} refs should validate"
            );
            assert_eq!(resolve_edit_references(&with_refs(n)).unwrap().len(), n);
        }
        // One past the cap → error.
        assert!(validate_request(&descriptor_edit(), &with_refs(MAX_EDIT_REFS + 1)).is_err());
        assert!(resolve_edit_references(&with_refs(MAX_EDIT_REFS + 1)).is_err());

        // A `MultiReference` list is accepted and flattened the same way.
        let multi = GenerationRequest {
            conditioning: vec![Conditioning::MultiReference {
                images: vec![img(512, 512), img(512, 512), img(512, 512)],
            }],
            ..req(512, 512)
        };
        assert!(validate_request(&descriptor_edit(), &multi).is_ok());
        assert_eq!(resolve_edit_references(&multi).unwrap().len(), 3);

        // A `MultiReference` list past the cap → error.
        let multi_over = GenerationRequest {
            conditioning: vec![Conditioning::MultiReference {
                images: (0..=MAX_EDIT_REFS).map(|_| img(512, 512)).collect(),
            }],
            ..req(512, 512)
        };
        assert!(validate_request(&descriptor_edit(), &multi_over).is_err());
    }

    #[test]
    fn load_rejects_single_file_and_adapters() {
        let file = LoadSpec::new(WeightsSource::File("/tmp/x.safetensors".into()));
        let e = load(&file).err().expect("error").to_string();
        assert!(e.contains("snapshot directory"), "got: {e}");
    }

    #[test]
    fn load_accepts_quant_spec() {
        for q in [Quant::Q4, Quant::Q8] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_quant(q);
            let e = load(&spec).err().expect("error").to_string();
            assert!(
                !e.contains("not supported"),
                "quant should be accepted: {e}"
            );
        }
    }

    #[test]
    fn all_three_reachable_via_registry_by_id() {
        for id in [BOOGU_IMAGE_ID, BOOGU_IMAGE_TURBO_ID, BOOGU_IMAGE_EDIT_ID] {
            assert!(
                gen_core::registry::generators().any(|r| (r.descriptor)().id == id),
                "id {id} not registered"
            );
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-boogu".into()));
            let e = gen_core::registry::load(id, &spec)
                .err()
                .expect("missing weights → err")
                .to_string();
            assert!(
                !e.contains("no generator registered"),
                "id {id} not resolved: {e}"
            );
        }
    }
}
