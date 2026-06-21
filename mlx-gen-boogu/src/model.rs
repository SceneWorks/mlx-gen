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

use mlx_gen::gen_core;
use mlx_gen::{
    default_seed, Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, ModelRegistration,
    Precision, Progress, Quant, Result, WeightsSource,
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
            samplers: Vec::new(),
            schedulers: Vec::new(),
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

/// Boogu **Turbo** identity + capabilities. Same surface as [`descriptor`] except it is **CFG-free**
/// — the DMD student distilled the guided velocity into the weights, so `guidance` is not offered
/// (no unconditional branch). Few-step ([`DEFAULT_TURBO_STEPS`]).
pub fn descriptor_turbo() -> ModelDescriptor {
    let mut d = descriptor();
    d.id = BOOGU_IMAGE_TURBO_ID;
    d.capabilities.supports_guidance = false;
    d
}

/// Boogu **Edit** identity + capabilities. Same true-CFG surface as [`descriptor`] plus a single
/// img2img/instruction-edit source [`ConditioningKind::Reference`]: the source image is read by the
/// Qwen3-VL vision tower (semantic edit) and VAE-encoded into the DiT's spatial reference latent.
pub fn descriptor_edit() -> ModelDescriptor {
    let mut d = descriptor();
    d.id = BOOGU_IMAGE_EDIT_ID;
    d.capabilities.conditioning = vec![ConditioningKind::Reference];
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

impl Generator for Boogu {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        validate_request(&self.descriptor, req).map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }
}

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
            // The source image arrives as the single `Reference`; the prompt is the edit instruction.
            let reference = resolve_edit_reference(req)?;
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
                };
                let img = self.pipeline.generate_edit_with_progress(
                    reference,
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

/// The single img2img/instruction-edit source [`Conditioning::Reference`] image. More than one
/// reference, or none, is an error (the Edit path needs exactly one source).
fn resolve_edit_reference(req: &GenerationRequest) -> Result<&Image> {
    let mut source: Option<&Image> = None;
    for c in &req.conditioning {
        if let Conditioning::Reference { image, .. } = c {
            if source.is_some() {
                return Err(Error::Msg(
                    "boogu_image_edit: only one reference (source) image is supported for edit"
                        .into(),
                ));
            }
            source = Some(image);
        }
    }
    source.ok_or_else(|| {
        Error::Msg("boogu_image_edit: an instruction edit requires a source reference image".into())
    })
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
    // The Edit variant needs exactly one source reference; the floor already rejects a reference on
    // Base/Turbo (their `conditioning` surface is empty).
    if id == BOOGU_IMAGE_EDIT_ID {
        let refs = req
            .conditioning
            .iter()
            .filter(|c| matches!(c, Conditioning::Reference { .. }))
            .count();
        if refs != 1 {
            return Err(Error::Msg(format!(
                "{id}: instruction edit requires exactly one source reference image (got {refs})"
            )));
        }
    }
    Ok(())
}

/// Registry adapter: the link-time registry's `load` slot is typed on the backend-neutral
/// [`gen_core::Result`]; bridge the crate's rich-`Result` loaders into it.
fn load_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load(spec).map_err(Into::into)
}

inventory::submit! {
    ModelRegistration { descriptor, load: load_registered }
}

fn load_turbo_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_turbo(spec).map_err(Into::into)
}

inventory::submit! {
    ModelRegistration { descriptor: descriptor_turbo, load: load_turbo_registered }
}

fn load_edit_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_edit(spec).map_err(Into::into)
}

inventory::submit! {
    ModelRegistration { descriptor: descriptor_edit, load: load_edit_registered }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn descriptor_edit_adds_reference() {
        let d = descriptor_edit();
        assert_eq!(d.id, "boogu_image_edit");
        assert!(d.capabilities.supports_guidance);
        assert!(d
            .capabilities
            .conditioning
            .contains(&ConditioningKind::Reference));
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
    fn edit_requires_exactly_one_reference() {
        // No reference → error.
        assert!(validate_request(&descriptor_edit(), &req(512, 512)).is_err());
        // One reference → ok.
        let one = GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: img(512, 512),
                strength: None,
            }],
            ..req(512, 512)
        };
        assert!(validate_request(&descriptor_edit(), &one).is_ok());
        assert!(resolve_edit_reference(&one).is_ok());
        // Two references → error.
        let two = GenerationRequest {
            conditioning: vec![
                Conditioning::Reference {
                    image: img(512, 512),
                    strength: None,
                },
                Conditioning::Reference {
                    image: img(512, 512),
                    strength: None,
                },
            ],
            ..req(512, 512)
        };
        assert!(validate_request(&descriptor_edit(), &two).is_err());
        assert!(resolve_edit_reference(&two).is_err());
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
