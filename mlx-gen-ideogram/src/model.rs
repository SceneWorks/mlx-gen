//! `Ideogram4` — the [`mlx_gen::Generator`] implementation for Ideogram 4.0, plus its
//! [`descriptor`]/[`load`] entry points and the `inventory` registration that wires it into
//! `mlx_gen`'s registry under id `"ideogram_4"` (sc-5988). Linking this crate is all the worker
//! needs to resolve the model by id.
//!
//! [`load`] assembles the pipeline (2 DiTs + Qwen3-VL TE + VAE + tokenizer) from a converted
//! snapshot directory ([`crate::pipeline::Ideogram4Pipeline`]); [`Ideogram4::generate`] runs the
//! full prompt→image flow per requested image — tokenize the (JSON-caption) prompt natively,
//! asymmetric-CFG flow-match denoise, VAE decode → RGB8 — honoring `req.cancel` and streaming
//! `Progress`. Quantization (Q4/Q8) is a follow-on slice (sc-5989) and is rejected here, not
//! silently ignored.

use mlx_gen::array::host_i32;
use mlx_gen::gen_core;
use mlx_gen::{
    default_seed, Capabilities, Error, GenerationOutput, GenerationRequest, Generator, Image,
    LoadSpec, Modality, ModelDescriptor, ModelRegistration, Precision, Progress, Result,
    WeightsSource,
};
use mlx_rs::{Array, Dtype};

use crate::config::{
    DEFAULT_GUIDANCE, DEFAULT_MU, DEFAULT_STEPS, IDEOGRAM_4_ID, RES_MAX, RES_MIN, RES_MULTIPLE,
};
use crate::pipeline::Ideogram4Pipeline;

/// Registry id (matches the SceneWorks worker's `payload.model`).
pub const MODEL_ID: &str = IDEOGRAM_4_ID;

/// Max images per request (the image-model standard, shared with the other MLX families).
const MAX_COUNT: u32 = 8;
/// Max aspect ratio (long:short) — the reference supports up to 6:1.
const MAX_ASPECT: u32 = 6;

/// Ideogram 4's identity + capabilities — constructible without loading weights (registry
/// introspection / capability advertisement).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "ideogram",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            // Asymmetric CFG runs a separate *unconditional* DiT — the "negative" is a fixed
            // trained model, not a user negative prompt — so `guidance` is offered but a negative
            // prompt is not.
            supports_negative_prompt: false,
            supports_guidance: true,
            supports_true_cfg: false,
            // Pure text-to-image (the prompt is the model's native JSON caption); no img2img /
            // control / reference conditioning in this engine.
            conditioning: Vec::new(),
            supports_lora: false,
            supports_lokr: false,
            samplers: Vec::new(),
            schedulers: Vec::new(),
            min_size: RES_MIN,
            max_size: RES_MAX,
            max_count: MAX_COUNT,
            mac_only: true,
            // Q4/Q8 is not yet wired (sc-5989) — advertise none so the worker does not offer a
            // quant `load` would reject.
            supported_quants: &[],
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// A loaded Ideogram 4 generator: the assembled pipeline plus the cached descriptor.
pub struct Ideogram4 {
    descriptor: ModelDescriptor,
    pipeline: Ideogram4Pipeline,
}

/// Construct an [`Ideogram4`] from a [`LoadSpec`]. `spec.weights` must be a [`WeightsSource::Dir`]
/// pointing at a converted snapshot (`transformer/ unconditional_transformer/ text_encoder/ vae/
/// tokenizer/`). Dense bf16 only: a precision override, on-the-fly quantization (sc-5989), and
/// LoRA/LoKr adapters are not wired and are rejected rather than silently ignored.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "ideogram_4: only dense bf16 is wired (drop the precision override)".into(),
        ));
    }
    if spec.quantize.is_some() {
        return Err(Error::Msg(
            "ideogram_4: Q4/Q8 quantization is not yet wired (sc-5989)".into(),
        ));
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(
            "ideogram_4: LoRA/LoKr adapters are not supported".into(),
        ));
    }
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p,
            WeightsSource::File(_) => return Err(Error::Msg(
                "ideogram_4 expects a snapshot directory (transformer/ unconditional_transformer/ \
                 text_encoder/ vae/ tokenizer/), not a single .safetensors file"
                    .into(),
            )),
        };
    Ok(Box::new(Ideogram4 {
        descriptor: descriptor(),
        pipeline: Ideogram4Pipeline::load(root)?,
    }))
}

impl Generator for Ideogram4 {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        validate_request(&self.descriptor.capabilities, req).map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }
}

impl Ideogram4 {
    /// The rich-`Result` body behind [`Generator::generate`] — kept on the crate's own
    /// [`mlx_gen::Error`] so `?` lifts `mlx_rs` device exceptions transparently; the trait wrapper
    /// bridges the tail into [`gen_core::Error`] (epic 3720).
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        validate_request(&self.descriptor.capabilities, req)?;

        let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
        let base_seed = req.seed.unwrap_or_else(default_seed);

        // Tokenize once — the JSON caption is identical across the count loop; only the seed varies.
        let ids = self.pipeline.tokenize(&req.prompt)?;

        let mut images = Vec::with_capacity(req.count as usize);
        for n in 0..req.count {
            let seed = base_seed.wrapping_add(n as u64);
            let arr = self.pipeline.generate_with_progress(
                &ids,
                req.height,
                req.width,
                steps,
                guidance,
                DEFAULT_MU,
                seed,
                &req.cancel,
                on_progress,
            )?;
            images.push(array_to_image(&arr)?);
        }
        Ok(GenerationOutput::Images(images))
    }
}

/// Capability-driven request validation, factored out so it can be unit-tested without loaded
/// weights. Layers Ideogram's model-specific constraints (non-empty prompt, size multiple-of-16,
/// aspect ≤ 6:1, steps ≥ 1) on top of the shared [`Capabilities::validate_request`] floor
/// (count/size range, negative/guidance/true_cfg flags, conditioning kinds).
pub(crate) fn validate_request(caps: &Capabilities, req: &GenerationRequest) -> Result<()> {
    if req.prompt.is_empty() {
        return Err(Error::Msg(
            "ideogram_4: prompt must not be empty (Ideogram 4 expects a JSON caption)".into(),
        ));
    }
    // `?` converts the shared floor's `gen_core::Error` into the crate's `Error` (From impl).
    caps.validate_request(MODEL_ID, req)?;
    if req.steps == Some(0) {
        return Err(Error::Msg("ideogram_4: steps must be >= 1".into()));
    }
    // The pipeline needs dims divisible by patch(2) × ae_scale(8) = 16, or `patchify`'s reshape
    // blows up deep in MLX — reject at the boundary.
    if !req.width.is_multiple_of(RES_MULTIPLE) || !req.height.is_multiple_of(RES_MULTIPLE) {
        return Err(Error::Msg(format!(
            "ideogram_4: {}x{} must be a multiple of {RES_MULTIPLE}",
            req.width, req.height
        )));
    }
    let (long, short) = (req.width.max(req.height), req.width.min(req.height));
    if long > short * MAX_ASPECT {
        return Err(Error::Msg(format!(
            "ideogram_4: aspect ratio of {}x{} exceeds the supported {MAX_ASPECT}:1",
            req.width, req.height
        )));
    }
    Ok(())
}

/// Host-extract the pipeline's `[H, W, 3]` u8 RGB array into an [`Image`].
fn array_to_image(img: &Array) -> Result<Image> {
    let sh = img.shape();
    let (h, w) = (sh[0] as u32, sh[1] as u32);
    let px = host_i32(&img.as_dtype(Dtype::Int32)?)?;
    Ok(Image {
        width: w,
        height: h,
        pixels: px.into_iter().map(|v| v as u8).collect(),
    })
}

/// Registry adapter: the link-time registry's `load` slot is typed on the backend-neutral
/// [`gen_core::Result`] (epic 3720); bridge the crate's rich-`Result` [`load`] into it.
fn load_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load(spec).map_err(Into::into)
}

inventory::submit! {
    ModelRegistration { descriptor, load: load_registered }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps() -> Capabilities {
        descriptor().capabilities
    }

    /// A valid request with a (stand-in) JSON-caption prompt.
    fn req(w: u32, h: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: r#"{"high_level_description":"a fox"}"#.into(),
            width: w,
            height: h,
            ..Default::default()
        }
    }

    #[test]
    fn descriptor_is_ideogram_4() {
        let d = descriptor();
        assert_eq!(d.id, "ideogram_4");
        assert_eq!(d.family, "ideogram");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.conditioning.is_empty());
        assert!(d.capabilities.supported_quants.is_empty()); // sc-5989 not yet wired
        assert_eq!(
            (d.capabilities.min_size, d.capabilities.max_size),
            (256, 2048)
        );
    }

    #[test]
    fn validate_accepts_in_surface() {
        assert!(validate_request(&caps(), &req(1024, 1024)).is_ok());
        // Exactly 6:1 is allowed (1536 / 256 = 6).
        assert!(validate_request(&caps(), &req(256, 1536)).is_ok());
        // guidance is supported.
        assert!(validate_request(
            &caps(),
            &GenerationRequest {
                guidance: Some(7.0),
                ..req(512, 512)
            }
        )
        .is_ok());
    }

    #[test]
    fn validate_rejects_empty_prompt() {
        let e = validate_request(&caps(), &GenerationRequest::default())
            .unwrap_err()
            .to_string();
        assert!(e.contains("empty"), "got: {e}");
    }

    #[test]
    fn validate_rejects_non_multiple_of_16() {
        for (w, h) in [(1000, 1000), (257, 256), (512, 520)] {
            let e = validate_request(&caps(), &req(w, h))
                .unwrap_err()
                .to_string();
            assert!(e.contains("multiple of 16"), "{w}x{h} got: {e}");
        }
    }

    #[test]
    fn validate_rejects_out_of_range_size() {
        assert!(validate_request(&caps(), &req(128, 128)).is_err()); // below min
        assert!(validate_request(&caps(), &req(2064, 256)).is_err()); // above max
    }

    #[test]
    fn validate_rejects_excessive_aspect() {
        // 1792 / 256 = 7:1 (> 6:1); in range and a multiple of 16, so only the aspect guard fires.
        let e = validate_request(&caps(), &req(256, 1792))
            .unwrap_err()
            .to_string();
        assert!(e.contains("aspect"), "got: {e}");
    }

    #[test]
    fn validate_rejects_zero_steps_and_negative_prompt() {
        assert!(validate_request(
            &caps(),
            &GenerationRequest {
                steps: Some(0),
                ..req(512, 512)
            }
        )
        .is_err());
        assert!(validate_request(
            &caps(),
            &GenerationRequest {
                negative_prompt: Some("x".into()),
                ..req(512, 512)
            }
        )
        .is_err());
    }

    #[test]
    fn validate_rejects_unsupported_conditioning() {
        let r = GenerationRequest {
            conditioning: vec![mlx_gen::Conditioning::Reference {
                image: Image::default(),
                strength: None,
            }],
            ..req(512, 512)
        };
        assert!(validate_request(&caps(), &r).is_err());
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/x.safetensors".into()));
        // `Box<dyn Generator>` isn't Debug → use `.err()`.
        let e = load(&spec).err().expect("expected an error").to_string();
        assert!(e.contains("snapshot directory"), "got: {e}");
    }

    #[test]
    fn load_rejects_quant_until_sc5989() {
        let spec =
            LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_quant(mlx_gen::Quant::Q8);
        let e = load(&spec).err().expect("expected an error").to_string();
        assert!(e.contains("quantization"), "got: {e}");
    }

    #[test]
    fn reachable_via_registry_by_id() {
        // Linking this crate self-registers ideogram_4; it must be discoverable and resolve to OUR
        // loader (a nonexistent dir fails inside load, NOT with "no generator registered").
        assert!(gen_core::registry::generators().any(|r| (r.descriptor)().id == MODEL_ID));
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-ideogram".into()));
        let e = gen_core::registry::load(MODEL_ID, &spec)
            .err()
            .expect("missing weights → err")
            .to_string();
        assert!(
            !e.contains("no generator registered"),
            "id not resolved: {e}"
        );
    }
}
