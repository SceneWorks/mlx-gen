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
    Capabilities, Conditioning, ConditioningKind, Error, FlowMatchEuler, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, ModelRegistration,
    Precision, Progress, Result, WeightsSource,
};
use mlx_rs::Dtype;

use crate::loader;
use crate::pipeline::{
    add_noise_by_interpolation, create_noise, decoded_to_image, denoise_with_progress,
    encode_init_latents, init_time_step, slice_valid, unpack_latents,
};
use crate::text_encoder::TextEncoder;
use crate::transformer::ZImageTransformer;
use crate::vae::Vae;

/// Z-Image-turbo is guidance-distilled to a fixed 4-step schedule; used when a request omits
/// `steps`.
const DEFAULT_STEPS: u32 = 4;

/// Flow-match time-shift for Z-Image-Turbo. Pinned by the model's own
/// `scheduler/scheduler_config.json` (`FlowMatchEulerDiscreteScheduler`, `shift=3.0`,
/// `use_dynamic_shifting=false`) — the static schedule used by the diffusers `ZImagePipeline`
/// (the SceneWorks production path) and approximated by mflux's `linear` scheduler. NOT the
/// empirical per-step `mu` of `FlowMatchEuler::for_image` (that is the *full* Z-Image model's
/// scheduler; using it here was the sc-2536 bug).
const SCHEDULE_SHIFT: f32 = 3.0;

/// Registry id for Z-Image-turbo (matches the SceneWorks worker's `payload.model`).
pub const MODEL_ID: &str = "z_image_turbo";

/// Z-Image-turbo's identity + capabilities — constructible without loading weights (registry
/// introspection). Values are conservative-but-real; sampler/scheduler lists fill in with the
/// scheduler port.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "z-image",
        modality: Modality::Image,
        capabilities: Capabilities {
            // Turbo is guidance-distilled: no CFG, no negative prompt.
            supports_negative_prompt: false,
            supports_guidance: false,
            supports_true_cfg: false,
            // img2img reference; ControlNet is a separate variant (sc-2349).
            conditioning: vec![ConditioningKind::Reference],
            supports_lora: true,
            supports_lokr: true,
            samplers: Vec::new(),
            schedulers: Vec::new(),
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: true,
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// A loaded Z-Image-turbo generator: the four model components assembled from a snapshot
/// directory, plus the cached descriptor.
pub struct ZImageTurbo {
    descriptor: ModelDescriptor,
    tokenizer: TextTokenizer,
    text_encoder: TextEncoder,
    transformer: ZImageTransformer,
    vae: Vae,
}

/// Construct a [`ZImageTurbo`] from a [`LoadSpec`].
///
/// `spec.weights` must be a [`WeightsSource::Dir`] pointing at a `Tongyi-MAI/Z-Image-Turbo`
/// snapshot (the diffusers multi-component tree — `tokenizer/`, `text_encoder/`, `transformer/`,
/// `vae/`). Weights load dense at their on-disk dtype (bf16); the text encoder promotes to f32
/// internally. `spec.quantize` (Q4/Q8) quantizes the **transformer only** (group_size 64) after
/// the dense load — the mflux fork's `nn.quantize` predicate matches every Linear in the
/// transformer. An fp32 precision override is not wired (the validated dense path is bf16) and is
/// rejected rather than silently ignored.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "z_image_turbo: only dense bf16 is wired in the Rust port; the text encoder already \
             runs f32 internally (drop the precision override)"
                .into(),
        ));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(Error::Msg(
                "z_image_turbo expects a snapshot directory (tokenizer/ text_encoder/ \
                 transformer/ vae/), not a single .safetensors file"
                    .into(),
            ))
        }
    };
    // Q4/Q8 quantizes the transformer in place after the dense bf16 load (the fork's
    // `nn.quantize` set, group_size 64). The text encoder + VAE run dense — the generate path's
    // parity is proven against the fork's quantized `cap_feats` (sc-2532).
    let mut transformer = loader::load_transformer(root)?;
    if let Some(q) = spec.quantize {
        transformer.quantize(q.bits())?;
    }
    Ok(Box::new(ZImageTurbo {
        descriptor: descriptor(),
        tokenizer: loader::load_tokenizer(root)?,
        text_encoder: loader::load_text_encoder(root)?,
        transformer,
        vae: loader::load_vae(root)?,
    }))
}

impl ZImageTurbo {
    /// Prompt → `cap_feats` (f32): tokenize with the Qwen chat template, run the text encoder,
    /// and slice off the padded tail to the valid caption tokens.
    fn encode_prompt(&self, prompt: &str) -> Result<mlx_rs::Array> {
        let t = self.tokenizer.tokenize(prompt)?;
        let num_valid: i32 = t.attention_mask.as_slice::<i32>().iter().sum();
        if num_valid == 0 {
            return Err(Error::Msg("z_image_turbo: empty prompt".into()));
        }
        let enc = self.text_encoder.forward(&t.input_ids, &t.attention_mask)?;
        slice_valid(&enc, num_valid)
    }

    /// Extract the single img2img init image + its strength from the request's conditioning. The
    /// per-reference strength wins over `req.strength`. Z-Image img2img conditions on exactly one
    /// init image, so more than one `Reference` is an error (multi-image is `MultiReference`, which
    /// this model doesn't advertise).
    fn resolve_reference<'a>(
        &self,
        req: &'a GenerationRequest,
    ) -> Result<Option<(&'a Image, Option<f32>)>> {
        let mut reference = None;
        for c in &req.conditioning {
            if let Conditioning::Reference { image, strength } = c {
                if reference.is_some() {
                    return Err(Error::Msg(
                        "z_image_turbo: multiple reference images are not supported (single \
                         img2img init only)"
                            .into(),
                    ));
                }
                reference = Some((image, strength.or(req.strength)));
            }
        }
        Ok(reference)
    }
}

impl Generator for ZImageTurbo {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> Result<()> {
        validate_request(&self.descriptor.capabilities, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;

        let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
        let base_seed = req.seed.unwrap_or_else(default_seed);

        // img2img: a single `Reference` image, with a per-reference strength overriding `req.strength`.
        let reference = self.resolve_reference(req)?;
        let start_step = match reference {
            Some((_, strength)) => init_time_step(steps, strength),
            None => 0,
        };
        let is_img2img = start_step > 0;

        // Prompt → cap_feats (f32). txt2img runs the DiT in bf16 (the parity-proven path); img2img
        // matches the fork's f32 init latents, so keep cap f32 too (so the unified stream is one
        // dtype). The DiT promotes per-op against the bf16 weights either way.
        let cap = self.encode_prompt(&req.prompt)?;
        let cap = if is_img2img {
            cap
        } else {
            cap.as_dtype(Dtype::Bfloat16)?
        };

        // Static shift=3.0 schedule (the model's scheduler_config.json), resolution- and
        // seed-independent — build it once. See SCHEDULE_SHIFT.
        let scheduler = FlowMatchEuler::for_static_shift(steps, SCHEDULE_SHIFT);

        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            // Distinct seed per image in a batch (the fork's `seed + i` convention).
            let seed = base_seed.wrapping_add(i as u64);
            // Seeded noise as bf16 (the fork's `create_noise` casts to model precision).
            let noise = create_noise(seed, req.width, req.height)?.as_dtype(Dtype::Bfloat16)?;
            let latents = if is_img2img {
                // VAE-encode the init image to clean latents (f32), then blend with the noise at
                // `sigma = sigmas[init_time_step]` (the fork's `create_for_txt2img_or_img2img`).
                let (image, _) = reference.expect("is_img2img implies a reference");
                let clean = encode_init_latents(&self.vae, image, req.width, req.height)?;
                let sigma = scheduler.sigmas[start_step];
                add_noise_by_interpolation(&clean, &noise, sigma)?
            } else {
                noise
            };
            let latents = denoise_with_progress(
                &self.transformer,
                &scheduler,
                latents,
                &cap,
                start_step,
                &req.cancel,
                on_progress,
            )?;

            on_progress(Progress::Decoding);
            // [16,1,H,W] -> [1,16,H,W] -> [1,16,1,H,W] for VAE decode.
            let unpacked = unpack_latents(&latents)?;
            let sh = unpacked.shape();
            let latent5 = unpacked.reshape(&[sh[0], sh[1], 1, sh[2], sh[3]])?;
            let decoded = self.vae.decode(&latent5)?.as_dtype(Dtype::Float32)?;
            images.push(decoded_to_image(&decoded)?);
        }
        Ok(GenerationOutput::Images(images))
    }
}

/// Seed when a request omits one: nanos since the epoch (any nonzero value works; this only sets
/// which sample is drawn, and a caller wanting reproducibility passes `req.seed`).
fn default_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Capability-driven request validation, factored out so it can be unit-tested without loaded
/// weights. Rejects unsupported guidance / negative prompt / conditioning / size / count.
pub(crate) fn validate_request(caps: &Capabilities, req: &GenerationRequest) -> Result<()> {
    if req.count == 0 || req.count > caps.max_count {
        return Err(mlx_gen::Error::Msg(format!(
            "count {} out of range 1..={}",
            req.count, caps.max_count
        )));
    }
    if req.width < caps.min_size
        || req.height < caps.min_size
        || req.width > caps.max_size
        || req.height > caps.max_size
    {
        return Err(mlx_gen::Error::Msg(format!(
            "{}x{} out of supported range {}..={}",
            req.width, req.height, caps.min_size, caps.max_size
        )));
    }
    if req.guidance.is_some() && !caps.supports_guidance {
        return Err(mlx_gen::Error::Msg(
            "z_image_turbo is guidance-distilled; `guidance` is not supported".into(),
        ));
    }
    if req.negative_prompt.is_some() && !caps.supports_negative_prompt {
        return Err(mlx_gen::Error::Msg(
            "z_image_turbo does not support a negative prompt".into(),
        ));
    }
    for c in &req.conditioning {
        let kind = match c {
            Conditioning::Reference { .. } => ConditioningKind::Reference,
            Conditioning::MultiReference { .. } => ConditioningKind::MultiReference,
            Conditioning::ReduxRefs { .. } => ConditioningKind::ReduxRefs,
            Conditioning::Control { .. } => ConditioningKind::Control,
            Conditioning::Depth { .. } => ConditioningKind::Depth,
            Conditioning::Mask { .. } => ConditioningKind::Mask,
        };
        if !caps.accepts(kind) {
            return Err(mlx_gen::Error::Msg(format!(
                "z_image_turbo does not accept {kind:?} conditioning"
            )));
        }
    }
    Ok(())
}

inventory::submit! {
    ModelRegistration { descriptor, load }
}

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
    fn validate_rejects_guidance_and_bad_size() {
        let caps = descriptor().capabilities;
        // guidance on a distilled model.
        let mut req = GenerationRequest {
            guidance: Some(4.0),
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_err());
        // out-of-range size.
        req = GenerationRequest {
            width: 64,
            height: 64,
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_err());
        // a plain valid request passes.
        req = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_ok());
    }

    #[test]
    fn validate_rejects_unsupported_conditioning() {
        let caps = descriptor().capabilities;
        let req = GenerationRequest {
            conditioning: vec![Conditioning::Depth {
                image: mlx_gen::Image::default(),
            }],
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_err());
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
        // Q4/Q8 is wired (transformer-only); a quant spec must get past the load entry point and
        // fail later on the missing snapshot, not on quantization being unsupported.
        for q in [mlx_gen::Quant::Q4, mlx_gen::Quant::Q8] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_quant(q);
            let err = load(&spec).err().expect("expected an error").to_string();
            assert!(!err.contains("quantization"), "got: {err}");
        }
    }
}
