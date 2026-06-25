//! `Sd3Large` — the SD3.5-Large implementation of [`mlx_gen::Generator`], plus its
//! [`descriptor`]/[`load`] entry points and the `inventory` registration that wires it into
//! `mlx_gen`'s registry (E5, sc-7864).
//!
//! [`load`] assembles the full model from a `stabilityai/stable-diffusion-3.5-large` snapshot
//! directory (see [`crate::loader`]) — CLIP BPE + T5 tokenizers, the triple text encoder, the MMDiT
//! transformer, the 16-ch VAE — and [`Sd3Large::generate`] runs the complete prompt→image pipeline:
//! tokenize → triple-TE conditioning → seeded noise → flow-match Euler denoise with true-CFG over the
//! MMDiT → VAE decode → RGB8 (see [`crate::pipeline`]).
//!
//! Registered engine id: **`sd3_5_large`** (the SceneWorker worker's `payload.model`). Turbo (E6) and
//! Medium (M3) are separate stories and register their own ids on the same crate scaffolding.

use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    default_seed, resolve_flow_schedule, Error, FlowMatchEuler, GenerationOutput,
    GenerationRequest, Generator, LoadSpec, ModelDescriptor, Precision, Progress, Result,
    WeightsSource,
};

use mlx_gen_sdxl::tokenizer::ClipBpeTokenizer;
use mlx_gen_z_image::vae::Vae;

use crate::config::{Sd3Variant, DEFAULT_GUIDANCE_LARGE, DEFAULT_STEPS_LARGE};
use crate::loader;
use crate::pipeline::{self, SCHEDULE_SHIFT};
use crate::text::Sd3TextEncoders;
use crate::transformer::Sd3Transformer;

/// Registry id for SD3.5-Large (matches the SceneWorks worker's `payload.model`).
pub const MODEL_ID: &str = crate::config::SD3_5_LARGE_ID;

/// SD3.5-Large's identity + capabilities — constructible without loading weights (registry
/// introspection). The full capability surface lives on [`Sd3Variant::Large`].
pub fn descriptor() -> ModelDescriptor {
    Sd3Variant::Large.descriptor()
}

/// A loaded SD3.5-Large generator: the tokenizers + three text encoders + MMDiT + VAE assembled from
/// a snapshot directory, plus the cached descriptor.
pub struct Sd3Large {
    descriptor: ModelDescriptor,
    clip_tokenizer: ClipBpeTokenizer,
    t5_tokenizer: TextTokenizer,
    encoders: Sd3TextEncoders,
    transformer: Sd3Transformer,
    vae: Vae,
}

/// Construct a [`Sd3Large`] from a [`LoadSpec`].
///
/// `spec.weights` must be a [`WeightsSource::Dir`] pointing at a `stabilityai/stable-diffusion-3.5-
/// large` snapshot (the diffusers multi-component tree). `spec.quantize` (Q4/Q8) quantizes the WHOLE
/// model — transformer + the three text encoders (group_size 64) — after the dense load, matching the
/// fork's `nn.quantize` over every quantizable Linear so a Q4/Q8 consumer gets the full memory saving.
/// The VAE stays dense (its decode quality dominates the final image; matches the other DiT families'
/// quantize-the-heavy-parts convention). An fp32 precision override is rejected (the validated dense
/// path is bf16/f32 internal).
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "sd3_5_large: only the default dense precision is wired (the CLIP encoders run f32 and \
             the T5 promotes internally; drop the precision override)"
                .into(),
        ));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(Error::Msg(
                "sd3_5_large expects a snapshot directory (transformer/ text_encoder{,_2,_3}/ \
                 tokenizer{,_2,_3}/ vae/), not a single .safetensors file"
                    .into(),
            ))
        }
    };
    let arch = Sd3Variant::Large.arch();
    let mut transformer = loader::load_transformer(root, &arch)?;
    let mut encoders = loader::load_text_encoders(root)?;
    let vae = loader::load_vae(root)?;
    if let Some(q) = spec.quantize {
        let bits = q.bits();
        transformer.quantize(bits)?;
        encoders.quantize(bits)?;
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(
            "sd3_5_large: LoRA/LoKr adapters are a later epic story (T1–T4); none are wired yet"
                .into(),
        ));
    }
    Ok(Box::new(Sd3Large {
        descriptor: descriptor(),
        clip_tokenizer: loader::load_clip_tokenizer(root)?,
        t5_tokenizer: loader::load_t5_tokenizer(root)?,
        encoders,
        transformer,
        vae,
    }))
}

mlx_gen::impl_generator!(Sd3Large {
    validate: |s, req| validate_request(&s.descriptor, req),
    generate: generate_impl,
});

impl Sd3Large {
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;

        let steps = req.steps.unwrap_or(DEFAULT_STEPS_LARGE) as usize;
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE_LARGE);

        // Conditioning is seed-independent — encode once. Cond = the prompt; uncond = the negative
        // prompt (empty string when unset), used only when CFG is active (guidance != 1.0).
        let cond = pipeline::encode_prompt(
            &self.encoders,
            &self.clip_tokenizer,
            &self.t5_tokenizer,
            &req.prompt,
        )?;
        let cfg_on = guidance != 1.0;
        let uncond = if cfg_on {
            let neg = req.negative_prompt.as_deref().unwrap_or("");
            Some(pipeline::encode_prompt(
                &self.encoders,
                &self.clip_tokenizer,
                &self.t5_tokenizer,
                neg,
            )?)
        } else {
            None
        };

        // Static shift=3.0 schedule (scheduler_config.json), resolution-independent — build once. An
        // unset req.scheduler keeps it byte-exact; a curated name re-shapes σ over the same mu=ln(3).
        let native = FlowMatchEuler::for_static_shift(steps, SCHEDULE_SHIFT);
        let scheduler = FlowMatchEuler::from_sigmas(resolve_flow_schedule(
            req.scheduler.as_deref(),
            SCHEDULE_SHIFT.ln(),
            steps,
            &native.sigmas,
        ));

        let sampler_name = req.sampler.as_deref();
        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            let seed = base_seed.wrapping_add(i as u64);
            let latents = pipeline::create_noise(seed, req.width, req.height)?;
            let latents = pipeline::denoise_cfg(
                &self.transformer,
                &scheduler,
                sampler_name,
                seed,
                latents,
                &cond,
                uncond.as_ref(),
                guidance,
                &req.cancel,
                on_progress,
            )?;
            on_progress(Progress::Decoding);
            images.push(pipeline::decode_to_image(&self.vae, &latents)?);
        }
        Ok(GenerationOutput::Images(images))
    }
}

/// Required divisor for requested image dims: VAE downsample (8) × transformer patch (2) = 16.
const SIZE_MULTIPLE: u32 = 16;

/// Capability-driven request validation, factored out so it can be unit-tested without loaded weights.
pub(crate) fn validate_request(desc: &ModelDescriptor, req: &GenerationRequest) -> Result<()> {
    let caps = &desc.capabilities;
    if req.prompt.is_empty() {
        return Err(Error::Msg("sd3_5_large: prompt must not be empty".into()));
    }
    if req.count == 0 || req.count > caps.max_count {
        return Err(Error::Msg(format!(
            "count {} out of range 1..={}",
            req.count, caps.max_count
        )));
    }
    if req.steps == Some(0) {
        return Err(Error::Msg("steps must be >= 1".into()));
    }
    if req.width < caps.min_size
        || req.height < caps.min_size
        || req.width > caps.max_size
        || req.height > caps.max_size
    {
        return Err(Error::Msg(format!(
            "{}x{} out of supported range {}..={}",
            req.width, req.height, caps.min_size, caps.max_size
        )));
    }
    if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
        return Err(Error::Msg(format!(
            "{}x{} must be a multiple of {SIZE_MULTIPLE} (VAE 8 × patch 2)",
            req.width, req.height
        )));
    }
    if req.guidance.is_some() && !caps.supports_guidance {
        return Err(Error::Msg(
            "sd3_5_large: guidance is not supported on this variant".into(),
        ));
    }
    if req.negative_prompt.is_some() && !caps.supports_negative_prompt {
        return Err(Error::Msg(
            "sd3_5_large: negative prompt is not supported on this variant".into(),
        ));
    }
    for c in &req.conditioning {
        let kind = c.kind();
        if !caps.accepts(kind) {
            return Err(Error::Msg(format!(
                "sd3_5_large does not accept {kind:?} conditioning (txt2img only)"
            )));
        }
    }
    Ok(())
}

// Link-time registration (epic 3720): the macro emits the `inventory::submit!` and bridges the
// crate's rich `Result` into the registry's backend-neutral `gen_core::Result`.
mlx_gen::register_generators! { descriptor => load }

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::Modality;

    #[test]
    fn descriptor_is_sd3_5_large() {
        let d = descriptor();
        assert_eq!(d.id, "sd3_5_large");
        assert_eq!(d.family, "sd3");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.supports_true_cfg);
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
    }

    #[test]
    fn validate_rejects_empty_prompt() {
        let d = descriptor();
        let req = GenerationRequest::default();
        let err = validate_request(&d, &req).unwrap_err().to_string();
        assert!(err.contains("empty"), "got: {err}");
    }

    #[test]
    fn validate_rejects_bad_size_and_count() {
        let d = descriptor();
        // non-multiple-of-16
        let req = GenerationRequest {
            prompt: "a fox".into(),
            width: 1000,
            height: 1000,
            ..Default::default()
        };
        assert!(validate_request(&d, &req).is_err());
        // out-of-range count
        let req = GenerationRequest {
            prompt: "a fox".into(),
            count: 99,
            ..Default::default()
        };
        assert!(validate_request(&d, &req).is_err());
        // zero steps
        let req = GenerationRequest {
            prompt: "a fox".into(),
            steps: Some(0),
            ..Default::default()
        };
        assert!(validate_request(&d, &req).is_err());
        // a plain valid request passes.
        let req = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        assert!(validate_request(&d, &req).is_ok());
    }

    #[test]
    fn validate_accepts_guidance_and_negative_prompt() {
        // Large is true-CFG: guidance + negative prompt are supported (unlike a distilled Turbo).
        let d = descriptor();
        let req = GenerationRequest {
            prompt: "a fox".into(),
            guidance: Some(4.5),
            negative_prompt: Some("blurry".into()),
            ..Default::default()
        };
        assert!(validate_request(&d, &req).is_ok());
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/sd3.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }

    #[test]
    fn load_accepts_quant_spec() {
        for q in [mlx_gen::Quant::Q4, mlx_gen::Quant::Q8] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_quant(q);
            let err = load(&spec).err().expect("expected an error").to_string();
            assert!(!err.contains("quantization"), "got: {err}");
        }
    }
}
