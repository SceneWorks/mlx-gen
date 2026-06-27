//! `ZImage` — the **base** (non-distilled, full-CFG) Z-Image implementation of
//! [`mlx_gen::Generator`] (sc-8320), registered as its own engine id `z_image` alongside the
//! distilled `z_image_turbo` ([`crate::model`]) and `z_image_turbo_control`
//! ([`crate::model_control`]).
//!
//! The base and Turbo share the **identical** `ZImageTransformer2DModel` architecture (n_layers=30,
//! dim=3840, n_heads=30, cap_feat_dim=2560, qk_norm, rope_theta=256, t_scale=1000) — verified against
//! both `transformer/config.json`s — so this generator **reuses** [`crate::transformer::ZImageTransformer`],
//! the loader, the VAE, and the text encoder unchanged. The deltas (all from the base model card /
//! `scheduler/scheduler_config.json`) are:
//!
//! * **Scheduler shift = 6.0** (Turbo = 3.0). `FlowMatchEulerDiscreteScheduler`, `shift=6.0`,
//!   `use_dynamic_shifting=false` — static, resolution-independent.
//! * **Default steps = 50** (Turbo = 4). The base is undistilled; the card recommends 28–50.
//! * **Real classifier-free guidance** (Turbo is guidance-distilled → CFG-free). The base is a
//!   non-distilled foundation model: it supports full CFG (`guidance_scale` 3.0–5.0, default 4.0) and
//!   a negative prompt. Each step runs the DiT twice (cond + uncond) and combines
//!   `v = v_uncond + guidance·(v_cond − v_uncond)` — see [`pipeline::denoise_cfg_with_progress`].
//!   `guidance == 1.0` collapses to a single cond forward (Turbo-equivalent cost).
//!
//! [`load`] assembles the model from a `Tongyi-MAI/Z-Image` snapshot directory (the same diffusers
//! multi-component tree the Turbo loader consumes); the coordinator points the catalog entry + re-pin
//! at that snapshot (a follow-up). Turbo and the control variant are untouched.

use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, default_seed, resolve_flow_schedule,
    Capabilities, ConditioningKind, Error, FlowMatchEuler, GenerationOutput, GenerationRequest,
    Generator, LatentDecoder, LoadSpec, Modality, ModelDescriptor, Precision, Progress, Quant,
    Result, WeightsSource,
};
use mlx_gen_pid::{resolve_pid_decoder, PidEngine};

use crate::loader;
use crate::model::validate_request;
use crate::pipeline::{self, denoise_cfg_with_progress, encode_init_latents, init_time_step};
use crate::text_encoder::TextEncoder;
use crate::transformer::ZImageTransformer;
use crate::vae::Vae;

/// Base Z-Image default steps — undistilled foundation model. The card recommends 28–50; 50 matches
/// the reference `ZImagePipeline` example (`num_inference_steps=50`). Used when a request omits `steps`.
pub(crate) const DEFAULT_STEPS: u32 = 50;

/// Flow-match time-shift for the **base** Z-Image: `scheduler/scheduler_config.json`
/// (`FlowMatchEulerDiscreteScheduler`, `shift=6.0`, `use_dynamic_shifting=false`) — static,
/// resolution-independent. **This is the sole scheduler delta vs Turbo (3.0).**
pub(crate) const SCHEDULE_SHIFT: f32 = 6.0;

/// Default CFG scale for the base — the card recommends 3.0–5.0; 4.0 matches the reference
/// `ZImagePipeline` example (`guidance_scale=4`). Used when a request omits `guidance`.
pub(crate) const DEFAULT_GUIDANCE: f32 = 4.0;

/// Registry id for the **base** Z-Image (non-Turbo). Coexists with `z_image_turbo` +
/// `z_image_turbo_control` — distinct id, separate `inventory` registration, no clash.
pub const MODEL_ID: &str = "z_image";

/// PiD backbone tag for the base Z-Image (epic 7840). Identical VAE latent space to Turbo (Flux1-dev's
/// 16-ch VAE), so it reuses the same `flux` PiD student via the `zimage-turbo` registry alias.
pub const PID_BACKBONE: &str = "zimage-turbo";

/// Base Z-Image's identity + capabilities — constructible without loading weights. Unlike Turbo, the
/// base is a non-distilled foundation model: real CFG (guidance + negative prompt) is supported.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "z-image",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            supported_quants: &[Quant::Q4, Quant::Q8],
            // Base is undistilled → full classifier-free guidance + negative prompting (the model card's
            // headline capabilities), unlike the guidance-distilled Turbo.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: true,
            // img2img reference; ControlNet is a separate variant (base control = sc-8251).
            conditioning: vec![ConditioningKind::Reference],
            supports_lora: true,
            supports_lokr: true,
            // Curated unified-framework integrator menu (epic 7114 P3); an unset `req.sampler` is the
            // curated Euler over the static shift=6.0 schedule.
            samplers: curated_sampler_names(),
            // Scheduler axis (epic 7114): the static shift=6.0 schedule is the byte-exact default; a
            // curated name re-shapes the σ schedule over the same `shift=6.0`.
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

/// A loaded base Z-Image generator — the four model components assembled from a snapshot directory,
/// plus the cached descriptor and an optional PiD overlay. Same component set as [`crate::model::ZImageTurbo`].
pub struct ZImage {
    descriptor: ModelDescriptor,
    tokenizer: TextTokenizer,
    text_encoder: TextEncoder,
    transformer: ZImageTransformer,
    vae: Vae,
    /// Optional PiD super-resolving decoder overlay (epic 7840). `Some` only when the `LoadSpec`
    /// carried `pid`; selected per-generation by `req.use_pid`. Same `zimage-turbo` PiD alias as Turbo.
    pid: Option<PidEngine>,
}

/// Construct a [`ZImage`] from a [`LoadSpec`].
///
/// `spec.weights` must be a [`WeightsSource::Dir`] pointing at a `Tongyi-MAI/Z-Image` snapshot (the
/// diffusers multi-component tree — `tokenizer/ text_encoder/ transformer/ vae/`). The load is
/// byte-identical to the Turbo loader (the transformer architecture is the same); only the generate-time
/// schedule + CFG differ. `spec.quantize` (Q4/Q8) quantizes the **whole model** after the dense bf16
/// load, exactly as Turbo.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "z_image: only dense bf16 is wired in the Rust port; the text encoder already runs f32 \
             internally (drop the precision override)"
                .into(),
        ));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => return Err(Error::Msg(
            "z_image expects a snapshot directory (tokenizer/ text_encoder/ transformer/ vae/), \
                 not a single .safetensors file"
                .into(),
        )),
    };
    let mut transformer = loader::load_transformer(root)?;
    let mut text_encoder = loader::load_text_encoder(root)?;
    let mut vae = loader::load_vae(root)?;
    if let Some(q) = spec.quantize {
        let bits = q.bits();
        transformer.quantize(bits)?;
        text_encoder.quantize(bits)?;
        vae.quantize(bits)?;
    }
    if !spec.adapters.is_empty() {
        crate::adapters::apply_z_image_adapters(&mut transformer, &spec.adapters)?;
    }
    let pid = spec
        .pid
        .as_ref()
        .map(|p| PidEngine::from_spec(p, PID_BACKBONE))
        .transpose()?;
    Ok(Box::new(ZImage {
        descriptor: descriptor(),
        tokenizer: loader::load_tokenizer(root)?,
        text_encoder,
        transformer,
        vae,
        pid,
    }))
}

mlx_gen::impl_generator!(ZImage {
    validate: |s, req| validate_request(&s.descriptor.capabilities, req),
    generate: generate_impl,
});

impl ZImage {
    /// The rich-`Result` body behind [`Generator::generate`].
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;

        let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
        let base_seed = req.seed.unwrap_or_else(default_seed);
        // Real CFG: the request's `guidance` is the classifier-free guidance scale; default 4.0. A
        // value of 1.0 turns CFG off (single cond forward, Turbo-equivalent cost).
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
        let cfg_on = guidance != 1.0;

        // img2img: a single `Reference` image, with a per-reference strength overriding `req.strength`.
        let reference = pipeline::resolve_reference(req, MODEL_ID)?;
        let start_step = match reference {
            Some((_, strength)) => init_time_step(steps, strength),
            None => 0,
        };
        let is_img2img = start_step > 0;

        // Prompt → cap_feats. The base is undistilled and runs real CFG; unlike Turbo there is no
        // bf16 seed-parity golden to match, so keep the conditioning at the text encoder's native
        // precision and let the DiT promote per-op against the bf16 weights.
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
        // `req.scheduler` keeps it byte-exact (epic 7114 N1); a curated name re-shapes σ over `shift=6.0`.
        let native = FlowMatchEuler::for_static_shift(steps, SCHEDULE_SHIFT);
        let scheduler = FlowMatchEuler::from_sigmas(resolve_flow_schedule(
            req.scheduler.as_deref(),
            SCHEDULE_SHIFT.ln(),
            steps,
            &native.sigmas,
        ));

        // VAE-encode the init image once (constant across the count loop — only the noise varies).
        let clean = if is_img2img {
            let (image, _) = reference.expect("is_img2img implies a reference");
            Some(encode_init_latents(
                &self.vae, image, req.width, req.height,
            )?)
        } else {
            None
        };

        let sampler_name = req.sampler.as_deref();
        let pid_decoder = resolve_pid_decoder(self.pid.as_ref(), req, base_seed, MODEL_ID)?;
        let neg_cap_ref = neg_cap.as_ref();
        let images = pipeline::render_batch(
            &self.vae,
            pid_decoder.as_ref().map(|d| d as &dyn LatentDecoder),
            &scheduler,
            clean.as_ref(),
            start_step,
            base_seed,
            req,
            on_progress,
            |latents, seed, op| {
                denoise_cfg_with_progress(
                    &self.transformer,
                    &scheduler,
                    sampler_name,
                    seed,
                    latents,
                    &cap,
                    neg_cap_ref,
                    guidance,
                    start_step,
                    &req.cancel,
                    op,
                )
            },
        )?;
        Ok(GenerationOutput::Images(images))
    }
}

// Link-time registration (epic 3720): emits `inventory::submit!` and bridges the rich `Result` into
// the registry's backend-neutral `gen_core::Result`. A distinct id (`z_image`) → no clash with the
// turbo / control submissions in the same crate.
mlx_gen::register_generators! { descriptor => load }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_is_base_z_image() {
        let d = descriptor();
        assert_eq!(d.id, "z_image");
        assert_eq!(d.family, "z-image");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.supports_lora && d.capabilities.supports_lokr);
        // The base delta vs Turbo: real CFG + negative prompt are supported.
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_true_cfg);
        assert!(d.capabilities.supports_negative_prompt);
    }

    #[test]
    fn base_descriptor_differs_from_turbo_only_in_cfg() {
        // The two share family/backend/modality/size envelope; the documented delta is CFG support.
        let base = descriptor();
        let turbo = crate::model::descriptor();
        assert_eq!(base.family, turbo.family);
        assert_eq!(base.backend, turbo.backend);
        assert_eq!(base.modality, turbo.modality);
        assert_eq!(base.capabilities.min_size, turbo.capabilities.min_size);
        assert_eq!(base.capabilities.max_size, turbo.capabilities.max_size);
        // Distinct ids — they coexist in the registry.
        assert_ne!(base.id, turbo.id);
        // Turbo is guidance-distilled (CFG off); base is full-CFG.
        assert!(!turbo.capabilities.supports_guidance);
        assert!(base.capabilities.supports_guidance);
    }

    #[test]
    fn base_schedule_and_steps_match_the_model_card() {
        // The base config's two scalar deltas vs Turbo: shift 6.0 (Turbo 3.0) and default steps 50
        // (Turbo 4). These are the load-bearing port values from scheduler_config.json + the card.
        assert_eq!(SCHEDULE_SHIFT, 6.0);
        assert_eq!(crate::model::SCHEDULE_SHIFT, 3.0);
        assert_eq!(DEFAULT_STEPS, 50);
        assert_eq!(crate::model::DEFAULT_STEPS, 4);
        assert_eq!(DEFAULT_GUIDANCE, 4.0);
    }

    #[test]
    fn validate_accepts_guidance_and_negative_prompt() {
        // Real-CFG variant: guidance + a negative prompt are accepted (the Turbo descriptor rejects them).
        let caps = descriptor().capabilities;
        let req = GenerationRequest {
            prompt: "a fox".into(),
            guidance: Some(4.0),
            negative_prompt: Some("blurry".into()),
            ..Default::default()
        };
        assert!(
            validate_request(&caps, &req).is_ok(),
            "base must accept guidance + negative prompt"
        );
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/z.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }
}
