//! `Sd3Large` — the SD3.5-Large / Large-Turbo implementation of [`mlx_gen::Generator`], plus its
//! [`descriptor`]/[`load`] entry points and the `inventory` registration that wires it into
//! `mlx_gen`'s registry (E5 sc-7864 = Large; E6 sc-7865 = Large-Turbo).
//!
//! [`load`] / [`load_turbo`] assemble the full model from a `stabilityai/stable-diffusion-3.5-large`
//! (resp. `-large-turbo`) snapshot directory (see [`crate::loader`]) — CLIP BPE + T5 tokenizers, the
//! triple text encoder, the MMDiT transformer, the 16-ch VAE — and [`Sd3Large::generate`] runs the
//! complete prompt→image pipeline: tokenize → triple-TE conditioning → seeded noise → flow-match Euler
//! denoise over the MMDiT → VAE decode → RGB8 (see [`crate::pipeline`]).
//!
//! ## Large vs Large-Turbo
//!
//! Both variants share **one MMDiT backbone arch** ([`Sd3Arch::large`](crate::config::Sd3Arch::large))
//! and one snapshot layout — the Turbo checkpoint is ADD-distilled to a few-step, guidance-baked
//! schedule, so it differs ONLY in the *sampling recipe*, not the tensor layout:
//!
//! * **Large** — true-CFG: default **28 steps**, **guidance 3.5**, negative prompt supported. Each
//!   denoise step runs the MMDiT TWICE (cond + uncond).
//! * **Large-Turbo** — distilled: default **4 steps**, **guidance 1.0 → CFG off**, no negative prompt.
//!   Guidance is baked into the distilled weights, so each step runs the MMDiT ONCE (cond only) — both
//!   faster and required (a CFG forward on a distilled model is wrong). The flow-match shift is the
//!   same `3.0` (verified against the Turbo `scheduler_config.json`).
//!
//! The variant-aware [`pipeline::denoise_cfg`] already skips the uncond forward when guidance is `1.0`
//! (and `uncond` is `None`), so the Turbo path reuses E5's pipeline unchanged — only the per-variant
//! step/guidance defaults + descriptor differ.
//!
//! Registered engine ids: **`sd3_5_large`** + **`sd3_5_large_turbo`** + **`sd3_5_medium`** (the
//! SceneWorks worker's `payload.model`). Medium (M3, sc-7869) reuses ALL of this scaffolding — the
//! same [`Sd3Large`] generator struct, loader, and pipeline — but is driven by the MMDiT-X arch
//! ([`Sd3Variant::Medium`](crate::config::Sd3Variant::Medium) →
//! [`Sd3Arch::medium`](crate::config::Sd3Arch::medium): 24 blocks,
//! hidden 1536, `pos_embed_max_size` 384, dual-attention blocks `0..=12`) and the Medium
//! true-CFG recipe (40 steps / guidance 5.0). The struct keeps the historical `Sd3Large` name; it is
//! variant-parameterized and serves all three variants.

use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    default_seed, resolve_flow_schedule, Error, FlowMatchEuler, GenerationOutput,
    GenerationRequest, Generator, LoadSpec, ModelDescriptor, Precision, Progress, Result,
    WeightsSource,
};

use mlx_gen_sdxl::tokenizer::ClipBpeTokenizer;
use mlx_gen_z_image::vae::Vae;

use crate::config::Sd3Variant;
use crate::loader;
use crate::pipeline::{self, SCHEDULE_SHIFT};
use crate::text::Sd3TextEncoders;
use crate::transformer::Sd3Transformer;

/// Registry id for SD3.5-Large (matches the SceneWorks worker's `payload.model`).
pub const MODEL_ID: &str = crate::config::SD3_5_LARGE_ID;
/// Registry id for SD3.5-Large-Turbo (the distilled few-step / CFG-off variant on the same backbone).
pub const TURBO_MODEL_ID: &str = crate::config::SD3_5_LARGE_TURBO_ID;
/// Registry id for SD3.5-Medium (the MMDiT-X variant — 24 blocks, hidden 1536, dual-attention in the
/// first 13 — true-CFG, on the same triple-TE + 16-ch VAE + flow-match-Euler scaffolding).
pub const MEDIUM_MODEL_ID: &str = crate::config::SD3_5_MEDIUM_ID;

/// SD3.5-Large's identity + capabilities — constructible without loading weights (registry
/// introspection). The full capability surface lives on [`Sd3Variant::Large`].
pub fn descriptor() -> ModelDescriptor {
    Sd3Variant::Large.descriptor()
}

/// SD3.5-Large-Turbo's identity + capabilities — the distilled few-step variant (no CFG / negative
/// prompt). The capability surface lives on [`Sd3Variant::LargeTurbo`].
pub fn turbo_descriptor() -> ModelDescriptor {
    Sd3Variant::LargeTurbo.descriptor()
}

/// SD3.5-Medium's identity + capabilities (M3, sc-7869) — the MMDiT-X variant: true-CFG (negative
/// prompt + guidance), default 40 steps / guidance 5.0, on the same triple-TE + 16-ch VAE +
/// flow-match-Euler (shift 3.0) pipeline. The capability surface lives on [`Sd3Variant::Medium`];
/// its `max_size` (1440) is the higher-res ceiling Medium's `pos_embed_max_size` (384) can span while
/// staying inside the Mac activation budget (see [`load_medium`]).
pub fn medium_descriptor() -> ModelDescriptor {
    Sd3Variant::Medium.descriptor()
}

/// A loaded SD3.5-Large / Large-Turbo generator: the tokenizers + three text encoders + MMDiT + VAE
/// assembled from a snapshot directory, plus the cached descriptor and the [`Sd3Variant`] that
/// selects the sampling recipe (step/guidance defaults, CFG on/off).
pub struct Sd3Large {
    variant: Sd3Variant,
    descriptor: ModelDescriptor,
    clip_tokenizer: ClipBpeTokenizer,
    t5_tokenizer: TextTokenizer,
    encoders: Sd3TextEncoders,
    transformer: Sd3Transformer,
    vae: Vae,
}

/// Construct a SD3.5-**Large** generator from a [`LoadSpec`]. See [`load_variant`] for the shared body.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(spec, Sd3Variant::Large)
}

/// Construct a SD3.5-**Large-Turbo** generator from a [`LoadSpec`]. Identical backbone/load path to
/// [`load`] (same `Sd3Arch::large` arch + snapshot layout); only the sampling recipe (4 steps,
/// guidance-baked CFG-off) and the advertised capabilities differ. See [`load_variant`].
pub fn load_turbo(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(spec, Sd3Variant::LargeTurbo)
}

/// Construct a SD3.5-**Medium** generator from a [`LoadSpec`] (M3, sc-7869). Same load path as
/// [`load`] but driven by the Medium **MMDiT-X** arch
/// ([`Sd3Arch::medium`](crate::config::Sd3Arch::medium): 24 blocks, hidden 1536,
/// `pos_embed_max_size` 384, dual-attention in blocks `0..=12`) and the
/// `stabilityai/stable-diffusion-3.5-medium` snapshot. The triple-TE, the 16-ch VAE, the snapshot
/// layout, and the flow-match-Euler (shift 3.0 — verified against Medium's `scheduler_config.json`)
/// pipeline are all shared with Large; only the transformer arch + the true-CFG sampling recipe
/// (40 steps / guidance 5.0) differ. See [`load_variant`].
pub fn load_medium(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(spec, Sd3Variant::Medium)
}

/// Construct a [`Sd3Large`] for `variant` from a [`LoadSpec`].
///
/// `spec.weights` must be a [`WeightsSource::Dir`] pointing at the matching
/// `stabilityai/stable-diffusion-3.5-{large,large-turbo}` snapshot (the diffusers multi-component
/// tree; both variants share the same layout). `spec.quantize` (Q4/Q8) quantizes the WHOLE model —
/// transformer + the three text encoders (group_size 64) — after the dense load, matching the fork's
/// `nn.quantize` over every quantizable Linear so a Q4/Q8 consumer gets the full memory saving. The
/// VAE stays dense (its decode quality dominates the final image; matches the other DiT families'
/// quantize-the-heavy-parts convention). An fp32 precision override is rejected (the validated dense
/// path is bf16/f32 internal).
pub fn load_variant(spec: &LoadSpec, variant: Sd3Variant) -> Result<Box<dyn Generator>> {
    let id = variant.id();
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{id}: only the default dense precision is wired (the CLIP encoders run f32 and the T5 \
             promotes internally; drop the precision override)"
        )));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(Error::Msg(format!(
                "{id} expects a snapshot directory (transformer/ text_encoder{{,_2,_3}}/ \
                 tokenizer{{,_2,_3}}/ vae/), not a single .safetensors file"
            )))
        }
    };
    let arch = variant.arch();
    let mut transformer = loader::load_transformer(root, &arch)?;
    let mut encoders = loader::load_text_encoders(root)?;
    let vae = loader::load_vae(root)?;
    if let Some(q) = spec.quantize {
        let bits = q.bits();
        transformer.quantize(bits)?;
        encoders.quantize(bits)?;
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(format!(
            "{id}: LoRA/LoKr adapters are a later epic story (T1–T4); none are wired yet"
        )));
    }
    Ok(Box::new(Sd3Large {
        variant,
        descriptor: variant.descriptor(),
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

        // Per-variant sampling recipe: Large = 28 steps / guidance 3.5 (true-CFG); Large-Turbo = 4
        // steps / guidance 1.0 (distilled, guidance-baked → CFG off). The pipeline below skips the
        // uncond forward whenever guidance == 1.0, so Turbo runs ONE forward per step.
        let steps = req.steps.unwrap_or_else(|| self.variant.default_steps()) as usize;
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let guidance = req
            .guidance
            .unwrap_or_else(|| self.variant.default_guidance());

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
    let id = desc.id;
    if req.prompt.is_empty() {
        return Err(Error::Msg(format!("{id}: prompt must not be empty")));
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
        return Err(Error::Msg(format!(
            "{id}: guidance is not supported on this variant (distilled Turbo bakes guidance in — \
             CFG off)"
        )));
    }
    if req.negative_prompt.is_some() && !caps.supports_negative_prompt {
        return Err(Error::Msg(format!(
            "{id}: negative prompt is not supported on this variant"
        )));
    }
    for c in &req.conditioning {
        let kind = c.kind();
        if !caps.accepts(kind) {
            return Err(Error::Msg(format!(
                "{id} does not accept {kind:?} conditioning (txt2img only)"
            )));
        }
    }
    Ok(())
}

// Link-time registration (epic 3720): the macro emits the `inventory::submit!` and bridges the
// crate's rich `Result` into the registry's backend-neutral `gen_core::Result`. Both the true-CFG
// Large (E5) and the distilled CFG-off Large-Turbo (E6) register here on the shared backbone.
mlx_gen::register_generators! {
    descriptor => load,
    turbo_descriptor => load_turbo,
    medium_descriptor => load_medium,
}

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
    fn turbo_descriptor_is_distilled_cfg_off() {
        // E6: Large-Turbo registers its own id, is NOT true-CFG, and rejects guidance / negative
        // prompt (guidance is baked into the distilled weights — CFG off).
        let d = turbo_descriptor();
        assert_eq!(d.id, "sd3_5_large_turbo");
        assert_eq!(d.family, "sd3");
        assert_eq!(d.modality, Modality::Image);
        assert!(!d.capabilities.supports_true_cfg);
        assert!(!d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_negative_prompt);
    }

    #[test]
    fn turbo_recipe_defaults() {
        // The distilled few-step / guidance-baked recipe: 4 steps, guidance 1.0 (CFG off).
        assert_eq!(Sd3Variant::LargeTurbo.default_steps(), 4);
        assert_eq!(Sd3Variant::LargeTurbo.default_guidance(), 1.0);
        // And Large stays true-CFG at 28 steps / guidance 3.5.
        assert_eq!(Sd3Variant::Large.default_steps(), 28);
        assert_eq!(Sd3Variant::Large.default_guidance(), 3.5);
    }

    #[test]
    fn turbo_rejects_guidance_and_negative_prompt() {
        // On Turbo, supplying guidance or a negative prompt is a validation error (CFG-off variant).
        let d = turbo_descriptor();
        let with_guidance = GenerationRequest {
            prompt: "a fox".into(),
            guidance: Some(3.5),
            ..Default::default()
        };
        assert!(validate_request(&d, &with_guidance).is_err());
        let with_neg = GenerationRequest {
            prompt: "a fox".into(),
            negative_prompt: Some("blurry".into()),
            ..Default::default()
        };
        assert!(validate_request(&d, &with_neg).is_err());
        // A plain txt2img request (no guidance/neg) passes.
        let plain = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        assert!(validate_request(&d, &plain).is_ok());
    }

    #[test]
    fn turbo_load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/sd3.safetensors".into()));
        let err = load_turbo(&spec)
            .err()
            .expect("expected an error")
            .to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
        // The error is namespaced to the Turbo id (not the Large id).
        assert!(err.contains("sd3_5_large_turbo"), "got: {err}");
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

    // --- M3 (sc-7869): SD3.5-Medium vertical -----------------------------------------------------

    #[test]
    fn medium_descriptor_is_sd3_5_medium_true_cfg() {
        // Medium registers its own engine id and is true-CFG (negative prompt + guidance), like Large
        // (NOT a distilled Turbo).
        let d = medium_descriptor();
        assert_eq!(d.id, "sd3_5_medium");
        assert_eq!(d.family, "sd3");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.supports_true_cfg);
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        // The higher-res ceiling: Medium's pos_embed_max_size (384) spans up to 1440² (patch grid
        // 90×90 ≤ 384²); the descriptor caps there to stay inside the Mac activation budget (2K is
        // activation-bound — the SCAIL-2 / SeedVR2 lesson).
        assert_eq!(d.capabilities.max_size, 1440);
    }

    #[test]
    fn medium_recipe_defaults() {
        // Medium's reference recipe: 40 steps / guidance 5.0 (true-CFG, more guidance-sensitive than
        // Large per Stability's model card). Shift stays 3.0 (Medium scheduler_config.json).
        assert_eq!(Sd3Variant::Medium.default_steps(), 40);
        assert_eq!(Sd3Variant::Medium.default_guidance(), 5.0);
    }

    #[test]
    fn medium_uses_mmdit_x_arch() {
        // The Medium generator is driven by the MMDiT-X arch (24 blocks, hidden 1536, dual-attention
        // in the first 13) — distinct from Large's plain MMDiT.
        let arch = Sd3Variant::Medium.arch();
        assert_eq!(arch.num_layers, 24);
        assert_eq!(arch.hidden(), 1536);
        assert_eq!(arch.dual_attention_layers, 13);
        assert_eq!(arch.pos_embed_max_size, 384);
        assert!(arch.is_dual_attention_block(12));
        assert!(!arch.is_dual_attention_block(13));
    }

    #[test]
    fn medium_validate_accepts_guidance_and_negative_prompt() {
        let d = medium_descriptor();
        let req = GenerationRequest {
            prompt: "a fox".into(),
            guidance: Some(5.0),
            negative_prompt: Some("blurry".into()),
            ..Default::default()
        };
        assert!(validate_request(&d, &req).is_ok());
    }

    #[test]
    fn medium_validate_guards_resolution_ceiling() {
        // 1440² is allowed (the validated Medium ceiling); anything above max_size is rejected by the
        // shared guard — the activation-budget cap (2K would be activation-bound).
        let d = medium_descriptor();
        let ok = GenerationRequest {
            prompt: "a fox".into(),
            width: 1440,
            height: 1440,
            ..Default::default()
        };
        assert!(validate_request(&d, &ok).is_ok());
        let too_big = GenerationRequest {
            prompt: "a fox".into(),
            width: 2048,
            height: 2048,
            ..Default::default()
        };
        let err = validate_request(&d, &too_big).unwrap_err().to_string();
        assert!(err.contains("out of supported range"), "got: {err}");
    }

    #[test]
    fn medium_load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/sd3.safetensors".into()));
        let err = load_medium(&spec)
            .err()
            .expect("expected an error")
            .to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
        assert!(err.contains("sd3_5_medium"), "got: {err}");
    }

    #[test]
    fn medium_load_accepts_quant_spec() {
        for q in [mlx_gen::Quant::Q4, mlx_gen::Quant::Q8] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_quant(q);
            let err = load_medium(&spec)
                .err()
                .expect("expected an error")
                .to_string();
            assert!(!err.contains("quantization"), "got: {err}");
        }
    }
}
