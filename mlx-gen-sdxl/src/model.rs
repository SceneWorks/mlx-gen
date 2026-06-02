//! `Sdxl` — the Stable Diffusion XL implementation of [`mlx_gen::Generator`], plus its
//! [`descriptor`]/[`load`] entry points and the `inventory` registration that wires it into
//! `mlx_gen`'s registry under the id `"sdxl"` (the SceneWorks worker's `payload.model`).
//!
//! SDXL is the in-process Apple `mlx-examples/stable_diffusion` path (vendored at
//! `_vendor/mlx_sd/`) brought into Rust — a **U-Net** generator (conv ResBlocks + spatial/cross
//! attention + time/`text_time` micro-conditioning), dual CLIP text encoders, an SDXL VAE, and a
//! discrete Euler-Ancestral sampler with real classifier-free guidance. Parity target = the
//! vendored fp16 reference path (`StableDiffusionXL.generate_latents`), validated stage-by-stage.
//!
//! Slices land incrementally (sc-2400): this module starts as the contract + capability surface;
//! [`load`] assembles components as each slice (tokenizer → text encoders → U-Net → VAE → sampler)
//! is wired and parity-proven.

use mlx_gen::{
    default_seed, Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, Precision, Progress,
    Result, WeightsSource,
};

use crate::config::DiffusionConfig;
use crate::loader;
use crate::pipeline::{
    decode_image, denoise, encode_conditioning, encode_init_latents, text_time_ids, Denoiser,
};
use crate::sampler::EulerSampler;
use crate::text_encoder::ClipTextEncoder;
use crate::tokenizer::ClipBpeTokenizer;
use crate::unet::UNet2DConditionModel;
use crate::vae::Autoencoder;

/// img2img default strength (the vendored `generate_latents_from_image` default).
const DEFAULT_STRENGTH: f32 = 0.8;

/// SDXL-base-1.0 production defaults (the SceneWorks `MlxSdxlAdapter`): 30 inference steps,
/// CFG 7.0, native 1024². Used when a request omits the corresponding field (consumed by the
/// `generate` pipeline slice, sc-2400 S5).
#[allow(dead_code)]
pub(crate) const DEFAULT_STEPS: u32 = 30;
#[allow(dead_code)]
pub(crate) const DEFAULT_GUIDANCE: f32 = 7.0;

/// Registry id — matches the SceneWorks worker's `payload.model` (`MODEL_TARGETS["sdxl"]`).
pub const MODEL_ID: &str = "sdxl";

/// SDXL's identity + capabilities — constructible without loading weights (registry
/// introspection). Capability flags are turned on as each slice lands and is parity-proven, so the
/// descriptor never advertises a path that isn't wired (avoids the false-capability trap —
/// [[false-green-gates-mask-descope]]).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "sdxl",
        modality: Modality::Image,
        capabilities: Capabilities {
            // SDXL uses real classifier-free guidance: honors the negative prompt + a CFG scale.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            // img2img Reference (sc-2638). LoRA/LoKr land in sc-2639/sc-2640 (advertised once wired).
            conditioning: vec![ConditioningKind::Reference],
            supports_lora: false,
            supports_lokr: false,
            // Only the wired + parity-proven sampler is advertised. The fork's SDXL path uses the
            // ancestral Euler step exclusively (there is no plain-`euler` SDXL golden), so a request
            // naming any other sampler is rejected in `validate_request` rather than silently
            // downgraded — same no-false-capability principle as the rest of this descriptor.
            samplers: vec!["euler_ancestral"],
            schedulers: vec!["discrete"],
            min_size: 512,
            max_size: 2048,
            max_count: 8,
            mac_only: true,
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// A loaded SDXL generator: the dual CLIP encoders + tokenizer, the U-Net, the VAE, and the
/// Euler-Ancestral sampler, assembled from a snapshot directory.
pub struct Sdxl {
    descriptor: ModelDescriptor,
    tokenizer: ClipBpeTokenizer,
    te1: ClipTextEncoder,
    te2: ClipTextEncoder,
    unet: UNet2DConditionModel,
    vae: Autoencoder,
    sampler: EulerSampler,
}

/// Construct an [`Sdxl`] from a [`LoadSpec`].
///
/// `spec.weights` must be a [`WeightsSource::Dir`] pointing at a
/// `stabilityai/stable-diffusion-xl-base-1.0` snapshot (the diffusers multi-component tree —
/// `tokenizer/`, `tokenizer_2/`, `text_encoder/`, `text_encoder_2/`, `unet/`, `vae/`). All weights
/// load + run f32.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        // The validated dense path runs f32 activations over f32 weights (sidesteps the pmetal
        // 16-bit dense-GEMM bug and is the f32-quality target); an fp32 precision override is the
        // default behaviour, not a separate mode, and a non-default precision flag is rejected
        // rather than silently ignored.
        return Err(Error::Msg(
            "sdxl: precision override is not wired; the dense path already runs f32 activations \
             (drop the precision override)"
                .into(),
        ));
    }
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p,
            WeightsSource::File(_) => return Err(Error::Msg(
                "sdxl expects a snapshot directory (tokenizer/ text_encoder/ unet/ vae/ …), not a \
                 single .safetensors file"
                    .into(),
            )),
        };
    if !spec.adapters.is_empty() {
        // LoRA/LoKr land in sc-2639/sc-2640 (the SDXL key→module map). Surface rather than
        // silently ignore a requested adapter.
        return Err(Error::Msg(
            "sdxl: LoRA/LoKr adapters are not yet wired (sc-2639/sc-2640)".into(),
        ));
    }
    if spec.quantize.is_some() {
        // Q4/Q8 parity for SDXL is validated + scoped in sc-2641 (the sc-1975 base-1.0 Q8 caveat
        // needs a dedicated check); don't ship an unvalidated quantized path.
        return Err(Error::Msg(
            "sdxl: Q4/Q8 quantization is not yet validated (sc-2641)".into(),
        ));
    }

    Ok(Box::new(Sdxl {
        descriptor: descriptor(),
        tokenizer: loader::load_tokenizer(root)?,
        te1: loader::load_text_encoder_1(root)?,
        te2: loader::load_text_encoder_2(root)?,
        unet: loader::load_unet(root)?,
        vae: loader::load_vae(root)?,
        sampler: EulerSampler::new(&DiffusionConfig::sdxl_base(), true),
    }))
}

impl Generator for Sdxl {
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
        let cfg = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
        let cfg_on = cfg > 1.0;
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let reference = self.resolve_reference(req)?;
        let max_time = self.sampler.max_time();

        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            // One image per iteration (the vendored `_run_one`, n_images=1), each with its own seed.
            let seed = base_seed.wrapping_add(i as u64);
            // Seed the global RNG up front. Conditioning + VAE-encode draw no RNG, so the first draw
            // is the init noise (txt2img prior / img2img add_noise) — matching the reference stream.
            mlx_rs::random::seed(seed)?;

            let tokens = self
                .tokenizer
                .tokenize_batch(&req.prompt, if cfg_on { Some(negative) } else { None })?;
            let (conditioning, pooled) = encode_conditioning(&self.te1, &self.te2, &tokens)?;
            let time_ids = text_time_ids(pooled.shape()[0]);

            // Init latents + the denoise start time/step count.
            let (latents, start_time, eff_steps) = match reference {
                Some((image, strength)) => {
                    // img2img (the vendored `generate_latents_from_image`): VAE-encode the init image,
                    // start at `max_time·strength`, run `int(steps·strength)` steps. Higher strength →
                    // later start → fewer steps → output stays closer to the init.
                    let strength = strength.unwrap_or(DEFAULT_STRENGTH).clamp(0.0, 1.0);
                    let x_0 = encode_init_latents(&self.vae, image, req.width, req.height)?;
                    let start_step = max_time * strength;
                    let x_t = self.sampler.add_noise(&x_0, start_step)?;
                    // Faithful to the reference's `int(num_steps · strength)` — NO min-1 floor.
                    // strength ≤ 1/steps ⇒ 0 steps ⇒ the init latents are returned unchanged. A floor
                    // here would force a denoise step at start_time 0, where σ = 0 makes the ancestral
                    // `σ_up = sqrt(σ_prev²·(σ²−σ_prev²)/σ²)` divide 0/0 → NaN (a real strength=0 bug).
                    let eff = (steps as f32 * strength) as usize;
                    (x_t, start_step, eff)
                }
                None => {
                    // txt2img: seeded prior.
                    let prior = self.sampler.sample_prior(&[
                        1,
                        (req.height / 8) as i32,
                        (req.width / 8) as i32,
                        4,
                    ])?;
                    (prior, max_time, steps)
                }
            };

            let d = Denoiser {
                unet: &self.unet,
                sampler: &self.sampler,
            };
            let latents = denoise(
                &d,
                latents,
                &conditioning,
                &pooled,
                &time_ids,
                eff_steps,
                start_time,
                cfg,
                &req.cancel,
                on_progress,
            )?;

            on_progress(Progress::Decoding);
            images.push(decode_image(&self.vae, &latents)?);
        }
        Ok(GenerationOutput::Images(images))
    }
}

impl Sdxl {
    /// Extract the single img2img init image + its strength from the request's conditioning (the
    /// per-reference strength wins over `req.strength`). SDXL img2img conditions on exactly one init
    /// image, so more than one `Reference` is an error.
    fn resolve_reference<'a>(
        &self,
        req: &'a GenerationRequest,
    ) -> Result<Option<(&'a Image, Option<f32>)>> {
        let mut reference = None;
        for c in &req.conditioning {
            if let Conditioning::Reference { image, strength } = c {
                if reference.is_some() {
                    return Err(Error::Msg(
                        "sdxl: multiple reference images are not supported (single img2img init only)"
                            .into(),
                    ));
                }
                reference = Some((image, strength.or(req.strength)));
            }
        }
        Ok(reference)
    }
}

/// Capability-driven request validation, factored out so it can be unit-tested without loaded
/// weights. Rejects unsupported guidance / negative prompt / conditioning / size / count.
pub(crate) fn validate_request(caps: &Capabilities, req: &GenerationRequest) -> Result<()> {
    if req.prompt.is_empty() {
        return Err(Error::Msg("sdxl: prompt must not be empty".into()));
    }
    if req.count == 0 || req.count > caps.max_count {
        return Err(Error::Msg(format!(
            "count {} out of range 1..={}",
            req.count, caps.max_count
        )));
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
    // SDXL works in latent space at /8; both dims must be multiples of 8.
    if req.width % 8 != 0 || req.height % 8 != 0 {
        return Err(Error::Msg(format!(
            "sdxl: width/height must be multiples of 8 (got {}x{})",
            req.width, req.height
        )));
    }
    if req.guidance.is_some() && !caps.supports_guidance {
        return Err(Error::Msg(
            "sdxl: `guidance` is not supported by this build".into(),
        ));
    }
    if req.negative_prompt.is_some() && !caps.supports_negative_prompt {
        return Err(Error::Msg(
            "sdxl: negative prompt is not supported by this build".into(),
        ));
    }
    // Reject an unsupported sampler instead of silently downgrading it to the ancestral default.
    if let Some(s) = &req.sampler {
        if !caps.samplers.contains(&s.as_str()) {
            return Err(Error::Msg(format!(
                "sdxl: unsupported sampler {s:?} (supported: {:?})",
                caps.samplers
            )));
        }
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
            return Err(Error::Msg(format!(
                "sdxl does not accept {kind:?} conditioning"
            )));
        }
    }
    Ok(())
}

inventory::submit! {
    ModelRegistration { descriptor, load }
}

use mlx_gen::ModelRegistration;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_is_sdxl() {
        let d = descriptor();
        assert_eq!(d.id, "sdxl");
        assert_eq!(d.family, "sdxl");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
    }

    #[test]
    fn registered_in_core_registry() {
        // Linking this crate must self-register the model (inventory link-time collection).
        assert!(
            mlx_gen::registry::generators().any(|r| (r.descriptor)().id == "sdxl"),
            "sdxl is not registered in mlx_gen's generator registry"
        );
    }

    #[test]
    fn validate_rejects_empty_prompt() {
        let caps = descriptor().capabilities;
        let req = GenerationRequest::default(); // default prompt is empty
        let err = validate_request(&caps, &req).unwrap_err().to_string();
        assert!(err.contains("empty"), "got: {err}");
    }

    #[test]
    fn validate_accepts_cfg_and_negative_prompt_rejects_bad_size() {
        let caps = descriptor().capabilities;
        // Real CFG + negative prompt are supported.
        let mut req = GenerationRequest {
            prompt: "a fox".into(),
            guidance: Some(7.0),
            negative_prompt: Some("blurry".into()),
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_ok());
        // Non-multiple-of-8 size is rejected.
        req = GenerationRequest {
            prompt: "a fox".into(),
            width: 1020,
            height: 1024,
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_err());
        // Out-of-range size is rejected.
        req = GenerationRequest {
            prompt: "a fox".into(),
            width: 256,
            height: 256,
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_err());
    }

    #[test]
    fn validate_sampler_selection() {
        let caps = descriptor().capabilities;
        let base = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        // The wired sampler is accepted; an unset sampler is accepted (defaults to ancestral).
        assert!(validate_request(&caps, &base).is_ok());
        assert!(validate_request(
            &caps,
            &GenerationRequest {
                sampler: Some("euler_ancestral".into()),
                ..base.clone()
            }
        )
        .is_ok());
        // `euler` (and any unknown sampler) is rejected, not silently downgraded.
        for bad in ["euler", "ddim", "nonsense"] {
            let err = validate_request(
                &caps,
                &GenerationRequest {
                    sampler: Some(bad.into()),
                    ..base.clone()
                },
            )
            .unwrap_err()
            .to_string();
            assert!(err.contains("unsupported sampler"), "got: {err}");
        }
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/sdxl.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }
}
