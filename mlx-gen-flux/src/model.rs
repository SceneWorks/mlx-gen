//! FLUX.1 provider registration and txt2img generation path.

use mlx_gen::image::decoded_to_image;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    default_seed, DiffusionSampler, Error, FlowMatchSampler, GenerationOutput, GenerationRequest,
    Generator, LoadSpec, ModelDescriptor, ModelRegistration, Precision, Progress, Result,
    WeightsSource,
};
use mlx_gen_z_image::vae::Vae;
use mlx_rs::Dtype;

use crate::config::{FluxVariant, DEFAULT_SAMPLER, HYPER_SAMPLER};
use crate::loader;
use crate::pipeline::{build_linear_sigmas, create_noise, unpack_latents};
use crate::text_encoder::FluxTextEncoders;
use crate::transformer::FluxTransformer;

pub fn descriptor_schnell() -> ModelDescriptor {
    descriptor_for(FluxVariant::Schnell)
}

pub fn descriptor_dev() -> ModelDescriptor {
    descriptor_for(FluxVariant::Dev)
}

pub fn descriptor_for(variant: FluxVariant) -> ModelDescriptor {
    variant.descriptor()
}

pub fn load_schnell(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(FluxVariant::Schnell, spec)
}

pub fn load_dev(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(FluxVariant::Dev, spec)
}

fn load_variant(variant: FluxVariant, spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{}: only dense bf16 is wired for the FLUX.1 port plan",
            variant.id()
        )));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(Error::Msg(format!(
                "{} expects a FLUX.1 snapshot directory (tokenizer/ tokenizer_2/ text_encoder/ \
                 text_encoder_2/ transformer/ vae/), not a single .safetensors file",
                variant.id()
            )))
        }
    };

    let t5_tokenizer = loader::load_t5_tokenizer(root, variant)?;
    let clip_tokenizer = loader::load_clip_tokenizer(root)?;
    let mut text_encoders = FluxTextEncoders {
        t5: loader::load_t5_encoder(root)?,
        clip: loader::load_clip_encoder(root)?,
    };
    let mut transformer = loader::load_transformer(root, variant)?;
    let mut vae = loader::load_vae(root)?;
    if let Some(q) = spec.quantize {
        let bits = q.bits();
        text_encoders.quantize(bits)?;
        transformer.quantize(bits)?;
        vae.quantize(bits)?;
    }
    // Install LoRA/LoKr adapters AFTER quantization (the fork merges/applies post-quantize too; a
    // forward-time residual over the now-quantized base, never a fused merge). No-op when empty; a
    // non-empty spec list that matches nothing — or any unmatched target — errors loudly (sc-2534).
    crate::adapters::apply_flux_adapters(&mut transformer, &spec.adapters)?;

    Ok(Box::new(Flux1 {
        descriptor: descriptor_for(variant),
        variant,
        t5_tokenizer: Some(t5_tokenizer),
        clip_tokenizer: Some(clip_tokenizer),
        text_encoders: Some(text_encoders),
        transformer: Some(transformer),
        vae: Some(vae),
    }))
}

pub struct Flux1 {
    descriptor: ModelDescriptor,
    variant: FluxVariant,
    t5_tokenizer: Option<TextTokenizer>,
    clip_tokenizer: Option<TextTokenizer>,
    text_encoders: Option<FluxTextEncoders>,
    transformer: Option<FluxTransformer>,
    vae: Option<Vae>,
}

impl Flux1 {
    pub fn new_for_tests(variant: FluxVariant) -> Self {
        Self {
            descriptor: descriptor_for(variant),
            variant,
            t5_tokenizer: None,
            clip_tokenizer: None,
            text_encoders: None,
            transformer: None,
            vae: None,
        }
    }

    pub fn encode_prompt(&self, prompt: &str) -> Result<(mlx_rs::Array, mlx_rs::Array)> {
        let t5_tokenizer = self.t5_tokenizer.as_ref().ok_or_else(|| {
            Error::Msg(format!(
                "{}: T5 tokenizer is not loaded in this test-only instance",
                self.descriptor.id
            ))
        })?;
        let clip_tokenizer = self.clip_tokenizer.as_ref().ok_or_else(|| {
            Error::Msg(format!(
                "{}: CLIP tokenizer is not loaded in this test-only instance",
                self.descriptor.id
            ))
        })?;
        let text_encoders = self.text_encoders.as_ref().ok_or_else(|| {
            Error::Msg(format!(
                "{}: text encoders are not loaded in this test-only instance",
                self.descriptor.id
            ))
        })?;
        let t5 = t5_tokenizer.tokenize(prompt)?;
        let clip = clip_tokenizer.tokenize(prompt)?;
        text_encoders.encode(&t5.input_ids, &clip.input_ids)
    }

    fn transformer(&self) -> Result<&FluxTransformer> {
        self.transformer.as_ref().ok_or_else(|| {
            Error::Msg(format!(
                "{}: transformer is not loaded in this test-only instance",
                self.descriptor.id
            ))
        })
    }

    fn vae(&self) -> Result<&Vae> {
        self.vae.as_ref().ok_or_else(|| {
            Error::Msg(format!(
                "{}: VAE is not loaded in this test-only instance",
                self.descriptor.id
            ))
        })
    }
}

impl Generator for Flux1 {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> Result<()> {
        validate_request(&self.descriptor, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        let transformer = self.transformer()?;
        let vae = self.vae()?;
        let base_seed = req.seed.unwrap_or_else(default_seed);
        // Sampler selection (sc-2908). FLUX is flow-match: the base render and the few-step `hyper`
        // profile share the SAME flow-match schedule (mflux's `LinearScheduler`) — `hyper` only
        // changes the default step count + guidance (the acceleration is a distilled LoRA the caller
        // loads at `scale≈0.125` via `spec.adapters`, not a different scheduler). An unset sampler is
        // the base flow-match path; `validate_request` rejects any name not in the descriptor.
        let sampler_name = req.sampler.as_deref().unwrap_or(DEFAULT_SAMPLER);
        let (def_steps, def_guidance) = profile_defaults(self.variant, sampler_name);
        let steps = req.steps.unwrap_or(def_steps) as usize;
        let guidance = if self.variant.supports_guidance() {
            req.guidance.unwrap_or(def_guidance)
        } else {
            0.0
        };
        // The FLUX diffusion path is MIXED precision, matching the mflux reference (sc-2787, verified
        // against the bf16 golden's per-tensor dtypes): the latents (`create_noise` → f32) and the
        // main residual stream stay f32 — the fork's scheduler casts the noise prediction to
        // `latents.dtype` (f32) and its T5 `prompt_embeds` is f32 (T5LayerNorm upcast). Only the CLIP
        // pooled embedding and the time/text/guidance conditioning run bf16 (handled in the encoders
        // and `TimeTextEmbed`). So latents are NOT cast to bf16 here — that would diverge from the
        // fork. (The old "f32 everywhere to dodge the x_embedder bf16 GEMM bug" is obsolete: that bug
        // is fixed by sc-2772, and the fork runs the x_embedder in f32 anyway because latents are f32.)
        let (prompt_embeds, pooled_prompt_embeds) = self.encode_prompt(&req.prompt)?;
        let sigmas = build_linear_sigmas(
            steps,
            req.width,
            req.height,
            self.variant.requires_sigma_shift(),
        )?;
        // Drive the denoise through the swappable `DiffusionSampler` seam (sc-2769). FLUX's impl is the
        // flow-match Euler sampler over these sigmas: `scale_model_input` is identity, `timestep(t)` is
        // `sigmas[t]` (fed straight to the transformer), and `step` is `x + v·(σ_{t+1}−σ_t)` — exactly
        // the proven inline loop, so the base render stays bit-exact (guarded by the e2e parity test).
        let sampler = FlowMatchSampler::new(sigmas);
        let n_steps = sampler.num_steps();

        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            let seed = base_seed.wrapping_add(i as u64);
            let mut latents = create_noise(seed, req.width, req.height)?;
            for t in 0..n_steps {
                if req.cancel.is_cancelled() {
                    return Err(Error::Msg("generation cancelled".into()));
                }
                let x_in = sampler.scale_model_input(&latents, t)?;
                let velocity = transformer.forward(
                    &x_in,
                    &prompt_embeds,
                    &pooled_prompt_embeds,
                    sampler.timestep(t),
                    guidance,
                    req.width,
                    req.height,
                )?;
                latents = sampler.step(&velocity, &latents, t)?;
                on_progress(Progress::Step {
                    current: t as u32 + 1,
                    total: n_steps as u32,
                });
            }

            on_progress(Progress::Decoding);
            let unpacked = unpack_latents(&latents, req.width, req.height)?;
            let decoded = vae.decode(&unpacked)?.as_dtype(Dtype::Float32)?;
            images.push(decoded_to_image(&decoded)?);
        }
        Ok(GenerationOutput::Images(images))
    }
}

/// Few-step profile defaults `(steps, guidance)` applied when the request omits them (sc-2908). The
/// base flow-match path uses the variant's own defaults; the `hyper` profile (Hyper-FLUX.1-dev) is 8
/// steps at guidance 3.5 — paired with the ByteDance Hyper-FLUX 8-step LoRA loaded at `scale≈0.125`
/// (the documented `lora_scale`) via `spec.adapters`. `hyper` is dev-only (it is a FLUX.1-dev LoRA)
/// and schnell never advertises it, so it never reaches here for schnell.
fn profile_defaults(variant: FluxVariant, sampler: &str) -> (u32, f32) {
    match sampler {
        HYPER_SAMPLER => (8, crate::config::DEFAULT_GUIDANCE),
        _ => (variant.default_steps(), crate::config::DEFAULT_GUIDANCE),
    }
}

fn validate_request(desc: &ModelDescriptor, req: &GenerationRequest) -> Result<()> {
    if req.prompt.trim().is_empty() {
        return Err(Error::Msg(format!("{}: prompt is required", desc.id)));
    }
    // Reject a sampler the variant does not advertise (e.g. `hyper` on schnell, or any typo) rather
    // than silently falling back to the base flow-match path.
    if let Some(s) = &req.sampler {
        if !desc.capabilities.samplers.contains(&s.as_str()) {
            return Err(Error::Msg(format!(
                "{}: unsupported sampler {s:?} (supported: {:?})",
                desc.id, desc.capabilities.samplers
            )));
        }
    }
    if !req.width.is_multiple_of(16) || !req.height.is_multiple_of(16) {
        return Err(Error::Msg(format!(
            "{}: width and height must be multiples of 16, got {}x{}",
            desc.id, req.width, req.height
        )));
    }
    let caps = &desc.capabilities;
    if req.width < caps.min_size
        || req.height < caps.min_size
        || req.width > caps.max_size
        || req.height > caps.max_size
    {
        return Err(Error::Msg(format!(
            "{}: size {}x{} outside supported range {}..={}",
            desc.id, req.width, req.height, caps.min_size, caps.max_size
        )));
    }
    if req.count == 0 || req.count > caps.max_count {
        return Err(Error::Msg(format!(
            "{}: count must be 1..={}",
            desc.id, caps.max_count
        )));
    }
    if req.negative_prompt.is_some() && !caps.supports_negative_prompt {
        return Err(Error::Msg(format!(
            "{}: negative prompts are not supported",
            desc.id
        )));
    }
    if req.guidance.is_some() && !caps.supports_guidance {
        return Err(Error::Msg(format!(
            "{}: guidance is not supported by this distilled variant",
            desc.id
        )));
    }
    if req.true_cfg.is_some() && !caps.supports_true_cfg {
        return Err(Error::Msg(format!(
            "{}: true_cfg is not supported",
            desc.id
        )));
    }
    if !req.conditioning.is_empty() {
        return Err(Error::Msg(format!(
            "{}: conditioning variants are not implemented in the base txt2img port yet",
            desc.id
        )));
    }
    Ok(())
}

inventory::submit! {
    ModelRegistration { descriptor: descriptor_schnell, load: load_schnell }
}

inventory::submit! {
    ModelRegistration { descriptor: descriptor_dev, load: load_dev }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FLUX1_DEV_ID, FLUX1_SCHNELL_ID};

    #[test]
    fn validates_base_txt2img_request() {
        let model = Flux1::new_for_tests(FluxVariant::Dev);
        let req = GenerationRequest {
            prompt: "a red fox".into(),
            guidance: Some(3.5),
            ..Default::default()
        };
        model.validate(&req).unwrap();
    }

    #[test]
    fn schnell_rejects_guidance() {
        let model = Flux1::new_for_tests(FluxVariant::Schnell);
        let req = GenerationRequest {
            prompt: "a red fox".into(),
            guidance: Some(3.5),
            ..Default::default()
        };
        let err = model.validate(&req).unwrap_err().to_string();
        assert!(err.contains("guidance is not supported"));
    }

    #[test]
    fn rejects_non_multiple_of_16() {
        let model = Flux1::new_for_tests(FluxVariant::Dev);
        let req = GenerationRequest {
            prompt: "a red fox".into(),
            width: 1025,
            ..Default::default()
        };
        let err = model.validate(&req).unwrap_err().to_string();
        assert!(err.contains("multiples of 16"));
    }

    #[test]
    fn constants_match_expected_ids() {
        assert_eq!(FluxVariant::Schnell.id(), FLUX1_SCHNELL_ID);
        assert_eq!(FluxVariant::Dev.id(), FLUX1_DEV_ID);
    }

    // ---- sc-2908: sampler capability surface + few-step profile -----------------------------

    #[test]
    fn dev_advertises_hyper_schnell_does_not() {
        // Hyper-FLUX is a FLUX.1-dev LoRA: dev exposes the base + `hyper` samplers; schnell (already
        // a distilled 4-step checkpoint) exposes only the base flow-match sampler.
        let dev = descriptor_for(FluxVariant::Dev).capabilities.samplers;
        assert_eq!(dev, vec![DEFAULT_SAMPLER, HYPER_SAMPLER]);
        let schnell = descriptor_for(FluxVariant::Schnell).capabilities.samplers;
        assert_eq!(schnell, vec![DEFAULT_SAMPLER]);
    }

    #[test]
    fn validate_accepts_base_and_hyper_on_dev() {
        let model = Flux1::new_for_tests(FluxVariant::Dev);
        for s in [DEFAULT_SAMPLER, HYPER_SAMPLER] {
            let req = GenerationRequest {
                prompt: "a red fox".into(),
                guidance: Some(3.5),
                sampler: Some(s.into()),
                ..Default::default()
            };
            assert!(model.validate(&req).is_ok(), "sampler {s:?} should be accepted on dev");
        }
        // An unset sampler is the base flow-match path.
        let req = GenerationRequest {
            prompt: "a red fox".into(),
            guidance: Some(3.5),
            ..Default::default()
        };
        assert!(model.validate(&req).is_ok());
    }

    #[test]
    fn validate_rejects_hyper_on_schnell_and_unknown_samplers() {
        // `hyper` is dev-only — schnell does not advertise it, so it is rejected, not downgraded.
        let schnell = Flux1::new_for_tests(FluxVariant::Schnell);
        let err = schnell
            .validate(&GenerationRequest {
                prompt: "a red fox".into(),
                sampler: Some(HYPER_SAMPLER.into()),
                ..Default::default()
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("unsupported sampler"), "got: {err}");
        // Any unknown sampler name is rejected on dev too.
        let dev = Flux1::new_for_tests(FluxVariant::Dev);
        for bad in ["lcm", "lightning", "euler", "nonsense"] {
            let err = dev
                .validate(&GenerationRequest {
                    prompt: "a red fox".into(),
                    guidance: Some(3.5),
                    sampler: Some(bad.into()),
                    ..Default::default()
                })
                .unwrap_err()
                .to_string();
            assert!(err.contains("unsupported sampler"), "sampler {bad:?}: {err}");
        }
    }

    #[test]
    fn hyper_profile_defaults_are_eight_steps_guidance_3_5() {
        // The few-step profile: 8 steps at guidance 3.5 (the Hyper-FLUX.1-dev recommendation).
        assert_eq!(profile_defaults(FluxVariant::Dev, HYPER_SAMPLER), (8, 3.5));
        // The base path keeps the variant's own defaults (dev 25, schnell 4).
        assert_eq!(
            profile_defaults(FluxVariant::Dev, DEFAULT_SAMPLER),
            (FluxVariant::Dev.default_steps(), crate::config::DEFAULT_GUIDANCE)
        );
        assert_eq!(
            profile_defaults(FluxVariant::Schnell, DEFAULT_SAMPLER),
            (FluxVariant::Schnell.default_steps(), crate::config::DEFAULT_GUIDANCE)
        );
    }
}
