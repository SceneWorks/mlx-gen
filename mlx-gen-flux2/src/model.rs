//! FLUX.2-klein provider registration + the txt2img generation path.
//!
//! `load()` assembles the tokenizer, Qwen3 text encoder, MMDiT transformer, and 32-ch VAE from a
//! snapshot directory. `generate()` runs the flow-match denoise loop (CFG dual-forward when
//! `guidance > 1`; distilled klein defaults to 1.0 = single forward), then BN-denormalizes +
//! 2×2-unpatchifies + VAE-decodes. Edit (`flux2_klein_9b_edit`) generation lands in S5.
//!
//! Activations run f32 (matmul(f32, bf16)→f32): dodges the dense 16-bit Metal GEMM bug and is the
//! quality target. Pixel-parity with the fork's bf16 render is therefore not the gate (see the
//! e2e test) — component f32 parity + visual correctness is.

use mlx_gen::array::scalar;
use mlx_gen::image::decoded_to_image;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    default_seed, Error, GenerationOutput, GenerationRequest, Generator, LoadSpec, ModelDescriptor,
    ModelRegistration, Precision, Progress, Result, WeightsSource,
};
use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::Array;

use crate::config::{Flux2Variant, DEFAULT_GUIDANCE};
use crate::pipeline::{
    create_noise, prepare_grid_ids, prepare_text_ids, schedule, timesteps_x1000,
};
use crate::text_encoder::Qwen3TextEncoder;
use crate::transformer::Flux2Transformer;
use crate::vae::Flux2Vae;
use crate::{loader, Flux2Config};

pub fn descriptor_klein_9b() -> ModelDescriptor {
    Flux2Variant::Klein9b.descriptor()
}

pub fn descriptor_klein_9b_edit() -> ModelDescriptor {
    Flux2Variant::Klein9bEdit.descriptor()
}

pub fn load_klein_9b(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(Flux2Variant::Klein9b, spec)
}

pub fn load_klein_9b_edit(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(Flux2Variant::Klein9bEdit, spec)
}

fn load_variant(variant: Flux2Variant, spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{}: only dense bf16 is wired (Q4/Q8 = sc-2643)",
            variant.id()
        )));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(Error::Msg(format!(
                "{} expects a FLUX.2-klein snapshot directory (tokenizer/ text_encoder/ \
                 transformer/ vae/), not a single .safetensors file",
                variant.id()
            )))
        }
    };
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(format!(
            "{}: LoRA/LoKr adapters are sc-2646",
            variant.id()
        )));
    }
    if spec.quantize.is_some() {
        return Err(Error::Msg(format!(
            "{}: Q4/Q8 quantization is sc-2643",
            variant.id()
        )));
    }

    Ok(Box::new(Flux2 {
        descriptor: variant.descriptor(),
        variant,
        config: variant.config(),
        tokenizer: Some(loader::load_tokenizer(root)?),
        text_encoder: Some(loader::load_text_encoder(root)?),
        transformer: Some(loader::load_transformer(root)?),
        vae: Some(loader::load_vae(root)?),
    }))
}

/// The FLUX.2-klein generator.
pub struct Flux2 {
    descriptor: ModelDescriptor,
    variant: Flux2Variant,
    config: Flux2Config,
    tokenizer: Option<TextTokenizer>,
    text_encoder: Option<Qwen3TextEncoder>,
    transformer: Option<Flux2Transformer>,
    vae: Option<Flux2Vae>,
}

impl Flux2 {
    /// Construct a weightless instance for validation tests.
    pub fn new_for_tests(variant: Flux2Variant) -> Self {
        Self {
            descriptor: variant.descriptor(),
            variant,
            config: variant.config(),
            tokenizer: None,
            text_encoder: None,
            transformer: None,
            vae: None,
        }
    }

    fn parts(
        &self,
    ) -> Result<(
        &TextTokenizer,
        &Qwen3TextEncoder,
        &Flux2Transformer,
        &Flux2Vae,
    )> {
        let err = |what: &str| Error::Msg(format!("{}: {what} is not loaded", self.descriptor.id));
        Ok((
            self.tokenizer.as_ref().ok_or_else(|| err("tokenizer"))?,
            self.text_encoder
                .as_ref()
                .ok_or_else(|| err("text encoder"))?,
            self.transformer
                .as_ref()
                .ok_or_else(|| err("transformer"))?,
            self.vae.as_ref().ok_or_else(|| err("VAE"))?,
        ))
    }

    /// Encode a prompt → `(prompt_embeds [1,512,joint], text_ids [1,512,4])`.
    fn encode(
        &self,
        tokenizer: &TextTokenizer,
        te: &Qwen3TextEncoder,
        prompt: &str,
    ) -> Result<(Array, Array)> {
        let tok = tokenizer.tokenize(prompt)?;
        let embeds = te.prompt_embeds(&tok.input_ids, &tok.attention_mask)?;
        let ids = prepare_text_ids(embeds.shape()[1] as usize);
        Ok((embeds, ids))
    }
}

impl Generator for Flux2 {
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
        if self.variant.is_edit() {
            return Err(Error::Msg(format!(
                "{}: image-conditioned edit generation is S5",
                self.descriptor.id
            )));
        }
        let (tokenizer, te, transformer, vae) = self.parts()?;
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let steps = req.steps.unwrap_or(crate::config::DEFAULT_STEPS) as usize;
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);

        let (prompt_embeds, text_ids) = self.encode(tokenizer, te, &req.prompt)?;
        // klein is distilled (guidance 1.0); CFG dual-forward only kicks in for base variants.
        let negative = if guidance > 1.0 {
            Some(self.encode(tokenizer, te, " ")?)
        } else {
            None
        };

        let sched = schedule(steps, req.width, req.height);
        let timesteps = timesteps_x1000(&sched);
        let lat_h = (req.height / 16) as usize;
        let lat_w = (req.width / 16) as usize;
        let latent_ids = prepare_grid_ids(lat_h, lat_w, 0);
        let in_channels = self.config.in_channels as i32;

        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            let seed = base_seed.wrapping_add(i as u64);
            let mut latents = create_noise(seed, req.width, req.height, self.config.in_channels)?;
            for (t, &ts) in timesteps.iter().enumerate() {
                if req.cancel.is_cancelled() {
                    return Err(Error::Msg("generation cancelled".into()));
                }
                let v =
                    transformer.forward(&latents, &prompt_embeds, &latent_ids, &text_ids, ts)?;
                let v = match &negative {
                    Some((neg_embeds, neg_ids)) => {
                        let vn =
                            transformer.forward(&latents, neg_embeds, &latent_ids, neg_ids, ts)?;
                        // noise = neg + guidance·(pos − neg)
                        add(&vn, &multiply(&subtract(&v, &vn)?, scalar(guidance))?)?
                    }
                    None => v,
                };
                latents = sched.step(&latents, &v, t)?;
                on_progress(Progress::Step {
                    current: t as u32 + 1,
                    total: steps as u32,
                });
            }
            on_progress(Progress::Decoding);
            let packed = latents.reshape(&[1, lat_h as i32, lat_w as i32, in_channels])?;
            let decoded = vae.decode_packed_latents(&packed)?; // NHWC [1,H,W,3]
            let nchw = decoded.transpose_axes(&[0, 3, 1, 2])?;
            images.push(decoded_to_image(&nchw)?);
        }
        Ok(GenerationOutput::Images(images))
    }
}

fn validate_request(desc: &ModelDescriptor, req: &GenerationRequest) -> Result<()> {
    if req.prompt.trim().is_empty() {
        return Err(Error::Msg(format!("{}: prompt is required", desc.id)));
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
            "{}: negative prompts are not supported by FLUX.2",
            desc.id
        )));
    }
    if req.true_cfg.is_some() && !caps.supports_true_cfg {
        return Err(Error::Msg(format!(
            "{}: true_cfg is not supported",
            desc.id
        )));
    }
    for c in &req.conditioning {
        let kind = conditioning_kind(c);
        if !caps.accepts(kind) {
            return Err(Error::Msg(format!(
                "{}: conditioning {kind:?} is not supported by this variant",
                desc.id
            )));
        }
    }
    Ok(())
}

fn conditioning_kind(c: &mlx_gen::Conditioning) -> mlx_gen::ConditioningKind {
    use mlx_gen::{Conditioning as C, ConditioningKind as K};
    match c {
        C::Reference { .. } => K::Reference,
        C::MultiReference { .. } => K::MultiReference,
        C::ReduxRefs { .. } => K::ReduxRefs,
        C::Control { .. } => K::Control,
        C::Depth { .. } => K::Depth,
        C::Mask { .. } => K::Mask,
    }
}

inventory::submit! {
    ModelRegistration { descriptor: descriptor_klein_9b, load: load_klein_9b }
}

inventory::submit! {
    ModelRegistration { descriptor: descriptor_klein_9b_edit, load: load_klein_9b_edit }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FLUX2_KLEIN_9B_EDIT_ID, FLUX2_KLEIN_9B_ID};
    use mlx_gen::media::Image;
    use mlx_gen::Conditioning;

    #[test]
    fn validates_basic_txt2img_request() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest {
            prompt: "a hummingbird".into(),
            ..Default::default()
        };
        model.validate(&req).unwrap();
    }

    #[test]
    fn rejects_empty_prompt() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest::default();
        let err = model.validate(&req).unwrap_err().to_string();
        assert!(err.contains("prompt is required"));
    }

    #[test]
    fn rejects_non_multiple_of_16() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest {
            prompt: "x".into(),
            width: 1023,
            ..Default::default()
        };
        let err = model.validate(&req).unwrap_err().to_string();
        assert!(err.contains("multiples of 16"));
    }

    #[test]
    fn txt2img_rejects_reference_conditioning() {
        // img2img (Reference) is sc-2644, not this story's txt2img variant.
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest {
            prompt: "x".into(),
            conditioning: vec![Conditioning::Reference {
                image: Image::default(),
                strength: None,
            }],
            ..Default::default()
        };
        let err = model.validate(&req).unwrap_err().to_string();
        assert!(err.contains("conditioning"));
    }

    #[test]
    fn edit_accepts_single_reference() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9bEdit);
        let req = GenerationRequest {
            prompt: "make it night".into(),
            conditioning: vec![Conditioning::Reference {
                image: Image::default(),
                strength: None,
            }],
            ..Default::default()
        };
        model.validate(&req).unwrap();
    }

    #[test]
    fn generate_without_weights_errors_not_loaded() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest {
            prompt: "x".into(),
            ..Default::default()
        };
        let mut progress = |_p: Progress| {};
        let err = model.generate(&req, &mut progress).unwrap_err().to_string();
        assert!(err.contains("not loaded"));
    }

    #[test]
    fn ids_match_expected() {
        assert_eq!(descriptor_klein_9b().id, FLUX2_KLEIN_9B_ID);
        assert_eq!(descriptor_klein_9b_edit().id, FLUX2_KLEIN_9B_EDIT_ID);
    }
}
