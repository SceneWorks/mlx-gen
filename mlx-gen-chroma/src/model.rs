//! Chroma provider registration + txt2img generation (sc-3839).
//!
//! Reuses the flux flow-match machinery (`build_linear_sigmas` for the raw `linspace(1,1/N,N)`;
//! `create_noise`/`unpack_latents`; the shared AutoencoderKL `Vae::decode`) and the core
//! `FlowMatchSampler` (Euler `x + v·Δσ`, `timestep(t)=σ`). Chroma's scheduler is **static-shift**
//! (`use_dynamic_shifting=false`, `σ' = shift·σ/(1+(shift-1)·σ)`), NOT FLUX's resolution-dependent
//! exp-shift, so the shift is applied here (see [`denoise`](Chroma::denoise)).
//! Chroma-specific: T5-only masked encode (sc-3838), the per-step **true CFG** (`neg + g·(pos−neg)`),
//! and the full-sequence MMDiT mask (text mask ++ image ones). The transformer runs f32 activations
//! over the bf16 weights (mlx promotes), matching a diffusers-bf16→f32 reference.

use mlx_gen::array::scalar;
use mlx_gen::image::decoded_to_image;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    default_seed, DiffusionSampler, Error, FlowMatchSampler, GenerationOutput, GenerationRequest,
    Generator, Image, LoadSpec, ModelDescriptor, ModelRegistration, Precision, Progress, Result,
    WeightsSource,
};
use mlx_gen_flux::{build_linear_sigmas, create_noise, unpack_latents, T5TextEncoder};
use mlx_gen_z_image::vae::Vae;
use mlx_rs::ops::{add, concatenate_axis, multiply, subtract};
use mlx_rs::Array;

use crate::config::{ChromaTransformerConfig, ChromaVariant};
use crate::loader;
use crate::text::encode_prompt;
use crate::transformer::ChromaTransformer;

pub fn descriptor_hd() -> ModelDescriptor {
    ChromaVariant::Hd.descriptor()
}

pub fn descriptor_base() -> ModelDescriptor {
    ChromaVariant::Base.descriptor()
}

pub fn descriptor_flash() -> ModelDescriptor {
    ChromaVariant::Flash.descriptor()
}

pub fn load_hd(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    Ok(Box::new(load_chroma(ChromaVariant::Hd, spec)?))
}

pub fn load_base(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    Ok(Box::new(load_chroma(ChromaVariant::Base, spec)?))
}

pub fn load_flash(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    Ok(Box::new(load_chroma(ChromaVariant::Flash, spec)?))
}

pub fn load_chroma(variant: ChromaVariant, spec: &LoadSpec) -> Result<Chroma> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{}: only dense bf16 is wired for the Chroma port (quant = sc-3841)",
            variant.id()
        )));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(Error::Msg(format!(
                "{} expects a Chroma diffusers snapshot directory (tokenizer/ text_encoder/ \
                 transformer/ vae/), not a single .safetensors file",
                variant.id()
            )))
        }
    };

    let cfg = ChromaTransformerConfig::default();
    let tokenizer = loader::load_tokenizer()?;
    let t5 = loader::load_t5_encoder(root)?;
    let transformer = loader::load_transformer(root, cfg)?;
    let vae = loader::load_vae(root)?;

    Ok(Chroma {
        descriptor: variant.descriptor(),
        variant,
        tokenizer: Some(tokenizer),
        t5: Some(t5),
        transformer: Some(transformer),
        vae: Some(vae),
    })
}

pub struct Chroma {
    descriptor: ModelDescriptor,
    variant: ChromaVariant,
    tokenizer: Option<TextTokenizer>,
    t5: Option<T5TextEncoder>,
    transformer: Option<ChromaTransformer>,
    vae: Option<Vae>,
}

/// FluxPosEmbed image position ids `[h2·w2, 3]` (axis 1 = row, axis 2 = col), row-major over the
/// packed `(height/16, width/16)` grid — diffusers `_prepare_latent_image_ids`.
fn latent_image_ids(h2: usize, w2: usize) -> Array {
    let mut data = vec![0f32; h2 * w2 * 3];
    for i in 0..h2 {
        for j in 0..w2 {
            let o = (i * w2 + j) * 3;
            data[o + 1] = i as f32;
            data[o + 2] = j as f32;
        }
    }
    Array::from_slice(&data, &[(h2 * w2) as i32, 3])
}

/// Text position ids `[L, 3]` — all zero (FluxPosEmbed places every text token at the origin).
fn zero_text_ids(l: usize) -> Array {
    Array::from_slice(&vec![0f32; l * 3], &[l as i32, 3])
}

impl Chroma {
    fn parts(&self) -> Result<(&TextTokenizer, &T5TextEncoder, &ChromaTransformer, &Vae)> {
        let err = |w: &str| Error::Msg(format!("{}: {w} not loaded", self.descriptor.id));
        Ok((
            self.tokenizer.as_ref().ok_or_else(|| err("tokenizer"))?,
            self.t5.as_ref().ok_or_else(|| err("t5"))?,
            self.transformer
                .as_ref()
                .ok_or_else(|| err("transformer"))?,
            self.vae.as_ref().ok_or_else(|| err("vae"))?,
        ))
    }

    /// The full-sequence MMDiT mask `[1, L + Si]` (0/1) = text mask ++ image ones.
    fn full_mask(text_mask: &Array, image_seq: i32) -> Result<Array> {
        let ones = Array::ones::<f32>(&[1, image_seq])?;
        Ok(concatenate_axis(&[text_mask, &ones], 1)?)
    }

    /// Run the true-CFG flow-match denoise from a given **packed** initial latent `[1, Si, 64]` →
    /// final packed latent. Public so the e2e parity test can inject the reference's initial latents
    /// (mlx and torch RNG differ); [`generate`](Self::generate) seeds it via `create_noise`.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(
        &self,
        prompt: &str,
        negative: &str,
        width: u32,
        height: u32,
        steps: u32,
        guidance: f32,
        latents: Array,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        let (tok, t5, tr, _) = self.parts()?;
        let (pos_embeds, pos_mask) = encode_prompt(tok, t5, prompt)?;
        let (neg_embeds, neg_mask) = encode_prompt(tok, t5, negative)?;

        let h2 = (height / 16) as usize;
        let w2 = (width / 16) as usize;
        let si = (h2 * w2) as i32;
        let img_ids = latent_image_ids(h2, w2);
        let txt_ids_pos = zero_text_ids(pos_embeds.shape()[1] as usize);
        let txt_ids_neg = zero_text_ids(neg_embeds.shape()[1] as usize);
        let mask_pos = Self::full_mask(&pos_mask, si)?;
        let mask_neg = Self::full_mask(&neg_mask, si)?;

        // Chroma's scheduler is `use_dynamic_shifting=false`. HD/Flash: static `shift` over the raw
        // mlx `linspace(1,1/N,N)` — `σ'=shift·σ/(1+(shift-1)·σ)` (shift=1.0 ⇒ identity). Base:
        // `use_beta_sigmas=true` — a beta-spaced schedule (sc-3840). NOT FLUX's resolution exp-shift.
        let sigmas = if self.variant.use_beta_sigmas() {
            crate::beta::base_sigmas(steps as usize)
        } else {
            let shift = self.variant.sigma_shift();
            let mut s = build_linear_sigmas(steps as usize, width, height, false)?;
            for v in s.iter_mut().take(steps as usize) {
                *v = shift * *v / (1.0 + (shift - 1.0) * *v);
            }
            s
        };
        let sampler = FlowMatchSampler::new(sigmas);
        let n = sampler.num_steps();

        let mut latents = latents;
        for t in 0..n {
            let ts = Array::from_slice(&[sampler.timestep(t)], &[1]);
            let pos = tr.forward(
                &latents,
                &pos_embeds,
                &ts,
                &img_ids,
                &txt_ids_pos,
                Some(&mask_pos),
            )?;
            let neg = tr.forward(
                &latents,
                &neg_embeds,
                &ts,
                &img_ids,
                &txt_ids_neg,
                Some(&mask_neg),
            )?;
            // true CFG: neg + g·(pos − neg).
            let pred = add(&neg, &multiply(&subtract(&pos, &neg)?, scalar(guidance))?)?;
            latents = sampler.step(&pred, &latents, t)?;
            on_progress(Progress::Step {
                current: t as u32 + 1,
                total: n as u32,
            });
        }
        Ok(latents)
    }

    /// Test accessors (real-weight e2e, sc-3839).
    #[doc(hidden)]
    pub fn transformer_ref(&self) -> &ChromaTransformer {
        self.transformer.as_ref().expect("transformer loaded")
    }
    #[doc(hidden)]
    pub fn tokenizer_ref(&self) -> &TextTokenizer {
        self.tokenizer.as_ref().expect("tokenizer loaded")
    }
    #[doc(hidden)]
    pub fn t5_ref(&self) -> &T5TextEncoder {
        self.t5.as_ref().expect("t5 loaded")
    }

    /// Unpack + VAE-decode a packed latent `[1, Si, 64]` → an [`Image`].
    pub fn decode(&self, latents: &Array, width: u32, height: u32) -> Result<Image> {
        let (_, _, _, vae) = self.parts()?;
        let unpacked = unpack_latents(latents, width, height)?;
        let decoded = vae.decode(&unpacked)?.as_dtype(mlx_rs::Dtype::Float32)?;
        decoded_to_image(&decoded)
    }
}

impl Generator for Chroma {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> Result<()> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;
        if req.prompt.trim().is_empty() {
            return Err(Error::Msg(format!(
                "{}: prompt must not be empty",
                self.descriptor.id
            )));
        }
        if !req.width.is_multiple_of(16) || !req.height.is_multiple_of(16) {
            return Err(Error::Msg(format!(
                "{}: width and height must be multiples of 16, got {}x{}",
                self.descriptor.id, req.width, req.height
            )));
        }
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        let steps = req.steps.unwrap_or_else(|| self.variant.default_steps());
        let guidance = req
            .true_cfg
            .unwrap_or_else(|| self.variant.default_true_cfg());
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let base_seed = req.seed.unwrap_or_else(default_seed);

        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            let seed = base_seed.wrapping_add(i as u64);
            let latents = create_noise(seed, req.width, req.height)?;
            let final_latents = self.denoise(
                &req.prompt,
                negative,
                req.width,
                req.height,
                steps,
                guidance,
                latents,
                on_progress,
            )?;
            on_progress(Progress::Decoding);
            images.push(self.decode(&final_latents, req.width, req.height)?);
        }
        Ok(GenerationOutput::Images(images))
    }
}

inventory::submit! {
    ModelRegistration { descriptor: descriptor_hd, load: load_hd }
}

inventory::submit! {
    ModelRegistration { descriptor: descriptor_base, load: load_base }
}

inventory::submit! {
    ModelRegistration { descriptor: descriptor_flash, load: load_flash }
}
