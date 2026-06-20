//! `Flux2DevControl` — the FLUX.2-dev **Fun-Controlnet-Union** variant (sc-2292): strict-pose
//! (VACE-style) conditioning via `alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union`, registered as its
//! own `Generator` (`flux2_dev_control`).
//!
//! Mirrors the Z-Image-turbo control port (sc-2257) onto the dev base: the transformer is a
//! [`Flux2ControlTransformer`] (the parity-proven dev DiT + the control branch) and `generate`
//! threads a VAE-encoded control context through it under the embedded-guidance denoise (dev is
//! guidance-distilled — a single forward, no true-CFG). [`load_dev_control`] needs the dev snapshot
//! (`spec.weights`) **and** the control checkpoint (`spec.control`); the base loads manifest-aware
//! (a pre-quantized dev snapshot loads packed, sc-5917) and the bf16 control overlay loads dense,
//! then `spec.quantize` packs the control branch in place (the packed base no-ops). The control
//! patch embedder stays dense (its 260 in-features is not a multiple of the quant group size).
//!
//! Architecture (`videox_fun/models/flux2_transformer2d_control.py`): a VACE ControlNet on the first
//! 4 of dev's 8 base double blocks. The control context is the VAE-encoded pose/union skeleton
//! (`control_latents` 128) concatenated with a zero inpaint mask (4) and a zero inpaint latent (128)
//! = 260 channels per image token (the union ControlNet's pose-only layout). See
//! [`crate::transformer::Flux2ControlBranch`] for the hint-injection forward.

use mlx_gen::image::decoded_to_image;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    default_seed, gen_core, Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, ModelRegistration,
    Precision, Progress, Quant, Result, WeightsSource,
};
use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

use crate::config::{Flux2Config, FLUX2_DEV_CONTROL_ID};
use crate::model::{crop_to_even, match_latent_spatial_size, validate_request};
use crate::pipeline::{
    add_noise_by_interpolation, create_noise, init_time_step, pack_latents, patchify_latents,
    prepare_grid_ids, prepare_text_ids, preprocess_ref_image, schedule, timesteps_x1000,
};
use crate::text_encoder::Qwen3TextEncoder;
use crate::transformer::Flux2ControlTransformer;
use crate::vae::Flux2Vae;
use crate::{loader, CONTROL_IN_DIM};

/// The control variant's identity + capabilities. The guidance-distilled dev base (embedded
/// guidance, no negative prompt / true-CFG) plus `Control` conditioning (the required pose/union
/// skeleton) and an optional `Reference` (an img2img init image, the fork's `inpaint_image`/`image`
/// init seed). Mac-only, like every FLUX.2 variant.
pub fn descriptor_dev_control() -> ModelDescriptor {
    ModelDescriptor {
        id: FLUX2_DEV_CONTROL_ID,
        family: "flux2",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: false,
            // dev consumes its guidance scale as an embedded scalar (FLUX.1-dev pattern), not CFG.
            supports_guidance: true,
            supports_true_cfg: false,
            // Control (required, the pose/union skeleton) + an optional img2img Reference init.
            conditioning: vec![ConditioningKind::Control, ConditioningKind::Reference],
            // LoRA/LoKr target the base DiT (the control branch is never an adapter target).
            supports_lora: true,
            supports_lokr: true,
            supported_quants: &[Quant::Q4, Quant::Q8],
            samplers: Vec::new(),
            schedulers: vec!["flow_match_euler"],
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: true,
            supports_kv_cache: false,
            requires_sigma_shift: true,
        },
    }
}

/// A loaded control generator: the dev base components + the control transformer assembled from the
/// dev snapshot and the Fun-Controlnet-Union overlay.
pub struct Flux2DevControl {
    descriptor: ModelDescriptor,
    config: Flux2Config,
    tokenizer: TextTokenizer,
    text_encoder: Qwen3TextEncoder,
    transformer: Flux2ControlTransformer,
    vae: Flux2Vae,
}

/// FLUX.2-dev strict pose (sc-2292): load the dev snapshot + the Fun-Controlnet-Union control
/// checkpoint and assemble the [`Flux2DevControl`] generator.
///
/// `spec.weights` must be the dev snapshot directory (tokenizer/ text_encoder/ transformer/ vae/);
/// `spec.control` (required) the Fun-Controlnet-Union checkpoint (a single `.safetensors` `File`, or
/// a `Dir`). The base loads manifest-aware (pre-quantized dev → packed); the bf16 control overlay
/// loads dense. `spec.quantize` (Q4/Q8) then quantizes the whole model — a no-op on the already
/// packed base, packing the dense control branch + the text encoder + VAE (the control patch
/// embedder stays dense, its in-features is not a multiple of 64).
pub fn load_dev_control(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{FLUX2_DEV_CONTROL_ID}: only the default precision is wired; drop the precision \
             override (Q4/Q8 = spec.quantize)"
        )));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(Error::Msg(format!(
                "{FLUX2_DEV_CONTROL_ID} expects a FLUX.2-dev snapshot directory (tokenizer/ \
                 text_encoder/ transformer/ vae/), not a single .safetensors file"
            )))
        }
    };
    let control = spec.control.as_ref().ok_or_else(|| {
        Error::Msg(format!(
            "{FLUX2_DEV_CONTROL_ID} requires the FLUX.2-dev-Fun-Controlnet-Union weights — set \
             LoadSpec::control (e.g. with_control(WeightsSource::File(...)))"
        ))
    })?;

    let mut transformer = loader::load_control_transformer_dev(root, control)?;
    let mut text_encoder = loader::load_text_encoder_dev(root)?;
    let mut vae = loader::load_vae(root)?;
    if let Some(q) = spec.quantize {
        let bits = q.bits();
        transformer.quantize(bits)?;
        text_encoder.quantize(bits)?;
        vae.quantize(bits)?;
    }
    // LoRA/LoKr (sc-2646): applied to the base DiT (the control branch is never an adapter target),
    // after quantization, as forward-time residuals. No-op when empty.
    if !spec.adapters.is_empty() {
        crate::adapters::apply_flux2_adapters(transformer.base_mut(), &spec.adapters)?;
    }
    Ok(Box::new(Flux2DevControl {
        descriptor: descriptor_dev_control(),
        config: Flux2Config::dev(),
        tokenizer: loader::load_tokenizer_dev(root)?,
        text_encoder,
        transformer,
        vae,
    }))
}

impl Flux2DevControl {
    /// Tokenize + encode the prompt into `(prompt_embeds, text_ids)` (the dev Mistral TE path; same
    /// as [`crate::model::Flux2`]'s `encode`).
    fn encode(&self, prompt: &str) -> Result<(Array, Array)> {
        let tok = self.tokenizer.tokenize(prompt)?;
        let (input_ids, attention_mask) = mlx_gen::tokenizer::to_arrays(&tok);
        let embeds = self
            .text_encoder
            .prompt_embeds(&input_ids, &attention_mask)?;
        let ids = prepare_text_ids(embeds.shape()[1] as usize);
        Ok((embeds, ids))
    }

    /// Extract the (required) control image + its `control_context_scale` from the request. The
    /// Fun-Controlnet-Union is a *union* ControlNet (pose / canny / depth / … share one VAE-encoded
    /// control path), so any [`mlx_gen::ControlKind`] is accepted — the pose skeleton is the
    /// validated use. A single control image is supported.
    fn resolve_control<'a>(&self, req: &'a GenerationRequest) -> Result<(&'a Image, f32)> {
        let mut found = None;
        for c in &req.conditioning {
            if let Conditioning::Control { image, scale, .. } = c {
                if found.is_some() {
                    return Err(Error::Msg(format!(
                        "{FLUX2_DEV_CONTROL_ID}: a single control image is supported"
                    )));
                }
                found = Some((image, *scale));
            }
        }
        found.ok_or_else(|| {
            Error::Msg(format!(
                "{FLUX2_DEV_CONTROL_ID} requires a Control conditioning (the pose/union skeleton)"
            ))
        })
    }

    /// The optional img2img init image (a single `Reference`) + its strength (the per-reference
    /// strength wins over `req.strength`). More than one `Reference` is an error.
    fn resolve_reference<'a>(
        &self,
        req: &'a GenerationRequest,
    ) -> Result<Option<(&'a Image, Option<f32>)>> {
        let mut reference = None;
        for c in &req.conditioning {
            if let Conditioning::Reference { image, strength } = c {
                if reference.is_some() {
                    return Err(Error::Msg(format!(
                        "{FLUX2_DEV_CONTROL_ID}: a single img2img init reference is supported"
                    )));
                }
                reference = Some((image, strength.or(req.strength)));
            }
        }
        Ok(reference)
    }

    /// img2img init conditioning (same encode chain as [`crate::model::Flux2`]): resize → VAE-encode
    /// → NCHW → crop-to-even → match the target latent grid → 2×2 patchify → BN-normalize → pack.
    /// Returns the **clean** packed init latents `[1, lat_h·lat_w, 128]` (seed-independent).
    fn encode_init_latents(&self, image: &Image, width: u32, height: u32) -> Result<Array> {
        let pre = preprocess_ref_image(image, width, height)?;
        let enc = self.vae.encode_mean(&pre)?;
        let enc = enc.transpose_axes(&[0, 3, 1, 2])?;
        let enc = crop_to_even(&enc)?;
        let enc = match_latent_spatial_size(&enc, (height / 8) as i32, (width / 8) as i32)?;
        let patchified = patchify_latents(&enc)?;
        let normed = self.vae.bn_normalize_nchw(&patchified)?;
        pack_latents(&normed)
    }

    /// Build the packed control context `[1, seq, 260]` from the pose/union control image — the
    /// fork's `pipeline_flux2_control.py`: VAE-encode → 2×2 patchify → BN-normalize → pack
    /// (`control_latents`, 128), concatenated with a zero inpaint **mask** (4) and a zero **inpaint
    /// latent** (128). For pure pose (no inpaint image / mask) the fork's mask is `1 − ones = 0` and
    /// the inpaint latent is a zeros tensor, so both are all-zero here. `seq` equals the target
    /// latent sequence (built at the same `width`/`height`), so the control context aligns 1:1 with
    /// the base image tokens.
    fn encode_control_context(&self, image: &Image, width: u32, height: u32) -> Result<Array> {
        let pre = preprocess_ref_image(image, width, height)?;
        let enc = self.vae.encode_mean(&pre)?; // NHWC [1,H/8,W/8,32]
        let enc = enc.transpose_axes(&[0, 3, 1, 2])?; // NCHW
        let enc = crop_to_even(&enc)?;
        let enc = match_latent_spatial_size(&enc, (height / 8) as i32, (width / 8) as i32)?;
        let patchified = patchify_latents(&enc)?; // [1,128,h,w]
        let control_lat = self.vae.bn_normalize_nchw(&patchified)?;
        let control_packed = pack_latents(&control_lat)?; // [1, seq, 128]
        let seq = control_packed.shape()[1];
        // Union pose-only layout: zero mask (1 latent channel × 2×2 patch = 4) + zero inpaint latent
        // (= in_channels, 128). Concatenated on the channel axis → 260 = CONTROL_IN_DIM.
        let in_ch = self.config.in_channels as i32;
        let mask_ch = in_ch / self.config.num_latent_channels as i32; // 128 / 32 = 4 (the 2×2 patch)
        let zeros = |c: i32| -> Result<Array> { Ok(mlx_rs::ops::zeros::<f32>(&[1, seq, c])?) };
        let mask = zeros(mask_ch)?;
        let inpaint = zeros(in_ch)?;
        let cc = concatenate_axis(&[&control_packed, &mask, &inpaint], 2)?;
        debug_assert_eq!(
            cc.shape()[2],
            CONTROL_IN_DIM,
            "control context must be 260ch"
        );
        Ok(cc)
    }

    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let steps = req.steps.unwrap_or(self.config_default_steps()) as usize;
        let guidance = req.guidance.unwrap_or(crate::config::DEFAULT_GUIDANCE_DEV);
        // dev is guidance-distilled: the scale is an embedded scalar (single forward), never true-CFG.
        let embedded_guidance = Some(guidance);

        let (control_image, control_scale) = self.resolve_control(req)?;
        // Optional img2img init (the fork's `image`/`inpaint_image` seed) via a single `Reference`.
        let img2img = self.resolve_reference(req)?;
        let start_step = match &img2img {
            Some((_, strength)) => init_time_step(steps, *strength),
            None => 0,
        };

        let (prompt_embeds, text_ids) = self.encode(&req.prompt)?;

        let sched = schedule(steps, req.width, req.height);
        let timesteps = timesteps_x1000(&sched);
        let lat_h = (req.height / 16) as usize;
        let lat_w = (req.width / 16) as usize;
        let latent_ids = prepare_grid_ids(lat_h, lat_w, 0);
        let in_channels = self.config.in_channels as i32;

        // The control context + the clean img2img init latents are constant across steps + the batch
        // (they depend only on the image + dims, not the per-seed noise) — encode once.
        let control_context = self.encode_control_context(control_image, req.width, req.height)?;
        let clean_init = match &img2img {
            Some((image, _)) if start_step > 0 => {
                Some(self.encode_init_latents(image, req.width, req.height)?)
            }
            _ => None,
        };

        // Compiled elementwise glue (sc-2963), shared with the base flux2 path. Scoped + restored on
        // drop by the RAII guard (F-007) instead of leaking the process-global toggle on.
        let _compile_glue = crate::transformer::CompileGlueGuard::enable();

        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            let seed = base_seed.wrapping_add(i as u64);
            let noise = create_noise(seed, req.width, req.height, self.config.in_channels)?;
            let mut latents = match &clean_init {
                Some(clean) => add_noise_by_interpolation(clean, &noise, sched.sigmas[start_step])?,
                None => noise,
            };
            for (t, &ts) in timesteps.iter().enumerate().skip(start_step) {
                if req.cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                let v = self.transformer.forward(
                    &latents,
                    &prompt_embeds,
                    &latent_ids,
                    &text_ids,
                    ts,
                    embedded_guidance,
                    &control_context,
                    control_scale,
                )?;
                latents = sched.step(&latents, &v, t)?;
                latents.eval()?;
                on_progress(Progress::Step {
                    current: t as u32 + 1,
                    total: steps as u32,
                });
            }
            on_progress(Progress::Decoding);
            let packed = latents.reshape(&[1, lat_h as i32, lat_w as i32, in_channels])?;
            let decoded = self.vae.decode_packed_latents(&packed)?; // NHWC [1,H,W,3]
            let nchw = decoded.transpose_axes(&[0, 3, 1, 2])?;
            images.push(decoded_to_image(&nchw)?);
        }
        Ok(GenerationOutput::Images(images))
    }

    fn config_default_steps(&self) -> u32 {
        crate::config::DEFAULT_STEPS_DEV
    }
}

impl Generator for Flux2DevControl {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // Shared capability floor (size/count/guidance/negative/accepted conditioning + multiple-of-16),
        // then the control-specific requirement that a Control conditioning is present.
        validate_request(&self.descriptor, req)?;
        if !req
            .conditioning
            .iter()
            .any(|c| matches!(c, Conditioning::Control { .. }))
        {
            return Err(gen_core::Error::Msg(format!(
                "{FLUX2_DEV_CONTROL_ID} requires a Control conditioning (the pose/union skeleton)"
            )));
        }
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }
}

/// Registry adapter: the link-time registry's `load` slot is typed on the backend-neutral
/// [`gen_core::Result`] (epic 3720); bridge the crate's rich-`Result` [`load_dev_control`] into it.
fn load_dev_control_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_dev_control(spec).map_err(Into::into)
}

inventory::submit! {
    ModelRegistration { descriptor: descriptor_dev_control, load: load_dev_control_registered }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_is_flux2_dev_control() {
        let d = descriptor_dev_control();
        assert_eq!(d.id, "flux2_dev_control");
        assert_eq!(d.family, "flux2");
        assert!(d.capabilities.accepts(ConditioningKind::Control));
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        // dev embedded guidance: guidance on, negative / true-CFG off; no KV cache; mac-only.
        assert!(d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_true_cfg);
        assert!(!d.capabilities.supports_kv_cache);
        assert!(d.capabilities.mac_only);
    }

    #[test]
    fn load_rejects_missing_control_weights() {
        // Without `spec.control`, load must fail on the missing control weights (proving the control
        // overlay is wired as a hard requirement) — not on the missing snapshot.
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let err = load_dev_control(&spec)
            .err()
            .expect("expected error")
            .to_string();
        assert!(err.contains("Fun-Controlnet-Union"), "got: {err}");
    }

    #[test]
    fn load_rejects_single_file_base() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/dev.safetensors".into()))
            .with_control(WeightsSource::File("/tmp/control.safetensors".into()));
        let err = load_dev_control(&spec)
            .err()
            .expect("expected error")
            .to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }
}
