//! `Flux1DevControl` — the FLUX.1-dev **Fun-Controlnet-Union** variant (sc-8238, epic 8236): VACE-like
//! strict structural conditioning (pose / canny / depth — input-agnostic, no discrete mode index) via
//! `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0`.
//!
//! This is the **E1 engine** story: the control branch struct, its weight load, the control-latent
//! encode, the residual computation, and the injection into the FLUX.1-dev denoise. The transformer is a
//! [`FluxControlTransformer`] (the parity-proven dev DiT + a diffusers-style [`FluxControlNet`] residual
//! branch); `generate` threads a VAE-encoded control latent through it under the embedded-guidance
//! denoise (dev is guidance-distilled — a single forward, no true-CFG). [`load_dev_control`] needs the
//! dev snapshot (`spec.weights`) **and** the control checkpoint (`spec.control`).
//!
//! ## Compose-readiness (constraint 2)
//! The control-residual injection coexists with the existing [`DitImageInjector`] seam (PuLID /
//! `FluxIpInjector`): the denoise routes through [`FluxControlTransformer::forward_composed`], which
//! threads an optional identity injector into the SAME base double-block stream as the control
//! residuals (see [`crate::transformer::FluxTransformer::forward_control`]). A follow-on epic can stack
//! identity (PuLID/IP-Adapter) + control in one denoise step by passing `Some(injector)` here — this
//! E1 story wires the seam and exposes [`Flux1DevControl::generate_with_injector`] for it; the worker
//! wiring + the registered descriptor are E2 (sc-8239).
//!
//! ## Scope note (E1 vs E2)
//! This file ships a descriptor + an `inventory` registration so the engine is testable end-to-end and
//! E2 has a concrete entry to build the worker `ModelDescriptor` onto. E2 (sc-8239) owns the
//! capability-surface finalization and the per-mode (canny/depth/pose) real-weight smokes.

use mlx_gen::gen_core;
use mlx_gen::image::decoded_to_image;
use mlx_gen::img2img::preprocess_init_image;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, default_seed, require_base_dir,
    require_control, run_flow_sampler, AcceptedControlKinds, Capabilities, ConditioningKind,
    ControlBranch, Error, GenerationOutput, GenerationRequest, Generator, Image, LoadSpec,
    Modality, ModelDescriptor, Precision, Progress, Quant, Result, TimestepConvention,
};
use mlx_rs::{Array, Dtype};

use crate::config::{FLUX1_DEV_CONTROL_ID, HYPER_SAMPLER};
use crate::control_transformer::FluxControlTransformer;
use crate::loader;
use crate::pipeline::{build_sigmas_with, create_noise, pack_latents, unpack_latents};
use crate::text_encoder::FluxTextEncoders;
use crate::transformer::DitImageInjector;
use mlx_gen_z_image::vae::Vae;

use crate::config::{FluxVariant, DEFAULT_GUIDANCE};

/// Default control-conditioning scale when a `Conditioning::Control` omits / does not override it. The
/// Shakker recommends ~0.7 for Union-Pro-2.0; the request's per-control `scale` wins when set.
const DEFAULT_CONTROL_SCALE: f32 = 0.7;

/// The control variant's identity + capabilities. The guidance-distilled dev base (embedded guidance,
/// no negative prompt / true-CFG) plus a required `Control` conditioning (the pose/canny/depth hint).
/// Mac-only, like every FLUX.1 variant. (E2 owns the final capability surface; this is the E1 stub the
/// engine + tests build against.)
pub fn descriptor_dev_control() -> ModelDescriptor {
    ModelDescriptor {
        id: FLUX1_DEV_CONTROL_ID,
        family: "flux",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: false,
            // dev consumes its guidance scale as an embedded scalar (FLUX.1-dev pattern), not CFG.
            supports_guidance: true,
            supports_true_cfg: false,
            // Control (required) — the structural hint (pose/canny/depth, input-agnostic).
            conditioning: vec![ConditioningKind::Control],
            supported_quants: &[Quant::Q4, Quant::Q8],
            // LoRA/LoKr target the base DiT (the control branch is never an adapter target).
            supports_lora: true,
            supports_lokr: true,
            samplers: {
                let mut s = curated_sampler_names();
                s.push(crate::config::DEFAULT_SAMPLER);
                s.push(HYPER_SAMPLER);
                s
            },
            schedulers: {
                let mut s = curated_scheduler_names();
                s.push("linear");
                s
            },
            supported_guidance_methods: vec![],
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: true,
            supports_kv_cache: false,
            requires_sigma_shift: FluxVariant::Dev.requires_sigma_shift(),
        },
    }
}

/// A loaded control generator: the dev base components + the control transformer assembled from the dev
/// snapshot and the Shakker Fun-Controlnet-Union overlay.
pub struct Flux1DevControl {
    descriptor: ModelDescriptor,
    t5_tokenizer: TextTokenizer,
    clip_tokenizer: TextTokenizer,
    text_encoders: FluxTextEncoders,
    transformer: FluxControlTransformer,
    vae: Vae,
}

/// FLUX.1-dev Fun-Controlnet-Union (sc-8238): load the dev snapshot + the Shakker control checkpoint and
/// assemble the [`Flux1DevControl`] generator.
///
/// `spec.weights` must be the dev snapshot directory (tokenizer/ tokenizer_2/ text_encoder/
/// text_encoder_2/ transformer/ vae/); `spec.control` (required) the
/// `FLUX.1-dev-ControlNet-Union-Pro-2.0` checkpoint (a single `.safetensors` `File`, or a `Dir`).
/// `spec.quantize` (Q4/Q8) quantizes the whole model — base DiT + control branch + text encoders + VAE.
pub fn load_dev_control(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{FLUX1_DEV_CONTROL_ID}: only dense bf16 is wired (Q4/Q8 = spec.quantize)"
        )));
    }
    // Shared load boilerplate (sc-8241): the base must be a snapshot dir, the control checkpoint is
    // required. The model id + labels keep the messages aligned with the other control ports.
    let root = require_base_dir(
        spec,
        FLUX1_DEV_CONTROL_ID,
        "a FLUX.1-dev snapshot directory",
    )?;
    let control = require_control(
        spec,
        FLUX1_DEV_CONTROL_ID,
        "FLUX.1-dev-ControlNet-Union-Pro-2.0",
    )?;

    let t5_tokenizer = loader::load_t5_tokenizer(root, FluxVariant::Dev)?;
    let clip_tokenizer = loader::load_clip_tokenizer()?;
    let mut text_encoders = FluxTextEncoders {
        t5: loader::load_t5_encoder(root)?,
        clip: loader::load_clip_encoder(root)?,
    };
    let mut transformer = loader::load_control_transformer_dev(root, control)?;
    let mut vae = loader::load_vae(root)?;
    if let Some(q) = spec.quantize {
        let bits = q.bits();
        text_encoders.quantize(bits)?;
        transformer.quantize(bits)?;
        vae.quantize(bits)?;
    }
    // LoRA/LoKr (sc-2657): applied to the base DiT (the control branch is never an adapter target),
    // after quantization, as forward-time residuals. No-op when empty.
    crate::adapters::apply_flux_adapters(transformer.base_mut(), &spec.adapters)?;

    Ok(Box::new(Flux1DevControl {
        descriptor: descriptor_dev_control(),
        t5_tokenizer,
        clip_tokenizer,
        text_encoders,
        transformer,
        vae,
    }))
}

impl Flux1DevControl {
    /// Tokenize + encode the prompt into `(prompt_embeds, pooled_prompt_embeds)` (the dev T5 + CLIP
    /// path; same as [`crate::model::Flux1::encode_prompt`]).
    fn encode_prompt(&self, prompt: &str) -> Result<(Array, Array)> {
        let (t5_ids, _) = mlx_gen::tokenizer::to_arrays(&self.t5_tokenizer.tokenize(prompt)?);
        let (clip_ids, _) = mlx_gen::tokenizer::to_arrays(&self.clip_tokenizer.tokenize(prompt)?);
        self.text_encoders.encode(&t5_ids, &clip_ids)
    }

    /// VAE-encode + pack the control hint into the packed control latent `[1, seq, 64]` (constant
    /// across steps + the batch). The Shakker FLUX.1 ControlNet encodes the control image with the same
    /// VAE and 2×2 pack as the noise latents (diffusers `prepare_image` → `vae.encode` →
    /// `_pack_latents`), so the control latent aligns 1:1 with the base image tokens.
    fn encode_control_latent(&self, image: &Image, width: u32, height: u32) -> Result<Array> {
        let image_nchw = preprocess_init_image(image, width, height)?;
        let encoded = self.vae.encode(&image_nchw)?; // [1, 16, H/8, W/8]
        pack_latents(&encoded, width, height)
    }

    /// As [`Generator::generate`], but threading an OPTIONAL identity injector (PuLID / XLabs
    /// IP-Adapter) into every control-denoise step — the **compose-ready** seam (constraint 2). With
    /// `injector = None` this is the plain control path; `Some(..)` stacks identity + control in one
    /// denoise (the seam a follow-on epic uses). `injector = None` IS [`Generator::generate`]'s path.
    pub fn generate_with_injector(
        &self,
        req: &GenerationRequest,
        injector: Option<&dyn DitImageInjector>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        let (control_image, control_scale) = self.resolve_control(req)?;
        let control_scale = if control_scale == 0.0 {
            // A request that supplied a `Control` but left `scale` at its default 0 still steers; map
            // the unset (0.0) to the Shakker-recommended default rather than running an inert branch.
            DEFAULT_CONTROL_SCALE
        } else {
            control_scale
        };

        let (prompt_embeds, pooled_prompt_embeds) = self.encode_prompt(&req.prompt)?;
        let control_latent = self.encode_control_latent(control_image, req.width, req.height)?;

        let base_seed = req.seed.unwrap_or_else(default_seed);
        let sampler_name = req
            .sampler
            .as_deref()
            .unwrap_or(crate::config::DEFAULT_SAMPLER);
        let (def_steps, def_guidance) = if sampler_name == HYPER_SAMPLER {
            (8, DEFAULT_GUIDANCE)
        } else {
            (FluxVariant::Dev.default_steps(), DEFAULT_GUIDANCE)
        };
        let steps = req.steps.unwrap_or(def_steps) as usize;
        let guidance = req.guidance.unwrap_or(def_guidance);

        let sigmas = build_sigmas_with(
            steps,
            req.width,
            req.height,
            FluxVariant::Dev.requires_sigma_shift(),
            req.scheduler.as_deref(),
        )?;

        // Compiled elementwise glue (sc-2963), shared with the base flux1 path. Scoped + restored on
        // drop by the RAII guard (F-007).
        let _compile_glue = crate::transformer::CompileGlueGuard::enable();

        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            let seed = base_seed.wrapping_add(i as u64);
            let latents = create_noise(seed, req.width, req.height)?;
            let final_latents = run_flow_sampler(
                Some(sampler_name),
                TimestepConvention::Sigma,
                &sigmas,
                latents,
                seed,
                &req.cancel,
                on_progress,
                |x_in, timestep| {
                    self.transformer.forward_composed(
                        x_in,
                        &control_latent,
                        &prompt_embeds,
                        &pooled_prompt_embeds,
                        timestep,
                        guidance,
                        req.width,
                        req.height,
                        control_scale,
                        injector,
                    )
                },
            )?;
            on_progress(Progress::Decoding);
            let unpacked = unpack_latents(&final_latents, req.width, req.height)?;
            let decoded = self.vae.decode(&unpacked)?.as_dtype(Dtype::Float32)?;
            images.push(decoded_to_image(&decoded)?);
        }
        Ok(GenerationOutput::Images(images))
    }

    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        // The plain control path is the compose-ready path with NO identity injector — exercising the
        // same seam keeps the two from drifting (constraint 2: control never bypasses the injector hook).
        self.generate_with_injector(req, None, on_progress)
    }

    fn validate_capability(&self, req: &GenerationRequest) -> Result<()> {
        if req.prompt.trim().is_empty() {
            return Err(Error::Msg(format!(
                "{FLUX1_DEV_CONTROL_ID}: prompt is required"
            )));
        }
        let caps = &self.descriptor.capabilities;
        if let Some(s) = &req.sampler {
            if !caps.samplers.contains(&s.as_str()) {
                return Err(Error::Msg(format!(
                    "{FLUX1_DEV_CONTROL_ID}: unsupported sampler {s:?}"
                )));
            }
        }
        if let Some(s) = &req.scheduler {
            if !caps.schedulers.contains(&s.as_str()) {
                return Err(Error::Msg(format!(
                    "{FLUX1_DEV_CONTROL_ID}: unsupported scheduler {s:?}"
                )));
            }
        }
        if !req.width.is_multiple_of(16) || !req.height.is_multiple_of(16) {
            return Err(Error::Msg(format!(
                "{FLUX1_DEV_CONTROL_ID}: width and height must be multiples of 16, got {}x{}",
                req.width, req.height
            )));
        }
        Ok(())
    }
}

/// The Shakker Union-Pro-2.0 is an *input-agnostic* union ControlNet (pose / canny / depth share one
/// VAE-encoded control path — the 2.0 checkpoint dropped the 1.0 discrete `control_mode` index), so the
/// default [`AcceptedControlKinds::Any`] applies and the control boilerplate (resolve/validate-present +
/// the load helpers) comes from the shared trait (sc-8241).
impl ControlBranch for Flux1DevControl {
    fn model_id(&self) -> &'static str {
        FLUX1_DEV_CONTROL_ID
    }

    fn accepted_control_kinds(&self) -> AcceptedControlKinds {
        AcceptedControlKinds::Any
    }
}

impl Generator for Flux1DevControl {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // Capability floor (size/count/sampler/scheduler/prompt), then the shared control-present check
        // (sc-8241's `ControlBranch::require_control_present`).
        self.validate_capability(req)?;
        self.require_control_present(req)?;
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

// Link-time registration (epic 3720): the `inventory::submit!` so E2's worker can resolve the
// `flux1_dev_control` generator by id. The `impl Generator` above stays hand-written because `validate`
// adds a control-conditioning check beyond the plain capability floor.
mlx_gen::register_generators! { descriptor_dev_control => load_dev_control }

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::WeightsSource;

    #[test]
    fn descriptor_is_flux1_dev_control() {
        let d = descriptor_dev_control();
        assert_eq!(d.id, "flux1_dev_control");
        assert_eq!(d.family, "flux");
        assert!(d.capabilities.accepts(ConditioningKind::Control));
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
        // overlay is a hard requirement) — not on the missing snapshot.
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let err = load_dev_control(&spec)
            .err()
            .expect("expected error")
            .to_string();
        assert!(
            err.contains("FLUX.1-dev-ControlNet-Union-Pro-2.0"),
            "got: {err}"
        );
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
