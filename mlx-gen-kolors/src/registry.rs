//! `KolorsGenerator` — the [`mlx_gen::Generator`] impl for Kolors, plus its [`descriptor`]/[`load`]
//! entry points and the `inventory` registration that wires it into `mlx_gen`'s registry under the
//! id `"kolors"` (sc-3874).
//!
//! The epic-3090 ports (sc-3091–3098) gave [`crate::Kolors`] the full capability surface but only as
//! a direct struct API (which the parity tests call). This module makes Kolors **dispatchable** —
//! `mlx_gen::load("kolors", spec).generate(req)`, the SceneWorks worker's in-process entry — by
//! mapping [`LoadSpec`]/[`GenerationRequest`] onto that API and looping `req.count` with per-image
//! seeds + cancel + streamed progress, mirroring `mlx-gen-sdxl/src/model.rs`.
//!
//! **Registration mechanism:** `inventory::submit!` here is collected by `mlx_gen`'s
//! `inventory::collect!` at *link* time — so the registration activates whenever a consumer (the
//! worker, or this crate's own test binary) links `mlx-gen-kolors`. The core `mlx-gen` crate does
//! **not** depend on the model crates (by design); there is no root-crate dependency to add.

use mlx_rs::{random, Dtype};

use mlx_gen::{
    curated_scheduler_names, default_seed, Capabilities, Conditioning, ConditioningKind,
    ControlKind, Error, GenerationOutput, GenerationRequest, Generator, Image, LoadSpec, Modality,
    ModelDescriptor, Progress, Quant, Result, Scheduler, Solver, WeightsSource,
};

use mlx_gen_sdxl::{
    decode_image, encode_init_latents, load_controlnet, ControlNet, IpImageEncoder,
};

use crate::ip_adapter::load_kolors_ip_adapter;
use crate::model::{DEFAULT_IMG2IMG_STRENGTH, SPATIAL_SCALE};
use crate::sampler::NUM_TRAIN_TIMESTEPS;
use crate::Kolors;

/// Registry id — the SceneWorks worker's `payload.model` for the Kolors family.
pub const MODEL_ID: &str = "kolors";

/// diffusers `KolorsPipeline` production defaults: 50 inference steps, CFG 5.0.
const DEFAULT_STEPS: u32 = 50;
const DEFAULT_GUIDANCE: f32 = 5.0;
/// Default IP-Adapter scale when a request doesn't override it (carried on the `Reference` strength
/// field in IP mode, mirroring the SDXL IP-Adapter convention).
const IP_DEFAULT_SCALE: f32 = 0.6;
/// Default img2img init strength for the combined strict-pose tier (sc-5012) when `req.strength` is
/// unset — the torch `_run_pose` default 1.0 (at full strength the init only seeds latent
/// dimensions; identity comes from the IP-Adapter, structure from the ControlNet).
const POSE_IMG2IMG_STRENGTH: f32 = 1.0;
/// The single Kolors sampler — diffusers `EulerDiscreteScheduler` (leading), see [`KolorsEulerSampler`].
const SAMPLER: &str = "euler_discrete";

/// Kolors' identity + capabilities — constructible without loading weights (registry
/// introspection). Advertises **only** the wired + parity-proven surface (the false-capability
/// guard): T2I + img2img (`Reference`) + ControlNet-pose (`Control`) + IP-Adapter (`Reference` in
/// IP mode) + Q8/Q4 + **LoRA/LoKr** (sc-4733 — merged into the SDXL-family U-Net at load via
/// [`Kolors::apply_lora`], the inference complement to the Kolors trainer sc-4568).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "kolors",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            // Kolors uses real classifier-free guidance over the ChatGLM3 conditioning.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            // Reference = img2img init (sc-3095) OR the IP-Adapter image prompt when an IP-Adapter is
            // loaded (sc-3098); Control = the Kolors ControlNet-pose branch (sc-3097).
            conditioning: vec![ConditioningKind::Reference, ConditioningKind::Control],
            // LoRA/LoKr merge into the SDXL-family U-Net at load (sc-4733).
            supports_lora: true,
            supports_lokr: true,
            // `euler_discrete` is the native leading-Euler default; the rest are the unified curated
            // solvers (epic 7114, sc-7121) — the additive k-diffusion path over `DiscreteModelSampling`.
            // Selecting one (or a non-`discrete` scheduler) routes to `Kolors::denoise_curated_latents`,
            // which now covers EVERY mode incl. the conditioned sub-providers (ControlNet-pose,
            // IP-Adapter, the combined pose tier — sc-7297, via `denoise_curated`'s control/ip support);
            // the native default stays byte-exact.
            samplers: {
                let mut s = vec![SAMPLER];
                s.extend([
                    "euler",
                    "euler_ancestral",
                    "heun",
                    "dpmpp_2m",
                    "dpmpp_sde",
                    "uni_pc",
                    "lcm",
                    "ddim",
                ]);
                s
            },
            // `discrete` is the native schedule; the rest are the curated σ schedulers (epic 7114).
            schedulers: {
                let mut s = vec!["discrete"];
                s.extend(curated_scheduler_names());
                s
            },
            min_size: 512,
            max_size: 2048,
            max_count: 8,
            mac_only: true,
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// A loaded, dispatchable Kolors generator: the [`Kolors`] pipeline plus the optionally-loaded
/// ControlNet branch and IP-Adapter image-token encoder (the decoupled-attn K/V pairs are already
/// installed into the U-Net at load).
pub struct KolorsGenerator {
    descriptor: ModelDescriptor,
    kolors: Kolors,
    control: Option<ControlNet>,
    ip_encoder: Option<IpImageEncoder>,
}

/// Build a [`KolorsGenerator`] from a [`LoadSpec`].
///
/// `spec.weights` is a `Kwai-Kolors/Kolors-diffusers` snapshot dir (the multi-component tree with
/// the materialized `tokenizer/tokenizer.json`). Dense runs **fp16** (the SDXL-family production
/// dtype; the VAE stays f32 via `load_vae`). `spec.quantize` ⇒ load-time Q8/Q4 (sc-3096);
/// `spec.control` ⇒ the Kolors ControlNet-Pose checkpoint (sc-3097); `spec.ip_adapter` ⇒ the
/// Kolors-IP-Adapter-Plus snapshot dir (sc-3098), whose K/V pairs are installed into the (pre-quant)
/// U-Net. `spec.adapters` (LoRA/LoKr) ⇒ merged into the dense U-Net before quantization (sc-4733).
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    // fp16 dense path (SDXL-family production dtype). `Precision::Bf16` is the registry's
    // "dense default / no override" sentinel here — NOT a literal bf16 request — mapped to fp16
    // for this SDXL-family loader (see the `Precision` enum note). A precision override is not
    // wired (the VAE is always f32, the rest fp16), so reject it rather than silently ignore.
    if spec.precision != mlx_gen::Precision::Bf16 {
        return Err(Error::Msg(
            "kolors: precision override is not wired; the dense path runs fp16 (SDXL-family \
             production dtype) — drop the precision override"
                .into(),
        ));
    }
    let dtype = Dtype::Float16;
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p.clone(),
            WeightsSource::File(_) => return Err(Error::Msg(
                "kolors expects a Kolors-diffusers snapshot directory (text_encoder/ tokenizer/ \
                 unet/ vae/), not a single .safetensors file"
                    .into(),
            )),
        };
    // Load the dense base, merge any LoRA/LoKr adapters into the dense U-Net, then quantize — the
    // SDXL ordering (sc-4733): the f32 delta lands in the dense weights, which are then packed.
    let mut kolors = Kolors::load(&root, dtype)?;
    if !spec.adapters.is_empty() {
        kolors.apply_lora(&spec.adapters)?;
    }
    if let Some(q) = spec.quantize {
        kolors.quantize(q.bits())?;
    }

    let control = match &spec.control {
        Some(src) => Some(load_controlnet(src, dtype)?),
        None => None,
    };

    let ip_encoder =
        match &spec.ip_adapter {
            Some(WeightsSource::Dir(p)) => {
                let (enc, pairs) = load_kolors_ip_adapter(p, dtype)?;
                kolors.install_ip_adapter(pairs)?;
                Some(enc)
            }
            Some(WeightsSource::File(_)) => return Err(Error::Msg(
                "kolors ip_adapter expects a Kolors-IP-Adapter-Plus snapshot directory, not a file"
                    .into(),
            )),
            None => None,
        };

    Ok(Box::new(KolorsGenerator {
        descriptor: descriptor(),
        kolors,
        control,
        ip_encoder,
    }))
}

mlx_gen::impl_generator!(KolorsGenerator {
    validate: |s, req| s.validate_impl(req),
    generate: generate_impl,
});

impl KolorsGenerator {
    /// The rich-`Result` body behind [`Generator::validate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the family
    /// helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    fn validate_impl(&self, req: &GenerationRequest) -> Result<()> {
        validate_request(&self.descriptor.capabilities, req)?;
        // Mode-combination guards. The Kolors conditioning paths are mutually exclusive EXCEPT the
        // combined strict-pose tier (sc-5012): Control (the pose skeleton) + a Reference (the
        // IP-Adapter identity, which also seeds the img2img init), which is supported when BOTH a
        // ControlNet and an IP-Adapter are loaded.
        let has_ref = req
            .conditioning
            .iter()
            .any(|c| matches!(c, Conditioning::Reference { .. }));
        let has_ctrl = req
            .conditioning
            .iter()
            .any(|c| matches!(c, Conditioning::Control { .. }));
        if has_ctrl && self.control.is_none() {
            return Err(Error::Msg(
                "kolors: a Control conditioning was passed but the model was loaded without a \
                 ControlNet (set LoadSpec::control)"
                    .into(),
            ));
        }
        // Control + Reference is the combined pose tier — allowed ONLY when an IP-Adapter is also
        // loaded (the Reference is the IP identity + img2img init). Plain Control + img2img (a
        // Reference with no IP-Adapter) is not a wired Kolors path.
        if has_ctrl && has_ref && self.ip_encoder.is_none() {
            return Err(Error::Msg(
                "kolors: combining ControlNet (Control) with a Reference requires an IP-Adapter (the \
                 combined pose tier — load LoadSpec::ip_adapter); plain Control + img2img is not \
                 supported in this build"
                    .into(),
            ));
        }
        // A loaded IP-Adapter + Control with no Reference can't run: the combined pass needs the
        // reference as the IP identity (and the IP image prompt is required in IP mode anyway).
        if has_ctrl && self.ip_encoder.is_some() && !has_ref {
            return Err(Error::Msg(
                "kolors: the combined ControlNet + IP-Adapter pass requires a Reference image (the \
                 IP-Adapter identity)"
                    .into(),
            ));
        }
        Ok(())
    }

    /// The rich-`Result` body behind [`Generator::generate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the family
    /// helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;

        let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
        let cfg = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let (h, w) = (req.height as i32, req.width as i32);
        let ip_mode = self.ip_encoder.is_some();

        let reference = self.resolve_reference(req)?;
        let control = self.resolve_control(req)?;
        if ip_mode && reference.is_none() {
            return Err(Error::Msg(
                "kolors: an IP-Adapter is loaded but no Reference image was provided (the Reference \
                 is the image prompt in IP mode)"
                    .into(),
            ));
        }

        // Curated unified-sampler path (epic 7114, sc-7121 + sc-7297): a curated solver name (≠ the
        // native `euler_discrete`) OR a non-`discrete` scheduler routes through the additive k-diffusion
        // `Kolors::denoise_curated_latents`. This now covers EVERY mode — txt2img / img2img AND the
        // conditioned sub-providers (ControlNet-pose, IP-Adapter, the combined pose tier sc-5012): the
        // engine `denoise_curated` already threads ControlNet residuals + the IP-Adapter decoupled-attn
        // tokens (it is the InstantID dual-conditioning path), so the conditioned modes ride the same
        // solver rather than being sampler-locked. The native `euler_discrete` default stays byte-exact
        // — the legacy `denoise_*_latents` assemblies are entered only when no curated knob is set (N1).
        let scheduler_curated = req
            .scheduler
            .as_deref()
            .and_then(Scheduler::from_name)
            .is_some();
        let sampler_curated = req
            .sampler
            .as_deref()
            .map(|s| Solver::from_name(s).is_some() && s != SAMPLER)
            .unwrap_or(false);
        let use_curated = scheduler_curated || sampler_curated;

        // Conditioning is seed-independent — encode the prompts once and hand the (context, pooled)
        // tuples to the per-mode `Kolors::denoise_*_latents` methods, which assemble the CFG batch +
        // time_ids. Routing every mode through those methods keeps a single denoise assembly shared
        // with the struct API + parity gates — only the real cancel/progress differ here (F-146).
        let pos = self.kolors.encode(&req.prompt)?;
        let neg = self.kolors.encode(negative)?;

        // IP-Adapter image tokens are seed-independent — encode the reference once (the
        // `denoise_ip_latents` method CFG-batches them per image). Carries the resolved scale.
        let ip = match (ip_mode, reference) {
            (true, Some((image, strength))) => {
                let tokens = self.ip_encoder.as_ref().unwrap().tokens(image)?;
                Some((tokens, strength.unwrap_or(IP_DEFAULT_SCALE)))
            }
            _ => None,
        };
        // img2img only when a Reference is present AND we're not in IP mode.
        let img2img = match (ip_mode, reference) {
            (false, Some((image, strength))) => Some((
                image,
                strength
                    .or(req.strength)
                    .unwrap_or(DEFAULT_IMG2IMG_STRENGTH),
            )),
            _ => None,
        };

        let (lh, lw) = (h / SPATIAL_SCALE, w / SPATIAL_SCALE);
        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            let seed = base_seed.wrapping_add(i as u64);
            random::seed(seed)?;

            // Draw this image's initial noise, then dispatch to the matching denoise assembly. Only
            // one global-RNG draw happens per image (the noise); the img2img VAE-encode below draws
            // none, so the per-image output stays byte-identical to the struct API's RNG order.
            let noise = random::normal::<f32>(&[1, lh, lw, 4], None, None, None)?;

            // Curated unified path (every mode): k-diffusion VE-σ sampling over the Kolors
            // `DiscreteModelSampling`, additive alongside the native leading-Euler default. Threads the
            // SAME conditioning the bespoke dispatch builds — ControlNet residuals, IP-Adapter tokens,
            // and the combined pose tier's img2img init — through `denoise_curated` (sc-7297).
            if use_curated {
                // ControlNet branch (Some when a Control image was provided + a ControlNet loaded — both
                // validated above) and the IP-Adapter tokens (Some in IP mode), passed straight through.
                let control_arg = control.map(|(image, scale)| {
                    (
                        self.control.as_ref().expect("validated above"),
                        image,
                        scale,
                    )
                });
                let ip_arg = ip.as_ref().map(|(tokens, scale)| (tokens, *scale));

                // Init mirrors the bespoke dispatch's per-mode choice: the combined pose tier seeds
                // img2img from the reference (== the IP image) at the strict-pose strength; plain
                // img2img seeds from its reference; ControlNet-only / IP-only / txt2img seed raw
                // `ε·σ_max` (no init).
                let (init_opt, strength) = if control.is_some() && ip_mode {
                    let (reference_image, _) = reference.expect("ip mode requires a reference");
                    (
                        Some(encode_init_latents(
                            self.kolors.vae(),
                            reference_image,
                            w as u32,
                            h as u32,
                        )?),
                        req.strength.unwrap_or(POSE_IMG2IMG_STRENGTH),
                    )
                } else if let Some((image, strength)) = img2img {
                    (
                        Some(encode_init_latents(
                            self.kolors.vae(),
                            image,
                            w as u32,
                            h as u32,
                        )?),
                        strength,
                    )
                } else {
                    (None, 0.0)
                };

                let latents = self.kolors.denoise_curated_latents(
                    req.sampler.as_deref(),
                    req.scheduler.as_deref(),
                    init_opt.as_ref(),
                    &noise,
                    &pos,
                    &neg,
                    steps,
                    strength,
                    cfg,
                    seed,
                    h,
                    w,
                    control_arg,
                    ip_arg,
                    &req.cancel,
                    on_progress,
                )?;
                on_progress(Progress::Decoding);
                images.push(decode_image(self.kolors.vae(), &latents)?);
                continue;
            }

            let latents = if let (Some((skeleton, control_scale)), Some((tokens, ip_scale))) =
                (control, &ip)
            {
                // Combined strict-pose tier (sc-5012): the pose ControlNet (skeleton) + the
                // IP-Adapter identity, on an img2img init from the SAME reference. `ip_mode` ⇒ a
                // Reference is present (validated), and it is both the IP image prompt and the init.
                let (reference_image, _) = reference.expect("ip mode requires a reference");
                let init_latents =
                    encode_init_latents(self.kolors.vae(), reference_image, w as u32, h as u32)?;
                let strength = req.strength.unwrap_or(POSE_IMG2IMG_STRENGTH);
                self.kolors.denoise_controlnet_ip_latents(
                    self.control.as_ref().expect("validated above"),
                    tokens,
                    &init_latents,
                    &noise,
                    skeleton,
                    &pos,
                    &neg,
                    steps,
                    strength,
                    cfg,
                    control_scale,
                    *ip_scale,
                    h,
                    w,
                    &req.cancel,
                    on_progress,
                )?
            } else if let Some((image, scale)) = control {
                self.kolors.denoise_controlnet_latents(
                    self.control.as_ref().expect("validated above"),
                    &noise,
                    image,
                    &pos,
                    &neg,
                    steps,
                    cfg,
                    scale,
                    h,
                    w,
                    &req.cancel,
                    on_progress,
                )?
            } else if let Some((tokens, scale)) = &ip {
                self.kolors.denoise_ip_latents(
                    tokens,
                    &noise,
                    &pos,
                    &neg,
                    steps,
                    cfg,
                    *scale,
                    h,
                    w,
                    &req.cancel,
                    on_progress,
                )?
            } else if let Some((image, strength)) = img2img {
                let x0 = encode_init_latents(self.kolors.vae(), image, w as u32, h as u32)?;
                self.kolors.denoise_img2img_latents(
                    &x0,
                    &noise,
                    &pos,
                    &neg,
                    steps,
                    strength,
                    cfg,
                    h,
                    w,
                    &req.cancel,
                    on_progress,
                )?
            } else {
                self.kolors.denoise_latents(
                    &noise,
                    &pos,
                    &neg,
                    steps,
                    cfg,
                    h,
                    w,
                    &req.cancel,
                    on_progress,
                )?
            };

            on_progress(Progress::Decoding);
            images.push(decode_image(self.kolors.vae(), &latents)?);
        }
        Ok(GenerationOutput::Images(images))
    }

    /// The single img2img / IP reference image + its strength (the per-reference strength wins). One
    /// reference only; more than one is an error.
    fn resolve_reference<'a>(
        &self,
        req: &'a GenerationRequest,
    ) -> Result<Option<(&'a Image, Option<f32>)>> {
        let mut reference = None;
        for c in &req.conditioning {
            if let Conditioning::Reference { image, strength } = c {
                if reference.is_some() {
                    return Err(Error::Msg(
                        "kolors: multiple reference images are not supported".into(),
                    ));
                }
                reference = Some((image, *strength));
            }
        }
        Ok(reference)
    }

    /// The single ControlNet control image + `conditioning_scale`. One control branch only; the
    /// Kolors ControlNet is pose-trained, so a non-pose `ControlKind` is rejected.
    fn resolve_control<'a>(&self, req: &'a GenerationRequest) -> Result<Option<(&'a Image, f32)>> {
        let mut control = None;
        for c in &req.conditioning {
            if let Conditioning::Control { image, kind, scale } = c {
                if control.is_some() {
                    return Err(Error::Msg(
                        "kolors: multiple control images are not supported".into(),
                    ));
                }
                if !matches!(kind, ControlKind::Pose) {
                    return Err(Error::Msg(format!(
                        "kolors: only Pose ControlNet is wired (got {kind:?})"
                    )));
                }
                control = Some((image, *scale));
            }
        }
        Ok(control)
    }
}

/// Capability-driven request validation (unit-testable without loaded weights).
pub(crate) fn validate_request(caps: &Capabilities, req: &GenerationRequest) -> Result<()> {
    // Shared capability contract: count/size range, negative_prompt/guidance/true_cfg support,
    // sampler, scheduler, and conditioning kinds. Delegating to core keeps Kolors from drifting
    // out of sync with the descriptor (F-132); this was previously a hand-rolled copy that had
    // already lost the negative_prompt/guidance/true_cfg/scheduler checks.
    caps.validate_request(MODEL_ID, req)?;

    // Kolors-specific checks layered on top of the shared contract:
    if req.prompt.is_empty() {
        return Err(Error::Msg("kolors: prompt must not be empty".into()));
    }
    // `steps == 0` divides by zero in `KolorsEulerSampler::new` (`num_train_timesteps / num_steps`),
    // and `steps > 1100` (the train-timestep count) makes `step_ratio == 0` so every timestep
    // collapses to 1 — silent garbage. Reject both at the request boundary (F-124). `None` falls back
    // to DEFAULT_STEPS.
    if let Some(steps) = req.steps {
        if steps == 0 || steps as usize > NUM_TRAIN_TIMESTEPS {
            return Err(Error::Msg(format!(
                "kolors: steps must be in 1..={NUM_TRAIN_TIMESTEPS} (got {steps})"
            )));
        }
    }
    // Kolors VAE downsamples by 8; non-multiple-of-8 dims would mismatch latent shapes.
    if !req.width.is_multiple_of(8) || !req.height.is_multiple_of(8) {
        return Err(Error::Msg(format!(
            "kolors: width/height must be multiples of 8 (got {}x{})",
            req.width, req.height
        )));
    }
    Ok(())
}

// Link-time registration (epic 3720): the macro emits the `inventory::submit!` and bridges the
// crate's rich `Result` into the registry's backend-neutral `gen_core::Result`.
mlx_gen::register_generators! { descriptor => load }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::KolorsEulerSampler;

    #[test]
    fn descriptor_is_kolors() {
        let d = descriptor();
        assert_eq!(d.id, "kolors");
        assert_eq!(d.family, "kolors");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(
            d.capabilities.supports_lora,
            "Kolors LoRA is wired (sc-4733)"
        );
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        assert!(d.capabilities.accepts(ConditioningKind::Control));
        assert!(!d.capabilities.accepts(ConditioningKind::Mask));
    }

    #[test]
    fn registered_in_inventory() {
        // The `inventory::submit!` above is linked into this test binary, so `mlx_gen::load`
        // resolves "kolors" (and fails on the bogus weights dir) — proving registration without
        // needing the real snapshot. A wrong/missing registration yields the registry's
        // "no generator registered for id" error instead.
        let spec = LoadSpec {
            weights: WeightsSource::Dir("/nonexistent/kolors".into()),
            quantize: None,
            precision: mlx_gen::Precision::Bf16,
            control: None,
            ip_adapter: None,
            adapters: Vec::new(),
            extra_controls: Vec::new(),
        };
        let err = match mlx_gen::load("kolors", &spec) {
            Ok(_) => panic!("bogus weights dir must fail to load"),
            Err(e) => e.to_string(),
        };
        assert!(
            !err.contains("no generator registered"),
            "kolors should resolve in the registry; got: {err}"
        );
    }

    #[test]
    fn validate_rejects_bad_steps() {
        // F-124: `steps == 0` would divide by zero in the sampler; `steps > NUM_TRAIN_TIMESTEPS`
        // collapses every timestep to 1. Both must be rejected at the request boundary; `None` and an
        // in-range count pass.
        let caps = descriptor().capabilities;
        let base = GenerationRequest {
            prompt: "a fox".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        for bad in [Some(0), Some(NUM_TRAIN_TIMESTEPS as u32 + 1)] {
            let req = GenerationRequest {
                steps: bad,
                ..base.clone()
            };
            let err = validate_request(&caps, &req).unwrap_err().to_string();
            assert!(err.contains("steps must be in"), "steps={bad:?} got: {err}");
        }
        for ok in [None, Some(1), Some(50), Some(NUM_TRAIN_TIMESTEPS as u32)] {
            let req = GenerationRequest {
                steps: ok,
                ..base.clone()
            };
            assert!(validate_request(&caps, &req).is_ok(), "steps={ok:?}");
        }
    }

    #[test]
    fn sampler_rejects_zero_steps() {
        // The defensive guard in `KolorsEulerSampler::new` (reached via `kolors`) returns a typed error
        // rather than panicking on the divide-by-zero (F-124).
        let err = match KolorsEulerSampler::kolors(0, mlx_rs::Dtype::Float32) {
            Ok(_) => panic!("num_steps == 0 must error, not build a sampler"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("num_steps must be >= 1"), "got: {err}");
    }

    #[test]
    fn validate_delegates_to_core_capability_checks() {
        // F-132: `validate_request` now delegates the shared contract to `Capabilities::validate_request`
        // rather than re-implementing it. Assert the checks the hand-rolled copy had dropped now fire:
        // an unsupported scheduler and a `true_cfg` the descriptor doesn't advertise.
        let caps = descriptor().capabilities;
        let base = GenerationRequest {
            prompt: "a fox".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };

        let bad_scheduler = GenerationRequest {
            scheduler: Some("ddim".into()),
            ..base.clone()
        };
        assert!(
            validate_request(&caps, &bad_scheduler).is_err(),
            "unsupported scheduler must be rejected (delegated to core)"
        );

        let bad_true_cfg = GenerationRequest {
            true_cfg: Some(4.0),
            ..base.clone()
        };
        assert!(
            validate_request(&caps, &bad_true_cfg).is_err(),
            "true_cfg must be rejected — Kolors advertises supports_true_cfg=false"
        );

        // The advertised scheduler still passes.
        let good = GenerationRequest {
            scheduler: Some("discrete".into()),
            ..base
        };
        assert!(validate_request(&caps, &good).is_ok());
    }

    #[test]
    fn validate_accepts_curated_samplers_and_schedulers() {
        // epic 7114 (sc-7121): the unified curated solver + scheduler menu is advertised additively
        // alongside the native `euler_discrete`/`discrete` and accepted by the shared capability check.
        let caps = descriptor().capabilities;
        let base = GenerationRequest {
            prompt: "a fox".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        for s in [
            "euler_discrete",
            "euler",
            "euler_ancestral",
            "heun",
            "dpmpp_2m",
            "dpmpp_sde",
            "uni_pc",
            "lcm",
            "ddim",
        ] {
            assert!(
                validate_request(
                    &caps,
                    &GenerationRequest {
                        sampler: Some(s.into()),
                        ..base.clone()
                    }
                )
                .is_ok(),
                "sampler {s:?} should be accepted"
            );
        }
        for s in [
            "discrete",
            "normal",
            "karras",
            "sgm_uniform",
            "beta",
            "ddim_uniform",
        ] {
            assert!(
                validate_request(
                    &caps,
                    &GenerationRequest {
                        scheduler: Some(s.into()),
                        ..base.clone()
                    }
                )
                .is_ok(),
                "scheduler {s:?} should be accepted"
            );
        }
        // Unknown names are still rejected (delegated to the shared contract).
        assert!(validate_request(
            &caps,
            &GenerationRequest {
                sampler: Some("nonsense".into()),
                ..base.clone()
            }
        )
        .is_err());
        assert!(validate_request(
            &caps,
            &GenerationRequest {
                scheduler: Some("leading".into()),
                ..base
            }
        )
        .is_err());
    }

    #[test]
    fn advertises_lora_adapters() {
        // sc-4733: LoRA/LoKr are wired — merged into the SDXL-family U-Net at load. The descriptor
        // advertises both (the real-weight merge + scale=0≡base parity is `tests/lora_parity.rs`).
        assert!(descriptor().capabilities.supports_lora);
        assert!(descriptor().capabilities.supports_lokr);
    }
}
