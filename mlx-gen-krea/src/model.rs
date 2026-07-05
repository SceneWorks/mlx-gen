//! `Krea` — the [`mlx_gen::Generator`] implementation for Krea 2 Turbo, plus its [`descriptor`] /
//! [`load`] entry points and the `inventory` registration that wires the engine into `mlx_gen`'s
//! registry under id `"krea_2_turbo"`. Linking this crate is all the worker needs to resolve the
//! model by id.
//!
//! **Status (P1 complete):** the provider crate + `krea_2_turbo` registration + architecture-validated
//! [`load`] + offline Q4/Q8 converter ([`crate::convert`]) landed in sc-7567; the DiT forward in
//! sc-7568 ([`crate::transformer`]); the Qwen3-VL-4B text encoder in sc-7569 ([`crate::text_encoder`]);
//! the VAE + rectified-flow sampler in sc-7570 ([`crate::vae`] / [`crate::schedule`]); and the
//! end-to-end Turbo t2i [`crate::pipeline`] in sc-7571. [`Krea::generate`] now renders real images
//! (CFG-free, few-step) through the assembled tokenizer → TE → DiT → VAE pipeline.

use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, default_seed, Capabilities, Error,
    GenerationOutput, GenerationRequest, Generator, LatentDecoder, LoadSpec, Modality,
    ModelDescriptor, Precision, Progress, Quant, Result, WeightsSource,
};
use mlx_gen_pid::{flow_capture_for_request, resolve_pid_decoder_at_sigma, PidEngine};
use mlx_gen_qwen_image::pipeline::PID_BACKBONE;

use std::path::Path;

use crate::pipeline::{base_schedule, turbo_schedule, KreaPipeline, TurboOptions};

/// Read the on-disk packed-quantization bits from `transformer/config.json` for a pre-quantized
/// (Group-B packed) Krea turnkey (`"quantization": {"bits", "group_size"}`); `None` for dense.
fn packed_quant_bits(root: &Path) -> Option<i32> {
    let cfg = std::fs::read(root.join("transformer").join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_slice(&cfg).ok()?;
    v.get("quantization")?
        .get("bits")?
        .as_i64()
        .map(|b| b as i32)
}

/// Registry id for the Krea 2 Turbo text-to-image variant. Matches the SceneWorks worker's
/// `payload.model` and the manifest `engine_id` (sc-7572).
pub const KREA_2_TURBO_ID: &str = "krea_2_turbo";

/// Max images per request (the image-model standard, shared with the other MLX families).
const MAX_COUNT: u32 = 8;
/// Resolution bounds (W/H). Turbo renders up to 2048²; the catalog/worker gate the UI options tighter.
const RES_MIN: u32 = 256;
const RES_MAX: u32 = 2048;
/// patch_size(2)·vae_downsample(8) = 16 — patchify requires W/H divisible by this.
const RES_MULTIPLE: u32 = 16;

/// Turbo defaults: the TDM-distilled few-step student renders CFG-free at 8 steps (reference
/// `is_distilled` + `guidance_scale 0`). Consumed by `generate` (`req.steps.unwrap_or(DEFAULT_STEPS)`);
/// the manifest `default_steps` mirrors this (sc-7572).
const DEFAULT_STEPS: u32 = 8;

/// Registry id for the undistilled **Raw** text-to-image variant (epic 9992). The SAME string as the
/// Krea LoRA *trainer* base ([`crate::training::KREA_2_RAW_TRAINER_ID`]) — Path 1 makes one id both the
/// training base and a first-class generator; the trainer + generator live in separate registries so
/// the shared id never collides. Matches the SceneWorks worker's `payload.model` + manifest `engine_id`.
pub const KREA_2_RAW_ID: &str = "krea_2_raw";

/// Raw defaults (the reference `sampling.py` Raw preset per the sc-7566 spike): full-CFG at 52 steps,
/// guidance 3.5, resolution-dynamic mu. Consumed by `generate_impl`
/// (`req.steps.unwrap_or(DEFAULT_RAW_STEPS)` / `req.guidance.unwrap_or(DEFAULT_RAW_GUIDANCE)`); the
/// manifest `default_steps` / `defaults.guidanceScale` mirror these (sc-9999 / sc-10003).
const DEFAULT_RAW_STEPS: u32 = 52;
const DEFAULT_RAW_GUIDANCE: f32 = 3.5;

/// Krea 2 Turbo identity + capabilities — constructible without loading weights (registry
/// introspection / capability advertisement). Distilled few-step text-to-image: **CFG-free** (the TDM
/// distillation baked the guided velocity into the weights, so no unconditional branch / `guidance`),
/// no user negative prompt, no img2img/control conditioning on the Turbo checkpoint.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: KREA_2_TURBO_ID,
        family: "krea_2",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: false,
            // CFG-free distilled student (like Ideogram Turbo / Boogu Turbo / SDXL-Lightning).
            supports_guidance: false,
            supports_true_cfg: false,
            // Turbo is text-to-image only.
            conditioning: Vec::new(),
            // LoRA/LoKr trained on the undistilled Raw DiT (sc-7577) apply at Turbo inference via the
            // shared `apply_adapters_strict` seam onto the `Krea2Transformer` adapter host (sc-7911).
            // Family-match cross-apply, no base-model gating (the Lens / Z-Image precedent).
            supports_lora: true,
            supports_lokr: true,
            // Rectified-flow v-param over the unified curated-sampler framework (epic 7114). The
            // distilled-coherent sampler subset is narrowed by the real-weight survey at e2e (sc-7571,
            // the Boogu Turbo precedent); the scaffold advertises the full curated menu as a starting
            // point. The native distilled loop stays the byte-exact default (`req.sampler == None`).
            samplers: curated_sampler_names(),
            schedulers: curated_scheduler_names(),
            supported_guidance_methods: vec![],
            min_size: RES_MIN,
            max_size: RES_MAX,
            max_count: MAX_COUNT,
            mac_only: true,
            // The turnkey ships pre-packed Q8/Q4 ([`crate::convert::assemble_quantized_snapshot`]);
            // load-time quantize over a dense bf16 build is a no-op on an already-packed snapshot.
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// Krea 2 **Raw** identity + capabilities — the undistilled 12B DiT run with **true classifier-free
/// guidance** (two DiT forwards/step: cond vs uncond) at 52 steps, unlike the CFG-free distilled Turbo.
/// Same architecture / snapshot layout as Turbo (only the DiT weights differ, distilled vs base), so it
/// shares [`load_variant`] + the whole [`KreaPipeline`]. Exposes a real guidance scale AND a user
/// negative prompt — the reference `sample()` accepts `negative_prompts` (richer than Boogu's base,
/// which fixes the uncond to the empty prompt). NOT guidance-distilled, so `supports_true_cfg` stays
/// false: there is no separate embedded-guidance axis to layer a `true_cfg_scale` over — the two-forward
/// CFG IS the guidance (the Boogu-base precedent). Derived from [`descriptor`] so the shared surface
/// (family/backend/samplers/quants/size/LoRA) stays in lockstep.
pub fn raw_descriptor() -> ModelDescriptor {
    let mut d = descriptor();
    d.id = KREA_2_RAW_ID;
    d.capabilities.supports_negative_prompt = true;
    d.capabilities.supports_guidance = true;
    d.capabilities.supports_true_cfg = false;
    d
}

/// A loaded Krea 2 generator (Turbo or Raw): the cached descriptor + the assembled pipeline (tokenizer +
/// Qwen3-VL-4B condition encoder + single-stream DiT + Qwen-Image VAE). The variant is read back off
/// `descriptor.id` at generate time (Turbo = CFG-free distilled; Raw = full-CFG undistilled).
pub struct Krea {
    descriptor: ModelDescriptor,
    pipeline: KreaPipeline,
    /// Optional PiD super-resolving decoder (epic 7840, sc-7845), loaded when `spec.pid` is set; Krea
    /// reuses the Qwen-Image latent space, so it shares the `qwenimage` PiD student. `req.use_pid`
    /// routes decode through it instead of the VAE. `None` for the plain VAE path.
    pid: Option<PidEngine>,
}

/// Load a Krea generator from a [`LoadSpec`]. `spec.weights` must be a [`WeightsSource::Dir`] pointing
/// at a Krea 2 snapshot (`transformer/ text_encoder/ vae/ tokenizer/`). Parses + validates the DiT
/// config against the spike architecture (catches a wrong/truncated snapshot at load); a precision
/// override is rejected rather than silently ignored. Raw-trained LoRA/LoKr adapters in `spec.adapters`
/// are installed onto the DiT (sc-7911).
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(spec, descriptor())
}

/// Load the undistilled **Raw** generator (`krea_2_raw`, epic 9992). Identical snapshot assembly to
/// [`load`] — the Raw + Turbo turnkeys share the exact architecture / weight layout (only distilled-vs-
/// base DiT weights differ), so one loader serves both — but stores the CFG-capable [`raw_descriptor`]
/// so `generate` runs the full-CFG path.
pub fn load_raw(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(spec, raw_descriptor())
}

/// Shared loader behind [`load`] / [`load_raw`]: assemble the pipeline from a snapshot dir, install any
/// Raw-trained LoRA/LoKr adapters, apply the optional (F-076-guarded) quantize, and overlay a PiD
/// decoder. `descriptor` selects the variant (Turbo vs Raw) the returned [`Krea`] renders.
fn load_variant(spec: &LoadSpec, descriptor: ModelDescriptor) -> Result<Box<dyn Generator>> {
    let id = descriptor.id;
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{id}: only the default dense precision is wired (drop the precision override)"
        )));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(Error::Msg(format!(
                "{id} expects a snapshot directory (transformer/ text_encoder/ vae/), not a single \
                 .safetensors file"
            )))
        }
    };
    // Assemble the full Turbo pipeline (tokenizer + TE + DiT + VAE); auto-detects a packed Q4/Q8
    // turnkey vs a dense bf16 snapshot. `spec.quantize` then quantizes the dense base in place (a no-op
    // on an already-packed snapshot — `AdaptableLinear::quantize` skips quantized bases).
    let mut pipeline = KreaPipeline::from_snapshot(root)?;
    // Install Raw-trained LoRA/LoKr adapters onto the DiT BEFORE the optional quantize, so the
    // residual stacks over the (possibly already-packed) base — the Lens load→apply→quantize order.
    // The shared seam errors (never silently drops) on an adapter target that matches no module.
    if !spec.adapters.is_empty() {
        pipeline.apply_adapters(&spec.adapters)?;
    }
    if let Some(q) = spec.quantize {
        // F-076: on an already-packed turnkey, `pipeline.quantize()` is a no-op, so e.g. Q4 over a
        // Q8 turnkey would silently serve Q8. Compare the requested bits against the config.json
        // `quantization.bits` marker the Group-B converter writes; error on mismatch. A dense
        // snapshot (no marker) takes the ordinary in-place quantize.
        if let Some(packed) = packed_quant_bits(root) {
            if packed != q.bits() {
                return Err(Error::Msg(format!(
                    "{id}: snapshot is a pre-quantized Q{packed} turnkey but Q{} was \
                     requested; quantize() is a no-op on packed weights so the request would \
                     silently serve Q{packed}. Point at a Q{} snapshot (or a dense one).",
                    q.bits(),
                    q.bits()
                )));
            }
        } else {
            pipeline.quantize(q.bits())?;
        }
    }
    // Optional PiD decoder overlay (sc-7845): Krea reuses the Qwen-Image latent space, so it loads the
    // same `qwenimage` student + Gemma-2 caption encoder when `spec.pid` is set.
    let pid = spec
        .pid
        .as_ref()
        .map(|p| PidEngine::from_spec(p, PID_BACKBONE))
        .transpose()?;
    Ok(Box::new(Krea {
        descriptor,
        pipeline,
        pid,
    }))
}

mlx_gen::impl_generator!(Krea {
    validate: |s, req| validate_request(&s.descriptor, req),
    generate: generate_impl,
});

impl Krea {
    /// The rich-`Result` body behind [`Generator::generate`] — kept on the crate's own
    /// [`mlx_gen::Error`] so `?` lifts `mlx_rs` device exceptions transparently; the trait wrapper
    /// bridges the tail into [`gen_core::Error`]. Renders `req.count` CFG-free Turbo images, one per
    /// seed (`seed + n`, mirroring the reference per-prompt seeding).
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        validate_request(&self.descriptor, req)?;
        let base_seed = req.seed.unwrap_or_else(default_seed);
        // Variant read back off the descriptor id: Raw = full-CFG undistilled (52-step, dynamic-mu);
        // Turbo = CFG-free distilled (8-step, fixed mu). One `Krea` struct, two render paths.
        let is_raw = self.descriptor.id == KREA_2_RAW_ID;
        let steps = req.steps.unwrap_or(if is_raw {
            DEFAULT_RAW_STEPS
        } else {
            DEFAULT_STEPS
        }) as usize;
        // Decode seam (sc-7845) + `from_ldm` early-stop (sc-7993): when `req.use_pid`, build one PiD
        // decoder from the prompt and reuse it across the batch — same prompt → same caption; per-image
        // variation comes from the per-seed latent. `None` → the native VAE. Errors if PiD was requested
        // but not loaded. With `req.pid_capture_sigma`, resolve the achieved degrade σ + the truncation
        // `keep` from the (seed-independent) schedule and decode the partially-denoised x_k; else the
        // clean σ=0 full-denoise path (`capture_sigma = 0`, `keep = MAX`). Raw uses the resolution-
        // dynamic schedule; both share the Qwen-Image latent space the PiD student decodes.
        let sigmas = if is_raw {
            base_schedule(steps, req.width, req.height, req.scheduler.as_deref())
        } else {
            turbo_schedule(steps, req.scheduler.as_deref())
        };
        let (capture_sigma, keep) = flow_capture_for_request(req, &sigmas, 0);
        let pid_decoder = resolve_pid_decoder_at_sigma(
            self.pid.as_ref(),
            req,
            base_seed,
            self.descriptor.id,
            capture_sigma,
        )?;
        let decoder = pid_decoder.as_ref().map(|d| d as &dyn LatentDecoder);
        // Raw CFG knobs: guidance defaults to the reference Raw preset, an empty/absent negative → ""
        // (reference `negative_prompts = [""] * n`). Inert on the Turbo (CFG-free) path.
        let guidance = req.guidance.unwrap_or(DEFAULT_RAW_GUIDANCE);
        let negative = req.negative_prompt.clone().unwrap_or_default();
        let mut images = Vec::with_capacity(req.count as usize);
        for n in 0..req.count {
            let opts = TurboOptions {
                width: req.width,
                height: req.height,
                steps,
                seed: base_seed.wrapping_add(n as u64),
                sampler: req.sampler.clone(),
                scheduler: req.scheduler.clone(),
            };
            let img = if is_raw {
                self.pipeline.generate_base_with_progress(
                    &req.prompt,
                    &negative,
                    guidance,
                    &opts,
                    decoder,
                    keep,
                    &req.cancel,
                    on_progress,
                )?
            } else {
                self.pipeline.generate_turbo_with_progress(
                    &req.prompt,
                    &opts,
                    decoder,
                    keep,
                    &req.cancel,
                    on_progress,
                )?
            };
            images.push(img);
        }
        Ok(GenerationOutput::Images(images))
    }
}

/// Capability-driven request validation, factored out so it can be unit-tested without loaded weights.
/// Layers Krea's model-specific constraints (non-empty prompt, size multiple-of-16, steps ≥ 1) on top
/// of the shared [`Capabilities::validate_request`] floor (count/size range, negative/guidance/true_cfg
/// flags, conditioning kinds).
pub(crate) fn validate_request(desc: &ModelDescriptor, req: &GenerationRequest) -> Result<()> {
    let id = desc.id;
    if req.prompt.is_empty() {
        return Err(Error::Msg(format!("{id}: prompt must not be empty")));
    }
    desc.capabilities.validate_request(id, req)?;
    if req.steps == Some(0) {
        return Err(Error::Msg(format!("{id}: steps must be >= 1")));
    }
    if !req.width.is_multiple_of(RES_MULTIPLE) || !req.height.is_multiple_of(RES_MULTIPLE) {
        return Err(Error::Msg(format!(
            "{id}: {}x{} must be a multiple of {RES_MULTIPLE}",
            req.width, req.height
        )));
    }
    Ok(())
}

// Link-time registration (epic 3720): the macro emits the `inventory::submit!` and bridges the
// crate's rich `Result` into the registry's backend-neutral `gen_core::Result`. Both variants register
// here — `krea_2_turbo` (distilled, CFG-free) and `krea_2_raw` (undistilled, full-CFG; epic 9992).
mlx_gen::register_generators! {
    descriptor => load,
    raw_descriptor => load_raw,
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core;
    use mlx_gen::{AdapterKind, AdapterSpec};

    fn req(w: u32, h: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "a red apple on a wooden table".into(),
            width: w,
            height: h,
            ..Default::default()
        }
    }

    #[test]
    fn descriptor_is_krea_2_turbo() {
        let d = descriptor();
        assert_eq!(d.id, "krea_2_turbo");
        assert_eq!(d.family, "krea_2");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        // CFG-free distilled Turbo: no guidance, no negative prompt, no conditioning surface.
        assert!(!d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.conditioning.is_empty());
        // Raw-trained LoRA/LoKr apply at Turbo inference (sc-7911).
        assert!(d.capabilities.supports_lora);
        assert!(d.capabilities.supports_lokr);
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert_eq!(DEFAULT_STEPS, 8);
        assert!(d.capabilities.mac_only);
    }

    #[test]
    fn validate_accepts_in_surface() {
        assert!(validate_request(&descriptor(), &req(1024, 1024)).is_ok());
        assert!(validate_request(&descriptor(), &req(2048, 2048)).is_ok());
    }

    #[test]
    fn validate_rejects_empty_prompt_and_bad_size() {
        assert!(validate_request(&descriptor(), &GenerationRequest::default()).is_err());
        for (w, h) in [(1000, 1000), (257, 256)] {
            let e = validate_request(&descriptor(), &req(w, h))
                .unwrap_err()
                .to_string();
            assert!(e.contains("multiple of 16"), "{w}x{h} got: {e}");
        }
        assert!(validate_request(&descriptor(), &req(128, 128)).is_err()); // below min
        assert!(validate_request(&descriptor(), &req(2064, 256)).is_err()); // above max
    }

    #[test]
    fn validate_rejects_guidance_and_negative_prompt() {
        // Turbo is CFG-free: the capability floor rejects a guidance override and a negative prompt.
        assert!(validate_request(
            &descriptor(),
            &GenerationRequest {
                guidance: Some(3.5),
                ..req(512, 512)
            }
        )
        .is_err());
        assert!(validate_request(
            &descriptor(),
            &GenerationRequest {
                negative_prompt: Some("x".into()),
                ..req(512, 512)
            }
        )
        .is_err());
    }

    #[test]
    fn load_rejects_single_file() {
        let file = LoadSpec::new(WeightsSource::File("/tmp/x.safetensors".into()));
        let e = load(&file).err().expect("error").to_string();
        assert!(e.contains("snapshot directory"), "got: {e}");
    }

    #[test]
    fn load_accepts_adapter_spec_without_rejecting() {
        // sc-7911: adapters are no longer rejected at the door; a LoadSpec carrying an adapter
        // resolves the snapshot first, so a missing snapshot — not an "unsupported adapters" error —
        // is what surfaces (the real install runs in the #[ignore] real-weight harness).
        let spec =
            LoadSpec::new(WeightsSource::Dir("/nonexistent-krea".into())).with_adapters(vec![
                AdapterSpec::new(
                    std::path::PathBuf::from("/nonexistent-krea/adapter.safetensors"),
                    1.0,
                    AdapterKind::Lora,
                ),
            ]);
        let e = load(&spec).err().expect("error").to_string();
        assert!(
            !e.to_lowercase().contains("not yet supported")
                && !e.to_lowercase().contains("not supported"),
            "adapters must be accepted, got: {e}"
        );
    }

    #[test]
    fn load_accepts_quant_spec_but_fails_on_missing_weights() {
        for q in [Quant::Q4, Quant::Q8] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-krea".into())).with_quant(q);
            let e = load(&spec).err().expect("error").to_string();
            // The quant is accepted (not the failure); the missing snapshot (the pipeline assembly
            // hits the absent tokenizer/config first) is.
            assert!(
                !e.contains("not supported"),
                "quant should be accepted: {e}"
            );
            assert!(
                e.contains("No such file")
                    || e.contains("config.json")
                    || e.contains("tokenizer")
                    || e.contains("read"),
                "expected a missing-snapshot error, got: {e}"
            );
        }
    }

    #[test]
    fn reachable_via_registry_by_id() {
        assert!(
            gen_core::registry::generators().any(|r| (r.descriptor)().id == KREA_2_TURBO_ID),
            "id {KREA_2_TURBO_ID} not registered"
        );
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-krea".into()));
        let e = gen_core::registry::load(KREA_2_TURBO_ID, &spec)
            .err()
            .expect("missing weights → err")
            .to_string();
        assert!(
            !e.contains("no generator registered"),
            "id not resolved: {e}"
        );
    }

    // --- Raw (undistilled, full-CFG) variant — epic 9992 ---

    #[test]
    fn raw_descriptor_is_krea_2_raw_and_cfg_capable() {
        let d = raw_descriptor();
        assert_eq!(d.id, "krea_2_raw");
        // The generator id MUST equal the LoRA-trainer base id (Path 1: one id, both roles).
        assert_eq!(KREA_2_RAW_ID, crate::training::KREA_2_RAW_TRAINER_ID);
        assert_eq!(d.family, "krea_2");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        // Undistilled base: real CFG guidance + a user negative prompt (unlike Turbo / Boogu base).
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        // Not guidance-distilled: no separate embedded-guidance axis, so no true_cfg toggle.
        assert!(!d.capabilities.supports_true_cfg);
        // Shared surface stays in lockstep with Turbo (derived from `descriptor()`).
        assert!(d.capabilities.supports_lora && d.capabilities.supports_lokr);
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert_eq!(d.capabilities.samplers, descriptor().capabilities.samplers);
        assert!(d.capabilities.mac_only);
        assert_eq!(DEFAULT_RAW_STEPS, 52);
        assert_eq!(DEFAULT_RAW_GUIDANCE, 3.5);
    }

    #[test]
    fn raw_validate_accepts_guidance_and_negative_prompt() {
        // The CFG floor that rejects these on Turbo must ACCEPT them on Raw.
        assert!(validate_request(
            &raw_descriptor(),
            &GenerationRequest {
                guidance: Some(3.5),
                negative_prompt: Some("blurry, lowres".into()),
                ..req(1024, 1024)
            }
        )
        .is_ok());
    }

    #[test]
    fn raw_reachable_via_registry_by_id() {
        assert!(
            gen_core::registry::generators().any(|r| (r.descriptor)().id == KREA_2_RAW_ID),
            "id {KREA_2_RAW_ID} not registered"
        );
        // Same snapshot loader as Turbo — a single-file weights source is rejected the same way.
        let file = LoadSpec::new(WeightsSource::File("/tmp/x.safetensors".into()));
        let e = load_raw(&file).err().expect("error").to_string();
        assert!(e.contains("snapshot directory"), "got: {e}");
    }
}
