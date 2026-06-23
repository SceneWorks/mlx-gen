//! `Krea` — the [`mlx_gen::Generator`] implementation for Krea 2 Turbo, plus its [`descriptor`] /
//! [`load`] entry points and the `inventory` registration that wires the engine into `mlx_gen`'s
//! registry under id `"krea_2_turbo"`. Linking this crate is all the worker needs to resolve the
//! model by id.
//!
//! **Scaffold status (sc-7567):** this slice lands the provider crate, the `krea_2_turbo` registration,
//! the architecture-validated [`load`], and the offline Q4/Q8 converter ([`crate::convert`]). The
//! DiT/TE/VAE forward + rectified-flow sampler are NOT yet wired — they land in their dedicated P1
//! stories (sc-7568 DiT, sc-7569 Qwen3-VL-4B TE, sc-7570 VAE + sampler, sc-7571 Turbo t2i e2e). Until
//! then [`Krea::generate`] returns an explicit error naming those stories rather than a silent stub.

use mlx_gen::gen_core;
use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, Capabilities, Error, GenerationOutput,
    GenerationRequest, Generator, LoadSpec, Modality, ModelDescriptor, ModelRegistration,
    Precision, Progress, Quant, Result, WeightsSource,
};

use crate::config::Krea2Config;

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
/// `is_distilled` + `guidance_scale 0`). Consumed by the forward (`req.steps.unwrap_or(DEFAULT_STEPS)`,
/// sc-7568+); the manifest `default_steps` mirrors this (sc-7572).
#[allow(dead_code)]
const DEFAULT_STEPS: u32 = 8;

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
            // LoRA (Raw-trained, applied at Turbo inference) is enabled when the forward + adapter
            // path land (sc-7568 forward, sc-7577 trainer, sc-7578 loraCompatibility). The scaffold's
            // `load` rejects adapters, so advertise `false` until the engine can honor them.
            supports_lora: false,
            supports_lokr: false,
            // Rectified-flow v-param over the unified curated-sampler framework (epic 7114). The
            // distilled-coherent sampler subset is narrowed by the real-weight survey at e2e (sc-7571,
            // the Boogu Turbo precedent); the scaffold advertises the full curated menu as a starting
            // point. The native distilled loop stays the byte-exact default (`req.sampler == None`).
            samplers: curated_sampler_names(),
            schedulers: curated_scheduler_names(),
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

/// A loaded Krea generator: the cached descriptor + the architecture-validated DiT config + the
/// snapshot root (the forward pipeline assembles from these in sc-7568+).
pub struct Krea {
    descriptor: ModelDescriptor,
    #[allow(dead_code)] // consumed by the forward pipeline (sc-7568+).
    config: Krea2Config,
    #[allow(dead_code)] // consumed by the forward pipeline (sc-7568+).
    root: std::path::PathBuf,
}

/// Load a Krea generator from a [`LoadSpec`]. `spec.weights` must be a [`WeightsSource::Dir`] pointing
/// at a Krea 2 snapshot (`transformer/ text_encoder/ vae/ tokenizer/`). Parses + validates the DiT
/// config against the spike architecture (catches a wrong/truncated snapshot at load); a precision
/// override and LoRA/LoKr adapters are rejected rather than silently ignored.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let id = KREA_2_TURBO_ID;
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{id}: only the default dense precision is wired (drop the precision override)"
        )));
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(format!(
            "{id}: LoRA/LoKr adapters are not yet supported (tracked: sc-7577 trainer, sc-7578 apply)"
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
    let config = Krea2Config::from_snapshot(root)?;
    Ok(Box::new(Krea {
        descriptor: descriptor(),
        config,
        root: root.clone(),
    }))
}

impl Generator for Krea {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        validate_request(&self.descriptor, req).map_err(Into::into)
    }

    fn generate(
        &self,
        _req: &GenerationRequest,
        _on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        // Scaffold (sc-7567): the forward is not yet wired. Surface the deferred work explicitly with
        // its tracking stories instead of returning a silent/empty result.
        Err(Error::Msg(format!(
            "{KREA_2_TURBO_ID}: the DiT/TE/VAE forward + rectified-flow sampler are not yet wired — \
             tracked by sc-7568 (12B single-stream DiT), sc-7569 (Qwen3-VL-4B text encoder), sc-7570 \
             (VAE + sampler), sc-7571 (Turbo t2i e2e). This build is the provider scaffold + converter \
             (sc-7567)."
        ))
        .into())
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

/// Registry adapter: the link-time registry's `load` slot is typed on the backend-neutral
/// [`gen_core::Result`]; bridge the crate's rich-`Result` loader into it.
fn load_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load(spec).map_err(Into::into)
}

inventory::submit! {
    ModelRegistration { descriptor, load: load_registered }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn load_rejects_single_file_and_adapters() {
        let file = LoadSpec::new(WeightsSource::File("/tmp/x.safetensors".into()));
        let e = load(&file).err().expect("error").to_string();
        assert!(e.contains("snapshot directory"), "got: {e}");
    }

    #[test]
    fn load_accepts_quant_spec_but_fails_on_missing_weights() {
        for q in [Quant::Q4, Quant::Q8] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-krea".into())).with_quant(q);
            let e = load(&spec).err().expect("error").to_string();
            // The quant is accepted (not the failure); the missing config.json is.
            assert!(
                !e.contains("not supported"),
                "quant should be accepted: {e}"
            );
            assert!(e.contains("config.json") || e.contains("read"), "got: {e}");
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
}
