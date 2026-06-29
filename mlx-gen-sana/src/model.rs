//! `Sana` — the [`mlx_gen::Generator`] implementation for SANA-1.6B 1024px, plus its
//! [`descriptor`]/[`load`] entry points and the `inventory` registration that wires it into
//! `mlx_gen`'s registry under the id `"sana_1600m"` (epic 8485, story sc-8489 **Phase B**).
//!
//! Phase A (sc-8486..8489 on mlx-gen) built the three native components and the composed
//! [`crate::pipeline::SanaPipeline`]; this module is the thin gen-core `Generator` adapter the
//! SceneWorks worker links and drives end-to-end. Linking this crate is all the worker needs to
//! resolve the model by id (the `inventory::submit!` below registers `descriptor`/[`load`]).
//!
//! ## Snapshot layout
//!
//! [`load`] assembles the pipeline from a `Sana_1600M_1024px_diffusers`-shaped snapshot directory
//! (the SceneWorks `SceneWorks/Sana_1600M_1024px_mlx` mirror ships this exact tree):
//!
//! ```text
//!   transformer/diffusion_pytorch_model.safetensors   → SanaTransformer   (the Linear-DiT trunk)
//!   vae/diffusion_pytorch_model.safetensors           → DcAeDecoder       (DC-AE f32c32 decoder)
//!   text_encoder/gemma-2-2b-it.safetensors            → SanaTextEncoder   (gemma-2-2b-it CHI TE)
//!   text_encoder/tokenizer.json                       ↗ (bundled gemma TE, from the un-gated
//!                                                        SceneWorks/gemma-2-2b-it mirror — epic 7840)
//! ```
//!
//! The gemma-2-2b-it caption encoder is bundled under `text_encoder/` exactly as LTX bundles its
//! gemma TE (the worker points the engine at the snapshot, no separate gemma download), so a single
//! [`WeightsSource::Dir`] is a complete, self-contained SANA load.
//!
//! ## Sampling recipe
//!
//! SANA-1.6B is a **true-CFG** flow-match model: default **20 steps / guidance 4.5** (diffusers
//! `SanaPipeline.__call__`), negative prompt supported, flow-match Euler over a static shift 3.0
//! schedule routed through the unified epic-7114 sampler. When `guidance <= 1.0` the uncond forward
//! is skipped (CFG off), matching diffusers' `do_classifier_free_guidance = guidance_scale > 1.0`.

use std::path::Path;

use mlx_gen::weights::Weights;
use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, default_seed, Capabilities, Error,
    GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality, ModelDescriptor, Precision,
    Progress, Result, WeightsSource,
};

use crate::config::{DcAeConfig, SanaTransformerConfig};
use crate::dc_ae::DcAeDecoder;
use crate::pipeline::{SanaGenerateRequest, SanaPipeline};
use crate::text_encoder::SanaTextEncoder;
use crate::transformer::SanaTransformer;

/// Registry id for SANA-1.6B 1024px (matches the SceneWorks worker's `payload.model`).
pub const MODEL_ID: &str = "sana_1600m";

/// SANA-1.6B's native generation resolution (the model is bucket-trained at 1024²; the catalog
/// gates the exposed buckets tighter, this is the engine validation range).
const RES_MIN: u32 = 256;
const RES_MAX: u32 = 2048;
/// DC-AE 32× spatial compression — requested dims must be a multiple of this so the latent edge
/// (`image / 32`) is integral.
const RES_MULTIPLE: u32 = crate::pipeline::SPATIAL_SCALE;
/// Max images per request (the image-model standard, shared with the other MLX families).
const MAX_COUNT: u32 = 8;

/// SANA-1.6B's identity + capabilities — constructible without loading weights (registry
/// introspection / capability advertisement). True-CFG text-to-image: negative prompt + guidance
/// scale, flow-match Euler over the unified curated sampler/scheduler menu (epic 7114). No img2img /
/// control conditioning is wired yet (plain txt2img — the Sprint CFG-free distill is a later story).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "sana",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: true,
            // Plain txt2img — no img2img/control conditioning on the base SANA checkpoint.
            conditioning: Vec::new(),
            // No SANA LoRA wiring yet (reserved for a later story).
            supports_lora: false,
            supports_lokr: false,
            // Flow-match Euler over the unified curated sampler/scheduler framework (epic 7114); the
            // native loop (`req.sampler == None`) stays the byte-exact default. `"default"` is the
            // engine-default sentinel the manifest drift guard always allows.
            samplers: {
                let mut s = curated_sampler_names();
                s.push("default");
                s
            },
            schedulers: {
                let mut s = curated_scheduler_names();
                s.push("default");
                s
            },
            supported_guidance_methods: vec![],
            min_size: RES_MIN,
            max_size: RES_MAX,
            max_count: MAX_COUNT,
            mac_only: true,
            // SANA is the bf16/fp16 weight path; the 2-bit Clark Labs quant is intentionally NOT
            // ported. No load-time quantization is wired for SANA yet — leave the set empty so the
            // catalog never records a quant the engine cannot honor.
            supported_quants: &[],
            supports_kv_cache: false,
            // Static flow-match shift 3.0, resolution-independent (handled by the unified sampler).
            requires_sigma_shift: false,
        },
    }
}

/// A loaded SANA generator: the composed pipeline plus the cached descriptor.
pub struct Sana {
    descriptor: ModelDescriptor,
    pipeline: SanaPipeline,
}

/// Construct a SANA generator from a [`LoadSpec`]. `spec.weights` must be a [`WeightsSource::Dir`]
/// pointing at a `Sana_1600M_1024px_diffusers`-shaped snapshot (`transformer/ vae/ text_encoder/`).
/// A precision override, load-time quantization, or LoRA/LoKr adapters are rejected rather than
/// silently ignored (none are wired for SANA yet).
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let descriptor = descriptor();
    let id = descriptor.id;
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{id}: only the default dense precision is wired (drop the precision override)"
        )));
    }
    if spec.quantize.is_some() {
        return Err(Error::Msg(format!(
            "{id}: load-time quantization is not supported (the 2-bit quant is not ported)"
        )));
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(format!(
            "{id}: LoRA/LoKr adapters are not supported"
        )));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.as_path(),
        WeightsSource::File(_) => {
            return Err(Error::Msg(format!(
                "{id} expects a snapshot directory (transformer/ vae/ text_encoder/), not a \
                 single .safetensors file"
            )))
        }
    };
    let pipeline = build_pipeline(root)?;
    Ok(Box::new(Sana {
        descriptor,
        pipeline,
    }))
}

/// Assemble the [`SanaPipeline`] from the snapshot tree — factored out so the load path is a single
/// `?`-threaded body and the snapshot layout lives in one place. `from_dir` is used for the
/// transformer/VAE subdirs so a sharded checkpoint loads transparently; the text encoder reuses
/// [`SanaTextEncoder::from_snapshot`] (the bundled gemma weights + `tokenizer.json`).
fn build_pipeline(root: &Path) -> Result<SanaPipeline> {
    let trunk_w = Weights::from_dir(root.join("transformer"))?;
    let trunk = SanaTransformer::from_weights(&trunk_w, SanaTransformerConfig::sana_1600m())?;

    let dcfg = DcAeConfig::sana_f32c32();
    let vae_w = Weights::from_dir(root.join("vae"))?;
    let decoder = DcAeDecoder::from_weights(&vae_w, dcfg.clone())?;

    let te = SanaTextEncoder::from_snapshot(root.join("text_encoder"))?;

    Ok(SanaPipeline::new(te, trunk, decoder, dcfg))
}

/// Capability-driven request validation, factored out so it can be unit-tested without loaded
/// weights. Delegates the shared size/count/guidance/negative/conditioning checks to the descriptor
/// (`Capabilities::validate_request`) and adds SANA's `RES_MULTIPLE` (32×, DC-AE) divisor rule.
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
            "{id}: {}x{} must be a multiple of {RES_MULTIPLE} (DC-AE 32× spatial compression)",
            req.width, req.height
        )));
    }
    Ok(())
}

mlx_gen::impl_generator!(Sana {
    validate: |s, req| validate_request(&s.descriptor, req),
    generate: generate_impl,
});

impl Sana {
    /// The rich-`Result` body behind [`Generator::generate`] — kept on the crate's own
    /// [`mlx_gen::Error`] so `?` lifts `mlx_rs` device exceptions transparently; the trait wrapper
    /// bridges the tail into [`gen_core::Error`]. Runs the composed [`SanaPipeline`] once per
    /// requested image, deriving each image's seed from the base seed so a `count > 1` batch is
    /// reproducible and distinct.
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        validate_request(&self.descriptor, req)?;

        let base_seed = req.seed.unwrap_or_else(default_seed);
        let steps = req.steps.map(|s| s as usize);
        let mut images = Vec::with_capacity(req.count as usize);
        for n in 0..req.count {
            let seed = base_seed.wrapping_add(n as u64);
            let sana_req = SanaGenerateRequest {
                prompt: &req.prompt,
                negative_prompt: req.negative_prompt.as_deref(),
                height: req.height,
                width: req.width,
                steps,
                guidance_scale: req.guidance,
                seed: Some(seed),
                sampler: req.sampler.as_deref(),
                scheduler: req.scheduler.as_deref(),
            };
            let img = self
                .pipeline
                .generate_with(&sana_req, &req.cancel, on_progress)?;
            images.push(img);
        }
        Ok(GenerationOutput::Images(images))
    }
}

// Link-time registration (epic 3720): the macro emits the `inventory::submit!` and bridges the
// crate's rich `Result` into the registry's backend-neutral `gen_core::Result`.
mlx_gen::register_generators! {
    descriptor => load,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{DEFAULT_GUIDANCE, DEFAULT_STEPS};
    use mlx_gen::{gen_core, Quant};

    fn req(w: u32, h: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "a red panda on a mossy log in a misty forest".into(),
            width: w,
            height: h,
            ..Default::default()
        }
    }

    #[test]
    fn descriptor_is_sana_1600m() {
        let d = descriptor();
        assert_eq!(d.id, "sana_1600m");
        assert_eq!(d.family, "sana");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.supports_true_cfg);
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.conditioning.is_empty());
        assert!(d.capabilities.supported_quants.is_empty());
        assert!(d.capabilities.mac_only);
    }

    #[test]
    fn descriptor_defaults_match_diffusers() {
        // The worker reads steps/guidance defaults from the catalog (MODEL_TABLE), but the engine's
        // own diffusers-parity defaults are the source of truth they mirror.
        assert_eq!(DEFAULT_STEPS, 20);
        assert!((DEFAULT_GUIDANCE - 4.5).abs() < 1e-6);
    }

    #[test]
    fn validate_accepts_1024_square() {
        let d = descriptor();
        assert!(validate_request(&d, &req(1024, 1024)).is_ok());
    }

    #[test]
    fn validate_rejects_empty_prompt() {
        let d = descriptor();
        let mut r = req(1024, 1024);
        r.prompt.clear();
        assert!(validate_request(&d, &r).is_err());
    }

    #[test]
    fn validate_rejects_non_multiple_of_32() {
        let d = descriptor();
        // 1024 % 32 == 0, 1000 % 32 != 0.
        assert!(validate_request(&d, &req(1000, 1024)).is_err());
    }

    #[test]
    fn validate_rejects_zero_steps() {
        let d = descriptor();
        let mut r = req(1024, 1024);
        r.steps = Some(0);
        assert!(validate_request(&d, &r).is_err());
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/x.safetensors".into()));
        let e = load(&spec).err().expect("error").to_string();
        assert!(e.contains("snapshot directory"), "got: {e}");
    }

    #[test]
    fn load_rejects_quantization() {
        let spec =
            LoadSpec::new(WeightsSource::Dir("/nonexistent-sana".into())).with_quant(Quant::Q8);
        let e = load(&spec).err().expect("error").to_string();
        assert!(e.contains("quantization"), "got: {e}");
    }

    #[test]
    fn registry_resolves_sana_descriptor() {
        // The `register_generators!` submission must surface in the gen-core registry so
        // `gen_core::load("sana_1600m")` resolves on the worker (the dead-strip trap that bit Kolors
        // — covered here by asserting the descriptor is present in the linked registry).
        let found = gen_core::registry::generators()
            .map(|reg| (reg.descriptor)())
            .any(|d| d.id == MODEL_ID);
        assert!(
            found,
            "sana_1600m must be registered in the gen-core registry"
        );
    }
}
