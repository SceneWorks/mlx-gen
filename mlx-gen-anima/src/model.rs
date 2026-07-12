//! The three Anima generators (`anima_base`, `anima_aesthetic`, `anima_turbo`) — [`Generator`]
//! implementations + descriptors + [`load`] entry points + the `inventory` registration. Linking this
//! crate is all the worker needs to resolve any Anima variant by id. All three share the same
//! architecture (Cosmos-Predict2 DiT + `AnimaTextConditioner` + Qwen3-0.6B TE + Qwen-Image VAE) and
//! differ only in the DiT weights file + default steps/CFG.

use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, default_seed, Capabilities, Error,
    GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality, ModelDescriptor, Precision,
    Progress, Quant, Result,
};

use crate::config::{Variant, RES_MULTIPLE};
use crate::pipeline::{AnimaPipeline, GenOptions};

const MAX_COUNT: u32 = 8;
const RES_MIN: u32 = 512;
/// Above ~1920 px/side the Cosmos RoPE would index out of its trained range; `rope.rs` **rejects**
/// (errors on) such a request rather than clamping. The shipped ceiling is 1536² (post-patch 96, well
/// within the 120-position max_size), so the guard is unreachable via the normal path. See [`crate::rope`].
const RES_MAX: u32 = 1536;

/// Build the descriptor for a variant. Turbo is the merged CFG-free student (no guidance / negative
/// prompt); Base/Aesthetic run true classifier-free guidance.
fn descriptor_for(variant: Variant) -> ModelDescriptor {
    let cfg_capable = variant.uses_cfg();
    ModelDescriptor {
        id: variant.id(),
        family: "anima",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: cfg_capable,
            supports_guidance: cfg_capable,
            supports_true_cfg: false,
            conditioning: vec![],
            // LoRA/LoKr injection is sc-10521; every projection is an adapter-ready `AdaptableLinear`.
            supports_lora: true,
            supports_lokr: true,
            // Rectified-flow over the unified curated-sampler framework (epic 7114). The native default
            // (req.sampler == None) is the recommended er_sde solver; the full menu is advertised.
            samplers: curated_sampler_names(),
            schedulers: curated_scheduler_names(),
            supported_guidance_methods: vec![],
            min_size: RES_MIN,
            max_size: RES_MAX,
            max_count: MAX_COUNT,
            mac_only: true,
            // Q4/Q8 quant tiers (sc-10517). Anima is convert-at-install: the SceneWorks worker packs
            // the Cosmos DiT on-device (the conditioner + Qwen3 TE + VAE stay dense bf16), and this
            // crate's loader packed-detects the tier off the on-disk `{base}.scales` — so `load`
            // ACCEPTS any `spec.quantize` (it is advisory; the resolved tier dir dictates precision,
            // like SANA). The worker reads `supported_quants` for its capability advertisement
            // (gen-core sc-3723); every advertised tier actually loads, so this is honest.
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: false,
            requires_sigma_shift: true,
            // Not wired onto the shared `Residency` seam (F-176); Sequential is a no-op fallback.
            supports_sequential_offload: false,
        },
    }
}

pub fn descriptor_base() -> ModelDescriptor {
    descriptor_for(Variant::Base)
}
pub fn descriptor_aesthetic() -> ModelDescriptor {
    descriptor_for(Variant::Aesthetic)
}
pub fn descriptor_turbo() -> ModelDescriptor {
    descriptor_for(Variant::Turbo)
}

/// A loaded Anima generator: the cached descriptor + variant + the assembled pipeline.
pub struct Anima {
    descriptor: ModelDescriptor,
    variant: Variant,
    pipeline: AnimaPipeline,
}

pub fn load_base(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(spec, Variant::Base)
}
pub fn load_aesthetic(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(spec, Variant::Aesthetic)
}
pub fn load_turbo(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(spec, Variant::Turbo)
}

fn load_variant(spec: &LoadSpec, variant: Variant) -> Result<Box<dyn Generator>> {
    let id = variant.id();
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{id}: only the default dense bf16 precision is wired (drop the precision override)"
        )));
    }
    // Q4/Q8 tiers (sc-10517) are NOT quantized at load. Anima is convert-at-install: the SceneWorks
    // worker packs the Cosmos DiT on-device (`convert::quantize_anima_dit`, conditioner + Qwen3 TE +
    // VAE kept dense bf16), and the DiT's `AdaptableLinear`s packed-detect the tier off the on-disk
    // `{base}.scales` inside `CosmosDiT::from_weights`. So a `spec.quantize` value is ADVISORY — the
    // resolved tier directory dictates the actual precision — and we accept any tier without a
    // load-time `.quantize()` (mirrors SANA, the Group-B packed-detect convert-at-install path;
    // Kolors/sd3 by contrast load-time-quantize, so SANA is the true precedent here).
    //
    // Quant + LoRA/LoKr together IS supported (sc-10578). No guard is needed here: the DiT's
    // `AdaptableLinear`s already carry a `LinearBase::Quantized` on a packed tier, and `AdaptableLinear`
    // evaluates `base(x) + Σ adapter.residual(x)` — i.e. the additive branch `y = xW_packed + scale·(xA)B`
    // (epic 10043) — leaving the packed codes untouched. A LoKr on a packed base installs as the
    // structured `Adapter::LokrStructured` (the Kronecker vec-trick), so it never materializes an
    // `[out,in]` delta; the shared `install_lycoris_groups` picks that form off `is_quantized()`.
    // (A LoHa has no deferred form, so it falls back to the materialized delta there — correct, but
    // memory-hungry. Whether a packed base should refuse that is sc-10678, not a load-gate concern.)
    let _ = spec.quantize;
    let mut pipeline = AnimaPipeline::from_source(&spec.weights, variant)?;
    // Bake any LoRA/LoKr adapters onto the still-mutable model (DiT + bundled conditioner), stacked
    // and mixed, strictly (an unmatched target errors rather than loading a partial distillation —
    // sc-10521 / sc-10274). No-op when `spec.adapters` is empty.
    if !spec.adapters.is_empty() {
        pipeline.apply_adapters(&spec.adapters)?;
    }
    Ok(Box::new(Anima {
        descriptor: descriptor_for(variant),
        variant,
        pipeline,
    }))
}

mlx_gen::impl_generator!(Anima {
    validate: |s, req| validate_request(&s.descriptor, req),
    generate: generate_impl,
});

impl Anima {
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        validate_request(&self.descriptor, req)?;
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let steps = req.steps.unwrap_or(self.variant.default_steps()) as usize;
        let guidance = if self.variant.uses_cfg() {
            req.guidance.unwrap_or(self.variant.default_guidance())
        } else {
            1.0
        };
        let negative = req.negative_prompt.clone().unwrap_or_default();

        let mut images = Vec::with_capacity(req.count as usize);
        for n in 0..req.count {
            // Release the MLX cache between images so a batch doesn't accumulate to a SIGKILL (sc-5567).
            if n > 0 {
                mlx_rs::memory::clear_cache();
            }
            let opts = GenOptions {
                width: req.width,
                height: req.height,
                steps,
                guidance,
                seed: base_seed.wrapping_add(n as u64),
                sampler: req.sampler.clone(),
                // Epic 7114 scheduler axis (sc-11123 / F-115): honor the advertised curated schedulers
                // instead of ignoring req.scheduler. The shared floor already validated the name against
                // curated_scheduler_names(); anima_schedule resolves it → σ (None ⇒ native schedule).
                scheduler: req.scheduler.clone(),
            };
            let img = self.pipeline.generate(
                &req.prompt,
                &negative,
                self.variant,
                &opts,
                &req.cancel,
                on_progress,
            )?;
            images.push(img);
        }
        Ok(GenerationOutput::Images(images))
    }
}

/// Capability-driven request validation (testable without loaded weights): non-empty prompt, size a
/// multiple of 16, steps ≥ 1, on top of the shared [`Capabilities::validate_request`] floor.
pub(crate) fn validate_request(desc: &ModelDescriptor, req: &GenerationRequest) -> Result<()> {
    let id = desc.id;
    if req.prompt.is_empty() {
        return Err(Error::Msg(format!("{id}: prompt must not be empty")));
    }
    desc.capabilities.validate_request(id, req)?;
    if !req.width.is_multiple_of(RES_MULTIPLE) || !req.height.is_multiple_of(RES_MULTIPLE) {
        return Err(Error::Msg(format!(
            "{id}: {}x{} must be a multiple of {RES_MULTIPLE}",
            req.width, req.height
        )));
    }
    Ok(())
}

// Link-time registration of all three variants.
mlx_gen::register_generators! {
    descriptor_base => load_base,
    descriptor_aesthetic => load_aesthetic,
    descriptor_turbo => load_turbo,
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core;
    use mlx_gen::WeightsSource;

    fn req(w: u32, h: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "an anime girl with silver hair, detailed".into(),
            width: w,
            height: h,
            ..Default::default()
        }
    }

    #[test]
    fn three_variants_registered() {
        for id in ["anima_base", "anima_aesthetic", "anima_turbo"] {
            assert!(
                gen_core::registry::generators().any(|r| (r.descriptor)().id == id),
                "id {id} not registered"
            );
        }
    }

    #[test]
    fn descriptors_surface() {
        let b = descriptor_base();
        assert_eq!(b.id, "anima_base");
        assert_eq!(b.family, "anima");
        assert_eq!(b.backend, "mlx");
        assert_eq!(b.modality, Modality::Image);
        assert!(b.capabilities.supports_guidance);
        assert!(b.capabilities.supports_negative_prompt);
        assert!(b.capabilities.requires_sigma_shift);
        assert!(b.capabilities.supports_lora && b.capabilities.supports_lokr);
        assert!(b.capabilities.mac_only);
        // Q4/Q8 tiers advertised (sc-10517): convert-at-install packs the DiT on-device and the loader
        // packed-detects each tier, so every advertised tier actually loads — an honest advertisement.
        assert_eq!(b.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert_eq!(b.capabilities.min_size, 512);
        assert_eq!(b.capabilities.max_size, 1536);
        // Turbo is the CFG-free merged student.
        let t = descriptor_turbo();
        assert!(!t.capabilities.supports_guidance);
        assert!(!t.capabilities.supports_negative_prompt);
        // er_sde is advertised in the curated menu.
        assert!(
            b.capabilities.samplers.contains(&"er_sde"),
            "er_sde not advertised"
        );
    }

    #[test]
    fn validate_rejects_bad_requests() {
        assert!(validate_request(&descriptor_base(), &GenerationRequest::default()).is_err()); // empty prompt
        assert!(validate_request(&descriptor_base(), &req(1000, 1024)).is_err()); // not mult of 16
        assert!(validate_request(&descriptor_base(), &req(256, 256)).is_err()); // below min
        assert!(validate_request(&descriptor_base(), &req(2048, 2048)).is_err()); // above max
        assert!(validate_request(&descriptor_base(), &req(1024, 1024)).is_ok());
        assert!(validate_request(&descriptor_base(), &req(1536, 1536)).is_ok());
        // Turbo rejects guidance / negative (CFG-free).
        assert!(validate_request(
            &descriptor_turbo(),
            &GenerationRequest {
                guidance: Some(4.5),
                ..req(1024, 1024)
            }
        )
        .is_err());
    }

    #[test]
    fn load_accepts_quant_spec() {
        // Q4/Q8 are wired (sc-10517) as packed-detected tiers: a quantize request must get PAST the
        // load gate (no "unsupported"/defer rejection) and fail later on the missing snapshot instead —
        // proving `spec.quantize` is accepted as advisory, not rejected.
        for q in [Quant::Q4, Quant::Q8] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-anima".into())).with_quant(q);
            let e = load_base(&spec).err().expect("error").to_string();
            assert!(
                !e.contains("quant") && !e.contains("sc-10517"),
                "quant spec must be accepted, got a quant-rejection: {e}"
            );
        }
    }

    #[test]
    fn load_accepts_quant_plus_adapter_sc10578() {
        // The inverse of the guard this story removed. A packed tier requested WITH an adapter must no
        // longer be rejected on CAPABILITY grounds: `AdaptableLinear` runs `base(x) + Σ residual(x)`
        // over a `LinearBase::Quantized` (the epic-10043 additive branch), and a packed LoKr installs as
        // the structured Kronecker form. The pair is supported.
        //
        // A nonexistent weights dir still errors — but it must now fail on WEIGHTS/IO, not on the pair.
        // Asserting the absence of the old rejection is what keeps a future "narrow the guard back"
        // refactor from silently re-breaking q4+LoRA, which is the single most common Anima workflow.
        //
        // This test only guards the load GATE. The numeric proof that the residual actually rides on
        // the packed codes lives in the real-weights `tests/packed_adapters.rs` (`#[ignore]`d, not run
        // in CI); the CI-covered proof of the install math is in the shared core unit tests,
        // `mlx-gen/src/adapters/loader.rs::lokr_on_packed_base_installs_structured_and_matches_dense`.
        use mlx_gen::runtime::{AdapterKind, AdapterSpec};
        for variant_load in [load_base, load_aesthetic, load_turbo] {
            let adapter = AdapterSpec::new(
                "/nonexistent-anima-lora.safetensors".into(),
                1.0,
                AdapterKind::Lora,
            );
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-anima".into()))
                .with_quant(Quant::Q8)
                .with_adapters(vec![adapter]);
            let e = variant_load(&spec)
                .err()
                .expect("a nonexistent weights dir still errors")
                .to_string();
            assert!(
                !e.contains("sc-10578") && !e.contains("not supported"),
                "packed tier + adapter must NOT be rejected as unsupported, got: {e}"
            );
        }
    }
}
