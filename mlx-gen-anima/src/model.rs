//! The three Anima generators (`anima_base`, `anima_aesthetic`, `anima_turbo`) — [`Generator`]
//! implementations + descriptors + [`load`] entry points + the `inventory` registration. Linking this
//! crate is all the worker needs to resolve any Anima variant by id. All three share the same
//! architecture (Cosmos-Predict2 DiT + `AnimaTextConditioner` + Qwen3-0.6B TE + Qwen-Image VAE) and
//! differ only in the DiT weights file + default steps/CFG.

use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, default_seed, Capabilities, Error,
    GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality, ModelDescriptor, Precision,
    Progress, Quant, Residency, Result,
};

use crate::config::{Variant, RES_MULTIPLE};
use crate::pipeline::{AnimaCondInputs, AnimaHeavy, AnimaText};

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
            // Wired onto the shared `Residency` seam (epic 10834, sc-10840); honors Sequential offload
            // (F-176). Under `Sequential` the Qwen3-0.6B text encoder is encoded, materialized, then
            // dropped before the DiT + bundled conditioner + VAE load — bounding peak unified memory to
            // `max(Qwen3-TE, DiT+conditioner+VAE)`. Q4/Q8 are packed convert-at-install tiers (no
            // load-time re-quant), so no F-181 dense-requant advisory is needed (mirrors SANA).
            supports_sequential_offload: true,
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

/// A loaded Anima generator: the cached descriptor + variant + the component-residency strategy
/// (epic 10834, sc-10840). Holds ONLY the [`Residency`] (no direct encoder/DiT/VAE fields — a retained
/// component would defeat the `Sequential` drop): `Resident` (default) holds the Qwen3 TE + DiT +
/// bundled conditioner + VAE warm for the whole job and across jobs; `Sequential` holds only the
/// per-phase loader closures and re-loads each per generation in phase order (encode → **drop the
/// Qwen3 TE** → conditioner/DiT/VAE), bounding peak unified memory to
/// `max(Qwen3-TE, DiT+conditioner+VAE)`.
pub struct Anima {
    descriptor: ModelDescriptor,
    variant: Variant,
    residency: Residency<AnimaText, AnimaHeavy>,
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
    Ok(Box::new(Anima {
        descriptor: descriptor_for(variant),
        variant,
        residency: build_residency(spec, variant)?,
    }))
}

/// The policy→[`Residency`] dispatch every Anima variant shares (sc-10840), routed through the single
/// [`Residency::from_policy`] seam (F-180) so no variant re-derives the `match offload_policy`.
/// `Resident` eager-loads the Qwen3 TE phase + heavy bundle now; `Sequential` captures the two loader
/// closures and loads nothing now, deferring each to [`Residency::run`]. Both use the same
/// [`AnimaText::load`] / [`AnimaHeavy::load`], so the `Resident` composition is byte-identical to the
/// pre-seam `AnimaComponents` (independent files, RNG-free load + adapter merge). Anima has no PiD
/// overlay, so the heavy loader's `use_pid` flag is ignored. Adapters are baked in the heavy loader
/// (the DiT + bundled conditioner both live there), stacked/mixed and strict — an unmatched target
/// errors rather than loading a partial distillation (sc-10521 / sc-10274). The deferral is
/// weight-free-testable: under `Sequential` this touches no component weights, so a dispatch that
/// ignored `offload_policy` would eager-load and fail the "Sequential defers" unit test.
pub(crate) fn build_residency(
    spec: &LoadSpec,
    variant: Variant,
) -> Result<Residency<AnimaText, AnimaHeavy>> {
    let spec_text = spec.clone();
    let spec_heavy = spec.clone();
    Residency::from_policy(
        spec.offload_policy,
        move || AnimaText::load(&spec_text.weights, variant),
        move |_use_pid| {
            let mut heavy = AnimaHeavy::load(&spec_heavy.weights, variant)?;
            if !spec_heavy.adapters.is_empty() {
                heavy.apply_adapters(&spec_heavy.adapters)?;
            }
            Ok(heavy)
        },
    )
}

mlx_gen::impl_generator!(Anima {
    validate: |s, req| validate_request(&s.descriptor, req),
    generate: generate_impl,
});

impl Anima {
    /// The staged residency lifecycle (encode the Qwen3 conditioner inputs → **drop the Qwen3 TE**
    /// under `Sequential` → conditioner forward + DiT denoise + VAE decode → free the heavy bundle) is
    /// driven by the shared [`Residency::run`] seam (sc-10840), which owns the eval/drop/clear
    /// discipline, the stage-boundary cancel checks, and the error-safe cache flush.
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        validate_request(&self.descriptor, req)?;
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let steps = req.steps.unwrap_or(self.variant.default_steps()) as usize;
        let variant = self.variant;
        let guidance = if variant.uses_cfg() {
            req.guidance.unwrap_or(variant.default_guidance())
        } else {
            1.0
        };
        let negative = req.negative_prompt.clone().unwrap_or_default();
        // Epic 7114 sampler/scheduler axis: `None` ⇒ the native er_sde default / native σ schedule.
        let sampler = req
            .sampler
            .clone()
            .unwrap_or_else(|| crate::pipeline::DEFAULT_SAMPLER.to_string());
        let scheduler = req.scheduler.clone();

        self.residency.run(
            &req.cancel,
            // Anima has no PiD overlay; the heavy loader ignores `use_pid`.
            false,
            on_progress,
            // ── Phase A: encode the conditioner INPUTS (Qwen3 forward + mask-multiply). Seed-independent
            // (no RNG). `uncond` is encoded iff the variant uses CFG (NOT gated on the guidance value —
            // preserving the pre-seam behavior of running the uncond forward even at guidance 1.0). Under
            // `Sequential` the shared seam materializes these + DROPS the Qwen3 TE before the heavy load.
            |text: &AnimaText| {
                let cond = text.encode_inputs(&req.prompt)?;
                let uncond = if variant.uses_cfg() {
                    Some(text.encode_inputs(&negative)?)
                } else {
                    None
                };
                Ok((cond, uncond))
            },
            // Materialize the masked Qwen3 states + T5 ids (cond + optional uncond) while the TE is still
            // alive (Sequential only) — MLX is lazy, so an un-evaluated `source` keeps the TE referenced
            // and dropping it would free nothing. The T5 weights are host data (no eval).
            |(cond, uncond): &(AnimaCondInputs, Option<AnimaCondInputs>)| {
                let mut arrays = vec![&cond.source, &cond.t5_ids];
                if let Some(u) = uncond {
                    arrays.push(&u.source);
                    arrays.push(&u.t5_ids);
                }
                mlx_rs::transforms::eval(arrays)?;
                Ok(())
            },
            // ── Phase B: conditioner forward (once per cond/uncond — seed-independent) then the count
            // loop of denoise/decode. Identical body for both residencies, so a Sequential job is
            // byte-identical to Resident.
            |heavy: &AnimaHeavy, (cond, uncond), on_progress: &mut dyn FnMut(Progress)| {
                let cond_enc = heavy.conditioner_forward(&cond)?;
                let uncond_enc = match &uncond {
                    Some(u) => Some(heavy.conditioner_forward(u)?),
                    None => None,
                };
                let mut images = Vec::with_capacity(req.count as usize);
                for n in 0..req.count {
                    // Release the MLX cache between images so a batch doesn't accumulate to a SIGKILL
                    // (sc-5567).
                    if n > 0 {
                        mlx_rs::memory::clear_cache();
                    }
                    let seed = base_seed.wrapping_add(n as u64);
                    let img = heavy.render_one(
                        &cond_enc,
                        uncond_enc.as_ref(),
                        req.width,
                        req.height,
                        steps,
                        guidance,
                        &sampler,
                        scheduler.as_deref(),
                        seed,
                        &req.cancel,
                        on_progress,
                    )?;
                    images.push(img);
                }
                Ok(GenerationOutput::Images(images))
            },
        )
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
    ; footprint = crate::loader::component_footprint
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

    // ── Sequential residency (epic 10834, sc-10840): weight-free proof the dispatch HONORS
    // `offload_policy`. `build_residency` points at a non-existent snapshot dir; the discriminator is
    // deferral:
    //   * `Sequential` captures the two per-phase loaders, touches NO weights → `Ok` + `is_sequential`.
    //   * `Resident` eager-loads the Qwen3 TE from the missing dir → `Err`.
    // A dispatch that ignored `offload_policy` (always `Resident`) would eager-load under a `Sequential`
    // request and fail the first assertion. The real-weights A/B is `#[ignore]`d; this runs by default.
    fn missing_snapshot_spec(policy: mlx_gen::OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(
            "/nonexistent/anima-residency-test-snapshot".into(),
        ))
        .with_offload_policy(policy)
    }

    #[test]
    fn build_residency_sequential_defers_all_component_loads() {
        // All three variants share the one dispatch — assert on Base.
        let res = build_residency(
            &missing_snapshot_spec(mlx_gen::OffloadPolicy::Sequential),
            Variant::Base,
        )
        .expect("Sequential must defer loads and not touch the (missing) snapshot dir");
        assert!(
            res.is_sequential(),
            "Sequential policy must build a Sequential (deferred) residency"
        );
    }

    #[test]
    fn build_residency_resident_eager_loads_and_fails_on_missing_snapshot() {
        let err = build_residency(
            &missing_snapshot_spec(mlx_gen::OffloadPolicy::Resident),
            Variant::Base,
        )
        .err()
        .expect("Resident must eager-load and fail on a missing snapshot dir");
        let msg = err.to_string();
        // An eager-load failure (missing split_files / TE file), not a policy/precision rejection.
        assert!(
            !msg.contains("precision override"),
            "expected an eager-load failure, not the up-front precision guard: {msg}"
        );
    }

    #[test]
    fn descriptors_advertise_sequential_offload() {
        // All three anima ids honor the shared Residency seam (the descriptor bit consumers read).
        for d in [
            descriptor_base(),
            descriptor_aesthetic(),
            descriptor_turbo(),
        ] {
            assert!(
                d.capabilities.supports_sequential_offload,
                "{} must advertise supports_sequential_offload",
                d.id
            );
        }
    }
}
