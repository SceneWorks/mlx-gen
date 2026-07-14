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
    curated_sampler_names, curated_scheduler_names, default_seed, Capabilities, Conditioning,
    ConditioningKind, Error, GenerationOutput, GenerationRequest, Generator, Image, LoadSpec,
    Modality, ModelDescriptor, Precision, Progress, Quant, Residency, Result, WeightsSource,
};

use crate::config::{DcAeConfig, SanaTransformerConfig};
use crate::dc_ae::{DcAeDecoder, DcAeEncoder};
use crate::pipeline::{
    encode_conditioning, SanaConditioning, SanaGenerateRequest, SanaHeavy, DEFAULT_GUIDANCE,
    SPRINT_DEFAULT_GUIDANCE,
};
use crate::text_encoder::SanaTextEncoder;
use crate::transformer::SanaTransformer;

/// Registry id for SANA-1.6B 1024px (matches the SceneWorks worker's `payload.model`).
pub const MODEL_ID: &str = "sana_1600m";

/// Registry id for **SANA-Sprint** 1.6B 1024px (the CFG-free, SCM/TrigFlow few-step variant, sc-8490).
pub const SPRINT_MODEL_ID: &str = "sana_sprint_1600m";

/// SANA-1.6B's native generation resolution. The model is bucket-trained at 1024² and the only
/// real-weight e2e that exists validates 1024² ([`real_weight_1024_e2e`]), so 1024 is the validated
/// engine envelope — and the advertised [`Capabilities::max_size`] is bounded to it.
///
/// **Why not 2048 (F-032, sc-9095):** the DC-AE decoder ([`crate::dc_ae::DcAeDecoder::decode`]) runs
/// the full f32 decode monolithically — no tiling — so at 2048² the shallow 128-channel stage
/// materializes ~2.1 GB tensors with several live at once (GLUMBConv expands 8×), an uncatchable
/// OOM/SIGKILL class the workspace already budgeted for wan (sc-4998) and seedvr2 (sc-8135/8261). DC-AE
/// is a deep conv stack that *could* be spatially tiled, but no larger-than-1024 output is validated,
/// so we advertise only what we can honor rather than tiling toward an unvalidated envelope. Raising
/// this ceiling later means porting wan's budgeted spatial tiling, not just bumping the constant.
const RES_MIN: u32 = 256;
const RES_MAX: u32 = 1024;
/// DC-AE 32× spatial compression — requested dims must be a multiple of this so the latent edge
/// (`image / 32`) is integral.
const RES_MULTIPLE: u32 = crate::pipeline::SPATIAL_SCALE;
/// Max images per request (the image-model standard, shared with the other MLX families).
const MAX_COUNT: u32 = 8;

/// SANA-1.6B's identity + capabilities — constructible without loading weights (registry
/// introspection / capability advertisement). True-CFG text-to-image: negative prompt + guidance
/// scale, flow-match Euler over the unified curated sampler/scheduler menu (epic 7114). Advertises
/// `Reference` img2img (sc-10190): a single reference image seeds the denoise from the DC-AE-encoded
/// init latent. ControlNet conditioning is a separate, later variant.
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
            // img2img (sc-10190): a single `Reference` image seeds the denoise from the DC-AE-encoded
            // init latent (`ui.img2img` in the catalog). ControlNet is a separate, later variant.
            conditioning: vec![ConditioningKind::Reference],
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
            // SANA ships pre-quantized Q4/Q8 turnkey tiers (sc-8489, epic 8506): the Linear-DiT
            // transformer + the Gemma-2 CHI TE are packed and PACKED-DETECTED on load (the DC-AE VAE
            // stays dense in every tier). Advertise Q4/Q8 so the catalog routes SANA through the
            // SAME quant-tier path as every other matrix model (tier selection + accurate recipe /
            // downgrade telemetry) rather than a no-quant special case. This is NOT the (still
            // unported) 2-bit Clark-Labs quant — it is the shared group-64 affine tier, packed
            // offline by `crate::convert` and self-describing on load, so a `spec.quantize` is
            // advisory (the resolved tier dir dictates the actual precision; see [`load`]).
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: false,
            // Static flow-match shift 3.0, resolution-independent (handled by the unified sampler).
            requires_sigma_shift: false,
            // Wired onto the shared `Residency` seam (epic 10834); honors Sequential offload (F-176).
            // Under `Sequential` the Gemma-2 CHI text encoder is encoded, materialized, then dropped
            // before the Linear-DiT trunk + DC-AE load — bounding peak to `max(Gemma-TE, DiT+DC-AE)`.
            // The Gemma encoder is comparable to (often ≥) the DiT, so the drop is a large win.
            supports_sequential_offload: true,
        },
    }
}

/// **SANA-Sprint** identity + capabilities (sc-8490). Same `sana` family / `mlx` backend / image
/// modality as the base, but the distilled variant is **CFG-free** (the guidance scale is an
/// *embedded scalar* fed to the trunk, not classifier-free guidance) and **few-step** (1–4, default
/// 2): so `supports_true_cfg = false`, `supports_negative_prompt = false`, and NO
/// `supported_guidance_methods` (the epic-7434 cfg/cfg_rescale/apg/cfg_pp combine operators do not
/// apply — there is no cond/uncond pair). `supports_guidance` stays `true` because the guidance scale
/// is still an honored request knob (it just modulates the embedded scalar). The SCM/TrigFlow sampler
/// is a dedicated few-step loop, so the curated epic-7114 sampler/scheduler menu is NOT advertised.
pub fn sprint_descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: SPRINT_MODEL_ID,
        family: "sana",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            // Embedded guidance scalar — honored knob, but NOT classifier-free (no uncond forward).
            supports_negative_prompt: false,
            supports_guidance: true,
            supports_true_cfg: false,
            // img2img (sc-10190): reference-seeded, via the SCM/TrigFlow renoise at the start angle.
            // Distilled/few-step → the strength window is narrow (validate on-device).
            conditioning: vec![ConditioningKind::Reference],
            supports_lora: false,
            supports_lokr: false,
            // The SCM/TrigFlow consistency loop is a dedicated few-step sampler, not a curated
            // epic-7114 `Solver`; only the engine-default sentinel is advertised.
            samplers: vec!["default"],
            schedulers: vec!["default"],
            // CFG-free: no cfg/cfg_rescale/apg/cfg_pp combine (the guidance axis embedded case).
            supported_guidance_methods: vec![],
            min_size: RES_MIN,
            max_size: RES_MAX,
            max_count: MAX_COUNT,
            mac_only: true,
            // Same Q4/Q8 packed turnkey tiers as base SANA (sc-8489): the Sprint Linear-DiT trunk +
            // Gemma-2 TE are packed/packed-detected, DC-AE VAE dense. Advertise Q4/Q8 for standard
            // quant-tier routing; `spec.quantize` is advisory (resolved tier dir dictates precision).
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: false,
            requires_sigma_shift: false,
            // Wired onto the shared `Residency` seam (epic 10834); honors Sequential offload (F-176).
            // Sprint drops the Gemma-2 CHI text encoder before the Sprint Linear-DiT trunk + DC-AE load,
            // bounding peak to `max(Gemma-TE, DiT+DC-AE)`.
            supports_sequential_offload: true,
        },
    }
}

/// A loaded SANA generator: the component-residency seam plus the cached descriptor and the variant
/// flag (`sprint`) the encode phase reads to gate the uncond forward / default guidance.
pub struct Sana {
    descriptor: ModelDescriptor,
    /// Component-residency strategy (epic 10834; the shared seam sc-11125), selected from
    /// [`LoadSpec::offload_policy`] at load. `Resident` (default) holds the Gemma-2 CHI text encoder +
    /// Linear-DiT trunk + DC-AE warm for the whole job and across jobs; `Sequential` holds only the
    /// per-phase loader closures and re-loads per generation in phase order (encode → **drop the Gemma
    /// encoder** → denoise/decode), bounding peak unified memory to `max(Gemma-TE, DiT+DC-AE)` instead
    /// of the sum — the Gemma encoder is comparable to (often ≥) the DiT, so the drop is a large win.
    /// The [`Residency`] seam owns the eval/drop/clear discipline, the stage-boundary cancel checks, and
    /// the error-safe cache flush once for all providers.
    residency: Residency<SanaTextEncoder, SanaHeavy>,
    /// `true` for SANA-Sprint (CFG-free SCM few-step) — read at encode time (before the heavy bundle is
    /// available under `Sequential`) to gate the uncond forward and resolve the default guidance.
    sprint: bool,
}

/// Construct a SANA generator from a [`LoadSpec`]. `spec.weights` must be a [`WeightsSource::Dir`]
/// pointing at a `Sana_1600M_1024px_diffusers`-shaped snapshot (`transformer/ vae/ text_encoder/`), or
/// a pre-quantized Q4/Q8 tier of the same shape (packed-detected on load, sc-8489). A precision
/// override or LoRA/LoKr adapters are rejected rather than silently ignored (neither is wired yet).
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_from(spec, descriptor(), false)
}

/// Construct a **SANA-Sprint** generator (sc-8490) from a [`LoadSpec`]. Identical snapshot contract to
/// [`load`] (`transformer/ vae/ text_encoder/`), but the transformer is loaded with the Sprint config
/// (guidance embedder + rms-norm-across-heads) and driven by the CFG-free SCM few-step pipeline.
pub fn load_sprint(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_from(spec, sprint_descriptor(), true)
}

/// Shared load body for [`load`] / [`load_sprint`]: fail-fast on the unsupported precision / adapter
/// overrides (via [`load_components`]) up front for BOTH residencies, then build the [`Residency`]
/// seam. `Resident` (default) builds the Gemma text encoder + heavy bundle now and holds them warm;
/// `Sequential` keeps only the per-phase loader closures and re-loads per generate in phase order
/// (encode → drop the Gemma encoder → denoise/decode) to bound peak memory to `max(Gemma-TE, DiT+DC-AE)`.
fn load_from(
    spec: &LoadSpec,
    descriptor: ModelDescriptor,
    sprint: bool,
) -> Result<Box<dyn Generator>> {
    let id = descriptor.id;
    // Fail-fast on precision / adapter / single-file guards for BOTH residencies (Sequential defers the
    // component build, but these overrides are still wrong, so reject them here).
    load_components(spec, id)?;
    let residency = build_residency(spec, sprint, id)?;
    Ok(Box::new(Sana {
        descriptor,
        residency,
        sprint,
    }))
}

/// The policy→[`Residency`] dispatch both SANA ids share (sc-11126, F-180), routed through the single
/// [`Residency::from_policy`] seam so neither variant re-derives the `match offload_policy`. `Resident`
/// eager-loads the Gemma text encoder + heavy bundle now (byte-identical to the pre-seam composition —
/// the same per-phase loaders over independent weight files; SANA's Q4/Q8 tiers are packed-detected on
/// load, so there is no RNG or load-time re-quantization to diverge); `Sequential` captures the two
/// loader closures and loads nothing now, deferring each to [`Residency::run`]. SANA has no PiD overlay,
/// so the heavy loader's `use_pid` flag is ignored. The deferral is weight-free-testable: under
/// `Sequential` this touches no component weights, so a dispatch that ignored `offload_policy` would
/// eager-load and fail the "Sequential defers" unit test.
pub(crate) fn build_residency(
    spec: &LoadSpec,
    sprint: bool,
    id: &'static str,
) -> Result<Residency<SanaTextEncoder, SanaHeavy>> {
    let spec_text = spec.clone();
    let spec_heavy = spec.clone();
    Residency::from_policy(
        spec.offload_policy,
        move || load_text_encoder_component(load_components(&spec_text, id)?),
        move |_use_pid| load_heavy(load_components(&spec_heavy, id)?, sprint),
    )
}

/// Load the Gemma-2 CHI text encoder (`text_encoder/`) — the phase-A component dropped first under
/// `Sequential`. Bundles the gemma weights + `tokenizer.json` (the un-gated mirror, epic 7840).
fn load_text_encoder_component(root: &Path) -> Result<SanaTextEncoder> {
    SanaTextEncoder::from_snapshot(root.join("text_encoder"))
}

/// Load the heavy render-phase components — the Linear-DiT trunk (base or Sprint config) + the DC-AE
/// encoder/decoder — everything but the Gemma text encoder. Factored so the `Sequential` path loads
/// these AFTER the encoder is dropped (bounding peak to `max(Gemma-TE, DiT+DC-AE)`), and the `Resident`
/// path builds the same bundle up front. Components are independent of the text encoder (separate weight
/// files, packed-detected quant — no RNG), so both residencies are byte-identical. The trunk is loaded
/// first (mirroring the pre-seam `build_pipeline` order), then the DC-AE from the shared `vae/` source.
fn load_heavy(root: &Path, sprint: bool) -> Result<SanaHeavy> {
    let trunk_w = Weights::from_dir(root.join("transformer"))?;
    let dcfg = DcAeConfig::sana_f32c32();
    let vae_w = Weights::from_dir(root.join("vae"))?;
    // The `vae/` snapshot ships BOTH `encoder.*` and `decoder.*` — build both from the one source.
    let encoder = DcAeEncoder::from_weights(&vae_w, dcfg.clone())?;
    let decoder = DcAeDecoder::from_weights(&vae_w, dcfg.clone())?;
    if sprint {
        let trunk_cfg = SanaTransformerConfig::sana_sprint_1600m();
        let guidance_embeds_scale = trunk_cfg.guidance_embeds_scale;
        let trunk = SanaTransformer::from_weights(&trunk_w, trunk_cfg)?;
        Ok(SanaHeavy::new_sprint(
            trunk,
            encoder,
            decoder,
            dcfg,
            guidance_embeds_scale,
        ))
    } else {
        let trunk = SanaTransformer::from_weights(&trunk_w, SanaTransformerConfig::sana_1600m())?;
        Ok(SanaHeavy::new(trunk, encoder, decoder, dcfg))
    }
}

/// Shared load preamble for [`load`] / [`load_sprint`] (F-090): reject the unsupported precision /
/// adapter overrides (neither is wired for SANA) — but ACCEPT a quant spec, since a pre-quantized tier
/// is packed-detected from disk (sc-8489) — then resolve the `LoadSpec` to the snapshot directory. The
/// `{id}` in each message comes from the descriptor, so the two paths' error text differs only by id).
fn load_components<'a>(spec: &'a LoadSpec, id: &str) -> Result<&'a Path> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{id}: only the default dense precision is wired (drop the precision override)"
        )));
    }
    // Quantization is NOT load-time here (the 2-bit Clark-Labs quant is still not ported). Instead a
    // pre-quantized Q4/Q8 tier is **packed-detected** from the on-disk `{base}.scales` by the shared
    // `mlx_gen::quant::lin` inside `SanaTransformer`/`Gemma2` `from_weights` (Group-B, sc-8489), so a
    // `spec.quantize` value is advisory: the resolved tier directory dictates the actual precision
    // (dense bf16 when no `.scales`). We therefore accept any `spec.quantize` and never quantize dense
    // weights at load — a request for a tier that is not on disk simply loads whichever tier is.
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(format!(
            "{id}: LoRA/LoKr adapters are not supported"
        )));
    }
    match &spec.weights {
        WeightsSource::Dir(p) => Ok(p.as_path()),
        WeightsSource::File(_) => Err(Error::Msg(format!(
            "{id} expects a snapshot directory (transformer/ vae/ text_encoder/), not a \
             single .safetensors file"
        ))),
    }
}

/// Resolve the optional img2img reference from the request conditioning (sc-10190): at most one
/// `Conditioning::Reference` (multiple → error), returning its image and the effective strength
/// (`per-reference strength.or(req.strength)`). Mirrors the sibling img2img families (Z-Image).
fn resolve_reference<'a>(
    req: &'a GenerationRequest,
    id: &str,
) -> Result<Option<(&'a Image, Option<f32>)>> {
    let mut reference = None;
    for c in &req.conditioning {
        if let Conditioning::Reference { image, strength } = c {
            if reference.is_some() {
                return Err(Error::Msg(format!(
                    "{id}: multiple reference images are not supported (single img2img init only)"
                )));
            }
            reference = Some((image, strength.or(req.strength)));
        }
    }
    Ok(reference)
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
    /// bridges the tail into [`gen_core::Error`]. The staged residency lifecycle (Gemma encode → **drop
    /// the Gemma encoder** under `Sequential` → load the trunk/DC-AE → denoise/decode per image → free
    /// the heavy bundle) is driven by the shared [`Residency::run`] seam (sc-11125): the prompt is
    /// encoded ONCE (seed-independent, so hoisting it above the per-image loop is byte-identical to the
    /// pre-seam per-image encode), then each image's seed is derived from the base seed so a `count > 1`
    /// batch is reproducible and distinct.
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        validate_request(&self.descriptor, req)?;

        // img2img (sc-10190): a single `Reference` conditioning, with a per-reference strength
        // overriding `req.strength`. Both render paths (base flow-match + Sprint SCM) seed the denoise
        // from the encoded init when an image + positive strength is present. Seed-independent; resolved
        // above the residency lifecycle.
        let reference = resolve_reference(req, self.descriptor.id)?;
        let (init_image, strength) = match reference {
            Some((image, strength)) => (Some(image), strength),
            None => (None, None),
        };

        let base_seed = req.seed.unwrap_or_else(default_seed);
        let steps = req.steps.map(|s| s as usize);
        // Resolve guidance against the variant default ONCE so the encode's uncond decision and the
        // render's denoise agree (base 4.5 true-CFG, Sprint's embedded 4.5).
        let guidance = req.guidance.unwrap_or(if self.sprint {
            SPRINT_DEFAULT_GUIDANCE
        } else {
            DEFAULT_GUIDANCE
        });

        self.residency.run(
            &req.cancel,
            // SANA has no PiD overlay; the heavy loader ignores `use_pid`.
            false,
            on_progress,
            // ── Phase A: Gemma CHI conditioning. Seed-independent (no RNG) — encodes cond (+ uncond for
            // base SANA with CFG active; Sprint is CFG-free). Under `Sequential` the shared seam LOADS
            // the Gemma encoder, encodes, materializes, then DROPS it + `clear_cache()` before the
            // trunk/DC-AE load. Under `Resident` it borrows the warm encoder (byte-identical to the
            // pre-seam `encode_with_mask`).
            |text_encoder: &SanaTextEncoder| {
                encode_conditioning(
                    text_encoder,
                    self.sprint,
                    &req.prompt,
                    req.negative_prompt.as_deref(),
                    guidance,
                )
            },
            // Materialize the CHI embedding + pad mask (and their uncond twins) while the encoder is
            // still alive (Sequential only) — MLX is lazy, so an un-evaluated output keeps the Gemma
            // encoder referenced through the graph and the drop would free nothing.
            |cond: &SanaConditioning| {
                let mut arrays = vec![&cond.cond, &cond.cond_mask];
                if let Some((u, um)) = &cond.uncond {
                    arrays.push(u);
                    arrays.push(um);
                }
                mlx_rs::transforms::eval(arrays)?;
                Ok(())
            },
            // ── Phase B: denoise/decode from the heavy bundle (trunk + DC-AE), one image per seed.
            // Runs identically for both residencies. `on_progress` is threaded through the seam (F-179).
            |heavy: &SanaHeavy, cond, on_progress: &mut dyn FnMut(Progress)| {
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
                        init_image,
                        strength,
                    };
                    let img =
                        heavy.render_one(&cond, &sana_req, guidance, &req.cancel, on_progress)?;
                    images.push(img);
                }
                Ok(GenerationOutput::Images(images))
            },
        )
    }
}

// Link-time registration (epic 3720): the macro emits the `inventory::submit!` and bridges the
// crate's rich `Result` into the registry's backend-neutral `gen_core::Result`.
/// Per-component on-disk footprint (sc-10894) for the MLX fit-gate's staged-residency split — the Gemma
/// text encoder (`text_encoder/`), the DiT (`transformer/`), and the DC-AE VAE (`vae/`), summed from the
/// exact snapshot subdirs [`build_pipeline`] loads. Shared by SANA + SANA-Sprint.
pub(crate) fn component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    mlx_gen::PerComponentBytes::from_spec_subdirs(
        spec,
        &["text_encoder"],
        &["transformer"],
        &["vae"],
    )
}

mlx_gen::register_generators! {
    descriptor => load,
    sprint_descriptor => load_sprint,
    ; footprint = component_footprint
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
        // sc-10190: img2img reference conditioning is now advertised.
        assert_eq!(
            d.capabilities.conditioning,
            vec![ConditioningKind::Reference]
        );
        // sc-8489/#653: SANA ships packed Q4/Q8 tiers (packed-detected on load), so the descriptor
        // advertises them for first-class quant-tier routing — no longer an empty set.
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
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

    fn ref_image() -> Image {
        Image {
            width: 32,
            height: 32,
            pixels: vec![128u8; 32 * 32 * 3],
        }
    }

    #[test]
    fn resolve_reference_extracts_single_img2img_init() {
        // sc-10190: a single Reference resolves to (image, strength); an img2img Reference is now an
        // ACCEPTED conditioning kind (advertised on the descriptor), so validate_request passes it.
        let mut r = req(1024, 1024);
        r.conditioning = vec![Conditioning::Reference {
            image: ref_image(),
            strength: Some(0.6),
        }];
        let (_, strength) = resolve_reference(&r, MODEL_ID)
            .unwrap()
            .expect("a reference");
        assert_eq!(strength, Some(0.6));
        assert!(validate_request(&descriptor(), &r).is_ok());
    }

    #[test]
    fn reference_strength_falls_back_to_request_strength() {
        // A per-reference `None` strength inherits `req.strength` (matches the sibling families).
        let mut r = req(1024, 1024);
        r.strength = Some(0.4);
        r.conditioning = vec![Conditioning::Reference {
            image: ref_image(),
            strength: None,
        }];
        let (_, strength) = resolve_reference(&r, MODEL_ID)
            .unwrap()
            .expect("a reference");
        assert_eq!(strength, Some(0.4));
    }

    #[test]
    fn resolve_reference_rejects_multiple_images() {
        let mut r = req(1024, 1024);
        r.conditioning = vec![
            Conditioning::Reference {
                image: ref_image(),
                strength: None,
            },
            Conditioning::Reference {
                image: ref_image(),
                strength: None,
            },
        ];
        assert!(resolve_reference(&r, MODEL_ID).is_err());
    }

    #[test]
    fn max_size_is_the_validated_1024_envelope() {
        // F-032 (sc-9095): the DC-AE decode is monolithic f32 (no tiling), so we advertise only the
        // validated 1024² envelope — not the old, un-decodable 2048² — on both the base and Sprint
        // descriptors. Advertising must match what we refuse.
        assert_eq!(descriptor().capabilities.max_size, 1024);
        assert_eq!(sprint_descriptor().capabilities.max_size, 1024);
    }

    #[test]
    fn validate_rejects_2048_over_the_dc_ae_envelope() {
        // A 2048² request (a legal multiple of 32) now falls outside the advertised max_size and is
        // refused up front with the shared size error — rather than proceeding into a monolithic f32
        // DC-AE decode that OOMs on a smaller Mac (F-032).
        let d = descriptor();
        let err = validate_request(&d, &req(2048, 2048))
            .expect_err("2048² is above the validated DC-AE envelope");
        assert!(
            err.to_string().contains("size"),
            "size-range refusal, got: {err}"
        );
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
    fn load_accepts_prequantized_tier() {
        // Group-B (sc-8489): a Q4/Q8 tier is packed-detected from the on-disk `.scales`, so the
        // loader no longer rejects a quant spec — it proceeds past the quant check and fails only on
        // the missing snapshot directory, NOT with the old "quantization is not supported" message.
        let spec =
            LoadSpec::new(WeightsSource::Dir("/nonexistent-sana".into())).with_quant(Quant::Q8);
        let e = load(&spec).err().expect("error").to_string();
        assert!(
            !e.contains("quantization"),
            "quant tier must be accepted, got: {e}"
        );
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

    #[test]
    fn registry_resolves_sana_sprint_descriptor() {
        // The Sprint variant (sc-8490) must register alongside the base under its own id.
        let found = gen_core::registry::generators()
            .map(|reg| (reg.descriptor)())
            .any(|d| d.id == SPRINT_MODEL_ID);
        assert!(
            found,
            "sana_sprint_1600m must be registered in the gen-core registry"
        );
    }

    #[test]
    fn sprint_descriptor_is_cfg_free_few_step() {
        let d = sprint_descriptor();
        assert_eq!(d.id, "sana_sprint_1600m");
        assert_eq!(d.family, "sana");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        // CFG-free distilled: embedded guidance scalar, NO true CFG / negative prompt / combine ops.
        assert!(!d.capabilities.supports_true_cfg);
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supported_guidance_methods.is_empty());
        // sc-10190: Sprint also advertises img2img reference conditioning.
        assert_eq!(
            d.capabilities.conditioning,
            vec![ConditioningKind::Reference]
        );
        // Sprint advertises the same packed Q4/Q8 tiers as base SANA (sc-8489).
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert!(d.capabilities.mac_only);
    }

    #[test]
    fn sprint_load_accepts_prequantized_tier() {
        // Sprint mirrors the base load path (sc-8489): a quant spec is packed-detected, not rejected.
        let spec =
            LoadSpec::new(WeightsSource::Dir("/nonexistent-sana".into())).with_quant(Quant::Q8);
        let e = load_sprint(&spec).err().expect("error").to_string();
        assert!(
            !e.contains("quantization"),
            "quant tier must be accepted, got: {e}"
        );
    }

    // ── F-180 (sc-11126): weight-free, default-run proof that SANA's dispatch HONORS `offload_policy`.
    // `build_residency` points at a non-existent snapshot *directory* (so the up-front single-file /
    // precision / adapter guard in `load_components` passes) and the discriminator is deferral:
    //   * `Sequential` captures the two per-phase loaders, touches NO weights → `Ok` + `is_sequential`.
    //   * `Resident` eager-loads the Gemma text encoder from the missing dir → `Err`.
    // A dispatch that ignored `offload_policy` (always `Resident`) would eager-load under a `Sequential`
    // request and fail the first assertion. The `sequential_residency_real_weights.rs` A/B is
    // `#[ignore]`d (needs a snapshot); this runs by default. Both `sprint` variants share the one
    // dispatch — asserted on the base here.
    fn missing_snapshot_spec(policy: mlx_gen::OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(
            "/nonexistent/sana-residency-test-snapshot".into(),
        ))
        .with_offload_policy(policy)
    }

    #[test]
    fn build_residency_sequential_defers_all_component_loads() {
        let res = build_residency(
            &missing_snapshot_spec(mlx_gen::OffloadPolicy::Sequential),
            false,
            MODEL_ID,
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
            false,
            MODEL_ID,
        )
        .err()
        .expect("Resident must eager-load and fail on a missing snapshot dir");
        let msg = err.to_string();
        assert!(
            !msg.contains("single .safetensors file") && !msg.contains("precision override"),
            "expected an eager-load failure, not the up-front guard: {msg}"
        );
    }

    #[test]
    fn descriptor_advertises_sequential_offload() {
        // Both SANA ids honor the shared Residency seam (the descriptor bit consumers read).
        assert!(descriptor().capabilities.supports_sequential_offload);
        assert!(sprint_descriptor().capabilities.supports_sequential_offload);
    }
}
