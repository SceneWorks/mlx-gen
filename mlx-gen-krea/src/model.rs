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

use mlx_gen::media::Image;
use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, default_seed, Capabilities, Conditioning,
    ConditioningKind, Error, GenerationOutput, GenerationRequest, Generator, LatentDecoder,
    LoadSpec, Modality, ModelDescriptor, Precision, Progress, Quant, Residency, Result,
    WeightsSource,
};
use mlx_gen_pid::{flow_capture_for_request, resolve_pid_decoder_at_sigma, PidEngine};
use mlx_gen_qwen_image::pipeline::PID_BACKBONE;

use mlx_rs::Array;
use std::path::Path;

use crate::pipeline::{base_schedule, turbo_schedule, KreaHeavy, KreaText, TurboOptions};

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

/// Registry id for the **image-edit** variant (epic 10871). The Kontext-style edit surface shares the
/// undistilled Raw pipeline (full-CFG, denoise-from-noise) but routes a single `Reference` — the SOURCE
/// image — through [`KreaPipeline::generate_edit_with_progress`] (in-context VAE tokens + Qwen3-VL
/// grounding) instead of the img2img latent-init. A DISTINCT engine id (the Qwen-Image-Edit /
/// FLUX.2-Klein-Edit pattern) is what disambiguates edit from img2img: the SAME source `Reference` means
/// "edit" or "img2img" purely by which generator the worker loaded. The community `krea2_identity_edit`
/// LoRA rides `spec.adapters`.
pub const KREA_2_EDIT_ID: &str = "krea_2_edit";

/// Registry id for the **CFG-free Turbo image-edit** variant (sc-11640, follow-on to epic 10871). Same
/// Kontext edit surface as [`KREA_2_EDIT_ID`] — a source image (or scene+person pair) drives the dual
/// conditioning (in-context VAE tokens + Qwen3-VL grounding) through
/// [`KreaPipeline::render_edit`] — but on the **distilled Turbo** checkpoint: the few-step
/// `turbo_schedule` run **CFG-free** (`guidance = 0`, a single conditional forward, no cond/uncond
/// split), the fast-path alternative to the ~52-step full-CFG Raw edit. The `krea2_identity_edit` LoRA
/// (trained on the Raw DiT, family-compatible with Turbo) folds in via `spec.adapters` exactly as on
/// Raw. A DISTINCT id so the worker's edit lane can select the fast tier by model, the same way
/// `krea_2_edit` disambiguates edit from img2img.
pub const KREA_2_TURBO_EDIT_ID: &str = "krea_2_turbo_edit";

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
            // Reference-image conditioning = img2img latent-init (epic 8588 slice A, sc-10135): a single
            // `Conditioning::Reference { image, strength }` seeds the denoise from the VAE-encoded
            // reference (see [`generate_impl`] → `generate_turbo_img2img_with_progress`). Turbo only; the
            // Raw descriptor clears this (no Raw img2img entrypoint yet).
            conditioning: vec![ConditioningKind::Reference],
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
            // Wired onto the shared `Residency` seam; honors Sequential offload (F-176).
            supports_sequential_offload: true,
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
    // img2img reference latent-init (epic 8588 slice A, sc-10224): Raw advertises `Reference` just like
    // Turbo, but routes to the CFG entrypoint `generate_base_img2img_with_progress` (honoring guidance +
    // negative prompt), NOT the CFG-free Turbo one. Inherited from `descriptor()` — same single-Reference
    // surface — so this is a no-op re-affirmation kept explicit for the reader.
    d.capabilities.conditioning = vec![ConditioningKind::Reference];
    d
}

/// Krea 2 **image-edit** identity + capabilities (epic 10871). Same full-CFG surface as
/// [`raw_descriptor`] — an edit denoises from noise under true CFG, honoring guidance + a negative
/// prompt — but with the distinct [`KREA_2_EDIT_ID`] so the worker's edit lane can select it. Carries the
/// single-`Reference` (source) conditioning + LoRA/LoKr (the `krea2_identity_edit` edit LoRA). Derived
/// from [`raw_descriptor`] so the shared surface (family/backend/samplers/quants/size/CFG) stays in
/// lockstep; only the id (→ the `generate_impl` edit branch) differs.
pub fn edit_descriptor() -> ModelDescriptor {
    let mut d = raw_descriptor();
    d.id = KREA_2_EDIT_ID;
    // Edit accepts a single source (`Reference`) OR a scene+person pair (`MultiReference`, epic 10871
    // P1.3 — scene = image 1, person = image 2, fixed order). The img2img Raw/Turbo descriptors stay
    // single-`Reference`; only the edit surface advertises `MultiReference`, so `validate_request`
    // accepts a two-source edit here while still rejecting it on the img2img path.
    d.capabilities.conditioning = vec![
        ConditioningKind::Reference,
        ConditioningKind::MultiReference,
    ];
    d
}

/// Krea 2 **CFG-free Turbo image-edit** identity + capabilities (sc-11640). Same Kontext edit
/// conditioning surface as [`edit_descriptor`] (single `Reference` source OR a scene+person
/// `MultiReference`, + the `krea2_identity_edit` LoRA) but derived from the distilled Turbo
/// [`descriptor`] rather than Raw: **CFG-free** (`supports_guidance = false`, no user negative prompt),
/// so the edit runs a single conditional forward on the few-step `turbo_schedule`. Only the id (→ the
/// `generate_impl` `is_turbo_edit` branch: `turbo_schedule` / 8-step default / `guidance = 0`) and the
/// widened `conditioning` differ from [`descriptor`].
pub fn turbo_edit_descriptor() -> ModelDescriptor {
    let mut d = descriptor();
    d.id = KREA_2_TURBO_EDIT_ID;
    // Same edit conditioning surface as `edit_descriptor` — a single source `Reference` or one
    // scene+person `MultiReference`. The Turbo img2img descriptor stays single-`Reference`; only the
    // edit surfaces advertise `MultiReference`.
    d.capabilities.conditioning = vec![
        ConditioningKind::Reference,
        ConditioningKind::MultiReference,
    ];
    d
}

/// A loaded Krea 2 generator (Turbo, Raw, or edit): the cached descriptor + a component-residency
/// strategy. The variant is read back off `descriptor.id` at generate time (Turbo = CFG-free distilled;
/// Raw = full-CFG undistilled; edit = the Raw pipeline routed to the Kontext edit entrypoint).
pub struct Krea {
    descriptor: ModelDescriptor,
    /// Component-residency strategy (epic 10834 Phase 1, sc-11101; hoisted to the shared seam in
    /// sc-11125), selected from [`LoadSpec::offload_policy`]. `Resident` (default) holds the Qwen3-VL-4B
    /// text phase + DiT + VAE warm for the whole job and across jobs; `Sequential` holds only the
    /// per-phase loader closures and re-loads per generation in phase order (encode → **drop the text
    /// phase** → denoise/decode), bounding peak unified memory to `max(text, DiT+VAE)` instead of the
    /// sum (the Qwen3-VL-4B text phase is the dropped ~4B component; the single-stream DiT is 12B). The
    /// [`Residency`] seam owns the eval/drop/clear discipline, the stage-boundary cancel checks, and
    /// the error-safe cache flush.
    residency: Residency<KreaText, KreaHeavyOwned>,
}

/// The heavy render-phase components (the single-stream DiT + VAE, via [`KreaHeavy`], plus the optional
/// PiD decoder) — everything but the text phase. Owned by the `Resident` components or by a
/// `Sequential` generate.
pub(crate) struct KreaHeavyOwned {
    heavy: KreaHeavy,
    /// Optional PiD super-resolving decoder (epic 7840, sc-7845), loaded when `spec.pid` is set; Krea
    /// reuses the Qwen-Image latent space, so it shares the `qwenimage` PiD student. `req.use_pid`
    /// routes decode through it instead of the VAE. `None` for the plain VAE path.
    pid: Option<PidEngine>,
}

/// A borrow of the heavy render-phase components, so the denoise/decode dispatch runs identically
/// whether they are held resident or were just loaded by the `Sequential` path.
struct KreaHeavyRef<'a> {
    heavy: &'a KreaHeavy,
    pid: Option<&'a PidEngine>,
}

impl KreaHeavyOwned {
    fn as_ref(&self) -> KreaHeavyRef<'_> {
        KreaHeavyRef {
            heavy: &self.heavy,
            pid: self.pid.as_ref(),
        }
    }
}

/// The pre-encoded DiT text context(s) a `generate` renders from (sc-11101): the conditional context
/// always, plus the unconditional one for true-CFG (`krea_2_raw` / `krea_2_edit` with `guidance > 0`).
/// Produced once by [`Krea::encode`] (Turbo/Raw = plain text encode; edit = Qwen3-VL grounded encode)
/// so a `Sequential` job can drop the text phase before the DiT loads.
struct KreaContexts {
    pos: Array,
    neg: Option<Array>,
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

/// Load the **image-edit** generator (`krea_2_edit`, epic 10871). Identical snapshot assembly to
/// [`load_raw`] — edit shares the Raw pipeline (the source is in-context conditioning, not a distinct
/// model) — but stores the [`edit_descriptor`] so `generate` routes a source `Reference` to the Kontext
/// edit entrypoint. The snapshot MUST carry the Qwen3-VL vision tower (`text_encoder/` `visual.*`) for
/// the grounded half of the dual conditioning; the turnkey keeps it dense ([`crate::convert`]).
pub fn load_edit(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(spec, edit_descriptor())
}

/// Load the **CFG-free Turbo image-edit** generator (`krea_2_turbo_edit`, sc-11640). Identical snapshot
/// assembly to [`load_edit`] — same dual-conditioning edit surface — but `spec.weights` must point at a
/// **Turbo** (distilled) snapshot and the stored [`turbo_edit_descriptor`] makes `generate` route the
/// source(s) to the edit entrypoint on the **few-step CFG-free** schedule (`turbo_schedule`, single
/// conditional forward). The snapshot MUST carry the Qwen3-VL vision tower (`text_encoder/` `visual.*`)
/// for the grounded conditioning — the Turbo turnkey shares Raw's dense text encoder, so it does.
pub fn load_turbo_edit(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(spec, turbo_edit_descriptor())
}

/// Shared loader behind [`load`] / [`load_raw`] / [`load_edit`]: build the residency from a snapshot
/// dir. `Resident` (default) assembles every component now and holds it warm; `Sequential` keeps only
/// the [`LoadSpec`] and re-loads per generate in phase order (encode → drop the text phase →
/// denoise/decode) to bound peak memory to `max(text, DiT+VAE)`. Both use the same per-phase loaders
/// ([`load_krea_text`] / [`load_krea_heavy`]), so the components are byte-identical. `descriptor`
/// selects the variant (Turbo vs Raw vs edit) the returned [`Krea`] renders.
fn load_variant(spec: &LoadSpec, descriptor: ModelDescriptor) -> Result<Box<dyn Generator>> {
    let residency = build_residency(spec, descriptor.id)?;
    Ok(Box::new(Krea {
        descriptor,
        residency,
    }))
}

/// The policy→[`Residency`] dispatch every Krea variant shares (sc-11101; routed through the single
/// [`Residency::from_policy`] seam in sc-11126, F-180), so no variant re-derives the
/// `match offload_policy`. `Resident` eager-loads the text phase + heavy bundle now (the heavy loader
/// with `use_pid = true` so any PiD overlay is loaded once and reused); `Sequential` captures the two
/// per-phase loaders and loads nothing now, deferring each to [`Residency::run`]. Both go through the
/// same [`load_krea_text`] / [`load_krea_heavy`], so the `Resident` composition is byte-identical to
/// the pre-seam one. The up-front [`resolve_root`] fails fast (precision + single-file rejection) for
/// BOTH policies. The deferral is weight-free-testable: under `Sequential` this touches no component
/// weights, so a dispatch that mapped `Sequential → Resident` (ignoring `offload_policy`) would
/// eager-load here and fail the "Sequential defers" unit test.
pub(crate) fn build_residency(
    spec: &LoadSpec,
    id: &'static str,
) -> Result<Residency<KreaText, KreaHeavyOwned>> {
    // Up-front fail-fast for both policies (precision override + single-file rejection).
    let _ = resolve_root(spec, id)?;
    let spec_text = spec.clone();
    let spec_heavy = spec.clone();
    Residency::from_policy(
        spec.offload_policy,
        move || load_krea_text(&spec_text, resolve_root(&spec_text, id)?, id),
        move |use_pid| load_krea_heavy(&spec_heavy, resolve_root(&spec_heavy, id)?, id, use_pid),
    )
}

/// Precision guard (only dense bf16 is wired) + snapshot-dir resolution (rejecting a single-file
/// source), shared by [`load_krea_text`] / [`load_krea_heavy`] and the `Sequential` per-phase loaders
/// (sc-11101).
fn resolve_root<'a>(spec: &'a LoadSpec, id: &str) -> Result<&'a Path> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{id}: only the default dense precision is wired (drop the precision override)"
        )));
    }
    match &spec.weights {
        WeightsSource::Dir(p) => Ok(p),
        WeightsSource::File(_) => Err(Error::Msg(format!(
            "{id} expects a snapshot directory (transformer/ text_encoder/ vae/), not a single \
             .safetensors file"
        ))),
    }
}

/// Resolve the load-time quantize for a component (F-076). Returns `Some(bits)` to quantize the dense
/// base in place, or `None` when there is no quant OR the turnkey is already packed at the requested
/// bits (`quantize()` would be a no-op). Errors on a packed-vs-requested mismatch so e.g. Q4 over a Q8
/// turnkey never silently serves Q8. Shared by the text + heavy loaders (the marker in
/// `transformer/config.json` is model-wide), so both phases decide identically.
pub(crate) fn load_time_quant_bits(spec: &LoadSpec, root: &Path, id: &str) -> Result<Option<i32>> {
    let Some(q) = spec.quantize else {
        return Ok(None);
    };
    match packed_quant_bits(root) {
        Some(packed) => {
            if packed != q.bits() {
                return Err(Error::Msg(format!(
                    "{id}: snapshot is a pre-quantized Q{packed} turnkey but Q{} was \
                     requested; quantize() is a no-op on packed weights so the request would \
                     silently serve Q{packed}. Point at a Q{} snapshot (or a dense one).",
                    q.bits(),
                    q.bits()
                )));
            }
            Ok(None)
        }
        None => Ok(Some(q.bits())),
    }
}

/// Load the Krea text phase (tokenizer + Qwen3-VL-4B condition encoder + vision tower) — the component
/// dropped first under `Sequential`. Applies the optional (F-076-guarded) text-encoder quantize; the
/// VAE + vision tower stay dense (the monolithic `KreaPipeline::quantize` quantized `te` + `dit`, not
/// the VAE/vision), so the `Resident` and `Sequential` paths build byte-identical text phases.
pub(crate) fn load_krea_text(spec: &LoadSpec, root: &Path, id: &str) -> Result<KreaText> {
    let mut text = KreaText::from_snapshot(root)?;
    if let Some(bits) = load_time_quant_bits(spec, root, id)? {
        text.quantize(bits)?;
    }
    Ok(text)
}

/// Load the Krea heavy render phase (single-stream DiT + VAE + the optional PiD overlay) — everything
/// but the text phase. Install Raw-trained LoRA/LoKr adapters onto the DiT BEFORE the optional quantize,
/// so the residual stacks over the (possibly already-packed) base (the Lens load→apply→quantize order);
/// the shared seam errors (never silently drops) on an adapter target that matches no module. Factored
/// so `Sequential` loads these AFTER the text phase is dropped (bounding peak to `max(text, DiT+VAE)`).
fn load_krea_heavy(
    spec: &LoadSpec,
    root: &Path,
    id: &str,
    load_pid: bool,
) -> Result<KreaHeavyOwned> {
    let mut heavy = KreaHeavy::from_snapshot(root)?;
    if !spec.adapters.is_empty() {
        heavy.apply_adapters(&spec.adapters)?;
    }
    if let Some(bits) = load_time_quant_bits(spec, root, id)? {
        heavy.quantize(bits)?;
    }
    // Optional PiD decoder overlay (sc-7845): Krea reuses the Qwen-Image latent space, so it loads the
    // same `qwenimage` student + Gemma-2 caption encoder when `spec.pid` is set AND this generate uses
    // it (`load_pid`, F-177) — Resident passes `true` (loaded once, reused), Sequential passes
    // `req.use_pid` so a non-PiD generate skips the student + its Gemma-2 caption encoder entirely.
    let pid = if load_pid {
        spec.pid
            .as_ref()
            .map(|p| PidEngine::from_spec(p, PID_BACKBONE))
            .transpose()?
    } else {
        None
    };
    Ok(KreaHeavyOwned { heavy, pid })
}

mlx_gen::impl_generator!(Krea {
    validate: |s, req| validate_request(&s.descriptor, req),
    generate: generate_impl,
});

impl Krea {
    /// Text-encode the prompt (and, for true CFG, the negative) per the residency (sc-11101). `Resident`
    /// borrows the warm Qwen3-VL-4B text phase (byte-identical to the pre-sc-11101 per-image re-encode —
    /// the encode is deterministic, the per-image variation comes from the seed inside `render`);
    /// `Sequential` loads the text phase, encodes, then the seam materializes + DROPS it + `clear_cache()`
    /// so its ~4 GB frees before the DiT/VAE load. `is_edit` uses the Qwen3-VL grounded encode over the
    /// source image; `is_raw`/Turbo the plain text encode. The unconditional context is built only when
    /// `guidance > 0` (reference `cfg = guidance > 0`; Turbo is CFG-free → always `None`). Called by the
    /// shared residency seam's encode closure with the phase-A `text` component.
    #[allow(clippy::too_many_arguments)]
    fn encode_contexts(
        &self,
        text: &KreaText,
        req: &GenerationRequest,
        is_raw: bool,
        is_edit: bool,
        guidance: f32,
        negative: &str,
        edit_source: Option<&Image>,
    ) -> Result<KreaContexts> {
        if is_edit {
            let src = edit_source.ok_or_else(|| {
                Error::Msg(format!(
                    "{}: edit requires a source image",
                    self.descriptor.id
                ))
            })?;
            let pos = text.encode_grounded(src, &req.prompt)?;
            let neg = if guidance > 0.0 {
                Some(text.encode_grounded(src, negative)?)
            } else {
                None
            };
            Ok(KreaContexts { pos, neg })
        } else if is_raw {
            let pos = text.encode(&req.prompt)?;
            let neg = if guidance > 0.0 {
                Some(text.encode(negative)?)
            } else {
                None
            };
            Ok(KreaContexts { pos, neg })
        } else {
            Ok(KreaContexts {
                pos: text.encode(&req.prompt)?,
                neg: None,
            })
        }
    }

    /// The rich-`Result` body behind [`Generator::generate`] — kept on the crate's own
    /// [`mlx_gen::Error`] so `?` lifts `mlx_rs` device exceptions transparently; the trait wrapper
    /// bridges the tail into [`gen_core::Error`]. Renders `req.count` images, one per seed (`seed + n`,
    /// mirroring the reference per-prompt seeding), through the residency (encode → drop text phase under
    /// `Sequential` → load heavy → per-image render → free heavy).
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        validate_request(&self.descriptor, req)?;
        let base_seed = req.seed.unwrap_or_else(default_seed);
        // Variant read back off the descriptor id: Raw = full-CFG undistilled (52-step, dynamic-mu);
        // Turbo = CFG-free distilled (8-step, fixed mu). One `Krea` struct, two render paths. The edit
        // variant (epic 10871) shares Raw's full-CFG sampler — an edit denoises from noise under true CFG
        // — so `is_raw` (the full-CFG path selector for schedule/steps/guidance) covers edit too; only the
        // per-image entrypoint below differs (`is_edit` → the Kontext edit path, not img2img/t2i).
        // The distilled CFG-free Turbo edit (sc-11640): routes to the SAME Kontext edit entrypoint as
        // `krea_2_edit`, but on the few-step `turbo_schedule` at `guidance = 0` (single conditional
        // forward). So it is an edit (`is_edit`) but NOT a full-CFG variant (`is_raw`).
        let is_turbo_edit = self.descriptor.id == KREA_2_TURBO_EDIT_ID;
        let is_edit = self.descriptor.id == KREA_2_EDIT_ID || is_turbo_edit;
        // `is_raw` gates the full-CFG sampler (52-step, dynamic-mu `base_schedule`, guidance) — Raw and
        // the full-CFG `krea_2_edit`, but NOT `krea_2_turbo_edit` (distilled few-step, CFG-free).
        let is_raw = self.descriptor.id == KREA_2_RAW_ID || (is_edit && !is_turbo_edit);
        let steps = req.steps.unwrap_or(if is_raw {
            DEFAULT_RAW_STEPS
        } else {
            DEFAULT_STEPS
        }) as usize;
        // Decode seam (sc-7845) + `from_ldm` early-stop (sc-7993): resolve the achieved degrade σ + the
        // truncation `keep` from the (seed-independent) schedule; the PiD decoder itself is built below
        // (it needs the heavy phase's PiD engine). Raw/edit use the resolution-dynamic schedule.
        let sigmas = if is_raw {
            base_schedule(steps, req.width, req.height, req.scheduler.as_deref())
        } else {
            turbo_schedule(steps, req.scheduler.as_deref())
        };
        let (capture_sigma, keep) = flow_capture_for_request(req, &sigmas, 0);
        // Edit extracts its own ordered source list (a single `Reference` or a `MultiReference`
        // scene+person pair, epic 10871 P1.3); img2img/t2i use the single-`Reference` helper. Kept
        // separate so an edit's `MultiReference` never trips `single_reference`'s "exactly one
        // Reference" img2img guard, and an img2img job still can't smuggle in two references.
        let edit_sources = if is_edit {
            edit_references(req)?
        } else {
            Vec::new()
        };
        let reference = if is_edit {
            None
        } else {
            single_reference(req)?
        };
        // A PiD `from_ldm` early-stop CAPTURE is not wired for img2img yet (sc-10121): reject THAT combo
        // rather than silently desync the decoder's σ from the img2img-sliced schedule. A capture is
        // active only when `keep` lands BEFORE the end of the schedule; the no-capture path reports
        // `keep == sigmas.len()` and is fine with img2img (`flow_capture_for_request` returns
        // `sigmas.len()`, NOT `usize::MAX`, for no-capture).
        if img2img_conflicts_with_capture(reference.is_some(), keep, sigmas.len()) {
            return Err(Error::Msg(format!(
                "{}: PiD from_ldm early-stop is not supported with img2img reference conditioning \
                 (tracked in sc-10121)",
                self.descriptor.id
            )));
        }
        // Raw CFG knobs: guidance defaults to the reference Raw preset, an empty/absent negative → ""
        // (reference `negative_prompts = [""] * n`). Inert on the Turbo (CFG-free) t2i/img2img path.
        // FORCED to 0 for `krea_2_turbo_edit` so the edit runs a single conditional forward — its
        // descriptor advertises no guidance, so `req.guidance` is already rejected upstream; this just
        // pins the CFG-free default that makes `encode` skip the unconditional grounded context.
        let guidance = if is_turbo_edit {
            0.0
        } else {
            req.guidance.unwrap_or(DEFAULT_RAW_GUIDANCE)
        };
        let negative = req.negative_prompt.clone().unwrap_or_default();

        // Phase A: prompt → context(s) (sc-11101; sc-11125). Under `Sequential` the shared seam loads
        // the Qwen3-VL-4B text phase, encodes, materializes, then DROPS it + `clear_cache()` so its
        // ~4 GB frees before the DiT/VAE load below — the peak-bounding win. Under `Resident` it borrows
        // the warm text phase. Edit grounds on the first source image.
        let edit_source = edit_sources.first().copied();
        self.residency.run(
            &req.cancel,
            req.use_pid,
            on_progress,
            |text: &KreaText| {
                self.encode_contexts(text, req, is_raw, is_edit, guidance, &negative, edit_source)
            },
            // Materialize pos (+neg) while the text phase is still alive (Sequential only) — MLX is
            // lazy, so an un-evaluated context keeps the encoder referenced and the drop frees nothing.
            |ctx: &KreaContexts| {
                match &ctx.neg {
                    Some(neg) => mlx_rs::transforms::eval([&ctx.pos, neg])?,
                    None => mlx_rs::transforms::eval([&ctx.pos])?,
                }
                Ok(())
            },
            // Phase B: heavy render components (DiT + VAE + PiD). The render dispatch below runs
            // identically for both residencies.
            |heavy_owned, ctx, on_progress| {
                let heavy = heavy_owned.as_ref();

                // PiD decode overlay (sc-7845): one decoder serves the whole count loop (same prompt → same
                // caption). Errors if `req.use_pid` but the model wasn't loaded with `LoadSpec::pid`; `None`
                // → the native VAE. Resolved against the heavy phase's PiD engine.
                let pid_decoder = resolve_pid_decoder_at_sigma(
                    heavy.pid,
                    req,
                    base_seed,
                    self.descriptor.id,
                    capture_sigma,
                )?;
                let decoder = pid_decoder.as_ref().map(|d| d as &dyn LatentDecoder);

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
                    // The one render body per path (sc-11101): the same `KreaHeavy::render_*` for both
                    // residencies, so a Sequential job (text phase already dropped) is byte-identical to Resident.
                    let img = if is_edit {
                        // Kontext-style edit (epic 10871): the source image(s) are kept as in-context
                        // conditioning (VAE tokens + the Qwen3-VL grounding baked into `ctx`) — NOT a noised
                        // img2img init. Denoises from PURE NOISE under full CFG, so a reference's strength is
                        // ignored; `edit_sources` is the ordered slice the pipeline VAE-encodes at successive
                        // RoPE frames. The `krea2_identity_edit` LoRA in `spec.adapters` steers the edit.
                        heavy.heavy.render_edit(
                            &ctx.pos,
                            ctx.neg.as_ref(),
                            guidance,
                            // Distilled Turbo edit → few-step `turbo_schedule`; Raw edit → dynamic-mu
                            // `base_schedule`. Matches the capture-σ `sigmas` selector above (`is_raw`).
                            is_turbo_edit,
                            &edit_sources,
                            &opts,
                            decoder,
                            &req.cancel,
                            on_progress,
                        )?
                    } else if let Some((init, ref_strength)) = reference {
                        // Reference fidelity: the Reference's own strength wins, else the request-level img2img
                        // strength, else the 0.5 mid default (the full-range slider's default; A2/A3).
                        let strength = ref_strength.or(req.strength).unwrap_or(0.5);
                        // img2img dispatch splits by variant: Raw takes the true-CFG entrypoint (guidance +
                        // negative prompt honored, sc-10224); Turbo takes the CFG-free distilled one (sc-10135).
                        if is_raw {
                            heavy.heavy.render_base_img2img(
                                &ctx.pos,
                                ctx.neg.as_ref(),
                                guidance,
                                init,
                                strength,
                                &opts,
                                decoder,
                                &req.cancel,
                                on_progress,
                            )?
                        } else {
                            heavy.heavy.render_turbo_img2img(
                                &ctx.pos,
                                init,
                                strength,
                                &opts,
                                decoder,
                                &req.cancel,
                                on_progress,
                            )?
                        }
                    } else if is_raw {
                        heavy.heavy.render_base(
                            &ctx.pos,
                            ctx.neg.as_ref(),
                            guidance,
                            &opts,
                            decoder,
                            keep,
                            &req.cancel,
                            on_progress,
                        )?
                    } else {
                        heavy.heavy.render_turbo(
                            &ctx.pos,
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
            },
        )
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

/// Extract the single reference image + its optional `strength` for img2img (epic 8588 slice A), or
/// `None` for plain txt2img. Krea conditions on exactly one reference image; `MultiReference` or more
/// than one `Reference` errors. Both variants advertise `Reference` (Turbo → CFG-free img2img sc-10135,
/// Raw → CFG img2img sc-10224), so this is reached on either path; the `generate_impl` dispatch then
/// picks the matching entrypoint by `is_raw`. Mirrors the FLUX single-reference idiom.
fn single_reference(req: &GenerationRequest) -> Result<Option<(&Image, Option<f32>)>> {
    match req.conditioning.as_slice() {
        [] => Ok(None),
        [Conditioning::Reference { image, strength }] => Ok(Some((image, *strength))),
        _ => Err(Error::Msg(
            "krea_2_turbo: img2img supports exactly one Reference image".into(),
        )),
    }
}

/// The most reference images a Krea edit accepts (epic 10871 P1.3): scene = image 1, person = image 2.
/// The edit LoRA was trained on this fixed pair order — swapping degrades identity — and the ComfyUI-
/// Krea2Edit node caps at two. Mirrors candle-gen-krea's `MAX_EDIT_REFERENCES`.
const MAX_EDIT_REFERENCES: usize = 2;

/// The ordered source image(s) for a Krea edit (epic 10871): one `Conditioning::Reference` (the common
/// single-source edit) or one `Conditioning::MultiReference` (scene, then person — the fixed P1.3
/// order). At least one is required; at most [`MAX_EDIT_REFERENCES`]. Distinct from [`single_reference`]
/// (img2img), which rejects `MultiReference` and any count > 1 — the edit surface is the only one that
/// advertises `MultiReference` ([`edit_descriptor`]). The returned slice is passed straight to
/// [`KreaPipeline::generate_edit_with_progress`], which VAE-encodes each at successive RoPE frames.
fn edit_references(req: &GenerationRequest) -> Result<Vec<&Image>> {
    let sources: Vec<&Image> = match req.conditioning.as_slice() {
        [Conditioning::Reference { image, .. }] => vec![image],
        [Conditioning::MultiReference { images }] => images.iter().collect(),
        [] => {
            return Err(Error::Msg(format!(
                "{KREA_2_EDIT_ID}: edit requires a source image (a Reference or a MultiReference)"
            )))
        }
        _ => {
            return Err(Error::Msg(format!(
                "{KREA_2_EDIT_ID}: edit expects a single Reference or one MultiReference of sources"
            )))
        }
    };
    if sources.is_empty() {
        return Err(Error::Msg(format!(
            "{KREA_2_EDIT_ID}: edit requires at least one source image"
        )));
    }
    if sources.len() > MAX_EDIT_REFERENCES {
        return Err(Error::Msg(format!(
            "{KREA_2_EDIT_ID}: at most {MAX_EDIT_REFERENCES} references are supported \
             (scene = image 1, person = image 2)"
        )));
    }
    Ok(sources)
}

/// Whether an img2img reference conflicts with an ACTIVE PiD `from_ldm` early-stop capture — the combo
/// deferred to sc-10121. A capture is active ONLY when [`flow_capture_for_request`]'s truncation `keep`
/// lands strictly before the end of the schedule (`keep < num_sigmas`); the no-capture full-denoise path
/// reports `keep == num_sigmas` (the PiD super-res decoder / native VAE then decodes the clean final
/// latent — fine with img2img). Extracted + tested because the original inline guard compared `keep`
/// against `usize::MAX`, a sentinel `flow_capture_for_request` never returns, so it rejected EVERY
/// img2img generation (the on-device break this fixes).
fn img2img_conflicts_with_capture(has_reference: bool, keep: usize, num_sigmas: usize) -> bool {
    has_reference && keep < num_sigmas
}

// Link-time registration (epic 3720): the macro emits the `inventory::submit!` and bridges the
// crate's rich `Result` into the registry's backend-neutral `gen_core::Result`. Four variants register
// here — `krea_2_turbo` (distilled t2i, CFG-free), `krea_2_raw` (undistilled t2i, full-CFG; epic 9992),
// `krea_2_edit` (the Raw pipeline routed to the Kontext edit entrypoint; epic 10871), and
// `krea_2_turbo_edit` (that edit surface on the distilled few-step CFG-free schedule; sc-11640).
mlx_gen::register_generators! {
    descriptor => load,
    raw_descriptor => load_raw,
    edit_descriptor => load_edit,
    turbo_edit_descriptor => load_turbo_edit,
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core;
    use mlx_gen::{AdapterKind, AdapterSpec, OffloadPolicy};

    fn req(w: u32, h: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "a red apple on a wooden table".into(),
            width: w,
            height: h,
            ..Default::default()
        }
    }

    fn tiny_image() -> Image {
        Image {
            width: 16,
            height: 16,
            pixels: vec![0u8; 16 * 16 * 3],
        }
    }

    /// A 1024² request carrying `refs` `Reference` conditionings (each with `strength`).
    fn ref_req(refs: usize, strength: Option<f32>) -> GenerationRequest {
        let mut r = req(1024, 1024);
        r.conditioning = (0..refs)
            .map(|_| Conditioning::Reference {
                image: tiny_image(),
                strength,
            })
            .collect();
        r
    }

    #[test]
    fn both_variants_advertise_reference_conditioning() {
        // img2img is now on BOTH variants: Turbo → CFG-free (sc-10135), Raw → CFG (sc-10224).
        assert_eq!(
            descriptor().capabilities.conditioning,
            vec![ConditioningKind::Reference]
        );
        assert_eq!(
            raw_descriptor().capabilities.conditioning,
            vec![ConditioningKind::Reference]
        );
    }

    #[test]
    fn validate_reference_accepted_on_both_variants() {
        // A single Reference (img2img) validates on Turbo AND Raw (sc-10224). The conditioning-floor
        // checks the KIND is allowed; the exactly-one-Reference count is enforced later by
        // `single_reference` (see `single_reference_extracts_one_or_errors`).
        assert!(validate_request(&descriptor(), &ref_req(1, Some(0.5))).is_ok());
        assert!(validate_request(&raw_descriptor(), &ref_req(1, Some(0.5))).is_ok());
    }

    #[test]
    fn img2img_conflicts_only_with_an_active_capture() {
        // sc-10121 sentinel regression: the no-capture full-denoise path reports `keep == num_sigmas`
        // (NOT usize::MAX). Plain img2img (no PiD capture) must be ACCEPTED; only a real from_ldm capture
        // (`keep < num_sigmas`) conflicts.
        let num_sigmas = 9; // an 8-step Turbo schedule has 9 sigmas.
                            // No reference → never a conflict (plain txt2img, any keep).
        assert!(!img2img_conflicts_with_capture(false, 5, num_sigmas));
        assert!(!img2img_conflicts_with_capture(
            false, num_sigmas, num_sigmas
        ));
        // Reference + NO capture (keep == num_sigmas) → allowed (the on-device break; must NOT conflict).
        assert!(!img2img_conflicts_with_capture(
            true, num_sigmas, num_sigmas
        ));
        // Reference + an ACTIVE from_ldm capture (keep truncates early) → the sc-10121 conflict.
        assert!(img2img_conflicts_with_capture(true, 5, num_sigmas));
        assert!(img2img_conflicts_with_capture(true, 0, num_sigmas));
        // The old bug: `keep == usize::MAX` was the (wrong) allowed sentinel; the real allowed sentinel
        // is `num_sigmas`. A capture value of usize::MAX (never produced) would be treated as no-conflict
        // here only because it is not < num_sigmas — but the point is num_sigmas itself is now allowed.
        assert!(!img2img_conflicts_with_capture(
            true,
            usize::MAX,
            num_sigmas
        ));
    }

    #[test]
    fn single_reference_extracts_one_or_errors() {
        // No conditioning → plain txt2img.
        assert!(single_reference(&req(1024, 1024)).unwrap().is_none());
        // Exactly one → the image + its strength.
        let r1 = ref_req(1, Some(0.4));
        let one = single_reference(&r1).unwrap();
        assert_eq!(one.map(|(_, s)| s), Some(Some(0.4)));
        // More than one → error (Krea conditions on a single reference).
        assert!(single_reference(&ref_req(2, None)).is_err());
    }

    #[test]
    fn edit_references_takes_one_reference_or_a_scene_person_pair() {
        // A single `Reference` → one source (the common single-image edit).
        assert_eq!(edit_references(&ref_req(1, None)).unwrap().len(), 1);

        // A `MultiReference` → the ordered source list (scene, then person; P1.3).
        let mut two = req(1024, 1024);
        two.conditioning = vec![Conditioning::MultiReference {
            images: vec![tiny_image(), tiny_image()],
        }];
        assert_eq!(edit_references(&two).unwrap().len(), 2);

        // Empty conditioning → error (an edit needs a source).
        assert!(edit_references(&req(1024, 1024)).is_err());

        // Past the scene/person cap → error naming the fixed order.
        let mut three = req(1024, 1024);
        three.conditioning = vec![Conditioning::MultiReference {
            images: vec![tiny_image(), tiny_image(), tiny_image()],
        }];
        let err = edit_references(&three).unwrap_err().to_string();
        assert!(err.contains("scene") && err.contains("person"), "{err}");

        // Two separate `Reference`s (not a `MultiReference`) → error: an edit takes one Reference or
        // one MultiReference, never a bare list.
        assert!(edit_references(&ref_req(2, None)).is_err());
    }

    #[test]
    fn descriptor_is_krea_2_turbo() {
        let d = descriptor();
        assert_eq!(d.id, "krea_2_turbo");
        assert_eq!(d.family, "krea_2");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        // CFG-free distilled Turbo: no guidance, no negative prompt. img2img reference conditioning
        // (sc-10135) IS advertised (the only conditioning surface).
        assert!(!d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_negative_prompt);
        assert_eq!(
            d.capabilities.conditioning,
            vec![ConditioningKind::Reference]
        );
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

    // --- Image-edit variant (Kontext-style) — epic 10871 ---

    #[test]
    fn edit_descriptor_is_krea_2_edit_and_cfg_capable() {
        let d = edit_descriptor();
        assert_eq!(d.id, "krea_2_edit");
        assert_eq!(d.id, KREA_2_EDIT_ID);
        assert_eq!(d.family, "krea_2");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        // Edit shares Raw's full-CFG surface (guidance + negative prompt; an edit denoises from noise
        // under true CFG), derived from `raw_descriptor()` — so it stays in lockstep with Raw.
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_true_cfg);
        // The source rides a single `Reference`, or a scene+person pair rides a `MultiReference`
        // (epic 10871 P1.3); the `krea2_identity_edit` LoRA rides `spec.adapters`.
        assert_eq!(
            d.capabilities.conditioning,
            vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference
            ]
        );
        assert!(d.capabilities.supports_lora && d.capabilities.supports_lokr);
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
    }

    #[test]
    fn edit_validate_accepts_reference_with_guidance_and_negative() {
        // An edit job: a source Reference + full-CFG knobs must pass the capability floor.
        let mut r = ref_req(1, None);
        r.guidance = Some(3.5);
        r.negative_prompt = Some("blurry, lowres".into());
        assert!(validate_request(&edit_descriptor(), &r).is_ok());
    }

    #[test]
    fn edit_reachable_via_registry_by_id() {
        assert!(
            gen_core::registry::generators().any(|r| (r.descriptor)().id == KREA_2_EDIT_ID),
            "id {KREA_2_EDIT_ID} not registered"
        );
        // Same snapshot loader as Raw/Turbo — a single-file weights source is rejected the same way.
        let file = LoadSpec::new(WeightsSource::File("/tmp/x.safetensors".into()));
        let e = load_edit(&file).err().expect("error").to_string();
        assert!(e.contains("snapshot directory"), "got: {e}");
    }

    // --- CFG-free Turbo image-edit variant — sc-11640 ---

    #[test]
    fn turbo_edit_descriptor_is_krea_2_turbo_edit_and_cfg_free() {
        let d = turbo_edit_descriptor();
        assert_eq!(d.id, "krea_2_turbo_edit");
        assert_eq!(d.id, KREA_2_TURBO_EDIT_ID);
        assert_eq!(d.family, "krea_2");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        // Derived from the distilled Turbo descriptor: CFG-free (no guidance, no user negative prompt),
        // UNLIKE the full-CFG `krea_2_edit`. This is the recipe difference the spike validates.
        assert!(!d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_true_cfg);
        // Same edit conditioning surface as `edit_descriptor` — a single `Reference` or a scene+person
        // `MultiReference`; the `krea2_identity_edit` LoRA rides `spec.adapters`.
        assert_eq!(
            d.capabilities.conditioning,
            vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference
            ]
        );
        assert!(d.capabilities.supports_lora && d.capabilities.supports_lokr);
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        // Same curated sampler/scheduler menu + size bounds as Turbo t2i (shared `descriptor()` base).
        assert_eq!(d.capabilities.samplers, descriptor().capabilities.samplers);
        assert!(d.capabilities.mac_only);
    }

    #[test]
    fn turbo_edit_rejects_guidance_and_negative_prompt() {
        // The CFG-free floor (like Turbo t2i) rejects both — the edit runs a single conditional forward.
        let mut r = ref_req(1, None);
        r.guidance = Some(3.5);
        assert!(validate_request(&turbo_edit_descriptor(), &r).is_err());
        let mut r = ref_req(1, None);
        r.negative_prompt = Some("blurry".into());
        assert!(validate_request(&turbo_edit_descriptor(), &r).is_err());
        // A source Reference with NO CFG knobs passes the capability floor.
        assert!(validate_request(&turbo_edit_descriptor(), &ref_req(1, None)).is_ok());
    }

    #[test]
    fn turbo_edit_reachable_via_registry_by_id() {
        assert!(
            gen_core::registry::generators().any(|r| (r.descriptor)().id == KREA_2_TURBO_EDIT_ID),
            "id {KREA_2_TURBO_EDIT_ID} not registered"
        );
        // Same snapshot loader as the other variants — a single-file weights source is rejected.
        let file = LoadSpec::new(WeightsSource::File("/tmp/x.safetensors".into()));
        let e = load_turbo_edit(&file).err().expect("error").to_string();
        assert!(e.contains("snapshot directory"), "got: {e}");
    }

    // ── F-180 (sc-11126): weight-free, default-run proof that Krea's dispatch HONORS
    // `offload_policy` — not a smoke test. `build_residency` points at a non-existent snapshot dir
    // (a *directory* source, so the up-front `resolve_root` precision/single-file guard passes). The
    // discriminator is the deferral:
    //   * `Sequential` must capture the two loaders and touch NO component weights → `Ok`, and the
    //     built residency is `Sequential` (`is_sequential()`).
    //   * `Resident` must eager-load the text encoder from that non-existent dir → `Err`.
    // A dispatch that ignored `offload_policy` and always built `Resident` (the F-172 bug class) would
    // eager-load under a `Sequential` request and turn the first assertion's `Ok` into an `Err` —
    // this test would fail. That is exactly the ignore-`offload_policy` regression the smoke tests miss.
    fn missing_snapshot_spec(policy: OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(
            "/nonexistent/krea-residency-test-snapshot".into(),
        ))
        .with_offload_policy(policy)
    }

    #[test]
    fn build_residency_sequential_defers_all_component_loads() {
        // Sequential defers every heavy/text load, so a missing snapshot dir is NOT touched here.
        let res = build_residency(
            &missing_snapshot_spec(OffloadPolicy::Sequential),
            KREA_2_TURBO_ID,
        )
        .expect("Sequential must defer loads and not touch the (missing) snapshot dir");
        assert!(
            res.is_sequential(),
            "Sequential policy must build a Sequential residency (the deferred state machine)"
        );
    }

    #[test]
    fn build_residency_resident_eager_loads_and_fails_on_missing_snapshot() {
        // Resident eager-loads the text encoder now, so the missing snapshot dir surfaces as an error
        // at construction — the flip side that proves the Sequential test's `Ok` came from deferral.
        let err = build_residency(
            &missing_snapshot_spec(OffloadPolicy::Resident),
            KREA_2_TURBO_ID,
        )
        .err()
        .expect("Resident must eager-load and fail on a missing snapshot dir");
        // A load/IO error, not the precision/single-file guard (which a Dir source passes).
        let msg = err.to_string();
        assert!(
            !msg.contains("single .safetensors file") && !msg.contains("precision override"),
            "expected an eager-load failure, got the up-front guard: {msg}"
        );
    }
}
