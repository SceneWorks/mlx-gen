//! `Sdxl` — the Stable Diffusion XL implementation of [`mlx_gen::Generator`], plus its
//! [`descriptor`]/[`load`] entry points and the `inventory` registration that wires it into
//! `mlx_gen`'s registry under the id `"sdxl"` (the SceneWorks worker's `payload.model`).
//!
//! SDXL is the in-process Apple `mlx-examples/stable_diffusion` path (vendored at
//! `_vendor/mlx_sd/`) brought into Rust — a **U-Net** generator (conv ResBlocks + spatial/cross
//! attention + time/`text_time` micro-conditioning), dual CLIP text encoders, an SDXL VAE, and a
//! discrete Euler-Ancestral sampler with real classifier-free guidance. Parity target = the
//! vendored fp16 reference path (`StableDiffusionXL.generate_latents`), validated stage-by-stage.
//!
//! Slices land incrementally (sc-2400): this module starts as the contract + capability surface;
//! [`load`] assembles components as each slice (tokenizer → text encoders → U-Net → VAE → sampler)
//! is wired and parity-proven.

use mlx_gen::{
    curated_scheduler_names, default_seed, schedule_sigmas, AlphaSchedule, Capabilities,
    Conditioning, ConditioningKind, DiffusionSampler, DiscreteModelSampling, Error,
    GenerationOutput, GenerationRequest, Generator, Image, LatentDecoder, LcmSampler,
    LightningSampler, LoadSpec, Modality, ModelDescriptor, OffloadPolicy, Precision, Progress,
    Quant, Residency, Result, Scheduler, Solver, TcdSampler, WeightsSource,
};
use mlx_rs::ops::{add, concatenate_axis, multiply};
use mlx_rs::Dtype;
use std::path::Path;

use mlx_gen::array::scalar;
use mlx_gen::gen_core::sampling::{vp_capture_plan, VpCapturePlan};
use mlx_gen_pid::{resolve_pid_decoder_at_sigma, PidEngine};

use crate::config::DiffusionConfig;
use crate::inpaint::{preprocess_mask, InpaintBlend};
use crate::ip_adapter::IpImageEncoder;
use crate::loader;
use crate::pipeline::{
    decode_image, denoise, denoise_cfgpp, denoise_curated, denoise_inpaint, denoise_ip,
    denoise_multi_control, encode_conditioning, encode_init_latents, preprocess_control_image,
    text_time_ids, ControlContext, Denoiser,
};
use crate::sampler::{AncestralEuler, EulerSampler};
use crate::text_encoder::ClipTextEncoder;
use crate::tokenizer::ClipBpeTokenizer;
use crate::unet::{ControlNet, UNet2DConditionModel};
use crate::vae::Autoencoder;

/// img2img default strength (the vendored `generate_latents_from_image` default).
const DEFAULT_STRENGTH: f32 = 0.8;
/// Masked-inpaint / outpaint default strength — the worker's `SdxlDiffusersAdapter` uses 0.85 for
/// `use_inpaint`/`outpaint` (vs 0.6 for a plain edit). An explicit request strength still wins.
const INPAINT_DEFAULT_STRENGTH: f32 = 0.85;
/// Default `ip_adapter_scale` (sc-3059) when a request doesn't override it (the worker's plus-face
/// default ≈ 0.6). In IP mode the `Reference` strength field carries the IP scale.
const IP_DEFAULT_SCALE: f32 = 0.6;
/// Default per-branch `conditioning_scale` for a ControlNet `Conditioning::Control` that leaves
/// `scale = None` (F-085) — the diffusers `controlnet_conditioning_scale` full-strength default. An
/// explicit `Some(x)`, including `Some(0.0)` for an inert branch, overrides it.
const DEFAULT_CONTROLNET_SCALE: f32 = 1.0;

/// The SDXL compute dtype: the U-Net + both CLIP text encoders run **fp16** (the production
/// reference `StableDiffusionXL(float16=True)`); the VAE loads f32 inside its own loader. Shared by
/// the eager (`Resident`) load and the per-generation (`Sequential`) component loaders so both build
/// byte-identical components.
const DTYPE: Dtype = Dtype::Float16;

/// SDXL-base-1.0 production defaults (the SceneWorks `MlxSdxlAdapter`): 30 inference steps,
/// CFG 7.0, native 1024². Used when a request omits the corresponding field (consumed by the
/// `generate` pipeline slice, sc-2400 S5).
pub(crate) const DEFAULT_STEPS: u32 = 30;
pub(crate) const DEFAULT_GUIDANCE: f32 = 7.0;

/// The few-step acceleration samplers (sc-2769). Selected per request via `req.sampler`; each is
/// paired with its acceleration LoRA at load (`spec.adapters`) by the caller (the SceneWorks
/// variant manifest, epic 2755) — selecting one without its LoRA loaded yields undertrained noise.
pub(crate) const ACCEL_SAMPLERS: [&str; 3] = ["lcm", "lightning", "hyper"];

/// `original_inference_steps` for the LCM/TCD timestep selection (diffusers' default).
const LCM_ORIGINAL_STEPS: usize = 50;

/// Per-variant few-step defaults `(steps, CFG, TCD eta)`, applied when the request omits `steps`/
/// `guidance`. **Locked by the sc-2758 SDXL acceleration A/B characterization** (re-tuned here per
/// sc-2907; `sdxl` and `realvisxl` came out identical, so the table keys on the sampler only). CFG is
/// 1.0 (off) for all three — Lightning/Hyper are trained CFG-free and LCM-LoRA runs at low/no CFG —
/// which also halves the per-step UNet work. Lightning's step count must match the loaded LoRA
/// (2/4/8); LCM uses a single LoRA at any step count.
fn accel_defaults(sampler: &str) -> (u32, f32, f32) {
    match sampler {
        // LCM is the weakest method and 4 steps is too soft as a default; sc-2758 locks 8 as the
        // quality floor (the LCM-LoRA is step-free, so this is a plain default, not LoRA-bound).
        "lcm" => (8, 1.0, 0.0),
        "lightning" => (4, 1.0, 0.0),
        // Hyper-SD: TCD, deterministic (eta=0) — sc-2758 locked eta=0 for the step-graded
        // (1/2/4/8-step) LoRAs, which is the default LoRA path here.
        "hyper" => (4, 1.0, 0.0),
        _ => (DEFAULT_STEPS, DEFAULT_GUIDANCE, 0.0),
    }
}

/// Registry id — matches the SceneWorks worker's `payload.model` (`MODEL_TARGETS["sdxl"]`).
pub const MODEL_ID: &str = "sdxl";

/// PiD latent-space backbone tag (epic 7840, sc-7848): the `sdxl` student in
/// [`mlx_gen_pid::registry`] (SDXL's 4-ch, `0.13025`-affine VAE latent). The whole SDXL family
/// shares this latent space, so [`mlx-gen-kolors`](mlx_gen_kolors) (and the RealVisXL variants,
/// which register under this same `"sdxl"` generator) reuse this tag rather than redeclaring it.
pub const PID_BACKBONE: &str = "sdxl";

/// SDXL's identity + capabilities — constructible without loading weights (registry
/// introspection). Capability flags are turned on as each slice lands and is parity-proven, so the
/// descriptor never advertises a path that isn't wired (avoids the false-capability trap —
/// [[false-green-gates-mask-descope]]).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "sdxl",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            // SDXL uses real classifier-free guidance: honors the negative prompt + a CFG scale.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            // img2img Reference (sc-2638) + masked inpaint/outpaint (Mask, sc-3057) + tile-ControlNet
            // detail (Control, sc-3058 — requires a control checkpoint via LoadSpec::control). LoRA
            // (kohya `lora_unet_` + PEFT, sc-2639) and LoKr (sc-2640 — Rust is more capable than the
            // vendored path, which rejects LoKr) are wired.
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::Mask,
                ConditioningKind::Control,
            ],
            supports_lora: true,
            supports_lokr: true,
            // `euler_ancestral` is the production default (full-CFG, 30-step, the bespoke vendored
            // ancestral loop); `lcm`/`lightning`/`hyper` are the few-step acceleration samplers
            // (sc-2769), each driven by its diffusers-faithful schedule and paired with an acceleration
            // LoRA at load. The remaining names (`euler`/`heun`/`dpmpp_2m`/`dpmpp_sde`/`uni_pc`/`ddim`)
            // are the unified curated solvers (epic 7114, sc-7121) — the additive k-diffusion path over
            // `DiscreteModelSampling`; selecting one (or a non-`discrete` scheduler) routes to
            // `denoise_curated` while the default stays byte-exact. A request naming any other sampler
            // is rejected in `validate_request` rather than silently downgraded.
            samplers: vec![
                "euler_ancestral",
                "lcm",
                "lightning",
                "hyper",
                "euler",
                "heun",
                "dpmpp_2m",
                "dpmpp_sde",
                "uni_pc",
                "ddim",
            ],
            // `discrete` is the native ancestral schedule; the rest are the curated σ schedulers
            // (epic 7114 scheduler axis) usable with any curated sampler.
            schedulers: {
                let mut s = vec!["discrete"];
                s.extend(curated_scheduler_names());
                s
            },
            // Plain CFG (the shared `gen_core::guidance::cfg` over `MlxLatentOps`, epic 7434 P3 sc-7443;
            // byte-identical to the retired hand form, shared by Kolors/InstantID/PuLID via
            // `denoise_core`), plus CFG++ (`cfg_pp`, sc-8256) on the curated path — gated at dispatch to a
            // CFG++-compatible base solver (euler/ddim/dpmpp_2m) + an active guidance gap.
            supported_guidance_methods: vec!["cfg", "cfg_pp"],
            min_size: 512,
            max_size: 2048,
            max_count: 8,
            mac_only: true,
            // On-the-fly Q4/Q8 over the U-Net + CLIP encoders + IdentityNet, conv_shortcut kept
            // dense (sc-2769 / sc-3329). Read by the worker capability advertisement (sc-3723).
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: false,
            requires_sigma_shift: false,
            // Wired onto the shared `Residency` seam (epic 10834); honors Sequential offload (F-176).
            supports_sequential_offload: true,
        },
    }
}

/// A loaded SDXL generator: the dual CLIP encoders + tokenizer, the U-Net, the VAE, the
/// Euler-Ancestral sampler (production default), and the `alphas_cumprod` schedule the few-step
/// acceleration samplers (LCM/Lightning/Hyper) build on — assembled from a snapshot directory.
pub struct Sdxl {
    descriptor: ModelDescriptor,
    tokenizer: ClipBpeTokenizer,
    sampler: EulerSampler,
    /// DDPM `alphas_cumprod` from the SDXL `scaled_linear` betas — shared by the acceleration
    /// samplers (sc-2769). Built once at load (the ancestral `sampler` keeps its own σ table).
    alpha_schedule: AlphaSchedule,
    /// Number of loaded ControlNet branches (`spec.control` + `spec.extra_controls`), computed at
    /// load so `generate` can reject a `Control`-count mismatch under either residency (the
    /// `Sequential` seam holds loader closures, not the spec, so this is captured up front).
    control_count: usize,
    /// Component-residency strategy (epic 10834 Phase 1, sc-10839; hoisted to the shared seam in
    /// sc-11125), selected from [`LoadSpec::offload_policy`] at [`load`]. `Resident` (default) holds
    /// every heavy component (both CLIP encoders + U-Net + control/IP/VAE/PiD) warm for the whole job
    /// and across jobs; `Sequential` holds only the per-phase loader closures and re-loads each
    /// component in phase order per generation (text encode → **drop the encoders** → U-Net/VAE
    /// denoise+decode), bounding peak unified memory to the largest single working set instead of the
    /// sum, at the cost of the warm cache — for Macs where the resident set would OOM. The
    /// [`Residency`] seam owns the eval/drop/clear discipline, the stage-boundary cancel checks, and
    /// the error-safe cache flush once for all providers.
    residency: Residency<(ClipTextEncoder, ClipTextEncoder), SdxlHeavyOwned>,
}

/// The heavy render-phase components (everything but the text encoders): the U-Net, its ControlNet
/// branches / IP-Adapter, the VAE, and the optional PiD decoder. Owned by the `Resident` components
/// (held for the whole job) or by a `Sequential` generate (loaded after the encoders are dropped,
/// freed when the job ends).
pub(crate) struct SdxlHeavyOwned {
    unet: UNet2DConditionModel,
    /// ControlNet branches (sc-3058; MultiControlNet sc-3378), loaded from `LoadSpec::control` +
    /// `LoadSpec::extra_controls`. Empty when no control checkpoint was supplied. `generate` requires
    /// exactly one `Control` conditioning per loaded branch (paired by order); their residuals are
    /// summed (the diffusers `MultiControlNetModel` rule).
    controls: Vec<ControlNet>,
    /// Optional IP-Adapter image-token source (sc-3059), loaded from `LoadSpec::ip_adapter`. When
    /// present, the model is in "IP mode": a `Reference` conditioning is the image prompt (txt2img +
    /// IP), not an img2img init. The decoupled-attn K/V projections are installed into `unet`.
    ip_adapter: Option<IpImageEncoder>,
    vae: Autoencoder,
    /// Optional PiD super-resolving decoder overlay (epic 7840, sc-7848): loaded when the spec
    /// carries [`LoadSpec::pid`]. `Some` ⇒ a `req.use_pid` generation decodes the final SDXL latent
    /// through the `sdxl` PiD student (4× SR) instead of the VAE. `None` ⇒ the default byte-exact
    /// VAE decode.
    pid: Option<PidEngine>,
}

/// A borrow of the heavy render-phase components, so the denoise/decode body is written once and
/// runs identically whether the components are held resident (borrowed out of [`ResidentComponents`])
/// or were just loaded by the `Sequential` path — mirrors candle's `DitRef` (sc-10769). Cheap refs.
struct SdxlHeavy<'a> {
    unet: &'a UNet2DConditionModel,
    controls: &'a [ControlNet],
    ip_adapter: Option<&'a IpImageEncoder>,
    vae: &'a Autoencoder,
    pid: Option<&'a PidEngine>,
}

impl SdxlHeavyOwned {
    fn as_ref(&self) -> SdxlHeavy<'_> {
        SdxlHeavy {
            unet: &self.unet,
            controls: &self.controls,
            ip_adapter: self.ip_adapter.as_ref(),
            vae: &self.vae,
            pid: self.pid.as_ref(),
        }
    }
}

/// Construct an [`Sdxl`] from a [`LoadSpec`].
///
/// `spec.weights` must be a [`WeightsSource::Dir`] pointing at a
/// `stabilityai/stable-diffusion-xl-base-1.0` snapshot (the diffusers multi-component tree —
/// `tokenizer/`, `tokenizer_2/`, `text_encoder/`, `text_encoder_2/`, `unet/`, `vae/`).
///
/// **Dtype:** the U-Net + both CLIP text encoders run **fp16**, matching the production reference
/// (`StableDiffusionXL(float16=True)`); the **VAE stays f32** (the vendored always loads the
/// autoencoder f32 — the SDXL VAE is fp16-unstable). The whole fp16 path is byte-identical to the
/// reference on MLX 0.31.2 (sc-2721; needs sc-2772's NAX 16-bit fix + the compiled `gelu_exact`).
/// The lower-level `load_unet`/`load_text_encoder_*` keep an f32 path for the tight stage gates.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        // `Precision::Bf16` is the registry's dense sentinel; the dense path runs fp16 (the
        // production dtype). A non-default precision flag is rejected rather than silently ignored.
        return Err(Error::Msg(
            "sdxl: precision override is not wired; the dense path runs fp16 (the production \
             reference dtype) — drop the precision override"
                .into(),
        ));
    }
    // Resolve the snapshot dir up front — a fail-fast for BOTH residencies (Sequential defers the
    // heavy component build to each generate, but a single-file source is still wrong, so reject it
    // here rather than at the first generate).
    let root = resolve_root(spec)?;

    let cfg = DiffusionConfig::sdxl_base();
    let alpha_schedule =
        AlphaSchedule::scaled_linear(cfg.num_train_steps, cfg.beta_start, cfg.beta_end);
    // Component residency (epic 10834 Phase 1, sc-10839): the default `Resident` builds every heavy
    // component now and holds it warm; `Sequential` keeps only the spec and re-loads per generate in
    // phase order (encode → drop encoders → denoise/decode) to bound peak memory. The `Resident`
    // build is byte-identical to the pre-sc-10839 `load` — the same loaders, adapter/quant order,
    // and PiD overlay, just assembled through the shared per-phase helpers.
    let control_count = spec.control.is_some() as usize + spec.extra_controls.len();
    // F-181: Sequential + a load-time quant over a dense snapshot re-quantizes every generate. An
    // already-packed turnkey loads packed (no re-quant); `Resident` quantizes once. So warn only for
    // the Sequential-over-dense combination that actually pays the repeated cost.
    if let Some(q) = spec.quantize {
        if matches!(spec.offload_policy, OffloadPolicy::Sequential)
            && loader::needs_load_time_quant(root, q.bits(), descriptor().id)?
        {
            mlx_gen::residency::warn_sequential_requantize(descriptor().id, q.bits());
        }
    }
    let residency = build_residency(spec)?;
    Ok(Box::new(Sdxl {
        descriptor: descriptor(),
        tokenizer: loader::load_tokenizer(root)?,
        sampler: EulerSampler::new_with_dtype(&cfg, true, DTYPE)?,
        alpha_schedule,
        control_count,
        residency,
    }))
}

/// The policy→[`Residency`] dispatch, routed through the single [`Residency::from_policy`] seam
/// (sc-10839; hoisted to the shared seam in sc-11126, F-180) so the `match offload_policy` lives in
/// exactly one place. `Resident` eager-loads the dual CLIP text encoders + heavy bundle now (the heavy
/// loader with `use_pid = true`, loading any PiD overlay once and reusing it); `Sequential` captures
/// the two per-phase loaders and loads nothing now, deferring each to [`Residency::run`]. Both use the
/// same [`load_text_encoders`] / [`load_heavy`], so the `Resident` composition is byte-identical to the
/// pre-seam one. The deferral is weight-free-testable: under `Sequential` this touches no component
/// weights, so a dispatch that ignored `offload_policy` would eager-load and fail the "Sequential
/// defers" unit test.
pub(crate) fn build_residency(
    spec: &LoadSpec,
) -> Result<Residency<(ClipTextEncoder, ClipTextEncoder), SdxlHeavyOwned>> {
    let spec_text = spec.clone();
    let spec_heavy = spec.clone();
    Residency::from_policy(
        spec.offload_policy,
        move || load_text_encoders(resolve_root(&spec_text)?, spec_text.quantize),
        move |use_pid| load_heavy(&spec_heavy, resolve_root(&spec_heavy)?, use_pid),
    )
}

/// Resolve the snapshot directory from the load spec, rejecting a single-file source (SDXL needs the
/// diffusers multi-component tree). Shared by [`load`] and the `Sequential` per-phase loaders.
fn resolve_root(spec: &LoadSpec) -> Result<&Path> {
    match &spec.weights {
        WeightsSource::Dir(p) => Ok(p),
        WeightsSource::File(_) => Err(Error::Msg(
            "sdxl expects a snapshot directory (tokenizer/ text_encoder/ unet/ vae/ …), not a \
             single .safetensors file"
                .into(),
        )),
    }
}

/// Load the dual CLIP text encoders (the phase-A component under `Sequential`, dropped before the
/// U-Net loads). Factored out of [`load`] so the `Resident` and `Sequential` paths build byte-
/// identical encoders: both run fp16 with the same optional Q4/Q8 (`group_size 64`) over every
/// quantizable Linear + the token Embedding, matching the sc-2604/sc-1975 quant scope.
fn load_text_encoders(
    root: &Path,
    quant: Option<Quant>,
) -> Result<(ClipTextEncoder, ClipTextEncoder)> {
    let mut te1 = loader::load_text_encoder_1_dtype(root, DTYPE)?;
    let mut te2 = loader::load_text_encoder_2_dtype(root, DTYPE)?;
    if let Some(q) = quant {
        let bits = q.bits();
        te1.quantize(bits)?;
        te2.quantize(bits)?;
    }
    Ok((te1, te2))
}

/// Load the heavy render-phase components — U-Net (+ LoRA/LoKr merge + IP-Adapter install + Q4/Q8),
/// ControlNet branches (+ Q4/Q8), VAE (f32), and the optional PiD overlay — everything but the text
/// encoders. Factored out of [`load`] so the `Sequential` path can load these AFTER the encoders are
/// dropped (bounding peak to `max(encoders, U-Net+VAE)`), and the `Resident` path builds the same
/// bundle up front. The operation order matches the pre-sc-10839 `load` (adapter merge before quant),
/// and the components are independent of the text encoders, so both residencies are byte-identical.
fn load_heavy(spec: &LoadSpec, root: &Path, load_pid: bool) -> Result<SdxlHeavyOwned> {
    let mut unet = loader::load_unet_dtype(root, DTYPE)?;
    if !spec.adapters.is_empty() {
        // Merge LoRA (kohya `lora_unet_` / PEFT, sc-2639) and LoKr (sc-2640) into the dense fp16
        // U-Net weights at load — the production reference merges into the `float16=True` U-Net too,
        // and merging (not a
        // forward-time residual) keeps the chaos-sensitive ancestral sampler bit-exact. Out-of-surface
        // keys (mid_block/ff/conv) are surfaced in the report, not dropped.
        //
        // Coverage (sc-2671): default to the strictly-more-correct COMPLETE surface — mid_block +
        // the GEGLU FF the vendored `lora.py` silently drops — so SDXL LoRAs apply in full, matching
        // diffusers (Michael's correctness-over-parity call, 2026-06-03). `SDXL_LORA_VENDORED` is the
        // escape hatch back to the legacy 515-module surface for byte-parity with the retired Python
        // path.
        let coverage = if std::env::var_os("SDXL_LORA_VENDORED").is_some() {
            eprintln!(
                "sdxl: SDXL_LORA_VENDORED set — restricting LoRA to the legacy vendored 515-module \
                 surface (mid_block + ff dropped; byte-parity with the retired Python path)"
            );
            crate::adapters::LoraCoverage::Vendored
        } else {
            crate::adapters::LoraCoverage::Complete
        };
        crate::adapters::apply_sdxl_adapters_with(&mut unet, &spec.adapters, coverage)?;
    }
    let vae = loader::load_vae(root)?; // VAE always f32 (vendored loads the autoencoder float16=False)

    // ControlNet branches (sc-3058; MultiControlNet sc-3378) — `spec.control` first, then each
    // `spec.extra_controls`, all at the U-Net dtype (fp16). Quantized with the U-Net below when
    // `spec.quantize` is set (the encoder-copy Linears; conv stem / cond-embedding / zero-convs stay
    // dense, matching the U-Net scope).
    let mut controls: Vec<ControlNet> = Vec::new();
    if let Some(src) = &spec.control {
        controls.push(loader::load_controlnet(src, DTYPE)?);
    }
    for src in &spec.extra_controls {
        controls.push(loader::load_controlnet(src, DTYPE)?);
    }

    // Optional IP-Adapter (sc-3059) — install the decoupled-attn K/V pairs into the still-mutable,
    // pre-quant U-Net (so they quantize with it) and keep the image-token encoder.
    let ip_adapter = match &spec.ip_adapter {
        Some(WeightsSource::Dir(p)) => {
            let (enc, pairs) = loader::load_ip_adapter(p, DTYPE)?;
            unet.install_ip_adapter(pairs)?;
            Some(enc)
        }
        Some(WeightsSource::File(_)) => {
            return Err(Error::Msg(
                "sdxl ip_adapter expects an h94/IP-Adapter snapshot directory, not a single file"
                    .into(),
            ));
        }
        None => None,
    };

    if let Some(q) = spec.quantize {
        // Q4/Q8 (group_size 64) over every quantizable Linear of the U-Net + control branches —
        // applied AFTER the adapter merge (the merge needs the dense weight; `merge_dense_delta`
        // errors on a quantized base, matching the fork's "LoRA merged pre-quantization"). The core
        // `AdaptableLinear::quantize` casts each weight to bf16 before packing (sc-2604): SDXL ships
        // fp16/fp32 on disk, and quantizing the as-loaded dtype would give drifted group scales — the
        // sc-1975 "Q8 broken on base-1.0". Convs / norms / token & position embeddings stay dense
        // (gather lookups, not matmuls). The text encoders are quantized in [`load_text_encoders`];
        // the **VAE stays f32** — its only Linears are the tiny quant/post-quant projections
        // (negligible memory), and a dense decode preserves output quality. Scope verified
        // empirically by the full `load(Q).generate()` gate (sc-2641).
        let bits = q.bits();
        unet.quantize(bits)?;
        for cn in &mut controls {
            cn.quantize(bits)?;
        }
    }

    // PiD decoder overlay (epic 7840, sc-7848): load the `sdxl` student + Gemma caption encoder once
    // when the spec carries it AND this generate uses it (`load_pid`, F-177) — Resident passes `true`
    // (loaded once, reused), Sequential passes `req.use_pid` so a non-PiD generate skips the student +
    // its Gemma caption encoder entirely. Shared across the whole SDXL family (sdxl/realvisxl) — and
    // Kolors, which loads its own engine via `mlx_gen_sdxl::model::PID_BACKBONE`.
    let pid = if load_pid {
        spec.pid
            .as_ref()
            .map(|p| PidEngine::from_spec(p, PID_BACKBONE))
            .transpose()?
    } else {
        None
    };

    Ok(SdxlHeavyOwned {
        unet,
        controls,
        ip_adapter,
        vae,
        pid,
    })
}

mlx_gen::impl_generator!(Sdxl {
    validate: |s, req| validate_request(&s.descriptor.capabilities, req),
    generate: generate_impl,
});

impl Sdxl {
    /// The rich-`Result` body behind [`Generator::generate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the family
    /// helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    ///
    /// The staged residency lifecycle (text encode → drop the CLIP encoders under `Sequential` → load
    /// the U-Net/control/IP/VAE/PiD → denoise/decode → free the heavy bundle) is driven by the shared
    /// [`Residency::run`] seam (sc-11125), which owns the eval/drop/clear discipline, the
    /// stage-boundary cancel checks, and the error-safe cache flush.
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;

        let sampler_name = req.sampler.as_deref().unwrap_or("euler_ancestral");
        let is_accel = ACCEL_SAMPLERS.contains(&sampler_name);
        // Curated unified path (epic 7114, sc-7121): a curated solver name (other than the bespoke
        // `euler_ancestral` default + the accel profiles) OR a non-`discrete` scheduler routes to the
        // additive k-diffusion `denoise_curated` over `DiscreteModelSampling`. The ancestral default
        // with no curated knob stays on the bespoke vendored loop, byte-exact (the N1 default gate).
        let scheduler_curated = req
            .scheduler
            .as_deref()
            .and_then(Scheduler::from_name)
            .is_some();
        let sampler_curated = Solver::from_name(sampler_name).is_some()
            && !is_accel
            && sampler_name != "euler_ancestral";
        let use_curated = !is_accel && (sampler_curated || scheduler_curated);
        // F-082: the accel samplers build their own distilled few-step schedule, so a request pairing
        // one with a curated σ scheduler used to validate and then silently drop the scheduler.
        // Reject the combination instead of misreporting the request as honored.
        if is_accel && scheduler_curated {
            return Err(Error::Msg(format!(
                "sdxl: the {sampler_name:?} acceleration sampler uses its own distilled schedule \
                 and cannot honor the {:?} scheduler — drop `scheduler` (or pick a curated sampler)",
                req.scheduler.as_deref().unwrap_or_default()
            )));
        }
        // Per-variant defaults for the few-step samplers; the production defaults otherwise.
        let (def_steps, def_cfg, eta) = if is_accel {
            accel_defaults(sampler_name)
        } else {
            (DEFAULT_STEPS, DEFAULT_GUIDANCE, 0.0)
        };
        let steps = req.steps.unwrap_or(def_steps) as usize;
        let cfg = req.guidance.unwrap_or(def_cfg);
        let cfg_on = cfg > 1.0;
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let reference = self.resolve_reference(req)?;
        let mask_img = self.resolve_mask(req)?;
        let max_time = self.sampler.max_time();

        // Acceleration variants are txt2img-only in v1 (epic 2755 "Image-only v1"); reject an init
        // image rather than silently ignoring it.
        if is_accel && reference.is_some() {
            return Err(Error::Msg(format!(
                "sdxl: the {sampler_name:?} acceleration sampler is txt2img-only (no img2img \
                 reference) in this build"
            )));
        }
        // Inpaint (Mask) rides the ancestral img2img path and needs an init image to blend against.
        if mask_img.is_some() {
            if is_accel {
                return Err(Error::Msg(
                    "sdxl: inpaint masks are not supported with the acceleration samplers".into(),
                ));
            }
            if use_curated {
                return Err(Error::Msg(
                    "sdxl: curated samplers/schedulers are not supported with an inpaint Mask (its \
                     per-step blend has no post-step hook in the callback Sampler) — use the default \
                     euler_ancestral"
                        .into(),
                ));
            }
            if reference.is_none() {
                return Err(Error::Msg(
                    "sdxl: inpaint requires an init image (a Reference) alongside the Mask".into(),
                ));
            }
        }
        // ControlNet (sc-3058; MultiControlNet sc-3378): each `Control` conditioning pairs, in order,
        // with a loaded control branch (`spec.control` + `spec.extra_controls`); their residuals are
        // summed. Needs the ancestral path; not combined with an inpaint mask in this build.
        let control_reqs = self.resolve_control(req)?;
        if !control_reqs.is_empty() {
            if is_accel {
                return Err(Error::Msg(
                    "sdxl: ControlNet is not supported with the acceleration samplers".into(),
                ));
            }
            if control_reqs.len() != self.control_count {
                return Err(Error::Msg(format!(
                    "sdxl: {} Control conditioning(s) passed but the model was loaded with {} control \
                     checkpoint(s) (set LoadSpec::control + extra_controls, one per Control, in order)",
                    control_reqs.len(),
                    self.control_count
                )));
            }
            if mask_img.is_some() {
                return Err(Error::Msg(
                    "sdxl: combining a ControlNet (Control) with an inpaint Mask is not supported"
                        .into(),
                ));
            }
        }

        // ── Phase A: text encode (epic 10834 Phase 1, sc-10839; sc-11125). Seed-independent (no RNG)
        // — tokenizing above the residency lifecycle, then hoisting the encode above the control/IP
        // builds and the per-image loop, is byte-identical to the F-068 order. Under `Sequential` the
        // shared seam LOADS the dual CLIP encoders, encodes, materializes, then DROPS them +
        // `clear_cache()` so their ~1 GB frees before the U-Net/control/IP bundle loads below —
        // bounding peak to `max(encoders, U-Net+VAE)`. Under `Resident` it borrows the warm encoders
        // (byte-identical to the pre-sc-10839 `encode_conditioning`).
        let tokens = self
            .tokenizer
            .tokenize_batch(&req.prompt, if cfg_on { Some(negative) } else { None })?;

        self.residency.run(
            &req.cancel,
            req.use_pid,
            on_progress,
            |text: &(ClipTextEncoder, ClipTextEncoder)| {
                encode_conditioning(&text.0, &text.1, &tokens)
            },
            // Materialize the conditioning + pooled while the encoders are still alive (Sequential
            // only) — MLX is lazy, so an un-evaluated output keeps the encoders referenced through the
            // graph and the drop would free nothing (cf. Wan's `encode_text_staged`).
            |enc| Ok(mlx_rs::transforms::eval([&enc.0, &enc.1])?),
            // ── Establish the heavy render components (U-Net + control/IP + VAE + PiD) and run the
            // denoise/decode body once against the `heavy` borrow — identical for both residencies.
            // `on_progress` is threaded through the seam (F-179) and shadows the outer sink here.
            |heavy_owned, enc, on_progress| {
        let heavy = heavy_owned.as_ref();
        let (conditioning, pooled) = enc;

        // Build the ControlNet contexts once (seed-independent): preprocess each control image to
        // [0,1] NHWC and CFG-batch it to match the U-Net input, paired by order with a loaded branch.
        let mut control_ctxs: Vec<ControlContext> = Vec::with_capacity(control_reqs.len());
        for ((image, scale), cn) in control_reqs.iter().zip(heavy.controls) {
            let img = preprocess_control_image(image, req.width, req.height)?;
            let img = if cfg_on {
                concatenate_axis(&[&img, &img], 0)?
            } else {
                img
            };
            control_ctxs.push(ControlContext {
                controlnet: cn,
                // Precompute the step-invariant conditioning embedding once per run (F-069).
                cond_embed: cn.embed_cond(&img)?,
                scale: *scale,
            });
        }

        // IP-Adapter (sc-3059): when the model carries IP weights and a Reference is present (no
        // mask/control/accel), the Reference is the image prompt (txt2img + IP), NOT an img2img init.
        // The IP scale rides the Reference `strength` field (default 0.6). Tokens are seed-independent
        // → built once, CFG-batched with a zeros uncond row so the negative pass gets no IP signal.
        let ip_mode = heavy.ip_adapter.is_some()
            && reference.is_some()
            && mask_img.is_none()
            && control_reqs.is_empty()
            && !is_accel;
        let ip_scale = reference.and_then(|(_, s)| s).unwrap_or(IP_DEFAULT_SCALE);
        let ip_tokens = if ip_mode {
            let enc = heavy.ip_adapter.expect("ip_adapter present in ip_mode");
            let (image, _) = reference.expect("reference present in ip_mode");
            let tokens = enc.tokens(image)?;
            Some(if cfg_on {
                let zeros = enc.zeros_like_tokens(tokens.dtype())?;
                concatenate_axis(&[&tokens, &zeros], 0)?
            } else {
                tokens
            })
        } else {
            None
        };

        let time_ids = text_time_ids(pooled.shape()[0]);
        let latent_shape = [1, (req.height / 8) as i32, (req.width / 8) as i32, 4];
        // img2img/inpaint init latents (the f32 VAE encode) and the inpaint mask are seed-independent
        // too (F-068). `init_latents` is Some exactly for the ancestral img2img/inpaint paths — a
        // Reference that is neither an accel run nor an IP image prompt; `mask_latent` adds the mask.
        let init_latents = match reference {
            Some((image, _)) if !is_accel && !ip_mode => Some(encode_init_latents(
                heavy.vae, image, req.width, req.height,
            )?),
            _ => None,
        };
        let mask_latent = match mask_img {
            Some(mask) if init_latents.is_some() => {
                Some(preprocess_mask(mask, req.width, req.height)?)
            }
            _ => None,
        };

        // PiD decode overlay (epic 7840, sc-7848) + `from_ldm` early-stop (sc-8049). SDXL is the lone
        // **variance-preserving** PiD student. Its two denoise paths keep the latent in DIFFERENT frames,
        // so the from_ldm capture is handled per-path:
        //   • ancestral (the default; txt2img / img2img / control / IP via `denoise_core`) stores the
        //     latent ALREADY renormalized to the VP frame `(x0+σε)/√(σ²+1)` = `√(1−σ_vp²)·x0 + σ_vp·ε`
        //     at every node (see `EulerSampler::step`/`add_noise`), so a truncated x_k is handed to PiD
        //     as-is (no rescale);
        //   • curated (opt-in k-diffusion) stores RAW VE latents `x0+σ·ε`, so a truncated x_k is mapped
        //     into the VP frame by the plan's rescale (`1/√(1+σ²)`) before decode.
        // Both stay 0.13025-normalized throughout (the loop runs in the scaled latent space `vae.decode`
        // consumes), so no extra normalization is applied. The plan is image-independent (the schedule is
        // fixed by steps / scheduler / strength), so resolve it once here and mint the decoder at the
        // achieved degrade σ. The clean σ=0 decode (`vp_plan = None`) is byte-identical to before. The
        // few-step accel path (decode-bound → no from_ldm benefit, sc-7993) and the inpaint mask-blend
        // (needs the full schedule to σ=0) keep the clean decode; a from_ldm request on either errors
        // loudly rather than silently dropping the knob.
        let vp_plan: Option<VpCapturePlan> = if req.use_pid && req.pid_capture_sigma.is_some() {
            if is_accel || mask_latent.is_some() {
                return Err(Error::Msg(format!(
                    "{}: pid_capture_sigma (from_ldm early-stop) is not supported on the SDXL {} path \
                     (it keeps the clean σ=0 decode); use the standard ancestral or curated denoise for \
                     from_ldm (sc-8049)",
                    self.descriptor.id,
                    if is_accel {
                        "few-step accel"
                    } else {
                        "inpaint mask-blend"
                    }
                )));
            }
            // Resolve the VP capture against the EXACT σ schedule this run will denoise, so `keep` and the
            // achieved degrade σ agree with the truncated trajectory. (This mirrors the per-path schedule
            // build in the count loop below — deterministic host math, no RNG, so it does not perturb the
            // ancestral noise stream.)
            let edm_sigmas = if use_curated {
                let ms = DiscreteModelSampling::sdxl(&self.alpha_schedule);
                let sched = req
                    .scheduler
                    .as_deref()
                    .and_then(Scheduler::from_name)
                    .unwrap_or(Scheduler::Normal);
                let full_sigmas = schedule_sigmas(sched, &ms, steps);
                if init_latents.is_some() {
                    let strength = reference
                        .and_then(|(_, s)| s)
                        .unwrap_or(DEFAULT_STRENGTH)
                        .clamp(0.0, 1.0);
                    let eff = (steps as f32 * strength) as usize;
                    let run_start = full_sigmas.len().saturating_sub(1).saturating_sub(eff);
                    full_sigmas[run_start..].to_vec()
                } else {
                    full_sigmas
                }
            } else {
                let (eff, start_time) = if init_latents.is_some() {
                    let strength = reference
                        .and_then(|(_, s)| s)
                        .unwrap_or(DEFAULT_STRENGTH)
                        .clamp(0.0, 1.0);
                    ((steps as f32 * strength) as usize, max_time * strength)
                } else {
                    (steps, max_time)
                };
                AncestralEuler::new(&self.sampler, eff, start_time)?.edm_sigmas()
            };
            vp_capture_plan(&edm_sigmas, req.pid_capture_sigma)
        } else {
            None
        };
        let capture_sigma = vp_plan.map(|p| p.sigma).unwrap_or(0.0);
        let pid_decoder = resolve_pid_decoder_at_sigma(
            heavy.pid,
            req,
            base_seed,
            self.descriptor.id,
            capture_sigma,
        )?;
        let pid_ref = pid_decoder.as_ref().map(|d| d as &dyn LatentDecoder);

        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            // One image per iteration (the vendored `_run_one`, n_images=1), each with its own seed.
            let seed = base_seed.wrapping_add(i as u64);
            // Seed the global RNG up front; the hoisted conditioning/VAE encodes drew no RNG, so the
            // first draw here is the init noise (the prior / img2img add_noise) — matching the
            // reference stream.
            mlx_rs::random::seed(seed)?;

            // Curated unified-sampler path (epic 7114, sc-7121): k-diffusion VE-σ sampling over a
            // `DiscreteModelSampling`, additive alongside the bespoke ancestral default. The latents
            // live in raw σ-space; the curated solver + scheduler are selected per request. Supports
            // txt2img / img2img / ControlNet / IP-Adapter (inpaint is guarded out above).
            if use_curated {
                let ms = DiscreteModelSampling::sdxl(&self.alpha_schedule);
                let sched = req
                    .scheduler
                    .as_deref()
                    .and_then(Scheduler::from_name)
                    .unwrap_or(Scheduler::Normal);
                let full_sigmas = schedule_sigmas(sched, &ms, steps);
                let noise = mlx_rs::random::normal::<f32>(&latent_shape, None, None, None)?;
                // Raw k-diffusion σ-space init: txt2img `ε·σ_max`; img2img runs the strength-tail of the
                // schedule, seeded `x₀ + ε·σ_start` (diffusers EulerDiscrete add_noise). A strength that
                // rounds to 0 effective steps leaves the schedule at `[0.0]` → the init is returned.
                let (run_sigmas, init) = if let Some(x_0) = &init_latents {
                    let strength = reference
                        .and_then(|(_, s)| s)
                        .unwrap_or(DEFAULT_STRENGTH)
                        .clamp(0.0, 1.0);
                    let eff = (steps as f32 * strength) as usize;
                    let run_start = full_sigmas.len().saturating_sub(1).saturating_sub(eff);
                    let rs = full_sigmas[run_start..].to_vec();
                    let init = add(x_0, &multiply(&noise, scalar(rs[0]))?)?;
                    (rs, init)
                } else {
                    let init = multiply(&noise, scalar(full_sigmas[0]))?;
                    (full_sigmas, init)
                };
                // PiD from_ldm early-stop (sc-8049): truncate the curated k-diffusion schedule to the
                // VP-capture `keep` nodes so the solver stops at the achieved degrade σ; the clean path
                // (`vp_plan = None`) runs the full schedule byte-identically.
                let keep_sigmas: &[f32] = match &vp_plan {
                    Some(p) => &run_sigmas[..p.keep],
                    None => &run_sigmas,
                };
                let ip = ip_tokens.as_ref().map(|t| (t, ip_scale));
                // CFG++ (sc-8256): opt-in via `guidance_method == "cfg_pp"`, only with a CFG++-compatible
                // base solver (euler/ddim/dpmpp_2m) and an active guidance gap (`cfg > 1`). Anything else
                // — including `cfg_pp` on an incompatible sampler — falls back to the plain curated path
                // (N3, never a hard-fail), so the default is byte-untouched.
                let want_cfgpp = req.guidance_method.as_deref() == Some("cfg_pp")
                    && cfg > 1.0
                    && Solver::from_name(sampler_name)
                        .is_some_and(mlx_gen::gen_core::sampling::base_supports_cfgpp);
                let latents = if want_cfgpp {
                    denoise_cfgpp(
                        heavy.unet,
                        Some(sampler_name),
                        &ms,
                        keep_sigmas,
                        init,
                        &conditioning,
                        &pooled,
                        &time_ids,
                        cfg,
                        &req.cancel,
                        on_progress,
                        &control_ctxs,
                        ip,
                        None,
                    )?
                } else {
                    denoise_curated(
                        heavy.unet,
                        Some(sampler_name),
                        &ms,
                        keep_sigmas,
                        init,
                        &conditioning,
                        &pooled,
                        &time_ids,
                        cfg,
                        seed,
                        &req.cancel,
                        on_progress,
                        &control_ctxs,
                        ip,
                        None,
                    )?
                };
                // Curated latents live in RAW VE σ-space (`x0+σ·ε`); an early-stop leaves x_k at σ>0, so
                // map it into the student's VP frame with the plan's rescale (`1/√(1+σ²)`) before decode.
                // The clean path (`vp_plan = None`) leaves it byte-identical. (sc-8049)
                let latents = match &vp_plan {
                    Some(p) => multiply(&latents, scalar(p.rescale))?,
                    None => latents,
                };
                on_progress(Progress::Decoding);
                images.push(decode_image(heavy.vae, &latents, pid_ref)?);
                continue;
            }

            // Build the run's sampler + its seeded init latents. The denoise loop is driven entirely
            // by the sampler's own schedule (`sampler.num_steps()`), so the trait owns the per-step
            // timestep, the input scaling, and the step math.
            let (latents, sampler, blend): (
                mlx_rs::Array,
                Box<dyn DiffusionSampler + '_>,
                Option<InpaintBlend>,
            ) = if is_accel {
                // Few-step acceleration (txt2img): unit-noise prior scaled into the sampler's space.
                let s = self.build_accel_sampler(sampler_name, steps, eta, seed);
                let noise = mlx_rs::random::normal::<f32>(&latent_shape, None, None, None)?;
                let lat = s.scale_initial_noise(&noise)?;
                (lat, s, None)
            } else if let (Some(x_0), Some(mask_latent)) = (&init_latents, &mask_latent) {
                // Masked inpaint (sc-3057): same ancestral img2img start, but keep the FIXED prior
                // noise so the per-step blend can pin the black (keep) region to the init noised to
                // each step's σ. Default strength 0.85 (the worker's inpaint default).
                let strength = reference
                    .and_then(|(_, s)| s)
                    .unwrap_or(INPAINT_DEFAULT_STRENGTH)
                    .clamp(0.0, 1.0);
                let start_step = max_time * strength;
                let noise = mlx_rs::random::normal::<f32>(&latent_shape, None, None, None)?;
                let x_t = self.sampler.add_noise_with(x_0, &noise, start_step)?;
                let eff = (steps as f32 * strength) as usize;
                // The kept region is noised to each step's "next" time `t_prev` (schedule[i].1).
                let t_prev: Vec<f32> = self
                    .sampler
                    .timesteps(eff, start_step)?
                    .into_iter()
                    .map(|(_, tp)| tp)
                    .collect();
                let blend = InpaintBlend::new(
                    &self.sampler,
                    mask_latent.clone(),
                    x_0.clone(),
                    noise,
                    t_prev,
                );
                (
                    x_t,
                    Box::new(AncestralEuler::new(&self.sampler, eff, start_step)?),
                    Some(blend),
                )
            } else if let Some(x_0) = &init_latents {
                // img2img (ancestral; the vendored `generate_latents_from_image`): start at
                // `max_time·strength`, run `int(steps·strength)` steps — NO min-1 floor (strength ≤
                // 1/steps ⇒ 0 steps ⇒ init returned unchanged, dodging the σ=0 ancestral `σ_up` 0/0
                // → NaN).
                let strength = reference
                    .and_then(|(_, s)| s)
                    .unwrap_or(DEFAULT_STRENGTH)
                    .clamp(0.0, 1.0);
                let start_step = max_time * strength;
                let x_t = self.sampler.add_noise(x_0, start_step)?;
                let eff = (steps as f32 * strength) as usize;
                // PiD from_ldm early-stop (sc-8049): truncate to the VP-capture `keep` steps; the stored
                // ancestral latent is already the VP frame, so it is handed to PiD as-is. Clean path
                // (`vp_plan = None`) keeps the full schedule.
                let sampler = AncestralEuler::new(&self.sampler, eff, start_step)?;
                (
                    x_t,
                    Box::new(match &vp_plan {
                        Some(p) => sampler.truncate_to(p.keep - 1),
                        None => sampler,
                    }),
                    None,
                )
            } else {
                // txt2img (ancestral): seeded prior.
                let prior = self.sampler.sample_prior(&latent_shape)?;
                // PiD from_ldm early-stop (sc-8049): truncate to the VP-capture `keep` steps (see the
                // img2img arm); clean path (`vp_plan = None`) keeps the full schedule.
                let sampler = AncestralEuler::new(&self.sampler, steps, max_time)?;
                (
                    prior,
                    Box::new(match &vp_plan {
                        Some(p) => sampler.truncate_to(p.keep - 1),
                        None => sampler,
                    }),
                    None,
                )
            };

            let d = Denoiser {
                unet: heavy.unet,
                sampler: sampler.as_ref(),
            };
            let latents = if let Some(tokens) = &ip_tokens {
                denoise_ip(
                    &d,
                    latents,
                    &conditioning,
                    &pooled,
                    &time_ids,
                    cfg,
                    &req.cancel,
                    on_progress,
                    tokens,
                    ip_scale,
                )?
            } else if !control_ctxs.is_empty() {
                denoise_multi_control(
                    &d,
                    latents,
                    &conditioning,
                    &pooled,
                    &time_ids,
                    cfg,
                    &req.cancel,
                    on_progress,
                    &control_ctxs,
                )?
            } else if let Some(b) = &blend {
                denoise_inpaint(
                    &d,
                    latents,
                    &conditioning,
                    &pooled,
                    &time_ids,
                    cfg,
                    &req.cancel,
                    on_progress,
                    b,
                )?
            } else {
                denoise(
                    &d,
                    latents,
                    &conditioning,
                    &pooled,
                    &time_ids,
                    cfg,
                    &req.cancel,
                    on_progress,
                )?
            };

            on_progress(Progress::Decoding);
            images.push(decode_image(heavy.vae, &latents, pid_ref)?);
        }
                Ok(GenerationOutput::Images(images))
            },
        )
    }
}

impl Sdxl {
    /// Build the per-run few-step acceleration sampler (sc-2769). `name` is one of
    /// [`ACCEL_SAMPLERS`]; `steps` is the inference step count (Lightning must match the loaded
    /// LoRA's 2/4/8); `eta` is the TCD stochasticity (Hyper-SD); `seed` is the request seed driving
    /// the deterministic between-step re-noise (D6). The samplers cast the U-Net input to fp16 (the
    /// loaded compute dtype) and run their step math in f32.
    fn build_accel_sampler(
        &self,
        name: &str,
        steps: usize,
        eta: f32,
        seed: u64,
    ) -> Box<dyn DiffusionSampler> {
        let n_train = self.alpha_schedule.alphas_cumprod.len();
        let sched = self.alpha_schedule.clone();
        match name {
            "lcm" => Box::new(LcmSampler::new(
                sched,
                n_train,
                LCM_ORIGINAL_STEPS,
                steps,
                Dtype::Float16,
                seed,
            )),
            "lightning" => Box::new(LightningSampler::new(
                &sched,
                n_train,
                steps,
                Dtype::Float16,
            )),
            "hyper" => Box::new(TcdSampler::new(
                sched,
                n_train,
                LCM_ORIGINAL_STEPS,
                steps,
                eta,
                Dtype::Float16,
                seed,
            )),
            // `generate` only calls this for `name ∈ ACCEL_SAMPLERS`.
            _ => unreachable!("build_accel_sampler: {name:?} is not an acceleration sampler"),
        }
    }

    /// Extract the single img2img init image + its strength from the request's conditioning (the
    /// per-reference strength wins over `req.strength`). SDXL img2img conditions on exactly one init
    /// image, so more than one `Reference` is an error.
    fn resolve_reference<'a>(
        &self,
        req: &'a GenerationRequest,
    ) -> Result<Option<(&'a Image, Option<f32>)>> {
        let mut reference = None;
        for c in &req.conditioning {
            if let Conditioning::Reference { image, strength } = c {
                if reference.is_some() {
                    return Err(Error::Msg(
                        "sdxl: multiple reference images are not supported (single img2img init only)"
                            .into(),
                    ));
                }
                reference = Some((image, strength.or(req.strength)));
            }
        }
        Ok(reference)
    }

    /// Extract the single inpaint mask from the request's conditioning (sc-3057). White = repaint,
    /// black = keep. SDXL supports one mask; more than one is an error.
    fn resolve_mask<'a>(&self, req: &'a GenerationRequest) -> Result<Option<&'a Image>> {
        let mut mask = None;
        for c in &req.conditioning {
            if let Conditioning::Mask { image } = c {
                if mask.is_some() {
                    return Err(Error::Msg(
                        "sdxl: multiple inpaint masks are not supported".into(),
                    ));
                }
                mask = Some(image);
            }
        }
        Ok(mask)
    }

    /// Collect the ControlNet control images + `conditioning_scale`s (sc-3058; MultiControlNet
    /// sc-3378), in request order. Each pairs with a loaded control branch (`spec.control` +
    /// `spec.extra_controls`); the count must match (validated in `generate`). A single `Control` is
    /// the common case; more than one runs as MultiControlNet (residuals summed).
    fn resolve_control<'a>(&self, req: &'a GenerationRequest) -> Result<Vec<(&'a Image, f32)>> {
        let mut controls = Vec::new();
        for c in &req.conditioning {
            if let Conditioning::Control { image, scale, .. } = c {
                // `None` → the diffusers `controlnet_conditioning_scale` default (full strength);
                // `Some(x)` — including `Some(0.0)` for an inert branch — is used verbatim (F-085).
                controls.push((image, scale.unwrap_or(DEFAULT_CONTROLNET_SCALE)));
            }
        }
        Ok(controls)
    }
}

/// Capability-driven request validation, factored out so it can be unit-tested without loaded
/// weights. Rejects unsupported guidance / negative prompt / conditioning / size / count.
pub(crate) fn validate_request(caps: &Capabilities, req: &GenerationRequest) -> Result<()> {
    // Shared capability floor (F-022): count/steps range, size range, negative_prompt/guidance/
    // true_cfg support gating + finiteness, sampler/scheduler/guidance_method membership, and accepted
    // conditioning kinds. Delegating to core (like Kolors, F-132) restores the `true_cfg` and
    // `guidance_method` checks this hand-rolled copy had dropped — a `cfg_pp` typo in `guidance_method`
    // previously slipped through and silently rendered plain CFG. `steps == Some(0)` is now the floor's
    // job too. The `?` keeps the typed `Error::Unsupported` for capability gaps.
    caps.validate_request(MODEL_ID, req)?;

    // SDXL-specific checks layered on top of the shared floor:
    if req.prompt.is_empty() {
        return Err(Error::Msg("sdxl: prompt must not be empty".into()));
    }
    // SDXL works in latent space at /8; both dims must be multiples of 8.
    if !req.width.is_multiple_of(8) || !req.height.is_multiple_of(8) {
        return Err(Error::Msg(format!(
            "sdxl: width/height must be multiples of 8 (got {}x{})",
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

    #[test]
    fn descriptor_is_sdxl() {
        let d = descriptor();
        assert_eq!(d.id, "sdxl");
        assert_eq!(d.family, "sdxl");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
    }

    #[test]
    fn registered_in_core_registry() {
        // Linking this crate must self-register the model (inventory link-time collection).
        assert!(
            mlx_gen::registry::generators().any(|r| (r.descriptor)().id == "sdxl"),
            "sdxl is not registered in mlx_gen's generator registry"
        );
    }

    #[test]
    fn validate_rejects_empty_prompt() {
        let caps = descriptor().capabilities;
        let req = GenerationRequest::default(); // default prompt is empty
        let err = validate_request(&caps, &req).unwrap_err().to_string();
        assert!(err.contains("empty"), "got: {err}");
    }

    #[test]
    fn validate_rejects_explicit_zero_steps() {
        let caps = descriptor().capabilities;
        // F-073: an explicit `steps: Some(0)` would VAE-decode pure scaled noise → reject loudly.
        let zero = GenerationRequest {
            prompt: "a fox".into(),
            steps: Some(0),
            ..Default::default()
        };
        let err = validate_request(&caps, &zero).unwrap_err().to_string();
        assert!(err.contains("steps"), "got: {err}");
        // `steps: None` (use the production default) and an explicit positive count are accepted.
        let unset = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        assert!(validate_request(&caps, &unset).is_ok());
        let one = GenerationRequest {
            prompt: "a fox".into(),
            steps: Some(1),
            ..Default::default()
        };
        assert!(validate_request(&caps, &one).is_ok());
    }

    #[test]
    fn validate_rejects_unadvertised_guidance_method_and_true_cfg() {
        // F-022: the hand-rolled copy dropped the `guidance_method` membership + `true_cfg` gate. A
        // `cfg_pp` typo (e.g. "cfgpp") previously slipped through and silently rendered plain CFG.
        let caps = descriptor().capabilities;
        // Advertised methods are ["cfg", "cfg_pp"]; a typo must be rejected (typed Unsupported).
        let typo = GenerationRequest {
            prompt: "a fox".into(),
            guidance_method: Some("cfgpp".into()),
            ..Default::default()
        };
        let err = validate_request(&caps, &typo).unwrap_err();
        assert!(
            matches!(err, Error::Unsupported(_)),
            "cfg_pp typo should be a typed Unsupported gap, got {err:?}"
        );
        // The correct spelling passes.
        let ok = GenerationRequest {
            prompt: "a fox".into(),
            guidance_method: Some("cfg_pp".into()),
            ..Default::default()
        };
        assert!(validate_request(&caps, &ok).is_ok());
        // SDXL doesn't support true_cfg — a request must be rejected, not ignored.
        let tcfg = GenerationRequest {
            prompt: "a fox".into(),
            true_cfg: Some(4.0),
            ..Default::default()
        };
        assert!(validate_request(&caps, &tcfg).is_err());
    }

    #[test]
    fn validate_accepts_cfg_and_negative_prompt_rejects_bad_size() {
        let caps = descriptor().capabilities;
        // Real CFG + negative prompt are supported.
        let mut req = GenerationRequest {
            prompt: "a fox".into(),
            guidance: Some(7.0),
            negative_prompt: Some("blurry".into()),
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_ok());
        // Non-multiple-of-8 size is rejected.
        req = GenerationRequest {
            prompt: "a fox".into(),
            width: 1020,
            height: 1024,
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_err());
        // Out-of-range size is rejected.
        req = GenerationRequest {
            prompt: "a fox".into(),
            width: 256,
            height: 256,
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_err());
    }

    #[test]
    fn validate_sampler_selection() {
        let caps = descriptor().capabilities;
        let base = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        // The default + every wired sampler is accepted (an unset sampler defaults to ancestral): the
        // accel profiles AND the unified curated solvers (epic 7114, sc-7121).
        assert!(validate_request(&caps, &base).is_ok());
        for ok in [
            "euler_ancestral",
            "lcm",
            "lightning",
            "hyper",
            "euler",
            "heun",
            "dpmpp_2m",
            "dpmpp_sde",
            "uni_pc",
            "ddim",
        ] {
            assert!(
                validate_request(
                    &caps,
                    &GenerationRequest {
                        sampler: Some(ok.into()),
                        ..base.clone()
                    }
                )
                .is_ok(),
                "sampler {ok:?} should be accepted"
            );
        }
        // An unknown sampler is rejected, not silently downgraded.
        for bad in ["plms", "dpm_fast", "nonsense"] {
            let err = validate_request(
                &caps,
                &GenerationRequest {
                    sampler: Some(bad.into()),
                    ..base.clone()
                },
            )
            .unwrap_err()
            .to_string();
            assert!(err.contains("unsupported sampler"), "got: {err}");
        }
    }

    #[test]
    fn validate_scheduler_selection() {
        let caps = descriptor().capabilities;
        let base = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        // `discrete` (the native ancestral schedule) + every curated σ scheduler is accepted (sc-7121).
        for ok in [
            "discrete",
            "normal",
            "simple",
            "karras",
            "exponential",
            "sgm_uniform",
            "beta",
            "ddim_uniform",
        ] {
            assert!(
                validate_request(
                    &caps,
                    &GenerationRequest {
                        scheduler: Some(ok.into()),
                        ..base.clone()
                    }
                )
                .is_ok(),
                "scheduler {ok:?} should be accepted"
            );
        }
        // diffusers timestep-spacing names are NOT curated scheduler names → rejected.
        for bad in ["leading", "trailing", "nonsense"] {
            let err = validate_request(
                &caps,
                &GenerationRequest {
                    scheduler: Some(bad.into()),
                    ..base.clone()
                },
            )
            .unwrap_err()
            .to_string();
            assert!(err.contains("unsupported scheduler"), "got: {err}");
        }
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/sdxl.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }

    // ── F-180 (sc-11126): weight-free, default-run proof that SDXL's dispatch HONORS `offload_policy`.
    // `build_residency` points at a non-existent snapshot *directory* (so the single-file guard in
    // `resolve_root` passes) and the discriminator is deferral:
    //   * `Sequential` captures the two per-phase loaders, touches NO weights → `Ok` + `is_sequential`.
    //   * `Resident` eager-loads the CLIP text encoders from the missing dir → `Err`.
    // A dispatch that ignored `offload_policy` (always `Resident`) would eager-load under a `Sequential`
    // request and fail the first assertion. The `sequential_residency_real_weights.rs` A/B is
    // `#[ignore]`d; this is the default-run guard.
    fn missing_snapshot_spec(policy: OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(
            "/nonexistent/sdxl-residency-test-snapshot".into(),
        ))
        .with_offload_policy(policy)
    }

    #[test]
    fn build_residency_sequential_defers_all_component_loads() {
        let res = build_residency(&missing_snapshot_spec(OffloadPolicy::Sequential))
            .expect("Sequential must defer loads and not touch the (missing) snapshot dir");
        assert!(
            res.is_sequential(),
            "Sequential policy must build a Sequential (deferred) residency"
        );
    }

    #[test]
    fn build_residency_resident_eager_loads_and_fails_on_missing_snapshot() {
        let err = build_residency(&missing_snapshot_spec(OffloadPolicy::Resident))
            .err()
            .expect("Resident must eager-load and fail on a missing snapshot dir");
        let msg = err.to_string();
        assert!(
            !msg.contains("single .safetensors file") && !msg.contains("precision override"),
            "expected an eager-load failure, not the up-front guard: {msg}"
        );
    }
}
