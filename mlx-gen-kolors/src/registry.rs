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

use mlx_rs::transforms::eval;
use mlx_rs::{random, Array, Dtype};

use std::path::Path;

use mlx_gen::gen_core::sampling::{vp_capture_plan, VpCapturePlan};
use mlx_gen::{
    curated_scheduler_names, default_seed, schedule_sigmas, AlphaSchedule, Capabilities,
    Conditioning, ConditioningKind, ControlKind, DiscreteModelSampling, Error, GenerationOutput,
    GenerationRequest, Generator, Image, LatentDecoder, LoadSpec, Modality, ModelDescriptor,
    OffloadPolicy, Progress, Quant, Residency, Result, Scheduler, Solver, WeightsSource,
};

use mlx_gen_pid::{resolve_pid_decoder_at_sigma, PidEngine};
use mlx_gen_sdxl::{
    decode_image, encode_init_latents, load_controlnet, ControlNet, IpImageEncoder, PID_BACKBONE,
};

use crate::ip_adapter::load_kolors_ip_adapter;
use crate::model::{KolorsHeavy, KolorsText, DEFAULT_IMG2IMG_STRENGTH, SPATIAL_SCALE};
use crate::sampler::{KolorsEulerSampler, BETA_END, BETA_START, NUM_TRAIN_TIMESTEPS};

/// Registry id — the SceneWorks worker's `payload.model` for the Kolors family.
pub const MODEL_ID: &str = "kolors";

/// diffusers `KolorsPipeline` production defaults: 50 inference steps, CFG 5.0.
const DEFAULT_STEPS: u32 = 50;
const DEFAULT_GUIDANCE: f32 = 5.0;
/// Default IP-Adapter scale when a request doesn't override it (carried on the `Reference` strength
/// field in IP mode, mirroring the SDXL IP-Adapter convention).
const IP_DEFAULT_SCALE: f32 = 0.6;
/// Default ControlNet `conditioning_scale` for a `Conditioning::Control` that leaves `scale = None`
/// (F-085) — the diffusers full-strength default. An explicit `Some(x)`, including `Some(0.0)` for an
/// inert branch, overrides it.
const DEFAULT_CONTROLNET_SCALE: f32 = 1.0;
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
            supported_guidance_methods: vec![],
            min_size: 512,
            max_size: 2048,
            max_count: 8,
            mac_only: true,
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: false,
            requires_sigma_shift: false,
            // Wired onto the shared `Residency` seam (epic 10834, sc-10840); honors Sequential offload
            // (F-176). The monolithic `Kolors` was split into a droppable `KolorsText` (6B ChatGLM3 +
            // tokenizer) phase and a `KolorsHeavy` (SDXL U-Net + VAE) phase (`crate::model`): under
            // `Sequential` the ChatGLM3 encoder — the dominant footprint — is encoded, materialized,
            // then dropped before the U-Net + VAE (+ ControlNet / IP-Adapter / PiD) load, bounding
            // peak unified memory to `max(ChatGLM3, U-Net+VAE)`. Kolors quantizes the U-Net + ChatGLM3
            // DENSE at load, so a `Sequential` + `quantize` load re-quantizes each generate (F-181
            // advisory in `load`).
            supports_sequential_offload: true,
        },
    }
}

/// A loaded, dispatchable Kolors generator: the component-residency strategy (epic 10834, sc-10840)
/// plus the load-time presence flags the validator reads. Holds ONLY the [`Residency`] (no direct
/// component fields — a retained component would defeat the `Sequential` drop): `Resident` (default)
/// holds the ChatGLM3 encoder + U-Net + VAE (+ ControlNet / IP-Adapter / PiD) warm; `Sequential` holds
/// only the per-phase loader closures and re-loads each per generation in phase order (encode →
/// **drop the ChatGLM3 encoder** → U-Net/VAE/ControlNet/IP/PiD), bounding peak to
/// `max(ChatGLM3, U-Net+VAE)`.
pub struct KolorsGenerator {
    descriptor: ModelDescriptor,
    /// Whether a ControlNet was requested (`spec.control`). Known at load without loading weights, so
    /// the validator's mode-combination guards work even under `Sequential` (the ControlNet itself is
    /// loaded lazily in the heavy phase).
    has_control: bool,
    /// Whether an IP-Adapter was requested (`spec.ip_adapter` is a dir). Same rationale as
    /// [`has_control`](Self::has_control).
    has_ip: bool,
    residency: Residency<KolorsText, KolorsHeavyOwned>,
}

/// The heavy render-phase components owned by a `Resident` build or a `Sequential` generate (mirrors
/// lens's `LensHeavyOwned`): the SDXL U-Net + VAE ([`KolorsHeavy`]) plus the optionally-loaded
/// ControlNet branch, IP-Adapter image-token encoder (its decoupled-attn K/V pairs already installed
/// into the U-Net), and PiD overlay.
pub(crate) struct KolorsHeavyOwned {
    heavy: KolorsHeavy,
    control: Option<ControlNet>,
    ip_encoder: Option<IpImageEncoder>,
    /// Optional PiD super-resolving decoder overlay (epic 7840, sc-7848): loaded when the spec carries
    /// [`LoadSpec::pid`] AND this generate uses it (F-177). Kolors shares the SDXL VAE latent space, so
    /// it reuses the `sdxl` PiD student (via [`PID_BACKBONE`]). `Some` ⇒ a `req.use_pid` generation
    /// decodes the final latent 4× through PiD instead of the VAE; `None` ⇒ the byte-exact VAE decode.
    pid: Option<PidEngine>,
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
    // Fail-fast the single-file source (and a File ip_adapter) up front for BOTH policies. Sequential
    // defers the component build, but these are still wrong regardless of policy.
    resolve_root(spec)?;
    if matches!(&spec.ip_adapter, Some(WeightsSource::File(_))) {
        return Err(Error::Msg(
            "kolors ip_adapter expects a Kolors-IP-Adapter-Plus snapshot directory, not a file"
                .into(),
        ));
    }
    // F-181: Kolors quantizes the U-Net + ChatGLM3 DENSE at load, so a `Sequential` + `quantize` load
    // re-quantizes each generate (repeated compute; the dense transient shrinks the memory win).
    if let Some(q) = spec.quantize {
        if matches!(spec.offload_policy, OffloadPolicy::Sequential) {
            mlx_gen::residency::warn_sequential_requantize(MODEL_ID, q.bits());
        }
    }
    Ok(Box::new(KolorsGenerator {
        descriptor: descriptor(),
        has_control: spec.control.is_some(),
        has_ip: matches!(&spec.ip_adapter, Some(WeightsSource::Dir(_))),
        residency: build_residency(spec)?,
    }))
}

/// Resolve the snapshot dir (rejecting a single-file source). Shared by the entry-point fail-fast and
/// the `Sequential` per-phase loaders.
fn resolve_root(spec: &LoadSpec) -> Result<std::path::PathBuf> {
    match &spec.weights {
        WeightsSource::Dir(p) => Ok(p.clone()),
        WeightsSource::File(_) => Err(Error::Msg(
            "kolors expects a Kolors-diffusers snapshot directory (text_encoder/ tokenizer/ \
             unet/ vae/), not a single .safetensors file"
                .into(),
        )),
    }
}

/// The policy→[`Residency`] dispatch (sc-10840), routed through the single [`Residency::from_policy`]
/// seam (F-180). `Resident` eager-loads the ChatGLM3 text phase + heavy bundle now (the heavy loader
/// with `use_pid = true`, loading any PiD overlay once and reusing it); `Sequential` captures the two
/// per-phase loaders and loads nothing now, deferring each to [`Residency::run`]. Both use the same
/// [`KolorsText::load`] / [`load_heavy_owned`], so the `Resident` composition is byte-identical to the
/// pre-seam whole-model load (independent snapshot subdirs, deterministic RNG-free LoRA-merge → quant
/// → IP-install). Weight-free-testable: under `Sequential` this touches no component weights.
pub(crate) fn build_residency(spec: &LoadSpec) -> Result<Residency<KolorsText, KolorsHeavyOwned>> {
    let dtype = Dtype::Float16;
    let spec_text = spec.clone();
    let spec_heavy = spec.clone();
    Residency::from_policy(
        spec.offload_policy,
        move || {
            let root = resolve_root(&spec_text)?;
            let mut text = KolorsText::load(&root, dtype)?;
            // Q4/Q8 quantizes the 6B ChatGLM3 encoder in place after the dense load (the U-Net
            // quantizes in `load_heavy_owned`). Deterministic, so byte-identical across residencies.
            if let Some(q) = spec_text.quantize {
                text.quantize(q.bits())?;
            }
            Ok(text)
        },
        move |use_pid| {
            let root = resolve_root(&spec_heavy)?;
            load_heavy_owned(&spec_heavy, &root, dtype, use_pid)
        },
    )
}

/// Load the heavy render bundle — the SDXL U-Net + VAE (+ optional ControlNet / IP-Adapter / PiD),
/// everything but the ChatGLM3 encoder. Merges any LoRA/LoKr into the dense U-Net, then quantizes it
/// (the SDXL ordering, sc-4733), then loads the ControlNet + installs the IP-Adapter K/V pairs into
/// the (possibly quantized) U-Net. The PiD student is loaded only when `use_pid` (F-177) — Resident
/// passes `true` (loaded once, reused), Sequential passes `req.use_pid` so a non-PiD generate skips it.
fn load_heavy_owned(
    spec: &LoadSpec,
    root: &Path,
    dtype: Dtype,
    use_pid: bool,
) -> Result<KolorsHeavyOwned> {
    let mut heavy = KolorsHeavy::load(root, dtype)?;
    if !spec.adapters.is_empty() {
        heavy.apply_lora(&spec.adapters)?;
    }
    if let Some(q) = spec.quantize {
        // F-144 (sc-11129): reject a requested-vs-packed tier mismatch before quantizing. `quantize()`
        // silently no-ops on already-packed weights, so a Q4 request over a pre-quantized Q8 snapshot
        // would serve Q8 with no diagnostic. Reuses the SDXL-family marker check — the Kolors U-Net is
        // the SDXL `UNet2DConditionModel` under `unet/`, the representative heavy component.
        mlx_gen_sdxl::loader::needs_load_time_quant(root, q.bits(), MODEL_ID)?;
        heavy.quantize_unet(q.bits())?;
    }

    let control = match &spec.control {
        Some(src) => Some(load_controlnet(src, dtype)?),
        None => None,
    };

    let ip_encoder = match &spec.ip_adapter {
        Some(WeightsSource::Dir(p)) => {
            let (enc, pairs) = load_kolors_ip_adapter(p, dtype)?;
            heavy.install_ip_adapter(pairs)?;
            Some(enc)
        }
        // A File ip_adapter was rejected up front in `load`.
        Some(WeightsSource::File(_)) => None,
        None => None,
    };

    // PiD decoder overlay (epic 7840, sc-7848): load the `sdxl` student + Gemma caption encoder once
    // when the spec carries it AND this generate uses it (Kolors = SDXL VAE latent space).
    let pid = if use_pid {
        spec.pid
            .as_ref()
            .map(|p| PidEngine::from_spec(p, PID_BACKBONE))
            .transpose()?
    } else {
        None
    };

    Ok(KolorsHeavyOwned {
        heavy,
        control,
        ip_encoder,
        pid,
    })
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
        if has_ctrl && !self.has_control {
            return Err(Error::Msg(
                "kolors: a Control conditioning was passed but the model was loaded without a \
                 ControlNet (set LoadSpec::control)"
                    .into(),
            ));
        }
        // Control + Reference is the combined pose tier — allowed ONLY when an IP-Adapter is also
        // loaded (the Reference is the IP identity + img2img init). Plain Control + img2img (a
        // Reference with no IP-Adapter) is not a wired Kolors path.
        if has_ctrl && has_ref && !self.has_ip {
            return Err(Error::Msg(
                "kolors: combining ControlNet (Control) with a Reference requires an IP-Adapter (the \
                 combined pose tier — load LoadSpec::ip_adapter); plain Control + img2img is not \
                 supported in this build"
                    .into(),
            ));
        }
        // A loaded IP-Adapter + Control with no Reference can't run: the combined pass needs the
        // reference as the IP identity (and the IP image prompt is required in IP mode anyway).
        if has_ctrl && self.has_ip && !has_ref {
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
    /// The staged residency lifecycle (ChatGLM3 encode pos+neg → **drop the encoder** under
    /// `Sequential` → load the U-Net/VAE/ControlNet/IP/PiD → per-mode denoise/decode → free the heavy
    /// bundle) is driven by the shared [`Residency::run`] seam (sc-10840). The per-mode denoise
    /// dispatch (curated + the five bespoke assemblies) + the PiD `from_ldm` plan run inside the render
    /// closure, byte-identical for both policies (the only global-RNG draw per image is the noise,
    /// after `random::seed(seed)` — hoisting the seed-independent encode/VAE-init out cannot perturb it).
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
        let ip_mode = self.has_ip;

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
        // `KolorsHeavy::denoise_curated_latents`. This covers EVERY mode — txt2img / img2img AND the
        // conditioned sub-providers (ControlNet-pose, IP-Adapter, the combined pose tier sc-5012). The
        // native `euler_discrete` default stays byte-exact — the legacy `denoise_*_latents` assemblies
        // are entered only when no curated knob is set (N1). Resolved from `req` (no weights).
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

        self.residency.run(
            &req.cancel,
            req.use_pid,
            on_progress,
            // ── Phase A: encode the positive (+ negative when CFG is on) ChatGLM3 conditioning.
            // Seed-independent; under `Sequential` the shared seam materializes these + DROPS the 6B
            // ChatGLM3 encoder before the U-Net/VAE load — bounding peak to `max(ChatGLM3, U-Net+VAE)`.
            // The negative encode is skipped when guidance is off (F-005, sc-9091): the per-mode
            // assemblies build B=1 conditioning for `cfg <= 1.0` and never read the uncond stream.
            |text: &KolorsText| {
                let pos = text.encode(&req.prompt)?;
                let neg = if cfg > 1.0 {
                    Some(text.encode(negative)?)
                } else {
                    None
                };
                Ok((pos, neg))
            },
            // Materialize the (context, pooled) tuples while the encoder is still alive (Sequential
            // only) — MLX is lazy, so un-evaluated outputs keep the 6B encoder referenced and the drop
            // would free nothing.
            |(pos, neg): &((Array, Array), Option<(Array, Array)>)| {
                let mut arrays = vec![&pos.0, &pos.1];
                if let Some(n) = neg {
                    arrays.push(&n.0);
                    arrays.push(&n.1);
                }
                eval(arrays)?;
                Ok(())
            },
            // ── Phase B: the IP-token/VAE-init/PiD-plan setup + the per-image denoise/decode count
            // loop. Identical body for both residencies.
            |heavy_owned: &KolorsHeavyOwned, (pos, neg), on_progress: &mut dyn FnMut(Progress)| {
                let heavy = &heavy_owned.heavy;

                // IP-Adapter image tokens are seed-independent — encode the reference once (the
                // `denoise_ip_latents` method CFG-batches them per image). Carries the resolved scale.
                let ip = match (ip_mode, reference) {
                    (true, Some((image, strength))) => {
                        let tokens = heavy_owned.ip_encoder.as_ref().unwrap().tokens(image)?;
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

                // F-083: the curated-path init latent (`encode_init_latents`) is seed-INDEPENDENT (the
                // VAE encode draws no RNG), so it is identical across the count loop (only the noise
                // varies per seed). Hoist it above the loop instead of re-encoding every iteration.
                let curated_init: Option<(Option<Array>, f32)> = if use_curated {
                    let (init_opt, strength) = if control.is_some() && ip_mode {
                        let (reference_image, _) = reference.expect("ip mode requires a reference");
                        (
                            Some(encode_init_latents(
                                heavy.vae(),
                                reference_image,
                                w as u32,
                                h as u32,
                            )?),
                            req.strength.unwrap_or(POSE_IMG2IMG_STRENGTH),
                        )
                    } else if let Some((image, strength)) = img2img {
                        (
                            Some(encode_init_latents(heavy.vae(), image, w as u32, h as u32)?),
                            strength,
                        )
                    } else {
                        (None, 0.0)
                    };
                    Some((init_opt, strength))
                } else {
                    None
                };

                // F-083: the legacy (native `euler_discrete`) dispatch's init latents are likewise
                // seed-INDEPENDENT, so hoist them above the loop too. Two sites mirror the two per-mode
                // branch arms below: the combined strict-pose tier (Control + IP-Adapter, seeded from
                // the reference == IP image) and plain img2img (seeded from its reference).
                let legacy_pose_init: Option<Array> =
                    if !use_curated && control.is_some() && ip.is_some() {
                        let (reference_image, _) = reference.expect("ip mode requires a reference");
                        Some(encode_init_latents(
                            heavy.vae(),
                            reference_image,
                            w as u32,
                            h as u32,
                        )?)
                    } else {
                        None
                    };
                let legacy_img2img_init: Option<Array> =
                    match (use_curated, control.is_none(), &ip, img2img) {
                        (false, true, None, Some((image, _))) => {
                            Some(encode_init_latents(heavy.vae(), image, w as u32, h as u32)?)
                        }
                        _ => None,
                    };

                // PiD decode overlay (epic 7840, sc-7848) + `from_ldm` early-stop (sc-8049). Kolors is a
                // **variance-preserving** PiD student on the shared SDXL backbone, but EVERY Kolors path
                // stores RAW variance-exploding latents `x0 + σ·ε`. So `from_ldm` is uniform: truncate
                // the schedule to the VP-capture `keep` nodes and map the captured VE latent into the
                // student's VP frame with the plan's `rescale` (`1/√(1+σ_edm²)`) before decode. The plan
                // is image-independent, so resolve it once here, against the EXACT schedule the active
                // mode denoises, BEFORE the per-image `random::seed`/noise draws (does not perturb the
                // noise stream). `pid_capture_sigma` unset ⇒ `vp_plan = None` ⇒ byte-identical clean decode.
                let vp_plan: Option<VpCapturePlan> = if req.use_pid
                    && req.pid_capture_sigma.is_some()
                {
                    let edm_sigmas: Vec<f32> = if use_curated {
                        let sched =
                            AlphaSchedule::scaled_linear(NUM_TRAIN_TIMESTEPS, BETA_START, BETA_END);
                        let ms = DiscreteModelSampling::sdxl(&sched);
                        let scheduler = req
                            .scheduler
                            .as_deref()
                            .and_then(Scheduler::from_name)
                            .unwrap_or(Scheduler::Normal);
                        let full_sigmas = schedule_sigmas(scheduler, &ms, steps);
                        // Curated init mirrors `curated_init`: combined-pose + img2img are
                        // strength-sliced; txt2img runs the full schedule.
                        let strength = if control.is_some() && ip_mode {
                            Some(req.strength.unwrap_or(POSE_IMG2IMG_STRENGTH))
                        } else {
                            img2img.map(|(_, s)| s)
                        };
                        match strength {
                            Some(strength) => {
                                let strength = strength.clamp(0.0, 1.0);
                                let eff = (steps as f32 * strength) as usize;
                                let run_start =
                                    full_sigmas.len().saturating_sub(1).saturating_sub(eff);
                                full_sigmas[run_start..].to_vec()
                            }
                            None => full_sigmas,
                        }
                    } else if control.is_some() && ip_mode {
                        // Combined strict-pose tier: img2img at POSE_IMG2IMG_STRENGTH (or `req.strength`).
                        let strength = req.strength.unwrap_or(POSE_IMG2IMG_STRENGTH);
                        KolorsEulerSampler::kolors_img2img(steps, strength, heavy.dtype())?
                            .edm_sigmas()
                            .to_vec()
                    } else if let Some((_, strength)) = img2img {
                        KolorsEulerSampler::kolors_img2img(steps, strength, heavy.dtype())?
                            .edm_sigmas()
                            .to_vec()
                    } else {
                        // txt2img / controlnet-only / ip-only: the full native leading-Euler schedule.
                        KolorsEulerSampler::kolors(steps, heavy.dtype())?
                            .edm_sigmas()
                            .to_vec()
                    };
                    vp_capture_plan(&edm_sigmas, req.pid_capture_sigma)
                } else {
                    None
                };
                let capture_sigma = vp_plan.map(|p| p.sigma).unwrap_or(0.0);
                let pid_decoder = resolve_pid_decoder_at_sigma(
                    heavy_owned.pid.as_ref(),
                    req,
                    base_seed,
                    self.descriptor.id,
                    capture_sigma,
                )?;
                let pid_ref = pid_decoder.as_ref().map(|d| d as &dyn LatentDecoder);
                // The run-step truncation each denoise method applies (keep nodes ⇒ keep-1 denoise steps).
                let run_steps = vp_plan.map(|p| p.keep - 1);

                let mut images = Vec::with_capacity(req.count as usize);
                for i in 0..req.count {
                    let seed = base_seed.wrapping_add(i as u64);
                    random::seed(seed)?;

                    // Draw this image's initial noise, then dispatch to the matching denoise assembly.
                    // Only one global-RNG draw happens per image (the noise); the img2img VAE-encode
                    // above draws none, so the per-image output stays byte-identical to the struct API.
                    let noise = random::normal::<f32>(&[1, lh, lw, 4], None, None, None)?;

                    if use_curated {
                        let control_arg = control.map(|(image, scale)| {
                            (
                                heavy_owned.control.as_ref().expect("validated above"),
                                image,
                                scale,
                            )
                        });
                        let ip_arg = ip.as_ref().map(|(tokens, scale)| (tokens, *scale));

                        let (init_opt, strength) = curated_init
                            .clone()
                            .expect("curated_init is Some iff use_curated (computed above)");

                        let latents = heavy.denoise_curated_latents(
                            req.sampler.as_deref(),
                            req.scheduler.as_deref(),
                            init_opt.as_ref(),
                            &noise,
                            &pos,
                            neg.as_ref(),
                            steps,
                            strength,
                            cfg,
                            seed,
                            h,
                            w,
                            control_arg,
                            ip_arg,
                            run_steps,
                            &req.cancel,
                            on_progress,
                        )?;
                        let latents = match &vp_plan {
                            Some(p) => {
                                mlx_rs::ops::multiply(&latents, mlx_gen::array::scalar(p.rescale))?
                            }
                            None => latents,
                        };
                        on_progress(Progress::Decoding);
                        images.push(decode_image(heavy.vae(), &latents, pid_ref)?);
                        continue;
                    }

                    let latents =
                        if let (Some((skeleton, control_scale)), Some((tokens, ip_scale))) =
                            (control, &ip)
                        {
                            // Combined strict-pose tier (sc-5012): pose ControlNet + IP-Adapter identity,
                            // on an img2img init from the SAME reference (the IP image).
                            let init_latents = legacy_pose_init.as_ref().expect(
                                "legacy_pose_init is Some in the non-curated combined-pose mode",
                            );
                            let strength = req.strength.unwrap_or(POSE_IMG2IMG_STRENGTH);
                            heavy.denoise_controlnet_ip_latents(
                                heavy_owned.control.as_ref().expect("validated above"),
                                tokens,
                                init_latents,
                                &noise,
                                skeleton,
                                &pos,
                                neg.as_ref(),
                                steps,
                                strength,
                                cfg,
                                control_scale,
                                *ip_scale,
                                h,
                                w,
                                run_steps,
                                &req.cancel,
                                on_progress,
                            )?
                        } else if let Some((image, scale)) = control {
                            heavy.denoise_controlnet_latents(
                                heavy_owned.control.as_ref().expect("validated above"),
                                &noise,
                                image,
                                &pos,
                                neg.as_ref(),
                                steps,
                                cfg,
                                scale,
                                h,
                                w,
                                run_steps,
                                &req.cancel,
                                on_progress,
                            )?
                        } else if let Some((tokens, scale)) = &ip {
                            heavy.denoise_ip_latents(
                                tokens,
                                &noise,
                                &pos,
                                neg.as_ref(),
                                steps,
                                cfg,
                                *scale,
                                h,
                                w,
                                run_steps,
                                &req.cancel,
                                on_progress,
                            )?
                        } else if let Some((_image, strength)) = img2img {
                            let x0 = legacy_img2img_init.as_ref().expect(
                                "legacy_img2img_init is Some in the non-curated img2img mode",
                            );
                            heavy.denoise_img2img_latents(
                                x0,
                                &noise,
                                &pos,
                                neg.as_ref(),
                                steps,
                                strength,
                                cfg,
                                h,
                                w,
                                run_steps,
                                &req.cancel,
                                on_progress,
                            )?
                        } else {
                            heavy.denoise_latents(
                                &noise,
                                &pos,
                                neg.as_ref(),
                                steps,
                                cfg,
                                h,
                                w,
                                run_steps,
                                &req.cancel,
                                on_progress,
                            )?
                        };

                    let latents = match &vp_plan {
                        Some(p) => {
                            mlx_rs::ops::multiply(&latents, mlx_gen::array::scalar(p.rescale))?
                        }
                        None => latents,
                    };
                    on_progress(Progress::Decoding);
                    images.push(decode_image(heavy.vae(), &latents, pid_ref)?);
                }
                Ok(GenerationOutput::Images(images))
            },
        )
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
                // `None` → the full-strength default; `Some(x)` — incl. `Some(0.0)` — verbatim (F-085).
                control = Some((image, scale.unwrap_or(DEFAULT_CONTROLNET_SCALE)));
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
    // `steps == 0` divides by zero in `KolorsEulerSampler::new` (`num_train_timesteps / num_steps`) —
    // now rejected by the shared floor above (F-007). `steps > 1100` (the train-timestep count) makes
    // `step_ratio == 0` so every timestep collapses to 1 — a silent-garbage upper bound the floor
    // doesn't know about, so keep it here (F-124). `None` falls back to DEFAULT_STEPS.
    if let Some(steps) = req.steps {
        if steps as usize > NUM_TRAIN_TIMESTEPS {
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
/// Per-component on-disk footprint (sc-10894) for the MLX fit-gate's staged-residency split — the
/// ChatGLM3-6B text encoder (`text_encoder/`), the SDXL U-Net (`unet/`), and the VAE (`vae/`), summed
/// from the exact snapshot subdirs Kolors loads (the IP-adapter / ControlNet checkpoints live in
/// separate LoadSpec sources, not under `spec.weights`).
pub(crate) fn component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    mlx_gen::PerComponentBytes::from_spec_subdirs(spec, &["text_encoder"], &["unet"], &["vae"])
}

mlx_gen::register_generators! { descriptor => load ; footprint = component_footprint }

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
            pid: None,
            identity: None,
            text_encoder: None,
            offload_policy: Default::default(),
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
        // `steps == 0` would divide by zero in the sampler — now rejected by the shared floor (F-007,
        // message "steps must be >= 1"). `steps > NUM_TRAIN_TIMESTEPS` collapses every timestep to 1 —
        // still Kolors' own upper-bound check (F-124). Both must be rejected; `None` and an in-range
        // count pass.
        let caps = descriptor().capabilities;
        let base = GenerationRequest {
            prompt: "a fox".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        // steps == 0 is rejected by the floor with its own message.
        let zero = GenerationRequest {
            steps: Some(0),
            ..base.clone()
        };
        let err = validate_request(&caps, &zero).unwrap_err().to_string();
        assert!(err.contains("steps must be >= 1"), "steps=0 got: {err}");
        // steps > NUM_TRAIN_TIMESTEPS is rejected by Kolors' upper-bound check.
        let over = GenerationRequest {
            steps: Some(NUM_TRAIN_TIMESTEPS as u32 + 1),
            ..base.clone()
        };
        let err = validate_request(&caps, &over).unwrap_err().to_string();
        assert!(err.contains("steps must be in"), "over-max got: {err}");
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

    #[test]
    fn advertises_sequential_offload() {
        // Kolors now honors the shared Residency seam (epic 10834, sc-10840).
        assert!(descriptor().capabilities.supports_sequential_offload);
    }

    // ── Sequential residency (epic 10834, sc-10840): weight-free proof the dispatch HONORS
    // `offload_policy`. `build_residency` points at a non-existent snapshot dir; the discriminator is
    // deferral:
    //   * `Sequential` captures the two per-phase loaders, touches NO weights → `Ok` + `is_sequential`.
    //   * `Resident` eager-loads the ChatGLM3 encoder from the missing dir → `Err`.
    // A dispatch that ignored `offload_policy` (always `Resident`) would eager-load under a `Sequential`
    // request and fail the first assertion. The real-weights A/B is `#[ignore]`d; this runs by default.
    fn missing_snapshot_spec(policy: OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(
            "/nonexistent/kolors-residency-test-snapshot".into(),
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
            !msg.contains("single .safetensors file"),
            "expected an eager-load failure, not the up-front single-file guard: {msg}"
        );
    }
}
