//! `QwenImageControl` — the Qwen-Image **ControlNet (strict pose)** variant (epic 3401 / sc-8267),
//! registered as its own `Generator` (`qwen_image_control`) via the alibaba-pai
//! `Qwen-Image-2512-Fun-Controlnet-Union` checkpoint (a VACE-style Fun-Controlnet-Union, Apache-2.0,
//! ungated — it **replaces** the retired InstantX `Qwen-Image-ControlNet-Union` on the Qwen path).
//!
//! Identical to [`crate::model::QwenImage`] (T2I) except it also loads a [`QwenFunControlBranch`]
//! VACE control branch and `generate` threads a VAE-encoded pose skeleton through it: each denoise
//! step the control branch computes 5 per-block hints from the post-embedder streams + the (constant)
//! 132-ch packed control context, which the frozen base 60-layer MMDiT adds into its image stream at
//! `control_layers = [0, 12, 24, 36, 48]` scaled by the request's control scale. [`load`] needs the
//! base snapshot (`spec.weights`) **and** the control checkpoint (`spec.control`); it applies both
//! dense, then quantizes base + control together (Q4/Q8, transformer-only — the fork's
//! overlay-then-quantize ordering). Identity comes from a character LoRA on the **base**
//! (`spec.adapters`); the control branch is never an adapter target.
//!
//! Accepts the three structural control signals — **pose/canny/depth** — which the 2512-Fun Union
//! admits via one input-agnostic VACE control path (no mode index; sc-8250). **Base pose-from-prompt**
//! (composing with the edit model is a later reach). Pose parity vs the fork's
//! `pipeline_qwenimage_control` is gated by `tests/control_real_weights.rs` (`#[ignore]`, M-series).
//!
//! Component residency (epic 10834 Phase 1, sc-11006 — the fan-out sibling of the T2I sc-11000):
//! under [`OffloadPolicy::Sequential`] the Qwen2.5-VL text encoder (~15 GB) is dropped after the
//! text-encode phase, so peak unified memory is bounded to `max(text-encoder, DiT+control+VAE)`
//! instead of the sum. The control branch is an **extra** heavy component (loaded + quantized with the
//! base transformer), so it stays on the heavy side of the split; the pose VAE-encode uses the VAE
//! (not the text encoder) and so runs after the drop, byte-identically.

use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    require_base_dir, require_control, AcceptedControlKinds, Capabilities, ConditioningKind,
    ControlBranch, ControlKind, Error, GenerationOutput, GenerationRequest, Generator,
    LatentDecoder, LoadSpec, Modality, ModelDescriptor, OffloadPolicy, Precision, Progress, Quant,
    Residency, Result, WeightsSource,
};
use mlx_gen_pid::{flow_capture_for_request, resolve_pid_decoder_at_sigma, PidEngine};
use std::path::Path;

use crate::control_transformer::QwenFunControlBranch;
use crate::loader;
use crate::model::validate_request;
use crate::pipeline::{
    create_noise, decode_and_collect, denoise_control_with_progress, encode_fun_control_context,
    encode_prompt, negative_or_fallback, qwen_samplers, qwen_schedulers, resolve_run_params,
    PID_BACKBONE,
};
use crate::text_encoder::QwenTextEncoder;
use crate::transformer::QwenTransformer;
use crate::vae::QwenVae;

/// Registry id for the Qwen-Image ControlNet (strict pose) variant.
pub const MODEL_ID: &str = "qwen_image_control";

/// The control variant's identity + capabilities — the base Qwen-Image T2I surface (true CFG /
/// negative prompt / guidance / Lightning) plus the **required** `Control` (pose skeleton)
/// conditioning. LoRA/LoKr (character identity) is on the base transformer.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "qwen-image",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: true,
            // Control (required, pose) only in v1 — no img2img Reference / edit compose yet.
            conditioning: vec![ConditioningKind::Control],
            supports_lora: true,
            supports_lokr: true,
            // Curated unified-framework integrator menu (epic 7114 P3) + the `lightning` profile.
            samplers: qwen_samplers(),
            schedulers: qwen_schedulers(),
            supported_guidance_methods: vec![],
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: true,
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: false,
            requires_sigma_shift: true,
            // Wired onto the shared `Residency` seam; honors Sequential offload (F-176).
            supports_sequential_offload: true,
        },
    }
}

/// A loaded control generator: the cached descriptor, the (tiny, always-warm) tokenizer, and the
/// heavy-component residency strategy (base components + the control branch) — sc-11006.
pub struct QwenImageControl {
    descriptor: ModelDescriptor,
    tokenizer: TextTokenizer,
    /// Component-residency strategy (epic 10834 Phase 1, sc-11006; hoisted to the shared seam in
    /// sc-11125), selected from [`LoadSpec::offload_policy`]. `Resident` (default) holds the Qwen2.5-VL
    /// text encoder + DiT + control branch + VAE warm; `Sequential` holds only the per-phase loader
    /// closures and re-loads per generation in phase order (encode → **drop the text encoder** →
    /// denoise/decode), bounding peak unified memory to `max(text-encoder, DiT+control+VAE)` instead of
    /// the sum. The [`Residency`] seam owns the eval/drop/clear discipline, the stage-boundary cancel
    /// checks, and the error-safe cache flush.
    residency: Residency<QwenTextEncoder, QwenControlHeavyOwned>,
}

/// The heavy render-phase components — the base MMDiT transformer, the VACE control branch, the VAE,
/// and the optional PiD decoder — everything but the text encoder. Owned by the `Resident` components
/// or by a `Sequential` generate. The control branch is an **extra** component vs T2I: it is loaded +
/// quantized alongside the base transformer and lives on the heavy side of the residency split.
struct QwenControlHeavyOwned {
    transformer: QwenTransformer,
    controlnet: QwenFunControlBranch,
    vae: QwenVae,
    /// Optional PiD super-resolving decoder (epic 7840, sc-7845); see [`crate::model::QwenImage`].
    pid: Option<PidEngine>,
}

/// A borrow of the heavy render-phase components, so the denoise/decode body runs identically whether
/// they are held resident or were just loaded by the `Sequential` path (candle's `DitRef`).
struct QwenControlHeavy<'a> {
    transformer: &'a QwenTransformer,
    controlnet: &'a QwenFunControlBranch,
    vae: &'a QwenVae,
    pid: Option<&'a PidEngine>,
}

impl QwenControlHeavyOwned {
    fn as_ref(&self) -> QwenControlHeavy<'_> {
        QwenControlHeavy {
            transformer: &self.transformer,
            controlnet: &self.controlnet,
            vae: &self.vae,
            pid: self.pid.as_ref(),
        }
    }
}

/// Construct a [`QwenImageControl`] from a [`LoadSpec`].
///
/// `spec.weights` must be a base `Qwen/Qwen-Image-2512` snapshot directory and `spec.control`
/// (required) the alibaba-pai `Qwen-Image-2512-Fun-Controlnet-Union` checkpoint (a single
/// `.safetensors` `File`, or a `Dir`). Base + control load dense (bf16); `spec.quantize` (Q4/Q8) then
/// quantizes both transformers (group_size 64). The text encoder + VAE stay dense (the fork's
/// transformer-only quant scope — see [`crate::model::load`]).
///
/// Component residency (epic 10834 Phase 1, sc-11006; hoisted to the shared [`Residency::from_policy`]
/// seam in sc-11126, F-180): `Resident` (default) builds every heavy component now via
/// [`build_residency`] and holds it warm; `Sequential` keeps only the spec and re-loads per generate in
/// phase order (encode → drop the text encoder → denoise/decode) to bound peak memory to
/// `max(text-encoder, DiT+control+VAE)`. Both use the same per-phase loaders, so the components are
/// byte-identical.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    // Resolve the base dir + required control checkpoint up front — fail-fast for BOTH policies — then
    // the always-warm tokenizer, then the shared [`build_residency`] dispatch.
    let (root, _control) = resolve_base_and_control(spec)?;
    // F-181: Sequential + a load-time quant over a dense snapshot re-quantizes every generate. An
    // already-packed turnkey loads packed (no re-quant); `Resident` quantizes once. So warn only for
    // the Sequential-over-dense combination that actually pays the repeated cost.
    if let Some(q) = spec.quantize {
        if matches!(spec.offload_policy, OffloadPolicy::Sequential)
            && loader::needs_load_time_quant(root, q.bits(), MODEL_ID)?
        {
            mlx_gen::residency::warn_sequential_requantize(MODEL_ID, q.bits());
        }
    }
    let tokenizer = loader::load_tokenizer(root)?;
    Ok(Box::new(QwenImageControl {
        descriptor: descriptor(),
        tokenizer,
        residency: build_residency(spec)?,
    }))
}

/// The policy→[`Residency`] dispatch, routed through the single [`Residency::from_policy`] seam
/// (sc-11006; hoisted to the shared seam in sc-11126, F-180) so the `match offload_policy` lives in one
/// place rather than a bespoke per-crate copy. `Resident` eager-loads the text encoder + heavy bundle
/// now (the heavy loader with `use_pid = true`, loading any PiD overlay once and reusing it);
/// `Sequential` captures the two per-phase loaders and loads nothing now, deferring each to
/// [`Residency::run`]. Both use the same [`load_text_encoder_only`] / [`load_heavy`], so the `Resident`
/// composition is byte-identical to the pre-seam one. The deferral is weight-free-testable: under
/// `Sequential` this touches no component weights, so a dispatch that ignored `offload_policy` would
/// eager-load and fail the "Sequential defers" unit test.
fn build_residency(spec: &LoadSpec) -> Result<Residency<QwenTextEncoder, QwenControlHeavyOwned>> {
    let spec_text = spec.clone();
    let spec_heavy = spec.clone();
    Residency::from_policy(
        spec.offload_policy,
        move || {
            let (root, _control) = resolve_base_and_control(&spec_text)?;
            load_text_encoder_only(root)
        },
        move |use_pid| {
            let (root, control) = resolve_base_and_control(&spec_heavy)?;
            load_heavy(&spec_heavy, root, control, use_pid)
        },
    )
}

/// Precision guard (only dense bf16 is wired) + base-snapshot-dir resolution + the **required**
/// control-checkpoint resolution, shared by [`load`] and [`build_residency`]'s per-phase loaders
/// (sc-11006). Preserves the original message order: a single-file base is rejected first (via
/// [`require_base_dir`]), then a missing control checkpoint (via [`require_control`]).
fn resolve_base_and_control(spec: &LoadSpec) -> Result<(&Path, &WeightsSource)> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "qwen_image_control: only dense bf16 is wired in the Rust port (drop the precision \
             override)"
                .into(),
        ));
    }
    // Shared load boilerplate (sc-8241): the base must be a snapshot dir, the control checkpoint is
    // required. The model id + labels keep the messages byte-identical to the hand-written originals.
    let root = require_base_dir(spec, MODEL_ID, "a base snapshot directory")?;
    let control = require_control(spec, MODEL_ID, "Qwen-Image-2512-Fun-Controlnet-Union")?;
    Ok((root, control))
}

/// Load the Qwen2.5-VL text encoder — the phase-A component dropped first under `Sequential`. Never
/// quantized (the fork's transformer-only quant scope), so the `Resident` and `Sequential` paths build
/// byte-identical encoders.
fn load_text_encoder_only(root: &Path) -> Result<QwenTextEncoder> {
    loader::load_text_encoder(root)
}

/// Load the heavy render-phase components — the base MMDiT transformer, the VACE control branch (both
/// Q4/Q8 + the base's LoRA/LoKr residuals), the VAE, and the optional PiD overlay — everything but the
/// text encoder. Factored so the `Sequential` path loads these AFTER the encoder is dropped (bounding
/// peak to `max(text-encoder, DiT+control+VAE)`). The overlay-then-quantize order (dense base + dense
/// control, THEN quantize both) matches the pre-sc-11006 `load`; the components are independent of the
/// text encoder (separate weight files, deterministic RNG-free quant), so the `Resident` composition
/// is byte-identical.
fn load_heavy(
    spec: &LoadSpec,
    root: &Path,
    control: &WeightsSource,
    load_pid: bool,
) -> Result<QwenControlHeavyOwned> {
    // Base + control applied dense first, THEN quantize together (the overlay-then-quantize ordering,
    // matching the Z-Image control port): quantizing before loading the control branch would not let
    // the dense control Linears compose. The text encoder + VAE stay dense (fork's quant scope).
    let mut transformer = loader::load_transformer(root)?;
    let mut controlnet = loader::load_controlnet(control)?;
    if let Some(q) = spec.quantize {
        let bits = q.bits();
        // F-076: reject a requested-vs-packed quant-tier mismatch on the base snapshot instead of
        // silently serving its tier; skip the no-op base quantize when the turnkey is already packed
        // at the requested bits (see `loader::needs_load_time_quant`). The control checkpoint has no
        // packed-marker convention (it ships dense), so it always takes the load-time quantize.
        if loader::needs_load_time_quant(root, bits, MODEL_ID)? {
            transformer.quantize(bits)?;
        }
        // F-076 parity for the control tier (sc-9517): a **published packed** control tier (built by
        // `convert::quantize_qwen_control_branch`) loads packed — its projections packed-detect via
        // `linear_from` — and `QwenFunControlBranch::quantize` no-ops on it (`AdaptableLinear::quantize`
        // is a no-op once quantized), so it renders as-is. But a requested-vs-packed BIT mismatch would
        // then silently serve the packed tier's bits — reject it, mirroring the base
        // `loader::needs_load_time_quant`. A dense checkpoint reports no packed bits → the request stands.
        match controlnet.packed_bits() {
            Some(packed) if packed != bits => {
                return Err(Error::Msg(format!(
                    "{MODEL_ID}: control checkpoint is a pre-quantized Q{packed} tier but Q{bits} was \
                     requested; quantize() is a no-op on packed weights so the request would silently \
                     serve Q{packed}. Point at a Q{bits} control tier (or a dense checkpoint)."
                )));
            }
            Some(_) => {}
            None => controlnet.quantize(bits)?,
        }
    }
    // Character-identity LoRA/LoKr targets the base transformer only (the control branch is never an
    // adapter target). No-op when `spec.adapters` is empty.
    if !spec.adapters.is_empty() {
        crate::adapters::apply_qwen_adapters(&mut transformer, &spec.adapters)?;
    }
    // Optional PiD overlay, loaded only when the spec carries it AND this generate uses it (`load_pid`,
    // F-177) — Resident passes `true`, Sequential passes `req.use_pid`.
    let pid = if load_pid {
        spec.pid
            .as_ref()
            .map(|p| PidEngine::from_spec(p, PID_BACKBONE))
            .transpose()?
    } else {
        None
    };
    let vae = loader::load_vae(root)?;
    Ok(QwenControlHeavyOwned {
        transformer,
        controlnet,
        vae,
        pid,
    })
}

/// The 2512-Fun-Controlnet-Union VACE checkpoint is input-agnostic: pose, canny, and depth differ
/// only by the preprocessor-produced control image (no mode index — sc-8250). Spelled out as
/// `Only([Pose, Canny, Depth])` so a free-form `ControlKind::Other` is rejected rather than silently
/// coerced into the union path. A free function so the policy is unit-testable without a loaded model.
fn accepted_kinds() -> AcceptedControlKinds {
    AcceptedControlKinds::Only(vec![
        ControlKind::Pose,
        ControlKind::Canny,
        ControlKind::Depth,
    ])
}

/// The 2512-Fun Union admits the three structural control signals — pose/canny/depth share one
/// VACE control path, so all are accepted (sc-8250); only a free-form `ControlKind::Other` is
/// rejected. The control boilerplate (resolve/validate-present + the load helpers above) comes from
/// the shared trait (sc-8241).
impl ControlBranch for QwenImageControl {
    fn model_id(&self) -> &'static str {
        MODEL_ID
    }

    fn accepted_control_kinds(&self) -> AcceptedControlKinds {
        accepted_kinds()
    }

    /// Fun-Union accepts pose/canny/depth; only the catch-all `Other` reaches this rejection, so the
    /// default Qwen "pose control only" wording is replaced with the union family's actual surface.
    fn unsupported_kind_message(&self, kind: &ControlKind) -> String {
        format!(
            "{MODEL_ID}: 2512-Fun-Controlnet-Union accepts pose/canny/depth control, got {kind:?}"
        )
    }

    fn missing_control_message(&self) -> String {
        format!("{MODEL_ID} requires a Control (pose/canny/depth) conditioning")
    }
}

mlx_gen::impl_generator!(QwenImageControl {
    validate: |s, req| s.validate_impl(req),
    generate: generate_impl,
});

impl QwenImageControl {
    fn validate_impl(&self, req: &GenerationRequest) -> Result<()> {
        // Shared capability floor, then the shared control-present check (sc-8241's
        // `ControlBranch::require_control_present`, which uses Qwen's "(pose skeleton)" message).
        validate_request(self.descriptor.id, &self.descriptor.capabilities, req)?;
        self.require_control_present(req)?;
        Ok(())
    }

    /// The rich-`Result` body behind [`Generator::generate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the family
    /// helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    ///
    /// The staged residency lifecycle (encode pos+neg → drop the Qwen2.5-VL encoder under `Sequential`
    /// → load the base DiT/control/VAE/PiD → denoise/decode → free the heavy bundle) is driven by the
    /// shared [`Residency::run`] seam (sc-11125).
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;

        // Shared step/sampler/guidance/seed resolution (F-117).
        let params = resolve_run_params(req, req.width, req.height);

        let (control_image, control_scale) = self.resolve_control(req)?;

        // Phase A: prompt → embeds (epic 10834 Phase 1, sc-11006; sc-11125). Under `Sequential` the
        // shared seam loads the Qwen2.5-VL encoder, encodes pos+neg, materializes, then DROPS it +
        // `clear_cache()` so its ~15 GB frees before the DiT/control load below. Under `Resident` it
        // borrows the warm encoder. `neg` is `None` under Lightning (CFG-distilled → one forward/step).
        self.residency.run(
            &req.cancel,
            req.use_pid,
            on_progress,
            |te: &QwenTextEncoder| {
                let pos = encode_prompt(&self.tokenizer, te, &req.prompt, MODEL_ID)?;
                let neg = if params.is_lightning {
                    None
                } else {
                    Some(encode_prompt(
                        &self.tokenizer,
                        te,
                        negative_or_fallback(req),
                        MODEL_ID,
                    )?)
                };
                Ok((pos, neg))
            },
            // Materialize pos (+neg) while the encoder is still alive (Sequential only).
            |(pos, neg)| {
                match neg {
                    Some(neg) => mlx_rs::transforms::eval([pos, neg])?,
                    None => mlx_rs::transforms::eval([pos])?,
                }
                Ok(())
            },
            // ── Establish the heavy render components (base DiT + control branch + VAE + PiD) and run
            // the denoise/decode body once against the `heavy` borrow — identical for both residencies.
            |heavy_owned, enc, on_progress| {
                let heavy = heavy_owned.as_ref();
                let (pos, neg) = enc;

                // VAE-encode + pack the pose skeleton to the 132-ch control context `[1, seq, 132]` (constant
                // across steps + the batch). The 2512-Fun control path VAE-encodes the control image and
                // concatenates a zero mask + zero inpaint latent before packing 2×2 (pose-only layout). This is
                // a deterministic VAE encode, independent of `pos`/`neg`, so under `Sequential` running it here
                // — after the text-encoder drop, with the VAE just loaded — is byte-identical to the Resident
                // order (same hoist argument as the T2I img2img `encode_init_latents`).
                let control_cond =
                    encode_fun_control_context(heavy.vae, control_image, req.width, req.height)?;

                // Decode seam (sc-7845) + `from_ldm` early-stop (sc-7993): the partially-denoised x_k at the
                // achieved σ (truncated schedule) when use_pid + pid_capture_sigma; else the clean σ=0 path.
                // Control denoises from full noise (the pose is a constant conditioning), so `start_step = 0`.
                let (capture_sigma, keep) = flow_capture_for_request(req, &params.sigmas, 0);
                let pid_decoder = resolve_pid_decoder_at_sigma(
                    heavy.pid,
                    req,
                    params.base_seed,
                    MODEL_ID,
                    capture_sigma,
                )?;
                let decoder: &dyn LatentDecoder = match &pid_decoder {
                    Some(d) => d,
                    None => heavy.vae,
                };
                let denoise_sigmas = &params.sigmas[..keep];
                let images = decode_and_collect(
                    decoder,
                    req.count,
                    params.base_seed,
                    req.width,
                    req.height,
                    on_progress,
                    |seed, progress| {
                        let noise = create_noise(seed, req.width, req.height)?;
                        denoise_control_with_progress(
                            heavy.transformer,
                            heavy.controlnet,
                            params.sampler_name.as_deref(),
                            denoise_sigmas,
                            seed,
                            noise,
                            &control_cond,
                            &pos,
                            neg.as_ref(),
                            params.guidance,
                            control_scale,
                            req.width,
                            req.height,
                            &req.cancel,
                            progress,
                        )
                    },
                )?;
                Ok(GenerationOutput::Images(images))
            },
        )
    }
}

// Link-time registration (epic 3720): the macro emits the `inventory::submit!` and bridges the
// crate's rich `Result` into the registry's backend-neutral `gen_core::Result`.
mlx_gen::register_generators! { descriptor => load }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_is_qwen_image_control() {
        let d = descriptor();
        assert_eq!(d.id, "qwen_image_control");
        assert_eq!(d.family, "qwen-image");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.accepts(ConditioningKind::Control));
        assert!(d.capabilities.supports_lora);
    }

    #[test]
    fn accepts_pose_canny_depth_via_control_branch() {
        // The 2512-Fun Union is input-agnostic: pose, canny, and depth are all accepted (they differ
        // only by the preprocessor-produced control image, no mode index — sc-8250). A free-form
        // `Other` kind is rejected. This is exactly the `accepted_control_kinds()` policy the
        // `ControlBranch` impl returns.
        let accepted = accepted_kinds();
        assert!(accepted.accepts(&ControlKind::Pose));
        assert!(accepted.accepts(&ControlKind::Canny));
        assert!(accepted.accepts(&ControlKind::Depth));
        assert!(!accepted.accepts(&ControlKind::Other("scribble".into())));
    }

    #[test]
    fn load_rejects_missing_control_weights() {
        // Without `spec.control`, load must fail on the missing control weights, proving the overlay
        // is a hard requirement (it fails here before touching the missing base snapshot).
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(
            err.contains("Qwen-Image-2512-Fun-Controlnet-Union"),
            "got: {err}"
        );
    }

    #[test]
    fn load_rejects_single_file_base() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/q.safetensors".into()))
            .with_control(WeightsSource::File("/tmp/control.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }

    // ── F-180 (sc-11126): weight-free, default-run proof that Qwen-Image-Control's dispatch HONORS
    // `offload_policy`. `build_residency` points at a non-existent base snapshot *directory* (with a
    // control checkpoint present so `resolve_base_and_control`'s up-front precision/single-file/missing-
    // control guards all pass) and the discriminator is deferral:
    //   * `Sequential` captures the two per-phase loaders, touches NO weights → `Ok` + `is_sequential`.
    //   * `Resident` eager-loads the Qwen2.5-VL text encoder from the missing dir → `Err`.
    // A dispatch that ignored `offload_policy` (always `Resident`) would eager-load under a `Sequential`
    // request and fail the first assertion. The A/B real-weight test is `#[ignore]`d; this runs by
    // default.
    fn missing_snapshot_spec(policy: OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(
            "/nonexistent/qwen-image-control-residency-test-snapshot".into(),
        ))
        .with_control(WeightsSource::Dir(
            "/nonexistent/qwen-image-control-residency-test-control".into(),
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
            !msg.contains("single .safetensors file")
                && !msg.contains("precision override")
                && !msg.contains("Qwen-Image-2512-Fun-Controlnet-Union"),
            "expected an eager-load failure, not an up-front guard: {msg}"
        );
    }
}
