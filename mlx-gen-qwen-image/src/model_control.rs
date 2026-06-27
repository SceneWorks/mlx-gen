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

use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    require_base_dir, require_control, AcceptedControlKinds, Capabilities, ConditioningKind,
    ControlBranch, ControlKind, Error, GenerationOutput, GenerationRequest, Generator,
    LatentDecoder, LoadSpec, Modality, ModelDescriptor, Precision, Progress, Quant, Result,
};
use mlx_gen_pid::{flow_capture_for_request, resolve_pid_decoder_at_sigma, PidEngine};

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
        },
    }
}

/// A loaded control generator: the base components + the control branch.
pub struct QwenImageControl {
    descriptor: ModelDescriptor,
    tokenizer: TextTokenizer,
    text_encoder: QwenTextEncoder,
    transformer: QwenTransformer,
    controlnet: QwenFunControlBranch,
    vae: QwenVae,
    /// Optional PiD super-resolving decoder (epic 7840, sc-7845); see [`crate::model::QwenImage`].
    pid: Option<PidEngine>,
}

/// Construct a [`QwenImageControl`] from a [`LoadSpec`].
///
/// `spec.weights` must be a base `Qwen/Qwen-Image-2512` snapshot directory and `spec.control`
/// (required) the alibaba-pai `Qwen-Image-2512-Fun-Controlnet-Union` checkpoint (a single
/// `.safetensors` `File`, or a `Dir`). Base + control load dense (bf16); `spec.quantize` (Q4/Q8) then
/// quantizes both transformers (group_size 64). The text encoder + VAE stay dense (the fork's
/// transformer-only quant scope — see [`crate::model::load`]).
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
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

    // Base + control applied dense first, THEN quantize together (the overlay-then-quantize ordering,
    // matching the Z-Image control port): quantizing before loading the control branch would not let
    // the dense control Linears compose. The text encoder + VAE stay dense (fork's quant scope).
    let mut transformer = loader::load_transformer(root)?;
    let mut controlnet = loader::load_controlnet(control)?;
    if let Some(q) = spec.quantize {
        let bits = q.bits();
        transformer.quantize(bits)?;
        controlnet.quantize(bits)?;
    }
    // Character-identity LoRA/LoKr targets the base transformer only (the control branch is never an
    // adapter target). No-op when `spec.adapters` is empty.
    if !spec.adapters.is_empty() {
        crate::adapters::apply_qwen_adapters(&mut transformer, &spec.adapters)?;
    }
    let pid = spec
        .pid
        .as_ref()
        .map(|p| PidEngine::from_spec(p, PID_BACKBONE))
        .transpose()?;
    Ok(Box::new(QwenImageControl {
        descriptor: descriptor(),
        tokenizer: loader::load_tokenizer(root)?,
        text_encoder: loader::load_text_encoder(root)?,
        transformer,
        controlnet,
        vae: loader::load_vae(root)?,
        pid,
    }))
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
        validate_request(&self.descriptor.capabilities, req)?;
        self.require_control_present(req)?;
        Ok(())
    }

    /// The rich-`Result` body behind [`Generator::generate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the family
    /// helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;

        // Shared step/sampler/guidance/seed resolution (F-117).
        let params = resolve_run_params(req, req.width, req.height);

        let (control_image, control_scale) = self.resolve_control(req)?;

        // Positive conditioning always; negative only for true CFG (Lightning is CFG-distilled).
        let pos = encode_prompt(&self.tokenizer, &self.text_encoder, &req.prompt, MODEL_ID)?;
        let neg = if params.is_lightning {
            None
        } else {
            Some(encode_prompt(
                &self.tokenizer,
                &self.text_encoder,
                negative_or_fallback(req),
                MODEL_ID,
            )?)
        };

        // VAE-encode + pack the pose skeleton to the 132-ch control context `[1, seq, 132]` (constant
        // across steps + the batch). The 2512-Fun control path VAE-encodes the control image and
        // concatenates a zero mask + zero inpaint latent before packing 2×2 (pose-only layout).
        let control_cond =
            encode_fun_control_context(&self.vae, control_image, req.width, req.height)?;

        // Decode seam (sc-7845) + `from_ldm` early-stop (sc-7993): the partially-denoised x_k at the
        // achieved σ (truncated schedule) when use_pid + pid_capture_sigma; else the clean σ=0 path.
        // Control denoises from full noise (the pose is a constant conditioning), so `start_step = 0`.
        let (capture_sigma, keep) = flow_capture_for_request(req, &params.sigmas, 0);
        let pid_decoder = resolve_pid_decoder_at_sigma(
            self.pid.as_ref(),
            req,
            params.base_seed,
            MODEL_ID,
            capture_sigma,
        )?;
        let decoder: &dyn LatentDecoder = match &pid_decoder {
            Some(d) => d,
            None => &self.vae,
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
                    &self.transformer,
                    &self.controlnet,
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
    }
}

// Link-time registration (epic 3720): the macro emits the `inventory::submit!` and bridges the
// crate's rich `Result` into the registry's backend-neutral `gen_core::Result`.
mlx_gen::register_generators! { descriptor => load }

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::WeightsSource;

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
}
