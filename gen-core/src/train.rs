//! The `Trainer` contract — LoRA/LoKr fine-tuning of a registered model (epic 3039), the training
//! analog of [`Generator`](crate::generator::Generator). See `docs/MODEL_ARCHITECTURE.md`.
//!
//! SceneWorks owns the training *product* surface (datasets, plan normalization, validation, the
//! queue) in Rust; this is the *execution* surface that replaces the Python kernel for the
//! MLX-native families. The worker maps its normalized `TrainingPlan` onto a [`TrainingRequest`]
//! and calls [`Trainer::train`] — exactly as it maps `ImageRequest` → `GenerationRequest` and calls
//! `Generator::generate`. mlx-gen owns these shapes; it does not depend on the SceneWorks contract.
//!
//! The spike (sc-3042) proved the per-family training mechanism (functional autograd over an
//! external LoRA factor map, re-injected as `Adapter::Lora`, stepped with `keyed_value_and_grad` +
//! AdamW). This module is the family-agnostic glue around it: the config/progress/request shapes,
//! the [`schedule`] LR helpers, dataset bucketing, and checkpointing. Each family crate implements
//! [`Trainer`] (Z-Image in sc-3044) and self-registers via [`crate::registry::TrainerRegistration`].

// The pure LR-schedule policy lives here (gen-core); the MLX training kernels
// (checkpoint/dataset/lora/optim, incl. `TrainOptimizer`) stay in mlx-gen's `train` module.
pub mod schedule;

use std::path::PathBuf;

pub use schedule::LrSchedule;

use crate::generator::Modality;
use crate::media::Image;
use crate::runtime::CancelFlag;

/// Adapter network parameterization (mirrors SceneWorks `network_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NetworkType {
    /// Standard low-rank `A·B` adapter.
    #[default]
    Lora,
    /// LyCORIS Kronecker-product adapter (LoKr); `decompose_factor` is the block-split knob.
    Lokr,
}

impl NetworkType {
    /// Parse the free-form contract string; unknown / empty → `Lora`.
    pub fn parse(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "lokr" => NetworkType::Lokr,
            _ => NetworkType::Lora,
        }
    }
}

/// Concrete training hyperparameters — the engine-side mirror of SceneWorks' `TrainingConfig`
/// (with the typed equivalents of the fields its Python kernel reads out of the plan's `advanced`
/// bag: LR schedule, warmup, timestep sampling, loss type, LoKr decompose factor, target modules).
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingConfig {
    /// LoRA/LoKr rank (network dimension).
    pub rank: u32,
    /// LoRA alpha; the residual is scaled by `alpha/rank`.
    pub alpha: f32,
    pub learning_rate: f32,
    /// Total training micro-steps (forward/backward passes; an optimizer update fires every
    /// `gradient_accumulation` of them).
    pub steps: u32,
    pub batch_size: u32,
    pub gradient_accumulation: u32,
    /// Gradient (activation) checkpointing: recompute each transformer block's activations during
    /// the backward pass instead of retaining them, bounding the first-step working set (sc-4874 —
    /// without it a production-resolution run can exceed unified memory and the OS hard-kills the
    /// worker). This is the engine-side home of the SceneWorks "Gradient Checkpointing" toggle (which
    /// was previously a no-op on the Rust path). Strictly opt-in — never auto-enabled; a run that
    /// would exceed the memory budget with it off is refused up front by the family trainer's
    /// pre-flight guard (a catchable error recommending this flag). Numerically it changes nothing
    /// (grads are bit-identical to the dense path); the cost is recompute time.
    pub gradient_checkpointing: bool,
    /// Training compute dtype for the model forward/backward: `"bf16"` (default — halves the
    /// activation working set; the ecosystem-standard mixed precision, sc-4887) or `"f32"` (full
    /// precision, the pre-sc-4887 behavior). The trainable adapter factors, loss, gradients, and
    /// optimizer state stay f32 either way (master-weights pattern); only the frozen base weights
    /// and the activation stream are cast. Unrecognized values mean f32.
    pub train_dtype: String,
    /// Square training resolution edge in pixels; bucketed down to a multiple of 32.
    pub resolution: u32,
    /// Adapter-checkpoint cadence, in micro-steps (`0` = no intermediate checkpoints).
    pub save_every: u32,
    pub seed: u64,
    /// Optimizer name, kept a free string to stay engine-agnostic (`adamw`/`adam`/`lion`/…); the
    /// family trainer maps it to an mlx-rs optimizer. Prodigy/Rose are not in mlx-rs (sc-3048).
    pub optimizer: String,
    pub weight_decay: f32,
    pub lr_scheduler: LrSchedule,
    pub lr_warmup_steps: u32,
    pub network_type: NetworkType,
    /// LoKr block-split factor (`-1` = auto). Ignored for plain LoRA.
    pub decompose_factor: i32,
    /// LoRA target module suffixes (e.g. `["to_q","to_k","to_v","to_out.0"]`); empty = the family
    /// default set.
    pub lora_target_modules: Vec<String>,
    /// ControlNet control type for a control-branch training run (`"pose"`/`"canny"`/`"depth"`/…);
    /// `None` for LoRA/LoKr training. Selects the branch's conditioning semantics and is recorded in
    /// the produced overlay's metadata so the model catalog / registration describes it correctly
    /// (rather than a hardcoded label). A control-branch trainer requires it set.
    pub control_type: Option<String>,
    /// Flow-match timestep sampling distribution (`sigmoid`/`linear`/`uniform`/…) — the *noise*
    /// schedule, distinct from `lr_scheduler`.
    pub timestep_type: String,
    /// Timestep sampling bias (`balanced`/`high_noise`/`low_noise`/…).
    pub timestep_bias: String,
    /// Loss type (`mse`/`mae`/…); families default to MSE on the velocity target.
    pub loss_type: String,
    /// Trigger word baked into captions / surfaced on the output adapter.
    pub trigger_word: Option<String>,
    /// Preview-sample cadence, in micro-steps (`0` = no preview samples). At each multiple the
    /// trainer renders preview images from the **in-progress adapter** (installed exactly as a train
    /// step installs it) so the user can watch the LoRA learn — the engine-side home of the
    /// SceneWorks "Sample cadence" control. The Python trainer did this; the native port dropped it
    /// (sc-5637). Samples are emitted via the [`TrainingProgress::Sample`] event on the existing
    /// `on_progress` callback — no extra trainer method.
    pub sample_every: u32,
    /// The prompts rendered at each [`sample_every`](Self::sample_every) cadence (the family trainer
    /// caps how many it renders per cadence — typically ≤4). Empty disables sampling regardless of
    /// `sample_every`. Encoded once during the dataset-caching pass (before families that free their
    /// text encoder post-cache do so), then reused for every cadence.
    pub sample_prompts: Vec<String>,
    /// Inference (denoise) steps per preview sample.
    pub sample_steps: u32,
    /// Guidance scale for preview samples. Guidance-distilled families (z-image-turbo,
    /// lens-turbo, …) ignore it / render at their fixed schedule; CFG families (sdxl, kolors) honor it.
    pub sample_guidance_scale: f32,
    /// Mid-schedule **resume** (sc-9560 / F-125): when `true`, the family trainer looks in
    /// [`output_dir`](TrainingRequest::output_dir) for the latest resume snapshot written by a prior
    /// interrupted run of the **same** output adapter (`file_name`) at [`save_every`](Self::save_every)
    /// — its trainable factors, optimizer state, and step/update index — and continues from there
    /// instead of restarting at step 0. `false` (the default) always trains from scratch. Requires
    /// `save_every > 0` on the interrupted run to have produced a snapshot; a run whose target `steps`
    /// is already reached by the snapshot is a no-op. Resume is bit-exact when a snapshot lands on an
    /// optimizer-update boundary (always for `gradient_accumulation = 1`; for `> 1`, use a `save_every`
    /// that is a multiple of it) — the in-flight accumulation buffer is not snapshotted.
    pub resume: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 16.0,
            learning_rate: 1e-4,
            steps: 1000,
            batch_size: 1,
            gradient_accumulation: 1,
            gradient_checkpointing: false,
            train_dtype: "bf16".to_string(),
            resolution: 1024,
            save_every: 250,
            seed: 0,
            optimizer: "adamw".to_string(),
            weight_decay: 0.0,
            lr_scheduler: LrSchedule::Constant,
            lr_warmup_steps: 0,
            network_type: NetworkType::Lora,
            decompose_factor: -1,
            lora_target_modules: Vec::new(),
            control_type: None,
            timestep_type: "sigmoid".to_string(),
            timestep_bias: "balanced".to_string(),
            loss_type: "mse".to_string(),
            trigger_word: None,
            // Preview sampling is OFF by default: a caller that does not opt in (every conformance
            // profile, every existing test that builds via `..Default::default()`) trains exactly as
            // before. The worker sets these explicitly from the plan's `advanced` bag (sc-5637).
            sample_every: 0,
            sample_prompts: Vec::new(),
            sample_steps: 20,
            sample_guidance_scale: 1.0,
            // Resume is OFF by default (F-125): a caller that does not opt in trains from scratch,
            // exactly as before. The worker sets it from the plan when re-running an interrupted job.
            resume: false,
        }
    }
}

/// One captioned training example. Paths are resolved by the caller (the worker resolves the
/// dataset's absolute image paths).
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingItem {
    pub image_path: PathBuf,
    pub caption: String,
    /// Optional per-item control-conditioning image (the ControlNet case): a rendered condition
    /// aligned to `image_path` — a pose skeleton, canny edge map, depth map, … `None` is the LoRA
    /// case, which every existing trainer ignores. A control-branch trainer requires it present on
    /// every item (its `validate` rejects the request otherwise).
    pub control_image_path: Option<PathBuf>,
}

impl TrainingItem {
    /// A captioned item with no control conditioning (the LoRA case) — the common constructor that
    /// keeps callers insulated from the optional `control_image_path` field.
    pub fn captioned(image_path: PathBuf, caption: String) -> Self {
        Self {
            image_path,
            caption,
            control_image_path: None,
        }
    }

    /// A captioned item paired with a control-conditioning image (the ControlNet case).
    pub fn with_control(image_path: PathBuf, caption: String, control_image_path: PathBuf) -> Self {
        Self {
            image_path,
            caption,
            control_image_path: Some(control_image_path),
        }
    }
}

/// A training run: the dataset, the hyperparameters, and where to write the adapter. The base model
/// is supplied at `load`-time via the [`LoadSpec`](crate::runtime::LoadSpec) (mirroring inference:
/// `load(id, spec)` then `generate(req)`), so it is not repeated here.
#[derive(Clone, Debug)]
pub struct TrainingRequest {
    pub items: Vec<TrainingItem>,
    pub config: TrainingConfig,
    /// Absolute directory the adapter (and any intermediate checkpoints) are written into.
    pub output_dir: PathBuf,
    /// Output adapter file name, e.g. `my_style.safetensors`.
    pub file_name: String,
    /// Trigger words surfaced on the produced adapter.
    pub trigger_words: Vec<String>,
    /// Cooperative cancellation, polled between steps (mirrors `GenerationRequest`).
    pub cancel: CancelFlag,
}

/// A progress event streamed during a long [`Trainer::train`] — the training analog of
/// [`Progress`](crate::runtime::Progress), with bands matching the kernel's
/// prepare→load→cache→train→checkpoint→save lifecycle.
#[derive(Clone, Debug, PartialEq)]
pub enum TrainingProgress {
    /// Resolving the dataset / building buckets.
    Preparing,
    /// Loading the (frozen) base model weights.
    LoadingModel,
    /// Encoding + caching VAE latents and prompt embeddings: item `current` of `total` (1-based).
    Caching { current: u32, total: u32 },
    /// Optimizer micro-step `step` of `total` (1-based) with the latest scalar `loss`.
    Training { step: u32, total: u32, loss: f32 },
    /// An intermediate adapter checkpoint was written at micro-step `step`.
    Checkpoint { step: u32 },
    /// A preview sample image, rendered from the in-progress adapter at micro-step `step` (sc-5637).
    /// `index`/`total` are the 1-based position within this cadence's prompt set; `prompt` is the
    /// rendered prompt; `image` is the decoded 8-bit RGB bitmap the consumer persists/streams (e.g.
    /// the worker writes it as a project asset and appends it to the job result the Training Studio
    /// renders). Emitted only when `config.sample_every > 0` and `config.sample_prompts` is non-empty.
    /// Consumers that don't care can ignore it — it is interleaved with `Training` and does not
    /// affect step/loss accounting.
    Sample {
        step: u32,
        index: u32,
        total: u32,
        prompt: String,
        image: Image,
    },
    /// Writing the final adapter.
    Saving,
}

/// What a [`Trainer::train`] produced.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingOutput {
    /// Absolute path to the final adapter safetensors.
    pub adapter_path: PathBuf,
    /// Micro-steps actually run (may be < `config.steps` if cancelled).
    pub steps: u32,
    /// The last training loss observed.
    pub final_loss: f32,
}

/// Identity + capabilities of a trainer (drives `validate` and consumer introspection). The
/// training analog of [`ModelDescriptor`](crate::generator::ModelDescriptor).
#[derive(Clone, Copy, Debug)]
pub struct TrainerDescriptor {
    /// Registry id, e.g. `"z_image_turbo"` (matches the generator id of the same base model).
    pub id: &'static str,
    pub family: &'static str,
    /// Tensor backend that registered this trainer ("mlx" | "candle"); used by the worker's
    /// per-backend capability advertisement (sc-4906, epic 3720).
    pub backend: &'static str,
    pub modality: Modality,
    pub supports_lora: bool,
    pub supports_lokr: bool,
    /// Whether this trainer can train a ControlNet-style **control branch** (the sc-10163
    /// `control_type` / per-item `control_image_path` contract), as opposed to a plain LoRA/LoKr
    /// adapter. The worker reads this to know whether a control-training plan is serviceable, and the
    /// shared [`validate_control_request`] floor uses it to decide whether a control request is a
    /// capability gap (reject typed) or must be checked for per-item control images. `false` for every
    /// LoRA/LoKr-only trainer shipped today — they must *reject* a control request, not silently train
    /// a plain adapter (F-006 / F-055).
    pub supports_control: bool,
}

/// The shared control-training validation floor (F-006) — the training analog of
/// [`Capabilities::validate_request`](crate::generator::Capabilities::validate_request). A family
/// trainer's `validate` calls this so the "control_type set ⇒ every item carries a control image"
/// invariant (and the "unsupported ⇒ typed reject" rule) lives in one place instead of being
/// re-implemented per family — the recurring fixes-don't-travel gap sc-10163 opened.
///
/// - `control_type` unset ⇒ the plain LoRA/LoKr path: no-op.
/// - `control_type` set on a trainer that does **not** advertise
///   [`supports_control`](TrainerDescriptor::supports_control) ⇒ typed [`Error::Unsupported`] (the
///   LoRA-only trainers must reject a control request, not silently produce a non-control adapter).
/// - `control_type` set on a control-capable trainer ⇒ every [`TrainingItem`] must carry a
///   `control_image_path`, else [`Error::Msg`].
pub fn validate_control_request(
    desc: &TrainerDescriptor,
    req: &TrainingRequest,
) -> crate::Result<()> {
    let Some(control_type) = req.config.control_type.as_deref() else {
        return Ok(());
    };
    if !desc.supports_control {
        return Err(crate::Error::Unsupported(format!(
            "{}: control-branch training (control_type {control_type:?}) is not supported by this \
             trainer",
            desc.id
        )));
    }
    if let Some(idx) = req
        .items
        .iter()
        .position(|it| it.control_image_path.is_none())
    {
        return Err(crate::Error::Msg(format!(
            "{}: control_type {control_type:?} requires a control image on every training item \
             (item {idx} has none)",
            desc.id
        )));
    }
    Ok(())
}

/// A LoRA/LoKr trainer for one model family — the training analog of
/// [`Generator`](crate::generator::Generator). `train` is **synchronous** (long/blocking; the
/// worker runs each job on its own thread) and takes `&mut self` because training mutates the
/// adapter parameters and optimizer state. The request carries a cancel flag and `on_progress`
/// streams the lifecycle.
pub trait Trainer {
    /// Identity + capabilities (drives `validate` and consumer UI introspection).
    fn descriptor(&self) -> &TrainerDescriptor;

    /// Reject a request this trainer cannot serve (LoKr when unsupported, empty dataset,
    /// unresolvable target modules, …) before doing expensive work.
    fn validate(&self, req: &TrainingRequest) -> crate::Result<()>;

    /// Run training to completion (or until `req.cancel` trips), writing the adapter to
    /// `req.output_dir`.
    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> crate::Result<TrainingOutput>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_item_ctors_set_control() {
        let img = PathBuf::from("a.png");
        let lora = TrainingItem::captioned(img.clone(), "a cat".into());
        assert_eq!(
            lora.control_image_path, None,
            "captioned = no control (LoRA)"
        );

        let ctrl =
            TrainingItem::with_control(img.clone(), "a cat".into(), PathBuf::from("a.pose.png"));
        assert_eq!(ctrl.control_image_path, Some(PathBuf::from("a.pose.png")));
        assert_eq!(ctrl.image_path, img);
    }

    #[test]
    fn config_default_has_no_control_type() {
        // Additive: LoRA callers building via `..Default::default()` get `control_type: None` and
        // are unaffected; only a control-branch trainer sets it.
        assert_eq!(TrainingConfig::default().control_type, None);
    }

    fn trainer_desc(supports_control: bool) -> TrainerDescriptor {
        TrainerDescriptor {
            id: "fam_trainer",
            family: "fam",
            backend: "mlx",
            modality: Modality::Image,
            supports_lora: true,
            supports_lokr: false,
            supports_control,
        }
    }

    fn train_req(control_type: Option<&str>, items: Vec<TrainingItem>) -> TrainingRequest {
        TrainingRequest {
            items,
            config: TrainingConfig {
                control_type: control_type.map(str::to_owned),
                ..Default::default()
            },
            output_dir: PathBuf::from("/out"),
            file_name: "a.safetensors".into(),
            trigger_words: Vec::new(),
            cancel: CancelFlag::default(),
        }
    }

    #[test]
    fn validate_control_request_floor() {
        // F-006: the shared control-training floor.
        let img = PathBuf::from("a.png");
        let ctrl = PathBuf::from("a.pose.png");

        // control_type unset ⇒ no-op on both trainer kinds.
        let lora_items = vec![TrainingItem::captioned(img.clone(), "a cat".into())];
        assert!(validate_control_request(
            &trainer_desc(false),
            &train_req(None, lora_items.clone())
        )
        .is_ok());

        // control_type set on a NON-control trainer ⇒ typed Unsupported (must reject, not silently
        // train a plain adapter — the F-055 class).
        let err = validate_control_request(
            &trainer_desc(false),
            &train_req(Some("pose"), lora_items.clone()),
        )
        .unwrap_err();
        assert!(
            matches!(err, crate::Error::Unsupported(_)),
            "control on a LoRA-only trainer is a capability gap → Unsupported, got {err:?}"
        );

        // control_type set on a control-capable trainer, but an item lacks a control image ⇒ Msg.
        let err =
            validate_control_request(&trainer_desc(true), &train_req(Some("pose"), lora_items))
                .unwrap_err();
        assert!(
            matches!(&err, crate::Error::Msg(_)) && err.to_string().contains("control image"),
            "got {err:?}"
        );

        // control_type set on a control-capable trainer with every item carrying a control image ⇒ ok.
        let ctrl_items = vec![TrainingItem::with_control(img, "a cat".into(), ctrl)];
        assert!(validate_control_request(
            &trainer_desc(true),
            &train_req(Some("pose"), ctrl_items)
        )
        .is_ok());
    }
}
