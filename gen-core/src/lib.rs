//! # gen-core
//!
//! The **backend-neutral contract layer** for SceneWorks generative inference. gen-core has
//! **zero tensor dependencies**: it owns the `Generator` / `Trainer` / `Captioner` / `Transform`
//! contracts, the request/output/conditioning/progress/cancel/error types, the link-time model
//! registry, and the pure host-side policy math (tokenization, PIL-compatible resize, tiling,
//! LR schedule). The tensor backends — `mlx-gen` (Apple MLX) and the forthcoming `candle-gen`
//! (Windows/CUDA) — implement these contracts and re-export this crate at their own paths.
//!
//! Numeric types here are restricted to `f32`/`f64`/`Vec<f32>`/`Vec<i32>`/`&[u8]` — never an
//! `mlx_rs::Array` or candle tensor. See epic 3720 (the unified-contract roadmap, Phase 0).

pub mod caption;
pub mod error;
pub mod face;
pub mod generator;
pub mod image_embed;
pub mod imageops;
pub mod json_constraint;
pub mod media;
pub mod registry;
pub mod runtime;
pub mod sampling;
pub mod text_embed;
pub mod textllm;
pub mod tiling;
pub mod tokenizer;
pub mod train;
pub mod transform;
pub mod weightsmeta;

pub use caption::{
    CaptionCapabilities, CaptionFinishReason, CaptionOptions, CaptionOutput, CaptionRequest,
    CaptionSampling, Captioner, CaptionerDescriptor,
};
pub use error::{Error, Result};
pub use face::{DetectedFace, FaceEmbedder, FaceEmbedderDescriptor};
pub use generator::{
    default_seed, Capabilities, Conditioning, ConditioningKind, ControlClipRef, ControlKind,
    GenerationOutput, GenerationRequest, Generator, KeyframeRef, Modality, ModelDescriptor,
    ReplacementMode, VideoClipRef,
};
pub use image_embed::{ImageEmbedder, ImageEmbedderDescriptor};
pub use json_constraint::JsonState;
pub use media::{AudioTrack, Image};
pub use registry::{
    load, load_captioner, load_image_embedder, load_text_embedder, load_transform,
    CaptionerRegistration, ImageEmbedderRegistration, ModelRegistration, TextEmbedderRegistration,
    TransformRegistration,
};
pub use registry::{load_textllm, TextLlmRegistration};
pub use registry::{load_trainer, TrainerRegistration};
pub use runtime::{
    AdapterKind, AdapterSpec, CancelFlag, LoadSpec, MoeExpert, Precision, Progress, Quant,
    WeightsSource,
};
pub use text_embed::{TextEmbedder, TextEmbedderDescriptor};
pub use textllm::{
    TextLlm, TextLlmCapabilities, TextLlmConstraint, TextLlmDescriptor, TextLlmFinishReason,
    TextLlmOutput, TextLlmRequest, TextLlmSampling,
};
pub use tiling::{TilingConfig, VaeTiling};
// NOTE: `TrainOptimizer` is intentionally NOT re-exported here — it wraps an mlx-rs optimizer and
// lives in mlx-gen (`mlx_gen::train::optim`). `LrSchedule` is pure policy and lives here.
pub use train::{
    LrSchedule, NetworkType, Trainer, TrainerDescriptor, TrainingConfig, TrainingItem,
    TrainingOutput, TrainingProgress, TrainingRequest,
};
pub use transform::{
    TargetSize, Transform, TransformCapabilities, TransformDescriptor, TransformRequest,
};

/// gen-core's package version, for the version-skew runtime guard (sc-4482).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
