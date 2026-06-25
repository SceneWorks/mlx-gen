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
mod macros;
pub mod media;
pub mod registry;
pub mod runtime;
pub mod sampling;
pub mod text_embed;
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
pub use registry::{load_trainer, TrainerRegistration};
pub use runtime::{
    AdapterKind, AdapterSpec, CancelFlag, LoadSpec, MoeExpert, Precision, Progress, Quant,
    WeightsSource,
};
pub use text_embed::{TextEmbedder, TextEmbedderDescriptor};
pub use tiling::{TilingConfig, VaeTiling};

// The independent LLM-serving library, re-exported at `gen_core::core_llm` (epic 7153, sc-7189). The
// dependency is INVERTED: gen-core CONSUMES `core-llm` — the same way mlx-gen re-exports gen-core via
// `pub use ::gen_core` — so a consumer that already pins gen-core reaches the unified
// `core_llm::TextLlm` engine (and `core_llm::load_for_model` model-first resolution) through this one
// path, with no separate core-llm pin. core-llm is itself tensor-free, preserving gen-core's invariant.
//
// `core_llm::TextLlm` is now the SOLE text-LLM contract: the legacy `gen_core::TextLlm` trait + its
// `load_textllm`/`TextLlmRegistration` registry plumbing were removed in sc-7189 Phase 3 once every
// provider had migrated (prompt-refine mac sc-7158 / candle sc-7404; JoyCaption sc-7265 / candle
// sc-7692). All text-LLM serving — including model-first resolution via `core_llm::load_for_model` —
// goes through this path.
pub use ::core_llm;
#[doc(hidden)]
pub use inventory;
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
