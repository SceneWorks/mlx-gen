//! Qwen2.5-VL **vision transformer** (sc-2465 slice 6a) — the image branch of the VL encoder used
//! by Qwen-Image-Edit. A patch-embed Conv3d → 32 pre-norm blocks (windowed attention, with full
//! attention at blocks `[7,15,23,31]`) → patch merger, producing vision embeds that get spliced
//! into the text stream (slice 6b).
//!
//! Stays **bf16** under Q8 (only the MMDiT quantizes), so — unlike the transformer — its linears
//! are plain `mlx_gen::nn` dense ops, mirroring the text encoder (`text_encoder::attention`).
//!
//! Ported 1:1 from the frozen fork's `model/qwen_text_encoder/qwen_vision_*.py`. Built micro-gated:
//! the weight-free index/RoPE math ([`grid`]) is verified first, then the weight-bearing modules
//! ([`patch_embed`], [`attention`], [`mlp`], [`block`], [`merger`]), then the full [`transformer`]
//! assembly.

pub mod attention;
pub mod block;
pub mod grid;
pub mod merger;
pub mod mlp;
pub mod patch_embed;
pub mod transformer;

pub use attention::VisionAttention;
pub use block::VisionBlock;
pub use merger::PatchMerger;
pub use mlp::VisionMlp;
pub use patch_embed::VisionPatchEmbed;
pub use transformer::{VisionConfig, VisionTransformer};
