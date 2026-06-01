//! Qwen2.5-VL text encoder (text path) — the decoder-only LM that turns the prompt into the
//! transformer's conditioning. Port of the fork's `QwenEncoder`/`QwenTextEncoder`.
//!
//! Qwen2.5-style (vs Z-Image's Qwen3): **biases on q/k/v** (o_proj bias-less), **no per-head
//! q_norm/k_norm**, and the **final RMSNorm is applied** (the returned hidden states are the
//! last layer's, normed). GQA (28 query / 4 kv heads), HF half-split RoPE (θ=1e6), SwiGLU MLP,
//! pre-norm residual blocks. The multimodal RoPE (`mrope_section [16,24,24]`) reduces to standard
//! RoPE for the text-only path (all three position sections are identical), so we use the plain
//! HF RoPE here.

pub mod attention;
pub mod encoder;
pub mod layer;
pub mod mlp;
pub mod rope;
pub mod vision;
pub mod vision_language;

pub use attention::QwenTextAttention;
pub use encoder::{QwenTextEncoder, QwenTextEncoderConfig};
pub use layer::QwenEncoderLayer;
pub use mlp::QwenMlp;
pub use rope::TextRope;
pub use vision_language::QwenVisionLanguageEncoder;

use mlx_rs::ops::matmul;
use mlx_rs::Array;

use mlx_gen::Result;

/// `y = x · Wᵀ` for a stored `[out, in]` weight (bias-less Linear, e.g. `o_proj` / MLP).
pub(crate) fn matmul_t(x: &Array, w: &Array) -> Result<Array> {
    Ok(matmul(x, w.t())?)
}

/// Join a module prefix with a leaf name, tolerating an empty prefix.
pub(crate) fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}
