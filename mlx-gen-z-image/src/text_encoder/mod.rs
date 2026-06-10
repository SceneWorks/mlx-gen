//! Z-Image text encoder — a Qwen3-style decoder-only LM that turns the prompt into `cap_feats`
//! (the DiT's `Conditioning`). Port of the fork's `z_image_text_encoder`.
//!
//! Qwen3-style (not Qwen2): per-head `q_norm`/`k_norm`, **no biases**, HF half-split RoPE,
//! GQA (32 query / 8 kv heads), pre-norm residual blocks. The encoder returns the **second-to-
//! last** layer's hidden states (no final norm). The sub-modules live here; the full
//! [`encoder::TextEncoder`] assembly + prompt encoding are in [`encoder`].

pub mod attention;
pub mod encoder;
pub mod layer;
pub mod mlp;

pub use attention::TextAttention;
pub use encoder::{TextEncoder, ZTextEncoderConfig};
pub use layer::EncoderLayer;
pub use mlp::TextMlp;
// The HF half-split text RoPE is identical across families and now lives in core (F-006).
pub use mlx_gen::nn::TextRope;

/// mlx `nn.RMSNorm` default eps — used by the per-head `q_norm`/`k_norm` (which the fork
/// constructs without an explicit eps). The block-level layer norms use `rms_norm_eps` (1e-6).
pub(crate) const QK_NORM_EPS: f32 = 1e-5;

/// Join a module prefix with a leaf name, tolerating an empty prefix (so flat fixtures and
/// real `layers.{i}` trees both resolve without a stray leading dot).
pub(crate) fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}
