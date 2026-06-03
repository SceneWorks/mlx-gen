//! FLUX.2's **Qwen3** text encoder — a 36-layer decoder-only LM whose intermediate hidden states
//! (layers 9, 18, 27) are concatenated into the transformer's `prompt_embeds`. Port of the fork's
//! `Qwen3TextEncoder` (`models/flux2/model/flux2_text_encoder/`) + the shared
//! `Qwen3VLDecoderLayer` (`models/common_models/qwen3_vl/`).
//!
//! Qwen3 vs the Qwen2.5-VL encoder in `mlx-gen-qwen-image`: **no q/k/v/o biases**
//! (`attention_bias=False`), **per-head q/k RMSNorm** on the head dim, and the prompt path
//! extracts **multiple intermediate layers** (no final norm) rather than the last normed hidden
//! state. GQA (32 query / 8 kv heads), HF half-split RoPE (θ=1e6), SwiGLU MLP, pre-norm residual
//! blocks. The text-only path uses plain 1-D RoPE (`mrope_section=None`).

pub mod attention;
pub mod encoder;
pub mod layer;
pub mod mlp;

pub use attention::Qwen3Attention;
pub use encoder::{Qwen3TextEncoder, Qwen3TextEncoderConfig};
pub use layer::Qwen3DecoderLayer;
pub use mlp::Qwen3Mlp;

// The HF half-split text RoPE is identical across families and lives in core (F-006).
pub use mlx_gen::nn::TextRope;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// Wrap a stored `[out, in]` weight as a bias-less dense [`AdaptableLinear`] — every Qwen3
/// projection is a bias-less Linear. Dense forward = `matmul(x, wᵀ)`; `quantize` swaps the base to
/// a Q4/Q8 `quantized_matmul`, the mlx-rs equivalent of the fork's `nn.quantize` over the TE.
pub(crate) fn lin(w: &Weights, key: &str) -> Result<AdaptableLinear> {
    Ok(AdaptableLinear::dense(w.require(key)?.clone(), None))
}

/// Join a module prefix with a leaf name, tolerating an empty prefix.
pub(crate) fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}
