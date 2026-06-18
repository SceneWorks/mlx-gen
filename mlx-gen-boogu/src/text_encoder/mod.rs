//! Boogu's **Qwen3-VL-8B-Instruct** condition encoder (text path; the vision tower is unused for
//! text-to-image). A 36-layer decoder-only LM whose **last_hidden_state** (all layers + final norm)
//! is the per-token `[B, L, 4096]` instruction features the DiT's caption embedder consumes.
//!
//! Reuses the same Qwen3-VL assembly as the ideogram crate over the shared `mlx-gen` core
//! primitives (`TextRope`, `TokenEmbedding`, `AdaptableLinear`, `rms_norm`, SDPA). GQA (32 query /
//! 8 kv heads), bias-less q/k/v/o, **per-head q/k RMSNorm**, HF half-split RoPE (θ = 5e6), SwiGLU
//! MLP, pre-norm residual blocks. The text-only path uses plain 1-D RoPE: Qwen3-VL's MRoPE sections
//! all index the same sequential text position when there are no image tokens.
//!
//! Boogu differs from the ideogram TE only in the head: it returns a **single** layer
//! (`last_hidden_state`) *with* the final norm applied, vs ideogram's 13-layer pre-final-norm concat.

pub mod attention;
pub mod encoder;
pub mod layer;
pub mod mlp;

pub use attention::Qwen3Attention;
pub use encoder::BooguTextEncoder;
pub use layer::Qwen3DecoderLayer;
pub use mlp::Qwen3Mlp;

// The HF half-split text RoPE is identical across families and lives in core.
pub use mlx_gen::nn::TextRope;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// Qwen3-VL-8B text-tower architecture (from `mllm/config.json` `text_config`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BooguTextEncoderConfig {
    pub num_layers: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl BooguTextEncoderConfig {
    pub fn qwen3_vl_8b() -> Self {
        Self {
            num_layers: 36,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
        }
    }
}

/// Load a bias-less Qwen3 projection from its `{base}.weight` `key`, auto-detecting a packed
/// snapshot. Every Qwen3 projection is a bias-less Linear (`attention_bias = false`).
pub(crate) fn lin(w: &Weights, key: &str) -> Result<AdaptableLinear> {
    let base = key.strip_suffix(".weight").unwrap_or(key);
    crate::quant::lin(w, base, false)
}

/// Join a module prefix with a leaf name, tolerating an empty prefix.
pub(crate) fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}
