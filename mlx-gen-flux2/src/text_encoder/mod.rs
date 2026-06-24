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
pub mod generate;
pub mod layer;
pub mod mlp;

pub use attention::Qwen3Attention;
pub use encoder::{Qwen3TextEncoder, Qwen3TextEncoderConfig};
pub use generate::UpsampleSampling;
pub use layer::Qwen3DecoderLayer;
pub use mlp::Qwen3Mlp;

// The HF half-split text RoPE is identical across families and lives in core (F-006).
pub use mlx_gen::nn::TextRope;
// Dotted-key assembly is the shared `mlx_gen::weights::join` (re-exported so submodules keep
// addressing it as `super::join`).
pub(crate) use mlx_gen::weights::join;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::Flux2Quant;

/// Wrap a stored `[out, in]` weight as a bias-less [`AdaptableLinear`] — every Qwen3/Mistral
/// projection is a bias-less Linear. With `quant == None` this is the dense path (`matmul(x, wᵀ)`);
/// `quantize` later swaps the base to a Q4/Q8 `quantized_matmul`, the mlx-rs equivalent of the
/// fork's `nn.quantize` over the TE.
///
/// With `quant == Some` AND this Linear's packed `{base}.scales` present on disk — a
/// **pre-quantized snapshot** (sc-5917) — build the quantized base directly from the packed
/// `{base}.weight` (u32 codes) / `.scales` / `.biases`, materializing no dense bf16 weight (the
/// dev Mistral TE is ~45 GB bf16; this is what keeps it off the load-time memory floor). `key` is
/// the `….weight` tensor name.
pub(crate) fn lin(w: &Weights, key: &str, quant: Option<Flux2Quant>) -> Result<AdaptableLinear> {
    if let Some(q) = quant {
        let base = key.strip_suffix(".weight").unwrap_or(key);
        if let Some(scales) = w.get(&format!("{base}.scales")) {
            return Ok(AdaptableLinear::from_quantized_parts(
                w.require(key)?.clone(),
                scales.clone(),
                w.require(&format!("{base}.biases"))?.clone(),
                None,
                q.group_size,
                q.bits,
            ));
        }
    }
    Ok(AdaptableLinear::dense(w.require(key)?.clone(), None))
}
