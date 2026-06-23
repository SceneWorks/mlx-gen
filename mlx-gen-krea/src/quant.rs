//! Quantization-group helpers for the Krea DiT — a thin wrapper that pins the Krea group size onto
//! the shared [`mlx_gen::quant::lin`]. Every quantizable projection in the DiT (and the Qwen3-VL text
//! tower, sc-7569) loads through here so the load path can't drift from [`crate::convert`]'s packer.

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::TokenEmbedding;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// Group size for every Krea group-wise-affine quantization (pack + load) — the codebase default
/// **64**. Every Krea quant-target Linear has an input dim divisible by 64 (DiT hidden 6144, FFN
/// 16384, text 2560, text-FFN 6912; Qwen3-VL TE 4096/1024/9728), so one group size packs the whole
/// model (unlike Boogu's 3360 hidden, which forced 32). Mirrors [`crate::convert::QUANT_GROUP_SIZE`].
pub(crate) const GROUP_SIZE: i32 = 64;

/// Load `{base}` as an [`AdaptableLinear`] at the Krea [`GROUP_SIZE`] — **packed** (Q4/Q8) when
/// `{base}.scales` is present (a [`crate::convert::assemble_quantized_snapshot`] turnkey), else
/// **dense**. `bias` additionally loads the dense `{base}.bias`. One loader serves the dense bf16 and
/// the pre-quantized snapshot identically.
pub(crate) fn lin(w: &Weights, base: &str, bias: bool) -> Result<AdaptableLinear> {
    mlx_gen::quant::lin(w, base, bias, GROUP_SIZE)
}

/// Load `{base}` as a [`TokenEmbedding`] at the Krea [`GROUP_SIZE`] — packed when `{base}.scales` is
/// present, else dense. The Qwen3-VL text-encoder token table (sc-7569).
pub(crate) fn embedding(w: &Weights, base: &str) -> Result<TokenEmbedding> {
    mlx_gen::quant::embedding(w, base, GROUP_SIZE)
}
