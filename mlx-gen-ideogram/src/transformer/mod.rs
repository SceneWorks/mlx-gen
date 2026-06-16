//! Ideogram 4's single-stream flow-matching DiT (`Ideogram4Transformer2DModel`): 34 layers over
//! one concatenated `[text ; image]` token sequence, AdaLN-modulated per block by the
//! flow-matching timestep, with interleaved 3D MRoPE and full (bidirectional, segment-masked)
//! attention. Port of upstream `modeling_ideogram4.py`.
//!
//! Built fresh on the shared `mlx-gen` core primitives (`AdaptableLinear`, `TokenEmbedding`,
//! `rms_norm`, SDPA) — it is a distinct architecture from flux2's MMDiT. Dense-only; the model is
//! instantiated twice (conditional + unconditional) for asymmetric CFG (pipeline, E5).

pub mod block;
pub mod model;
pub mod mrope;

pub use block::Ideogram4Block;
pub use model::Ideogram4Transformer;
pub use mrope::Ideogram4MRoPE;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// Build a Linear from `{base}.weight` (+ optional `{base}.bias`). Ideogram's DiT projections are
/// biased (input_proj / llm_cond_proj / adaln_* / t_embedding / final_layer) except the per-block
/// attention `qkv`/`o` and the SwiGLU `w1`/`w2`/`w3`, which are bias-less.
pub(crate) fn lin(w: &Weights, base: &str, bias: bool) -> Result<AdaptableLinear> {
    let weight = w.require(&format!("{base}.weight"))?.clone();
    let b = if bias {
        Some(w.require(&format!("{base}.bias"))?.clone())
    } else {
        None
    };
    Ok(AdaptableLinear::dense(weight, b))
}

/// Join a module prefix with a leaf name, tolerating an empty prefix (the DiT keys are top-level).
pub(crate) fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}
