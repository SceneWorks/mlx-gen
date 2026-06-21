//! Activation-peak control for the shared Wan-family video DiT forward (sc-5681).
//!
//! At the high-resolution/duration buckets a 14B Wan video DiT (SCAIL-2, Wan2.2-T2V/I2V, the Bernini
//! renderer) is **activation-bound**, not weight-bound — even with a pre-quantized Q4 snapshot
//! (sc-5445). These knobs bound the **DiT-denoise** per-step activation high-water:
//! 1. **Lazy-graph depth.** The whole 40-block forward is one lazy graph evaluated once per denoise
//!    step, so without intervention the peak is ~the **sum** of every block's transients rather than
//!    one block's; [`DitMemoryConfig::eval_per_block`] caps it at one block.
//! 2. **The FFN intermediate.** The gated-GELU FFN materializes a `[L, ffn_dim]` (≈`dim`×2.7) tensor,
//!    the largest single denoise transient; [`DitMemoryConfig::ffn_seq_chunk`] bounds it.
//! 3. **The self-attention score matrix.** *If* MLX's `scaled_dot_product_attention` falls back from
//!    flash to a materialized `[heads, L, L]` matrix (`heads · L² · 2 B`), one block's attention can
//!    dwarf everything; [`DitMemoryConfig::attn_query_chunk`] bounds it to `[heads, chunk, L]`. In
//!    practice SDPA *is* flash for the SCAIL-2 buckets measured (sc-5681), so this lever is off by
//!    default — kept for buckets/models where the fallback bites.
//!
//! IMPORTANT (sc-5681 measurement): for SCAIL-2 the 832×480/5s OOM was **not** the DiT denoise — that
//! fits. It was the **VAE decode** of the whole segment in one pass (fixed by tiling the decode, the
//! way the shared wan pipeline already does). These DiT levers are denoise *headroom* + the
//! generalizable shared-layer "practice" for Wan/Bernini, not the thing that unblocked 832×480.
//!
//! **Equivalence.** `eval_per_block` is exactly bit-identical (it only forces materialization). The
//! sequence-chunking levers are *numerically* equivalent, not bit-identical: the FFN / QKV projections
//! are per-token and attention softmax is per-query-row, so the math is unchanged, but MLX's Metal
//! GEMM/SDPA kernels are tile-specialized by the row (M) dimension, so a `[chunk, k]` matmul rounds
//! slightly differently from the full `[L, k]` one (cosine ≈ 1, max|Δ| ~1e-3 — the same class as the
//! model's own torch parity). The knobs default **off** (the historical whole-sequence behaviour) and
//! are opt-in per consumer + overridable from the environment.
//!
//! ## Consumers / rollout (sc-5681)
//! - **SCAIL-2** (`mlx-gen-scail2`) is the **only current consumer** (the driver + first validation
//!   target): its reimplemented blocks (`model.rs`) call [`map_seq_chunks`] + use [`DitMemoryConfig`]
//!   for the FFN + the attention query path and eval-to-free in the block loop; `generate.rs` opts
//!   into the production config. Bit-exactness vs. the un-chunked forward is gated by
//!   `mlx-gen-scail2/tests/dit_chunk_equiv.rs`.
//! - **(Prospective)** **Wan2.2-T2V / Bernini** share [`crate::transformer::WanTransformer`] (Bernini via
//!   `forward_packed`), so they inherit the same levers once wired — the integration points are the
//!   gated-GELU FFN in `transformer::Block::forward` (wrap in [`map_seq_chunks`]) and a per-block
//!   `eval` in `forward_with_modulation` / `forward_packed`, gated by a `WanTransformer` memory
//!   config. That wiring + its own high-resolution validation is the **sibling rollout** (epics 4699
//!   Bernini / 5594 Wan), deliberately not done here to avoid unvalidated changes to Wan's
//!   training/inference paths.

use mlx_gen::Result;
use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

/// Knobs that bound the per-step activation high-water of the shared Wan-family video DiT so the
/// existing high-resolution/duration buckets run without OOM (sc-5681). All bit-identical to the
/// un-chunked forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DitMemoryConfig {
    /// Run each block's FFN (`[L, ffn_dim]` intermediate) over sequence row-blocks of at most this
    /// many tokens. `None` ⇒ the whole sequence at once. A secondary transient (see the module doc).
    pub ffn_seq_chunk: Option<usize>,
    /// Run self-attention over query row-blocks of at most this many tokens, bounding the score
    /// matrix from `[heads, L, L]` to `[heads, chunk, L]` (K/V stay full, so the softmax is
    /// unchanged). `None` ⇒ the whole sequence at once. **The critical lever** at high resolution —
    /// MLX SDPA materializes the score matrix here (module doc).
    pub attn_query_chunk: Option<usize>,
    /// Force-evaluate (and free) each transformer block's output before starting the next, so the
    /// peak is ~one block's activations instead of the whole-depth lazy graph. Bit-exact.
    pub eval_per_block: bool,
}

impl DitMemoryConfig {
    /// No activation chunking — whole-sequence FFN/attention with one eval at the end of the step
    /// (the historical behaviour). Used by paths not yet validated under chunking; bit-identical to
    /// every other [`DitMemoryConfig`] in output.
    pub const OFF: Self = Self {
        ffn_seq_chunk: None,
        attn_query_chunk: None,
        eval_per_block: false,
    };

    /// `true` if no lever is active (the [`OFF`](Self::OFF) fast path — skip the chunk plumbing).
    pub fn is_off(&self) -> bool {
        self.ffn_seq_chunk.is_none() && self.attn_query_chunk.is_none() && !self.eval_per_block
    }

    /// Overlay the environment onto `base` so a deployment can tune the memory/throughput tradeoff
    /// without a recompile (the sc-5681 acceptance asks for steerable chunk sizes):
    ///   * `MLX_GEN_WAN_FFN_SEQ_CHUNK` — FFN sequence chunk (`0` disables; unset keeps `base`).
    ///   * `MLX_GEN_WAN_ATTN_QUERY_CHUNK` — attention query chunk (`0` disables; unset keeps `base`).
    ///   * `MLX_GEN_WAN_EVAL_PER_BLOCK` — `1`/`true`/`on` or `0`/`false`/`off` (unset keeps `base`).
    pub fn from_env(base: Self) -> Self {
        Self {
            ffn_seq_chunk: env_chunk("MLX_GEN_WAN_FFN_SEQ_CHUNK", base.ffn_seq_chunk),
            attn_query_chunk: env_chunk("MLX_GEN_WAN_ATTN_QUERY_CHUNK", base.attn_query_chunk),
            eval_per_block: env_bool("MLX_GEN_WAN_EVAL_PER_BLOCK", base.eval_per_block),
        }
    }
}

impl Default for DitMemoryConfig {
    fn default() -> Self {
        Self::OFF
    }
}

/// A `usize` chunk knob from `var`: a positive integer enables, `0` disables (`None`), anything else
/// (unset / unparseable) keeps `base`.
fn env_chunk(var: &str, base: Option<usize>) -> Option<usize> {
    match std::env::var(var) {
        Ok(s) => match s.trim().parse::<usize>() {
            Ok(0) => None,
            Ok(n) => Some(n),
            Err(_) => base,
        },
        Err(_) => base,
    }
}

/// A boolean knob from `var` (`1`/`true`/`on` vs `0`/`false`/`off`, case-insensitive); unset /
/// unrecognized keeps `base`.
fn env_bool(var: &str, base: bool) -> bool {
    match std::env::var(var) {
        Ok(s) => match s.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "on" | "yes" => true,
            "0" | "false" | "off" | "no" => false,
            _ => base,
        },
        Err(_) => base,
    }
}

/// Map a per-token function `f` over sequence row-blocks of `x` `[B, L, *]` and concatenate the
/// results back along the sequence axis. For chunk `i`, `f` receives the contiguous slice
/// `x[:, start:start+len, :]` (`len ≤ chunk`) **and** that block's `start` offset, so a caller can
/// slice position-dependent side inputs (RoPE cos/sin) to match the query block.
///
/// `chunk` `None` / `0` / `≥ L` runs a single `f(x, 0)` — the no-op fast path. Bit-identical to
/// `f(&x, 0)` for any per-token `f`: no op here reduces across the sequence axis and
/// `concatenate(split(x)) == x`.
pub fn map_seq_chunks<F>(x: &Array, chunk: Option<usize>, mut f: F) -> Result<Array>
where
    F: FnMut(&Array, i32) -> Result<Array>,
{
    let l = x.shape()[1] as usize;
    let c = match chunk {
        Some(c) if c > 0 && c < l => c,
        _ => return f(x, 0),
    };
    let mut outs: Vec<Array> = Vec::with_capacity(l.div_ceil(c));
    let mut start = 0usize;
    while start < l {
        let len = c.min(l - start);
        let part = slice_seq(x, start as i32, len as i32)?;
        outs.push(f(&part, start as i32)?);
        start += len;
    }
    let refs: Vec<&Array> = outs.iter().collect();
    Ok(concatenate_axis(&refs, 1)?)
}

/// Contiguous `[:, start:start+len, …]` slice along the sequence axis (axis 1), via the arange-index
/// `take_axis` idiom used throughout this crate.
fn slice_seq(x: &Array, start: i32, len: i32) -> Result<Array> {
    let idx = Array::from_slice(&(start..start + len).collect::<Vec<i32>>(), &[len]);
    Ok(x.take_axis(&idx, 1)?)
}

/// Contiguous `[start:start+len, …]` slice along axis 0 — for slicing the RoPE `cos`/`sin`
/// `[seq_len, 1, half_d]` tables to a query block inside [`map_seq_chunks`].
pub fn slice_axis0(x: &Array, start: i32, len: i32) -> Result<Array> {
    let idx = Array::from_slice(&(start..start + len).collect::<Vec<i32>>(), &[len]);
    Ok(x.take_axis(&idx, 0)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Dtype;

    fn flat(a: &Array) -> Vec<f32> {
        a.reshape(&[-1])
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec()
    }

    #[test]
    fn off_config_is_off() {
        assert!(DitMemoryConfig::OFF.is_off());
        assert!(DitMemoryConfig::default().is_off());
        assert!(!DitMemoryConfig {
            eval_per_block: true,
            ..DitMemoryConfig::OFF
        }
        .is_off());
    }

    #[test]
    fn map_seq_chunks_is_bit_identical() {
        // [B=2, L=37, D=5] so the last block is a ragged remainder for several chunk sizes.
        let l = 37;
        let d = 5;
        let n = 2 * l * d;
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.013 - 1.7).collect();
        let x = Array::from_slice(&data, &[2, l, d]);

        // A non-trivial per-token op that also reads `start` (position-dependent, like RoPE): scale
        // each row by a function of its absolute sequence index. Still per-token ⇒ chunk-invariant.
        let apply = |chunk: Option<usize>| -> Array {
            map_seq_chunks(&x, chunk, |part, start| {
                let len = part.shape()[1];
                let idx: Vec<f32> = (0..len).map(|i| 1.0 + 0.5 * (start + i) as f32).collect();
                let w = Array::from_slice(&idx, &[1, len, 1]);
                Ok(mlx_rs::ops::multiply(part, &w)?)
            })
            .unwrap()
        };

        let full = apply(None);
        for chunk in [Some(1), Some(7), Some(16), Some(37), Some(100)] {
            let chunked = apply(chunk);
            assert_eq!(chunked.shape(), full.shape(), "chunk {chunk:?} shape");
            let (fa, fb) = (flat(&full), flat(&chunked));
            let max_abs = fa
                .iter()
                .zip(&fb)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert_eq!(
                max_abs, 0.0,
                "chunk {chunk:?} not bit-identical (max|Δ| {max_abs})"
            );
        }
    }

    #[test]
    fn env_helpers_parse() {
        assert_eq!(env_chunk("definitely_unset_var_xyz", Some(99)), Some(99));
        assert!(env_bool("definitely_unset_var_xyz", true));
        assert!(!env_bool("definitely_unset_var_xyz", false));
    }
}
