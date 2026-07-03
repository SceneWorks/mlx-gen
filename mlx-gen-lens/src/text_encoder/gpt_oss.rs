//! gpt-oss-20b attention core (sc-3165): GQA + learned **attention sinks** + alternating
//! sliding/full causal masks + **YaRN RoPE** + RMSNorm — a faithful port of
//! `transformers.models.gpt_oss.modeling_gpt_oss` (`GptOssAttention` / `eager_attention_forward` /
//! `GptOssRotaryEmbedding`).
//!
//! ## Parity-critical details (from the reference)
//! - **RoPE is NeoX "half-split"** (`_apply_rotary_emb` chunks the head_dim in two; cos/sin have
//!   length `head_dim/2`) with the YaRN `attention_scaling` folded into cos/sin. mlx
//!   `fast::rope` does **not** reproduce this layout with custom `freqs` (verified: both
//!   `traditional` settings diverge ~1.7), so the rotation is applied explicitly here — cheap, since
//!   the encoder runs a single short forward.
//! - **Attention sinks**: per-head learnable logit appended as an extra softmax column, then dropped
//!   after the softmax. The reference subtracts the row-wise max *over the combined scores+sink* for
//!   bf16 stability; we reproduce that exactly with an explicit `−max` / exp / denominator softmax
//!   (`softmax([scores, sink])[..., :L]` ≡ `exp(scores−m) / (Σ exp(scores−m) + exp(sink−m))`).
//! - **No q/k-norm** (unlike Gemma). attention scale = `head_dim^-0.5`. Projections **carry biases**.
//! - **GQA**: 64 query heads over 8 KV heads (`repeat_kv`, n_rep = 8).
//!
//! The MoE feed-forward + decoder-layer/residual assembly is sc-3166; this module is the attention
//! sub-block only (it consumes an already-RMSNorm'd hidden state, exactly like the reference
//! `GptOssAttention.forward`).

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::indexing::argmax_axis;
use mlx_rs::ops::{
    add, argsort_axis, broadcast_to, concatenate_axis, cos as cos_op, divide, floor_divide,
    gather_mm, gather_qmm, matmul, max_axes, maximum, minimum, multiply, quantize, sigmoid,
    sin as sin_op, softmax_axis, split, split_sections, subtract, sum_axes,
};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Quant, Result};

use crate::config::GptOssConfig;
use crate::text_encoder::mxfp4::dequantize_mxfp4;

/// A scalar `[1]` array for broadcasting multiplies.
fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// `y = x · Wᵀ + b` for a stored `[out, in]` weight and `[out]` bias (the gpt-oss attention
/// projections all have biases — `attention_bias: true`).
struct LinearBias {
    w: Array, // [out, in]
    b: Array, // [out]
}

impl LinearBias {
    fn load(w: &Weights, key: &str, dtype: Dtype) -> Result<Self> {
        Ok(Self {
            w: w.require(&format!("{key}.weight"))?.as_dtype(dtype)?,
            b: w.require(&format!("{key}.bias"))?.as_dtype(dtype)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        Ok(add(&matmul(x, self.w.t())?, &self.b)?)
    }
}

/// One gpt-oss decoder layer's attention (`self_attn`). Consumes the RMSNorm'd hidden state and
/// returns the attention output *before* the residual add (matching `GptOssAttention.forward`).
pub struct GptOssAttention {
    q_proj: LinearBias,
    k_proj: LinearBias,
    v_proj: LinearBias,
    o_proj: LinearBias,
    /// Per-head sink logits, `[num_heads]`.
    sinks: Array,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl GptOssAttention {
    /// Load `self_attn` at `{prefix}` (e.g. `model.layers.0.self_attn`) at `dtype` (bf16 production /
    /// f32 for the correctness gate). The attention weights are dense in the checkpoint
    /// (`modules_to_not_convert` keeps `self_attn` out of MXFP4).
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: &GptOssConfig,
        dtype: Dtype,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: LinearBias::load(w, &format!("{prefix}.q_proj"), dtype)?,
            k_proj: LinearBias::load(w, &format!("{prefix}.k_proj"), dtype)?,
            v_proj: LinearBias::load(w, &format!("{prefix}.v_proj"), dtype)?,
            o_proj: LinearBias::load(w, &format!("{prefix}.o_proj"), dtype)?,
            sinks: w.require(&format!("{prefix}.sinks"))?.as_dtype(dtype)?,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            scale: (cfg.head_dim as f32).powf(-0.5),
        })
    }

    /// `x`: `[B, L, hidden]` RMSNorm'd hidden state. `inv_freq`: the YaRN frequencies `[head_dim/2]`.
    /// `attn_scaling`: the YaRN mscale. `mask`: additive `[1, 1, L, L]` (or broadcastable) causal /
    /// sliding mask. Returns `[B, L, hidden]`.
    pub fn forward(
        &self,
        x: &Array,
        inv_freq: &Array,
        attn_scaling: f32,
        mask: &Array,
    ) -> Result<Array> {
        let sh = x.shape();
        let (b, l) = (sh[0], sh[1]);
        let (h, kv, d) = (self.num_heads, self.num_kv_heads, self.head_dim);

        let q = self
            .q_proj
            .forward(x)?
            .reshape(&[b, l, h, d])?
            .transpose_axes(&[0, 2, 1, 3])?; // [B,H,L,d]
        let k = self
            .k_proj
            .forward(x)?
            .reshape(&[b, l, kv, d])?
            .transpose_axes(&[0, 2, 1, 3])?; // [B,kv,L,d]
        let v = self
            .v_proj
            .forward(x)?
            .reshape(&[b, l, kv, d])?
            .transpose_axes(&[0, 2, 1, 3])?; // [B,kv,L,d]

        // RoPE: the reference uses a NeoX **half-split** rotation (`_apply_rotary_emb` chunks the
        // head_dim in two; cos/sin have length head_dim/2) with the YaRN `attention_scaling` folded
        // into cos/sin. mlx `fast::rope` does not reproduce this layout with custom `freqs`, so we
        // apply it explicitly (cheap: encoder-only, short sequence).
        let (cos, sin) = yarn_cos_sin(l, inv_freq, attn_scaling, x.dtype())?;
        let q = apply_half_rope(&q, &cos, &sin)?;
        let k = apply_half_rope(&k, &cos, &sin)?;

        // GQA: repeat K/V from `kv` heads to `h` heads (n_rep = h/kv).
        let k = repeat_kv(&k, h)?; // [B,H,L,d]
        let v = repeat_kv(&v, h)?; // [B,H,L,d]

        // scores = (q·kᵀ)·scale + mask   → [B,H,L,L]
        let scores = multiply(
            &matmul(&q, &k.transpose_axes(&[0, 1, 3, 2])?)?,
            scalar(self.scale),
        )?;
        let scores = add(&scores, mask)?;

        // Sink column: sinks[h] → [1,H,1,1] → broadcast [B,H,L,1].
        let sink = broadcast_to(&self.sinks.reshape(&[1, h, 1, 1])?, &[b, h, l, 1])?;

        // Softmax over [scores, sink] with the reference's −(row-max incl. sink) stabilization, then
        // drop the sink column: probs = exp(scores−m) / (Σ exp(scores−m) + exp(sink−m)).
        let row_max = max_axes(&scores, &[-1], true)?; // [B,H,L,1]
        let m = maximum(&row_max, &sink)?; // [B,H,L,1]
        let exp_scores = subtract(&scores, &m)?.exp()?; // [B,H,L,L]
        let exp_sink = subtract(&sink, &m)?.exp()?; // [B,H,L,1]
        let denom = add(&sum_axes(&exp_scores, &[-1], true)?, &exp_sink)?; // [B,H,L,1]
        let probs = divide(&exp_scores, &denom)?; // [B,H,L,L]

        let out = matmul(&probs, &v)?; // [B,H,L,d]
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, l, h * d])?;
        self.o_proj.forward(&out)
    }

    /// Incremental (cached) attention for autoregressive generation (sc-3176). Processes the `T` new
    /// tokens in `x` `[B, T, hidden]` at **absolute** positions `position..position+T`, appends their
    /// (post-RoPE) K / (pre-repeat) V to `cache`, attends the new queries over the whole cache, then —
    /// for a sliding-window layer — evicts the cache to the last `window` keys for the next step.
    /// `mask` is the additive `[1, 1, T, cache_len]` causal(+sliding) mask for the prefill (`T > 1`);
    /// for a single decode token (`T == 1`) every cached key is valid, so `mask` is `None`.
    ///
    /// `position` is the **true** sequence offset of the first new token, passed explicitly because a
    /// sliding layer's `cache.len()` is capped at `window` and so does *not* track the absolute
    /// position — the RoPE rotation must use the real position (a bug if derived from `cache.len()`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_cached(
        &self,
        x: &Array,
        inv_freq: &Array,
        attn_scaling: f32,
        position: i32,
        cache: &mut KvCache,
        sliding_window: Option<i32>,
        mask: Option<&Array>,
    ) -> Result<Array> {
        let sh = x.shape();
        let (b, t) = (sh[0], sh[1]);
        let (h, kv, d) = (self.num_heads, self.num_kv_heads, self.head_dim);
        let past = position;

        let q = self
            .q_proj
            .forward(x)?
            .reshape(&[b, t, h, d])?
            .transpose_axes(&[0, 2, 1, 3])?; // [B,H,T,d]
        let k = self
            .k_proj
            .forward(x)?
            .reshape(&[b, t, kv, d])?
            .transpose_axes(&[0, 2, 1, 3])?; // [B,kv,T,d]
        let v = self
            .v_proj
            .forward(x)?
            .reshape(&[b, t, kv, d])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // RoPE the new q/k at their absolute positions; cache the post-RoPE K so relative rotations
        // stay correct under sliding-window eviction.
        let (cos, sin) = yarn_cos_sin_at(past, t, inv_freq, attn_scaling, x.dtype())?;
        let q = apply_half_rope(&q, &cos, &sin)?;
        let k = apply_half_rope(&k, &cos, &sin)?;

        cache.append(&k, &v)?;
        // Sliding window: a **decode** query (`T == 1`) attends to exactly the last `window` keys
        // (positions `p-window+1..=p`), so evict the stale key BEFORE attending — appending the new
        // key made the cache `window+1`. (A `T > 1` prefill instead carries the sliding **mask** over
        // the full prompt, so it evicts *after* attending, leaving the window primed for the next
        // step.) This keeps the cached decode bit-identical to a masked full recompute.
        let prefill = t > 1;
        if !prefill {
            if let Some(w) = sliding_window {
                cache.truncate_last(w)?;
            }
        }
        let k_all = repeat_kv(cache.k.as_ref().unwrap(), h)?; // [B,H,cache_len,d]
        let v_all = repeat_kv(cache.v.as_ref().unwrap(), h)?;

        let mut scores = multiply(
            &matmul(&q, &k_all.transpose_axes(&[0, 1, 3, 2])?)?,
            scalar(self.scale),
        )?; // [B,H,T,cache_len]
        if let Some(m) = mask {
            scores = add(&scores, m)?;
        }
        let sink = broadcast_to(&self.sinks.reshape(&[1, h, 1, 1])?, &[b, h, t, 1])?;
        let row_max = max_axes(&scores, &[-1], true)?;
        let m = maximum(&row_max, &sink)?;
        let exp_scores = subtract(&scores, &m)?.exp()?;
        let exp_sink = subtract(&sink, &m)?.exp()?;
        let denom = add(&sum_axes(&exp_scores, &[-1], true)?, &exp_sink)?;
        let probs = divide(&exp_scores, &denom)?;

        let out = matmul(&probs, &v_all)?; // [B,H,T,d]
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, t, h * d])?;

        // Prefill: prime the sliding cache to the last `window` keys for the next (decode) step.
        if prefill {
            if let Some(w) = sliding_window {
                cache.truncate_last(w)?;
            }
        }
        self.o_proj.forward(&out)
    }
}

/// A per-layer key/value cache for incremental decode (sc-3176). Stores the **post-RoPE K** and **V**
/// at `[B, kv_heads, seq, head_dim]` (pre-`repeat_kv`); a sliding-window layer truncates to the last
/// `window` after each step.
#[derive(Default)]
pub struct KvCache {
    k: Option<Array>,
    v: Option<Array>,
}

impl KvCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of cached key positions so far.
    pub fn len(&self) -> i32 {
        self.k.as_ref().map(|k| k.shape()[2]).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append the new `[B, kv, T, d]` K / V and return the full cached `(K, V)`.
    fn append(&mut self, k: &Array, v: &Array) -> Result<(Array, Array)> {
        let k_all = match &self.k {
            Some(prev) => concatenate_axis(&[prev, k], 2)?,
            None => k.clone(),
        };
        let v_all = match &self.v {
            Some(prev) => concatenate_axis(&[prev, v], 2)?,
            None => v.clone(),
        };
        self.k = Some(k_all.clone());
        self.v = Some(v_all.clone());
        Ok((k_all, v_all))
    }

    /// Keep only the last `max` key positions (sliding-window eviction).
    fn truncate_last(&mut self, max: i32) -> Result<()> {
        let len = self.len();
        if len <= max {
            return Ok(());
        }
        // `[:, :, len-max:, :]` — split at `len-max` along the sequence axis, keep the tail.
        let tail =
            |a: &Array| -> Result<Array> { Ok(split_sections(a, &[len - max], 2)?[1].clone()) };
        self.k = self.k.as_ref().map(tail).transpose()?;
        self.v = self.v.as_ref().map(tail).transpose()?;
        Ok(())
    }
}

/// Build the YaRN RoPE `cos`/`sin` for positions `0..l`, each `[1, 1, l, head_dim/2]`, with the
/// `attention_scaling` (mscale) folded in (`cos = cos(p·inv_freq)·scaling`), matching
/// `GptOssRotaryEmbedding.forward`. Cast to `dtype` so they multiply cleanly against q/k.
fn yarn_cos_sin(l: i32, inv_freq: &Array, scaling: f32, dtype: Dtype) -> Result<(Array, Array)> {
    yarn_cos_sin_at(0, l, inv_freq, scaling, dtype)
}

/// As [`yarn_cos_sin`] but for the absolute positions `start..start+l` — used by the incremental
/// decode path (sc-3176), where the `l` new tokens sit at positions offset by the cache length.
fn yarn_cos_sin_at(
    start: i32,
    l: i32,
    inv_freq: &Array,
    scaling: f32,
    dtype: Dtype,
) -> Result<(Array, Array)> {
    let half = inv_freq.shape()[0];
    let pos: Vec<f32> = (start..start + l).map(|i| i as f32).collect();
    let pos = Array::from_slice(&pos, &[l, 1]);
    let freqs = multiply(&pos, &inv_freq.reshape(&[1, half])?)?; // [l, half]
    let s = scalar(scaling);
    let cos = multiply(&cos_op(&freqs)?, &s)?
        .reshape(&[1, 1, l, half])?
        .as_dtype(dtype)?;
    let sin = multiply(&sin_op(&freqs)?, &s)?
        .reshape(&[1, 1, l, half])?
        .as_dtype(dtype)?;
    Ok((cos, sin))
}

/// Apply the NeoX half-split rotation to `[B, H, L, d]` given `cos`/`sin` `[1, 1, L, d/2]`:
/// `out = cat(first·cos − second·sin, second·cos + first·sin)` where `first`/`second` are the two
/// halves of the head dim. Bit-identical to `transformers`' `_apply_rotary_emb`.
fn apply_half_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let parts = split(x, 2, -1)?;
    let (first, second) = (&parts[0], &parts[1]);
    let out_first = subtract(&multiply(first, cos)?, &multiply(second, sin)?)?;
    let out_second = add(&multiply(second, cos)?, &multiply(first, sin)?)?;
    Ok(concatenate_axis(&[out_first, out_second], -1)?)
}

/// `repeat_kv`: expand `[B, kv, L, d]` to `[B, H, L, d]` by repeat-interleaving each KV head
/// `H/kv` times (matching `transformers.repeat_kv`).
fn repeat_kv(x: &Array, num_heads: i32) -> Result<Array> {
    let sh = x.shape();
    let (b, kv, l, d) = (sh[0], sh[1], sh[2], sh[3]);
    if kv == num_heads {
        return Ok(x.clone());
    }
    let n_rep = num_heads / kv;
    let expanded = broadcast_to(&x.reshape(&[b, kv, 1, l, d])?, &[b, kv, n_rep, l, d])?;
    Ok(expanded.reshape(&[b, num_heads, l, d])?)
}

/// Build the additive attention mask `[1, 1, L, L]` for a single un-padded sequence: causal, and —
/// for sliding-window (local) layers — additionally masking keys older than `window` (`i − j ≥
/// window`). Matches `create_causal_mask` / `create_sliding_window_causal_mask` for the no-padding
/// case the Lens encoder runs.
pub fn attention_mask(l: i32, sliding_window: Option<i32>, dtype: Dtype) -> Result<Array> {
    let l = l as usize;
    let neg = f32::MIN / 2.0;
    let mut data = vec![0f32; l * l];
    for i in 0..l {
        for j in 0..l {
            let causal_ok = j <= i;
            let window_ok = match sliding_window {
                Some(w) => (i as i64 - j as i64) < w as i64,
                None => true,
            };
            data[i * l + j] = if causal_ok && window_ok { 0.0 } else { neg };
        }
    }
    Array::from_slice(&data, &[1, 1, l as i32, l as i32])
        .as_dtype(dtype)
        .map_err(Error::from)
}

// =====================================================================================================
// MoE feed-forward + decoder-layer assembly (sc-3166)
// =====================================================================================================

/// One expert projection during **offline (pre-)quantization** — either the dense
/// MXFP4-dequantized weight (`[in, out]`, the eager `GptOssExperts` `x · w` layout) or an
/// MLX-quantized (Q4/Q8) pack of `wᵀ` (`[out, in]`). Retained as the per-expert staging type for
/// [`prequantize_expert_proj`] / [`crate::convert`]; the *forward* path no longer uses it — the MoE
/// now runs a grouped GEMM over the stacked [`ExpertBank`] (F-021, sc-9500).
enum Proj {
    /// `w`: `[in, out]`; the eager `GptOssExperts` `x · w` layout.
    Dense { w: Array, b: Array },
    /// MLX group-wise affine pack of `wᵀ` (`[out, in]`) + dense `[out]` bias.
    Quant {
        wq: Array,
        scales: Array,
        biases: Array,
        b: Array,
    },
}

impl Proj {
    /// Quantize a dense proj to `bits`-bit MLX affine (group `group_size`). The dense `w` is
    /// `[in, out]`; MLX `quantize` expects `[out, in]` (it groups along the last/`in` axis), so
    /// transpose first. The weight + bias are cast to bf16 before packing — the fork-parity
    /// convention shared with [`AdaptableLinear::quantize`] (`gather_qmm` accumulates in fp32
    /// regardless). No-op if already quantized.
    fn into_quantized(self, bits: i32, group_size: i32) -> Result<Self> {
        match self {
            Proj::Dense { w, b } => {
                let w_oi = w.t().as_dtype(Dtype::Bfloat16)?; // [out, in]
                let (wq, scales, biases) = quantize(&w_oi, group_size, bits)?;
                Ok(Proj::Quant {
                    wq,
                    scales,
                    biases,
                    b: b.as_dtype(Dtype::Bfloat16)?,
                })
            }
            q => Ok(q),
        }
    }
}

/// The stacked packed triple + dense bias for one MoE expert projection across all `E` experts —
/// the on-disk representation an offline pre-quantized turnkey stores (sc-8763,
/// [`crate::convert`]). `weight`/`scales`/`biases` are `[E, out, …]`; `bias` is bf16 `[E, out]`.
pub(crate) struct StackedExpertPack {
    pub weight: Array,
    pub scales: Array,
    pub biases: Array,
    pub bias: Array,
}

/// Offline pre-quantize one MoE expert projection (all `E` experts) from its MXFP4 source
/// (`blocks` `[E, out, G, 16]` + `scales_u8` `[E, out, G]` + `bias` `[E, out]`) to `bits`-bit MLX
/// affine (group 64), returning the **stacked** packed triple (sc-8763). Reuses the exact load-time
/// path — [`dequantize_mxfp4`] then [`Proj::into_quantized`] per expert — so the pack is
/// byte-identical to what [`GptOssMoe::from_weights`]'s dense-then-quantize branch produces. The
/// per-expert packs are re-stacked along a fresh leading E axis (`expand_dims(0)` + `concat`), which
/// [`crate::quant::load_packed_experts`] splits back — valid because affine quant is per-row, so the
/// stack/split around the per-expert quantize commute.
pub(crate) fn prequantize_expert_proj(
    blocks: &Array,
    scales_u8: &Array,
    bias: &Array,
    bits: i32,
    group_size: i32,
) -> Result<StackedExpertPack> {
    // Dequantize MXFP4 → dense `[E, in, out]` (bf16 — the dtype the load path dequants to before
    // `into_quantized` re-casts to bf16 anyway; f32 would give identical packs after the bf16 cast).
    let dense = dequantize_mxfp4(blocks, scales_u8, Dtype::Bfloat16)?; // [E, in, out]
    let e = dense.shape()[0];
    let per_expert = split(&dense, e, 0)?; // E × [1, in, out]
    let bias_e = split(bias, e, 0)?; // E × [1, out]

    let mut wq_stack: Vec<Array> = Vec::with_capacity(e as usize);
    let mut sc_stack: Vec<Array> = Vec::with_capacity(e as usize);
    let mut bi_stack: Vec<Array> = Vec::with_capacity(e as usize);
    let mut bs_stack: Vec<Array> = Vec::with_capacity(e as usize);
    for i in 0..e as usize {
        let (in_c, out_c) = (per_expert[i].shape()[1], per_expert[i].shape()[2]);
        // Build the exact `Proj::Dense` the load path builds (`[in, out]` weight + `[out]` bias),
        // then run the identical `into_quantized` (transpose → bf16 → quantize group 64).
        let proj = Proj::Dense {
            w: per_expert[i].reshape(&[in_c, out_c])?,
            b: bias_e[i].reshape(&[out_c])?,
        };
        match proj.into_quantized(bits, group_size)? {
            Proj::Quant {
                wq,
                scales,
                biases,
                b,
            } => {
                // Re-add the leading E axis for stacking.
                wq_stack.push(wq.expand_dims(0)?);
                sc_stack.push(scales.expand_dims(0)?);
                bi_stack.push(biases.expand_dims(0)?);
                bs_stack.push(b.expand_dims(0)?);
            }
            Proj::Dense { .. } => unreachable!("into_quantized always yields Quant"),
        }
    }
    let stack = |v: &[Array]| -> Result<Array> {
        let refs: Vec<&Array> = v.iter().collect();
        Ok(concatenate_axis(&refs, 0)?)
    };
    let pack = StackedExpertPack {
        weight: stack(&wq_stack)?,
        scales: stack(&sc_stack)?,
        biases: stack(&bi_stack)?,
        bias: stack(&bs_stack)?,
    };
    // Materialize so the dense bf16 dequant transient frees before the next layer (the converter
    // packs one layer at a time; a live lazy graph would keep the whole 20 B bf16 stack resident).
    mlx_rs::transforms::eval([&pack.weight, &pack.scales, &pack.biases, &pack.bias])?;
    Ok(pack)
}

/// Token count (`n·k`) at/above which the MoE forward switches to the expert-sorted gather (mlx-lm's
/// `SwitchGLU` uses the same `indices.size >= 64`): below it the sort overhead outweighs the win and
/// the direct broadcast gather is faster (measured crossover ≈ n·k a few hundred; 64 leaves margin).
const GATHER_SORT_THRESHOLD: i64 = 64;

/// One MoE expert projection's Q4/Q8 pack across all `E` experts, stored **stacked** for
/// `gather_qmm` (F-021): `wq [E, out, in·bits/32]`, `scales`/`biases [E, out, in/gs]`, `b [E, out]`.
struct StackedQuant {
    wq: Array,
    scales: Array,
    biases: Array,
    b: Array,
    group_size: i32,
    bits: i32,
}

impl StackedQuant {
    /// `gather_qmm` (`transpose = true` recovers `x·wᵀ` from the `[out, in]` pack) + the gathered
    /// bias. `x`'s batch dims broadcast against `idx`; `sorted` sets `gather_qmm`'s sorted-index
    /// fast path (valid only when `idx` is the flattened, expert-sorted routing, see [`GptOssMoe`]).
    fn forward(&self, x: &Array, idx: &Array, sorted: bool) -> Result<Array> {
        let y = gather_qmm(
            x,
            &self.wq,
            &self.scales,
            &self.biases,
            None,
            idx,
            true,
            self.group_size,
            self.bits,
            sorted,
        )?;
        let bias = self.b.take_axis(idx, 0)?.expand_dims(-2)?; // [.., 1, out]
        Ok(add(&y, &bias)?)
    }
}

/// The MoE expert bank, stored **stacked** `[E, …]` for grouped-GEMM routing (F-021) — dense bf16
/// (the dequantized-MXFP4 encoder path) or MLX group-wise affine Q4/Q8 (the ~12 GB path, sc-3172).
/// Replaces the former per-expert `Vec<Expert>` + all-32-dense-then-mask loop.
enum ExpertBank {
    /// `gu_w [E, hidden, 2·inter]`, `gu_b [E, 2·inter]`, `dn_w [E, inter, hidden]`, `dn_b [E, hidden]`
    /// — the `GptOssExperts` `[E, in, out]` layout, forward `x·w` via `gather_mm`.
    Dense {
        gu_w: Array,
        gu_b: Array,
        dn_w: Array,
        dn_b: Array,
    },
    /// Stacked Q4/Q8 packs for `gate_up` / `down`.
    Quant { gu: StackedQuant, dn: StackedQuant },
}

impl ExpertBank {
    /// gate_up projection: `x [.., 1, hidden]` → `[.., 1, 2·inter]`.
    fn gate_up(&self, x: &Array, idx: &Array, sorted: bool) -> Result<Array> {
        match self {
            ExpertBank::Dense { gu_w, gu_b, .. } => dense_gather(x, gu_w, gu_b, idx, sorted),
            ExpertBank::Quant { gu, .. } => gu.forward(x, idx, sorted),
        }
    }

    /// down projection: `gated [.., 1, inter]` → `[.., 1, hidden]`.
    fn down(&self, gated: &Array, idx: &Array, sorted: bool) -> Result<Array> {
        match self {
            ExpertBank::Dense { dn_w, dn_b, .. } => dense_gather(gated, dn_w, dn_b, idx, sorted),
            ExpertBank::Quant { dn, .. } => dn.forward(gated, idx, sorted),
        }
    }
}

/// Dense-bank projection: `gather_mm(x, w[E, in, out], rhs = idx)` + the gathered bias.
fn dense_gather(x: &Array, w: &Array, b: &Array, idx: &Array, sorted: bool) -> Result<Array> {
    let y = gather_mm(x, w, None, idx, sorted)?; // [.., 1, out]
    let bias = b.take_axis(idx, 0)?.expand_dims(-2)?;
    Ok(add(&y, &bias)?)
}

/// gpt-oss MoE feed-forward: a top-`k` linear router + `E` **clamped-SwiGLU** experts. Faithful port
/// of `GptOssTopKRouter` + `GptOssExperts`: router → top-`k` softmax over the selected logits; each
/// expert computes `(up+1)·(gate·σ(α·gate))` with `gate` clamped `≤limit` and `up` clamped `±limit`,
/// weighted by its router score.
///
/// F-021 (sc-9500): routing runs **on device** (top-`k` via iterated `argmax`, no per-layer host
/// sync) and the expert math is a **grouped GEMM** over the stacked [`ExpertBank`] — two `gather_qmm`
/// (Q4/Q8) / `gather_mm` (dense) calls that touch only the `top_k` selected experts per token
/// (mlx-lm's gpt-oss `SwitchGLU` construction, with `gate_up` kept fused → one gather). This drops
/// the per-token expert FLOPs from `E`→`top_k` (32→4) and removes the ×24-per-reasoner-token host
/// sync of the previous dense-all-experts path. The Q4/Q8 packs stay packed and are indexed per token
/// via `gather_qmm`'s `rhs_indices` — no dequant, so the memory footprint is unchanged.
///
/// Two dispatch shapes (mlx-lm SwitchGLU): for small token counts (decode / short prompts) the direct
/// broadcast gather; for `n·k ≥` [`GATHER_SORT_THRESHOLD`] (long prefill) the `(token, expert)` pairs
/// are argsort-ed by expert so `gather_qmm`'s `sorted_indices` fast path runs contiguous per-expert
/// GEMM, then scatter-unsorted back — without it the gathered path regresses vs dense at long prompts.
pub struct GptOssMoe {
    router_w: Array, // [E, hidden]
    router_b: Array, // [E]
    bank: ExpertBank,
    top_k: i32,
    inter: i32,
    alpha: f32,
    limit: f32,
}

impl GptOssMoe {
    /// Load `mlp` at `{prefix}` (e.g. `model.layers.0.mlp`). The router stays dense bf16; the experts
    /// load **stacked** `[E, …]` for grouped GEMM from one of three sources:
    ///
    /// * **packed turnkey** (sc-8763) — `experts.{gate_up,down}_proj.{weight,scales,biases}` already
    ///   stored stacked Q4/Q8; loaded as-is with no dequant transient (the on-disk memory win).
    /// * **MXFP4 + load-time quant** (`quant = Some`, the ~12 GB path, sc-3172) — each projection is
    ///   dequantized then re-quantized to MLX Q4/Q8 via [`prequantize_expert_proj`], which `eval`s the
    ///   pack and frees the per-layer bf16 transient before the next layer loads (the full `~38 GB`
    ///   bf16 expert stack across 24 layers never co-resides).
    /// * **MXFP4 dense** (`quant = None`) — dequantized to a stacked bf16 [`ExpertBank::Dense`].
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: &GptOssConfig,
        dtype: Dtype,
        quant: Option<Quant>,
    ) -> Result<Self> {
        let req = |k: &str| -> Result<Array> { Ok(w.require(k)?.as_dtype(dtype)?) };

        // Packed-detect (sc-8763): a **pre-quantized turnkey** stores the experts as the stacked packed
        // triple `experts.{gate_up,down}_proj.{weight,scales,biases}` (`crate::convert`), which loads
        // directly with NO MXFP4 dequant + re-quant transient — the memory win realized on disk. A
        // dense (MXFP4) source has no `experts.gate_up_proj.scales`, so it falls through to the
        // dequant-then-(optionally)-quantize paths below.
        let bank = if crate::quant::has_packed_experts(w, prefix, "gate_up") {
            ExpertBank::Quant {
                gu: stacked_quant_from_pack(crate::quant::load_packed_stack(w, prefix, "gate_up")?),
                dn: stacked_quant_from_pack(crate::quant::load_packed_stack(w, prefix, "down")?),
            }
        } else if let Some(q) = quant {
            // MXFP4 source + load-time quant: reuse the exact per-expert prequantize (returns the
            // stacked pack, byte-identical to the packed turnkey, and `eval`s it so the bf16 dequant
            // transient frees before the next layer loads).
            let (bits, gs) = (q.bits(), mlx_gen::quant::DEFAULT_GROUP_SIZE);
            let pack = |name: &str| -> Result<StackedQuant> {
                let blocks = w.require(&format!("{prefix}.experts.{name}_proj_blocks"))?;
                let scales = w.require(&format!("{prefix}.experts.{name}_proj_scales"))?;
                let bias = w.require(&format!("{prefix}.experts.{name}_proj_bias"))?;
                let p = prequantize_expert_proj(blocks, scales, bias, bits, gs)?;
                Ok(StackedQuant {
                    wq: p.weight,
                    scales: p.scales,
                    biases: p.biases,
                    b: p.bias,
                    group_size: gs,
                    bits,
                })
            };
            ExpertBank::Quant {
                gu: pack("gate_up")?,
                dn: pack("down")?,
            }
        } else {
            // MXFP4 dense (bf16): dequantize to the stacked `[E, in, out]` layout; no per-expert split.
            let gu_w = dequantize_mxfp4(
                w.require(&format!("{prefix}.experts.gate_up_proj_blocks"))?,
                w.require(&format!("{prefix}.experts.gate_up_proj_scales"))?,
                dtype,
            )?; // [E, hidden, 2*inter]
            let dn_w = dequantize_mxfp4(
                w.require(&format!("{prefix}.experts.down_proj_blocks"))?,
                w.require(&format!("{prefix}.experts.down_proj_scales"))?,
                dtype,
            )?; // [E, inter, hidden]
            ExpertBank::Dense {
                gu_w,
                gu_b: req(&format!("{prefix}.experts.gate_up_proj_bias"))?, // [E, 2*inter]
                dn_w,
                dn_b: req(&format!("{prefix}.experts.down_proj_bias"))?, // [E, hidden]
            }
        };

        Ok(Self {
            router_w: req(&format!("{prefix}.router.weight"))?,
            router_b: req(&format!("{prefix}.router.bias"))?,
            bank,
            top_k: cfg.experts_per_tok,
            inter: cfg.intermediate,
            alpha: cfg.swiglu_alpha,
            limit: cfg.swiglu_limit,
        })
    }

    /// `x`: `[B, L, hidden]`. Returns `[B, L, hidden]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, l, hidden) = (sh[0], sh[1], sh[2]);
        let n = b * l;
        let xf = x.reshape(&[n, hidden])?;

        // Router logits [n, E] → on-device top-`k` routing (selected indices + softmax weights).
        let logits = add(&matmul(&xf, self.router_w.t())?, &self.router_b)?;
        let (idx, weights) = self.route(&logits)?; // idx [n, k], weights [n, k]
        let k = self.top_k;

        // Gathered clamped-SwiGLU experts → `[n, k, hidden]`, weighted-summed over the `k` selected
        // experts (ascending expert order → matches the former dense path's accumulation order).
        let experts = self.gathered_experts(&xf, &idx, n, k, hidden)?;
        let weighted = multiply(&experts, &weights.reshape(&[n, k, 1])?)?; // [n, k, hidden]
        Ok(sum_axes(&weighted, &[-2], false)?.reshape(&[b, l, hidden])?)
    }

    /// Per-`(token, expert)` clamped-SwiGLU expert outputs → `[n, k, hidden]`, choosing the direct
    /// broadcast gather (small `n·k`) or the expert-sorted gather (large `n·k`, see [`GptOssMoe`]).
    fn gathered_experts(
        &self,
        xf: &Array,
        idx: &Array,
        n: i32,
        k: i32,
        hidden: i32,
    ) -> Result<Array> {
        if (n as i64) * (k as i64) >= GATHER_SORT_THRESHOLD {
            // mlx-lm `_gather_sort`: sort the `n·k` (token, expert) pairs by expert so `gather_qmm`'s
            // `sorted_indices` fast path does contiguous per-expert GEMM, then scatter-unsort. Each
            // pair's math is x[token]·W[expert] — identical values, so this is numerically equivalent
            // to the broadcast path; the unsort restores the original ascending-expert `k` order.
            let idx_flat = idx.reshape(&[n * k])?;
            let order = argsort_axis(&idx_flat, 0)?; // [n*k] pairs sorted by expert
            let inv_order = argsort_axis(&order, 0)?; // [n*k] undo permutation
            let sorted_idx = idx_flat.take(&order)?; // [n*k] ascending expert
            let token = floor_divide(&order, Array::from_slice(&[k], &[1]))?; // [n*k] token per pair
            let x_rows = xf.take_axis(&token, 0)?.reshape(&[n * k, 1, hidden])?; // [n*k, 1, hidden]
            let out = self.expert_mlp(&x_rows, &sorted_idx, true)?; // [n*k, 1, hidden]
            let out = out.reshape(&[n * k, hidden])?.take_axis(&inv_order, 0)?; // unsort
            Ok(out.reshape(&[n, k, hidden])?)
        } else {
            // Direct broadcast: `x [n, 1, 1, hidden]` against `idx [n, k]` → `[n, k, 1, hidden]`.
            let x4 = xf.reshape(&[n, 1, 1, hidden])?;
            Ok(self.expert_mlp(&x4, idx, false)?.squeeze_axes(&[2])?) // [n, k, hidden]
        }
    }

    /// The clamped-SwiGLU expert body over gathered rows: `gate_up` (fused, one gather) →
    /// de-interleave → clamp → SwiGLU → `down`. Shape-agnostic in the batch dims: `x` is `[.., 1,
    /// hidden]` (broadcasting against `idx`) and the result is `[.., 1, hidden]`.
    fn expert_mlp(&self, x: &Array, idx: &Array, sorted: bool) -> Result<Array> {
        let gate_up = self.bank.gate_up(x, idx, sorted)?; // [.., 1, 2*inter]

        // De-interleave gate/up: reshape the last axis `2*inter` → `(inter, 2)`, split (gate = `[..,
        // 0]`, up = `[.., 1]`), matching the dense `GptOssExperts` layout.
        let base: Vec<i32> = gate_up.shape()[..gate_up.shape().len() - 1].to_vec(); // [.., 1]
        let mut split_shape = base.clone();
        split_shape.push(self.inter);
        split_shape.push(2);
        let mut half_shape = base;
        half_shape.push(self.inter);
        let halves = split(&gate_up.reshape(&split_shape)?, 2, -1)?;
        let gate = halves[0].reshape(&half_shape)?;
        let up = halves[1].reshape(&half_shape)?;

        // Clamped SwiGLU: gate ≤ limit; up ∈ [−limit, limit]; (up+1)·(gate·σ(α·gate)).
        let limit = scalar(self.limit);
        let neg_limit = scalar(-self.limit);
        let alpha = scalar(self.alpha);
        let one = scalar(1.0);
        let gate = minimum(&gate, &limit)?;
        let up = maximum(&minimum(&up, &limit)?, &neg_limit)?;
        let glu = multiply(&gate, &sigmoid(&multiply(&gate, &alpha)?)?)?;
        let gated = multiply(&add(&up, &one)?, &glu)?; // [.., 1, inter]

        self.bank.down(&gated, idx, sorted) // [.., 1, hidden]
    }

    /// On-device top-`k` routing. Returns `(idx, weights)`: `idx [n, k]` the selected expert indices
    /// sorted **ascending** per row, `weights [n, k]` the softmax-over-selected routing weights in
    /// `logits`' dtype, aligned to `idx`.
    ///
    /// Selection is `k` iterations of `argmax` + mask-to-−∞. `argmax` returns the **first** occurrence
    /// of the maximum (verified on the pinned build), so this reproduces `torch.topk`'s
    /// descending-value / tie-by-lower-index selection the reference router uses — which matters only
    /// when two bf16 logits are exactly equal at the top-`k` boundary (which does occur). The ascending
    /// sort makes the `k`-term accumulation order match the former dense path (ascending expert index)
    /// and keeps the gathered bias / weights aligned. All on device: no per-layer host sync (the
    /// ×24-per-reasoner-token win). A NaN router logit (bf16 overflow — not observed on real weights,
    /// but the former host path guarded it) would select an arbitrary index here rather than panic;
    /// the real-weight goldens gate this.
    fn route(&self, logits: &Array) -> Result<(Array, Array)> {
        let n = logits.shape()[0];
        let k = self.top_k;
        let logit_dtype = logits.dtype();
        let lf = logits.as_dtype(Dtype::Float32)?;

        // Top-`k` indices in descending-value / tie-by-lower-index order.
        let neg_inf = broadcast_to(scalar(f32::NEG_INFINITY), &[n, 1])?;
        let mut work = lf.clone();
        let mut cols: Vec<Array> = Vec::with_capacity(k as usize);
        for _ in 0..k {
            let j = argmax_axis(&work, -1, true)?; // [n, 1] uint32
            work = work.put_along_axis(&j, &neg_inf, -1)?;
            cols.push(j);
        }
        let col_refs: Vec<&Array> = cols.iter().collect();
        let idx_desc = concatenate_axis(&col_refs, 1)?; // [n, k]

        // Softmax over the selected logits (f32, precise) in the selection order.
        let sel = lf.take_along_axis(&idx_desc, -1)?; // [n, k]
        let w_desc = softmax_axis(&sel, -1, true)?; // [n, k] f32

        // Sort experts ascending; reorder the weights identically (selected indices are distinct).
        let order = argsort_axis(&idx_desc, -1)?; // [n, k]
        let idx = idx_desc.take_along_axis(&order, -1)?; // [n, k] ascending
        let weights = w_desc.take_along_axis(&order, -1)?.as_dtype(logit_dtype)?;
        Ok((idx, weights))
    }
}

/// Assemble a forward-time [`StackedQuant`] from a loaded packed-turnkey stack (sc-8763).
fn stacked_quant_from_pack(p: crate::quant::StackedPack) -> StackedQuant {
    StackedQuant {
        wq: p.wq,
        scales: p.scales,
        biases: p.biases,
        b: p.bias,
        group_size: p.group_size,
        bits: p.bits,
    }
}

/// One full gpt-oss decoder layer: pre-norm sandwich `h + attn(rms(h))` then `h + moe(rms(h))`
/// (`GptOssDecoderLayer.forward`).
pub struct GptOssDecoderLayer {
    input_ln: Array,
    post_attn_ln: Array,
    attn: GptOssAttention,
    moe: GptOssMoe,
    eps: f32,
}

impl GptOssDecoderLayer {
    /// Load the layer at `{prefix}` (e.g. `model.layers.0`). `quant` (when `Some`) quantizes only the
    /// MoE experts to Q4/Q8 (sc-3172); attention/router/norms stay dense `dtype`.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: &GptOssConfig,
        dtype: Dtype,
        quant: Option<Quant>,
    ) -> Result<Self> {
        Ok(Self {
            input_ln: w
                .require(&format!("{prefix}.input_layernorm.weight"))?
                .as_dtype(dtype)?,
            post_attn_ln: w
                .require(&format!("{prefix}.post_attention_layernorm.weight"))?
                .as_dtype(dtype)?,
            attn: GptOssAttention::from_weights(w, &format!("{prefix}.self_attn"), cfg, dtype)?,
            moe: GptOssMoe::from_weights(w, &format!("{prefix}.mlp"), cfg, dtype, quant)?,
            eps: cfg.rms_eps,
        })
    }

    /// The MoE sub-block (exposed for isolated validation).
    pub fn moe(&self) -> &GptOssMoe {
        &self.moe
    }

    /// `x`: `[B, L, hidden]`. `inv_freq`/`attn_scaling`: YaRN constants. `mask`: additive attention
    /// mask. Returns `[B, L, hidden]`.
    pub fn forward(
        &self,
        x: &Array,
        inv_freq: &Array,
        attn_scaling: f32,
        mask: &Array,
    ) -> Result<Array> {
        let normed = rms_norm(x, &self.input_ln, self.eps)?;
        let h = add(
            x,
            &self.attn.forward(&normed, inv_freq, attn_scaling, mask)?,
        )?;
        let normed = rms_norm(&h, &self.post_attn_ln, self.eps)?;
        Ok(add(&h, &self.moe.forward(&normed)?)?)
    }

    /// Incremental (cached) decoder layer for generation (sc-3176): the same pre-norm sandwich, with
    /// the attention sub-block running [`GptOssAttention::forward_cached`] over `cache`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_cached(
        &self,
        x: &Array,
        inv_freq: &Array,
        attn_scaling: f32,
        position: i32,
        cache: &mut KvCache,
        sliding_window: Option<i32>,
        mask: Option<&Array>,
    ) -> Result<Array> {
        let normed = rms_norm(x, &self.input_ln, self.eps)?;
        let h = add(
            x,
            &self.attn.forward_cached(
                &normed,
                inv_freq,
                attn_scaling,
                position,
                cache,
                sliding_window,
                mask,
            )?,
        )?;
        let normed = rms_norm(&h, &self.post_attn_ln, self.eps)?;
        Ok(add(&h, &self.moe.forward(&normed)?)?)
    }
}

#[cfg(test)]
mod prequant_tests {
    use super::*;
    use mlx_rs::ops::eq;

    fn byte_equal(a: &Array, b: &Array) -> bool {
        a.shape() == b.shape()
            && a.dtype() == b.dtype()
            && eq(a, b).unwrap().all(None).unwrap().item::<bool>()
    }

    /// The offline stacked pack ([`prequantize_expert_proj`]) sliced back per-expert
    /// ([`crate::quant::load_packed_experts_from_stack`], axis-0 split) is byte-identical to the
    /// load-time dense path (`dequantize_mxfp4` → per-expert `Proj::into_quantized`) — the sc-8763
    /// round-trip guarantee for the MXFP4→MLX-affine encoder experts (the grouped-GEMM forward
    /// consumes the same stack whole). Uses a tiny synthetic MXFP4 tensor (`E=2`, `out=4`, `G=2` ⇒
    /// `in=64`, group-aligned).
    #[test]
    fn stacked_expert_pack_slice_byte_identical_to_load_time() {
        let (e, out, g) = (2usize, 4usize, 2usize);
        let in_c = g * 32; // 64, group-aligned
                           // Deterministic pseudo-random nibbles + e8m0 scales (127 ⇒ 2^0).
        let blocks: Vec<u8> = (0..e * out * g * 16)
            .map(|i| (i * 37 % 256) as u8)
            .collect();
        let scales: Vec<u8> = (0..e * out * g).map(|i| (120 + (i % 8)) as u8).collect();
        let bias: Vec<f32> = (0..e * out).map(|i| (i as f32).cos()).collect();
        let blocks = Array::from_slice(&blocks, &[e as i32, out as i32, g as i32, 16]);
        let scales = Array::from_slice(&scales, &[e as i32, out as i32, g as i32]);
        let bias = Array::from_slice(&bias, &[e as i32, out as i32]);

        let bits = 4;
        let gs = 64;
        // Offline stacked pack.
        let pack = prequantize_expert_proj(&blocks, &scales, &bias, bits, gs).unwrap();
        let sliced = crate::quant::load_packed_experts_from_stack(
            &pack.weight,
            &pack.scales,
            &pack.biases,
            &pack.bias,
        )
        .unwrap();

        // Load-time reference: dequantize MXFP4 then quantize each expert independently.
        let dense = dequantize_mxfp4(&blocks, &scales, Dtype::Bfloat16).unwrap(); // [E, in, out]
        let per = split(&dense, e as i32, 0).unwrap();
        let bias_e = split(&bias, e as i32, 0).unwrap();
        for i in 0..e {
            let proj = Proj::Dense {
                w: per[i].reshape(&[in_c as i32, out as i32]).unwrap(),
                b: bias_e[i].reshape(&[out as i32]).unwrap(),
            };
            match proj.into_quantized(bits, gs).unwrap() {
                Proj::Quant {
                    wq,
                    scales,
                    biases,
                    b,
                } => {
                    assert!(byte_equal(&sliced[i].wq, &wq), "expert {i} wq mismatch");
                    assert!(
                        byte_equal(&sliced[i].scales, &scales),
                        "expert {i} scales mismatch"
                    );
                    assert!(
                        byte_equal(&sliced[i].biases, &biases),
                        "expert {i} biases mismatch"
                    );
                    assert!(byte_equal(&sliced[i].bias, &b), "expert {i} bias mismatch");
                }
                Proj::Dense { .. } => unreachable!(),
            }
        }
    }
}

/// F-021 (sc-9500) default-suite tests for the grouped-GEMM MoE — committed-fixture-only (synthetic
/// random weights, no snapshot), so `cargo test` stays green on a fresh clone. Real-weight parity
/// (`encoder_parity` / `encoder_quant_parity` / `reasoner_parity`) is the `#[ignore]`d gate.
#[cfg(test)]
mod moe_grouped_tests {
    use super::*;
    use mlx_rs::ops::{abs, max, quantized_matmul};

    fn rn(shape: &[i32], scale: f32, dtype: Dtype) -> Array {
        let a = mlx_rs::random::normal::<f32>(shape, None, None, None).unwrap();
        multiply(&a, scalar(scale))
            .unwrap()
            .as_dtype(dtype)
            .unwrap()
    }

    /// Slice expert `e` off a stacked `[E, …]` array, dropping the leading axis.
    fn slice_e(a: &Array, e: usize) -> Array {
        let ecount = a.shape()[0];
        split(a, ecount, 0).unwrap()[e].squeeze_axes(&[0]).unwrap()
    }

    fn build_dense(e: i32, k: i32, hidden: i32, inter: i32, dtype: Dtype) -> GptOssMoe {
        let s_in = 1.0 / (hidden as f32).sqrt();
        GptOssMoe {
            router_w: rn(&[e, hidden], s_in, dtype),
            router_b: rn(&[e], 0.1, dtype),
            bank: ExpertBank::Dense {
                gu_w: rn(&[e, hidden, 2 * inter], s_in, dtype),
                gu_b: rn(&[e, 2 * inter], 0.1, dtype),
                dn_w: rn(&[e, inter, hidden], 1.0 / (inter as f32).sqrt(), dtype),
                dn_b: rn(&[e, hidden], 0.1, dtype),
            },
            top_k: k,
            inter,
            alpha: 1.702,
            limit: 7.0,
        }
    }

    /// Quantize a dense bank's stacked weights to Q`bits` (the same transpose→bf16→`quantize` seam as
    /// `Proj::into_quantized`, applied to the whole `[E, out, in]` stack — affine quant is per-row).
    fn quantize_bank(dense: &ExpertBank, bits: i32) -> ExpertBank {
        let ExpertBank::Dense {
            gu_w,
            gu_b,
            dn_w,
            dn_b,
        } = dense
        else {
            unreachable!("build_dense yields Dense")
        };
        let pack = |w_in_out: &Array, b: &Array| -> StackedQuant {
            let w_oi = w_in_out
                .swap_axes(-1, -2)
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            let (wq, scales, biases) = quantize(&w_oi, 64, bits).unwrap();
            StackedQuant {
                wq,
                scales,
                biases,
                b: b.as_dtype(Dtype::Bfloat16).unwrap(),
                group_size: 64,
                bits,
            }
        };
        ExpertBank::Quant {
            gu: pack(gu_w, gu_b),
            dn: pack(dn_w, dn_b),
        }
    }

    /// Host `torch.topk`-semantics routing → dense `[n, E]` weight matrix (zero off the top-k). The
    /// pre-F-021 production routing, kept as the equivalence/tie oracle.
    fn host_routing(logits: &Array, top_k: i32) -> Array {
        let sh = logits.shape();
        let (n, e) = (sh[0] as usize, sh[1] as usize);
        let k = top_k as usize;
        let l32 = logits.as_dtype(Dtype::Float32).unwrap();
        let data = l32.as_slice::<f32>();
        let mut out = vec![0f32; n * e];
        for row in 0..n {
            let s = &data[row * e..row * e + e];
            let mut idx: Vec<usize> = (0..e).collect();
            idx.sort_by(|&a, &b| s[b].total_cmp(&s[a]).then(a.cmp(&b)));
            let top = &idx[..k];
            let maxv = top.iter().map(|&i| s[i]).fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0f32;
            let exps: Vec<f32> = top
                .iter()
                .map(|&i| {
                    let ev = (s[i] - maxv).exp();
                    denom += ev;
                    ev
                })
                .collect();
            for (j, &i) in top.iter().enumerate() {
                out[row * e + i] = exps[j] / denom;
            }
        }
        Array::from_slice(&out, &[n as i32, e as i32])
            .as_dtype(logits.dtype())
            .unwrap()
    }

    /// The pre-F-021 dense-all-experts forward (validated against goldens historically), rebuilt in
    /// the test as the reference the new grouped-GEMM forward must match.
    fn dense_reference(moe: &GptOssMoe, x: &Array) -> Array {
        let sh = x.shape();
        let (b, l, hidden) = (sh[0], sh[1], sh[2]);
        let n = b * l;
        let xf = x.reshape(&[n, hidden]).unwrap();
        let logits = add(matmul(&xf, moe.router_w.t()).unwrap(), &moe.router_b).unwrap();
        let ecount = moe.router_w.shape()[0];
        let routing = host_routing(&logits, moe.top_k);
        let routing_cols = split(&routing, ecount, 1).unwrap();

        let limit = scalar(moe.limit);
        let neg_limit = scalar(-moe.limit);
        let alpha = scalar(moe.alpha);
        let one = scalar(1.0);
        let mut acc: Option<Array> = None;
        for (e, col) in routing_cols.iter().enumerate() {
            let gate_up = proj_ref(&moe.bank, true, e, &xf);
            let gu = gate_up.reshape(&[n, moe.inter, 2]).unwrap();
            let halves = split(&gu, 2, -1).unwrap();
            let gate = halves[0].reshape(&[n, moe.inter]).unwrap();
            let up = halves[1].reshape(&[n, moe.inter]).unwrap();
            let gate = minimum(&gate, &limit).unwrap();
            let up = maximum(minimum(&up, &limit).unwrap(), &neg_limit).unwrap();
            let glu = multiply(&gate, sigmoid(multiply(&gate, &alpha).unwrap()).unwrap()).unwrap();
            let gated = multiply(add(&up, &one).unwrap(), &glu).unwrap();
            let out_e = proj_ref(&moe.bank, false, e, &gated);
            let weighted = multiply(&out_e, col).unwrap();
            acc = Some(match acc {
                None => weighted,
                Some(a) => add(&a, &weighted).unwrap(),
            });
        }
        acc.unwrap().reshape(&[b, l, hidden]).unwrap()
    }

    /// Reference single-expert projection: `x·w + b` (dense) or per-expert `quantized_matmul` (quant).
    fn proj_ref(bank: &ExpertBank, gate_up: bool, e: usize, x: &Array) -> Array {
        match bank {
            ExpertBank::Dense {
                gu_w,
                gu_b,
                dn_w,
                dn_b,
            } => {
                let (w, b) = if gate_up { (gu_w, gu_b) } else { (dn_w, dn_b) };
                add(matmul(x, slice_e(w, e)).unwrap(), slice_e(b, e)).unwrap()
            }
            ExpertBank::Quant { gu, dn } => {
                let sq = if gate_up { gu } else { dn };
                let y = quantized_matmul(
                    x,
                    slice_e(&sq.wq, e),
                    slice_e(&sq.scales, e),
                    &slice_e(&sq.biases, e),
                    true,
                    sq.group_size,
                    sq.bits,
                )
                .unwrap();
                add(&y, slice_e(&sq.b, e)).unwrap()
            }
        }
    }

    fn peak_rel(got: &Array, want: &Array) -> f32 {
        let g = got.as_dtype(Dtype::Float32).unwrap();
        let w = want.as_dtype(Dtype::Float32).unwrap();
        let diff = abs(subtract(&g, &w).unwrap()).unwrap();
        let denom = max(abs(&w).unwrap(), None).unwrap().item::<f32>().max(1e-6);
        max(&diff, None).unwrap().item::<f32>() / denom
    }

    /// The two forward shapes to exercise per equivalence test: `(2,5)` = n·k 20 → broadcast gather;
    /// `(8,16)` = n·k 256 ≥ [`GATHER_SORT_THRESHOLD`] → the expert-sorted gather path. Both must match
    /// the dense reference (the sort is numerically equivalent — same per-pair math, reordered).
    const EQUIV_SHAPES: [(i32, i32); 2] = [(2, 5), (8, 16)];

    /// Gathered grouped-GEMM forward ≡ the dense-all-experts reference, at f32. Not bit-exact:
    /// `gather_mm` is a different Metal GEMM kernel than the loop's `matmul`, and MLX fp32 matmul is
    /// itself reduced-precision on Metal (~1e-3, per the crate tolerance convention), so the observed
    /// peak_rel ≈ 1.8e-3 is the kernel-tiling floor — well inside the crate's ~1e-2 parity bar.
    #[test]
    fn gathered_matches_dense_reference_f32() {
        mlx_rs::random::seed(7).unwrap();
        let (e, k, hidden, inter) = (8, 2, 64, 128);
        let moe = build_dense(e, k, hidden, inter, Dtype::Float32);
        for (bb, ll) in EQUIV_SHAPES {
            let x = rn(&[bb, ll, hidden], 1.0, Dtype::Float32);
            let pr = peak_rel(&moe.forward(&x).unwrap(), &dense_reference(&moe, &x));
            eprintln!(
                "dense f32 n={} grouped-vs-loop peak_rel = {pr:.3e}",
                bb * ll
            );
            assert!(
                pr < 4e-3,
                "dense f32 n={} peak_rel {pr:.3e} exceeds 4e-3",
                bb * ll
            );
        }
    }

    /// Same at bf16 (the encoder's production dtype) — looser, since bf16 gather vs matmul differ at
    /// the 8-bit mantissa.
    #[test]
    fn gathered_matches_dense_reference_bf16() {
        mlx_rs::random::seed(11).unwrap();
        let (e, k, hidden, inter) = (8, 2, 64, 128);
        let moe = build_dense(e, k, hidden, inter, Dtype::Bfloat16);
        for (bb, ll) in EQUIV_SHAPES {
            let x = rn(&[bb, ll, hidden], 1.0, Dtype::Bfloat16);
            let pr = peak_rel(&moe.forward(&x).unwrap(), &dense_reference(&moe, &x));
            eprintln!(
                "dense bf16 n={} grouped-vs-loop peak_rel = {pr:.3e}",
                bb * ll
            );
            assert!(
                pr < 3e-2,
                "dense bf16 n={} peak_rel {pr:.3e} exceeds 3e-2",
                bb * ll
            );
        }
    }

    /// Q8 / Q4 gathered `gather_qmm` ≡ per-expert `quantized_matmul` loop over the same packs (f32
    /// activations isolate the gather kernel from bf16 rounding).
    #[test]
    fn gathered_quant_matches_perexpert_loop() {
        for (bits, tol) in [(8, 3e-3f32), (4, 3e-2f32)] {
            mlx_rs::random::seed(3).unwrap();
            let (e, k, hidden, inter) = (8, 2, 64, 128);
            let dense = build_dense(e, k, hidden, inter, Dtype::Float32);
            let moe = GptOssMoe {
                router_w: dense.router_w.clone(),
                router_b: dense.router_b.clone(),
                bank: quantize_bank(&dense.bank, bits),
                top_k: k,
                inter,
                alpha: 1.702,
                limit: 7.0,
            };
            for (bb, ll) in EQUIV_SHAPES {
                let x = rn(&[bb, ll, hidden], 1.0, Dtype::Float32);
                let pr = peak_rel(&moe.forward(&x).unwrap(), &dense_reference(&moe, &x));
                eprintln!("Q{bits} n={} grouped-vs-loop peak_rel = {pr:.3e}", bb * ll);
                assert!(
                    pr < tol,
                    "Q{bits} n={} peak_rel {pr:.3e} exceeds {tol:.0e}",
                    bb * ll
                );
            }
        }
    }

    /// Device `route` reproduces `torch.topk`'s descending / tie-by-lower-index selection AND the
    /// softmax weights exactly (bf16 logits with deliberate exact ties, incl. a top-k boundary tie).
    #[test]
    fn route_tie_semantics_match_host_oracle() {
        let (e, k) = (6i32, 3i32);
        let moe = build_dense(e, k, 8, 16, Dtype::Bfloat16);
        // row0: three-way tie at 5.0 (idx 1,2,4) → select {1,2,4}.
        // row1: 5.0 at idx0 then three 4.0 (idx1,2,3) → boundary tie → select {0,1,2}.
        let logits = Array::from_slice(
            &[
                3.0f32, 5.0, 5.0, 1.0, 5.0, 2.0, 5.0, 4.0, 4.0, 4.0, 1.0, 0.0,
            ],
            &[2, e],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();

        let (idx, weights) = moe.route(&logits).unwrap();
        let idx_v: Vec<u32> = idx
            .as_dtype(Dtype::Uint32)
            .unwrap()
            .as_slice::<u32>()
            .to_vec();
        assert_eq!(
            idx_v,
            vec![1, 2, 4, 0, 1, 2],
            "tie selection (ascending) {idx_v:?}"
        );

        // Weights vs host oracle (dense [n,E]), gathered at the selected ascending indices.
        let oracle = host_routing(&logits, k);
        let want = oracle.take_along_axis(&idx, -1).unwrap();
        let pr = peak_rel(&weights, &want);
        eprintln!("tie weights peak_rel = {pr:.3e}");
        assert!(pr < 1e-2, "tie weights peak_rel {pr:.3e}");
    }

    /// Shape coverage: n=1 (decode), k=E (all experts), B>1 — output `[B, L, hidden]`, finite.
    #[test]
    fn shapes_decode_all_experts_batched() {
        mlx_rs::random::seed(5).unwrap();
        let (e, hidden, inter) = (8, 32, 64);
        for (bb, ll, k) in [(1, 1, 2), (1, 1, e), (3, 4, 2)] {
            let moe = build_dense(e, k, hidden, inter, Dtype::Float32);
            let x = rn(&[bb, ll, hidden], 1.0, Dtype::Float32);
            let out = moe.forward(&x).unwrap();
            assert_eq!(out.shape(), &[bb, ll, hidden], "shape (B{bb} L{ll} k{k})");
            let s = out.as_slice::<f32>();
            assert!(
                s.iter().all(|v| v.is_finite()),
                "non-finite (B{bb} L{ll} k{k})"
            );
        }
    }

    /// Resolve the cached SceneWorks Lens-Turbo **q4** text-encoder safetensors (the real Q4-packed
    /// gpt-oss MoE), or `$LENS_TURBO_TE` if set. Real-weight, licensed → the test is `#[ignore]`d.
    fn lens_turbo_te_path() -> Option<std::path::PathBuf> {
        if let Ok(p) = std::env::var("LENS_TURBO_TE") {
            return Some(p.into());
        }
        let base = std::path::PathBuf::from(std::env::var("HOME").ok()?)
            .join(".cache/huggingface/hub/models--SceneWorks--lens-turbo-mlx/snapshots");
        let snap = std::fs::read_dir(&base)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.is_dir())?;
        let f = snap.join("q4/text_encoder/model.safetensors");
        f.exists().then_some(f)
    }

    /// sc-9500 real-weight gate (same-backend). On the **real Q4-packed** Lens-Turbo experts, the new
    /// grouped-GEMM forward (`gather_qmm` + on-device top-k) must match the pre-F-021 per-expert
    /// `quantized_matmul` loop over the identical packs (the path that was cross-backend-validated
    /// against the torch encoder golden in prior sessions). The MLX↔torch goldens themselves need the
    /// original `microsoft/Lens-Turbo` transformers snapshot (not in the local cache — only the
    /// MLX-converted q4 pipeline is), so this same-backend equivalence — the crate-preferred end-to-end
    /// check — closes `new ≈ old ≈ golden` transitively. Sampled across layers to exercise the router
    /// + both projections on genuine weights (incl. any real bf16 router ties).
    #[test]
    #[ignore = "needs the cached SceneWorks/lens-turbo-mlx q4 text_encoder (Q4-packed, ~10GB) — sc-9500 real-weight gate"]
    fn real_weight_gathered_matches_dense_loop_q4() {
        let path = lens_turbo_te_path().expect("no lens-turbo q4 text_encoder; set LENS_TURBO_TE");
        eprintln!("loading real Q4 text_encoder: {}", path.display());
        let w = Weights::from_file(&path).expect("load lens-turbo q4 text_encoder");
        let cfg = GptOssConfig::lens();

        // f32 activations give a fine-grained (discriminating) comparison; bf16 is the encoder's
        // production dtype (where the ~1e-3 f32 intermediate gap collapses below bf16 resolution, so
        // the two paths land on bit-identical output). `seq` 6 hits the broadcast gather, 128 the
        // expert-sorted gather (both must match). Assert both dtypes × both dispatch paths.
        for (dtype, tol) in [(Dtype::Float32, 5e-3f32), (Dtype::Bfloat16, 2e-2f32)] {
            let mut worst = 0f32;
            for seq in [6i32, 128] {
                mlx_rs::random::seed(0).unwrap();
                let x = rn(&[1, seq, cfg.hidden_size], 1.0, dtype);
                for layer in [0usize, 6, 12, 18, 23] {
                    let moe = GptOssMoe::from_weights(
                        &w,
                        &format!("model.layers.{layer}.mlp"),
                        &cfg,
                        dtype,
                        None,
                    )
                    .unwrap();
                    assert!(
                        matches!(moe.bank, ExpertBank::Quant { .. }),
                        "layer {layer} expected the packed Q4 path"
                    );
                    let got = moe.forward(&x).unwrap();
                    let want = dense_reference(&moe, &x);
                    let pr = peak_rel(&got, &want);
                    worst = worst.max(pr);
                    eprintln!("{dtype:?} seq={seq} layer {layer}: peak_rel = {pr:.3e}");
                }
            }
            eprintln!(
                "sc-9500 {dtype:?} worst peak_rel (both paths, sampled layers) = {worst:.3e}"
            );
            assert!(
                worst < tol,
                "real-weight Q4 {dtype:?} peak_rel {worst:.3e} exceeds {tol:.0e}"
            );
        }
    }

    /// sc-9500 Phase 5 — real-weight MoE perf (single layer). Times the new grouped-GEMM forward
    /// against `dense_reference` (which recomputes the pre-F-021 all-32-experts loop + one host-sync
    /// route, i.e. the "before"), at decode (n=1) and prefill (n=512) shapes. Not an assertion —
    /// prints the measured speedup for the story record. Median of a few timed iters after a warmup.
    #[test]
    #[ignore = "perf measurement — needs the cached SceneWorks/lens-turbo-mlx q4 text_encoder"]
    fn real_weight_moe_perf_q4() {
        use std::time::Instant;
        let path = lens_turbo_te_path().expect("no lens-turbo q4 text_encoder; set LENS_TURBO_TE");
        let w = Weights::from_file(&path).expect("load lens-turbo q4 text_encoder");
        let cfg = GptOssConfig::lens();
        let moe =
            GptOssMoe::from_weights(&w, "model.layers.0.mlp", &cfg, Dtype::Bfloat16, None).unwrap();

        let time = |f: &dyn Fn() -> Array| -> f64 {
            let out = f();
            mlx_rs::transforms::eval([&out]).unwrap(); // warmup
            let iters = 20;
            let t0 = Instant::now();
            for _ in 0..iters {
                let o = f();
                mlx_rs::transforms::eval([&o]).unwrap();
            }
            t0.elapsed().as_secs_f64() * 1e3 / iters as f64 // ms/iter
        };

        for n in [1i32, 8, 32, 64, 128, 256, 512] {
            mlx_rs::random::seed(1).unwrap();
            let x = rn(&[1, n, cfg.hidden_size], 1.0, Dtype::Bfloat16);
            let new_ms = time(&|| moe.forward(&x).unwrap());
            let old_ms = time(&|| dense_reference(&moe, &x));
            eprintln!(
                "n={n:>4}: grouped {new_ms:7.3} ms  dense-all-32 {old_ms:7.3} ms  speedup {:.2}×",
                old_ms / new_ms
            );
        }
    }
}
