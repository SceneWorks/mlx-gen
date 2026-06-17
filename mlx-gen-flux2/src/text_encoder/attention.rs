//! GQA self-attention for the FLUX.2 decoder-LM text encoders: GQA (32 query / 8 kv heads),
//! **bias-less** q/k/v/o projections, HF half-split RoPE, masked SDPA. Port of `Qwen3VLAttention`
//! for the text path (`mrope_section=None`).
//!
//! **Per-head q/k RMSNorm** on the head dim is the Qwen3 addition over Qwen2.5 — present for the
//! klein Qwen3 encoder, **absent** for the FLUX.2-dev Mistral encoder (sc-5915), which is otherwise
//! identical. Gated by `qk_norm`; when off, q/k flow straight into RoPE.

use mlx_rs::fast::{rms_norm, scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::ops::{add, broadcast_to, concatenate_axis, multiply, split};
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::generate::Qwen3KvCache;
use super::{join, lin};
use crate::config::Flux2Quant;

pub struct Qwen3Attention {
    q_w: AdaptableLinear,
    k_w: AdaptableLinear,
    v_w: AdaptableLinear,
    o_w: AdaptableLinear,
    /// Per-head q/k RMSNorm weights — `Some` for Qwen3 (klein), `None` for Mistral (dev).
    q_norm: Option<Array>,
    k_norm: Option<Array>,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    eps: f32,
}

impl Qwen3Attention {
    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        eps: f32,
        qk_norm: bool,
        quant: Option<Flux2Quant>,
    ) -> Result<Self> {
        let (q_norm, k_norm) = if qk_norm {
            (
                Some(w.require(&join(prefix, "q_norm.weight"))?.clone()),
                Some(w.require(&join(prefix, "k_norm.weight"))?.clone()),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            q_w: lin(w, &join(prefix, "q_proj.weight"), quant)?,
            k_w: lin(w, &join(prefix, "k_proj.weight"), quant)?,
            v_w: lin(w, &join(prefix, "v_proj.weight"), quant)?,
            o_w: lin(w, &join(prefix, "o_proj.weight"), quant)?,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            eps,
        })
    }

    /// Quantize the q/k/v/o projections (group_size 64). q/k RMSNorm stays full precision.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.q_w.quantize(bits, None)?;
        self.k_w.quantize(bits, None)?;
        self.v_w.quantize(bits, None)?;
        self.o_w.quantize(bits, None)?;
        Ok(())
    }

    /// `x`: `[b, s, hidden]`; `cos`/`sin`: `[1, s, head_dim]`; `mask`: additive `[b,1,s,s]`.
    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array, mask: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);

        let q = self
            .q_w
            .forward(x)?
            .reshape(&[b, s, self.num_heads, self.head_dim])?;
        let k = self
            .k_w
            .forward(x)?
            .reshape(&[b, s, self.num_kv_heads, self.head_dim])?;
        let v = self
            .v_w
            .forward(x)?
            .reshape(&[b, s, self.num_kv_heads, self.head_dim])?;

        // Per-head q/k RMSNorm over the head dim (Qwen3), before RoPE — matches the fork order.
        // Mistral (dev) has no qk-norm, so q/k pass through unchanged.
        let q = match &self.q_norm {
            Some(g) => rms_norm(&q, g, self.eps)?,
            None => q,
        };
        let k = match &self.k_norm {
            Some(g) => rms_norm(&k, g, self.eps)?,
            None => k,
        };

        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;

        // GQA: repeat each kv head `groups` times to match the query heads.
        let groups = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;

        // [b,s,h,hd] → [b,h,s,hd]
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        let mask = mask.as_dtype(q.dtype())?;
        let o = scaled_dot_product_attention(&q, &k, &v, self.scale, &mask, None)?;
        let o =
            o.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, s, self.num_heads * self.head_dim])?;
        self.o_w.forward(&o)
    }

    /// KV-cached **causal** decode step (caption-upsampling generate, sc-6030). Mirrors
    /// [`forward`](Self::forward) but appends this step's K/V to the per-layer growable `cache` and
    /// attends causally over the full cached sequence via the implicit bottom-right causal mask
    /// (MLX aligns the `q_len` queries to the last cached positions, exactly the joycaption/
    /// prompt-refine decode rule). `cos`/`sin` are the RoPE table at the step's **absolute** positions
    /// (`TextRope::forward_offset`), so prefill (offset 0, `q_len = prompt_len`) and each 1-token
    /// decode (offset = cache length) both index the right frequencies. Used only by the Mistral dev
    /// tower (`qk_norm` is `None`); the branch is kept for symmetry with `forward`.
    pub(crate) fn forward_step(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        cache: &mut Qwen3KvCache,
        layer_idx: usize,
    ) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);

        let q = self
            .q_w
            .forward(x)?
            .reshape(&[b, s, self.num_heads, self.head_dim])?;
        let k = self
            .k_w
            .forward(x)?
            .reshape(&[b, s, self.num_kv_heads, self.head_dim])?;
        let v = self
            .v_w
            .forward(x)?
            .reshape(&[b, s, self.num_kv_heads, self.head_dim])?;

        let q = match &self.q_norm {
            Some(g) => rms_norm(&q, g, self.eps)?,
            None => q,
        };
        let k = match &self.k_norm {
            Some(g) => rms_norm(&k, g, self.eps)?,
            None => k,
        };

        let q = apply_rope(&q, cos, sin)?.transpose_axes(&[0, 2, 1, 3])?;
        let k = apply_rope(&k, cos, sin)?.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;
        let (k_all, v_all) = cache.append(layer_idx, k, v)?;

        let groups = self.num_heads / self.num_kv_heads;
        let k_all = repeat_kv_cache(&k_all, groups)?;
        let v_all = repeat_kv_cache(&v_all, groups)?;

        let o = scaled_dot_product_attention(
            &q,
            &k_all,
            &v_all,
            self.scale,
            ScaledDotProductAttentionMask::Causal,
            None,
        )?;
        let o =
            o.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, s, self.num_heads * self.head_dim])?;
        self.o_w.forward(&o)
    }
}

/// HF half-split RoPE: `x*cos + rotate_half(x)*sin`, `rotate_half(x) = [-x2, x1]`. `cos`/`sin`
/// `[1,s,hd]` → broadcast over heads (axis 2).
fn apply_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let cos = cos.expand_dims(2)?; // [1,s,1,hd]
    let sin = sin.expand_dims(2)?;
    let parts = split(x, 2, 3)?; // halves along the head dim
    let rot = concatenate_axis(&[&parts[1].negative()?, &parts[0]], 3)?;
    Ok(add(&multiply(x, &cos)?, &multiply(&rot, &sin)?)?)
}

/// Expand `[b,s,hkv,hd]` → `[b,s,hkv*groups,hd]`, repeating each kv head `groups` times
/// consecutively (matching `mx.repeat(x, groups, axis=2)`).
fn repeat_kv(x: &Array, groups: i32) -> Result<Array> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let sh = x.shape();
    let (b, s, hkv, hd) = (sh[0], sh[1], sh[2], sh[3]);
    let x = x.expand_dims(3)?; // [b,s,hkv,1,hd]
    let x = broadcast_to(&x, &[b, s, hkv, groups, hd])?;
    Ok(x.reshape(&[b, s, hkv * groups, hd])?)
}

/// GQA expand for the **decode** layout `[b,hkv,s,hd]` → `[b,hkv*groups,s,hd]` (the post-transpose,
/// post-cache K/V), repeating each kv head `groups` times consecutively. The decode companion to
/// [`repeat_kv`] (which runs pre-transpose).
fn repeat_kv_cache(x: &Array, groups: i32) -> Result<Array> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let sh = x.shape();
    let (b, hkv, s, hd) = (sh[0], sh[1], sh[2], sh[3]);
    let x = x.expand_dims(2)?; // [b,hkv,1,s,hd]
    let x = broadcast_to(&x, &[b, hkv, groups, s, hd])?;
    Ok(x.reshape(&[b, hkv * groups, s, hd])?)
}
