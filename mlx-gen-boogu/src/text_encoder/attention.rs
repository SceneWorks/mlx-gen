//! GQA self-attention for the Boogu Qwen3-VL text encoder: GQA (32 query / 8 kv heads), bias-less
//! q/k/v/o, per-head q/k RMSNorm (on the head dim, before RoPE), HF half-split RoPE, masked SDPA.
//! Text-only path → plain 1-D RoPE. Dense-or-packed via `AdaptableLinear`.

use mlx_rs::fast::{rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, broadcast_to, concatenate_axis, multiply, split};
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, lin};

pub struct Qwen3Attention {
    q_w: AdaptableLinear,
    k_w: AdaptableLinear,
    v_w: AdaptableLinear,
    o_w: AdaptableLinear,
    q_norm: Array,
    k_norm: Array,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    eps: f32,
}

impl Qwen3Attention {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            q_w: lin(w, &join(prefix, "q_proj.weight"))?,
            k_w: lin(w, &join(prefix, "k_proj.weight"))?,
            v_w: lin(w, &join(prefix, "v_proj.weight"))?,
            o_w: lin(w, &join(prefix, "o_proj.weight"))?,
            q_norm: w.require(&join(prefix, "q_norm.weight"))?.clone(),
            k_norm: w.require(&join(prefix, "k_norm.weight"))?.clone(),
            num_heads,
            num_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            eps,
        })
    }

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

        // Per-head q/k RMSNorm over the head dim (Qwen3), before RoPE.
        let q = rms_norm(&q, &self.q_norm, self.eps)?;
        let k = rms_norm(&k, &self.k_norm, self.eps)?;

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
        let o = o
            .transpose_axes(&[0, 2, 1, 3])?
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

/// Expand `[b,s,hkv,hd]` → `[b,s,hkv*groups,hd]`, repeating each kv head `groups` times.
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
