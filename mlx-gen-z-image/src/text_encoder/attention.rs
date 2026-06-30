//! Text-encoder self-attention: GQA (q heads > kv heads) with per-head `q_norm`/`k_norm`,
//! HF half-split RoPE, and masked SDPA. Port of the fork's `Attention` (no biases).

use mlx_rs::fast::{rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, broadcast_to, concatenate_axis, multiply, split};
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, QK_NORM_EPS};

pub struct TextAttention {
    q_proj: AdaptableLinear,
    k_proj: AdaptableLinear,
    v_proj: AdaptableLinear,
    o_proj: AdaptableLinear,
    q_norm: Array,
    k_norm: Array,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl TextAttention {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> Result<Self> {
        // Packed-detect (sc-8670): the GQA projections load packed from a pre-quantized snapshot or
        // dense otherwise. Qwen3 attention has no biases.
        let lin = |name: &str| crate::quant::lin(w, &join(prefix, name), false);
        Ok(Self {
            q_proj: lin("q_proj")?,
            k_proj: lin("k_proj")?,
            v_proj: lin("v_proj")?,
            o_proj: lin("o_proj")?,
            q_norm: w.require(&join(prefix, "q_norm.weight"))?.clone(),
            k_norm: w.require(&join(prefix, "k_norm.weight"))?.clone(),
            num_heads,
            num_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// Quantize the QKV + output projections to Q4/Q8 (group_size 64) — the fork quantizes every
    /// Linear in the text encoder. `q_norm`/`k_norm` are RMSNorm scales, so they stay dense.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for lin in [
            &mut self.q_proj,
            &mut self.k_proj,
            &mut self.v_proj,
            &mut self.o_proj,
        ] {
            lin.quantize(bits, None)?;
        }
        Ok(())
    }

    /// `x`: `[b, s, hidden]`; `cos`/`sin`: `[1, s, head_dim]`; `mask`: additive `[1,1,s,s]`.
    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array, mask: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);

        let q = self
            .q_proj
            .forward(x)?
            .reshape(&[b, s, self.num_heads, self.head_dim])?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape(&[b, s, self.num_kv_heads, self.head_dim])?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape(&[b, s, self.num_kv_heads, self.head_dim])?;

        // per-head RMSNorm (q_norm/k_norm use mlx's default eps, not rms_norm_eps).
        let q = rms_norm(&q, &self.q_norm, QK_NORM_EPS)?;
        let k = rms_norm(&k, &self.k_norm, QK_NORM_EPS)?;

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

        // trailing `None` is the MLX ≥0.30 `sinks` arg (no attention sinks).
        let o = scaled_dot_product_attention(&q, &k, &v, self.scale, mask, None)?;
        let o =
            o.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, s, self.num_heads * self.head_dim])?;
        self.o_proj.forward(&o)
    }
}

/// HF half-split RoPE: `x*cos + rotate_half(x)*sin`, where `rotate_half(x) = [-x2, x1]`
/// (x split in halves along the head dim). `cos`/`sin` `[1,s,hd]` → broadcast over heads.
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
