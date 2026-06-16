//! `PixtralAttention`: **split** bias-less q/k/v/o → per-head split → **`rotate_half` 2-D RoPE (in
//! f32)** → block-diagonal SDPA (one window per reference image, via `cu_seqlens`) → bias-less o.
//! Port of `PixtralAttention`.
//!
//! Same shape as `mlx-gen-qwen-image`'s vision attention except the projections are four separate
//! bias-less Linears (Pixtral) rather than one fused biased `qkv` (Qwen). The RoPE is the
//! **non-interleaved** `rotate_half` form (`[-x2, x1]`), computed in f32 then cast back — distinct
//! from the MMDiT's interleaved-complex RoPE; they are NOT interchangeable.

use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::{add, concatenate_axis, matmul, multiply, split};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::text_encoder::join;

pub struct PixtralAttention {
    q_w: Array,
    k_w: Array,
    v_w: Array,
    o_w: Array,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl PixtralAttention {
    pub fn from_weights(w: &Weights, prefix: &str, num_heads: i32, head_dim: i32) -> Result<Self> {
        Ok(Self {
            q_w: w.require(&join(prefix, "q_proj.weight"))?.clone(),
            k_w: w.require(&join(prefix, "k_proj.weight"))?.clone(),
            v_w: w.require(&join(prefix, "v_proj.weight"))?.clone(),
            o_w: w.require(&join(prefix, "o_proj.weight"))?.clone(),
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// `x`: `[seq, embed]`; `cos`/`sin`: `[seq, head_dim]`; `cu`: cumulative seqlens
    /// (`[0, …, seq]`). `cu.len() > 2` ⇒ block-diagonal (per-image) attention.
    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array, cu: &[i32]) -> Result<Array> {
        let seq = x.shape()[0];
        let (h, hd) = (self.num_heads, self.head_dim);

        // [seq, embed] → matmul(x, Wᵀ) → [seq, h, hd] → [h, seq, hd] (bias-less projection).
        let to_heads = |proj_w: &Array| -> Result<Array> {
            Ok(matmul(x, proj_w.t())?
                .reshape(&[seq, h, hd])?
                .transpose_axes(&[1, 0, 2])?)
        };
        let q = apply_rope(&to_heads(&self.q_w)?, cos, sin)?;
        let k = apply_rope(&to_heads(&self.k_w)?, cos, sin)?;
        let v = to_heads(&self.v_w)?;

        let attn = if cu.len() > 2 {
            // block-diagonal: SDPA per image window, then concat back over the sequence axis.
            let mut outs = Vec::with_capacity(cu.len() - 1);
            for i in 0..cu.len() - 1 {
                let (off, len) = (cu[i], cu[i + 1] - cu[i]);
                let idx = Array::from_slice(&(off..off + len).collect::<Vec<i32>>(), &[len]);
                let qc = q.take_axis(&idx, 1)?.reshape(&[1, h, len, hd])?;
                let kc = k.take_axis(&idx, 1)?.reshape(&[1, h, len, hd])?;
                let vc = v.take_axis(&idx, 1)?.reshape(&[1, h, len, hd])?;
                let o = scaled_dot_product_attention(&qc, &kc, &vc, self.scale, None, None)?;
                outs.push(o.reshape(&[h, len, hd])?);
            }
            let refs: Vec<&Array> = outs.iter().collect();
            concatenate_axis(&refs, 1)? // [h, seq, hd]
        } else {
            let q4 = q.reshape(&[1, h, seq, hd])?;
            let k4 = k.reshape(&[1, h, seq, hd])?;
            let v4 = v.reshape(&[1, h, seq, hd])?;
            scaled_dot_product_attention(&q4, &k4, &v4, self.scale, None, None)?
                .reshape(&[h, seq, hd])?
        };

        let out = attn.transpose_axes(&[1, 0, 2])?.reshape(&[seq, h * hd])?;
        Ok(matmul(&out, self.o_w.t())?)
    }
}

/// `rotate_half` RoPE in f32: `x·cos + rotate_half(x)·sin`, `rotate_half(x) = [-x2, x1]` (split on
/// the head dim). `x`: `[h, seq, hd]`; `cos`/`sin`: `[seq, hd]` → broadcast over heads.
fn apply_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let orig = x.dtype();
    let xf = x.as_dtype(Dtype::Float32)?;
    let cos = cos.expand_dims(0)?.as_dtype(Dtype::Float32)?; // [1, seq, hd]
    let sin = sin.expand_dims(0)?.as_dtype(Dtype::Float32)?;
    let axis = (xf.shape().len() - 1) as i32;
    let halves = split(&xf, 2, axis)?;
    let rotated = concatenate_axis(&[&halves[1].negative()?, &halves[0]], axis)?;
    let out = add(&multiply(&xf, &cos)?, &multiply(&rotated, &sin)?)?;
    Ok(out.as_dtype(orig)?)
}
