//! `VisionAttention`: biased fused `qkv` → per-head split → **`rotate_half` 2-D RoPE (in f32)** →
//! block-diagonal SDPA (one chunk per window, via `cu_seqlens`) → biased `proj`. Port of the fork's
//! `qwen_vision_attention.py`.
//!
//! The RoPE here is the **non-interleaved** `rotate_half` form (`[-x2, x1]`), distinct from the
//! MMDiT's interleaved-complex RoPE — they are NOT interchangeable. Computed in f32 then cast back,
//! matching the fork.

use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::{add, concatenate_axis, multiply, split};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::linear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::text_encoder::join;

pub struct VisionAttention {
    qkv_w: Array,
    qkv_b: Array,
    proj_w: Array,
    proj_b: Array,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl VisionAttention {
    pub fn from_weights(w: &Weights, prefix: &str, num_heads: i32, head_dim: i32) -> Result<Self> {
        Ok(Self {
            qkv_w: w.require(&join(prefix, "qkv.weight"))?.clone(),
            qkv_b: w.require(&join(prefix, "qkv.bias"))?.clone(),
            proj_w: w.require(&join(prefix, "proj.weight"))?.clone(),
            proj_b: w.require(&join(prefix, "proj.bias"))?.clone(),
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// `x`: `[seq, embed]`; `cos`/`sin`: `[seq, head_dim]`; `cu`: cumulative seqlens for this block's
    /// attention windows (`[0, …, seq]`). `cu.len() > 2` ⇒ block-diagonal (windowed) attention.
    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array, cu: &[i32]) -> Result<Array> {
        let seq = x.shape()[0];
        let (h, hd) = (self.num_heads, self.head_dim);

        let qkv = linear(x, &self.qkv_w, &self.qkv_b)?.reshape(&[seq, 3, h, hd])?;
        let parts = split(&qkv, 3, 1)?; // each [seq, 1, h, hd]
        let to_heads = |p: &Array| -> Result<Array> {
            // [seq, 1, h, hd] → [h, seq, hd]
            Ok(p.reshape(&[seq, h, hd])?.transpose_axes(&[1, 0, 2])?)
        };
        let q = apply_rope(&to_heads(&parts[0])?, cos, sin)?;
        let k = apply_rope(&to_heads(&parts[1])?, cos, sin)?;
        let v = to_heads(&parts[2])?;

        let attn = if cu.len() > 2 {
            // block-diagonal: SDPA per window, then concat back over the sequence axis.
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
        linear(&out, &self.proj_w, &self.proj_b)
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
