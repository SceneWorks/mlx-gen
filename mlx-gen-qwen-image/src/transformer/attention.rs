//! Joint (dual-stream) attention. Port of the fork's `QwenAttention`: separate q/k/v projections
//! for the image (`to_*`) and text (`add_*_proj`) streams, per-head q/k RMSNorm, **interleaved**
//! complex RoPE, then attention over the concatenated `[txt, img]` sequence, split back into the
//! two streams and projected (`attn_to_out.0` / `to_add_out`). All eight projections are
//! [`AdaptableLinear`] (Q8-quantizable); the q/k RMSNorm weights stay dense.

use mlx_rs::fast::{rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, concatenate_axis, multiply, split, subtract};
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, linear_from};

const RMS_EPS: f32 = 1e-6;

pub struct QwenJointAttention {
    to_q: AdaptableLinear,
    to_k: AdaptableLinear,
    to_v: AdaptableLinear,
    add_q: AdaptableLinear,
    add_k: AdaptableLinear,
    add_v: AdaptableLinear,
    to_out: AdaptableLinear,
    to_add_out: AdaptableLinear,
    norm_q: Array,
    norm_k: Array,
    norm_added_q: Array,
    norm_added_k: Array,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl QwenJointAttention {
    pub fn from_weights(w: &Weights, prefix: &str, num_heads: i32, head_dim: i32) -> Result<Self> {
        let g = |s: &str| w.require(&join(prefix, s)).cloned();
        Ok(Self {
            to_q: linear_from(w, &join(prefix, "to_q"), true)?,
            to_k: linear_from(w, &join(prefix, "to_k"), true)?,
            to_v: linear_from(w, &join(prefix, "to_v"), true)?,
            add_q: linear_from(w, &join(prefix, "add_q_proj"), true)?,
            add_k: linear_from(w, &join(prefix, "add_k_proj"), true)?,
            add_v: linear_from(w, &join(prefix, "add_v_proj"), true)?,
            to_out: linear_from(w, &join(prefix, "attn_to_out.0"), true)?,
            to_add_out: linear_from(w, &join(prefix, "to_add_out"), true)?,
            norm_q: g("norm_q.weight")?,
            norm_k: g("norm_k.weight")?,
            norm_added_q: g("norm_added_q.weight")?,
            norm_added_k: g("norm_added_k.weight")?,
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.to_q.quantize(bits, None)?;
        self.to_k.quantize(bits, None)?;
        self.to_v.quantize(bits, None)?;
        self.add_q.quantize(bits, None)?;
        self.add_k.quantize(bits, None)?;
        self.add_v.quantize(bits, None)?;
        self.to_out.quantize(bits, None)?;
        self.to_add_out.quantize(bits, None)?;
        Ok(())
    }

    /// `img`/`txt`: `[B, seq, dim]`; rope tables `[seq, head_dim/2]`; `mask`: optional additive
    /// `[B, 1, 1, txt+img]`. Returns `(img_attn, txt_attn)`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        img: &Array,
        txt: &Array,
        img_cos: &Array,
        img_sin: &Array,
        txt_cos: &Array,
        txt_sin: &Array,
        mask: Option<&Array>,
    ) -> Result<(Array, Array)> {
        let (b, img_seq) = (img.shape()[0], img.shape()[1]);
        let txt_seq = txt.shape()[1];
        let (h, hd) = (self.num_heads, self.head_dim);
        let to_heads = |lin: &AdaptableLinear, x: &Array, seq: i32| -> Result<Array> {
            Ok(lin.forward(x)?.reshape(&[b, seq, h, hd])?)
        };

        let img_q = rms_norm(&to_heads(&self.to_q, img, img_seq)?, &self.norm_q, RMS_EPS)?;
        let img_k = rms_norm(&to_heads(&self.to_k, img, img_seq)?, &self.norm_k, RMS_EPS)?;
        let img_v = to_heads(&self.to_v, img, img_seq)?;
        let txt_q = rms_norm(
            &to_heads(&self.add_q, txt, txt_seq)?,
            &self.norm_added_q,
            RMS_EPS,
        )?;
        let txt_k = rms_norm(
            &to_heads(&self.add_k, txt, txt_seq)?,
            &self.norm_added_k,
            RMS_EPS,
        )?;
        let txt_v = to_heads(&self.add_v, txt, txt_seq)?;

        let img_q = apply_rope_qwen(&img_q, img_cos, img_sin)?;
        let img_k = apply_rope_qwen(&img_k, img_cos, img_sin)?;
        let txt_q = apply_rope_qwen(&txt_q, txt_cos, txt_sin)?;
        let txt_k = apply_rope_qwen(&txt_k, txt_cos, txt_sin)?;

        // joint [txt, img] over the sequence axis, then to [B, heads, seq, head_dim] for SDPA.
        let q = concatenate_axis(&[&txt_q, &img_q], 1)?.transpose_axes(&[0, 2, 1, 3])?;
        let k = concatenate_axis(&[&txt_k, &img_k], 1)?.transpose_axes(&[0, 2, 1, 3])?;
        let v = concatenate_axis(&[&txt_v, &img_v], 1)?.transpose_axes(&[0, 2, 1, 3])?;

        let o = match mask {
            Some(m) => scaled_dot_product_attention(&q, &k, &v, self.scale, m, None)?,
            None => scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?,
        };
        let joint = txt_seq + img_seq;
        let o = o
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, joint, h * hd])?;

        // split back along the sequence axis: text first, then image.
        let txt_idx = Array::from_slice(&(0..txt_seq).collect::<Vec<i32>>(), &[txt_seq]);
        let img_idx = Array::from_slice(&(txt_seq..joint).collect::<Vec<i32>>(), &[img_seq]);
        let txt_attn = self.to_add_out.forward(&o.take_axis(&txt_idx, 1)?)?;
        let img_attn = self.to_out.forward(&o.take_axis(&img_idx, 1)?)?;
        Ok((img_attn, txt_attn))
    }
}

/// Interleaved complex RoPE: pairs `(x_2i, x_2i+1)` rotated by `(cos_i, sin_i)`. `x`:
/// `[B, seq, heads, head_dim]`; `cos`/`sin`: `[seq, head_dim/2]`.
fn apply_rope_qwen(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let sh = x.shape();
    let (b, seq, heads, hd) = (sh[0], sh[1], sh[2], sh[3]);
    let half = hd / 2;
    let x5 = x.reshape(&[b, seq, heads, half, 2])?;
    let parts = split(&x5, 2, 4)?; // even/odd lanes, each [B,seq,heads,half,1]
    let xr = parts[0].reshape(&[b, seq, heads, half])?;
    let xi = parts[1].reshape(&[b, seq, heads, half])?;
    let cos = cos.reshape(&[1, seq, 1, half])?;
    let sin = sin.reshape(&[1, seq, 1, half])?;
    let out_r = subtract(&multiply(&xr, &cos)?, &multiply(&xi, &sin)?)?;
    let out_i = add(&multiply(&xr, &sin)?, &multiply(&xi, &cos)?)?;
    let stacked = concatenate_axis(&[&out_r.expand_dims(4)?, &out_i.expand_dims(4)?], 4)?;
    Ok(stacked.reshape(&[b, seq, heads, hd])?)
}
