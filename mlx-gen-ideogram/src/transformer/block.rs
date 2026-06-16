//! Ideogram 4 DiT block: attention + SwiGLU MLP with AdaLN "sandwich" norms (a pre-norm scaled by
//! `1+scale`, a post-norm gated by `tanh(gate)`), full segment-masked attention, per-head q/k
//! RMSNorm, and interleaved 3D MRoPE. Port of `Ideogram4Attention` / `Ideogram4MLP` /
//! `Ideogram4TransformerBlock`.

use mlx_rs::fast::{rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, concatenate_axis, multiply, split, tanh};
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, lin};

/// `1.0 + a`, broadcasting the scalar.
fn plus1(a: &Array) -> Result<Array> {
    Ok(add(a, Array::from_f32(1.0))?)
}

// ── Attention ────────────────────────────────────────────────────────────────────────────
pub struct Ideogram4Attention {
    qkv: AdaptableLinear,
    o: AdaptableLinear,
    norm_q: Array,
    norm_k: Array,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
    eps: f32,
}

impl Ideogram4Attention {
    pub fn from_weights(w: &Weights, prefix: &str, num_heads: i32, head_dim: i32) -> Result<Self> {
        Ok(Self {
            qkv: lin(w, &join(prefix, "qkv"), false)?,
            o: lin(w, &join(prefix, "o"), false)?,
            norm_q: w.require(&join(prefix, "norm_q.weight"))?.clone(),
            norm_k: w.require(&join(prefix, "norm_k.weight"))?.clone(),
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            eps: 1e-5,
        })
    }

    /// `x`: `[B, L, hidden]`; `cos`/`sin`: `[B, L, head_dim]`; `mask`: additive `[B, 1, L, L]`.
    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array, mask: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);

        // qkv → [B, L, 3, H, hd] → q,k,v [B, L, H, hd]
        let qkv = self
            .qkv
            .forward(x)?
            .reshape(&[b, s, 3, self.num_heads, self.head_dim])?;
        let parts = split(&qkv, 3, 2)?;
        let q = parts[0].reshape(&[b, s, self.num_heads, self.head_dim])?;
        let k = parts[1].reshape(&[b, s, self.num_heads, self.head_dim])?;
        let v = parts[2].reshape(&[b, s, self.num_heads, self.head_dim])?;

        // Per-head q/k RMSNorm over the head dim, before transpose + RoPE.
        let q = rms_norm(&q, &self.norm_q, self.eps)?;
        let k = rms_norm(&k, &self.norm_k, self.eps)?;

        // [B,L,H,hd] → [B,H,L,hd]
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;

        let mask = mask.as_dtype(q.dtype())?;
        let o = scaled_dot_product_attention(&q, &k, &v, self.scale, &mask, None)?;
        let o =
            o.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, s, self.num_heads * self.head_dim])?;
        self.o.forward(&o)
    }
}

/// HF half-split RoPE in `[B, H, L, hd]` layout: `cos`/`sin` `[B, L, hd]` → broadcast over heads.
fn apply_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let cos = cos.expand_dims(1)?; // [B,1,L,hd]
    let sin = sin.expand_dims(1)?;
    let parts = split(x, 2, 3)?;
    let rot = concatenate_axis(&[&parts[1].negative()?, &parts[0]], 3)?;
    Ok(add(&multiply(x, &cos)?, &multiply(&rot, &sin)?)?)
}

// ── SwiGLU MLP ───────────────────────────────────────────────────────────────────────────
pub struct Ideogram4Mlp {
    w1: AdaptableLinear,
    w2: AdaptableLinear,
    w3: AdaptableLinear,
}

impl Ideogram4Mlp {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w1: lin(w, &join(prefix, "w1"), false)?,
            w2: lin(w, &join(prefix, "w2"), false)?,
            w3: lin(w, &join(prefix, "w3"), false)?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let gated = multiply(&silu(&self.w1.forward(x)?)?, &self.w3.forward(x)?)?;
        self.w2.forward(&gated)
    }
}

// ── Block ────────────────────────────────────────────────────────────────────────────────
pub struct Ideogram4Block {
    attention: Ideogram4Attention,
    feed_forward: Ideogram4Mlp,
    attention_norm1: Array,
    attention_norm2: Array,
    ffn_norm1: Array,
    ffn_norm2: Array,
    adaln_modulation: AdaptableLinear,
    eps: f32,
}

impl Ideogram4Block {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_heads: i32,
        head_dim: i32,
        norm_eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            attention: Ideogram4Attention::from_weights(
                w,
                &join(prefix, "attention"),
                num_heads,
                head_dim,
            )?,
            feed_forward: Ideogram4Mlp::from_weights(w, &join(prefix, "feed_forward"))?,
            attention_norm1: w.require(&join(prefix, "attention_norm1.weight"))?.clone(),
            attention_norm2: w.require(&join(prefix, "attention_norm2.weight"))?.clone(),
            ffn_norm1: w.require(&join(prefix, "ffn_norm1.weight"))?.clone(),
            ffn_norm2: w.require(&join(prefix, "ffn_norm2.weight"))?.clone(),
            adaln_modulation: lin(w, &join(prefix, "adaln_modulation"), true)?,
            eps: norm_eps,
        })
    }

    /// `x`: `[B, L, hidden]`; `adaln_input`: `[B, 1, adaln_dim]`; `cos`/`sin`: `[B, L, head_dim]`;
    /// `mask`: additive `[B, 1, L, L]`.
    pub fn forward(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        mask: &Array,
        adaln_input: &Array,
    ) -> Result<Array> {
        let mod_ = self.adaln_modulation.forward(adaln_input)?; // [B,1,4*hidden]
        let chunks = split(&mod_, 4, 2)?;
        let (scale_msa, gate_msa, scale_mlp, gate_mlp) =
            (&chunks[0], &chunks[1], &chunks[2], &chunks[3]);
        let gate_msa = tanh(gate_msa)?;
        let gate_mlp = tanh(gate_mlp)?;
        let scale_msa = plus1(scale_msa)?;
        let scale_mlp = plus1(scale_mlp)?;

        let normed = multiply(&rms_norm(x, &self.attention_norm1, self.eps)?, &scale_msa)?;
        let attn_out = self.attention.forward(&normed, cos, sin, mask)?;
        let x = add(
            x,
            &multiply(
                &gate_msa,
                &rms_norm(&attn_out, &self.attention_norm2, self.eps)?,
            )?,
        )?;

        let normed2 = multiply(&rms_norm(&x, &self.ffn_norm1, self.eps)?, &scale_mlp)?;
        let ff = self.feed_forward.forward(&normed2)?;
        let x = add(
            &x,
            &multiply(&gate_mlp, &rms_norm(&ff, &self.ffn_norm2, self.eps)?)?,
        )?;
        Ok(x)
    }
}
