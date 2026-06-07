//! EVA `SwiGLU` FFN with sub-LN: `w3( ffn_ln( silu(w1·x) * (w2·x) ) )`. Port of
//! `eva_vit_model.py SwiGLU` (naiveswiglu + subln). All three linears are biased; `ffn_ln` is a
//! LayerNorm over the hidden dim (2730).

use mlx_rs::fast::layer_norm;
use mlx_rs::ops::multiply;
use mlx_rs::Array;

use mlx_gen::nn::{linear, silu};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::eva_clip::{join, EPS};

pub struct SwiGlu {
    w1_w: Array,
    w1_b: Array,
    w2_w: Array,
    w2_b: Array,
    ffn_ln_w: Array,
    ffn_ln_b: Array,
    w3_w: Array,
    w3_b: Array,
}

impl SwiGlu {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w1_w: w.require(&join(prefix, "w1.weight"))?.clone(),
            w1_b: w.require(&join(prefix, "w1.bias"))?.clone(),
            w2_w: w.require(&join(prefix, "w2.weight"))?.clone(),
            w2_b: w.require(&join(prefix, "w2.bias"))?.clone(),
            ffn_ln_w: w.require(&join(prefix, "ffn_ln.weight"))?.clone(),
            ffn_ln_b: w.require(&join(prefix, "ffn_ln.bias"))?.clone(),
            w3_w: w.require(&join(prefix, "w3.weight"))?.clone(),
            w3_b: w.require(&join(prefix, "w3.bias"))?.clone(),
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let x1 = linear(x, &self.w1_w, &self.w1_b)?;
        let x2 = linear(x, &self.w2_w, &self.w2_b)?;
        let hidden = multiply(&silu(&x1)?, &x2)?;
        let hidden = layer_norm(&hidden, Some(&self.ffn_ln_w), Some(&self.ffn_ln_b), EPS)?;
        linear(&hidden, &self.w3_w, &self.w3_b)
    }
}
