//! Pixtral transformer block (pre-norm residual): `h += attn(attention_norm(h))`, then
//! `h += ffn(ffn_norm(h))`. Both norms are RMSNorm. Port of `PixtralAttentionLayer`.

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::add;
use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{PixtralAttention, PixtralMlp, PixtralVisionConfig};
use crate::text_encoder::join;

pub struct PixtralBlock {
    attention_norm: Array,
    ffn_norm: Array,
    attn: PixtralAttention,
    ffn: PixtralMlp,
    eps: f32,
}

impl PixtralBlock {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &PixtralVisionConfig) -> Result<Self> {
        Ok(Self {
            attention_norm: w.require(&join(prefix, "attention_norm.weight"))?.clone(),
            ffn_norm: w.require(&join(prefix, "ffn_norm.weight"))?.clone(),
            attn: PixtralAttention::from_weights(
                w,
                &join(prefix, "attention"),
                cfg.num_heads,
                cfg.head_dim,
            )?,
            ffn: PixtralMlp::from_weights(w, &join(prefix, "feed_forward"))?,
            eps: cfg.rms_norm_eps,
        })
    }

    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array, cu: &[i32]) -> Result<Array> {
        let normed = rms_norm(x, &self.attention_norm, self.eps)?;
        let h = add(x, &self.attn.forward(&normed, cos, sin, cu)?)?;
        let normed2 = rms_norm(&h, &self.ffn_norm, self.eps)?;
        Ok(add(&h, &self.ffn.forward(&normed2)?)?)
    }
}
