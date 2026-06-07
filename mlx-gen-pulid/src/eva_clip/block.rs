//! EVA `Block`: pre-norm residual `x += attn(norm1(x)); x += mlp(norm2(x))` (no LayerScale —
//! EVA02-CLIP-L has `init_values=None`, postnorm=False). `norm1`/`norm2` are LayerNorm(weight+bias,
//! ε=1e-6). Port of `eva_vit_model.py Block`.

use mlx_rs::fast::layer_norm;
use mlx_rs::ops::add;
use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::eva_clip::attention::Attention;
use crate::eva_clip::mlp::SwiGlu;
use crate::eva_clip::rope::VisionRope;
use crate::eva_clip::{join, EPS};

pub struct Block {
    norm1_w: Array,
    norm1_b: Array,
    norm2_w: Array,
    norm2_b: Array,
    attn: Attention,
    mlp: SwiGlu,
}

impl Block {
    pub fn from_weights(w: &Weights, prefix: &str, num_heads: i32, head_dim: i32) -> Result<Self> {
        Ok(Self {
            norm1_w: w.require(&join(prefix, "norm1.weight"))?.clone(),
            norm1_b: w.require(&join(prefix, "norm1.bias"))?.clone(),
            norm2_w: w.require(&join(prefix, "norm2.weight"))?.clone(),
            norm2_b: w.require(&join(prefix, "norm2.bias"))?.clone(),
            attn: Attention::from_weights(w, &join(prefix, "attn"), num_heads, head_dim)?,
            mlp: SwiGlu::from_weights(w, &join(prefix, "mlp"))?,
        })
    }

    pub fn forward(&self, x: &Array, rope: &VisionRope) -> Result<Array> {
        let n1 = layer_norm(x, Some(&self.norm1_w), Some(&self.norm1_b), EPS)?;
        let x = add(x, &self.attn.forward(&n1, rope)?)?;
        let n2 = layer_norm(&x, Some(&self.norm2_w), Some(&self.norm2_b), EPS)?;
        Ok(add(&x, &self.mlp.forward(&n2)?)?)
    }
}
