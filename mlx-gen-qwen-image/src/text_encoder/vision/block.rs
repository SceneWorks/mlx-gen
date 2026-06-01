//! `VisionBlock`: pre-norm residual block — `x += attn(rms_norm(x)); x += mlp(rms_norm(x))`, both
//! norms RMSNorm(ε=1e-6). Port of the fork's `qwen_vision_block.py`.

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::add;
use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{VisionAttention, VisionMlp};
use crate::text_encoder::join;

const EPS: f32 = 1e-6;

pub struct VisionBlock {
    norm1: Array,
    norm2: Array,
    attn: VisionAttention,
    mlp: VisionMlp,
}

impl VisionBlock {
    pub fn from_weights(w: &Weights, prefix: &str, num_heads: i32, head_dim: i32) -> Result<Self> {
        Ok(Self {
            norm1: w.require(&join(prefix, "norm1.weight"))?.clone(),
            norm2: w.require(&join(prefix, "norm2.weight"))?.clone(),
            attn: VisionAttention::from_weights(w, &join(prefix, "attn"), num_heads, head_dim)?,
            mlp: VisionMlp::from_weights(w, &join(prefix, "mlp"))?,
        })
    }

    /// `x`: `[seq, embed]`; `cos`/`sin`: `[seq, head_dim]`; `cu`: this block's window seqlens.
    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array, cu: &[i32]) -> Result<Array> {
        let attn = self
            .attn
            .forward(&rms_norm(x, &self.norm1, EPS)?, cos, sin, cu)?;
        let x = add(x, &attn)?;
        let mlp = self.mlp.forward(&rms_norm(&x, &self.norm2, EPS)?)?;
        Ok(add(&x, &mlp)?)
    }
}
