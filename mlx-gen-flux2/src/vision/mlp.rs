//! Pixtral SwiGLU FFN: `down(silu(gate(x)) · up(x))`. Bias-less. Port of `PixtralMLP`.

use mlx_rs::ops::{matmul, multiply};
use mlx_rs::Array;

use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::text_encoder::join;

pub struct PixtralMlp {
    gate_w: Array,
    up_w: Array,
    down_w: Array,
}

impl PixtralMlp {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate_w: w.require(&join(prefix, "gate_proj.weight"))?.clone(),
            up_w: w.require(&join(prefix, "up_proj.weight"))?.clone(),
            down_w: w.require(&join(prefix, "down_proj.weight"))?.clone(),
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let gated = multiply(
            &silu(&matmul(x, self.gate_w.t())?)?,
            &matmul(x, self.up_w.t())?,
        )?;
        Ok(matmul(&gated, self.down_w.t())?)
    }
}
