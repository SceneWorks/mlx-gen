//! `VisionMlp`: a biased SwiGLU MLP (`down(silu(gate(x)) · up(x))`). Port of the fork's
//! `qwen_vision_mlp.py`.

use mlx_rs::ops::multiply;
use mlx_rs::Array;

use mlx_gen::nn::{linear, silu};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::text_encoder::join;

pub struct VisionMlp {
    gate_w: Array,
    gate_b: Array,
    up_w: Array,
    up_b: Array,
    down_w: Array,
    down_b: Array,
}

impl VisionMlp {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate_w: w.require(&join(prefix, "gate_proj.weight"))?.clone(),
            gate_b: w.require(&join(prefix, "gate_proj.bias"))?.clone(),
            up_w: w.require(&join(prefix, "up_proj.weight"))?.clone(),
            up_b: w.require(&join(prefix, "up_proj.bias"))?.clone(),
            down_w: w.require(&join(prefix, "down_proj.weight"))?.clone(),
            down_b: w.require(&join(prefix, "down_proj.bias"))?.clone(),
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let gate = silu(&linear(x, &self.gate_w, &self.gate_b)?)?;
        let up = linear(x, &self.up_w, &self.up_b)?;
        linear(&multiply(&gate, &up)?, &self.down_w, &self.down_b)
    }
}
