//! Qwen3 SwiGLU MLP: `down(silu(gate(x)) · up(x))`. No biases.

use mlx_rs::ops::multiply;
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, lin};

pub struct Qwen3Mlp {
    gate: AdaptableLinear,
    up: AdaptableLinear,
    down: AdaptableLinear,
}

impl Qwen3Mlp {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate: lin(w, &join(prefix, "gate_proj.weight"))?,
            up: lin(w, &join(prefix, "up_proj.weight"))?,
            down: lin(w, &join(prefix, "down_proj.weight"))?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let gated = multiply(&silu(&self.gate.forward(x)?)?, &self.up.forward(x)?)?;
        self.down.forward(&gated)
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.gate.quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        self.up.quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        self.down.quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        Ok(())
    }
}
