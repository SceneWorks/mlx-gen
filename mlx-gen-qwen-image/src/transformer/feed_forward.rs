//! Per-stream feed-forward: `mlp_out(gelu_approx(mlp_in(x)))` (both biased, 4× expansion).
//! Port of the fork's `QwenFeedForward`. Both Linears are [`AdaptableLinear`] so the transformer
//! can be quantized (Q8) without changing the forward.

use mlx_rs::nn::gelu_approximate;
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, linear_from};

pub struct FeedForward {
    mlp_in: AdaptableLinear,
    mlp_out: AdaptableLinear,
}

impl FeedForward {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            mlp_in: linear_from(w, &join(prefix, "mlp_in"), true)?,
            mlp_out: linear_from(w, &join(prefix, "mlp_out"), true)?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let h = gelu_approximate(self.mlp_in.forward(x)?)?;
        self.mlp_out.forward(&h)
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.mlp_in.quantize(bits, None)?;
        self.mlp_out.quantize(bits, None)?;
        Ok(())
    }
}
