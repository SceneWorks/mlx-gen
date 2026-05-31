//! Timestep → conditioning embedding. Port of the fork's `QwenTimeTextEmbed`: sinusoidal
//! `time_proj` (256, scale 1000) → `timestep_embedder` (linear_1 → SiLU → linear_2). `[B] → [B, inner]`.
//! Both Linears are [`AdaptableLinear`] (Q8-quantizable).

use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::timesteps::timestep_proj;
use super::{join, linear_from};

const PROJ_DIM: i32 = 256;
const SCALE: f32 = 1000.0;

pub struct TimeTextEmbed {
    linear_1: AdaptableLinear,
    linear_2: AdaptableLinear,
}

impl TimeTextEmbed {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let p = join(prefix, "timestep_embedder");
        Ok(Self {
            linear_1: linear_from(w, &join(&p, "linear_1"), true)?,
            linear_2: linear_from(w, &join(&p, "linear_2"), true)?,
        })
    }

    /// `timestep`: `[B]` f32 → `[B, inner]`.
    pub fn forward(&self, timestep: &Array) -> Result<Array> {
        let proj = timestep_proj(timestep, PROJ_DIM, SCALE)?;
        let x = silu(&self.linear_1.forward(&proj)?)?;
        self.linear_2.forward(&x)
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.linear_1.quantize(bits, None)?;
        self.linear_2.quantize(bits, None)?;
        Ok(())
    }
}
