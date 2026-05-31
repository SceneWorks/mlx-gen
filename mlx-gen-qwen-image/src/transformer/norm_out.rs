//! `AdaLayerNormContinuous` (FLUX-style): affine-less LayerNorm scaled/shifted by a conditioning
//! projection. `linear(silu(c))` → `[scale | shift]`; `x = norm(x)·(1+scale) + shift`. The fork's
//! linear is **bias-less** (`bias=False`); the diffusers checkpoint ships a `linear.bias` but the
//! fork ignores it. The Linear is an [`AdaptableLinear`] (Q8-quantizable).

use mlx_rs::fast::layer_norm;
use mlx_rs::ops::{add, multiply, split};
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, linear_from};

const LN_EPS: f32 = 1e-6;

pub struct AdaLayerNormContinuous {
    linear: AdaptableLinear,
}

impl AdaLayerNormContinuous {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            linear: linear_from(w, &join(prefix, "linear"), false)?,
        })
    }

    /// `x`: `[B, S, H]`, `c` (conditioning): `[B, H]` → `[B, S, H]`.
    pub fn forward(&self, x: &Array, c: &Array) -> Result<Array> {
        let mod_params = self.linear.forward(&silu(c)?)?; // [B, 2H], no bias
        let parts = split(&mod_params, 2, 1)?; // scale, shift each [B, H]
        let scale = add(&parts[0], Array::from_slice(&[1.0f32], &[1]))?.expand_dims(1)?; // [B,1,H]
        let shift = parts[1].expand_dims(1)?;
        let normed = layer_norm(x, None, None, LN_EPS)?;
        Ok(add(&multiply(&normed, &scale)?, &shift)?)
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.linear.quantize(bits, None)
    }
}
