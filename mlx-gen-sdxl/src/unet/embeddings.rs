//! UNet time/conditioning embeddings: the `SinusoidalPositionalEncoding` (used for both the
//! timestep and the SDXL `text_time` micro-conditioning) and the 2-layer `TimestepEmbedding` MLP.
//! Faithful ports of the vendored `unet.py` + mlx `nn.SinusoidalPositionalEncoding`.

use std::f32::consts::PI;

use mlx_rs::ops::{add, concatenate_axis, cos, exp, multiply, sin};
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::array::scalar;
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// Port of mlx `nn.SinusoidalPositionalEncoding`. Precomputes `sigmas` from `(dims, min_freq,
/// max_freq, full_turns)` and on call returns `concat([cos(x·σ), sin(x·σ)])` (when `cos_first`)
/// scaled by `scale`.
///
/// The `sigmas` table is computed with **MLX ops** (not host `f32::exp`) so it is bit-identical to
/// the reference `mx.exp(one_zero·(max−min)+min)` — a host-exp `sigmas` differs at the ULP level,
/// and that ~2e-4 timestep-embedding error chaotically amplifies under CFG=7 ancestral sampling
/// (sc-2400 S5). The `min_freq`/`max_freq` logs are taken in f64 (matching the reference's
/// `math.log`) before the f32 array math.
pub struct SinusoidalPositionalEncoding {
    sigmas: Array, // [dims/2]
    scale: f32,
    cos_first: bool,
}

impl SinusoidalPositionalEncoding {
    pub fn new(
        dims: i32,
        min_freq: f64,
        max_freq: f64,
        scale: f32,
        cos_first: bool,
        full_turns: bool,
    ) -> Result<Self> {
        let half = dims / 2;
        // one_zero = 1 - arange(half)/(half-1)  (f32 array, matching mx.arange int→float division).
        let arange: Vec<f32> = (0..half).map(|i| i as f32).collect();
        let arange = Array::from_slice(&arange, &[half]);
        let one_zero = mlx_rs::ops::subtract(
            scalar(1.0),
            &mlx_rs::ops::divide(&arange, scalar((half - 1) as f32))?,
        )?;
        // sigmas = exp(one_zero·(max_l − min_l) + min_l), logs in f64 then cast to f32 (as the
        // reference does: math.log is f64, the mlx scalar broadcast casts to f32).
        let min_l = min_freq.ln();
        let max_l = max_freq.ln();
        let scaled = add(
            &multiply(&one_zero, scalar((max_l - min_l) as f32))?,
            scalar(min_l as f32),
        )?;
        let mut sigmas = exp(&scaled)?;
        if full_turns {
            sigmas = multiply(&sigmas, scalar(2.0 * PI))?;
        }
        Ok(Self {
            sigmas,
            scale,
            cos_first,
        })
    }

    /// The SDXL timestep encoder: `SinusoidalPositionalEncoding(dim, min_freq=exp(-ln(10000) +
    /// 2·ln(10000)/dim), max_freq=1, scale=1, cos_first=true, full_turns=false)` — the vendored
    /// `UNetModel.timesteps` / `add_time_proj` construction.
    pub fn timestep(dim: i32) -> Result<Self> {
        let ln10000 = 10000f64.ln();
        let min_freq = (-ln10000 + 2.0 * ln10000 / dim as f64).exp();
        Self::new(dim, min_freq, 1.0, 1.0, true, false)
    }

    /// `x[..., None] * sigmas → concat([cos, sin], -1)` (cos-first) `· scale`. `x` is any shape;
    /// output appends a `dims` axis.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let axis = x.shape().len() as i32; // append a trailing axis
        let y = multiply(&x.expand_dims(axis)?, &self.sigmas)?;
        let cosy = cos(&y)?;
        let siny = sin(&y)?;
        let order: [&Array; 2] = if self.cos_first {
            [&cosy, &siny]
        } else {
            [&siny, &cosy]
        };
        let mut y = concatenate_axis(&order, -1)?;
        if self.scale != 1.0 {
            y = multiply(&y, scalar(self.scale))?;
        }
        Ok(y)
    }
}

/// The 2-layer time-embedding MLP (`linear_1 → SiLU → linear_2`). Used both for the timestep
/// (`time_embedding`) and the SDXL added-conditioning (`add_embedding`).
pub struct TimestepEmbedding {
    linear1: AdaptableLinear,
    linear2: AdaptableLinear,
}

impl TimestepEmbedding {
    /// `prefix` is e.g. `time_embedding` or `add_embedding`; leaves are `linear_1`/`linear_2`.
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let dense = |n: &str| -> Result<AdaptableLinear> {
            Ok(AdaptableLinear::dense(
                w.require(&format!("{prefix}.{n}.weight"))?.clone(),
                Some(w.require(&format!("{prefix}.{n}.bias"))?.clone()),
            ))
        };
        Ok(Self {
            linear1: dense("linear_1")?,
            linear2: dense("linear_2")?,
        })
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.linear1.quantize(bits, None)?;
        self.linear2.quantize(bits, None)?;
        Ok(())
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let x = self.linear1.forward(x)?;
        let x = silu(&x)?;
        self.linear2.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinusoidal_shape_and_cos_first() {
        // dims=8 -> output appends an 8-wide axis; cos-first means first half is cos.
        let enc = SinusoidalPositionalEncoding::timestep(8).unwrap();
        let x = Array::from_slice(&[0.0f32, 1.0], &[2]);
        let y = enc.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 8]);
        // At x=0: cos(0)=1 for the first half, sin(0)=0 for the second.
        let row0 = y.reshape(&[2, 8]).unwrap();
        let s = row0.as_slice::<f32>();
        for v in &s[0..4] {
            assert!((v - 1.0).abs() < 1e-6, "cos(0) half should be 1");
        }
        for v in &s[4..8] {
            assert!(v.abs() < 1e-6, "sin(0) half should be 0");
        }
    }
}
