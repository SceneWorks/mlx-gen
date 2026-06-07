//! SVD time/frame embeddings — the diffusers `Timesteps` sinusoidal encoder + the 2-layer
//! `TimestepEmbedding` MLP, shared by the UNet (timestep + `added_time_ids`) and each
//! `TransformerSpatioTemporalModel` (per-frame `time_pos_embed`).

use mlx_rs::ops::{concatenate_axis, cos, divide, exp, multiply, sin};
use mlx_rs::Array;

use mlx_gen::array::scalar;
use mlx_gen::nn::{linear, silu};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// diffusers `get_timestep_embedding(x, dim, flip_sin_to_cos=True, downscale_freq_shift=0,
/// max_period=10000)`: `freq_i = 10000^(−i/half)` (`i∈[0,half)`), `emb = x[:,None]·freq`, output
/// `concat([cos(emb), sin(emb)], -1)` (cos first). `x` is `[N]` → returns `[N, dim]`. The `ln(10000)`
/// is taken in f64 (matching `math.log`) and the steps mirror diffusers (`(−ln·arange)/half`) so the
/// f32 rounding matches.
pub fn sinusoidal_timestep(x: &Array, dim: i32) -> Result<Array> {
    let half = dim / 2;
    let arange: Vec<f32> = (0..half).map(|i| i as f32).collect();
    let arange = Array::from_slice(&arange, &[half]);
    let neg_ln = -(10000f64.ln());
    let exponent = divide(
        &multiply(&arange, scalar(neg_ln as f32))?,
        scalar(half as f32),
    )?;
    let freqs = exp(&exponent)?; // [half]
    let axis = x.shape().len() as i32;
    let emb = multiply(&x.expand_dims(axis)?, &freqs)?; // [N, half]
    Ok(concatenate_axis(&[&cos(&emb)?, &sin(&emb)?], -1)?)
}

/// The 2-layer time-embedding MLP (`linear_1 → SiLU → linear_2`). `out_dim` differs from the input
/// only for the transformer `time_pos_embed` (C→C·4→C); the UNet's `time_embedding`/`add_embedding`
/// map into the 1280-wide embedding. Dims come from the loaded weights.
pub struct TimestepEmbedding {
    lin1: (Array, Array),
    lin2: (Array, Array),
}

impl TimestepEmbedding {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let lin = |n: &str| -> Result<(Array, Array)> {
            Ok((
                w.require(&format!("{prefix}.{n}.weight"))?.clone(),
                w.require(&format!("{prefix}.{n}.bias"))?.clone(),
            ))
        };
        Ok(Self {
            lin1: lin("linear_1")?,
            lin2: lin("linear_2")?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let x = linear(x, &self.lin1.0, &self.lin1.1)?;
        let x = silu(&x)?;
        linear(&x, &self.lin2.0, &self.lin2.1)
    }
}
