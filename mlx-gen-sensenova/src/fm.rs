//! Flow-matching head, timestep/noise-scale embedders, and the FM sampler math (sc-3184).
//!
//! For the 8B-MoT checkpoint (`use_pixel_head=false`, `fm_head_layers=2`) the FM head is a plain
//! `Linear → erf-GELU → Linear` (no AdaLN/ResBlocks — those are the deep `FlowMatchingHead` for
//! other configs). [`TimestepEmbedder`] (GLIDE sinusoidal → SiLU MLP) backs both the
//! `timestep_embedder` and the `noise_scale_embedder`. The sampler is the reference's
//! `_apply_time_schedule` / `_euler_step` / velocity formula and `patchify`/`unpatchify`.
//!
//! These are the weight-bearing + pure pieces of image generation; the full denoise loop that
//! threads them through the AR backbone lands in sc-3187/sc-3188.
//!
//! NOTE: `_apply_time_schedule` overwrites `self.time_schedule = "standard"` on entry, so the
//! "dynamic"/`_calculate_dynamic_mu` branch is dead code — the effective schedule is always
//! `σ = shift·σ / (1 + (shift−1)·σ)` with `σ = 1−t`. [`apply_time_schedule`] implements that.

use mlx_rs::ops::{add, concatenate_axis, divide, matmul, multiply, subtract};
use mlx_rs::Array;

use mlx_gen::nn::{gelu_exact, linear, silu};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

fn require(w: &Weights, key: &str) -> Result<Array> {
    Ok(w.require(key)?.clone())
}

/// The shallow flow-matching head: `Linear → erf-GELU → Linear`. Maps a generation-path hidden
/// state `[…, llm_hidden]` to a patch latent `[…, 3·(patch·merge)²]`.
pub struct FmHead {
    l0_w: Array,
    l0_b: Array,
    l2_w: Array,
    l2_b: Array,
}

impl FmHead {
    /// `prefix` = e.g. `"fm_modules.fm_head"` (Sequential indices 0 = first Linear, 2 = second).
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            l0_w: require(w, &format!("{prefix}.0.weight"))?,
            l0_b: require(w, &format!("{prefix}.0.bias"))?,
            l2_w: require(w, &format!("{prefix}.2.weight"))?,
            l2_b: require(w, &format!("{prefix}.2.bias"))?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let h = gelu_exact(&linear(x, &self.l0_w, &self.l0_b)?)?;
        linear(&h, &self.l2_w, &self.l2_b)
    }
}

/// GLIDE-style sinusoidal timestep embedding → 2-layer SiLU MLP. Used for both the timestep and the
/// noise-scale conditioning (`fm_modules.{timestep_embedder,noise_scale_embedder}`).
pub struct TimestepEmbedder {
    mlp0_w: Array,
    mlp0_b: Array,
    mlp2_w: Array,
    mlp2_b: Array,
    freq_size: i32,
}

impl TimestepEmbedder {
    /// `prefix` = e.g. `"fm_modules.timestep_embedder"`. `frequency_embedding_size` is 256 in the
    /// reference (`TimestepEmbedder` default).
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            mlp0_w: require(w, &format!("{prefix}.mlp.0.weight"))?,
            mlp0_b: require(w, &format!("{prefix}.mlp.0.bias"))?,
            mlp2_w: require(w, &format!("{prefix}.mlp.2.weight"))?,
            mlp2_b: require(w, &format!("{prefix}.mlp.2.bias"))?,
            freq_size: 256,
        })
    }

    /// Embed scalar timesteps `t` `[N]` → `[N, hidden]`.
    pub fn forward(&self, t: &Array) -> Result<Array> {
        let freq = timestep_embedding(t, self.freq_size)?;
        let h = silu(&linear(&freq, &self.mlp0_w, &self.mlp0_b)?)?;
        linear(&h, &self.mlp2_w, &self.mlp2_b)
    }
}

/// GLIDE sinusoidal embedding: `freqs = exp(-ln(max_period)·arange(half)/half)`, then
/// `cat(cos(t·freqs), sin(t·freqs))`. `dim` is even (256), so no zero-pad branch.
fn timestep_embedding(t: &Array, dim: i32) -> Result<Array> {
    const MAX_PERIOD: f32 = 10000.0;
    let half = (dim / 2) as usize;
    let log_max = MAX_PERIOD.ln(); // natural log of 10000
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-log_max * i as f32 / half as f32).exp())
        .collect();
    let n = t.shape()[0];
    let t = t.reshape(&[n, 1])?;
    let freqs = Array::from_slice(&freqs, &[1, half as i32]);
    let args = matmul(&t, &freqs)?; // [N, half]
    concatenate_axis(&[&args.cos()?, &args.sin()?], 1).map_err(Error::from)
}

/// The flow-matching time schedule (always the standard branch): `σ = 1−t`,
/// `σ ← shift·σ / (1 + (shift−1)·σ)`, return `1−σ`. Elementwise over `t`.
pub fn apply_time_schedule(t: &Array, shift: f32) -> Result<Array> {
    let one = Array::from_f32(1.0);
    let sigma = subtract(&one, t)?;
    let num = multiply(&sigma, Array::from_f32(shift))?;
    let denom = add(&one, &multiply(&sigma, Array::from_f32(shift - 1.0))?)?;
    let sigma = divide(&num, &denom)?;
    subtract(&one, &sigma).map_err(Error::from)
}

/// One forward-Euler step: `z + (t_next − t)·v_pred`.
pub fn euler_step(v_pred: &Array, z: &Array, t: f32, t_next: f32) -> Result<Array> {
    add(z, &multiply(v_pred, Array::from_f32(t_next - t))?).map_err(Error::from)
}

/// Flow-matching velocity: `(x_pred − z) / max(1 − t, t_eps)`.
pub fn velocity(x_pred: &Array, z: &Array, t: f32, t_eps: f32) -> Result<Array> {
    let denom = (1.0 - t).max(t_eps);
    divide(&subtract(x_pred, z)?, Array::from_f32(denom)).map_err(Error::from)
}

/// `images` `[N,3,H,W]` → patches `[N, (H/ps)·(W/ps), ps²·3]` (channel-last patch layout, matching
/// the reference `patchify(..., channel_first=False)`: `nchpwq → nhwpqc`).
pub fn patchify(images: &Array, patch_size: i32) -> Result<Array> {
    let sh = images.shape();
    let (n, h, w) = (sh[0], sh[2] / patch_size, sh[3] / patch_size);
    let x = images.reshape(&[n, 3, h, patch_size, w, patch_size])?;
    let x = x.transpose_axes(&[0, 2, 4, 3, 5, 1])?; // nchpwq -> nhwpqc
    x.reshape(&[n, h * w, patch_size * patch_size * 3])
        .map_err(Error::from)
}

/// Inverse of [`patchify`]: patches `[N,L,ps²·3]` → `[N,3,H,W]` (`nhwpqc → nchpwq`). `h`/`w` are
/// the token-grid dims; if `None`, a square grid is assumed.
pub fn unpatchify(x: &Array, patch_size: i32, h: Option<i32>, w: Option<i32>) -> Result<Array> {
    let n = x.shape()[0];
    let (h, w) = match (h, w) {
        (Some(h), Some(w)) => (h, w),
        _ => {
            let g = (x.shape()[1] as f64).sqrt() as i32;
            (g, g)
        }
    };
    let x = x.reshape(&[n, h, w, patch_size, patch_size, 3])?;
    let x = x.transpose_axes(&[0, 5, 1, 3, 2, 4])?; // nhwpqc -> nchpwq
    x.reshape(&[n, 3, h * patch_size, w * patch_size])
        .map_err(Error::from)
}
