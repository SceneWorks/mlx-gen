//! SDXL sampler — port of the vendored `_vendor/mlx_sd/sampler.py`
//! `SimpleEulerSampler` / `SimpleEulerAncestralSampler`. SDXL's `StableDiffusionXL` uses the
//! **ancestral** variant. The noise schedule comes from the `scaled_linear` betas; `sigmas` is a
//! 1001-entry table (a leading 0 + 1000 training-step sigmas). Per-step `sigma(t)` is a linear
//! interpolation of the table at the (float) time `t`.
//!
//! The ancestral step draws fresh noise from the **global** MLX RNG each step (seeded once at the
//! start of generation). `mlx.random.normal` is f32 regardless of the model's compute dtype, so the
//! draw sequence is precision-independent — running the Rust pipeline f32 reproduces the reference's
//! exact noise stream for a given seed (validated by the e2e gate, sc-2400 S5).

use mlx_rs::ops::multiply;
use mlx_rs::{random, Array};

use mlx_gen::array::scalar;
use mlx_gen::Result;

use crate::config::{BetaSchedule, DiffusionConfig};

/// A discrete Euler / Euler-Ancestral sampler over a precomputed sigma table.
pub struct EulerSampler {
    /// `[0, σ_1, …, σ_1000]` (length `num_train_steps + 1`).
    sigmas: Vec<f32>,
    ancestral: bool,
}

impl EulerSampler {
    /// Build the sampler from a [`DiffusionConfig`]. `ancestral` selects the
    /// `SimpleEulerAncestralSampler` step (SDXL) vs the plain Euler step.
    pub fn new(cfg: &DiffusionConfig, ancestral: bool) -> Self {
        Self::try_new(cfg, ancestral).expect("sigma table construction")
    }

    /// The sigma table is built with **MLX ops** (`cumprod`/`sqrt`/`square`), not host f32, so it is
    /// bit-identical to the reference `_sigmas` — a host cumprod over 1000 steps differs by ~2e-7,
    /// and that is the only remaining chaos seed once the U-Net is bit-exact (sc-2400 S5). `_linspace`
    /// (the vendored `arange/(N-1)·(b−a)+a`) is reproduced exactly, with the `**0.5` taken in f64
    /// (matching python) before the f32 array math.
    pub fn try_new(cfg: &DiffusionConfig, ancestral: bool) -> Result<Self> {
        use mlx_rs::ops::{add, concatenate_axis, divide, multiply, sqrt, square, subtract, zeros};
        let n = cfg.num_train_steps as i32;
        // _linspace(a, b, n) = arange(n)/(n-1) · (b−a) + a, as f32 arrays.
        let arange: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let x = divide(Array::from_slice(&arange, &[n]), scalar((n - 1) as f32))?;
        let betas = match cfg.beta_schedule {
            BetaSchedule::ScaledLinear => {
                let (a, b) = ((cfg.beta_start as f64).sqrt(), (cfg.beta_end as f64).sqrt());
                let lin = add(&multiply(&x, scalar((b - a) as f32))?, scalar(a as f32))?;
                square(&lin)?
            }
            BetaSchedule::Linear => {
                let (a, b) = (cfg.beta_start as f64, cfg.beta_end as f64);
                add(&multiply(&x, scalar((b - a) as f32))?, scalar(a as f32))?
            }
        };
        // alphas_cumprod = cumprod(1 − betas); sigmas = concat([0], sqrt((1−acp)/acp)).
        let alphas = subtract(scalar(1.0), &betas)?;
        let acp = alphas.cumprod(0, false, true)?;
        let body = sqrt(&divide(&subtract(scalar(1.0), &acp)?, &acp)?)?;
        let table = concatenate_axis(&[&zeros::<f32>(&[1])?, &body], 0)?;
        let sigmas = table.as_slice::<f32>().to_vec();
        Ok(Self { sigmas, ancestral })
    }

    /// The maximum (start) time index: `len(sigmas) - 1` = `num_train_steps`.
    pub fn max_time(&self) -> f32 {
        (self.sigmas.len() - 1) as f32
    }

    /// Linearly interpolate the sigma table at the (float) time `t` (the vendored `_interp`), host
    /// f32. Used for `init_noise_scale` + tests; the denoise math uses [`Self::sigma_arr`] so the
    /// interp runs through MLX (bit-exact to the reference at non-integer `t` — sc-2400 S6).
    pub fn sigma(&self, t: f32) -> f32 {
        let last = self.sigmas.len() - 1;
        let lo = (t as usize).min(last);
        let hi = (lo + 1).min(last);
        let delta = t - lo as f32;
        self.sigmas[lo] * (1.0 - delta) + delta * self.sigmas[hi]
    }

    /// `σ(t)` as an MLX scalar via the reference's `_interp` op order — `y_lo·(1−δ) + δ·y_hi` in MLX.
    /// At integer `t` (δ=0) this equals `sigmas[t]`; at the non-integer `t` produced by an
    /// inexact-fraction linspace (e.g. strength 0.7), the MLX interp is bit-exact to the reference
    /// where host f32 diverges by 1 ULP and the ancestral sampler amplifies it.
    fn sigma_arr(&self, t: f32) -> Result<Array> {
        use mlx_rs::ops::{add, multiply, subtract};
        let last = self.sigmas.len() - 1;
        let lo = (t as usize).min(last);
        let hi = (lo + 1).min(last);
        let y_lo = scalar(self.sigmas[lo]);
        let y_hi = scalar(self.sigmas[hi]);
        let delta = scalar(t - lo as f32);
        Ok(add(
            &multiply(&y_lo, &subtract(scalar(1.0), &delta)?)?,
            &multiply(&delta, &y_hi)?,
        )?)
    }

    /// The `(t, t_prev)` step pairs: `_linspace(start_time, 0, num_steps + 1)` zipped with its tail.
    ///
    /// Op order matches the vendored `_linspace` EXACTLY: `x = arange/(n)`, then `(0−start)·x +
    /// start` — NOT `start + (0−start)·i/n` (which divides last). `i/n` is f32-inexact (e.g. 1/5),
    /// so the two orders differ by 1 ULP in the timestep `t`, and a 1-ULP `t` shifts the U-Net's
    /// sinusoidal embedding enough to seed the chaotic ancestral trajectory (sc-2400 S6).
    pub fn timesteps(&self, num_steps: usize, start_time: f32) -> Vec<(f32, f32)> {
        let n = num_steps as f32;
        let steps: Vec<f32> = (0..=num_steps)
            .map(|i| {
                let x = i as f32 / n; // arange/(num_steps) first, matching _linspace
                (0.0 - start_time) * x + start_time
            })
            .collect();
        steps.windows(2).map(|w| (w[0], w[1])).collect()
    }

    /// The latent-noise scale for `sample_prior`: `σ_last · (σ_last² + 1)^-0.5` (host f32; for tests).
    pub fn init_noise_scale(&self) -> f32 {
        let s = *self.sigmas.last().unwrap();
        s * (s * s + 1.0).powf(-0.5)
    }

    /// Sample the prior latents `noise · σ_last · (σ_last² + 1).rsqrt()` (f32, global RNG). `shape`
    /// is NHWC `[B, H/8, W/8, 4]`.
    ///
    /// The scale path **byte-matches the reference's exact op order**: `(noise · σ_last) ·
    /// rsqrt(σ_last²+1)` — two left-to-right array multiplies, NOT `noise · (σ_last · rsqrt(…))`.
    /// f32 multiply is non-associative, so precomputing the scalar `σ_last·rsqrt(…)` first differs by
    /// 1 ULP (~1e-7). Fed through the (bit-exact) ancestral trajectory at CFG=7, that single-ULP
    /// prior perturbation is the *only* remaining chaos seed once the U-Net + sigma table are
    /// bit-exact, and alone it moves the full render from pixel-parity to ~34% px>8 (sc-2400 S5).
    /// The host `powf(-0.5)`→MLX `rsqrt` swap matters for the same reason. (The per-step `step()` math
    /// stays host f32 — already bit-exact, proven by `denoise_per_step_matches_golden`.)
    pub fn sample_prior(&self, shape: &[i32]) -> Result<Array> {
        use mlx_rs::ops::{add, rsqrt, square};
        let noise = random::normal::<f32>(shape, None, None, None)?;
        let s = scalar(*self.sigmas.last().unwrap());
        let factor = rsqrt(&add(&square(&s)?, scalar(1.0))?)?;
        // (noise · σ_last) · rsqrt(…) — reference order.
        Ok(multiply(&multiply(&noise, &s)?, &factor)?)
    }

    /// Add noise to clean latents at (float) time `t` — the vendored `add_noise`, used to seed
    /// img2img: `(x + noise·σ(t)) · rsqrt(σ(t)²+1)`, drawing one global-RNG normal. The op order
    /// matches the reference exactly (f32 non-associativity, like `sample_prior` — sc-2400 S6).
    pub fn add_noise(&self, x: &Array, t: f32) -> Result<Array> {
        use mlx_rs::ops::{add, rsqrt, square};
        let noise = random::normal::<f32>(x.shape(), None, None, None)?;
        let s = self.sigma_arr(t)?;
        let noised = add(x, &multiply(&noise, &s)?)?;
        let factor = rsqrt(&add(&square(&s)?, scalar(1.0))?)?;
        Ok(multiply(&noised, &factor)?)
    }

    /// One denoise step from `x_t` (at time `t`) to `x_{t_prev}`. Euler-ancestral when `ancestral`
    /// (draws fresh global-RNG noise scaled by `σ_up`); plain Euler otherwise.
    ///
    /// All scalar math runs through **MLX ops** op-for-op with the reference, NOT host f32. Two
    /// subtleties matter (each a 1-ULP that the chaos-sensitive ancestral sampler amplifies to a
    /// visible whole-image divergence at non-round σ — sc-2400 S6):
    /// - host `sqrt`/`powf(-0.5)` ≠ MLX `sqrt`/`rsqrt` at some σ → use MLX `sqrt`/`rsqrt`;
    /// - `σ_up²` must be MLX **`power(σ_up, 2)`** (the reference's `σ_up**2`), NOT `square(σ_up)` —
    ///   `mx.power(x,2)` and `mx.square(x)` differ by 1 ULP at some x (confirmed: σ_up=0.96565 →
    ///   square `0x3f6eb715`, power `0x3f6eb714`), which shifts `σ_down` by 1 ULP.
    ///
    /// `σ(t)` itself is the host interp of the MLX-exact table (bit-exact at integer `t`).
    pub fn step(&self, eps_pred: &Array, x_t: &Array, t: f32, t_prev: f32) -> Result<Array> {
        use mlx_rs::ops::{add, divide, multiply, power, rsqrt, sqrt, square, subtract};
        let sigma = self.sigma_arr(t)?;
        let sigma_prev = self.sigma_arr(t_prev)?;
        let sigma2 = square(&sigma)?;
        let sigma_prev2 = square(&sigma_prev)?;
        let one = scalar(1.0);
        // x' = sqrt(σ²+1)·x_t + eps·dt, with dt = σ_down − σ (ancestral) or σ_prev − σ (euler).
        let scaled_x = |dt: &Array| -> Result<Array> {
            Ok(add(
                &multiply(&sqrt(&add(&sigma2, &one)?)?, x_t)?,
                &multiply(eps_pred, dt)?,
            )?)
        };
        let renorm = rsqrt(&add(&sigma_prev2, &one)?)?; // (σ_prev²+1)^-0.5
        if self.ancestral {
            // σ_up = sqrt(σ_prev²·(σ²−σ_prev²)/σ²); σ_down = sqrt(σ_prev² − σ_up²).
            let sigma_up = sqrt(&divide(
                &multiply(&sigma_prev2, &subtract(&sigma2, &sigma_prev2)?)?,
                &sigma2,
            )?)?;
            // `power(σ_up, 2)` (the reference's `σ_up**2`), NOT `square` — they differ by 1 ULP.
            let sigma_down = sqrt(&subtract(&sigma_prev2, &power(&sigma_up, scalar(2.0))?)?)?;
            let mut x = scaled_x(&subtract(&sigma_down, &sigma)?)?;
            let noise = random::normal::<f32>(x.shape(), None, None, None)?;
            x = add(&x, &multiply(&noise, &sigma_up)?)?;
            Ok(multiply(&x, &renorm)?)
        } else {
            let x = scaled_x(&subtract(&sigma_prev, &sigma)?)?;
            Ok(multiply(&x, &renorm)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigma_table_endpoints_and_interp() {
        let s = EulerSampler::new(&DiffusionConfig::sdxl_base(), true);
        assert_eq!(s.sigmas.len(), 1001);
        assert_eq!(s.sigmas[0], 0.0);
        assert_eq!(s.max_time(), 1000.0);
        // Monotonic increasing table.
        assert!(s.sigmas.windows(2).all(|w| w[1] >= w[0]));
        // Linear interp at a half index.
        let mid = s.sigma(10.5);
        assert!((mid - 0.5 * (s.sigmas[10] + s.sigmas[11])).abs() < 1e-6);
    }

    #[test]
    fn zero_steps_yield_no_pairs() {
        // img2img at strength ≤ 1/steps rounds to 0 steps; the schedule must produce no `(t, t_prev)`
        // pairs so the denoise loop is a no-op (and never invokes the σ=0 ancestral step → NaN).
        let s = EulerSampler::new(&DiffusionConfig::sdxl_base(), true);
        assert!(s.timesteps(0, 0.0).is_empty());
        assert!(s.timesteps(0, 1000.0).is_empty());
    }

    #[test]
    fn timesteps_span_start_to_zero() {
        let s = EulerSampler::new(&DiffusionConfig::sdxl_base(), true);
        let ts = s.timesteps(4, 1000.0);
        assert_eq!(ts.len(), 4);
        assert_eq!(ts[0].0, 1000.0);
        assert!((ts.last().unwrap().1 - 0.0).abs() < 1e-4);
    }
}
