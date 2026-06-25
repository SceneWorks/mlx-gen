//! The distilled few-step pixel-diffusion sampler — faithful port of
//! `pid_distill_model.py::_student_sample_loop` (+ `_velocity_to_x0`). The released students run the
//! **SDE / velocity-prediction** schedule (`student_t_list=[0.999,0.866,0.634,0.342,0.0]`,
//! `fm_timescale=1000`, cfg 1 — no classifier-free guidance). PiD denoises directly in high-res
//! **pixel** space: `noise`/`x` are `[B, 3, H, W]` at the *output* resolution, conditioned on the LQ
//! latent + caption + degrade σ.
//!
//! Per step `(t_cur, t_next)`: `v = net(x, t_cur·timescale, …)`, `x0 = x − t_cur·v`; then for an SDE
//! interior step `x = (1−t_next)·x0 + t_next·ε` (fresh noise), and the final `t_next=0` step takes
//! `x = x0`. Output is clamped to `[-1, 1]`.
//!
//! The step math is RNG-free and deterministic — [`Sampler::run`] takes the initial noise and the
//! per-step ε injected, so it parity-tests bit-for-bit against the torch loop. [`Sampler::sample`]
//! draws them from MLX's PRNG for production (cross-backend RNG does not match torch — a same-backend
//! decode, per the repo's full-trajectory chaos note).

use mlx_rs::ops::{add, clip, multiply, subtract};
use mlx_rs::{random, Array};

use mlx_gen::array::scalar;
use mlx_gen::Result;

use crate::config::{SampleType, SamplerConfig};
use crate::lq::PidNet;

/// The distilled few-step sampler.
pub struct Sampler {
    t_list: Vec<f32>,
    timescale: f32,
    sde: bool,
}

impl Sampler {
    pub fn new(cfg: &SamplerConfig) -> Self {
        Self {
            t_list: cfg.student_t_list.clone(),
            timescale: cfg.fm_timescale,
            sde: cfg.sample_type == SampleType::Sde,
        }
    }

    /// Number of denoising steps (`len(t_list) − 1`).
    pub fn steps(&self) -> usize {
        self.t_list.len().saturating_sub(1)
    }

    /// Number of fresh-noise draws the SDE loop consumes (one per interior step with `t_next>0`).
    pub fn num_eps(&self) -> usize {
        if !self.sde {
            return 0;
        }
        (1..self.t_list.len())
            .filter(|&i| self.t_list[i] > 0.0)
            .count()
    }

    /// velocity-prediction `x0 = x − t·v`.
    fn velocity_to_x0(x: &Array, v: &Array, t: f32) -> Result<Array> {
        Ok(subtract(x, &multiply(v, scalar(t))?)?)
    }

    /// Deterministic loop with the initial `noise` and the per-step `eps` injected (one ε per SDE
    /// interior step, in order). `caption`/`lq_latent`/`sigma` condition the net every step.
    pub fn run(
        &self,
        net: &PidNet,
        noise: &Array,
        eps: &[Array],
        caption: &Array,
        lq_latent: &Array,
        sigma: &Array,
    ) -> Result<Array> {
        let b = noise.shape()[0];
        let mut x = noise.clone();
        let mut ei = 0usize;
        for i in 0..self.steps() {
            let t_cur = self.t_list[i];
            let t_next = self.t_list[i + 1];
            let t_scaled = Array::from_slice(&vec![t_cur * self.timescale; b as usize], &[b]);
            let v = net.forward(&x, &t_scaled, caption, lq_latent, sigma)?;
            if t_next > 0.0 {
                if self.sde {
                    let x0 = Self::velocity_to_x0(&x, &v, t_cur)?;
                    let e = &eps[ei];
                    ei += 1;
                    x = add(
                        &multiply(&x0, scalar(1.0 - t_next))?,
                        &multiply(e, scalar(t_next))?,
                    )?;
                } else {
                    // ODE: x = x + (t_next − t_cur)·v (velocity prediction).
                    x = add(&x, &multiply(&v, scalar(t_next - t_cur))?)?;
                }
            } else {
                x = Self::velocity_to_x0(&x, &v, t_cur)?;
            }
        }
        Ok(clip(&x, (-1.0, 1.0))?)
    }

    /// Production entry: draw the initial noise + per-step ε from MLX's PRNG (seeded), then run the
    /// loop. Returns clamped pixels `[B, 3, H, W]`.
    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &self,
        net: &PidNet,
        caption: &Array,
        lq_latent: &Array,
        sigma: &Array,
        b: i32,
        h: i32,
        w: i32,
        seed: u64,
    ) -> Result<Array> {
        let (k_noise, mut k_rest) = random::split(&random::key(seed)?, 2)?;
        let noise = random::normal::<f32>(&[b, 3, h, w], None, None, Some(&k_noise))?;
        let mut eps = Vec::with_capacity(self.num_eps());
        for _ in 0..self.num_eps() {
            let (k_e, k_n) = random::split(&k_rest, 2)?;
            eps.push(random::normal::<f32>(
                &[b, 3, h, w],
                None,
                None,
                Some(&k_e),
            )?);
            k_rest = k_n;
        }
        self.run(net, &noise, &eps, caption, lq_latent, sigma)
    }
}
