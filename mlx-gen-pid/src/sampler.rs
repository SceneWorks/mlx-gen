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
use mlx_rs::transforms::eval;
use mlx_rs::{random, Array};

use mlx_gen::array::scalar;
use mlx_gen::{CancelFlag, Error, Result};

use crate::config::{SampleType, SamplerConfig};
use crate::lq::PidNet;
use crate::tiling::forward_tiled;

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
    ///
    /// `cancel` is the cooperative cancellation handle (F-006): checked at each of the ~4 step
    /// boundaries and forced-eval'd there so a cancel actually interrupts this multi-second decode
    /// (MLX is lazy — without the per-step `eval` the whole graph would schedule at once and a
    /// mid-loop check could never observe the trip). Returns [`Error::Canceled`] on trip.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        net: &PidNet,
        noise: &Array,
        eps: &[Array],
        caption: &Array,
        lq_latent: &Array,
        sigma: &Array,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        self.run_inner(noise, eps, cancel, |x, t_scaled| {
            net.forward(x, t_scaled, caption, lq_latent, sigma)
        })
    }

    /// Like [`Self::run`] but the per-step **velocity** forward is spatially tiled (sc-10087):
    /// [`crate::tiling::forward_tiled`] runs the net on overlapping `tile`-px pixel windows and
    /// feather-blends them, so the whole-image `PidNet::forward` peak (and its single long Metal command
    /// buffer) never materializes. The 4-step SDE loop stays whole-image — `noise`/`eps` are the same
    /// full-res seeded draws, so the sampler math + RNG sequence are unchanged and only the forward is
    /// approximated. `overlap` is the feather width (px).
    #[allow(clippy::too_many_arguments)]
    pub fn run_tiled(
        &self,
        net: &PidNet,
        noise: &Array,
        eps: &[Array],
        caption: &Array,
        lq_latent: &Array,
        sigma: &Array,
        tile: i32,
        overlap: i32,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        self.run_inner(noise, eps, cancel, |x, t_scaled| {
            forward_tiled(net, x, t_scaled, caption, lq_latent, sigma, tile, overlap)
        })
    }

    /// Shared SDE step loop. `forward(x, t_scaled) -> v` is the per-step velocity predictor — either the
    /// whole-image `PidNet::forward` ([`Self::run`]) or the tiled forward ([`Self::run_tiled`]). The step
    /// math, ε injection, cancel handling, and per-step `eval` are identical between the two paths.
    fn run_inner(
        &self,
        noise: &Array,
        eps: &[Array],
        cancel: Option<&CancelFlag>,
        forward: impl Fn(&Array, &Array) -> Result<Array>,
    ) -> Result<Array> {
        // F-100: the SDE loop consumes one `eps[ei]` per interior step; a caller that supplies fewer
        // than `num_eps()` draws would panic OOB mid-loop. Validate the contract up front.
        if eps.len() < self.num_eps() {
            return Err(Error::Msg(format!(
                "pid sampler: need {} eps draws for this schedule, got {}",
                self.num_eps(),
                eps.len()
            )));
        }
        let b = noise.shape()[0];
        let mut x = noise.clone();
        let mut ei = 0usize;
        for i in 0..self.steps() {
            if cancel.is_some_and(CancelFlag::is_cancelled) {
                return Err(Error::Canceled);
            }
            let t_cur = self.t_list[i];
            let t_next = self.t_list[i + 1];
            let t_scaled = Array::from_slice(&vec![t_cur * self.timescale; b as usize], &[b]);
            let v = forward(&x, &t_scaled)?;
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
            // Materialize this step so a cancel between steps is actually observed (MLX is lazy),
            // and to bound transient peak memory (F-013): the 4-step graph no longer schedules at once.
            eval([&x])?;
        }
        Ok(clip(&x, (-1.0, 1.0))?)
    }

    /// Production entry: draw the initial noise + per-step ε from MLX's PRNG (seeded), then run the
    /// loop. Returns clamped pixels `[B, 3, H, W]`. `cancel` is threaded into [`Self::run`] (F-006).
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
        cancel: Option<&CancelFlag>,
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
        self.run(net, &noise, &eps, caption, lq_latent, sigma, cancel)
    }

    /// Tiled production entry (sc-10087): identical seeded noise/ε draw as [`Self::sample`] (full-res, so
    /// the RNG sequence is byte-for-byte the same), then [`Self::run_tiled`] with the spatial `tile` /
    /// `overlap`. Use for large outputs where the whole-image forward overflows the memory /
    /// command-buffer envelope. `tile`/`overlap` are output-pixel units (rounded to the pixel→latent
    /// factor internally).
    #[allow(clippy::too_many_arguments)]
    pub fn sample_tiled(
        &self,
        net: &PidNet,
        caption: &Array,
        lq_latent: &Array,
        sigma: &Array,
        b: i32,
        h: i32,
        w: i32,
        seed: u64,
        tile: i32,
        overlap: i32,
        cancel: Option<&CancelFlag>,
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
        self.run_tiled(
            net, &noise, &eps, caption, lq_latent, sigma, tile, overlap, cancel,
        )
    }
}
