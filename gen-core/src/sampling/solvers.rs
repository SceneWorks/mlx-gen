//! The curated callback-solver library (epic 7114, P1, sc-7117): the integration-method half of the
//! unified framework. Each solver is a [`Sampler`] impl driving a `denoise(x, σ) -> x0` callback over
//! a sigma schedule, advancing the latents through [`LatentOps`] only — backend-neutral, prediction-
//! type-agnostic (the [`super::model_sampling::ModelSampling`] already mapped the raw output to `x0`).
//!
//! Faithful ports of ComfyUI / k-diffusion in the unified VE-sigma space (`λ = −ln σ`, `sigma_fn(t) =
//! e^{−t}`): `euler` (in [`super::unified`]), `euler_ancestral`, `heun`, `dpmpp_2m`, `dpmpp_sde`,
//! `uni_pc`, `lcm`, `ddim` (decision 2). The multistep solvers (`dpmpp_2m`, `uni_pc`) lift the structure
//! of Wan's flow-mode `dpmpp_2m`/`uni_pc` (`mlx-gen-wan/src/scheduler.rs`) generalised to VE space —
//! where each reduces to plain Euler at first order, so a constant-velocity field integrates EXACTLY
//! (the per-solver coherence test).

use super::unified::{is_terminal, to_d, DenoiseFn, Euler, Sampler};
use super::LatentOps;
use crate::Result;

// =================================================================================================
// Scalar helpers (host f64) — the VE-sigma `λ = −ln σ` time and the ancestral noise split.
// =================================================================================================

/// `λ(σ) = −ln σ` — the VE half-log-SNR "time" the DPM/UniPC multistep solvers integrate in.
#[inline]
fn lambda(sigma: f32) -> f64 {
    -(sigma.max(1e-12) as f64).ln()
}

/// `σ(λ) = e^{−λ}` — inverse of [`lambda`].
#[inline]
fn sigma_of(l: f64) -> f64 {
    (-l).exp()
}

/// k-diffusion `get_ancestral_step`: split the `σ_from -> σ_to` step into a deterministic descent to
/// `σ_down` plus `σ_up` of fresh noise, scaled by `eta`. `σ_to == 0` -> `(0, 0)` (no noise on the
/// terminal step).
fn ancestral_step(sigma_from: f32, sigma_to: f32, eta: f32) -> (f32, f32) {
    if sigma_to <= 0.0 {
        return (0.0, 0.0);
    }
    let (sf, st) = (sigma_from as f64, sigma_to as f64);
    let su = (eta as f64 * (st * st * (sf * sf - st * st) / (sf * sf)).max(0.0).sqrt()).min(st);
    let sd = (st * st - su * su).max(0.0).sqrt();
    (sd as f32, su as f32)
}

// =================================================================================================
// euler_ancestral — k-diffusion `sample_euler_ancestral` (1st order, stochastic).
// =================================================================================================

/// Ancestral Euler: a forward-Euler descent to `σ_down` plus `σ_up` of fresh noise each step
/// (`eta = 1`). The added noise makes it non-deterministic-in-σ but seed-reproducible.
#[derive(Clone, Copy, Debug)]
pub struct EulerAncestral {
    /// Stochasticity scale (k-diffusion default `1.0`).
    pub eta: f32,
}

impl Default for EulerAncestral {
    fn default() -> Self {
        Self { eta: 1.0 }
    }
}

impl<L: LatentOps> Sampler<L> for EulerAncestral {
    fn sample(
        &self,
        ops: &L,
        _ms: &dyn super::ModelSampling,
        denoise: &mut DenoiseFn<'_, L>,
        mut x: L::Latent,
        sigmas: &[f32],
        seed: u64,
    ) -> Result<L::Latent> {
        for i in 0..sigmas.len().saturating_sub(1) {
            let sigma = sigmas[i];
            let x0 = denoise(&x, sigma)?;
            if is_terminal(sigma) {
                x = x0;
                continue;
            }
            let (sd, su) = ancestral_step(sigma, sigmas[i + 1], self.eta);
            let d = to_d(ops, &x, sigma, &x0)?;
            x = ops.axpy(1.0, &x, sd - sigma, &d)?;
            if su > 0.0 {
                let noise = ops.randn_like(&x, seed, i)?;
                x = ops.axpy(1.0, &x, su, &noise)?;
            }
        }
        Ok(x)
    }
}

// =================================================================================================
// heun — k-diffusion `sample_heun` (2nd order, deterministic; 2 model evals/step).
// =================================================================================================

/// Heun's method: an Euler predictor to `σ_{i+1}`, a second model eval there, then a step with the
/// averaged derivative `(d + d')/2`. The terminal step (`σ_{i+1} = 0`) falls back to plain Euler.
#[derive(Clone, Copy, Debug, Default)]
pub struct Heun;

impl<L: LatentOps> Sampler<L> for Heun {
    fn sample(
        &self,
        ops: &L,
        _ms: &dyn super::ModelSampling,
        denoise: &mut DenoiseFn<'_, L>,
        mut x: L::Latent,
        sigmas: &[f32],
        _seed: u64,
    ) -> Result<L::Latent> {
        for i in 0..sigmas.len().saturating_sub(1) {
            let sigma = sigmas[i];
            let s_next = sigmas[i + 1];
            let x0 = denoise(&x, sigma)?;
            if is_terminal(sigma) {
                x = x0;
                continue;
            }
            let d = to_d(ops, &x, sigma, &x0)?;
            let dt = s_next - sigma;
            if s_next == 0.0 {
                x = ops.axpy(1.0, &x, dt, &d)?;
            } else {
                let x2 = ops.axpy(1.0, &x, dt, &d)?;
                let x0_2 = denoise(&x2, s_next)?;
                let d2 = to_d(ops, &x2, s_next, &x0_2)?;
                let d_prime = ops.axpy(0.5, &d, 0.5, &d2)?;
                x = ops.axpy(1.0, &x, dt, &d_prime)?;
            }
        }
        Ok(x)
    }
}

// =================================================================================================
// dpmpp_2m — k-diffusion `sample_dpmpp_2m` (DPM-Solver++(2M), 2nd-order multistep).
// =================================================================================================

/// DPM-Solver++(2M): a multistep solver that reuses the previous step's denoised estimate for a
/// 2nd-order update with a single model eval per step. Falls back to 1st order on the first and
/// terminal steps. Wan's flow-mode `dpmpp_2m` is the structural reference (generalised to VE space).
#[derive(Clone, Copy, Debug, Default)]
pub struct Dpmpp2m;

impl<L: LatentOps> Sampler<L> for Dpmpp2m {
    fn sample(
        &self,
        ops: &L,
        _ms: &dyn super::ModelSampling,
        denoise: &mut DenoiseFn<'_, L>,
        mut x: L::Latent,
        sigmas: &[f32],
        _seed: u64,
    ) -> Result<L::Latent> {
        let mut old_x0: Option<L::Latent> = None;
        for i in 0..sigmas.len().saturating_sub(1) {
            let sigma = sigmas[i];
            let s_next = sigmas[i + 1];
            let x0 = denoise(&x, sigma)?;
            if s_next == 0.0 {
                // Terminal: land on the denoised estimate (the 1st-order limit).
                x = x0.clone();
                old_x0 = Some(x0);
                continue;
            }
            let t = lambda(sigma);
            let t_next = lambda(s_next);
            let h = t_next - t;
            let coeff_x = s_next / sigma; // sigma_fn(t_next)/sigma_fn(t)
            let coeff_x0 = (-(-h).exp_m1()) as f32; // −expm1(−h)
            let denoised_d = match &old_x0 {
                None => x0.clone(),
                Some(old) => {
                    let h_last = t - lambda(sigmas[i - 1]);
                    let r = h_last / h;
                    // denoised_d = (1 + 1/(2r))·x0 − (1/(2r))·old_x0
                    let a = 1.0 + 1.0 / (2.0 * r);
                    let b = -1.0 / (2.0 * r);
                    ops.axpy(a as f32, &x0, b as f32, old)?
                }
            };
            x = ops.axpy(coeff_x, &x, coeff_x0, &denoised_d)?;
            old_x0 = Some(x0);
        }
        Ok(x)
    }
}

// =================================================================================================
// dpmpp_sde — k-diffusion `sample_dpmpp_sde` (stochastic midpoint, 2nd order; 2 evals/step).
// =================================================================================================

/// DPM-Solver++ SDE (midpoint, `r = 1/2`): a stochastic 2nd-order solver — a noised half-step, a
/// second model eval there, then a noised full step using the midpoint denoised. Seed-reproducible.
#[derive(Clone, Copy, Debug)]
pub struct DpmppSde {
    /// Stochasticity scale (k-diffusion default `1.0`).
    pub eta: f32,
}

impl Default for DpmppSde {
    fn default() -> Self {
        Self { eta: 1.0 }
    }
}

impl<L: LatentOps> Sampler<L> for DpmppSde {
    fn sample(
        &self,
        ops: &L,
        _ms: &dyn super::ModelSampling,
        denoise: &mut DenoiseFn<'_, L>,
        mut x: L::Latent,
        sigmas: &[f32],
        seed: u64,
    ) -> Result<L::Latent> {
        const R: f64 = 0.5;
        for i in 0..sigmas.len().saturating_sub(1) {
            let sigma = sigmas[i];
            let s_next = sigmas[i + 1];
            let x0 = denoise(&x, sigma)?;
            if is_terminal(sigma) {
                x = x0;
                continue;
            }
            if s_next == 0.0 {
                // Terminal: plain Euler onto the denoised estimate.
                let d = to_d(ops, &x, sigma, &x0)?;
                x = ops.axpy(1.0, &x, s_next - sigma, &d)?;
                continue;
            }
            let t = lambda(sigma);
            let t_next = lambda(s_next);
            let h = t_next - t;
            let s = t + h * R; // midpoint time
            let sigma_s = sigma_of(s) as f32;

            // Half-step to the midpoint (noised).
            let (sd1, su1) = ancestral_step(sigma, sigma_s, self.eta);
            let s_lam = lambda(sd1);
            let coeff_x = (sigma_of(s_lam) / sigma_of(t)) as f32; // sd1/sigma
            let coeff_x0 = -(t - s_lam).exp_m1() as f32;
            let mut x2 = ops.axpy(coeff_x, &x, coeff_x0, &x0)?;
            if su1 > 0.0 {
                let noise1 = ops.randn_like(&x2, seed, 2 * i)?;
                x2 = ops.axpy(1.0, &x2, su1, &noise1)?;
            }
            let x0_2 = denoise(&x2, sigma_s)?;

            // Full step using the midpoint denoised (fac = 1/(2r) = 1 -> denoised_d = x0_2), noised.
            let (sd2, su2) = ancestral_step(sigma, s_next, self.eta);
            let t_next_lam = lambda(sd2);
            let coeff_x = (sigma_of(t_next_lam) / sigma_of(t)) as f32; // sd2/sigma
            let coeff_x0 = -(t - t_next_lam).exp_m1() as f32;
            x = ops.axpy(coeff_x, &x, coeff_x0, &x0_2)?;
            if su2 > 0.0 {
                let noise2 = ops.randn_like(&x, seed, 2 * i + 1)?;
                x = ops.axpy(1.0, &x, su2, &noise2)?;
            }
        }
        Ok(x)
    }
}

// =================================================================================================
// uni_pc — UniPC predictor-corrector (order 2, bh2). Wan's flow-mode uni_pc generalised to VE space.
// =================================================================================================

/// UniPC (order 2): a multistep predictor-corrector. Each step refines the previous prediction with
/// the fresh denoised estimate (the corrector, a trapezoidal blend), then predicts the next sample
/// with a 2nd-order extrapolation. At order 1 / constant denoised it reduces to Euler. Wan's flow-mode
/// `uni_pc` is the structural reference (generalised to VE `λ = −ln σ`, `α = 1`).
#[derive(Clone, Copy, Debug, Default)]
pub struct UniPc;

impl<L: LatentOps> Sampler<L> for UniPc {
    fn sample(
        &self,
        ops: &L,
        _ms: &dyn super::ModelSampling,
        denoise: &mut DenoiseFn<'_, L>,
        mut x: L::Latent,
        sigmas: &[f32],
        _seed: u64,
    ) -> Result<L::Latent> {
        let mut prev_x0: Option<L::Latent> = None; // x0_{i-1}
        let mut last_sample: Option<L::Latent> = None; // corrected x_{i-1}
        for i in 0..sigmas.len().saturating_sub(1) {
            let sigma = sigmas[i];
            let s_next = sigmas[i + 1];
            let x0 = denoise(&x, sigma)?;

            // Corrector: refine x_i (predicted at step i-1) using the fresh x0_i. Order-2 corrector
            // is the trapezoidal blend 0.5·(x0_{i-1} + x0_i).
            if let (Some(prev), Some(ls)) = (&prev_x0, &last_sample) {
                let s_prev = sigmas[i - 1];
                let phi = (sigma / s_prev) - 1.0; // expm1(−h_corr)
                let blend = ops.axpy(0.5, prev, 0.5, &x0)?; // 0.5·(x0_{i-1}+x0_i)
                x = ops.axpy(sigma / s_prev, ls, -phi, &blend)?;
            }

            if s_next == 0.0 {
                // Terminal: land on the (corrected-then-)denoised estimate.
                x = x0;
                break;
            }

            // Predictor: σ_i -> σ_{i+1}, 2nd order when history is available.
            let t = lambda(sigma);
            let t_next = lambda(s_next);
            let h = t_next - t;
            let phi = (s_next / sigma) - 1.0; // expm1(−h)
            let mut x_next = ops.axpy(s_next / sigma, &x, -phi, &x0)?;
            if let Some(prev) = &prev_x0 {
                let r = (lambda(sigmas[i - 1]) - t) / h; // (λ_{i-1} − λ_i)/h
                let d1 = ops.axpy((1.0 / r) as f32, prev, (-1.0 / r) as f32, &x0)?; // (x0_{i-1}−x0_i)/r
                x_next = ops.axpy(1.0, &x_next, -phi * 0.5, &d1)?;
            }

            prev_x0 = Some(x0);
            last_sample = Some(x);
            x = x_next;
        }
        Ok(x)
    }
}

// =================================================================================================
// lcm — ComfyUI `sample_lcm` (consistency: x <- denoised, re-noise between steps).
// =================================================================================================

/// Latent Consistency Model sampler (ComfyUI `sample_lcm`): each step jumps straight to the denoised
/// estimate, then re-noises to `σ_{i+1}`. ~2–8 steps. (Distinct from the legacy diffusers-faithful
/// `LcmPolicy` accel path, which the engines keep for distilled LoRAs; this is the user-selectable
/// curated sampler.)
#[derive(Clone, Copy, Debug, Default)]
pub struct Lcm;

impl<L: LatentOps> Sampler<L> for Lcm {
    fn sample(
        &self,
        ops: &L,
        ms: &dyn super::ModelSampling,
        denoise: &mut DenoiseFn<'_, L>,
        mut x: L::Latent,
        sigmas: &[f32],
        seed: u64,
    ) -> Result<L::Latent> {
        for i in 0..sigmas.len().saturating_sub(1) {
            let x0 = denoise(&x, sigmas[i])?;
            x = x0;
            let s_next = sigmas[i + 1];
            if s_next > 0.0 {
                let noise = ops.randn_like(&x, seed, i)?;
                // Re-noise through the model's own noise_scaling (ComfyUI `sample_lcm`): `x = k_x0·x0 +
                // k_noise·noise`. VE/EDM/DDPM keep x0 at full scale (k_x0 = 1, k_noise = σ); FLOW uses the
                // convex blend (k_x0 = 1−σ, k_noise = σ) so a flow-distilled student is re-noised in its
                // training regime instead of the OOD VE form (sc-7491).
                let (k_noise, k_x0) = ms.noise_scaling_coeffs(s_next);
                x = ops.axpy(k_x0, &x, k_noise, &noise)?;
            }
        }
        Ok(x)
    }
}

// =================================================================================================
// ddim — DDIM (η = 0): the deterministic x0-interpolation step.
// =================================================================================================

/// DDIM (η = 0): the deterministic update `x_{i+1} = (σ_{i+1}/σ_i)·x_i + (1 − σ_{i+1}/σ_i)·x0`. In the
/// unified VE-sigma space this coincides with Euler, but is exposed as a named curated sampler.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ddim;

impl<L: LatentOps> Sampler<L> for Ddim {
    fn sample(
        &self,
        ops: &L,
        _ms: &dyn super::ModelSampling,
        denoise: &mut DenoiseFn<'_, L>,
        mut x: L::Latent,
        sigmas: &[f32],
        _seed: u64,
    ) -> Result<L::Latent> {
        for i in 0..sigmas.len().saturating_sub(1) {
            let sigma = sigmas[i];
            let s_next = sigmas[i + 1];
            let x0 = denoise(&x, sigma)?;
            if is_terminal(sigma) {
                x = x0;
                continue;
            }
            let ratio = s_next / sigma;
            x = ops.axpy(ratio, &x, 1.0 - ratio, &x0)?;
        }
        Ok(x)
    }
}

// =================================================================================================
// Registry — name <-> solver, the per-request selection seam the worker/engine drives (sc-7127).
// =================================================================================================

/// The curated sampler vocabulary (epic 7114 decision 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Solver {
    Euler,
    EulerAncestral,
    Heun,
    Dpmpp2m,
    DpmppSde,
    UniPc,
    Lcm,
    Ddim,
}

impl Solver {
    /// Parse the canonical lowercase name (the UI / recipe vocabulary). Unknown -> `None` (callers
    /// fall back to the model default + emit an event, N3).
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "euler" => Self::Euler,
            "euler_ancestral" => Self::EulerAncestral,
            "heun" => Self::Heun,
            "dpmpp_2m" => Self::Dpmpp2m,
            "dpmpp_sde" => Self::DpmppSde,
            "uni_pc" => Self::UniPc,
            "lcm" => Self::Lcm,
            "ddim" => Self::Ddim,
            _ => return None,
        })
    }

    /// The canonical lowercase name (round-trips with [`Self::from_name`]).
    pub fn name(self) -> &'static str {
        match self {
            Self::Euler => "euler",
            Self::EulerAncestral => "euler_ancestral",
            Self::Heun => "heun",
            Self::Dpmpp2m => "dpmpp_2m",
            Self::DpmppSde => "dpmpp_sde",
            Self::UniPc => "uni_pc",
            Self::Lcm => "lcm",
            Self::Ddim => "ddim",
        }
    }

    /// Whether the solver draws fresh per-step noise (needs a request seed for reproducibility).
    pub fn is_stochastic(self) -> bool {
        matches!(self, Self::EulerAncestral | Self::DpmppSde | Self::Lcm)
    }

    /// Every curated solver, in menu order.
    pub const ALL: [Solver; 8] = [
        Self::Euler,
        Self::EulerAncestral,
        Self::Heun,
        Self::Dpmpp2m,
        Self::DpmppSde,
        Self::UniPc,
        Self::Lcm,
        Self::Ddim,
    ];

    /// Box the matching [`Sampler`] for a backend `L`.
    pub fn boxed<L: LatentOps + 'static>(self) -> Box<dyn Sampler<L>> {
        match self {
            Self::Euler => Box::new(Euler),
            Self::EulerAncestral => Box::new(EulerAncestral::default()),
            Self::Heun => Box::new(Heun),
            Self::Dpmpp2m => Box::new(Dpmpp2m),
            Self::DpmppSde => Box::new(DpmppSde::default()),
            Self::UniPc => Box::new(UniPc),
            Self::Lcm => Box::new(Lcm),
            Self::Ddim => Box::new(Ddim),
        }
    }
}

/// Build the [`Sampler`] for a canonical solver name, or `None` if unknown (N3 fallback).
pub fn sampler_by_name<L: LatentOps + 'static>(name: &str) -> Option<Box<dyn Sampler<L>>> {
    Solver::from_name(name).map(Solver::boxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::model_sampling::{denoise, FlowModelSampling};
    use crate::sampling::{
        build_flow_sigmas, compute_mu, image_seq_len, CpuLatentOps, TimestepConvention,
    };

    fn flow_sigmas(steps: usize) -> Vec<f32> {
        build_flow_sigmas(steps, compute_mu(image_seq_len(1024, 1024), steps))
    }

    /// Drive a solver over a FLOW model with a CONSTANT velocity `v` (independent of x and t). The
    /// rectified-flow ODE `dx/dσ = v` is linear, so every *consistent* deterministic solver must land
    /// EXACTLY on `x_init + v·(σ_final − σ_init) = x_init − v·σ_0` (σ_final = 0). This catches any
    /// coefficient sign/scale bug.
    fn run_const_velocity<S: Sampler<CpuLatentOps>>(
        solver: &S,
        steps: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let sigmas = flow_sigmas(steps);
        let v = vec![0.37_f32, -0.12, 0.8, -0.5];
        let x_init = vec![0.3_f32, -1.1, 2.0, 0.05];
        let mut dn = |xx: &Vec<f32>, s: f32| denoise(&ops, &ms, xx, s, |_xin, _t| Ok(v.clone()));
        let got = solver
            .sample(&ops, &ms, &mut dn, x_init.clone(), &sigmas, 0)
            .unwrap();
        // Exact: x_init − v·σ_0.
        let want: Vec<f32> = x_init
            .iter()
            .zip(&v)
            .map(|(&xi, &vi)| xi - vi * sigmas[0])
            .collect();
        (got, want)
    }

    fn assert_close(got: &[f32], want: &[f32], tol: f32, label: &str) {
        assert_eq!(got.len(), want.len(), "{label}: length");
        let max = got
            .iter()
            .zip(want)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max < tol, "{label}: max abs diff {max:e} (tol {tol:e})");
    }

    #[test]
    fn deterministic_solvers_integrate_constant_velocity_exactly() {
        // Euler (from unified) + the deterministic curated solvers must all hit the exact solution.
        let (g, w) = run_const_velocity(&Euler, 12);
        assert_close(&g, &w, 1e-4, "euler");
        let (g, w) = run_const_velocity(&Heun, 12);
        assert_close(&g, &w, 1e-4, "heun");
        let (g, w) = run_const_velocity(&Dpmpp2m, 12);
        assert_close(&g, &w, 1e-4, "dpmpp_2m");
        let (g, w) = run_const_velocity(&UniPc, 12);
        assert_close(&g, &w, 1e-4, "uni_pc");
        let (g, w) = run_const_velocity(&Ddim, 12);
        assert_close(&g, &w, 1e-4, "ddim");
    }

    #[test]
    fn stochastic_solvers_are_finite_and_seed_reproducible() {
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let sigmas = flow_sigmas(10);
        let x_init = vec![0.3_f32, -1.1, 2.0, 0.05, 0.9, -0.3];
        let run = |solver: &dyn Sampler<CpuLatentOps>, seed: u64| {
            let mut dn = |xx: &Vec<f32>, s: f32| {
                denoise(&ops, &ms, xx, s, |xin, _t| {
                    Ok(xin.iter().map(|&v| 0.2 * v + 0.05).collect())
                })
            };
            solver
                .sample(&ops, &ms, &mut dn, x_init.clone(), &sigmas, seed)
                .unwrap()
        };
        for solver in [
            &EulerAncestral::default() as &dyn Sampler<CpuLatentOps>,
            &DpmppSde::default(),
            &Lcm,
        ] {
            let a = run(solver, 7);
            let b = run(solver, 7);
            let c = run(solver, 8);
            assert!(a.iter().all(|v| v.is_finite()), "non-finite output");
            assert_eq!(a, b, "same seed must reproduce");
            assert_ne!(a, c, "different seed must differ");
        }
    }

    #[test]
    fn ddim_equals_euler_on_flow() {
        // In VE-sigma space DDIM(η=0) coincides with Euler — verify on a non-trivial velocity field.
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let sigmas = flow_sigmas(8);
        let x_init = vec![0.4_f32, -0.7, 1.3];
        let mut model = |xx: &Vec<f32>, _t: f32| -> Result<Vec<f32>> {
            Ok(xx.iter().map(|&v| 0.3 * v + 0.1).collect())
        };
        let mut dn_e = |xx: &Vec<f32>, s: f32| denoise(&ops, &ms, xx, s, &mut model);
        let euler = Euler
            .sample(&ops, &ms, &mut dn_e, x_init.clone(), &sigmas, 0)
            .unwrap();
        let mut dn_d = |xx: &Vec<f32>, s: f32| denoise(&ops, &ms, xx, s, &mut model);
        let ddim = Ddim
            .sample(&ops, &ms, &mut dn_d, x_init.clone(), &sigmas, 0)
            .unwrap();
        assert_close(&euler, &ddim, 1e-5, "ddim_vs_euler");
    }

    #[test]
    fn second_order_solvers_track_euler_on_smooth_field() {
        // dpmpp_2m / uni_pc / heun should produce sane (finite, close-to-Euler) trajectories on a
        // smooth field — not identical (they're higher order) but in the same neighbourhood.
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let sigmas = flow_sigmas(16);
        let x_init = vec![0.4_f32, -0.7, 1.3, 0.2, -0.9];
        let model = |xx: &Vec<f32>, t: f32| -> Vec<f32> {
            xx.iter().map(|&v| 0.25 * v + 0.1 * (t).sin()).collect()
        };
        let mut dn_e =
            |xx: &Vec<f32>, s: f32| denoise(&ops, &ms, xx, s, |xin, t| Ok(model(xin, t)));
        let euler = Euler
            .sample(&ops, &ms, &mut dn_e, x_init.clone(), &sigmas, 0)
            .unwrap();
        for solver in [&Heun as &dyn Sampler<CpuLatentOps>, &Dpmpp2m, &UniPc] {
            let mut dn =
                |xx: &Vec<f32>, s: f32| denoise(&ops, &ms, xx, s, |xin, t| Ok(model(xin, t)));
            let out = solver
                .sample(&ops, &ms, &mut dn, x_init.clone(), &sigmas, 0)
                .unwrap();
            assert!(out.iter().all(|v| v.is_finite()));
            assert_close(&out, &euler, 0.5, "2nd-order near euler");
        }
    }

    #[test]
    fn registry_round_trips_and_boxes_all() {
        for s in Solver::ALL {
            assert_eq!(Solver::from_name(s.name()), Some(s));
            let _boxed: Box<dyn Sampler<CpuLatentOps>> = s.boxed();
        }
        assert!(sampler_by_name::<CpuLatentOps>("euler").is_some());
        assert!(sampler_by_name::<CpuLatentOps>("nope").is_none());
        assert!(Solver::Lcm.is_stochastic());
        assert!(!Solver::Heun.is_stochastic());
    }
}
