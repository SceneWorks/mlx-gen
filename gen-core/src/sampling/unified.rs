//! The callback `Sampler` trait + the first solver (Euler) of the unified framework (epic 7114, P1).
//!
//! Replaces the legacy precomputed-`StepCoeffs` design (`super`), which folds the whole step into one
//! affine `a_x·x + a_out·out + a_noise·ε` and so structurally cannot host a solver that needs more
//! than one model evaluation per step (Heun) or the previous steps' denoised history (DPM++ 2M,
//! UniPC). Here a [`Sampler`] drives a `denoise(x, σ) -> x0` callback as many times as its algorithm
//! needs, in ComfyUI / k-diffusion sigma space, advancing the latents through [`LatentOps`] only.
//!
//! The curated solver set (euler_ancestral, heun, dpmpp_2m, dpmpp_sde, uni_pc, lcm, ddim) lands on
//! this trait in sc-7117; this module ships the trait + plain [`Euler`] and proves the new path
//! reproduces the legacy [`super::FlowMatchPolicy`] within the epic's N1 parity gate.

use super::LatentOps;
use crate::Result;

/// A `denoise(x, σ) -> x0` callback the samplers drive — the engine's model wrapper (input scaling →
/// model forward → prediction-type recombination, see [`super::model_sampling::denoise`]). Boxed
/// `FnMut` form so the [`Sampler`] trait stays object-safe; `'a` is the closure's borrow.
pub type DenoiseFn<'a, L> =
    dyn FnMut(&<L as LatentOps>::Latent, f32) -> Result<<L as LatentOps>::Latent> + 'a;

/// A denoise integrator over a sigma schedule. Implementors call `denoise(x, σ)` — the model wrapper
/// returning a denoised `x0` estimate — and advance the latents using only [`LatentOps`]. The trait
/// is object-safe (`denoise` is a `&mut` [`DenoiseFn`], the method has no generic params) so an
/// engine selects a solver by name into a `Box<dyn Sampler<Ops>>` and the denoise loop never knows
/// which is running.
///
/// Generic over the backend `L` rather than per-method so the box can be built once per request.
pub trait Sampler<L: LatentOps> {
    /// Integrate `x` from `sigmas[0]` down to `sigmas[last]` (the schedule carries a trailing `0.0`).
    ///
    /// - `ops`: backend tensor ops.
    /// - `ms`: the engine's prediction-type + noise-schedule contract. Most solvers ignore it (the
    ///   `denoise` closure already folds it in); the consistency re-noise (`lcm`) reads
    ///   [`super::ModelSampling::noise_scaling_coeffs`] so its between-step re-noise matches the model's
    ///   convention (the flow convex blend vs the VE additive form).
    /// - `denoise`: `denoise(x, σ) -> x0`; called at least once per step.
    /// - `x`: starting latents (already at `sigmas[0]`).
    /// - `sigmas`: descending schedule, length `num_steps + 1`, trailing `0.0`.
    /// - `seed`: request seed for stochastic solvers' per-step noise (deterministic solvers ignore it).
    fn sample(
        &self,
        ops: &L,
        ms: &dyn super::ModelSampling,
        denoise: &mut DenoiseFn<'_, L>,
        x: L::Latent,
        sigmas: &[f32],
        seed: u64,
    ) -> Result<L::Latent>;
}

/// `to_d(x, σ, x0) = (x − x0) / σ` — the k-diffusion derivative (score-scaled) every ODE solver in
/// the curated set advances along. At `σ == 0` (a terminal node a solver should never integrate
/// *from*) the derivative is undefined; callers guard with [`is_terminal`].
pub(crate) fn to_d<L: LatentOps>(
    ops: &L,
    x: &L::Latent,
    sigma: f32,
    x0: &L::Latent,
) -> Result<L::Latent> {
    ops.scale(&ops.sub(x, x0)?, 1.0 / sigma)
}

/// Whether a schedule node is the terminal clean node (`σ == 0`), from which no step is integrated.
#[inline]
pub(crate) fn is_terminal(sigma: f32) -> bool {
    sigma == 0.0
}

/// Forward-Euler (1st-order) ODE solver — the k-diffusion `sample_euler` with no churn (`gamma = 0`).
///
/// Per step: `x0 = denoise(x, σ_i)`, `d = (x − x0)/σ_i`, `x ← x + d·(σ_{i+1} − σ_i)`. Driven over a
/// FLOW [`super::model_sampling::ModelSampling`] this is algebraically the legacy flow-match step
/// `x + v·(σ_{i+1} − σ_i)` (`d == v`); they differ only by the f32 rounding of the `to_d` round-trip,
/// which the N1 parity gate covers (the production Wan UniPC default already integrates through the
/// same `x0 = sample − v·σ` round-trip). See the `euler_matches_legacy_flow_match_within_eps` test.
#[derive(Clone, Copy, Debug, Default)]
pub struct Euler;

impl<L: LatentOps> Sampler<L> for Euler {
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
            let x0 = denoise(&x, sigma)?;
            if is_terminal(sigma) {
                // Degenerate leading 0 (no real schedule starts here): land on x0, no division.
                x = x0;
                continue;
            }
            let dt = sigmas[i + 1] - sigma;
            let d = to_d(ops, &x, sigma, &x0)?;
            x = ops.axpy(1.0, &x, dt, &d)?;
        }
        Ok(x)
    }
}

/// Apply one legacy [`super::StepCoeffs`] through [`LatentOps`]:
/// `x_next = a_x·x + a_out·out + a_noise·ε`.
///
/// The bridge that lets the legacy coefficient policies — FlowMatch / Lightning / LCM / TCD, the
/// accel lineage that does NOT route through an `x0` callback — run on the unified backend layer
/// during the P3/P4 per-engine migration, bit-identically to mlx-gen's `apply_step`. The byte-parity
/// branch (`a_x == 1 && a_noise == 0 ⇒ x + out·a_out`, via `axpy(1.0, …)` whose `1.0·x` is an f32
/// identity) is preserved. `out` is the RAW (CFG-combined) model output; stochastic re-noise draws
/// from [`LatentOps::randn_like`] (the noise *source* is backend-specific by design — cross-backend
/// bitwise equality of the draw is not a goal, per `StepRng`).
pub fn apply_coeffs<L: LatentOps>(
    ops: &L,
    c: &super::StepCoeffs,
    x: &L::Latent,
    out: &L::Latent,
    seed: u64,
    step: usize,
) -> Result<L::Latent> {
    let acc = ops.axpy(c.a_x, x, c.a_out, out)?;
    if c.a_noise != 0.0 {
        let noise = ops.randn_like(&acc, seed, step)?;
        return ops.axpy(1.0, &acc, c.a_noise, &noise);
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::model_sampling::{denoise, FlowModelSampling};
    use crate::sampling::{
        build_flow_sigmas, compute_mu, image_seq_len, AlphaSchedule, CpuLatentOps, FlowMatchPolicy,
        LcmPolicy, LightningPolicy, SamplerPolicy, TcdPolicy, TimestepConvention,
    };

    /// A deterministic, mild flow "model": velocity `v = 0.3·x + 0.1` (ignores the timestep). Stable
    /// enough that the legacy and unified trajectories stay comparable over a full schedule.
    fn stub_velocity(x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| 0.3 * v + 0.1).collect()
    }

    #[test]
    fn euler_single_flow_step_is_x_plus_v_dt() {
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        // One step from σ=0.8 to σ=0.5 over a constant velocity.
        let sigmas = [0.8_f32, 0.5];
        let v = vec![0.7_f32, -0.2];
        let x = vec![0.3_f32, 1.0];
        let mut dn = |xx: &Vec<f32>, s: f32| denoise(&ops, &ms, xx, s, |_xin, _t| Ok(v.clone()));
        let got = Euler
            .sample(&ops, &ms, &mut dn, x.clone(), &sigmas, 0)
            .unwrap();
        // Expected: x + v·(0.5−0.8) = x − 0.3·v.
        for ((g, &xi), &vi) in got.iter().zip(&x).zip(&v) {
            assert!((g - (xi + vi * (0.5 - 0.8))).abs() < 1e-5, "got {g}");
        }
    }

    #[test]
    fn euler_last_step_lands_on_x0() {
        // A schedule whose final node is 0 integrates exactly onto the denoised estimate.
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let sigmas = [0.5_f32, 0.0];
        let v = vec![0.4_f32, 0.4];
        let x = vec![0.2_f32, -0.6];
        let mut dn = |xx: &Vec<f32>, s: f32| denoise(&ops, &ms, xx, s, |_xin, _t| Ok(v.clone()));
        let got = Euler
            .sample(&ops, &ms, &mut dn, x.clone(), &sigmas, 0)
            .unwrap();
        // x0 at σ=0.5 = x − 0.5·v.
        for ((g, &xi), &vi) in got.iter().zip(&x).zip(&v) {
            assert!((g - (xi - 0.5 * vi)).abs() < 1e-5, "got {g}");
        }
    }

    /// THE byte-equivalence proof (sc-7115 acceptance): the new callback Euler over a FLOW
    /// ModelSampling reproduces the legacy `FlowMatchPolicy` (`x + v·(σ_{i+1}−σ_i)`, byte-parity
    /// branch) within the epic's N1 parity tolerance. Both paths feed the SAME (x, σ) to the SAME
    /// stub velocity model; the only difference is the unified path's `to_d` round-trip, an
    /// f32-cancellation that stays < 1e-5 over a full realistic shifted-flow schedule.
    #[test]
    fn euler_matches_legacy_flow_match_within_eps() {
        let ops = CpuLatentOps;
        // A realistic shifted-flow schedule (the FLUX/Qwen world): 8 steps, empirical μ.
        let mu = compute_mu(image_seq_len(1024, 1024), 8);
        let sigmas = build_flow_sigmas(8, mu);
        let x_init = vec![0.3_f32, -1.1, 2.0, 0.05, -0.4, 1.7];

        // Legacy path: FlowMatchPolicy + the apply_step byte-parity branch (x + out·a_out).
        let policy = FlowMatchPolicy::new(sigmas.clone(), TimestepConvention::Sigma);
        let mut legacy = x_init.clone();
        for i in 0..policy.num_steps() {
            let c = policy.coeffs(i); // c_in=1, timestep=σ_i, a_x=1, a_out=σ_{i+1}−σ_i, a_noise=0
            let out = stub_velocity(&legacy); // model input = c_in·x = x; timestep unused by the stub
            legacy = legacy
                .iter()
                .zip(&out)
                .map(|(&xi, &oi)| xi + oi * c.a_out)
                .collect();
        }

        // Unified path: callback Euler over a FLOW ModelSampling, same stub model.
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let mut dn =
            |xx: &Vec<f32>, s: f32| denoise(&ops, &ms, xx, s, |xin, _t| Ok(stub_velocity(xin)));
        let unified = Euler
            .sample(&ops, &ms, &mut dn, x_init.clone(), &sigmas, 0)
            .unwrap();

        assert_eq!(legacy.len(), unified.len());
        let max_abs = legacy
            .iter()
            .zip(&unified)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs < 1e-5,
            "unified Euler diverged from legacy FlowMatchPolicy by {max_abs:e} (> 1e-5)"
        );
    }

    #[test]
    fn euler_is_object_safe_boxed() {
        // Runtime solver selection: the trait must box (`Box<dyn Sampler<Ops>>`).
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let sampler: Box<dyn Sampler<CpuLatentOps>> = Box::new(Euler);
        let v = vec![0.5_f32];
        let mut dn = |xx: &Vec<f32>, s: f32| denoise(&ops, &ms, xx, s, |_xin, _t| Ok(v.clone()));
        let out = sampler
            .sample(&ops, &ms, &mut dn, vec![0.1_f32], &[0.6_f32, 0.0], 0)
            .unwrap();
        assert_eq!(out.len(), 1);
    }

    /// `apply_coeffs` reproduces the legacy affine step (`a_x·x + a_out·out`) BIT-FOR-BIT for every
    /// policy family — the accel lineage the story names (FlowMatch / Lightning / LCM / TCD). This is
    /// the foundation-level "accel policies on the new interface, byte-equivalence proven": the
    /// per-engine callback samplers for `lcm`/`ddim` land in sc-7117 and the engine adoption in
    /// P3/P4, but the LatentOps layer already subsumes the legacy `apply_step` exactly here.
    #[test]
    fn apply_coeffs_reproduces_legacy_policy_steps_bit_exact() {
        let ops = CpuLatentOps;
        let sched = AlphaSchedule::scaled_linear(1000, 0.00085, 0.012).unwrap();
        let x = vec![0.3_f32, -1.2, 2.5, 0.07];
        let out = vec![0.7_f32, 0.1, -0.4, 1.3];
        // Deterministic coeff sets (a_noise = 0) from each policy family.
        let policies: Vec<(&str, super::super::StepCoeffs)> = vec![
            (
                "flow",
                FlowMatchPolicy::new(vec![1.0, 0.75, 0.5, 0.25, 0.0], TimestepConvention::Sigma)
                    .coeffs(1),
            ),
            ("lightning", LightningPolicy::new(&sched, 1000, 4).coeffs(0)),
            (
                "lcm_last", // final step never re-noises -> deterministic
                LcmPolicy::new(sched.clone(), 1000, 50, 4).coeffs(3),
            ),
            (
                "tcd_eta0",
                TcdPolicy::new(sched.clone(), 1000, 50, 4, 0.0).coeffs(0),
            ),
        ];
        for (name, c) in policies {
            assert_eq!(c.a_noise, 0.0, "{name}: expected a deterministic coeff set");
            let got = apply_coeffs(&ops, &c, &x, &out, 0, 0).unwrap();
            // Legacy reference: a_x·x + a_out·out (the byte-parity branch is a_x==1 special-case;
            // both reduce to the same f32 arithmetic on the CPU backend).
            let want: Vec<f32> = x
                .iter()
                .zip(&out)
                .map(|(&xi, &oi)| c.a_x * xi + c.a_out * oi)
                .collect();
            assert_eq!(
                got, want,
                "{name}: apply_coeffs diverged from legacy affine step"
            );
        }
    }
}
