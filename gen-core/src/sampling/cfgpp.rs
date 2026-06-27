//! CFG++ solver variants (epic 7434, sc-8255): the sampler-layer half of CFG++ (Chung et al., ICLR
//! 2025 — https://github.com/CFGpp-diffusion/CFGpp; no upstream LICENSE, the *formula* is reimplemented
//! here, nothing is vendored).
//!
//! ## Why this is a separate module, not a flag on the existing solvers
//! The unified [`super::unified::Sampler`] drives a `denoise(x, σ) -> x0` callback that returns ONE
//! denoised estimate — the engine's CFG combine is already collapsed inside the `predict` closure
//! before gen-core ever sees it. Standard guidance methods (`cfg` / `cfg_rescale` / `apg`) are pure
//! output-combine and so ride that single-`x0` callback unchanged. **CFG++ cannot**: it anchors the
//! step on the *guided* `x0` but builds the renoise derivative from the *unconditional* `x0`, so it
//! needs BOTH per step. Rather than widen the shared `Sampler`/`DenoiseFn` contract (which every
//! solver and every adopted engine depends on), CFG++ is a parallel [`CfgPpSampler`] trait driven by a
//! [`CfgPpDenoiseFn`] that yields `(guided_x0, uncond_x0)`. Existing solvers/engines are untouched;
//! the engine opts into this path only when the guidance method is `cfg_pp` (sc-8256 adoption).
//!
//! ## What ships (spike sc-8254, locked)
//! `euler_cfg++` and `ddim_cfg++` (identical in the unified VE-sigma space, exactly as the base
//! `ddim` coincides with `euler` — see `ddim_equals_euler_on_flow`) plus `dpmpp_2m_cfg++` — the two
//! solvers the CFG++ paper validates, faithfully ported from ComfyUI's `sample_euler_cfg_pp` /
//! `sample_dpmpp_2m_cfg_pp`. `heun` / `uni_pc` / `lcm` / `dpmpp_sde` / `euler_ancestral` have no
//! paper-validated CFG++ form and are gated off ([`base_supports_cfgpp`]).

use super::unified::{is_terminal, to_d};
use super::{LatentOps, Solver};
use crate::Result;

/// A `denoise(x, σ) -> (guided_x0, uncond_x0)` callback the CFG++ samplers drive. Unlike the standard
/// [`super::DenoiseFn`] (one `x0`), it surfaces the unconditional estimate the engine would otherwise
/// discard, so the solver can renoise from it. Boxed `FnMut` so [`CfgPpSampler`] stays object-safe.
pub type CfgPpDenoiseFn<'a, L> = dyn FnMut(
        &<L as LatentOps>::Latent,
        f32,
    ) -> Result<(<L as LatentOps>::Latent, <L as LatentOps>::Latent)>
    + 'a;

/// A CFG++ denoise integrator. Mirror of [`super::unified::Sampler`] but over the guided+uncond
/// [`CfgPpDenoiseFn`]. All shipped variants are deterministic-in-sigma (no per-step noise, no `ms`
/// dependency), so the signature drops `seed`/`ms`; a future ancestral CFG++ variant would reintroduce
/// them on its own trait or via an extension.
pub trait CfgPpSampler<L: LatentOps> {
    /// Integrate `x` from `sigmas[0]` down to `sigmas[last]` (trailing `0.0`), renoising each step
    /// from the unconditional estimate while landing on the guided one.
    fn sample(
        &self,
        ops: &L,
        denoise: &mut CfgPpDenoiseFn<'_, L>,
        x: L::Latent,
        sigmas: &[f32],
    ) -> Result<L::Latent>;
}

/// `λ(σ) = −ln σ` — the VE half-log-SNR time the DPM multistep solver integrates in (local copy of
/// the `solvers` helper; gen-core keeps schedule math in host f64).
#[inline]
fn lambda(sigma: f32) -> f64 {
    -(sigma.max(1e-12) as f64).ln()
}

// =================================================================================================
// euler_cfg++ / ddim_cfg++ — ComfyUI `sample_euler_cfg_pp` (= euler_ancestral_cfg_pp, eta=0).
// =================================================================================================

/// CFG++ Euler (= CFG++ DDIM in VE-sigma space). Per step: anchor on the guided `x0`, but take the
/// renoise derivative from the unconditional `x0`:
///
/// `x_{i+1} = guided_x0 + σ_{i+1}·(x − uncond_x0)/σ_i`.
///
/// Algebraically `= (σ_{i+1}/σ_i)·x + guided_x0 − (σ_{i+1}/σ_i)·uncond_x0`, i.e. plain Euler with the
/// renoise term's `x0` swapped to the unconditional branch. With no guidance gap (uncond == guided)
/// it reduces to plain Euler within the framework's `to_d` round-trip tolerance (the N1 gate).
#[derive(Clone, Copy, Debug, Default)]
pub struct EulerCfgPp;

impl<L: LatentOps> CfgPpSampler<L> for EulerCfgPp {
    fn sample(
        &self,
        ops: &L,
        denoise: &mut CfgPpDenoiseFn<'_, L>,
        mut x: L::Latent,
        sigmas: &[f32],
    ) -> Result<L::Latent> {
        for i in 0..sigmas.len().saturating_sub(1) {
            let sigma = sigmas[i];
            let (guided, uncond) = denoise(&x, sigma)?;
            if is_terminal(sigma) {
                x = guided;
                continue;
            }
            // d = (x − uncond_x0)/σ  (the UNCONDITIONAL derivative); land on the GUIDED x0.
            let d = to_d(ops, &x, sigma, &uncond)?;
            x = ops.axpy(1.0, &guided, sigmas[i + 1], &d)?;
        }
        Ok(x)
    }
}

// =================================================================================================
// dpmpp_2m_cfg++ — ComfyUI `sample_dpmpp_2m_cfg_pp` (DPM-Solver++(2M), CFG++ multistep).
// =================================================================================================

/// CFG++ DPM-Solver++(2M). Faithful port of ComfyUI `sample_dpmpp_2m_cfg_pp`:
/// with `a = exp(−h) = σ_{i+1}/σ_i` and the multistep ratio `r = h_last/h`,
/// ```text
///   first step / σ_{i+1}==0:  x = a·x + guided − a·uncond
///   otherwise:                x = a·x + guided − a·uncond + (1−a)/(2r)·(guided − old_uncond)
/// ```
/// The history carried between steps is the **unconditional** denoised (`old_uncond`), per ComfyUI.
/// Its first-order limit (no history, or terminal) is exactly [`EulerCfgPp`] — the consistency the
/// `dpmpp2m_first_step_matches_euler_cfgpp` test pins.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dpmpp2mCfgPp;

impl<L: LatentOps> CfgPpSampler<L> for Dpmpp2mCfgPp {
    fn sample(
        &self,
        ops: &L,
        denoise: &mut CfgPpDenoiseFn<'_, L>,
        mut x: L::Latent,
        sigmas: &[f32],
    ) -> Result<L::Latent> {
        let mut old_uncond: Option<L::Latent> = None;
        for i in 0..sigmas.len().saturating_sub(1) {
            let sigma = sigmas[i];
            let s_next = sigmas[i + 1];
            let (guided, uncond) = denoise(&x, sigma)?;
            let a = s_next / sigma; // exp(−h)
                                    // x = a·x + guided − a·uncond  (the first-order CFG++ step)
            let mut x_next = ops.axpy(a, &x, 1.0, &guided)?;
            x_next = ops.axpy(1.0, &x_next, -a, &uncond)?;
            // Second-order correction once history exists and we are not on the terminal step.
            if s_next != 0.0 {
                if let Some(old) = &old_uncond {
                    let h = lambda(s_next) - lambda(sigma);
                    let h_last = lambda(sigma) - lambda(sigmas[i - 1]);
                    let r = h_last / h;
                    let c = ((1.0 - a as f64) / (2.0 * r)) as f32; // (1−a)/(2r)
                                                                   // + c·(guided − old_uncond)
                    let diff = ops.axpy(1.0, &guided, -1.0, old)?;
                    x_next = ops.axpy(1.0, &x_next, c, &diff)?;
                }
            }
            x = x_next;
            old_uncond = Some(uncond);
        }
        Ok(x)
    }
}

// =================================================================================================
// Registry — map a curated base [`Solver`] + the `cfg_pp` guidance method to its CFG++ variant.
// =================================================================================================

/// Whether a curated base solver has a paper-validated CFG++ variant (spike sc-8254 ship/gate list).
/// `euler`/`ddim`/`dpmpp_2m` → yes; everything else → no (the worker gates `cfg_pp` off + N3-falls
/// back when an incompatible sampler is paired with it).
pub fn base_supports_cfgpp(base: Solver) -> bool {
    matches!(base, Solver::Euler | Solver::Ddim | Solver::Dpmpp2m)
}

/// Box the CFG++ sampler for a curated base solver, or `None` if that solver has no CFG++ form.
/// `euler`/`ddim` → [`EulerCfgPp`] (they coincide in VE-sigma space); `dpmpp_2m` → [`Dpmpp2mCfgPp`].
pub fn cfgpp_sampler_for<L: LatentOps + 'static>(base: Solver) -> Option<Box<dyn CfgPpSampler<L>>> {
    match base {
        Solver::Euler | Solver::Ddim => Some(Box::new(EulerCfgPp)),
        Solver::Dpmpp2m => Some(Box::new(Dpmpp2mCfgPp)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::model_sampling::{denoise, FlowModelSampling};
    use crate::sampling::unified::{Euler, Sampler};
    use crate::sampling::{
        build_flow_sigmas, compute_mu, image_seq_len, CpuLatentOps, TimestepConvention,
    };

    fn flow_sigmas(steps: usize) -> Vec<f32> {
        build_flow_sigmas(steps, compute_mu(image_seq_len(1024, 1024), steps))
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    /// Build a CFG++ denoise callback over a FLOW model from a guided velocity and an uncond velocity.
    #[allow(clippy::type_complexity)] // a test helper mirroring the boxed `CfgPpDenoiseFn` shape
    fn pair_denoise<'a>(
        ops: &'a CpuLatentOps,
        ms: &'a FlowModelSampling,
        v_guided: &'a [f32],
        v_uncond: &'a [f32],
    ) -> impl FnMut(&Vec<f32>, f32) -> Result<(Vec<f32>, Vec<f32>)> + 'a {
        move |x: &Vec<f32>, s: f32| {
            let g = denoise(ops, ms, x, s, |_xin, _t| Ok(v_guided.to_vec()))?;
            let u = denoise(ops, ms, x, s, |_xin, _t| Ok(v_uncond.to_vec()))?;
            Ok((g, u))
        }
    }

    /// N1 no-op: with NO guidance gap (uncond == guided), CFG++ Euler reproduces plain Euler within
    /// the epic's parity tolerance (< 1e-5 — the `to_d` round-trip residual, NOT bit-exact, because
    /// `guided + σ_next·d` is a different f32 op-order than euler's `x + dt·d`).
    #[test]
    fn euler_cfgpp_matches_euler_when_no_gap() {
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let sigmas = flow_sigmas(10);
        let x_init = vec![0.3_f32, -1.1, 2.0, 0.05, -0.4];
        let v = |x: &Vec<f32>| -> Vec<f32> { x.iter().map(|&t| 0.3 * t + 0.1).collect() };

        let mut dn_e = |xx: &Vec<f32>, s: f32| denoise(&ops, &ms, xx, s, |xin, _t| Ok(v(xin)));
        let euler = Euler
            .sample(&ops, &ms, &mut dn_e, x_init.clone(), &sigmas, 0)
            .unwrap();

        // guided == uncond == the same model output.
        let mut dn_p = |xx: &Vec<f32>, s: f32| {
            let x0 = denoise(&ops, &ms, xx, s, |xin, _t| Ok(v(xin)))?;
            Ok((x0.clone(), x0))
        };
        let cfgpp = EulerCfgPp
            .sample(&ops, &mut dn_p, x_init.clone(), &sigmas)
            .unwrap();
        assert!(
            max_abs(&euler, &cfgpp) < 1e-5,
            "euler_cfg++ diverged from euler with no gap: {:e}",
            max_abs(&euler, &cfgpp)
        );
    }

    /// Constant-velocity exactness (the gen-core per-solver coherence gate). With a constant velocity
    /// and no guidance gap, both CFG++ solvers must integrate EXACTLY onto x_init − v·σ_0.
    #[test]
    fn cfgpp_solvers_integrate_constant_velocity_exactly() {
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let sigmas = flow_sigmas(12);
        let v = vec![0.37_f32, -0.12, 0.8, -0.5, 0.2];
        let x_init = vec![0.3_f32, -1.1, 2.0, 0.05, -0.4];
        let want: Vec<f32> = x_init
            .iter()
            .zip(&v)
            .map(|(&xi, &vi)| xi - vi * sigmas[0])
            .collect();

        for (label, run) in [("euler_cfg++", 0u8), ("dpmpp_2m_cfg++", 1u8)] {
            let mut dn = pair_denoise(&ops, &ms, &v, &v);
            let got = if run == 0 {
                EulerCfgPp
                    .sample(&ops, &mut dn, x_init.clone(), &sigmas)
                    .unwrap()
            } else {
                Dpmpp2mCfgPp
                    .sample(&ops, &mut dn, x_init.clone(), &sigmas)
                    .unwrap()
            };
            assert!(
                max_abs(&got, &want) < 1e-4,
                "{label}: const-velocity not exact: {:e}",
                max_abs(&got, &want)
            );
        }
    }

    /// The renoise really comes from the UNCONDITIONAL branch, and the trajectory genuinely DIFFERS
    /// from what plain CFG (renoise from the guided estimate) would produce — the whole point of CFG++.
    #[test]
    fn euler_cfgpp_renoises_from_uncond_and_differs_from_plain_cfg() {
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let sigma = 0.8_f32;
        let s_next = 0.5_f32;
        let x = vec![0.4_f32, -0.2, 1.0];
        let v_cond = vec![0.6_f32, 0.1, -0.3];
        let v_uncond = vec![0.2_f32, 0.0, -0.1];
        let lambda_g = 0.7_f32;
        let v_guided: Vec<f32> = v_uncond
            .iter()
            .zip(&v_cond)
            .map(|(&u, &c)| u + lambda_g * (c - u))
            .collect();
        let x0_guided = denoise(&ops, &ms, &x, sigma, |_x, _t| Ok(v_guided.clone())).unwrap();
        let x0_uncond = denoise(&ops, &ms, &x, sigma, |_x, _t| Ok(v_uncond.clone())).unwrap();

        let mut dn = |_x: &Vec<f32>, _s: f32| Ok((x0_guided.clone(), x0_uncond.clone()));
        let got = EulerCfgPp
            .sample(&ops, &mut dn, x.clone(), &[sigma, s_next])
            .unwrap();

        // Hand reference: guided + s_next·(x − uncond)/σ.
        let d_u = to_d(&ops, &x, sigma, &x0_uncond).unwrap();
        let hand = ops.axpy(1.0, &x0_guided, s_next, &d_u).unwrap();
        assert!(max_abs(&got, &hand) < 1e-6, "cfg++ step != hand formula");

        // Plain-CFG step renoises from the guided estimate — must be a distinct trajectory.
        let d_g = to_d(&ops, &x, sigma, &x0_guided).unwrap();
        let plain = ops.axpy(1.0, &x0_guided, s_next, &d_g).unwrap();
        assert!(
            max_abs(&got, &plain) > 1e-3,
            "cfg++ collapsed onto plain CFG (the uncond swap is not live)"
        );
    }

    /// Internal consistency: dpmpp_2m_cfg++'s first-order limit (first step / no history) is exactly
    /// euler_cfg++. Validates the multistep port's base case against the simpler, hand-derived solver.
    #[test]
    fn dpmpp2m_first_step_matches_euler_cfgpp() {
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let sigmas = [0.8_f32, 0.5]; // single step -> dpmpp_2m has no history, must equal euler_cfg++
        let v_g = vec![0.6_f32, 0.1, -0.3, 0.2];
        let v_u = vec![0.2_f32, 0.0, -0.1, 0.05];
        let x_init = vec![0.4_f32, -0.2, 1.0, 0.3];

        let mut dn_e = pair_denoise(&ops, &ms, &v_g, &v_u);
        let e = EulerCfgPp
            .sample(&ops, &mut dn_e, x_init.clone(), &sigmas)
            .unwrap();
        let mut dn_d = pair_denoise(&ops, &ms, &v_g, &v_u);
        let d = Dpmpp2mCfgPp
            .sample(&ops, &mut dn_d, x_init.clone(), &sigmas)
            .unwrap();
        assert!(
            max_abs(&e, &d) < 1e-5,
            "dpmpp_2m_cfg++ first-order limit != euler_cfg++: {:e}",
            max_abs(&e, &d)
        );
    }

    /// Gating + registry: exactly euler/ddim/dpmpp_2m resolve a CFG++ sampler; the rest are gated off.
    #[test]
    fn registry_gates_to_compatible_solvers() {
        for s in [Solver::Euler, Solver::Ddim, Solver::Dpmpp2m] {
            assert!(base_supports_cfgpp(s), "{s:?} should support cfg++");
            assert!(cfgpp_sampler_for::<CpuLatentOps>(s).is_some());
        }
        for s in [
            Solver::Heun,
            Solver::UniPc,
            Solver::Lcm,
            Solver::DpmppSde,
            Solver::EulerAncestral,
        ] {
            assert!(!base_supports_cfgpp(s), "{s:?} must be gated off");
            assert!(cfgpp_sampler_for::<CpuLatentOps>(s).is_none());
        }
    }
}
