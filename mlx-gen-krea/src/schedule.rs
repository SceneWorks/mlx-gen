//! Krea 2's rectified-flow (v-parameterization) timestep schedule + Euler sampler (reference
//! `sampling.py`). The published `model_index.json` names `FlowMatchEulerDiscreteScheduler`, but the
//! authoritative K2 sampler is the functional `timesteps` schedule below — there is no Scheduler
//! class; the loop is a plain forward-Euler integration of the flow ODE `t: 1 → 0`.
//!
//! ## The schedule (reference `sampling.py::timesteps`)
//! A uniform `linspace(1, 0, steps+1)` grid is **exponentially time-shifted** by `mu`:
//! ```text
//!   ts = exp(mu) / (exp(mu) + (1/ts − 1)^sigma)
//! ```
//! (`sigma = 1` for K2). The shift fixes the endpoints (`shift(1) = 1`, `shift(0) = 0`), so the result
//! is a descending sigma schedule `[1.0 … 0.0]` of length `steps+1` with a trailing `0.0` — exactly
//! what the core [`mlx_gen::FlowMatchSampler`] integrates (`x + v·(σ_{i+1} − σ_i)`, the raw σ fed to
//! the DiT as its timestep, which scales ×1000 internally — `crate::transformer::temb`).
//!
//! `mu` is either:
//! - **fixed** — the TDM-distilled **Turbo** checkpoint was trained at `mu = 1.15` regardless of
//!   resolution ([`TURBO_MU`]); or
//! - **resolution-dynamic** — linearly interpolated in image-sequence length between the published
//!   scheduler-config endpoints (`base_image_seq_len 256 → base_shift 0.5`,
//!   `max_image_seq_len 6400 → max_shift 1.15`), the undistilled **Raw** path ([`mu_for_seq_len`]).
//!
//! The Euler loop itself (and CFG, which calls the model a second time) is the pipeline's job (sc-7571
//! Turbo e2e; Raw CFG inference is a later P3 concern) — this module owns only the schedule + the
//! core-sampler construction, the family-neutral seam the rest of the workspace shares.

use mlx_gen::FlowMatchSampler;

/// Turbo's fixed timestep-shift `mu` — the value the TDM distillation was trained at (reference CLI
/// default `--mu`, applied resolution-independently for the distilled student).
pub const TURBO_MU: f64 = 1.15;
/// Turbo default denoising steps (the few-step distilled student; reference `is_distilled`).
pub const TURBO_STEPS: usize = 8;
/// The reference shift exponent `sigma` (always `1.0` for K2; kept explicit to mirror `timesteps`).
pub const SHIFT_EXPONENT: f64 = 1.0;

/// Resolution → `mu` interpolation endpoints, in **image-sequence-length** space. Mirror the published
/// `scheduler_config.json` (`base_image_seq_len`/`max_image_seq_len`, `base_shift`/`max_shift`) and the
/// reference CLI defaults (`minres 256`, `maxres 1280`; `compression·patch = 8·2 = 16`, so
/// `x = (res/16)²` → `x1 = 256`, `x2 = 6400`).
pub const BASE_SEQ_LEN: f64 = 256.0;
pub const MAX_SEQ_LEN: f64 = 6400.0;
pub const BASE_SHIFT: f64 = 0.5;
pub const MAX_SHIFT: f64 = 1.15;

/// Linearly interpolate `mu` in image-sequence length (reference `mu = slope·seq_len + (y1 −
/// slope·x1)`, `slope = (y2 − y1)/(x2 − x1)`) — the Raw dynamic-shift path. Not clamped (the reference
/// extrapolates beyond the endpoints).
pub fn mu_for_seq_len(seq_len: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let slope = (y2 - y1) / (x2 - x1);
    slope * seq_len + (y1 - slope * x1)
}

/// `mu` for an image token count using the published scheduler-config endpoints — the convenience form
/// of [`mu_for_seq_len`] for the Raw dynamic path.
pub fn dynamic_mu(seq_len: f64) -> f64 {
    mu_for_seq_len(seq_len, BASE_SEQ_LEN, BASE_SHIFT, MAX_SEQ_LEN, MAX_SHIFT)
}

/// Reference `sampling.py::timesteps`: the exponentially `mu`-shifted `linspace(1, 0, steps+1)` sigma
/// schedule (descending, length `steps+1`, endpoints `1.0 … 0.0`). Computed in f64 then narrowed to
/// the `f32` the core sampler stores (the reference computes in f32; the f64 intermediate only tightens
/// the rounding, well within the flow-match tolerance). The `t = 0` node maps to exactly `0.0`
/// (`1/0 → ∞ → exp(mu)/∞ = 0`), giving the trailing terminal `0.0` [`FlowMatchSampler`] expects.
pub fn krea_sigmas(steps: usize, mu: f64) -> Vec<f32> {
    let n = steps.max(1);
    let e = mu.exp();
    (0..=n)
        .map(|i| {
            let t = 1.0 - (i as f64) / (n as f64); // linspace(1, 0, n+1)
            let shifted = e / (e + (1.0 / t - 1.0).powf(SHIFT_EXPONENT));
            shifted as f32
        })
        .collect()
}

/// The fixed-`mu` **Turbo** sigma schedule (`mu = 1.15`) — the byte-exact distilled default.
pub fn turbo_sigmas(steps: usize) -> Vec<f32> {
    krea_sigmas(steps, TURBO_MU)
}

/// Build the core flow-match Euler sampler for the **Turbo** schedule. The sampler walks the descending
/// sigmas `x + v·(σ_{i+1} − σ_i)` (= the reference `img + (tprev − tcurr)·v`) with `timestep(i) = σ_i`
/// passed raw to the DiT.
pub fn turbo_sampler(steps: usize) -> FlowMatchSampler {
    FlowMatchSampler::new(turbo_sigmas(steps))
}

/// Build the resolution-**dynamic** (Raw) flow-match sampler: `mu` interpolated in `seq_len` (image
/// token count) via [`dynamic_mu`], then the same exponential shift.
pub fn dynamic_sampler(steps: usize, seq_len: f64) -> FlowMatchSampler {
    FlowMatchSampler::new(krea_sigmas(steps, dynamic_mu(seq_len)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::DiffusionSampler;

    /// Assert a sigma schedule matches the reference values (kept at the f64 precision the reference
    /// `sampling.py` prints, so the literals stay verbatim) to within f32-narrowing tolerance.
    fn assert_close(got: &[f32], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "schedule length");
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert!((g as f64 - w).abs() < 1e-5, "sigma[{i}] = {g}, want {w}");
        }
    }

    /// Reference `timesteps(steps=8, mu=1.15)` (the Turbo schedule), values from the upstream
    /// `sampling.py` run on the published checkpoint settings.
    #[test]
    fn turbo_schedule_matches_reference() {
        let want = [
            1.0, 0.95672369, 0.90453076, 0.84034878, 0.75951093, 0.65456682, 0.51284409,
            0.31090108, 0.0,
        ];
        assert_close(&turbo_sigmas(TURBO_STEPS), &want);
        // Endpoints are exact (shift fixes 1→1, 0→0) — the terminal 0.0 the sampler integrates to.
        let s = turbo_sigmas(8);
        assert_eq!(s.first().copied(), Some(1.0));
        assert_eq!(s.last().copied(), Some(0.0));
    }

    /// The dynamic `mu` is linear in image-sequence length through the published scheduler endpoints.
    #[test]
    fn dynamic_mu_matches_reference() {
        assert!((dynamic_mu(256.0) - 0.5).abs() < 1e-9, "base endpoint");
        assert!((dynamic_mu(6400.0) - 1.15).abs() < 1e-9, "max endpoint");
        assert!(
            (dynamic_mu(4096.0) - 0.90625).abs() < 1e-9,
            "1024² interior"
        );
    }

    /// Reference `timesteps(seq_len=4096, steps=4, x1=256, x2=6400)` — the Raw dynamic-shift path
    /// (`mu = 0.90625`).
    #[test]
    fn dynamic_schedule_matches_reference() {
        let want = [1.0, 0.88130659, 0.71223223, 0.45205718, 0.0];
        assert_close(&krea_sigmas(4, dynamic_mu(4096.0)), &want);
    }

    /// The schedule drives the core flow-match sampler: `num_steps = steps`, `timestep(i) = σ_i`.
    #[test]
    fn turbo_sampler_wires_core_flow_match() {
        let sigmas = turbo_sigmas(TURBO_STEPS);
        let s = turbo_sampler(TURBO_STEPS);
        assert_eq!(s.num_steps(), TURBO_STEPS);
        for (i, &sig) in sigmas.iter().take(TURBO_STEPS).enumerate() {
            assert!((s.timestep(i) - sig).abs() < 1e-6, "timestep[{i}]");
            assert!((s.sigma(i) - sig).abs() < 1e-6, "sigma[{i}]");
        }
    }

    /// `steps = 1` is the degenerate `[1.0, 0.0]` (a single Euler hop) — not a panic.
    #[test]
    fn single_step_schedule() {
        let s = turbo_sigmas(1);
        assert_eq!(s.len(), 2);
        assert_eq!(s, vec![1.0, 0.0]);
    }
}
