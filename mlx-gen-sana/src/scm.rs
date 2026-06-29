//! SCM (sCM / TrigFlow continuous-time consistency) scheduler — the few-step sampler behind
//! **SANA-Sprint** (epic 8485, story sc-8490). Faithful port of diffusers `SCMScheduler`
//! (`schedulers/scheduling_scm.py`) + the Sprint pipeline's per-step trigflow recombination
//! (`pipelines/sana/pipeline_sana_sprint.py`).
//!
//! ## Why this is NOT the unified flow-match sampler
//!
//! The epic-7114 unified framework ([`mlx_gen::run_flow_sampler`] / the `Solver` menu) integrates in
//! **sigma space** with an `x0 = k_x·x + k_out·output` recombination. SCM is a different
//! parameterization: it works in **angle / atan space** (`t ∈ [0, π/2]`, the TrigFlow continuous-time
//! consistency map), predicts `x0 = cos(s)·x − sin(s)·output`, and **renoises** to the next angle
//! `x_{t} = cos(t)·x0 + sin(t)·noise·σ_data`. The model output itself is recombined trigonometrically
//! *before* the scheduler step (the Sprint pipeline's `noise_pred = …` block). None of that maps onto
//! the flow-match `ModelSampling`/`Solver` contract, so SCM is a dedicated scheduler here — the
//! consistency analog of the crate-local [`mlx_gen::FlowMatchEuler`] (which is itself a small
//! scheduler the engines drive directly, not a `Solver`). The unified framework's **seam** we DO reuse
//! is the run-loop contract (per-step `eval` boundary + cooperative cancel + monotone progress),
//! mirrored by [`crate::pipeline::denoise_sprint`].
//!
//! ## Guidance axis (epic 7434)
//!
//! Sprint is **CFG-free**: a single trunk forward per step with the guidance scale fed as an
//! *embedded scalar* (the trunk's guidance embedder, [`crate::transformer::SanaTransformer::forward_with_guidance`]).
//! There is no conditional/unconditional pair to combine, so the epic-7434 guidance library (`cfg` /
//! `cfg_rescale` / `apg` / `cfg_pp` — all *combine* operators) does not apply; Sprint slots into the
//! guidance axis as the embedded / CFG-free case (the descriptor advertises NO
//! `supported_guidance_methods`, [`crate::model`]).

use std::f32::consts::FRAC_PI_2;

/// diffusers `SCMScheduler` default `max_timesteps` (`arctan(80/0.5)`-adjacent; the Sprint default is
/// `1.57080 ≈ π/2`).
pub const DEFAULT_MAX_TIMESTEP: f32 = FRAC_PI_2;
/// diffusers Sprint default `intermediate_timesteps` (only consulted when `num_inference_steps == 2`).
pub const DEFAULT_INTERMEDIATE_TIMESTEP: f32 = 1.3;
/// diffusers `SCMScheduler` `sigma_data` — the std-dev of the multi-step renoise.
pub const SIGMA_DATA: f32 = 0.5;

/// A SANA-Sprint SCM (TrigFlow consistency) denoising schedule: the descending angle timesteps
/// `t ∈ [max_timesteps … 0]`, length `num_steps + 1` (the trailing `0` is the clean endpoint).
///
/// Mirrors diffusers `SCMScheduler.set_timesteps`:
/// * `num_steps == 2` → `[max_timesteps, intermediate_timesteps, 0]`;
/// * otherwise → `linspace(max_timesteps, 0, num_steps + 1)`.
#[derive(Clone, Debug)]
pub struct ScmScheduler {
    /// Descending angle timesteps, length `num_steps + 1`.
    pub timesteps: Vec<f32>,
    /// `sigma_data` (the renoise std-dev; `0.5` for Sprint).
    pub sigma_data: f32,
}

impl ScmScheduler {
    /// Build the Sprint schedule for `num_steps` (the diffusers `set_timesteps` default path:
    /// `max_timesteps = π/2`, `intermediate_timesteps = 1.3`). 1–4 steps is the Sprint operating band.
    pub fn new(num_steps: usize) -> Self {
        Self::with_timesteps(
            num_steps,
            DEFAULT_MAX_TIMESTEP,
            DEFAULT_INTERMEDIATE_TIMESTEP,
        )
    }

    /// Build the schedule for `num_steps` with explicit `max_timesteps` / `intermediate_timesteps`
    /// (the latter only used for the `num_steps == 2` two-point schedule, matching diffusers).
    pub fn with_timesteps(
        num_steps: usize,
        max_timesteps: f32,
        intermediate_timesteps: f32,
    ) -> Self {
        let timesteps = if num_steps == 2 {
            vec![max_timesteps, intermediate_timesteps, 0.0]
        } else {
            let n = num_steps;
            (0..=n)
                .map(|i| max_timesteps * (1.0 - i as f32 / n as f32))
                .collect()
        };
        Self {
            timesteps,
            sigma_data: SIGMA_DATA,
        }
    }

    /// Wrap an explicit descending angle schedule (length `num_steps + 1`).
    pub fn from_timesteps(timesteps: Vec<f32>) -> Self {
        Self {
            timesteps,
            sigma_data: SIGMA_DATA,
        }
    }

    /// Number of denoise iterations (loop count) = `timesteps.len() - 1`.
    pub fn num_steps(&self) -> usize {
        self.timesteps.len() - 1
    }

    /// Whether this is a true single-step schedule (one renoise-free step). diffusers skips the
    /// between-step noise when `len(timesteps) == 1` *after* the pipeline drops the trailing `0`
    /// (`timesteps = timesteps[:-1]`), i.e. when `num_steps == 1`.
    pub fn is_single_step(&self) -> bool {
        self.num_steps() <= 1
    }

    /// The **SCM conditioning timestep** the trunk embeds at step `i` (diffusers
    /// `scm_timestep = sin(t)/(cos(t)+sin(t))` where `t` is the angle timestep). The trunk consumes
    /// this, NOT the raw angle.
    pub fn scm_timestep(&self, i: usize) -> f32 {
        let t = self.timesteps[i];
        let (s, c) = (t.sin(), t.cos());
        s / (c + s)
    }

    /// The model-input scale at step `i` (diffusers
    /// `sqrt(scm_t² + (1 - scm_t)²)`, applied to `latents / sigma_data`).
    pub fn input_scale(&self, i: usize) -> f32 {
        let st = self.scm_timestep(i);
        (st * st + (1.0 - st) * (1.0 - st)).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-5, "got {a} want {b}");
    }

    #[test]
    fn two_step_schedule_is_diffusers_intermediate() {
        let s = ScmScheduler::new(2);
        assert_eq!(s.timesteps.len(), 3);
        approx(s.timesteps[0], DEFAULT_MAX_TIMESTEP);
        approx(s.timesteps[1], DEFAULT_INTERMEDIATE_TIMESTEP);
        approx(s.timesteps[2], 0.0);
        assert_eq!(s.num_steps(), 2);
    }

    #[test]
    fn four_step_schedule_is_linspace() {
        let s = ScmScheduler::new(4);
        assert_eq!(s.timesteps.len(), 5);
        // linspace(pi/2, 0, 5).
        for (i, &t) in s.timesteps.iter().enumerate() {
            approx(t, DEFAULT_MAX_TIMESTEP * (1.0 - i as f32 / 4.0));
        }
        assert!(s.timesteps.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn single_step_flag() {
        assert!(ScmScheduler::new(1).is_single_step());
        assert!(!ScmScheduler::new(2).is_single_step());
    }

    #[test]
    fn scm_timestep_endpoints() {
        let s = ScmScheduler::new(4);
        // t = pi/2 -> sin=1, cos=0 -> scm = 1.
        approx(s.scm_timestep(0), 1.0);
        // t = 0 -> sin=0 -> scm = 0.
        approx(s.scm_timestep(4), 0.0);
        // input_scale at scm=1 is sqrt(1+0)=1; at scm=0 is sqrt(0+1)=1.
        approx(s.input_scale(0), 1.0);
        approx(s.input_scale(4), 1.0);
    }
}
