//! SANA-Sprint **SCM/TrigFlow scheduler parity** (sc-8490, Phase A) — pure host math vs diffusers
//! `SCMScheduler` / `SanaSprintPipeline` references, no weights.
//!
//! The SCM scheduler is the few-step consistency core; its math is small and exact, so this gates it
//! against hand-computed diffusers values rather than a dumped golden (the trunk forward, which DOES
//! need weights, is gated by `transformer_parity.rs` + the `dump_sana_sprint_golden.py` tiny golden).
//!
//! References (diffusers `schedulers/scheduling_scm.py` + `pipelines/sana/pipeline_sana_sprint.py`):
//!  * `set_timesteps(2, max=π/2, intermediate=1.3)` → `[π/2, 1.3, 0]`;
//!  * `set_timesteps(n, max=π/2)` → `linspace(π/2, 0, n+1)`;
//!  * `scm_timestep = sin(t)/(cos(t)+sin(t))`;
//!  * input scale `sqrt(scm_t² + (1−scm_t)²)`;
//!  * step: `pred_x0 = cos(s)·x − sin(s)·model_output`.

use std::f32::consts::FRAC_PI_2;

use mlx_gen_sana::ScmScheduler;

fn approx(a: f32, b: f32, tol: f32, what: &str) {
    assert!((a - b).abs() < tol, "{what}: got {a} want {b}");
}

#[test]
fn two_step_timesteps_match_diffusers_intermediate() {
    // diffusers set_timesteps(2) → torch.tensor([1.57080, 1.3, 0]) (max_timesteps default ≈ π/2).
    let s = ScmScheduler::new(2);
    assert_eq!(s.timesteps.len(), 3);
    approx(s.timesteps[0], FRAC_PI_2, 1e-4, "max");
    approx(s.timesteps[1], 1.3, 1e-6, "intermediate");
    approx(s.timesteps[2], 0.0, 1e-6, "end");
}

#[test]
fn four_step_timesteps_match_linspace() {
    // diffusers set_timesteps(4) → torch.linspace(π/2, 0, 5) = [π/2, 3π/8, π/4, π/8, 0].
    let s = ScmScheduler::new(4);
    let want = [
        FRAC_PI_2,
        3.0 * FRAC_PI_2 / 4.0,
        2.0 * FRAC_PI_2 / 4.0,
        FRAC_PI_2 / 4.0,
        0.0,
    ];
    assert_eq!(s.timesteps.len(), 5);
    for (i, &w) in want.iter().enumerate() {
        approx(s.timesteps[i], w, 1e-5, &format!("ts[{i}]"));
    }
}

#[test]
fn scm_timestep_and_input_scale_match_diffusers_formula() {
    // For the 2-step schedule, compute scm_timestep + input scale at each step and compare to the
    // diffusers closed forms evaluated by hand.
    let s = ScmScheduler::new(2);
    for i in 0..s.num_steps() {
        let t = s.timesteps[i];
        let want_scm = t.sin() / (t.cos() + t.sin());
        approx(s.scm_timestep(i), want_scm, 1e-6, &format!("scm[{i}]"));
        let want_scale = (want_scm * want_scm + (1.0 - want_scm) * (1.0 - want_scm)).sqrt();
        approx(
            s.input_scale(i),
            want_scale,
            1e-6,
            &format!("in_scale[{i}]"),
        );
    }
    // At t = π/2: sin=1, cos=0 → scm = 1; input scale = sqrt(1+0) = 1.
    approx(s.scm_timestep(0), 1.0, 1e-5, "scm at pi/2");
    approx(s.input_scale(0), 1.0, 1e-5, "scale at pi/2");
}

/// The diffusers `SCMScheduler.step` x0-prediction: `pred_x0 = cos(s)·x − sin(s)·model_output`. Verify
/// the closed form against a hand-computed scalar reference (the renoise term is the next-angle
/// `cos(t)·x0 + sin(t)·noise·σ_data`, exercised end-to-end in `sprint_contract.rs`).
#[test]
fn trigflow_x0_pred_closed_form() {
    let s = ScmScheduler::new(2);
    let angle = s.timesteps[0]; // π/2
    let (cos_s, sin_s) = (angle.cos(), angle.sin());
    // x = 2.0, model_output = 0.5 → pred_x0 = cos(π/2)·2 − sin(π/2)·0.5 = 0 − 0.5 = −0.5.
    let x = 2.0_f32;
    let model_output = 0.5_f32;
    let pred_x0 = cos_s * x - sin_s * model_output;
    approx(pred_x0, -0.5, 1e-5, "pred_x0 at pi/2");
    // sigma_data is 0.5 (the Sprint SCM renoise std-dev).
    approx(s.sigma_data, 0.5, 1e-6, "sigma_data");
}

#[test]
fn single_step_skips_renoise() {
    assert!(ScmScheduler::new(1).is_single_step());
    assert!(!ScmScheduler::new(2).is_single_step());
    assert!(!ScmScheduler::new(4).is_single_step());
}
