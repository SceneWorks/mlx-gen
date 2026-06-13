//! Wan2.2 flow-match schedulers — port of `models/wan/scheduler.py`.
//!
//! Three rectified-flow solvers share one **shifted-sigma schedule** with **integer timesteps**
//! (`(σ·1000).trunc`): [`FlowMatchEuler`] (1st-order), [`FlowDpmpp2m`] (DPM-Solver++(2M)), and
//! [`FlowUniPC`] (the **default** — order-2 predictor-corrector). This is *not* the core mflux
//! `FlowMatchEuler` (`linspace(1,1/n,n)` + exponential time-shift); Wan builds
//! `linspace(σ_max, σ_min, n+1)[:-1]` from the unshifted training bounds and applies the algebraic
//! shift `σ' = shift·σ / (1 + (shift−1)·σ)` once, then appends a terminal `0.0`.
//!
//! Coefficient math runs in **f64** (the reference's Python-float / numpy path); only the final
//! array combinations run in the latent dtype (f32 for the 5B T2V/TI2V path). `_sigmas_float` in the
//! reference is `sigmas.tolist()` on a float32 array, i.e. f64 values that equal the f32 sigmas
//! exactly — mirrored here by storing f32 sigmas and widening to f64 for scalar math.

use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::Array;

use mlx_gen::{Error, Result};

/// Which flow-match solver to use. `UniPC` is the Wan2.2 default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverKind {
    Euler,
    Dpmpp2m,
    UniPC,
}

impl SolverKind {
    /// Resolve a sampler name (CLI/string) to a solver; unknown names fall back to UniPC, matching
    /// `generate_wan.py`'s default.
    pub fn from_name(name: &str) -> SolverKind {
        match name.to_ascii_lowercase().as_str() {
            "euler" => SolverKind::Euler,
            "dpm++" | "dpmpp" | "dpmpp2m" | "dpm++2m" => SolverKind::Dpmpp2m,
            _ => SolverKind::UniPC,
        }
    }
}

/// The common scheduler interface used by the denoise loop.
pub trait WanScheduler {
    /// Build the schedule for `num_steps` with time-`shift`.
    fn set_timesteps(&mut self, num_steps: usize, shift: f32);
    /// Sigmas, length `num_steps + 1` (trailing `0.0`).
    fn sigmas(&self) -> &[f32];
    /// Integer-valued model timesteps, length `num_steps` (`(σ·num_train).trunc`).
    fn timesteps(&self) -> &[f32];
    /// One denoise step; advances the internal step index. `model_output` is the velocity `v`.
    fn step(&mut self, model_output: &Array, sample: &Array) -> Result<Array>;
    /// Reset to step 0 (clears multistep history).
    fn reset(&mut self);
}

/// Guard the per-step `sigmas_f64[step_index(+1)]` indexing every `step` impl performs (F-019).
/// `WanScheduler` is public, so a caller that miscounts steps — over-steps past `num_steps`, or steps
/// before `set_timesteps` builds the schedule — must get a typed `Error`, not an out-of-bounds panic
/// (and, for UniPC, a `num_steps - i` underflow). A built schedule has `num_steps + 1` sigmas
/// (trailing terminal `0.0`), so a valid step needs `step_index + 1 < sigmas.len()`.
fn ensure_step_in_range(step_index: usize, sigmas_len: usize) -> Result<()> {
    if step_index + 1 >= sigmas_len {
        return Err(Error::Msg(format!(
            "WanScheduler::step over-stepped: step_index {step_index} but the schedule has {} \
             step(s) ({sigmas_len} sigmas) — call set_timesteps and run exactly that many steps",
            sigmas_len.saturating_sub(1)
        )));
    }
    Ok(())
}

/// Construct a boxed scheduler for `kind`.
pub fn make_scheduler(kind: SolverKind, num_train_timesteps: usize) -> Box<dyn WanScheduler> {
    match kind {
        SolverKind::Euler => Box::new(FlowMatchEuler::new(num_train_timesteps)),
        SolverKind::Dpmpp2m => Box::new(FlowDpmpp2m::new(num_train_timesteps)),
        SolverKind::UniPC => Box::new(FlowUniPC::new(num_train_timesteps, 2)),
    }
}

/// Shifted-sigma schedule (`_compute_sigmas`): `num_steps + 1` f32 sigmas, trailing `0.0`.
///
/// Computed in f64 (numpy path) and cast to f32 at the end. `σ_max = (N−1)/N`, `σ_min = 0`, where
/// `N = num_train_timesteps`; interpolate `linspace(σ_max, σ_min, n+1)[:-1]`, apply the shift once,
/// then append `0.0`.
pub fn compute_sigmas(num_steps: usize, shift: f32, num_train_timesteps: usize) -> Vec<f32> {
    let n_train = num_train_timesteps as f64;
    let sigma_max = 1.0 - 1.0 / n_train; // (N-1)/N
    let sigma_min = 0.0_f64;
    let shift = shift as f64;
    let n = num_steps as f64;
    // numpy linspace(σ_max, σ_min, num_steps+1) with endpoint=True evaluates `start + step·k`,
    // step = (stop−start)/(num−1) = (σ_min−σ_max)/num_steps; `[:-1]` keeps k = 0..num_steps-1.
    let step = (sigma_min - sigma_max) / n;

    let mut out = Vec::with_capacity(num_steps + 1);
    for k in 0..num_steps {
        let s = sigma_max + step * (k as f64);
        let shifted = shift * s / (1.0 + (shift - 1.0) * s);
        out.push(shifted as f32);
    }
    out.push(0.0_f32);
    out
}

/// Integer-valued model timesteps from sigmas: `(σ[:-1] · num_train).trunc` (int64 cast truncates
/// toward zero), back to f32.
fn integer_timesteps(sigmas: &[f32], num_train_timesteps: usize) -> Vec<f32> {
    let nt = num_train_timesteps as f32;
    sigmas[..sigmas.len() - 1]
        .iter()
        .map(|&s| (s * nt).trunc())
        .collect()
}

/// Multiply an array by an f64 scalar, rounded to the array's (f32) latent dtype — mirrors the
/// reference's `python_float * mx.array(f32)` (the weak scalar adopts the array dtype).
fn fscale(a: &Array, c: f64) -> Result<Array> {
    Ok(multiply(a, Array::from_f32(c as f32))?)
}

// =====================================================================================
// Euler
// =====================================================================================

/// 1st-order Euler flow-match scheduler: `x_next = x + (σ_next − σ_cur)·v`.
pub struct FlowMatchEuler {
    num_train_timesteps: usize,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    sigmas_f64: Vec<f64>,
    step_index: usize,
}

impl FlowMatchEuler {
    pub fn new(num_train_timesteps: usize) -> Self {
        Self {
            num_train_timesteps,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            sigmas_f64: Vec::new(),
            step_index: 0,
        }
    }
}

impl WanScheduler for FlowMatchEuler {
    fn set_timesteps(&mut self, num_steps: usize, shift: f32) {
        self.sigmas = compute_sigmas(num_steps, shift, self.num_train_timesteps);
        self.timesteps = integer_timesteps(&self.sigmas, self.num_train_timesteps);
        self.sigmas_f64 = self.sigmas.iter().map(|&s| s as f64).collect();
        self.step_index = 0;
    }
    fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }
    fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }
    fn step(&mut self, model_output: &Array, sample: &Array) -> Result<Array> {
        let i = self.step_index;
        ensure_step_in_range(i, self.sigmas_f64.len())?;
        let dt = self.sigmas_f64[i + 1] - self.sigmas_f64[i];
        let x_next = add(sample, &fscale(model_output, dt)?)?;
        self.step_index += 1;
        Ok(x_next)
    }
    fn reset(&mut self) {
        self.step_index = 0;
    }
}

// =====================================================================================
// DPM-Solver++(2M)
// =====================================================================================

/// log-SNR `λ(σ) = log((1−σ)/σ)`, with ±∞ at the σ=1 / σ=0 boundaries (matching `torch.log`).
fn log_snr(sigma: f64) -> f64 {
    if sigma >= 1.0 {
        return f64::NEG_INFINITY;
    }
    if sigma <= 0.0 {
        return f64::INFINITY;
    }
    ((1.0 - sigma) / sigma).ln()
}

/// DPM-Solver++(2M): 2nd-order multistep, falls back to 1st order on the first/last step.
pub struct FlowDpmpp2m {
    num_train_timesteps: usize,
    lower_order_final: bool,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    sigmas_f64: Vec<f64>,
    step_index: usize,
    num_steps: usize,
    prev_x0: Option<Array>,
}

impl FlowDpmpp2m {
    pub fn new(num_train_timesteps: usize) -> Self {
        Self {
            num_train_timesteps,
            lower_order_final: true,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            sigmas_f64: Vec::new(),
            step_index: 0,
            num_steps: 0,
            prev_x0: None,
        }
    }
}

impl WanScheduler for FlowDpmpp2m {
    fn set_timesteps(&mut self, num_steps: usize, shift: f32) {
        self.sigmas = compute_sigmas(num_steps, shift, self.num_train_timesteps);
        self.timesteps = integer_timesteps(&self.sigmas, self.num_train_timesteps);
        self.sigmas_f64 = self.sigmas.iter().map(|&s| s as f64).collect();
        self.step_index = 0;
        self.num_steps = num_steps;
        self.prev_x0 = None;
    }
    fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }
    fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }
    fn step(&mut self, model_output: &Array, sample: &Array) -> Result<Array> {
        let i = self.step_index;
        ensure_step_in_range(i, self.sigmas_f64.len())?;
        let s = &self.sigmas_f64;
        let sigma_cur = s[i];
        let sigma_next = s[i + 1];

        // x0 = sample − σ_cur · v
        let x0 = subtract(sample, &fscale(model_output, sigma_cur)?)?;

        let use_first_order = self.prev_x0.is_none()
            || (self.lower_order_final && i == self.num_steps - 1 && self.num_steps < 15);

        let x_next = if use_first_order || sigma_next == 0.0 {
            if sigma_next == 0.0 {
                x0.clone()
            } else {
                let lambda_cur = log_snr(sigma_cur);
                let lambda_next = log_snr(sigma_next);
                let h = lambda_next - lambda_cur;
                let alpha_next = 1.0 - sigma_next;
                let coeff_x = sigma_next / sigma_cur;
                let coeff_x0 = alpha_next * (-h).exp_m1();
                subtract(&fscale(sample, coeff_x)?, &fscale(&x0, coeff_x0)?)?
            }
        } else {
            let sigma_prev = s[i - 1];
            let lambda_prev = log_snr(sigma_prev);
            let lambda_cur = log_snr(sigma_cur);
            let lambda_next = log_snr(sigma_next);
            let h = lambda_next - lambda_cur;
            let h_0 = lambda_cur - lambda_prev;
            let r0 = h_0 / h;
            let alpha_next = 1.0 - sigma_next;
            let exp_neg_h_m1 = (-h).exp_m1(); // expm1(-h)

            // D0 = x0 ; D1 = (1/r0)·(x0 − prev_x0)
            let prev_x0 = self.prev_x0.as_ref().expect("2nd order requires history");
            let d1 = fscale(&subtract(&x0, prev_x0)?, 1.0 / r0)?;
            // x_next = (σ_next/σ_cur)·sample − (α·e)·D0 − 0.5·(α·e)·D1
            let term0 = fscale(sample, sigma_next / sigma_cur)?;
            let term1 = fscale(&x0, alpha_next * exp_neg_h_m1)?;
            let term2 = fscale(&d1, 0.5 * alpha_next * exp_neg_h_m1)?;
            subtract(&subtract(&term0, &term1)?, &term2)?
        };

        self.prev_x0 = Some(x0);
        self.step_index += 1;
        Ok(x_next)
    }
    fn reset(&mut self) {
        self.step_index = 0;
        self.prev_x0 = None;
    }
}

// =====================================================================================
// UniPC (default)
// =====================================================================================

/// UniPC predictor-corrector (B(h)=expm1(−h) / bh2 variant), `solver_order` (=2 for all Wan
/// configs). The order-2 corrector solves a 2×2 system via [`solve_linear`] (the reference's
/// `np.linalg.solve`), assembled with the f64 factorial / `h_phi` recurrence.
pub struct FlowUniPC {
    num_train_timesteps: usize,
    solver_order: usize,
    lower_order_final: bool,
    use_corrector: bool,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    sigmas_f64: Vec<f64>,
    step_index: usize,
    num_steps: usize,
    lower_order_nums: usize,
    model_outputs: Vec<Option<Array>>, // x0 history, newest-last
    last_sample: Option<Array>,
    this_order: usize,
}

impl FlowUniPC {
    pub fn new(num_train_timesteps: usize, solver_order: usize) -> Self {
        Self {
            num_train_timesteps,
            solver_order,
            lower_order_final: true,
            use_corrector: true,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            sigmas_f64: Vec::new(),
            step_index: 0,
            num_steps: 0,
            lower_order_nums: 0,
            model_outputs: vec![None; solver_order],
            last_sample: None,
            this_order: 1,
        }
    }

    /// Predictor `multistep_uni_p_bh_update`: predict `x_next` from x0 history. Returns the base
    /// prediction plus the order-≥2 correction (analytic ρ=[0.5] when effective_order ≤ 2).
    fn uni_p_bh2(&self, x0: &Array, sample: &Array, order: usize) -> Result<Array> {
        let i = self.step_index;
        let s = &self.sigmas_f64;
        let sigma_s0 = s[i];
        let sigma_t = s[i + 1];
        if sigma_t == 0.0 {
            return Ok(x0.clone());
        }
        let lambda_s0 = log_snr(sigma_s0);
        let lambda_t = log_snr(sigma_t);
        let h = lambda_t - lambda_s0;
        let hh = -h;
        let alpha_t = 1.0 - sigma_t;
        let h_phi_1 = hh.exp_m1();
        let b_h = h_phi_1;

        let m0 = self.model_outputs[self.solver_order - 1]
            .as_ref()
            .expect("history newest must be set before predict");
        // Base: (σ_t/σ_s0)·sample − (α_t·h_phi_1)·m0
        let mut x_t = subtract(
            &fscale(sample, sigma_t / sigma_s0)?,
            &fscale(m0, alpha_t * h_phi_1)?,
        )?;

        if order >= 2 {
            let mut rks: Vec<f64> = Vec::new();
            let mut d1s: Vec<Array> = Vec::new();
            for k in 1..order {
                let si_idx = i as isize - k as isize;
                let hist = self
                    .model_outputs
                    .get(self.solver_order.wrapping_sub(k + 1));
                let mk = match (si_idx >= 0, hist) {
                    (true, Some(Some(mk))) => mk,
                    _ => break,
                };
                let sigma_sk = s[si_idx as usize];
                let lambda_sk = log_snr(sigma_sk);
                let rk = (lambda_sk - lambda_s0) / h;
                if rk.is_infinite() {
                    break;
                }
                rks.push(rk);
                d1s.push(fscale(&subtract(mk, m0)?, 1.0 / rk)?);
            }

            if !d1s.is_empty() {
                let effective_order = d1s.len() + 1;
                let rhos_p: Vec<f64> = if effective_order <= 2 {
                    vec![0.5]
                } else {
                    let (r, b) = build_rb(&rks, h_phi_1, hh, b_h, 1, effective_order)?;
                    solve_linear(&r, &b)?
                };
                // pred_res = Σ ρ·D1 ; x_t -= (α_t·B_h)·pred_res. (rhos_p.len() == d1s.len().)
                let mut pred_res = fscale(&d1s[0], rhos_p[0])?;
                for (rho, d1) in rhos_p.iter().zip(d1s.iter()).skip(1) {
                    pred_res = add(&pred_res, &fscale(d1, *rho)?)?;
                }
                x_t = subtract(&x_t, &fscale(&pred_res, alpha_t * b_h)?)?;
            }
        }
        Ok(x_t)
    }

    /// Corrector `multistep_uni_c_bh_update`: refine the current sample using the freshly-computed
    /// model output. effective_order==1 → ρ=[0.5]; else solve the (effective_order)² system.
    fn uni_c_bh2(
        &self,
        model_x0: &Array,
        last_sample: &Array,
        this_sample: &Array,
        order: usize,
    ) -> Result<Array> {
        let i = self.step_index;
        let s = &self.sigmas_f64;
        let sigma_s0 = s[i - 1];
        let sigma_t = s[i];
        if sigma_t == 0.0 {
            return Ok(this_sample.clone());
        }
        let lambda_s0 = log_snr(sigma_s0);
        let lambda_t = log_snr(sigma_t);
        let h = lambda_t - lambda_s0;
        let hh = -h;
        let alpha_t = 1.0 - sigma_t;
        let h_phi_1 = hh.exp_m1();
        let b_h = h_phi_1;

        let m0 = self.model_outputs[self.solver_order - 1]
            .as_ref()
            .expect("history newest must be set before correct");
        // Re-derive base from last_sample.
        let x_t_ = subtract(
            &fscale(last_sample, sigma_t / sigma_s0)?,
            &fscale(m0, alpha_t * h_phi_1)?,
        )?;
        let d1_t = subtract(model_x0, m0)?;

        let mut rks: Vec<f64> = Vec::new();
        let mut d1s: Vec<Array> = Vec::new();
        for k in 1..order {
            let si_idx = i as isize - (k as isize + 1);
            let hist = self
                .model_outputs
                .get(self.solver_order.wrapping_sub(k + 1));
            let mk = match (si_idx >= 0, hist) {
                (true, Some(Some(mk))) => mk,
                _ => break,
            };
            let sigma_sk = s[si_idx as usize];
            let lambda_sk = log_snr(sigma_sk);
            let rk = (lambda_sk - lambda_s0) / h;
            if rk.is_infinite() {
                break;
            }
            rks.push(rk);
            d1s.push(fscale(&subtract(mk, m0)?, 1.0 / rk)?);
        }
        rks.push(1.0);
        let effective_order = rks.len(); // = d1s.len() + 1

        let rhos_c: Vec<f64> = if effective_order == 1 {
            vec![0.5]
        } else {
            let (r, b) = build_rb(&rks, h_phi_1, hh, b_h, 1, effective_order + 1)?;
            solve_linear(&r, &b)?
        };

        // corr = Σ ρ_k·D1_k + ρ_last·D1_t ; x_t = x_t_ − (α_t·B_h)·corr
        let mut corr = fscale(&d1_t, *rhos_c.last().unwrap())?;
        for (k_idx, d1) in d1s.iter().enumerate() {
            corr = add(&corr, &fscale(d1, rhos_c[k_idx])?)?;
        }
        Ok(subtract(&x_t_, &fscale(&corr, alpha_t * b_h)?)?)
    }
}

impl WanScheduler for FlowUniPC {
    fn set_timesteps(&mut self, num_steps: usize, shift: f32) {
        self.sigmas = compute_sigmas(num_steps, shift, self.num_train_timesteps);
        self.timesteps = integer_timesteps(&self.sigmas, self.num_train_timesteps);
        self.sigmas_f64 = self.sigmas.iter().map(|&s| s as f64).collect();
        self.step_index = 0;
        self.num_steps = num_steps;
        self.lower_order_nums = 0;
        self.model_outputs = vec![None; self.solver_order];
        self.last_sample = None;
        self.this_order = 1;
    }
    fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }
    fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }
    fn step(&mut self, model_output: &Array, sample: &Array) -> Result<Array> {
        let i = self.step_index;
        ensure_step_in_range(i, self.sigmas_f64.len())?;
        // x0 = sample − σ[i]·v
        let x0 = subtract(sample, &fscale(model_output, self.sigmas_f64[i])?)?;

        // 1. Corrector (uses last step's `this_order`).
        let mut sample_cur = sample.clone();
        let use_corrector = self.use_corrector && i > 0 && self.last_sample.is_some();
        if use_corrector {
            let last = self.last_sample.as_ref().unwrap().clone();
            sample_cur = self.uni_c_bh2(&x0, &last, &sample_cur, self.this_order)?;
        }

        // 2. Shift x0 history (newest-last), append current x0.
        for k in 0..self.solver_order - 1 {
            self.model_outputs[k] = self.model_outputs[k + 1].clone();
        }
        let last_idx = self.solver_order - 1;
        self.model_outputs[last_idx] = Some(x0.clone());

        // 3. Prediction order.
        let this_order = if self.lower_order_final {
            self.solver_order.min(self.num_steps - i)
        } else {
            self.solver_order
        };
        self.this_order = this_order.min(self.lower_order_nums + 1);

        // 4. Predict next.
        self.last_sample = Some(sample_cur.clone());
        let x_next = self.uni_p_bh2(&x0, &sample_cur, self.this_order)?;

        if self.lower_order_nums < self.solver_order {
            self.lower_order_nums += 1;
        }
        self.step_index += 1;
        Ok(x_next)
    }
    fn reset(&mut self) {
        self.step_index = 0;
        self.lower_order_nums = 0;
        self.model_outputs = vec![None; self.solver_order];
        self.last_sample = None;
        self.this_order = 1;
    }
}

/// Build the (R, b) system for the UniPC ρ solve. `rks` are the rk values (Vandermonde base);
/// rows `j = 1..j_end` give `R_row = rks.^(j−1)` and `b = h_phi_k·factorial_i / B_h`, advancing the
/// factorial / `h_phi_k` recurrence in f64. The predictor passes `j_end = effective_order`; the
/// corrector passes `j_end = effective_order + 1`.
fn build_rb(
    rks: &[f64],
    h_phi_1: f64,
    hh: f64,
    b_h: f64,
    j_start: usize,
    j_end: usize,
) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    // `hh = λ_{s0} − λ_t`; equal consecutive sigmas (a degenerate/single-step schedule) make it 0,
    // so `h_phi_1 / hh` and the `h_phi_k / hh` recurrence below become ±Inf/NaN and would silently
    // corrupt the predictor coefficients (solve_linear's near-zero-pivot guard only fires *after*
    // NaN has entered the matrix). Reject the degenerate step here instead (F-008).
    if hh.abs() < f64::EPSILON {
        return Err(Error::Msg(format!(
            "wan scheduler: degenerate schedule (equal consecutive sigmas, hh={hh:.3e}) in build_rb"
        )));
    }
    let mut h_phi_k = h_phi_1 / hh - 1.0;
    let mut factorial_i = 1.0_f64;
    let mut r_rows: Vec<Vec<f64>> = Vec::new();
    let mut b_vals: Vec<f64> = Vec::new();
    for j in j_start..j_end {
        r_rows.push(rks.iter().map(|&rk| rk.powi((j - 1) as i32)).collect());
        b_vals.push(h_phi_k * factorial_i / b_h);
        factorial_i *= (j + 1) as f64;
        h_phi_k = h_phi_k / hh - 1.0 / factorial_i;
    }
    Ok((r_rows, b_vals))
}

/// Solve `R·x = b` for a small dense square system via Gaussian elimination with partial pivoting
/// (f64) — the host-side stand-in for `np.linalg.solve` (LAPACK LU). For the order-2 systems Wan
/// actually builds this matches LAPACK to f64 round-off. A (near-)singular system returns `Err`
/// rather than silently propagating NaN/Inf into the latents, mirroring how `np.linalg.solve`
/// raises `LinAlgError` (F-020); reachable only via a custom `solver_order > 2` with degenerate `rks`.
// Index loops read clearer than iterator adapters for a matrix solve.
#[allow(clippy::needless_range_loop)]
fn solve_linear(r: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
    let n = b.len();
    let mut a: Vec<Vec<f64>> = r.to_vec();
    let mut x = b.to_vec();
    for col in 0..n {
        // Partial pivot.
        let mut piv = col;
        let mut best = a[col][col].abs();
        for row in (col + 1)..n {
            let v = a[row][col].abs();
            if v > best {
                best = v;
                piv = row;
            }
        }
        a.swap(col, piv);
        x.swap(col, piv);
        let diag = a[col][col];
        // The largest-magnitude pivot in this column is ~0 ⇒ the system is (near-)singular; dividing
        // by it (here and in back-substitution) would yield NaN/Inf that silently corrupts the
        // trajectory. Error instead, like `np.linalg.solve` (F-020).
        if diag.abs() < 1e-12 {
            return Err(Error::Msg(format!(
                "wan scheduler: singular {n}x{n} system in solve_linear (near-zero pivot \
                 {diag:.3e} at column {col}) — degenerate rks for the configured solver_order"
            )));
        }
        for row in (col + 1)..n {
            let factor = a[row][col] / diag;
            if factor != 0.0 {
                for c in col..n {
                    a[row][c] -= factor * a[col][c];
                }
                x[row] -= factor * x[col];
            }
        }
    }
    // Back-substitution (each `a[col][col]` is a pivot already checked non-zero above).
    for col in (0..n).rev() {
        let mut sum = x[col];
        for c in (col + 1)..n {
            sum -= a[col][c] * x[c];
        }
        x[col] = sum / a[col][col];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_step_in_range_guards_over_and_under_run() {
        // F-019: a built schedule has num_steps+1 sigmas; a valid step needs step_index+1 < len.
        assert!(ensure_step_in_range(0, 0).is_err()); // empty (set_timesteps not called)
        assert!(ensure_step_in_range(0, 3).is_ok()); // 2-step schedule, first step
        assert!(ensure_step_in_range(1, 3).is_ok()); // last valid step
        assert!(ensure_step_in_range(2, 3).is_err()); // over-stepped
    }

    #[test]
    fn step_errors_instead_of_panicking_when_over_stepped() {
        // F-019: driving a scheduler past its step count returns a typed error, not an OOB panic.
        let mut sched = make_scheduler(SolverKind::Euler, 1000);
        sched.set_timesteps(2, 5.0);
        let x = Array::from_slice(&[0.5f32, -0.5], &[1, 2]);
        let v = Array::from_slice(&[0.1f32, 0.2], &[1, 2]);
        assert!(sched.step(&v, &x).is_ok()); // step 0
        assert!(sched.step(&v, &x).is_ok()); // step 1 (last)
        let err = sched.step(&v, &x).unwrap_err().to_string(); // step 2 → over-stepped
        assert!(err.contains("over-stepped"), "got: {err}");

        // Stepping before set_timesteps (empty schedule) also errors rather than panicking.
        let mut fresh = make_scheduler(SolverKind::Euler, 1000);
        assert!(fresh.step(&v, &x).is_err());
    }

    #[test]
    fn from_name_maps_advertised_samplers_and_defaults_to_unipc() {
        // F-021: the single sampler→solver mapping the production model entries now share. The
        // advertised set (validate_request) is unipc/euler/dpmpp2m; an unset/unknown name → UniPC.
        assert_eq!(SolverKind::from_name("euler"), SolverKind::Euler);
        assert_eq!(SolverKind::from_name("dpmpp2m"), SolverKind::Dpmpp2m);
        assert_eq!(SolverKind::from_name("dpm++"), SolverKind::Dpmpp2m);
        assert_eq!(SolverKind::from_name("unipc"), SolverKind::UniPC);
        assert_eq!(SolverKind::from_name(""), SolverKind::UniPC); // the unset-sampler sentinel
        assert_eq!(SolverKind::from_name("nope"), SolverKind::UniPC);
        // Case-insensitive (the model entries lower-case via the same path).
        assert_eq!(SolverKind::from_name("Euler"), SolverKind::Euler);
        assert_eq!(SolverKind::from_name("DPM++"), SolverKind::Dpmpp2m);
    }

    #[test]
    fn sigma_schedule_endpoints_and_shape() {
        let s = compute_sigmas(40, 5.0, 1000);
        assert_eq!(s.len(), 41);
        assert_eq!(*s.last().unwrap(), 0.0);
        // First sigma = shift·σ_max/(1+(shift−1)·σ_max), σ_max = 0.999, shift = 5.
        let sm = 0.999_f64;
        let want0 = (5.0 * sm / (1.0 + 4.0 * sm)) as f32;
        assert!((s[0] - want0).abs() < 1e-6, "s0 {} want {}", s[0], want0);
        // Strictly decreasing.
        assert!(s.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn integer_timesteps_truncate() {
        let s = compute_sigmas(40, 5.0, 1000);
        let t = integer_timesteps(&s, 1000);
        assert_eq!(t.len(), 40);
        // Each timestep is trunc(sigma*1000) and integer-valued.
        for (ts, sig) in t.iter().zip(s.iter()) {
            assert_eq!(*ts, (sig * 1000.0).trunc());
            assert_eq!(*ts, ts.trunc());
        }
        // First timestep with shift=5, σ0≈0.9998 → ~999.
        assert!(t[0] >= 995.0 && t[0] <= 999.0, "t0 = {}", t[0]);
    }

    #[test]
    fn euler_step_matches_formula() {
        let mut sched = FlowMatchEuler::new(1000);
        sched.set_timesteps(4, 5.0);
        let sample = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let v = Array::from_slice(&[0.5_f32, 0.5, 0.5], &[3]);
        let out = sched.step(&v, &sample).unwrap();
        let dt = sched.sigmas[1] - sched.sigmas[0];
        let got = out.as_slice::<f32>();
        assert!((got[0] - (1.0 + dt * 0.5)).abs() < 1e-6);
        assert!((got[2] - (3.0 + dt * 0.5)).abs() < 1e-6);
    }

    #[test]
    fn solve_linear_2x2() {
        // [[2,1],[1,3]] x = [3,5] → x = [0.8, 1.4].
        let r = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![3.0, 5.0];
        let x = solve_linear(&r, &b).unwrap();
        assert!((x[0] - 0.8).abs() < 1e-12);
        assert!((x[1] - 1.4).abs() < 1e-12);
    }

    #[test]
    fn solve_linear_singular_errs() {
        // F-020: a singular system (row 2 = 2 × row 1) errors instead of returning NaN/Inf.
        let r = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let b = vec![1.0, 2.0];
        match solve_linear(&r, &b) {
            Ok(x) => panic!("singular system must error, got {x:?}"),
            Err(e) => assert!(e.to_string().contains("singular"), "got: {e}"),
        }
    }

    #[test]
    fn unipc_runs_full_trajectory() {
        // Smoke: UniPC default order-2 over 8 steps with constant velocity does not panic and stays
        // finite (exercises corrector + 2×2 solve path).
        let mut sched = FlowUniPC::new(1000, 2);
        sched.set_timesteps(8, 5.0);
        let mut x = Array::from_slice(&[0.3_f32; 16], &[16]);
        for _ in 0..8 {
            let v = multiply(&x, Array::from_f32(0.1)).unwrap();
            x = sched.step(&v, &x).unwrap();
        }
        let xs = x.as_slice::<f32>();
        assert!(xs.iter().all(|v| v.is_finite()));
    }
}
