//! Backend-neutral sampler **policy**: schedule construction (sigma/alpha tables) and the per-step
//! affine **coefficients** every diffusion sampler reduces to. Zero tensor deps — gen-core's only
//! numeric types here are `f32`/`f64`/`Vec<f32>`/`Vec<usize>`.
//!
//! Every sampler step in the mlx-gen (and forthcoming candle-gen) stack is the same affine update:
//!
//! ```text
//! x_in   = cast(c_in · x, model_dtype)                  (model-input scaling)
//! x_next = a_x·x + a_out·model_output + a_noise·ε        (ε ~ N(0,1), fresh per step)
//! ```
//!
//! so the only thing that differs between LCM / SDXL-Lightning / Hyper-SD (TCD) / flow-match is the
//! *schedule* and the *scalar coefficients*. Those live here; each backend writes a ~5-line tensor
//! `apply_step` that consumes a [`StepCoeffs`]. See epic 3720 §3 (the sampler policy / application
//! split, D5). The mlx-gen sampler types (`LcmSampler`/`LightningSampler`/`TcdSampler`/
//! `FlowMatchSampler`/`FlowMatchEuler`) are thin wrappers holding one of these policies, so the
//! family-crate call sites are unchanged.
//!
//! Reference: faithful ports of the **diffusers** schedulers each acceleration method trains against
//! (`LCMScheduler`, `EulerDiscreteScheduler(timestep_spacing="trailing")`, `TCDScheduler`,
//! `FlowMatchEulerDiscreteScheduler`). Intermediates are computed in `f64` and emitted as `f32` to
//! match the original mlx-gen code; the FlowMatch `a_out` is an `f32` subtraction of `f32` sigmas so
//! the backend's byte-parity rule reproduces `flow_match_euler_step` exactly (F-009 / FLUX goldens).

use crate::Result;

/// The scalar coefficients of one denoise step. The backend applies
/// `x_next = a_x·x + a_out·model_output + a_noise·ε`, scaling the model input by `c_in` first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepCoeffs {
    /// The conditioning timestep fed to the model at this step (the value the network embeds).
    pub timestep: f32,
    /// Model-input scale (1.0 = identity). Applied before casting to the model dtype.
    pub c_in: f32,
    /// Coefficient on the current latents `x`.
    pub a_x: f32,
    /// Coefficient on the (already CFG-combined) model output.
    pub a_out: f32,
    /// Coefficient on fresh unit-normal noise. `0.0` = deterministic step (no noise drawn).
    pub a_noise: f32,
}

/// The precision in which the backend computes the step update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepDtype {
    /// Upcast `x` and `model_output` to f32 for the update (the DDPM samplers; diffusers parity).
    F32,
    /// Compute in the latents' dtype, no upcast (flow-match; byte-parity with the original
    /// `flow_match_euler_step` is REQUIRED).
    Latents,
}

/// A swappable denoise schedule, reduced to backend-neutral policy. The backend drives, per step `i`:
/// `x_in = cast(coeffs(i).c_in · x, dt)` → `out = model(x_in, coeffs(i).timestep)` → (CFG) →
/// `x = a_x·x + a_out·out + a_noise·ε`. The starting latents are `unit_noise · init_noise_scale()`.
pub trait SamplerPolicy {
    /// Number of denoise iterations (loop count).
    fn num_steps(&self) -> usize;
    /// Multiply unit-normal starting noise by this (txt2img prior). 1.0 for LCM/TCD/flow-match;
    /// max-sigma for Lightning.
    fn init_noise_scale(&self) -> f32;
    /// The precision the backend should compute the step update in.
    fn step_dtype(&self) -> StepDtype;
    /// The affine coefficients for step `i` (0-based, `i < num_steps()`).
    fn coeffs(&self, i: usize) -> StepCoeffs;
    /// The schedule sigma at node `i` (length `num_steps()+1` semantics). Used by flow-match
    /// img2img start-step blending. `None` for samplers without a sigma-schedule semantic.
    fn sigma(&self, i: usize) -> Option<f32>;
}

// =================================================================================================
// Schedule builders
// =================================================================================================

/// A discrete DDPM noise schedule: the `alphas_cumprod` table built from `scaled_linear` betas,
/// shared by the diffusers-derived acceleration samplers. Mirrors diffusers'
/// `betas = linspace(√β₀, √β₁, N)²; alphas_cumprod = cumprod(1 - betas)` (torch float32).
#[derive(Clone)]
pub struct AlphaSchedule {
    /// `alphas_cumprod[t]`, length `num_train_timesteps`.
    pub alphas_cumprod: Vec<f32>,
}

impl AlphaSchedule {
    /// Build from `scaled_linear` betas (SDXL: `β₀=0.00085`, `β₁=0.012`, `N=1000`). The cumprod is a
    /// **sequential host f32 accumulation** (`acc *= 1 - beta`, f32 each step) — torch CPU `cumprod`
    /// semantics — which keeps the table tensor-free here while matching the reference to f32 (the
    /// original mlx-gen ran the same product through `Array::cumprod`).
    pub fn scaled_linear(
        num_train_timesteps: usize,
        beta_start: f32,
        beta_end: f32,
    ) -> Result<Self> {
        let n = num_train_timesteps;
        // betas = linspace(√β₀, √β₁, N)²  (the √ endpoints taken in f64 like diffusers' Python).
        let (a, b) = ((beta_start as f64).sqrt(), (beta_end as f64).sqrt());
        let mut alphas_cumprod = Vec::with_capacity(n);
        let mut acc = 1.0f32;
        for i in 0..n {
            let t = if n <= 1 {
                0.0
            } else {
                i as f32 / (n - 1) as f32
            };
            let v = a + (b - a) * t as f64;
            let beta = (v * v) as f32;
            acc *= 1.0 - beta;
            alphas_cumprod.push(acc);
        }
        Ok(Self { alphas_cumprod })
    }

    /// `alphas_cumprod[t]` as f64.
    pub(crate) fn acp(&self, t: usize) -> f64 {
        self.alphas_cumprod[t] as f64
    }

    /// The per-train-step Karras-style sigma `√((1-ᾱ)/ᾱ)` at integer index `t` (diffusers Euler).
    pub(crate) fn sigma_at(&self, t: usize) -> f64 {
        let acp = self.acp(t);
        ((1.0 - acp) / acp).sqrt()
    }
}

/// Select the LCM/TCD inference timesteps (the shared diffusers logic): take `original_steps`
/// linearly-spaced training timesteps `arange(1, original_steps+1)·k − 1` (`k = N/original_steps`),
/// reverse, then pick `num_steps` evenly-spaced-by-index entries. For SDXL `N=1000`,
/// `original_steps=50` → e.g. 4 steps = `[999, 759, 499, 259]`.
pub fn lcm_style_timesteps(
    num_train_timesteps: usize,
    original_steps: usize,
    num_steps: usize,
) -> Vec<usize> {
    // Clamp caller-supplied counts so a 0 can't divide-by-zero (`num_train/original_steps`) or
    // underflow `reversed.len()-1` on an empty table. Mirrors `build_flow_sigmas`'s clamp; the real
    // floor is `validate_request` enforcing steps>=1 upstream (F-037).
    let original_steps = original_steps.max(1);
    let num_steps = num_steps.max(1);
    let k = num_train_timesteps / original_steps;
    // lcm_origin_timesteps = arange(1, original_steps+1)·k − 1, then reversed (descending).
    let origin: Vec<i64> = (1..=original_steps as i64)
        .map(|i| i * k as i64 - 1)
        .collect();
    let reversed: Vec<i64> = origin.into_iter().rev().collect();
    // inference_indices = floor(linspace(0, len, num_steps, endpoint=False)).
    let len = reversed.len() as f64;
    (0..num_steps)
        .map(|j| {
            let idx = (len * j as f64 / num_steps as f64).floor() as usize;
            reversed[idx.min(reversed.len() - 1)] as usize
        })
        .collect()
}

/// Latent sequence length used for the flow-match empirical `mu` fit: `(height/16) * (width/16)`.
/// Each dim is floored to `/16` before the multiply (matching the fork). Callers validate the
/// resolution upstream so a sub-16 dim never reaches the `mu` fit (F-089), but the `.max(1)` per
/// floored dim guards a direct caller from a degenerate 0-length sequence (which would make
/// [`compute_mu`] fit on `seq_len = 0`). For any valid (`>= 16`) dim the floor is `>= 1`, so the
/// `.max(1)` is a no-op and the result is byte-identical (L-E).
pub fn image_seq_len(width: u32, height: u32) -> usize {
    ((height / 16).max(1) * (width / 16).max(1)) as usize
}

/// Port of the fork's `_compute_empirical_mu`: a piecewise-linear fit of the time-shift `mu`
/// from the latent sequence length and step count.
//  Constants mirror the fork's Python float64 literals verbatim (8.73809524e-05 / 1.89833333 /
//  0.00016927 / 0.45666666) for parity auditing; f32 rounds the extra digits harmlessly.
#[allow(clippy::excessive_precision)]
pub fn compute_mu(image_seq_len: usize, num_steps: usize) -> f32 {
    let (a1, b1) = (8.738_095_24e-5_f32, 1.898_333_33_f32);
    let (a2, b2) = (0.000_169_27_f32, 0.456_666_66_f32);
    let seq = image_seq_len as f32;
    if image_seq_len > 4300 {
        return a2 * seq + b2;
    }
    let m_200 = a2 * seq + b2;
    let m_10 = a1 * seq + b1;
    let a = (m_200 - m_10) / 190.0;
    let b = m_200 - 200.0 * a;
    a * num_steps as f32 + b
}

/// `exp(mu) / (exp(mu) + (1/t - 1))` — the fork's `_time_shift_exponential_array` at
/// `sigma_power = 1`.
fn time_shift_exponential(mu: f32, t: f32) -> f32 {
    let e = mu.exp();
    e / (e + (1.0 / t - 1.0))
}

/// Build the flow-match sigma schedule (`linspace(1, 1/n, n)` run through the exponential time-shift,
/// with a trailing `0.0`). The fork's `LinearScheduler` / `FlowMatchEulerDiscreteScheduler`.
pub fn build_flow_sigmas(num_steps: usize, mu: f32) -> Vec<f32> {
    let n = num_steps.max(1);
    let (start, end) = (1.0_f32, 1.0_f32 / n as f32);
    let mut sigmas: Vec<f32> = (0..n)
        .map(|i| {
            // linspace(1.0, 1.0/n, n)
            let t = if n == 1 {
                start
            } else {
                start + (end - start) * (i as f32) / ((n - 1) as f32)
            };
            time_shift_exponential(mu, t)
        })
        .collect();
    sigmas.push(0.0);
    sigmas
}

// =================================================================================================
// Concrete policies
// =================================================================================================

/// How a flow-match family feeds the schedule sigma to the model as its conditioning timestep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimestepConvention {
    /// FLUX / Qwen / Chroma feed the raw sigma `σᵢ` (their time embedding scales it ×1000 inside).
    Sigma,
    /// The `FlowMatchEuler`-style DiT families (Z-Image) feed `1 − σᵢ`.
    OneMinusSigma,
}

/// Flow-match (rectified-flow) Euler policy: a forward Euler integration of the velocity field over a
/// precomputed sigma schedule, `x ← x + v·(σ_{i+1} − σ_i)`. Holds the sigmas (length `num_steps + 1`,
/// trailing `0.0`) and the timestep convention.
#[derive(Clone)]
pub struct FlowMatchPolicy {
    sigmas: Vec<f32>,
    conv: TimestepConvention,
}

impl FlowMatchPolicy {
    /// Build from a precomputed sigma schedule (length `num_steps + 1`, trailing `0.0`). Panics if
    /// fewer than two entries are supplied (a schedule needs at least one step + the terminal `0`).
    pub fn new(sigmas: Vec<f32>, conv: TimestepConvention) -> Self {
        // Internal invariant: the schedule helpers always build `sigmas` as `num_steps + 1` (>= 2);
        // a `debug_assert` documents it without changing this pub constructor's signature (request-
        // derived step counts are validated at the provider boundary, not here) (F-020/L-A).
        debug_assert!(
            sigmas.len() >= 2,
            "FlowMatchPolicy needs sigmas of length num_steps+1 (>= 2), got {}",
            sigmas.len()
        );
        Self { sigmas, conv }
    }

    /// The raw schedule sigma at node `i` (length `num_steps + 1`).
    pub fn sigma_at_node(&self, i: usize) -> f32 {
        self.sigmas[i]
    }
}

impl SamplerPolicy for FlowMatchPolicy {
    fn num_steps(&self) -> usize {
        self.sigmas.len() - 1
    }

    fn init_noise_scale(&self) -> f32 {
        1.0
    }

    fn step_dtype(&self) -> StepDtype {
        StepDtype::Latents
    }

    fn coeffs(&self, i: usize) -> StepCoeffs {
        let timestep = match self.conv {
            TimestepConvention::Sigma => self.sigmas[i],
            TimestepConvention::OneMinusSigma => 1.0 - self.sigmas[i],
        };
        StepCoeffs {
            timestep,
            c_in: 1.0,
            a_x: 1.0,
            // f32 subtraction of f32 sigmas — the exact `dt` of `flow_match_euler_step`, so the
            // backend's byte-parity branch reproduces it bit-for-bit (F-009 / FLUX goldens).
            a_out: self.sigmas[i + 1] - self.sigmas[i],
            a_noise: 0.0,
        }
    }

    fn sigma(&self, i: usize) -> Option<f32> {
        Some(self.sigmas[i])
    }
}

/// SDXL-Lightning policy: trailing-spaced Euler, ε-prediction, no churn, `final_sigmas_type="zero"`.
/// Port of diffusers `EulerDiscreteScheduler(timestep_spacing="trailing")`.
#[derive(Clone)]
pub struct LightningPolicy {
    /// Interpolated sigmas at the trailing timesteps, length `num_steps + 1` (trailing `0.0`).
    sigmas: Vec<f32>,
    /// The (float) trailing timesteps fed to the model, length `num_steps`.
    timesteps: Vec<f32>,
}

impl LightningPolicy {
    /// Build for `num_steps` (2/4/8). Timesteps are diffusers' trailing spacing
    /// `round(arange(N, 0, −N/num_steps)) − 1`; sigmas are `√((1-ᾱ)/ᾱ)` linearly interpolated at
    /// those (float) timesteps, with a trailing `0` (`final_sigmas_type="zero"`).
    pub fn new(sched: &AlphaSchedule, num_train_timesteps: usize, num_steps: usize) -> Self {
        // Guard /0 (F-037); the real floor is `validate_request` enforcing steps>=1 upstream.
        let num_steps = num_steps.max(1);
        let step_ratio = num_train_timesteps as f64 / num_steps as f64;
        // arange(N, 0, -step_ratio): N, N-step_ratio, … (num_steps entries), round, then −1.
        let timesteps: Vec<f32> = (0..num_steps)
            .map(|j| {
                let v = num_train_timesteps as f64 - step_ratio * j as f64;
                (v.round() - 1.0) as f32
            })
            .collect();
        // Full per-train-step sigma table for the linear interp.
        let full: Vec<f64> = (0..num_train_timesteps)
            .map(|t| sched.sigma_at(t))
            .collect();
        let interp = |t: f32| -> f32 {
            // np.interp over xp = arange(0, N), fp = full. t is in [0, N-1] here.
            let tt = (t as f64).clamp(0.0, (num_train_timesteps - 1) as f64);
            let lo = tt.floor() as usize;
            let hi = (lo + 1).min(num_train_timesteps - 1);
            let frac = tt - lo as f64;
            (full[lo] * (1.0 - frac) + full[hi] * frac) as f32
        };
        let mut sigmas: Vec<f32> = timesteps.iter().map(|&t| interp(t)).collect();
        sigmas.push(0.0); // final_sigmas_type = "zero"
        Self { sigmas, timesteps }
    }

    /// The largest sigma — `init_noise_sigma` for trailing spacing.
    pub fn init_noise_sigma(&self) -> f32 {
        self.sigmas.iter().copied().fold(0.0_f32, f32::max)
    }
}

impl SamplerPolicy for LightningPolicy {
    fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    fn init_noise_scale(&self) -> f32 {
        self.init_noise_sigma()
    }

    fn step_dtype(&self) -> StepDtype {
        StepDtype::F32
    }

    fn coeffs(&self, i: usize) -> StepCoeffs {
        // x / √(σ²+1) input scaling; Euler ε-pred step `x + eps·(σ_{i+1} − σ_i)`, gamma=0.
        let sigma = self.sigmas[i] as f64;
        StepCoeffs {
            timestep: self.timesteps[i],
            c_in: (1.0 / (sigma * sigma + 1.0).sqrt()) as f32,
            a_x: 1.0,
            a_out: self.sigmas[i + 1] - self.sigmas[i],
            a_noise: 0.0,
        }
    }

    fn sigma(&self, i: usize) -> Option<f32> {
        Some(self.sigmas[i])
    }
}

/// LCM policy: diffusers `LCMScheduler` (ε-prediction; SDXL world: scaled_linear betas,
/// timestep_scaling=10, sigma_data=0.5, `set_alpha_to_one=True` → final ᾱ = 1).
#[derive(Clone)]
pub struct LcmPolicy {
    sched: AlphaSchedule,
    timesteps: Vec<usize>,
    timestep_scaling: f32,
}

impl LcmPolicy {
    /// Build for `num_steps` inference steps. `original_inference_steps` is diffusers' default 50.
    pub fn new(
        sched: AlphaSchedule,
        num_train_timesteps: usize,
        original_inference_steps: usize,
        num_steps: usize,
    ) -> Self {
        Self {
            timesteps: lcm_style_timesteps(
                num_train_timesteps,
                original_inference_steps,
                num_steps,
            ),
            sched,
            timestep_scaling: 10.0,
        }
    }

    /// The consistency-prediction (`denoised`) coefficients at step `i` — diffusers' deterministic
    /// `denoised` before the between-step re-noise. `x_next = d_x·x + d_eps·eps`.
    pub fn denoised_coeffs(&self, i: usize) -> StepCoeffs {
        let t = self.timesteps[i];
        let apt = self.sched.acp(t);
        let bpt = 1.0 - apt;
        // Boundary-condition scalings (sigma_data=0.5 → sigma_data²=0.25).
        let scaled_t = t as f64 * self.timestep_scaling as f64;
        let c_skip = 0.25 / (scaled_t * scaled_t + 0.25);
        let c_out = scaled_t / (scaled_t * scaled_t + 0.25).sqrt();
        // denoised = c_out·pred_x0 + c_skip·x, pred_x0 = (x − √β̄·eps)/√ᾱ
        //          = (c_out/√ᾱ + c_skip)·x + (−c_out·√β̄/√ᾱ)·eps.
        let d_x = c_out / apt.sqrt() + c_skip;
        let d_eps = -c_out * bpt.sqrt() / apt.sqrt();
        StepCoeffs {
            timestep: t as f32,
            c_in: 1.0,
            a_x: d_x as f32,
            a_out: d_eps as f32,
            a_noise: 0.0,
        }
    }
}

impl SamplerPolicy for LcmPolicy {
    fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    fn init_noise_scale(&self) -> f32 {
        1.0
    }

    fn step_dtype(&self) -> StepDtype {
        StepDtype::F32
    }

    fn coeffs(&self, i: usize) -> StepCoeffs {
        let det = self.denoised_coeffs(i);
        if i == self.timesteps.len() - 1 {
            // No re-noise on the final step (also: one-step sampling never re-noises).
            return det;
        }
        // prev = √ᾱ_prev·denoised + √β̄_prev·noise.
        let apt_prev = self.sched.acp(self.timesteps[i + 1]);
        let bpt_prev = 1.0 - apt_prev;
        let s = apt_prev.sqrt() as f32;
        StepCoeffs {
            timestep: det.timestep,
            c_in: 1.0,
            a_x: det.a_x * s,
            a_out: det.a_out * s,
            a_noise: bpt_prev.sqrt() as f32,
        }
    }

    fn sigma(&self, _i: usize) -> Option<f32> {
        None
    }
}

/// TCD (Hyper-SD) policy: diffusers `TCDScheduler` (ε-prediction). Like LCM but steps to an
/// intermediate noise level `s = ⌊(1−η)·t_prev⌋` and (for `η>0`) re-noises across the `t_prev`/`s`
/// gap.
#[derive(Clone)]
pub struct TcdPolicy {
    sched: AlphaSchedule,
    timesteps: Vec<usize>,
    eta: f32,
}

impl TcdPolicy {
    /// Build for `num_steps`. `original_inference_steps` is diffusers' default 50; `eta` is the
    /// stochasticity (`0.0` = deterministic; ByteDance's unified LoRA recommends ~`0.3`).
    pub fn new(
        sched: AlphaSchedule,
        num_train_timesteps: usize,
        original_inference_steps: usize,
        num_steps: usize,
        eta: f32,
    ) -> Self {
        Self {
            timesteps: lcm_style_timesteps(
                num_train_timesteps,
                original_inference_steps,
                num_steps,
            ),
            sched,
            eta,
        }
    }

    /// The deterministic noised-prediction (`pred_noised`) coefficients at step `i` — diffusers'
    /// `pred_noised_sample` before the `η>0` re-noise. `x_next = P_x·x + P_eps·eps`.
    pub fn pred_noised_coeffs(&self, i: usize) -> StepCoeffs {
        let t = self.timesteps[i];
        let last = i == self.timesteps.len() - 1;
        // prev_timestep = timesteps[i+1] if it exists else 0; timestep_s = floor((1−η)·prev_t).
        let prev_t = if last { 0 } else { self.timesteps[i + 1] };
        let timestep_s = ((1.0 - self.eta as f64) * prev_t as f64).floor() as usize;
        let apt = self.sched.acp(t);
        let bpt = 1.0 - apt;
        let aps = self.sched.acp(timestep_s);
        let bps = 1.0 - aps;
        // pred_noised = √ᾱ_s·pred_x0 + √β̄_s·eps, pred_x0 = (x − √β̄_t·eps)/√ᾱ_t
        //            = √(ᾱ_s/ᾱ_t)·x + (√β̄_s − √ᾱ_s·√β̄_t/√ᾱ_t)·eps.
        let p_x = (aps / apt).sqrt();
        let p_eps = bps.sqrt() - aps.sqrt() * bpt.sqrt() / apt.sqrt();
        StepCoeffs {
            timestep: t as f32,
            c_in: 1.0,
            a_x: p_x as f32,
            a_out: p_eps as f32,
            a_noise: 0.0,
        }
    }
}

impl SamplerPolicy for TcdPolicy {
    fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    fn init_noise_scale(&self) -> f32 {
        1.0
    }

    fn step_dtype(&self) -> StepDtype {
        StepDtype::F32
    }

    fn coeffs(&self, i: usize) -> StepCoeffs {
        let det = self.pred_noised_coeffs(i);
        let last = i == self.timesteps.len() - 1;
        if self.eta > 0.0 && !last {
            // prev = √(ᾱ_prev/ᾱ_s)·pred_noised + √(1 − ᾱ_prev/ᾱ_s)·noise.
            let prev_t = self.timesteps[i + 1];
            let timestep_s = ((1.0 - self.eta as f64) * prev_t as f64).floor() as usize;
            let ratio = self.sched.acp(prev_t) / self.sched.acp(timestep_s);
            let r = ratio.sqrt() as f32;
            return StepCoeffs {
                timestep: det.timestep,
                c_in: 1.0,
                a_x: det.a_x * r,
                a_out: det.a_out * r,
                a_noise: (1.0 - ratio).max(0.0).sqrt() as f32,
            };
        }
        det
    }

    fn sigma(&self, _i: usize) -> Option<f32> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sdxl_sched() -> AlphaSchedule {
        AlphaSchedule::scaled_linear(1000, 0.00085, 0.012).unwrap()
    }

    #[test]
    fn alphas_cumprod_is_monotonic_decreasing() {
        let s = sdxl_sched();
        assert_eq!(s.alphas_cumprod.len(), 1000);
        // ᾱ starts near 1 and decreases toward 0.
        assert!(s.alphas_cumprod[0] > 0.99);
        assert!(*s.alphas_cumprod.last().unwrap() < 0.01);
        assert!(s.alphas_cumprod.windows(2).all(|w| w[0] >= w[1]));
    }

    #[test]
    fn lcm_4step_timesteps_match_diffusers() {
        // diffusers LCMScheduler.set_timesteps(4) on N=1000, original=50.
        assert_eq!(lcm_style_timesteps(1000, 50, 4), vec![999, 759, 499, 259]);
    }

    #[test]
    fn lcm_8step_timesteps_descend_from_999() {
        let ts = lcm_style_timesteps(1000, 50, 8);
        assert_eq!(ts.len(), 8);
        assert_eq!(ts[0], 999);
        assert!(ts.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn lightning_trailing_timesteps_match_diffusers() {
        // diffusers EulerDiscreteScheduler(timestep_spacing="trailing").set_timesteps(4).
        let p = LightningPolicy::new(&sdxl_sched(), 1000, 4);
        assert_eq!(p.timesteps, vec![999.0, 749.0, 499.0, 249.0]);
        assert_eq!(p.sigmas.len(), 5);
        assert_eq!(*p.sigmas.last().unwrap(), 0.0);
        assert!(p.sigmas.windows(2).all(|w| w[0] > w[1]));
        assert_eq!(p.init_noise_scale(), p.sigmas[0]);
    }

    #[test]
    fn tcd_shares_lcm_timesteps() {
        let p = TcdPolicy::new(sdxl_sched(), 1000, 50, 4, 0.3);
        assert_eq!(p.timesteps, lcm_style_timesteps(1000, 50, 4));
    }

    #[test]
    fn policies_report_step_count() {
        assert_eq!(LcmPolicy::new(sdxl_sched(), 1000, 50, 4).num_steps(), 4);
        assert_eq!(LightningPolicy::new(&sdxl_sched(), 1000, 2).num_steps(), 2);
        assert_eq!(
            TcdPolicy::new(sdxl_sched(), 1000, 50, 8, 0.0).num_steps(),
            8
        );
    }

    #[test]
    fn image_seq_len_floors_to_16_and_guards_sub16() {
        // Valid (≥16) dims: floor-to-/16 then multiply, unchanged by the `.max(1)` guard.
        assert_eq!(image_seq_len(1024, 1024), (1024 / 16) * (1024 / 16));
        assert_eq!(image_seq_len(48, 32), 3 * 2);
        // L-E: a sub-16 dim floors to 0 without the guard, collapsing the sequence to length 0; the
        // `.max(1)` keeps it at ≥1 so a direct caller never fits `compute_mu` on an empty sequence.
        // width 8 → max(8/16, 1) = 1, so the result is height/16 = 64 (not 0).
        assert_eq!(image_seq_len(8, 1024), 1024 / 16);
        assert_eq!(image_seq_len(8, 8), 1);
    }

    // Coefficient goldens (epic 3720 §3.5): the same scalar references the original inline
    // sampler.rs tests asserted, now recombined through the neutral StepCoeffs (`a_x·x + a_out·out`).
    #[test]
    fn lcm_step0_denoised_matches_diffusers() {
        // step 0 of the 4-step schedule, t=999, eps=0.7, x=0.3 → diffusers denoised ≈ −5.835607.
        let c = LcmPolicy::new(sdxl_sched(), 1000, 50, 4).denoised_coeffs(0);
        let got = c.a_x * 0.3 + c.a_out * 0.7;
        assert!((got - (-5.835_607)).abs() < 1e-3, "got {got}");
    }

    #[test]
    fn lightning_step0_matches_diffusers() {
        let c = LightningPolicy::new(&sdxl_sched(), 1000, 4).coeffs(0);
        // scaled input = c_in·x at x=0.3.
        let scaled = c.c_in * 0.3;
        assert!((scaled - 0.020_479_47).abs() < 1e-4, "scaled {scaled}");
        // step at eps=0.7, x=0.3 → ≈ −7.073041 (a_x=1, a_noise=0).
        let got = c.a_x * 0.3 + c.a_out * 0.7;
        assert!((got - (-7.073_041)).abs() < 1e-3, "got {got}");
    }

    #[test]
    fn tcd_eta0_step0_pred_noised_matches_diffusers() {
        let c = TcdPolicy::new(sdxl_sched(), 1000, 50, 4, 0.0).pred_noised_coeffs(0);
        let got = c.a_x * 0.3 + c.a_out * 0.7;
        assert!((got - (-0.651_963_8)).abs() < 1e-4, "got {got}");
    }

    #[test]
    fn flow_match_static_shift_sigmas() {
        // diffusers FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=false, shift=3.0):
        // sigma' = 3·t/(1+2·t), t = linspace(1, 1/n, n); n=4 → [1, 0.9, 0.75, 0.5, 0].
        let sigmas = build_flow_sigmas(4, 3.0_f32.ln());
        let expected = [1.0_f32, 0.9, 0.75, 0.5, 0.0];
        assert_eq!(sigmas.len(), 5);
        for (got, want) in sigmas.iter().zip(expected) {
            assert!(
                (got - want).abs() < 1e-5,
                "static shift: got {got} want {want}"
            );
        }
    }

    #[test]
    fn flow_match_coeffs_are_pure_euler() {
        // a_x=1, a_noise=0, a_out = σ_{i+1}−σ_i; timestep = σ (Sigma convention).
        let sigmas = vec![1.0_f32, 0.75, 0.5, 0.25, 0.0];
        let p = FlowMatchPolicy::new(sigmas.clone(), TimestepConvention::Sigma);
        assert_eq!(p.num_steps(), 4);
        for i in 0..4 {
            let c = p.coeffs(i);
            assert_eq!(c.a_x, 1.0);
            assert_eq!(c.a_noise, 0.0);
            assert_eq!(c.timestep, sigmas[i]);
            assert_eq!(c.a_out, sigmas[i + 1] - sigmas[i]);
        }
        // OneMinusSigma convention flips the timestep.
        let p2 = FlowMatchPolicy::new(sigmas.clone(), TimestepConvention::OneMinusSigma);
        assert_eq!(p2.coeffs(0).timestep, 1.0 - sigmas[0]);
    }

    #[test]
    fn image_seq_len_matches_definition() {
        assert_eq!(image_seq_len(1024, 1024), 4096);
        assert_eq!(image_seq_len(256, 256), 256);
        assert_eq!(image_seq_len(1280, 1280), 6400);
    }

    #[test]
    fn mu_large_seq_branch_independent_of_steps() {
        assert!((compute_mu(6400, 4) - compute_mu(6400, 8)).abs() < 1e-6);
    }

    /// The AC demonstration: a candle implementer drives the policy with a plain `Vec<f32>` "tensor"
    /// and a hand-written `apply_step`, producing a working sampler with no backend tensor library.
    #[test]
    fn policy_drives_a_plain_vec_tensor_backend() {
        // A toy "backend": x and model_output are Vec<f32>; one affine step + optional noise.
        fn apply_step(c: &StepCoeffs, x: &[f32], out: &[f32], noise: &[f32]) -> Vec<f32> {
            // Byte-parity rule mirror: a_x==1 && a_noise==0 ⇒ x + out·a_out.
            if c.a_x == 1.0 && c.a_noise == 0.0 {
                return x.iter().zip(out).map(|(&a, &b)| a + b * c.a_out).collect();
            }
            x.iter()
                .zip(out)
                .zip(noise)
                .map(|((&a, &b), &n)| c.a_x * a + c.a_out * b + c.a_noise * n)
                .collect()
        }

        let sigmas = vec![1.0_f32, 0.75, 0.5, 0.25, 0.0];
        let policy = FlowMatchPolicy::new(sigmas, TimestepConvention::Sigma);
        let mut x = vec![0.3_f32];
        for i in 0..policy.num_steps() {
            let c = policy.coeffs(i);
            // velocity-prediction model stub: out = 0.7 at every step.
            x = apply_step(&c, &x, &[0.7], &[0.0]);
            assert!(x[0].is_finite());
        }
        // step 0: 0.3 + 0.7·(0.75−1.0) = 0.125.
        let c0 = policy.coeffs(0);
        assert!((0.3 + 0.7 * c0.a_out - 0.125).abs() < 1e-6);
    }
}
