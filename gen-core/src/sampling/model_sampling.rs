//! The prediction-type layer of the unified sampler framework (epic 7114, P1): convert a raw model
//! output into a denoised `x0` estimate in normalized sigma space, and expose the sigma↔timestep
//! mapping the solvers integrate over.
//!
//! This is the decoupling layer mlx-gen lacks today. The legacy `SamplerPolicy` (`super`) bakes the
//! prediction type into each policy's precomputed coefficients, entangling the *integration method*
//! with the *prediction type*. Here they split: a [`ModelSampling`] owns ONLY the locked prediction
//! type (EPS / V / FLOW), and the callback [`super::unified::Sampler`] owns ONLY the solver. An engine
//! composes them — it builds a `denoise(x, σ) -> x0` closure (input scaling → model forward →
//! [`ModelSampling::denoised_coeffs`]) via [`denoise`] and hands it to any sampler.
//!
//! Mirrors ComfyUI's `comfy/model_sampling.py` (`EPS` / `V_PREDICTION` / `CONST`, with
//! `ModelSamplingDiscrete` / `ModelSamplingContinuousEDM` / `ModelSamplingFlux` schedules) reduced to
//! backend-neutral scalar coefficients: the tensor blends are applied by the caller through
//! [`super::LatentOps`], so this module stays pure host math (gen-core's zero-tensor-dep invariant).

use super::{AlphaSchedule, LatentOps, TimestepConvention};
use crate::Result;

/// The locked prediction type a model was trained with — what the raw network output *means*.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredictionType {
    /// ε-prediction (the DDPM / SDXL / Kolors world). `x0 = x − σ·ε`; model input scaled by
    /// `1/√(σ² + σ_data²)`.
    Eps,
    /// v-prediction (EDM / Stable-Video-Diffusion). `x0` mixes `x` and the output through `σ_data`.
    V,
    /// Rectified-flow / ComfyUI `CONST` (FLUX / Qwen / Z-Image / Boogu). `x0 = x − σ·v`; input
    /// unscaled.
    Flow,
}

/// The prediction-type + noise-schedule contract a solver integrates over.
///
/// A `ModelSampling` answers four questions for any sigma on the schedule:
/// 1. how to scale the latent before the model forward ([`Self::input_scale`], the `c_in`);
/// 2. how to recombine `(x, raw_output)` into a denoised `x0` ([`Self::denoised_coeffs`]);
/// 3. what conditioning value the model embeds at this sigma ([`Self::timestep`]);
/// 4. the inverse sigma↔timestep map ([`Self::sigma`]) + the schedule endpoints
///    ([`Self::sigma_min`] / [`Self::sigma_max`]), for schedule construction and img2img start-step.
pub trait ModelSampling {
    /// The locked prediction type (for introspection / capability reporting).
    fn prediction(&self) -> PredictionType;

    /// Smallest schedule sigma (the near-clean end).
    fn sigma_min(&self) -> f32;
    /// Largest schedule sigma (the pure-noise end).
    fn sigma_max(&self) -> f32;

    /// `c_in`: the scalar the latent is multiplied by before the model forward. `1.0` (FLOW) means
    /// the caller skips the multiply entirely (byte-identical no-op).
    fn input_scale(&self, sigma: f32) -> f32;

    /// `(k_x, k_out)` such that the denoised estimate is `x0 = k_x·x + k_out·raw_output`, where `x`
    /// is the **un-scaled** latent (NOT the [`Self::input_scale`] output) — matching ComfyUI's
    /// `calculate_denoised(sigma, model_output, model_input)`.
    fn denoised_coeffs(&self, sigma: f32) -> (f32, f32);

    /// The conditioning value the model embeds at `sigma` (what the time-embedding consumes).
    fn timestep(&self, sigma: f32) -> f32;

    /// Inverse of [`Self::timestep`]: the sigma at a (float) conditioning value. Used to seed the
    /// img2img / video start-step noise blend and to build schedules in timestep space.
    fn sigma(&self, timestep: f32) -> f32;

    /// The number of discrete training-timestep nodes the table-indexed schedulers (simple / ddim /
    /// beta, sc-7116) sample over — ComfyUI's `len(model_sampling.sigmas)`. Default 1000 (the
    /// standard DDPM / flow training-step count); the discrete schedule overrides it to its table
    /// length.
    fn num_timesteps(&self) -> usize {
        1000
    }

    /// The discrete per-node sigma table the table-indexed schedulers sample, ASCENDING
    /// (`table[0]` ≈ [`Self::sigma_min`] … `table[last]` ≈ [`Self::sigma_max`]). ComfyUI's
    /// `model_sampling.sigmas`. The default samples [`Self::sigma`] across the conditioning grid
    /// `[timestep(σ_min) .. timestep(σ_max)]`, which is EXACT for the discrete schedule (where
    /// `timestep` is an integer index), log-linear for EDM, and a linear `σ` ramp for flow.
    fn sigma_table(&self) -> Vec<f32> {
        let n = self.num_timesteps();
        let t_lo = self.timestep(self.sigma_min());
        let t_hi = self.timestep(self.sigma_max());
        (0..n)
            .map(|i| {
                let f = if n <= 1 {
                    0.0
                } else {
                    i as f32 / (n - 1) as f32
                };
                self.sigma(t_lo + (t_hi - t_lo) * f)
            })
            .collect()
    }
}

/// Compute a denoised `x0 = denoise(x, σ)` from a `ModelSampling` and a raw-model closure.
///
/// This is the bridge an engine wraps its DiT/U-Net forward in: `run_model(scaled_input, timestep)`
/// returns the raw network output, and this applies the `c_in` input scaling and the prediction-type
/// `x0` recombination through [`LatentOps`]. The resulting `denoise` callback is exactly what the
/// callback [`super::unified::Sampler`] consumes — the sampler never sees the prediction type.
pub fn denoise<L, M>(
    ops: &L,
    ms: &dyn ModelSampling,
    x: &L::Latent,
    sigma: f32,
    mut run_model: M,
) -> Result<L::Latent>
where
    L: LatentOps,
    M: FnMut(&L::Latent, f32) -> Result<L::Latent>,
{
    let s = ms.input_scale(sigma);
    let x_in = if s == 1.0 {
        x.clone()
    } else {
        ops.scale(x, s)?
    };
    let raw = run_model(&x_in, ms.timestep(sigma))?;
    let (k_x, k_out) = ms.denoised_coeffs(sigma);
    ops.axpy(k_x, x, k_out, &raw)
}

// =================================================================================================
// FLOW / CONST — rectified-flow (FLUX / Qwen / Z-Image / Boogu). The byte-equivalence anchor.
// =================================================================================================

/// Rectified-flow (ComfyUI `CONST`) model sampling. `σ ∈ [0, 1]`, input is unscaled, and the model
/// output is the velocity: `x0 = x − σ·v`. With the Euler solver this reproduces the legacy
/// [`super::FlowMatchPolicy`] step `x + v·(σ_{i+1} − σ_i)` (the `to_d` round-trip is an f32-cancellation
/// away — see [`super::unified`]). The [`TimestepConvention`] selects whether the model is fed the raw
/// sigma (FLUX / Qwen) or `1 − σ` (the Z-Image-style DiTs).
#[derive(Clone, Copy, Debug)]
pub struct FlowModelSampling {
    conv: TimestepConvention,
    /// Time-shift `mu` (`exp(mu)` is the diffusers/ComfyUI `shift`). `0.0` is the identity (no shift) —
    /// the byte-exact pre-shift behaviour. An engine that builds a resolution-/config-shifted native
    /// schedule passes its own `mu` (e.g. `compute_mu(image_seq_len, steps)` for FLUX.2/Qwen, or
    /// `shift.ln()` for a static-shift model) so a curated scheduler stays consistent with the engine's
    /// native time-shift (epic 7114 scheduler axis, sc-7120).
    mu: f32,
}

impl FlowModelSampling {
    /// Build for a timestep convention with NO time-shift (`mu = 0`). FLUX / Qwen / Chroma feed the raw
    /// sigma ([`TimestepConvention::Sigma`]); Z-Image feeds `1 − σ`
    /// ([`TimestepConvention::OneMinusSigma`]).
    pub fn new(conv: TimestepConvention) -> Self {
        Self { conv, mu: 0.0 }
    }

    /// Build with an explicit time-shift `mu`, so a curated `normal` / `sgm_uniform` / `simple` / `beta`
    /// / `ddim_uniform` schedule built over this model reproduces the engine's resolution-/config-shift.
    /// `mu = 0` reduces to [`Self::new`]. The shift modifies only the schedule-construction map
    /// ([`Self::sigma`]), NOT the model conditioning ([`Self::timestep`]): schedule construction is
    /// convention-independent (the σ ramp is the same noise-fraction schedule however the model consumes
    /// σ), so engines build curated schedules with [`TimestepConvention::Sigma`] regardless of their own
    /// conditioning convention, and the conditioning flip is applied separately at the model forward.
    pub fn with_shift(conv: TimestepConvention, mu: f32) -> Self {
        Self { conv, mu }
    }
}

impl ModelSampling for FlowModelSampling {
    fn prediction(&self) -> PredictionType {
        PredictionType::Flow
    }
    fn sigma_min(&self) -> f32 {
        // Smallest POSITIVE scheduled sigma — the flow clean-end node (σ = 0 is the terminal, not a
        // schedulable sigma). ComfyUI flux derives σ_min from its table's first entry; for the
        // unshifted flow schedule that is `1/num_timesteps`. Keeps the σ-schedulers (sc-7116) off
        // `log(0)` and the `normal`/`sgm_uniform` schedules from ending in a spurious second zero.
        1.0 / self.num_timesteps() as f32
    }
    fn sigma_max(&self) -> f32 {
        1.0
    }
    fn input_scale(&self, _sigma: f32) -> f32 {
        1.0
    }
    fn denoised_coeffs(&self, sigma: f32) -> (f32, f32) {
        // x0 = 1·x + (−σ)·v.
        (1.0, -sigma)
    }
    fn timestep(&self, sigma: f32) -> f32 {
        // The model conditioning is the post-shift sigma (FLUX/Qwen/Chroma) or `1 − σ` (Z-Image): the
        // shift lives in the schedule, not here, so this is unchanged by `mu`.
        match self.conv {
            TimestepConvention::Sigma => sigma,
            TimestepConvention::OneMinusSigma => 1.0 - sigma,
        }
    }
    fn sigma(&self, timestep: f32) -> f32 {
        match self.conv {
            // The schedule coordinate `t ∈ [0,1]` maps to the shifted sigma through the exponential
            // time-shift (`mu = 0` ⇒ identity `t`). This is exactly the per-node shift `build_flow_sigmas`
            // applies, so the `normal` scheduler over a shifted FlowModelSampling reproduces the engine's
            // native `linspace(1, 1/N, N)`-through-shift schedule. Schedule construction always uses the
            // Sigma convention (see [`Self::with_shift`]), so this is the only branch the schedulers hit.
            TimestepConvention::Sigma => super::time_shift_exponential(self.mu, timestep),
            // OneMinusSigma keeps the un-shifted timestep-inverse form (never the schedule map).
            TimestepConvention::OneMinusSigma => 1.0 - timestep,
        }
    }
}

// =================================================================================================
// EPS — DDPM discrete (SDXL / Kolors). ComfyUI `ModelSamplingDiscrete` + `EPS`.
// =================================================================================================

/// Discrete-schedule ε / v model sampling (ComfyUI `ModelSamplingDiscrete`). Sigmas are
/// `√((1−ᾱ_t)/ᾱ_t)` over the training timesteps (the [`AlphaSchedule`] table); `timestep(σ)` is the
/// nearest training index in log-sigma space and `sigma(t)` interpolates — matching ComfyUI's
/// `timestep`/`sigma`. `σ_data = 1.0` for the standard SDXL/SD `scaled_linear` world.
#[derive(Clone)]
pub struct DiscreteModelSampling {
    /// log of `√((1−ᾱ_t)/ᾱ_t)` per training timestep `t` (ascending in `t`, so ascending in σ).
    log_sigmas: Vec<f32>,
    prediction: PredictionType,
    sigma_data: f32,
}

impl DiscreteModelSampling {
    /// Build from a DDPM `alphas_cumprod` schedule (e.g. `AlphaSchedule::scaled_linear`). `prediction`
    /// is [`PredictionType::Eps`] (SDXL/Kolors) or [`PredictionType::V`] (SD2.x-v on a discrete
    /// schedule); `sigma_data` is `1.0` for the standard world.
    pub fn new(sched: &AlphaSchedule, prediction: PredictionType, sigma_data: f32) -> Self {
        let n = sched.alphas_cumprod.len();
        let log_sigmas: Vec<f32> = (0..n).map(|t| (sched.sigma_at(t) as f32).ln()).collect();
        Self {
            log_sigmas,
            prediction,
            sigma_data,
        }
    }

    /// SDXL/Kolors default: ε-prediction over `scaled_linear` betas, `σ_data = 1`.
    pub fn sdxl(sched: &AlphaSchedule) -> Self {
        Self::new(sched, PredictionType::Eps, 1.0)
    }
}

impl ModelSampling for DiscreteModelSampling {
    fn prediction(&self) -> PredictionType {
        self.prediction
    }
    fn sigma_min(&self) -> f32 {
        self.log_sigmas.first().copied().unwrap_or(0.0).exp()
    }
    fn sigma_max(&self) -> f32 {
        self.log_sigmas.last().copied().unwrap_or(0.0).exp()
    }
    fn input_scale(&self, sigma: f32) -> f32 {
        1.0 / (sigma * sigma + self.sigma_data * self.sigma_data).sqrt()
    }
    fn denoised_coeffs(&self, sigma: f32) -> (f32, f32) {
        prediction_denoised_coeffs(self.prediction, sigma, self.sigma_data)
    }
    fn timestep(&self, sigma: f32) -> f32 {
        // Nearest training timestep in log-sigma space (ComfyUI `timestep`: argmin|log σ − log σ_t|).
        let log_sigma = sigma.max(1e-12).ln();
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for (t, &ls) in self.log_sigmas.iter().enumerate() {
            let d = (log_sigma - ls).abs();
            if d < best_d {
                best_d = d;
                best = t;
            }
        }
        best as f32
    }
    fn sigma(&self, timestep: f32) -> f32 {
        // Interpolate log-sigmas at the float timestep (ComfyUI `sigma`).
        let n = self.log_sigmas.len();
        if n == 0 {
            return 0.0;
        }
        let t = timestep.clamp(0.0, (n - 1) as f32);
        let lo = t.floor() as usize;
        let hi = (lo + 1).min(n - 1);
        let w = t - lo as f32;
        (self.log_sigmas[lo] * (1.0 - w) + self.log_sigmas[hi] * w).exp()
    }
    fn num_timesteps(&self) -> usize {
        self.log_sigmas.len()
    }
}

// =================================================================================================
// V — continuous EDM (Stable Video Diffusion). ComfyUI `ModelSamplingContinuousEDM` + `V_PREDICTION`.
// =================================================================================================

/// Continuous-EDM model sampling (ComfyUI `ModelSamplingContinuousEDM`). Used by SVD (v-prediction).
/// `timestep(σ) = 0.25·ln(σ)` (the EDM `c_noise`), `sigma(t) = exp(4t)`; endpoints are the model's
/// configured `[σ_min, σ_max]`.
#[derive(Clone, Copy, Debug)]
pub struct EdmModelSampling {
    prediction: PredictionType,
    sigma_min: f32,
    sigma_max: f32,
    sigma_data: f32,
}

impl EdmModelSampling {
    /// Build for a prediction type with explicit EDM endpoints + `σ_data`.
    pub fn new(
        prediction: PredictionType,
        sigma_min: f32,
        sigma_max: f32,
        sigma_data: f32,
    ) -> Self {
        Self {
            prediction,
            sigma_min,
            sigma_max,
            sigma_data,
        }
    }

    /// SVD default: v-prediction, `σ_data = 1`, the EDM range ComfyUI configures for Stable Video
    /// Diffusion (`σ_min = 0.002`, `σ_max = 700`).
    pub fn svd() -> Self {
        Self::new(PredictionType::V, 0.002, 700.0, 1.0)
    }
}

impl ModelSampling for EdmModelSampling {
    fn prediction(&self) -> PredictionType {
        self.prediction
    }
    fn sigma_min(&self) -> f32 {
        self.sigma_min
    }
    fn sigma_max(&self) -> f32 {
        self.sigma_max
    }
    fn input_scale(&self, sigma: f32) -> f32 {
        1.0 / (sigma * sigma + self.sigma_data * self.sigma_data).sqrt()
    }
    fn denoised_coeffs(&self, sigma: f32) -> (f32, f32) {
        prediction_denoised_coeffs(self.prediction, sigma, self.sigma_data)
    }
    fn timestep(&self, sigma: f32) -> f32 {
        0.25 * sigma.max(1e-12).ln()
    }
    fn sigma(&self, timestep: f32) -> f32 {
        (4.0 * timestep).exp()
    }
}

/// The prediction-type-only part of `x0 = k_x·x + k_out·raw_output` (schedule-independent). Shared by
/// the discrete and EDM model samplings; mirrors ComfyUI's `EPS` / `V_PREDICTION` / `CONST`
/// `calculate_denoised`.
fn prediction_denoised_coeffs(p: PredictionType, sigma: f32, sigma_data: f32) -> (f32, f32) {
    match p {
        // x0 = x − σ·ε.
        PredictionType::Eps | PredictionType::Flow => (1.0, -sigma),
        // x0 = x·(σd²/(σ²+σd²)) − v·(σ·σd/√(σ²+σd²)).
        PredictionType::V => {
            let sd2 = sigma_data * sigma_data;
            let denom = sigma * sigma + sd2;
            (sd2 / denom, -(sigma * sigma_data) / denom.sqrt())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::CpuLatentOps;

    fn sdxl_sched() -> AlphaSchedule {
        AlphaSchedule::scaled_linear(1000, 0.00085, 0.012).unwrap()
    }

    #[test]
    fn flow_is_pure_velocity_euler_coeffs() {
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        assert_eq!(ms.prediction(), PredictionType::Flow);
        assert_eq!(ms.input_scale(0.7), 1.0);
        assert_eq!(ms.denoised_coeffs(0.7), (1.0, -0.7));
        assert_eq!(ms.timestep(0.7), 0.7);
        assert_eq!(ms.sigma(0.7), 0.7);
        // OneMinusSigma flips the conditioning both ways.
        let z = FlowModelSampling::new(TimestepConvention::OneMinusSigma);
        assert_eq!(z.timestep(0.3), 0.7);
        assert_eq!(z.sigma(0.7), 0.3);
    }

    #[test]
    fn flow_shift_only_touches_sigma_map_not_conditioning() {
        // mu = 0 (`new`) is the identity: `sigma(t) == t` — the byte-exact pre-shift behaviour.
        let plain = FlowModelSampling::new(TimestepConvention::Sigma);
        for &t in &[0.05_f32, 0.3, 0.5, 0.8, 1.0] {
            assert!(
                (plain.sigma(t) - t).abs() < 1e-7,
                "mu=0 must be identity at {t}"
            );
        }
        // A shifted flow model maps the schedule coordinate through the exponential time-shift, which is
        // exactly the diffusers static-shift `shift·t/(1+(shift−1)·t)` with `shift = exp(mu)`.
        let mu = 3.0_f32.ln();
        let shifted = FlowModelSampling::with_shift(TimestepConvention::Sigma, mu);
        for &t in &[0.1_f32, 0.25, 0.5, 0.75, 0.9] {
            let want = 3.0 * t / (1.0 + (3.0 - 1.0) * t); // shift = 3.0
            assert!(
                (shifted.sigma(t) - want).abs() < 1e-6,
                "shift map at {t}: got {} want {want}",
                shifted.sigma(t)
            );
        }
        // The model conditioning (`timestep`) is unchanged by the shift — only the schedule map moves.
        assert_eq!(shifted.timestep(0.7), 0.7);
        assert_eq!(shifted.denoised_coeffs(0.7), (1.0, -0.7));
        assert_eq!(shifted.input_scale(0.7), 1.0);
    }

    #[test]
    fn curated_scheduler_over_shifted_flow_is_valid_and_materially_shifted() {
        // The scheduler-axis guarantee: building a curated `normal`/`sgm_uniform` schedule over a
        // SHIFTED FlowModelSampling produces a valid descending-to-zero schedule that is MATERIALLY
        // different from the unshifted one — i.e. the engine's `mu` flows through `schedule_sigmas` and
        // actually bends the schedule (without it a high-shift model would get a near-linear σ ramp and
        // be starved of high-noise steps). It is NOT meant to reproduce the engine's native schedule
        // byte-for-byte — ComfyUI's `normal` floors σ_min at `1/num_timesteps`, not `1/steps`, so it is a
        // distinct (alternative) schedule; the native default stays byte-exact via the `None` path.
        use crate::sampling::{schedule_sigmas, Scheduler};
        let mu = 3.0_f32.ln(); // shift = 3.0
        let steps = 8;
        let shifted = FlowModelSampling::with_shift(TimestepConvention::Sigma, mu);
        let plain = FlowModelSampling::new(TimestepConvention::Sigma);
        for sched in [Scheduler::Normal, Scheduler::SgmUniform, Scheduler::Simple] {
            let s = schedule_sigmas(sched, &shifted, steps);
            assert!(s.len() >= 2);
            assert_eq!(*s.last().unwrap(), 0.0, "{} trailing 0", sched.name());
            assert!(
                s.windows(2).all(|w| w[0] >= w[1]),
                "{} not descending: {s:?}",
                sched.name()
            );
            assert!(
                s[..s.len() - 1].iter().all(|&v| v > 0.0),
                "{} has a non-positive interior node: {s:?}",
                sched.name()
            );
            // The shift moved the schedule vs the unshifted build.
            let u = schedule_sigmas(sched, &plain, steps);
            let gap = s
                .iter()
                .zip(&u)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                gap > 0.02,
                "{} shift had no effect (gap {gap})",
                sched.name()
            );
        }
    }

    #[test]
    fn flow_denoise_recovers_x0_from_velocity() {
        // denoise(x, σ) with a constant-velocity stub returns x0 = x − σ·v exactly.
        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let x = vec![0.3_f32, -1.0, 2.0];
        let v = vec![0.7_f32, 0.5, -0.25];
        let x0 = denoise(&ops, &ms, &x, 0.6, |_xin, _t| Ok(v.clone())).unwrap();
        for ((g, &xi), &vi) in x0.iter().zip(&x).zip(&v) {
            assert!((g - (xi - 0.6 * vi)).abs() < 1e-6, "got {g}");
        }
    }

    #[test]
    fn eps_input_scale_and_denoised_match_comfy() {
        let ms = DiscreteModelSampling::sdxl(&sdxl_sched());
        assert_eq!(ms.prediction(), PredictionType::Eps);
        // c_in = 1/√(σ²+1).
        let sigma = 2.0_f32;
        assert!((ms.input_scale(sigma) - 1.0 / (sigma * sigma + 1.0).sqrt()).abs() < 1e-7);
        // x0 = x − σ·ε.
        assert_eq!(ms.denoised_coeffs(sigma), (1.0, -sigma));
    }

    #[test]
    fn discrete_timestep_sigma_roundtrip() {
        // sigma(timestep(σ_t)) ≈ σ_t at a training-grid sigma, and timestep is the right index.
        let sched = sdxl_sched();
        let ms = DiscreteModelSampling::sdxl(&sched);
        let t = 500usize;
        let sigma_t = sched.sigma_at(t) as f32;
        assert_eq!(ms.timestep(sigma_t), t as f32);
        assert!((ms.sigma(t as f32) - sigma_t).abs() / sigma_t < 1e-4);
        // Endpoints: σ_min near t=0 (clean), σ_max near t=N−1 (noisy).
        assert!(ms.sigma_min() < ms.sigma_max());
        assert!(ms.sigma_min() < 0.1);
    }

    #[test]
    fn edm_v_prediction_coeffs_match_formula() {
        let ms = EdmModelSampling::svd();
        assert_eq!(ms.prediction(), PredictionType::V);
        let sigma = 3.0_f32;
        let (k_x, k_out) = ms.denoised_coeffs(sigma);
        // σ_data = 1: k_x = 1/(σ²+1), k_out = −σ/√(σ²+1).
        assert!(
            (k_x - 1.0 / (sigma * sigma + 1.0)).abs() < 1e-7,
            "k_x {k_x}"
        );
        assert!(
            (k_out - (-(sigma) / (sigma * sigma + 1.0).sqrt())).abs() < 1e-7,
            "k_out {k_out}"
        );
        // EDM timestep/sigma inverse: sigma(0.25·ln σ) == σ.
        assert!((ms.sigma(ms.timestep(sigma)) - sigma).abs() < 1e-4);
        assert_eq!(ms.sigma_min(), 0.002);
        assert_eq!(ms.sigma_max(), 700.0);
    }
}
