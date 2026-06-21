//! Swappable diffusion samplers — the engine-agnostic seam behind the few-step acceleration
//! variants (LCM / SDXL-Lightning / Hyper-SD), sc-2769.
//!
//! As of sc-3722 the **policy** (schedules + per-step affine coefficients) lives in the
//! backend-neutral [`gen_core::sampling`] crate; this module keeps only the thin **tensor
//! application**. Each sampler type below is a wrapper holding a [`gen_core::sampling::SamplerPolicy`]
//! plus the MLX compute dtype, so the family-crate call sites are unchanged (D5). The neutral
//! coefficients (`a_x`/`a_out`/`a_noise`/`c_in`) are applied by one shared [`apply_step`]; a candle
//! backend implements the same ~5 lines against the same policies.
//!
//! A [`DiffusionSampler`] owns a model's **denoise schedule**: the per-step conditioning timestep,
//! the model-input scaling, the initial-noise scaling, and the per-step update. The generic denoise
//! loop drives `&dyn DiffusionSampler` so a model can swap samplers per request without the loop
//! knowing which one is running. Each model family supplies its own impls:
//! - SDXL's production default is the crate-local ancestral Euler sampler (`mlx-gen-sdxl`), which
//!   folds the input scaling into its step → [`DiffusionSampler::scale_model_input`] is identity.
//! - The acceleration samplers here are faithful ports of the **diffusers** schedulers each method
//!   is trained against (`LCMScheduler`, `EulerDiscreteScheduler(timestep_spacing="trailing")`,
//!   `TCDScheduler`); their schedule math (the DDPM `alphas_cumprod` world) is the policy layer.
//!
//! FLUX-MLX and Qwen-MLX acceleration both drive the shared [`FlowMatchSampler`] (the rectified-flow
//! world, sc-2908 / sc-2909); the Qwen-specific Lightning sigma schedule is built in
//! `mlx-gen-qwen-image` and wrapped in this same sampler (deduped in sc-2950).

use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::{random, Array, Dtype};

use gen_core::sampling::{
    LatentOps, LcmPolicy, LightningPolicy, SamplerPolicy, StepCoeffs, StepDtype, TcdPolicy,
    TimestepConvention,
};

use crate::array::scalar;
use crate::Result;

/// The DDPM `alphas_cumprod` noise schedule, re-exported from gen-core at the historical
/// `mlx_gen::sampler::AlphaSchedule` path (SDXL/Kolors build it for the acceleration samplers and
/// training).
pub use gen_core::sampling::{AlphaSchedule, FlowMatchPolicy};

/// A swappable denoise schedule. The generic loop calls, per step `i`:
/// `x_in = scale_model_input(latents, i)` → `eps = model(x_in, timestep(i))` → (CFG) →
/// `latents = step(eps, latents, i)`. The starting latents are `scale_initial_noise(unit_noise)`.
pub trait DiffusionSampler {
    /// Number of denoise iterations (loop count).
    fn num_steps(&self) -> usize;

    /// The conditioning timestep fed to the model at step `i` (the value the U-Net embeds).
    fn timestep(&self, i: usize) -> f32;

    /// Scale the latents into the model's expected input space at step `i`. The default is identity
    /// (samplers that fold the scaling into [`Self::step`], e.g. the ancestral Euler sampler, and
    /// the flow-match sampler whose `c_in = 1`); diffusers' Euler divides by `√(σ²+1)`.
    fn scale_model_input(&self, x: &Array, _i: usize) -> Result<Array> {
        Ok(x.clone())
    }

    /// Scale unit-normal noise into the sampler's starting latent space (the txt2img prior).
    fn scale_initial_noise(&self, noise: &Array) -> Result<Array>;

    /// One denoise step: latents at step `i` → latents at step `i+1`, given the (already
    /// CFG-combined) model output. `x` is the **un-scaled** latents (NOT the
    /// [`Self::scale_model_input`] output), matching diffusers' `step(model_output, t, sample)`.
    fn step(&self, model_output: &Array, x: &Array, i: usize) -> Result<Array>;
}

// =================================================================================================
// Shared tensor application — the only numeric code that stays per-backend (the policy is neutral).
// =================================================================================================

/// Seed-derived per-step noise source (D6). Stochastic samplers (LCM re-noise, TCD `η>0`) draw their
/// between-step noise from a subkey split off the request seed by step index, so the trajectory is
/// deterministic for a given seed regardless of the global RNG draw order (the previous unseeded
/// `random::normal(…, None)` was order-dependent). **Same-backend determinism only** — cross-backend
/// bitwise equality is explicitly NOT a goal (RNG algorithms differ).
pub struct StepRng {
    seed: u64,
}

impl StepRng {
    /// A step-RNG keyed off the request seed. Deterministic samplers pass any value (the byte-parity
    /// branch never draws), so wrappers without a request seed use `StepRng::new(0)`.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Unit-normal noise for step `step`, drawn from a distinct subkey. The multiplier de-correlates
    /// consecutive steps; the `+1` keeps step 0 off the raw seed used for the init-noise prior.
    fn normal(&self, shape: &[i32], step: usize) -> Result<Array> {
        let sub = self
            .seed
            .wrapping_add(0x9E37_79B9_7F4A_7C15_u64.wrapping_mul(step as u64 + 1));
        let key = random::key(sub)?;
        Ok(random::normal::<f32>(shape, None, None, Some(&key))?)
    }
}

/// DDPM model-input scaling: `cast(c_in · x, model_dtype)`. `c_in = 1` (LCM/TCD) skips the multiply
/// to stay byte-identical to the original `x.as_dtype(model_dtype)`.
fn scale_input(c_in: f32, model_dtype: Dtype, x: &Array) -> Result<Array> {
    let scaled = if c_in == 1.0 {
        x.clone()
    } else {
        multiply(x, scalar(c_in))?
    };
    Ok(scaled.as_dtype(model_dtype)?)
}

/// Scale unit-normal noise by `init_noise_scale` (the txt2img prior). `scale = 1` (LCM/TCD/flow-match)
/// is the identity cast to f32; Lightning multiplies by its max sigma.
fn scale_initial(scale: f32, noise: &Array) -> Result<Array> {
    let n = noise.as_dtype(Dtype::Float32)?;
    if scale == 1.0 {
        Ok(n)
    } else {
        Ok(multiply(&n, scalar(scale))?)
    }
}

/// Apply one neutral [`StepCoeffs`] to the latents: `x_next = a_x·x + a_out·out + a_noise·ε`.
///
/// **Byte-parity rule (§3.3):** when `a_x == 1.0 && a_noise == 0.0`, emit exactly `x + out·a_out`
/// (the original `flow_match_euler_step` / Lightning Euler expression), NOT `x·1.0 + …` — the F-009
/// `scheduler_and_sampler_steps_are_identical` test and the FLUX golden images must stay
/// byte-identical. `StepDtype::F32` upcasts both operands (the DDPM samplers, diffusers parity);
/// `StepDtype::Latents` computes in the latents' dtype (flow-match).
fn apply_step(
    c: &StepCoeffs,
    dt: StepDtype,
    x: &Array,
    out: &Array,
    step: usize,
    rng: &StepRng,
) -> Result<Array> {
    let (x, out) = match dt {
        StepDtype::F32 => (x.as_dtype(Dtype::Float32)?, out.as_dtype(Dtype::Float32)?),
        StepDtype::Latents => (x.clone(), out.clone()),
    };
    if c.a_x == 1.0 && c.a_noise == 0.0 {
        return Ok(add(&x, &multiply(&out, scalar(c.a_out))?)?);
    }
    let mut acc = add(
        &multiply(&x, scalar(c.a_x))?,
        &multiply(&out, scalar(c.a_out))?,
    )?;
    if c.a_noise != 0.0 {
        let noise = rng.normal(acc.shape(), step)?;
        acc = add(&acc, &multiply(&noise, scalar(c.a_noise))?)?;
    }
    Ok(acc)
}

// =================================================================================================
// Unified framework backend (epic 7114 P2, sc-7118): the gen-core `LatentOps` impl over MLX `Array`.
// =================================================================================================

/// The mlx-gen backend impl of [`gen_core::sampling::LatentOps`] — the tensor primitives the unified
/// curated samplers (Euler / Heun / DPM++ 2M·SDE / UniPC / ancestral / LCM / DDIM, sc-7117) are
/// written against. Carries the same byte-parity rules as the legacy [`apply_step`] so an engine's
/// DEFAULT sampler stays bit-identical after it migrates onto the unified framework (the N1 gate):
/// `scale(x, 1.0)` and `axpy(1.0, x, b, y) = x + y·b` elide the multiply-by-one, and `randn_like`
/// reuses the seed-derived [`StepRng`] subkey so a stochastic sampler is deterministic per request
/// seed regardless of global RNG draw order.
#[derive(Clone, Copy, Debug, Default)]
pub struct MlxLatentOps;

/// Lift a raw MLX op error into the backend-neutral [`gen_core::Error`] (the `LatentOps` trait is
/// declared in gen-core, so its methods return `gen_core::Result`). Routes through the existing
/// mlx-gen bridge: `Exception -> mlx_gen::Error -> gen_core::Error`.
#[inline]
fn ge<T>(r: std::result::Result<T, mlx_rs::error::Exception>) -> gen_core::Result<T> {
    r.map_err(|e| crate::Error::from(e).into())
}

impl LatentOps for MlxLatentOps {
    type Latent = Array;

    fn scale(&self, x: &Array, scale: f32) -> gen_core::Result<Array> {
        if scale == 1.0 {
            return Ok(x.clone());
        }
        ge(multiply(x, scalar(scale)))
    }

    fn add(&self, a: &Array, b: &Array) -> gen_core::Result<Array> {
        ge(add(a, b))
    }

    fn sub(&self, a: &Array, b: &Array) -> gen_core::Result<Array> {
        ge(subtract(a, b))
    }

    fn axpy(&self, a: f32, x: &Array, b: f32, y: &Array) -> gen_core::Result<Array> {
        // Byte-parity with apply_step's a_x==1 branch: emit `x + y·b` (multiply-by-one elided), so a
        // migrated engine's default step is bit-identical to the legacy `flow_match_euler_step`.
        let by = ge(multiply(y, scalar(b)))?;
        if a == 1.0 {
            return ge(add(x, &by));
        }
        let ax = ge(multiply(x, scalar(a)))?;
        ge(add(&ax, &by))
    }

    fn randn_like(&self, x: &Array, seed: u64, step: usize) -> gen_core::Result<Array> {
        // Reuse the legacy per-step subkey derivation (D6) — same trajectory determinism guarantees.
        // `StepRng::normal` already returns `crate::Result`, which `?` lifts into `gen_core::Error`.
        Ok(StepRng::new(seed).normal(x.shape(), step)?)
    }
}

// =================================================================================================
// LCM — diffusers `LCMScheduler` (epsilon prediction; SDXL world). Policy: gen_core LcmPolicy.
// =================================================================================================

/// Latent Consistency Model sampler. Predicts `x₀` from `eps`, applies the consistency boundary
/// scalings `c_skip`/`c_out`, and re-noises between steps. ~2–8 steps; CFG ≈ 1.
pub struct LcmSampler {
    policy: LcmPolicy,
    /// The compute dtype the model's forward expects (latents are cast to this in
    /// [`DiffusionSampler::scale_model_input`]); the step math runs f32.
    model_dtype: Dtype,
    rng: StepRng,
}

impl LcmSampler {
    /// Build for `num_steps` inference steps. `original_inference_steps` is diffusers' default 50.
    /// `seed` is the request seed driving the deterministic between-step re-noise (D6).
    pub fn new(
        sched: AlphaSchedule,
        num_train_timesteps: usize,
        original_inference_steps: usize,
        num_steps: usize,
        model_dtype: Dtype,
        seed: u64,
    ) -> Self {
        Self {
            policy: LcmPolicy::new(
                sched,
                num_train_timesteps,
                original_inference_steps,
                num_steps,
            ),
            model_dtype,
            rng: StepRng::new(seed),
        }
    }

    /// The deterministic consistency prediction at step `i` — diffusers' `denoised` (before the
    /// between-step re-noise). Used by the scheduler-isolation parity gate.
    pub fn denoised(&self, model_output: &Array, x: &Array, i: usize) -> Result<Array> {
        apply_step(
            &self.policy.denoised_coeffs(i),
            StepDtype::F32,
            x,
            model_output,
            i,
            &self.rng,
        )
    }
}

impl DiffusionSampler for LcmSampler {
    fn num_steps(&self) -> usize {
        self.policy.num_steps()
    }

    fn timestep(&self, i: usize) -> f32 {
        self.policy.coeffs(i).timestep
    }

    fn scale_model_input(&self, x: &Array, i: usize) -> Result<Array> {
        scale_input(self.policy.coeffs(i).c_in, self.model_dtype, x)
    }

    fn scale_initial_noise(&self, noise: &Array) -> Result<Array> {
        scale_initial(self.policy.init_noise_scale(), noise)
    }

    fn step(&self, model_output: &Array, x: &Array, i: usize) -> Result<Array> {
        apply_step(
            &self.policy.coeffs(i),
            self.policy.step_dtype(),
            x,
            model_output,
            i,
            &self.rng,
        )
    }
}

// =================================================================================================
// SDXL-Lightning — diffusers `EulerDiscreteScheduler(timestep_spacing="trailing")`. Deterministic.
// =================================================================================================

/// SDXL-Lightning sampler: trailing-spaced Euler. The latents live in diffusers' un-normalized
/// (σ-scaled) space; [`DiffusionSampler::scale_model_input`] divides by `√(σ²+1)` before the U-Net.
pub struct LightningSampler {
    policy: LightningPolicy,
    model_dtype: Dtype,
}

impl LightningSampler {
    /// Build for `num_steps` (2/4/8). Trailing-spaced timesteps + interpolated sigmas (policy layer).
    pub fn new(
        sched: &AlphaSchedule,
        num_train_timesteps: usize,
        num_steps: usize,
        model_dtype: Dtype,
    ) -> Self {
        Self {
            policy: LightningPolicy::new(sched, num_train_timesteps, num_steps),
            model_dtype,
        }
    }
}

impl DiffusionSampler for LightningSampler {
    fn num_steps(&self) -> usize {
        self.policy.num_steps()
    }

    fn timestep(&self, i: usize) -> f32 {
        self.policy.coeffs(i).timestep
    }

    fn scale_model_input(&self, x: &Array, i: usize) -> Result<Array> {
        // x · c_in (= 1/√(σ²+1)), then cast to the model's compute dtype.
        scale_input(self.policy.coeffs(i).c_in, self.model_dtype, x)
    }

    fn scale_initial_noise(&self, noise: &Array) -> Result<Array> {
        // latents = randn · init_noise_sigma (the largest sigma).
        scale_initial(self.policy.init_noise_scale(), noise)
    }

    fn step(&self, model_output: &Array, x: &Array, i: usize) -> Result<Array> {
        // Euler ε-pred step, gamma=0: `x + eps·(σ_{i+1} − σ_i)`, upcast to f32 (a_x=1, a_noise=0).
        apply_step(
            &self.policy.coeffs(i),
            self.policy.step_dtype(),
            x,
            model_output,
            i,
            &StepRng::new(0),
        )
    }
}

// =================================================================================================
// Hyper-SD — diffusers `TCDScheduler` (epsilon prediction). Policy: gen_core TcdPolicy.
// =================================================================================================

/// Hyper-SD sampler: Trajectory Consistency Distillation. Like LCM but steps to an intermediate
/// noise level `s = ⌊(1−η)·t_prev⌋` and (for `η>0`) re-noises across the `t_prev`/`s` gap.
pub struct TcdSampler {
    policy: TcdPolicy,
    model_dtype: Dtype,
    rng: StepRng,
}

impl TcdSampler {
    /// Build for `num_steps`. `original_inference_steps` is diffusers' default 50; `eta` is the
    /// stochasticity (`0.0` = deterministic; ByteDance's unified LoRA recommends ~`0.3`). `seed`
    /// drives the deterministic `η>0` re-noise (D6).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sched: AlphaSchedule,
        num_train_timesteps: usize,
        original_inference_steps: usize,
        num_steps: usize,
        eta: f32,
        model_dtype: Dtype,
        seed: u64,
    ) -> Self {
        Self {
            policy: TcdPolicy::new(
                sched,
                num_train_timesteps,
                original_inference_steps,
                num_steps,
                eta,
            ),
            model_dtype,
            rng: StepRng::new(seed),
        }
    }

    /// The deterministic noised prediction `x_s` at step `i` — diffusers' `pred_noised_sample`
    /// (before the `η>0` re-noise). Used by the scheduler-isolation parity gate.
    pub fn pred_noised(&self, model_output: &Array, x: &Array, i: usize) -> Result<Array> {
        apply_step(
            &self.policy.pred_noised_coeffs(i),
            StepDtype::F32,
            x,
            model_output,
            i,
            &self.rng,
        )
    }
}

impl DiffusionSampler for TcdSampler {
    fn num_steps(&self) -> usize {
        self.policy.num_steps()
    }

    fn timestep(&self, i: usize) -> f32 {
        self.policy.coeffs(i).timestep
    }

    fn scale_model_input(&self, x: &Array, i: usize) -> Result<Array> {
        scale_input(self.policy.coeffs(i).c_in, self.model_dtype, x)
    }

    fn scale_initial_noise(&self, noise: &Array) -> Result<Array> {
        scale_initial(self.policy.init_noise_scale(), noise)
    }

    fn step(&self, model_output: &Array, x: &Array, i: usize) -> Result<Array> {
        apply_step(
            &self.policy.coeffs(i),
            self.policy.step_dtype(),
            x,
            model_output,
            i,
            &self.rng,
        )
    }
}

// =================================================================================================
// Flow-match — the rectified-flow world (FLUX.1 / Qwen-Image). Policy: gen_core FlowMatchPolicy.
// =================================================================================================

/// A flow-match (rectified-flow) Euler sampler driven by a precomputed sigma schedule. The schedule
/// is built by the model family (FLUX's `build_linear_sigmas`, Qwen's `qwen_scheduler` and its
/// Lightning builder), so this sampler is family-neutral — it owns only the flow-match update. The
/// model is velocity-prediction, the latents stay f32, and the prior is unit noise.
pub struct FlowMatchSampler {
    policy: FlowMatchPolicy,
}

impl FlowMatchSampler {
    /// Build from a precomputed sigma schedule (length `num_steps + 1`, trailing `0.0`). A schedule
    /// needs at least one step + the terminal `0`; this is debug-asserted here (the downstream `step`
    /// indexing requires it) — previously the doc promised a panic the code never enforced (F-086).
    /// FLUX/Qwen feed the raw sigma as the model timestep ([`TimestepConvention::Sigma`]).
    pub fn new(sigmas: Vec<f32>) -> Self {
        debug_assert!(
            sigmas.len() >= 2,
            "FlowMatchSampler::new: schedule needs >= 2 entries (>=1 step + terminal 0), got {}",
            sigmas.len()
        );
        Self {
            policy: FlowMatchPolicy::new(sigmas, TimestepConvention::Sigma),
        }
    }

    /// The schedule sigma at step `i` (length `num_steps + 1`, trailing `0.0`). For flow-match this
    /// equals [`DiffusionSampler::timestep`]; img2img seeds its noise blend at `sigma(start_step)`.
    pub fn sigma(&self, i: usize) -> f32 {
        self.policy.sigma_at_node(i)
    }
}

impl DiffusionSampler for FlowMatchSampler {
    fn num_steps(&self) -> usize {
        self.policy.num_steps()
    }

    fn timestep(&self, i: usize) -> f32 {
        self.policy.coeffs(i).timestep
    }

    fn scale_initial_noise(&self, noise: &Array) -> Result<Array> {
        // init_noise_scale = 1 → identity cast to f32 (FLUX seeds its own noise via `create_noise`).
        scale_initial(self.policy.init_noise_scale(), noise)
    }

    fn step(&self, model_output: &Array, x: &Array, i: usize) -> Result<Array> {
        // Forward Euler on the velocity field: `x + v·(σ_{i+1} − σ_i)` (a_x=1, a_noise=0 → the
        // byte-parity branch, computed in the latents' dtype — identical to `FlowMatchEuler::step`
        // and the original `flow_match_euler_step`, F-009).
        apply_step(
            &self.policy.coeffs(i),
            self.policy.step_dtype(),
            x,
            model_output,
            i,
            &StepRng::new(0),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sdxl_sched() -> AlphaSchedule {
        AlphaSchedule::scaled_linear(1000, 0.00085, 0.012).unwrap()
    }

    fn scalar1(v: f32) -> Array {
        Array::from_slice(&[v], &[1])
    }
    fn val(a: &Array) -> f32 {
        a.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>()[0]
    }

    #[test]
    fn samplers_report_step_count() {
        let lcm = LcmSampler::new(sdxl_sched(), 1000, 50, 4, Dtype::Float32, 0);
        assert_eq!(lcm.num_steps(), 4);
        let light = LightningSampler::new(&sdxl_sched(), 1000, 2, Dtype::Float32);
        assert_eq!(light.num_steps(), 2);
        let tcd = TcdSampler::new(sdxl_sched(), 1000, 50, 8, 0.0, Dtype::Float32, 0);
        assert_eq!(tcd.num_steps(), 8);
    }

    // The per-step tensor application reproduces the diffusers scalars via the neutral coefficients
    // (the same references the gen_core::sampling policy goldens assert, now through MLX arrays).
    #[test]
    fn lcm_step0_denoised_matches_diffusers() {
        let s = LcmSampler::new(sdxl_sched(), 1000, 50, 4, Dtype::Float32, 0);
        let d = s.denoised(&scalar1(0.7), &scalar1(0.3), 0).unwrap();
        assert!((val(&d) - (-5.835_607)).abs() < 1e-3, "got {}", val(&d));
    }

    #[test]
    fn lightning_step0_matches_diffusers() {
        let s = LightningSampler::new(&sdxl_sched(), 1000, 4, Dtype::Float32);
        let scaled = s.scale_model_input(&scalar1(0.3), 0).unwrap();
        assert!(
            (val(&scaled) - 0.020_479_47).abs() < 1e-4,
            "scaled {}",
            val(&scaled)
        );
        let prev = s.step(&scalar1(0.7), &scalar1(0.3), 0).unwrap();
        assert!(
            (val(&prev) - (-7.073_041)).abs() < 1e-3,
            "prev {}",
            val(&prev)
        );
    }

    #[test]
    fn tcd_eta0_step0_pred_noised_matches_diffusers() {
        let s = TcdSampler::new(sdxl_sched(), 1000, 50, 4, 0.0, Dtype::Float32, 0);
        let pn = s.pred_noised(&scalar1(0.7), &scalar1(0.3), 0).unwrap();
        assert!((val(&pn) - (-0.651_963_8)).abs() < 1e-4, "got {}", val(&pn));
    }

    // Flow-match (FLUX): the sampler must reproduce the proven inline FLUX loop `x + v·(σ_{i+1}−σ_i)`
    // exactly, with `timestep(i)=σ_i` and `num_steps = len-1`. Schnell-style 4-step linear sigmas.
    #[test]
    fn flow_match_step_matches_inline_euler() {
        let sigmas = vec![1.0_f32, 0.75, 0.5, 0.25, 0.0];
        let s = FlowMatchSampler::new(sigmas.clone());
        assert_eq!(s.num_steps(), 4);
        for (i, &sig) in sigmas.iter().take(4).enumerate() {
            assert_eq!(s.timestep(i), sig);
        }
        // step 0: x=0.3, v=0.7 → 0.3 + 0.7·(0.75−1.0) = 0.125 (the exact inline-loop arithmetic).
        let out = s.step(&scalar1(0.7), &scalar1(0.3), 0).unwrap();
        assert!((val(&out) - 0.125).abs() < 1e-6, "got {}", val(&out));
        // last step integrates to σ=0: dt = 0.0 − 0.25 = −0.25.
        let last = s.step(&scalar1(0.4), &scalar1(0.2), 3).unwrap();
        assert!(
            (val(&last) - (0.2 - 0.1)).abs() < 1e-6,
            "got {}",
            val(&last)
        );
    }

    #[test]
    fn flow_match_initial_noise_is_unit_identity_f32() {
        let s = FlowMatchSampler::new(vec![1.0_f32, 0.5, 0.0]);
        let n = Array::from_slice(&[0.3_f32, -0.7, 1.1], &[3]);
        let scaled = s.scale_initial_noise(&n).unwrap();
        // init_noise_sigma = 1 → identity (×1), dtype f32.
        assert_eq!(scaled.dtype(), Dtype::Float32);
        let got = scaled.as_slice::<f32>();
        for (a, b) in got.iter().zip([0.3_f32, -0.7, 1.1]) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    // --- Unified framework backend: MlxLatentOps (epic 7114 P2, sc-7118) ---------------------------

    use gen_core::sampling::{
        build_flow_sigmas, compute_mu, denoise, image_seq_len, Euler, FlowModelSampling, Sampler,
    };
    use mlx_rs::ops::eq;

    fn arr(v: &[f32]) -> Array {
        Array::from_slice(v, &[v.len() as i32])
    }
    fn arrays_eq(a: &Array, b: &Array) -> bool {
        eq(a, b).unwrap().all(None).unwrap().item::<bool>()
    }
    /// A reference flow velocity model `v = 0.3·x + 0.1` over MLX (matches the gen-core byte-equiv).
    fn stub_velocity(xin: &Array) -> Result<Array> {
        Ok(add(&multiply(xin, scalar(0.3))?, scalar(0.1))?)
    }

    #[test]
    fn mlx_latent_ops_scale_add_sub() {
        let ops = MlxLatentOps;
        let a = arr(&[1.0, 2.0, 3.0]);
        let b = arr(&[0.5, -1.0, 4.0]);
        assert!(arrays_eq(
            &ops.scale(&a, 2.0).unwrap(),
            &arr(&[2.0, 4.0, 6.0])
        ));
        // scale by 1.0 is a byte-identical clone (no kernel).
        assert!(arrays_eq(&ops.scale(&a, 1.0).unwrap(), &a));
        assert!(arrays_eq(&ops.add(&a, &b).unwrap(), &arr(&[1.5, 1.0, 7.0])));
        assert!(arrays_eq(
            &ops.sub(&a, &b).unwrap(),
            &arr(&[0.5, 3.0, -1.0])
        ));
    }

    #[test]
    fn mlx_axpy_a1_is_byte_identical_to_legacy_branch() {
        // axpy(1.0, x, b, y) must equal `x + y·b` exactly (the apply_step byte-parity branch).
        let ops = MlxLatentOps;
        let x = arr(&[0.3, -1.2, 2.5]);
        let y = arr(&[0.7, 0.1, -0.4]);
        let got = ops.axpy(1.0, &x, 0.25, &y).unwrap();
        let want = add(&x, multiply(&y, scalar(0.25)).unwrap()).unwrap();
        assert!(arrays_eq(&got, &want), "axpy a=1 not byte-identical");
        // General a: 2·x + (−3)·y.
        let got2 = ops.axpy(2.0, &x, -3.0, &y).unwrap();
        let want2 = add(
            multiply(&x, scalar(2.0)).unwrap(),
            multiply(&y, scalar(-3.0)).unwrap(),
        )
        .unwrap();
        assert!(arrays_eq(&got2, &want2));
    }

    #[test]
    fn mlx_randn_like_is_deterministic_shaped_and_seed_keyed() {
        let ops = MlxLatentOps;
        let x = arr(&[0.0, 0.0, 0.0, 0.0, 0.0]);
        let a = ops.randn_like(&x, 42, 0).unwrap();
        assert_eq!(a.shape(), &[5]);
        assert!(arrays_eq(&a, &ops.randn_like(&x, 42, 0).unwrap()));
        assert!(!arrays_eq(&a, &ops.randn_like(&x, 42, 1).unwrap()));
        assert!(!arrays_eq(&a, &ops.randn_like(&x, 43, 0).unwrap()));
    }

    #[test]
    fn mlx_unified_euler_matches_legacy_flowmatch() {
        // The N1 proof on the real backend: gen-core Euler over MlxLatentOps + a FLOW ModelSampling
        // reproduces the legacy FlowMatchSampler trajectory (same stub velocity, same sigmas).
        let ops = MlxLatentOps;
        let sigmas = build_flow_sigmas(8, compute_mu(image_seq_len(1024, 1024), 8));
        let x_init = arr(&[0.3, -1.1, 2.0, 0.05, -0.4, 1.7]);

        // Legacy FlowMatchSampler path (the byte-parity apply_step).
        let legacy_sampler = FlowMatchSampler::new(sigmas.clone());
        let mut legacy = x_init.clone();
        for i in 0..legacy_sampler.num_steps() {
            let v = stub_velocity(&legacy).unwrap(); // c_in=1 -> model input = x
            legacy = legacy_sampler.step(&v, &legacy, i).unwrap();
        }

        // Unified Euler over MlxLatentOps.
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let mut dn = |x: &Array, s: f32| {
            denoise(&ops, &ms, x, s, |xin, _t| {
                stub_velocity(xin).map_err(Into::into)
            })
        };
        let unified = Euler
            .sample(&ops, &mut dn, x_init.clone(), &sigmas, 0)
            .unwrap();

        let (lg, un) = (legacy.as_slice::<f32>(), unified.as_slice::<f32>());
        let max = lg
            .iter()
            .zip(un)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max < 1e-4,
            "unified Euler diverged from legacy FlowMatch: {max:e}"
        );
    }

    #[test]
    fn mlx_drives_every_curated_solver_to_finite_output() {
        // The P2 deliverable: every gen-core curated sampler runs end-to-end over mlx_rs::Array.
        let ops = MlxLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let sigmas = build_flow_sigmas(6, compute_mu(image_seq_len(512, 512), 6));
        let x_init = arr(&[0.2, -0.5, 1.0, 0.3]);
        for name in [
            "euler",
            "euler_ancestral",
            "heun",
            "dpmpp_2m",
            "dpmpp_sde",
            "uni_pc",
            "lcm",
            "ddim",
        ] {
            let sampler =
                gen_core::sampling::sampler_by_name::<MlxLatentOps>(name).expect("known solver");
            let mut dn = |x: &Array, s: f32| {
                denoise(&ops, &ms, x, s, |xin, _t| {
                    stub_velocity(xin).map_err(Into::into)
                })
            };
            let out = sampler
                .sample(&ops, &mut dn, x_init.clone(), &sigmas, 7)
                .unwrap();
            assert!(
                out.as_slice::<f32>().iter().all(|v| v.is_finite()),
                "{name} produced non-finite output"
            );
        }
    }
}
