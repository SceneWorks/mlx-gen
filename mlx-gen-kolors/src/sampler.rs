//! Kolors scheduler (sc-3094) — a faithful port of diffusers `EulerDiscreteScheduler` with the
//! Kolors config: `scaled_linear` betas (β₀=0.00085, β₁=0.014), **num_train_timesteps=1100**,
//! `timestep_spacing="leading"`, `steps_offset=1`, `interpolation_type="linear"`,
//! `final_sigmas_type="zero"`, epsilon prediction, `s_churn=0`.
//!
//! This is the non-ancestral Euler — **no per-step RNG** — so a denoise run is fully deterministic
//! given the initial latents, which is what makes pixel parity vs `KolorsPipeline` achievable. It
//! differs from the core [`mlx_gen::sampler::LightningSampler`] (which is the same scheduler with
//! `timestep_spacing="trailing"`) only in the timestep selection and `init_noise_sigma`:
//!  - **leading**: `timesteps = (arange(0,N)·(1100//N)).round()[::-1] + steps_offset`;
//!  - leading ⇒ `init_noise_sigma = (max_sigma² + 1)^0.5` (trailing/linspace use `max_sigma`).
//!
//! The latents live in diffusers' σ-scaled space; [`DiffusionSampler::scale_model_input`] divides by
//! `√(σ²+1)` before the U-Net, and the Euler step is `x + eps·(σ_next − σ)`.

use mlx_rs::ops::{divide, multiply};
use mlx_rs::{Array, Dtype};

use mlx_gen::array::scalar;
use mlx_gen::sampler::AlphaSchedule;
use mlx_gen::{DiffusionSampler, Error, Result};

/// Kolors' `num_train_timesteps` — the length of the `scaled_linear` schedule the sampler interpolates
/// over. `num_steps` must lie in `1..=NUM_TRAIN_TIMESTEPS`: `0` divides by zero below, and a value above
/// it makes `step_ratio == 0` so every timestep collapses to a single value.
pub const NUM_TRAIN_TIMESTEPS: usize = 1100;

/// Kolors' `scaled_linear` β schedule endpoints (diffusers `EulerDiscreteScheduler` config: β₀ =
/// 0.00085, β₁ = 0.014). Single source of truth (F-083) — the native sampler
/// ([`KolorsEulerSampler::kolors`]), the curated-path `DiscreteModelSampling` (`model.rs`), and the
/// trainer's DDPM schedule (`training.rs`) all build the *same* `scaled_linear` betas over
/// [`NUM_TRAIN_TIMESTEPS`], so hoist the literals here rather than re-spelling them at each site.
pub const BETA_START: f32 = 0.00085;
pub const BETA_END: f32 = 0.014;

/// Kolors' EulerDiscrete (leading) sampler over the 1100-step `scaled_linear` schedule.
pub struct KolorsEulerSampler {
    /// Interpolated sigmas at the (effective) timesteps, length `num_steps + 1` (trailing `0.0`).
    /// For img2img this is the schedule **sliced** at the strength-derived start (`begin_index`).
    sigmas: Vec<f32>,
    /// The (float) leading timesteps fed to the U-Net, length `num_steps`.
    timesteps: Vec<f32>,
    init_noise_sigma: f32,
    /// The σ at the img2img start (`sigmas[begin_index]`) — what diffusers' `add_noise` scales the
    /// noise by when seeding the init latents. Equals `sigmas[0]` after the img2img slice; for the
    /// full txt2img schedule it is just the max σ and is unused (txt2img seeds via init noise).
    start_sigma: f32,
    model_dtype: Dtype,
}

impl KolorsEulerSampler {
    /// Kolors defaults: `num_train_timesteps=1100`, β₀=0.00085, β₁=0.014, `steps_offset=1`.
    pub fn kolors(num_steps: usize, model_dtype: Dtype) -> Result<Self> {
        Self::new(
            NUM_TRAIN_TIMESTEPS,
            BETA_START,
            BETA_END,
            1,
            num_steps,
            model_dtype,
        )
    }

    /// The img2img variant of [`Self::kolors`]: build the full `num_steps` schedule, then slice it at
    /// the strength-derived start exactly as diffusers' `KolorsImg2ImgPipeline.get_timesteps` +
    /// `set_begin_index` do. With `init_timestep = min(int(num_steps·strength), num_steps)` and
    /// `t_start = num_steps − init_timestep`, the run uses `timesteps[t_start..]` / `sigmas[t_start..]`
    /// and seeds the init latents with [`Self::add_noise`] at `σ = sigmas[t_start]`. `strength ≤
    /// 1/num_steps` ⇒ 0 effective steps ⇒ the (un-noised) init is returned unchanged.
    pub fn kolors_img2img(num_steps: usize, strength: f32, model_dtype: Dtype) -> Result<Self> {
        let full = Self::kolors(num_steps, model_dtype)?;
        let init_timestep = ((num_steps as f32 * strength) as usize).min(num_steps);
        let t_start = num_steps - init_timestep;
        // sigmas has length num_steps+1 (trailing 0); slicing from t_start keeps that trailing 0.
        let sigmas = full.sigmas[t_start..].to_vec();
        let timesteps = full.timesteps[t_start..].to_vec();
        let start_sigma = sigmas[0];
        Ok(Self {
            sigmas,
            timesteps,
            init_noise_sigma: full.init_noise_sigma,
            start_sigma,
            model_dtype,
        })
    }

    /// Seed img2img: noise the clean (VAE-encoded, scaled) init latents at the start σ —
    /// diffusers' `EulerDiscreteScheduler.add_noise` with `begin_index = t_start`, which is the
    /// **raw** `x₀ + noise·σ` (the latents stay in un-normalized σ-space; `scale_model_input`
    /// normalizes before each U-Net call). Draws no RNG — the caller supplies `noise`.
    pub fn add_noise(&self, x0: &Array, noise: &Array) -> Result<Array> {
        use mlx_rs::ops::add;
        let x0 = x0.as_dtype(Dtype::Float32)?;
        let noise = noise.as_dtype(Dtype::Float32)?;
        Ok(add(&x0, &multiply(&noise, scalar(self.start_sigma))?)?)
    }

    /// The per-node EDM/k-diffusion σ schedule of this baked run — `[σ(t₀), …, σ(t_{n-1}), 0]`
    /// (`self.sigmas`, length `num_steps + 1`, strictly descending to the terminal `0`). Used by the
    /// PiD `from_ldm` early-stop (epic 7840, sc-8049) to resolve the variance-preserving capture plan
    /// ([`mlx_gen::gen_core::sampling::vp_capture_plan`]) against the *exact* leading-Euler schedule this
    /// run denoises, so the `keep` index and the achieved degrade σ agree with the truncated trajectory.
    /// Kolors stores RAW variance-exploding latents (`x0 + σ·ε`) at every node (plain k-diffusion
    /// `x + eps·dt`; `scale_model_input` only divides the U-Net input by `√(σ²+1)`), so the captured
    /// latent is mapped into the student's VP frame by the plan's rescale before decode.
    pub fn edm_sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// Truncate the baked schedule to its first `run_steps` steps — the PiD `from_ldm` early-stop
    /// (epic 7840, sc-8049). This keeps the FULL schedule's spacing and stops after `run_steps` denoise
    /// steps, leaving `x_k` at node `run_steps` (EDM σ = `edm_sigmas()[run_steps]`) — unlike rebuilding
    /// with fewer steps, which would re-space the whole leading-Euler schedule. `run_steps >=
    /// self.timesteps.len()` is a no-op (`Vec::truncate` clamps, so neither truncate panics), the clean
    /// full-denoise path. `timesteps` keeps `num_steps` entries and `sigmas` keeps `num_steps + 1`
    /// (the trailing `0`), matching the invariant `step`/`scale_model_input` index into.
    pub fn truncate_to(mut self, run_steps: usize) -> Self {
        self.timesteps.truncate(run_steps);
        self.sigmas.truncate(run_steps + 1);
        self
    }

    pub fn new(
        num_train_timesteps: usize,
        beta_start: f32,
        beta_end: f32,
        steps_offset: i64,
        num_steps: usize,
        model_dtype: Dtype,
    ) -> Result<Self> {
        // Defensive: `num_steps == 0` divides by zero at `num_train_timesteps / num_steps` below
        // (F-124). The request boundary rejects this, but guard here so any caller gets a typed error.
        if num_steps == 0 {
            return Err(Error::Msg("kolors sampler: num_steps must be >= 1".into()));
        }
        let sched = AlphaSchedule::scaled_linear(num_train_timesteps, beta_start, beta_end);
        // Per-train-step Karras sigma √((1-ᾱ)/ᾱ) (the `alphas_cumprod` field is public).
        let full: Vec<f64> = sched
            .alphas_cumprod
            .iter()
            .map(|&acp| {
                let a = acp as f64;
                ((1.0 - a) / a).sqrt()
            })
            .collect();

        // leading: timesteps = (arange(0,N)·step_ratio).round()[::-1] + steps_offset.
        let step_ratio = (num_train_timesteps / num_steps) as i64;
        let timesteps: Vec<f32> = (0..num_steps)
            .rev()
            .map(|j| ((j as i64 * step_ratio) + steps_offset) as f32)
            .collect();

        // np.interp(timesteps, arange(0, N), full), then append 0 (final_sigmas_type="zero").
        let interp = |t: f32| -> f32 {
            let tt = (t as f64).clamp(0.0, (num_train_timesteps - 1) as f64);
            let lo = tt.floor() as usize;
            let hi = (lo + 1).min(num_train_timesteps - 1);
            let frac = tt - lo as f64;
            (full[lo] * (1.0 - frac) + full[hi] * frac) as f32
        };
        let mut sigmas: Vec<f32> = timesteps.iter().map(|&t| interp(t)).collect();
        let max_sigma = sigmas.iter().copied().fold(0.0_f32, f32::max);
        sigmas.push(0.0);

        Ok(Self {
            sigmas,
            timesteps,
            // leading spacing ⇒ init_noise_sigma = (max_sigma² + 1)^0.5.
            init_noise_sigma: (max_sigma * max_sigma + 1.0).sqrt(),
            // The full txt2img schedule starts at the max σ; the img2img slice overrides this.
            start_sigma: max_sigma,
            model_dtype,
        })
    }
}

impl DiffusionSampler for KolorsEulerSampler {
    fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    fn timestep(&self, i: usize) -> f32 {
        self.timesteps[i]
    }

    fn scale_model_input(&self, x: &Array, i: usize) -> Result<Array> {
        let sigma = self.sigmas[i] as f64;
        let scaled = divide(x, scalar(((sigma * sigma + 1.0).sqrt()) as f32))?;
        Ok(scaled.as_dtype(self.model_dtype)?)
    }

    fn scale_initial_noise(&self, noise: &Array) -> Result<Array> {
        Ok(multiply(
            &noise.as_dtype(Dtype::Float32)?,
            scalar(self.init_noise_sigma),
        )?)
    }

    fn step(&self, model_output: &Array, x: &Array, i: usize) -> Result<Array> {
        use mlx_rs::ops::add;
        // Euler, epsilon prediction, gamma=0: prev = x + eps·(σ_next − σ). (diffusers upcasts to f32.)
        let eps = model_output.as_dtype(Dtype::Float32)?;
        let x = x.as_dtype(Dtype::Float32)?;
        let dt = self.sigmas[i + 1] - self.sigmas[i];
        Ok(add(&x, &multiply(&eps, scalar(dt))?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edm_sigmas_descend_to_zero_with_node_count() {
        // The from_ldm VP capture (sc-8049) resolves `vp_capture_plan` against this per-node σ schedule
        // — one σ per node, descending, trailing terminal 0 (mirrors the SDXL anchor's assertion).
        let s = KolorsEulerSampler::kolors(8, Dtype::Float32).unwrap();
        let edm = s.edm_sigmas();
        assert_eq!(
            edm.len(),
            s.num_steps() + 1,
            "one σ per node incl. terminal 0"
        );
        assert_eq!(edm.len(), 9);
        assert_eq!(*edm.last().unwrap(), 0.0, "terminal node σ=0");
        assert!(edm.windows(2).all(|w| w[0] >= w[1]), "descending");
        assert!(edm[0] > 0.0, "starts at a positive σ");
    }

    #[test]
    fn truncate_to_preserves_full_schedule_spacing() {
        // The from_ldm early-stop truncates the FULL leading-Euler schedule (keeping its spacing)
        // rather than rebuilding with fewer steps (which would re-space it) — sc-8049.
        let full_edm = KolorsEulerSampler::kolors(8, Dtype::Float32)
            .unwrap()
            .edm_sigmas()
            .to_vec();
        let trunc = KolorsEulerSampler::kolors(8, Dtype::Float32)
            .unwrap()
            .truncate_to(3);
        assert_eq!(trunc.num_steps(), 3, "3 steps kept");
        let trunc_edm = trunc.edm_sigmas();
        assert_eq!(trunc_edm.len(), 4, "3 steps → 4 nodes (incl. trailing 0)");
        for i in 0..4 {
            assert!(
                (trunc_edm[i] - full_edm[i]).abs() < 1e-6,
                "node {i}: truncation must keep the full schedule's leading nodes"
            );
        }
        // A natively-built 3-step schedule re-spaces (leading spacing → different interior node), so its
        // interior node differs from the truncation.
        let native3 = KolorsEulerSampler::kolors(3, Dtype::Float32)
            .unwrap()
            .edm_sigmas()
            .to_vec();
        assert!(
            (trunc_edm[1] - native3[1]).abs() > 1e-4,
            "truncation must NOT equal a re-spaced 3-step schedule"
        );
    }

    #[test]
    fn truncate_to_full_or_larger_is_a_noop() {
        // `run_steps >= timesteps.len()` leaves the schedule intact (the clean full-denoise path); the
        // `sigmas.truncate(run_steps + 1)` must not panic past the end (Vec::truncate clamps).
        let full = KolorsEulerSampler::kolors(4, Dtype::Float32)
            .unwrap()
            .edm_sigmas()
            .to_vec();
        for rs in [4usize, 5, 100] {
            let t = KolorsEulerSampler::kolors(4, Dtype::Float32)
                .unwrap()
                .truncate_to(rs);
            assert_eq!(t.num_steps(), 4, "run_steps {rs}: no-op keeps all 4 steps");
            assert_eq!(
                t.edm_sigmas(),
                full.as_slice(),
                "run_steps {rs}: schedule unchanged"
            );
        }
    }
}
