//! Qwen-Image flow-match Lightning schedule (sc-2909), built on the core
//! [`mlx_gen::FlowMatchSampler`] seam (deduped onto that one wrapper in sc-2950).
//!
//! Qwen-Image is **flow-match**, so the DDPM `alphas_cumprod`-world acceleration samplers shipped
//! with sc-2769 (`LcmSampler`/`LightningSampler`/`TcdSampler`) do not apply. Both the production
//! schedule (`qwen_scheduler`, the fork's `LinearScheduler`) and the few-step **Lightning** schedule
//! drive the generic denoise loop through the same core [`mlx_gen::FlowMatchSampler`] (forward Euler over a
//! precomputed sigma schedule, `x + v·(σ_{i+1} − σ_i)`). Only the *schedule construction* differs per
//! family — FLUX builds its `linspace(1, 1/n, n)` mu-shift sigmas, Qwen builds the two below — so this
//! module owns just the Qwen-specific sigma builders; the wrapper type itself lives in core.
//!
//! - Production: `FlowMatchSampler::new(qwen_scheduler(..).sigmas)` (resolution-dependent μ).
//! - Lightning: [`lightning`] builds the **official lightx2v Qwen-Image-Lightning** schedule,
//!   reproducing diffusers' `FlowMatchEulerDiscreteScheduler` under that LoRA's model-card config: a
//!   static flow-match shift of `3.0` (`base_shift = max_shift = ln 3`, which collapses dynamic
//!   shifting to a resolution-independent constant) with **no terminal rescale** (`shift_terminal =
//!   None`), over the **full** sigma span `linspace(1, 1/num_train_timesteps, n)` — NOT the mflux
//!   `linspace(1, 1/n, n)` of [`crate::pipeline::qwen_scheduler`]. The matching distillation LoRA
//!   (e.g. `lightx2v/Qwen-Image-Lightning`) must be supplied via `spec.adapters`; the CFG-distilled
//!   LoRAs run CFG-off (a single forward). Validated bit-exact-ish vs diffusers
//!   (`tests/lightning_parity.rs`, `tools/dump_qwen_lightning_golden.py`).
//!
//! Timestep convention: Qwen feeds the **raw sigma** as the model timestep — the transformer's
//! `QwenTimesteps` time-proj scales by ×1000 internally (so `embed(sigma·1000)` matches diffusers'
//! `timesteps = sigmas·1000` fed to a scale-1 embedding). The core [`mlx_gen::FlowMatchSampler`]'s
//! `timestep` already returns `sigmas[i]`, exactly what the pixel-parity production loop passes.

pub use mlx_gen::FlowMatchSampler;

/// The official lightx2v Qwen-Image-Lightning flow-match shift (`exp(μ)`, μ = `ln 3`). The model
/// card sets `base_shift = max_shift = ln 3`, so the per-resolution dynamic shift collapses to this
/// constant; `shift_terminal = None` means no terminal-sigma rescale (unlike the production
/// `qwen_scheduler`'s 0.02).
pub const LIGHTNING_SHIFT: f32 = 3.0;

/// Flow-match training timesteps (diffusers `num_train_timesteps`) — the Lightning sigma span runs
/// down to `1/LIGHTNING_NUM_TRAIN_TIMESTEPS`, the full diffusers minimum (not the mflux `1/n`).
const LIGHTNING_NUM_TRAIN_TIMESTEPS: f32 = 1000.0;

/// Build the few-step **Lightning** [`mlx_gen::FlowMatchSampler`] for `num_steps` (typically 4 or 8,
/// matching the loaded distillation LoRA): the official diffusers Lightning sigmas (see
/// `lightning_sigmas`), wrapped in the core flow-match Euler sampler.
pub fn lightning(num_steps: usize) -> FlowMatchSampler {
    FlowMatchSampler::new(lightning_sigmas(num_steps))
}

/// Build the Lightning sigmas, reproducing diffusers' `FlowMatchEulerDiscreteScheduler.set_timesteps`
/// under the official config: exponential time-shift `exp(μ)/(exp(μ) + (1/σ − 1))` with `exp(μ) =
/// 3.0`, applied over `linspace(1.0, 1/num_train_timesteps, n)`, then a trailing `0.0`. The `1/1000`
/// floor (vs the production schedule's `1/n`) is the whole difference — proven bit-exact vs diffusers
/// in `tests/lightning_parity.rs` (e.g. 4-step → `[1.0, 0.857.., 0.601.., 0.00299.., 0.0]`).
pub fn lightning_sigmas(num_steps: usize) -> Vec<f32> {
    let n = num_steps.max(1);
    let e = LIGHTNING_SHIFT; // exp(μ)
    let sigma_min = 1.0 / LIGHTNING_NUM_TRAIN_TIMESTEPS;
    let mut sigmas: Vec<f32> = (0..n)
        .map(|i| {
            // linspace(1.0, sigma_min, n)
            let s = if n == 1 {
                1.0
            } else {
                1.0 + (sigma_min - 1.0) * (i as f32) / ((n - 1) as f32)
            };
            e / (e + (1.0 / s - 1.0))
        })
        .collect();
    sigmas.push(0.0);
    sigmas
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::qwen_scheduler;
    use mlx_gen::DiffusionSampler;

    #[test]
    fn lightning_4step_sigmas_match_diffusers() {
        // The official recipe as realized in diffusers FlowMatchEulerDiscreteScheduler (shift=3.0,
        // shift_terminal=None) over linspace(1, 1/1000, 4): bit-exact values from
        // `tools/dump_qwen_lightning_golden.py` (the tight cross-impl gate is `tests/lightning_parity.rs`).
        let s = lightning(4);
        assert_eq!(s.num_steps(), 4);
        let expected = [1.0_f32, 0.857_326_5, 0.600_719_4, 0.002_994_012, 0.0];
        for (i, want) in expected.iter().enumerate() {
            assert!(
                (s.sigma(i) - want).abs() < 1e-5,
                "lightning sigma[{i}] = {} want {want}",
                s.sigma(i)
            );
        }
        // No terminal rescale: the span runs to the diffusers 1/1000 floor (≈0.003), NOT 0.02.
        assert!(s.sigma(3) < 0.01);
    }

    #[test]
    fn lightning_8step_sigmas_match_diffusers() {
        let s = lightning(8);
        assert_eq!(s.num_steps(), 8);
        let expected = [
            1.0_f32,
            0.947_426_5,
            0.882_498_3,
            0.800_279_9,
            0.692_804_4,
            0.546_321_5,
            0.334_886_8,
            0.002_994_012,
            0.0,
        ];
        for (i, want) in expected.iter().enumerate() {
            assert!(
                (s.sigma(i) - want).abs() < 1e-5,
                "lightning sigma[{i}] = {} want {want}",
                s.sigma(i)
            );
        }
    }

    /// F-121: `lightning(n)` takes no width/height (μ is the constant `ln 3`, base_shift == max_shift),
    /// unlike the production `qwen_scheduler` whose μ comes from the latent sequence length. Test the
    /// CONTRAST that gives "resolution-independent" meaning: the production schedule genuinely changes
    /// across resolutions, while Lightning is the single schedule that differs from both. (The old test
    /// compared `lightning(8)` to itself — only proving determinism.)
    #[test]
    fn lightning_is_resolution_independent() {
        let light = lightning(8);
        let light_sigmas: Vec<f32> = (0..=light.num_steps()).map(|i| light.sigma(i)).collect();

        let lo = qwen_scheduler(8, 512, 512).sigmas;
        let hi = qwen_scheduler(8, 1024, 1024).sigmas;

        assert_ne!(
            lo, hi,
            "production qwen_scheduler must depend on resolution"
        );
        assert_ne!(
            light_sigmas, lo,
            "lightning must differ from the 512² production schedule"
        );
        assert_ne!(
            light_sigmas, hi,
            "lightning must differ from the 1024² production schedule"
        );
    }

    #[test]
    fn timestep_is_raw_sigma() {
        // The model timestep is the raw sigma (the time-proj scales ×1000), matching the production
        // loop; NOT FlowMatchEuler::timestep's `1 - sigma` (used by other families).
        let s = lightning(4);
        for i in 0..4 {
            assert_eq!(s.timestep(i), s.sigma(i));
        }
    }

    #[test]
    fn wrapping_production_scheduler_preserves_sigmas() {
        // Routing the production `qwen_scheduler` through the core wrapper must expose the identical
        // schedule (the base path stays bit-for-bit unchanged).
        let sched = qwen_scheduler(8, 1024, 1024);
        let sigmas = sched.sigmas.clone();
        let s = FlowMatchSampler::new(sched.sigmas);
        assert_eq!(s.num_steps(), 8);
        for (i, want) in sigmas.iter().enumerate() {
            assert_eq!(s.sigma(i), *want);
            if i < 8 {
                assert_eq!(s.timestep(i), *want);
            }
        }
    }
}
