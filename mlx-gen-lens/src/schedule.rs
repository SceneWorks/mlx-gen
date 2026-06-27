//! Lens sampling schedule + CFG (sc-3170). The schedule is the core [`mlx_gen::FlowMatchEuler`]
//! verbatim: the Lens `compute_empirical_mu` is **byte-identical** to the core
//! [`mlx_gen::scheduler::compute_mu`] (same calibrated constants + `>4300` branch), and the Lens
//! `np.linspace(1, 1/n, n)` → `set_timesteps(sigmas, mu)` dynamic-shift is exactly
//! `FlowMatchEuler::new(num_steps, mu)`. Only two pieces are Lens-specific:
//!
//! 1. **Timestep convention** — Lens feeds the transformer the *shifted sigma* directly (the
//!    reference `timestep / 1000`, where `scheduler.timesteps = shifted_sigma · 1000`), **not** the
//!    `1 − sigma` other mlx-gen DiTs use. [`timesteps`] returns those shifted sigmas.
//! 2. **Norm-rescaled CFG** — `comb = uncond + g·(cond − uncond)`, then rescale `comb` to carry
//!    `cond`'s per-token norm (`comb · ‖cond‖ / ‖comb‖` along the channel axis). This now lives in the
//!    backend-neutral `gen_core::guidance::cfg_rescale` (epic 7434), wired in from the
//!    [`pipeline`](crate::pipeline) (`cfg_rescale`).
//!
//! The denoise step itself is the core flow-match Euler step ([`FlowMatchEuler::step`]).

use mlx_gen::scheduler::compute_mu;
use mlx_gen::FlowMatchEuler;

/// Per-variant sampling defaults (`num_steps`, `guidance_scale`).
#[derive(Clone, Copy, Debug)]
pub struct LensSamplingDefaults {
    pub num_steps: usize,
    pub guidance_scale: f32,
}

/// `microsoft/Lens-Turbo`: distilled **4 steps, guidance 1.0** (≈ no CFG).
pub const TURBO: LensSamplingDefaults = LensSamplingDefaults {
    num_steps: 4,
    guidance_scale: 1.0,
};
/// `microsoft/Lens` (base): **20 steps, guidance 5.0**.
pub const BASE: LensSamplingDefaults = LensSamplingDefaults {
    num_steps: 20,
    guidance_scale: 5.0,
};

/// Build the Lens flow-match schedule for `num_steps` at the given latent grid. The empirical
/// time-shift `mu` is fit from the latent token count `latent_h · latent_w` (== the reference
/// `compute_empirical_mu(seq_len, num_steps)`).
pub fn lens_schedule(num_steps: usize, latent_h: usize, latent_w: usize) -> FlowMatchEuler {
    let mu = compute_mu(latent_h * latent_w, num_steps);
    FlowMatchEuler::new(num_steps, mu)
}

/// The per-step transformer timesteps: the **shifted sigmas** `sigmas[0..num_steps]` (Lens feeds the
/// sigma directly, the reference `timestep / 1000`).
pub fn timesteps(schedule: &FlowMatchEuler) -> Vec<f32> {
    schedule.sigmas[..schedule.num_steps()].to_vec()
}
