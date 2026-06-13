//! Lens sampling schedule + CFG (sc-3170). The schedule is the core [`mlx_gen::FlowMatchEuler`]
//! verbatim: the Lens `compute_empirical_mu` is **byte-identical** to the core
//! [`mlx_gen::scheduler::compute_mu`] (same calibrated constants + `>4300` branch), and the Lens
//! `np.linspace(1, 1/n, n)` → `set_timesteps(sigmas, mu)` dynamic-shift is exactly
//! `FlowMatchEuler::new(num_steps, mu)`. Only two pieces are Lens-specific:
//!
//! 1. **Timestep convention** — Lens feeds the transformer the *shifted sigma* directly (the
//!    reference `timestep / 1000`, where `scheduler.timesteps = shifted_sigma · 1000`), **not** the
//!    `1 − sigma` other mlx-gen DiTs use. [`timesteps`] returns those shifted sigmas.
//! 2. **Norm-rescaled CFG** — [`cfg_rescale`]: `comb = uncond + g·(cond − uncond)`, then rescale
//!    `comb` to carry `cond`'s per-token norm (`comb · ‖cond‖ / ‖comb‖` along the channel axis).
//!
//! The denoise step itself is the core flow-match Euler step ([`FlowMatchEuler::step`]).

use mlx_rs::ops::{add, gt, maximum, multiply, ones_like, sqrt, subtract, sum_axes, which};
use mlx_rs::Array;

use mlx_gen::scheduler::compute_mu;
use mlx_gen::{FlowMatchEuler, Result};

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

/// Norm-rescaled classifier-free guidance (the reference per-step CFG).
///
/// `cond`/`uncond`: `[B, seq, C]` predictions. Returns
/// `comb · (‖cond‖ / ‖comb‖)` per token (channel-axis L2 norm), with `comb = uncond + g·(cond −
/// uncond)`; where `‖comb‖ = 0` the scale is `1` (matching the reference `torch.where`).
pub fn cfg_rescale(cond: &Array, uncond: &Array, guidance: f32) -> Result<Array> {
    let g = Array::from_f32(guidance);
    let comb = add(uncond, &multiply(&subtract(cond, uncond)?, &g)?)?;

    let norm = |x: &Array| -> Result<Array> {
        Ok(sqrt(&sum_axes(&multiply(x, x)?, &[-1], true)?)?) // [B, seq, 1]
    };
    let cond_norm = norm(cond)?;
    let comb_norm = norm(&comb)?;

    let denom = maximum(&comb_norm, Array::from_f32(1e-12))?;
    let ratio = mlx_rs::ops::divide(&cond_norm, &denom)?;
    // scale = where(comb_norm > 0, cond_norm / max(comb_norm, 1e-12), 1).
    let positive = gt(&comb_norm, Array::from_f32(0.0))?;
    let scale = which(&positive, &ratio, &ones_like(&comb_norm)?)?;
    Ok(multiply(&comb, &scale)?)
}
