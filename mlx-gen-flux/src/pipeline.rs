//! FLUX.1 sampling-pipeline primitives whose math is stable before the model blocks land.
//! These mirror `FluxLatentCreator` and the fork's default `LinearScheduler`.

use mlx_gen::image::validate_multiple_of_16;
use mlx_gen::{resolve_flow_schedule, Result};
use mlx_rs::ops::{add, divide, linspace, subtract};
use mlx_rs::{random, Array};

pub fn image_seq_len(width: u32, height: u32) -> usize {
    ((height / 16) * (width / 16)) as usize
}

/// The FLUX.1 time-shift `mu` for the `requires_sigma_shift` (dev) path — the fork's `LinearScheduler`
/// resolution-dependent linear fit `mu = m·(w·h/256) + b` (constants base `(256, 0.5)`, max
/// `(4096, 1.15)`). `0.0` for FLUX.1-schnell (unshifted). Exposed so the epic 7114 scheduler axis can
/// build a curated schedule over FLUX.1's OWN mu (which is the analytic linear fit, NOT the mflux
/// empirical `compute_mu`). Kept identical to [`build_linear_sigmas`] so the native default is byte-exact.
pub fn linear_sigma_mu(width: u32, height: u32, requires_sigma_shift: bool) -> f32 {
    if !requires_sigma_shift {
        return 0.0;
    }
    let base_seq_len = 256.0_f64;
    let max_seq_len = 4096.0_f64;
    let base_shift = 0.5_f64;
    let max_shift = 1.15_f64;
    let m = (max_shift - base_shift) / (max_seq_len - base_seq_len);
    let b = base_shift - m * base_seq_len;
    (m * (width as f64) * (height as f64) / 256.0 + b) as f32
}

/// [`build_linear_sigmas`] honoring a per-generation curated `scheduler` (epic 7114 scheduler axis). An
/// unset / unknown / `linear`-aliased name keeps the native schedule byte-exact (N1); a curated name
/// (`normal` / `sgm_uniform` / `karras` / …) re-shapes σ over FLUX.1's own [`linear_sigma_mu`] (schnell:
/// `mu = 0`, an unshifted ramp).
pub fn build_sigmas_with(
    num_steps: usize,
    width: u32,
    height: u32,
    requires_sigma_shift: bool,
    scheduler_name: Option<&str>,
) -> Result<Vec<f32>> {
    let native = build_linear_sigmas(num_steps, width, height, requires_sigma_shift)?;
    let mu = linear_sigma_mu(width, height, requires_sigma_shift);
    Ok(resolve_flow_schedule(
        scheduler_name,
        mu,
        num_steps,
        &native,
    ))
}

/// Seeded FLUX txt2img latent noise: `[1, (height/16) * (width/16), 64]`.
pub fn create_noise(seed: u64, width: u32, height: u32) -> Result<Array> {
    validate_multiple_of_16(width, height, "flux1")?;
    let key = random::key(seed)?;
    let shape = [1, image_seq_len(width, height) as i32, 64];
    Ok(random::normal::<f32>(&shape[..], None, None, Some(&key))?)
}

/// Pack VAE latents `[1, 16, height/8, width/8]` into FLUX DiT tokens
/// `[1, (height/16) * (width/16), 64]`.
pub fn pack_latents(latents: &Array, width: u32, height: u32) -> Result<Array> {
    validate_multiple_of_16(width, height, "flux1")?;
    let h = (height / 16) as i32;
    let w = (width / 16) as i32;
    let latents = latents.reshape(&[1, 16, h, 2, w, 2])?;
    let latents = latents.transpose_axes(&[0, 2, 4, 1, 3, 5])?;
    Ok(latents.reshape(&[1, h * w, 64])?)
}

/// Unpack FLUX DiT tokens `[1, (height/16) * (width/16), 64]` back to VAE latents
/// `[1, 16, height/8, width/8]`.
pub fn unpack_latents(latents: &Array, width: u32, height: u32) -> Result<Array> {
    validate_multiple_of_16(width, height, "flux1")?;
    let h = (height / 16) as i32;
    let w = (width / 16) as i32;
    let latents = latents.reshape(&[1, h, w, 16, 2, 2])?;
    let latents = latents.transpose_axes(&[0, 3, 1, 4, 2, 5])?;
    Ok(latents.reshape(&[1, 16, h * 2, w * 2])?)
}

/// Fork `LinearScheduler` sigmas. `requires_sigma_shift` is true for FLUX.1-dev and false for
/// FLUX.1-schnell. The shift constants are the fork's defaults: base `(256, 0.5)`, max
/// `(4096, 1.15)`, no terminal stretch for FLUX.1.
pub fn build_linear_sigmas(
    num_steps: usize,
    width: u32,
    height: u32,
    requires_sigma_shift: bool,
) -> Result<Vec<f32>> {
    let n = num_steps.max(1) as i32;
    // `mx.linspace(1.0, 1.0/n, n)` computed in MLX (sc-2787): the host interpolation differs from the
    // MLX op by ~6e-8, which the chaotic FLUX sampler amplifies into a different image. Pass the stop
    // as f64 like the fork's Python `1.0/num_steps`. Default linspace dtype is f32.
    let sigmas = linspace::<f64, f32>(1.0, 1.0 / n as f64, n)?;

    let sigmas = if requires_sigma_shift {
        // FLUX.1-dev mu-shift, mirroring `LinearScheduler` exactly: the resolution-dependent `mu` is the
        // shared [`linear_sigma_mu`] (byte-identical f64 fit), the shift division/exp run in MLX.
        let mu = linear_sigma_mu(width, height, true);
        let e = Array::from_slice(&[mu], &[1]).exp()?;
        let one = Array::from_slice(&[1.0_f32], &[1]);
        // shifted = exp(mu) / (exp(mu) + (1/sigmas - 1))
        let inv = divide(&one, &sigmas)?;
        divide(&e, &add(&e, &subtract(&inv, &one)?)?)?
    } else {
        sigmas
    };

    let mut out = sigmas.as_slice::<f32>().to_vec();
    out.push(0.0);
    Ok(out)
}
