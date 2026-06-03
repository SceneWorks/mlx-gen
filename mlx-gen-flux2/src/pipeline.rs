//! FLUX.2 sampling-pipeline primitives whose math is stable before the model blocks land.
//! Mirror the fork's `Flux2LatentCreator` (`models/flux2/latent_creator/`), the prompt
//! encoder's `prepare_text_ids`, and the shared flow-match schedule.
//!
//! Latent geometry (klein, vae_scale_factor = 8, 2×2 patch): a `width × height` image →
//! VAE latents `[1, 32, height/8, width/8]` → **2×2 patchify** `[1, 128, height/16, width/16]`
//! → **pack** to the transformer token sequence `[1, (height/16)·(width/16), 128]`. txt2img
//! samples noise directly in the packed 128-channel space.

use mlx_gen::image::resize_lanczos_u8;
use mlx_gen::media::Image;
use mlx_gen::{Error, FlowMatchEuler, Result};
use mlx_rs::ops::{add, multiply};
use mlx_rs::{random, Array};

/// Transformer token-sequence length: `(height/16) · (width/16)`.
pub fn image_seq_len(width: u32, height: u32) -> usize {
    ((height / 16) * (width / 16)) as usize
}

/// 2×2 patchify: `[B, C, H, W]` → `[B, C·4, H/2, W/2]` (the fork's `patchify_latents`).
/// Folds each 2×2 spatial block into the channel axis; ordering matches the fork exactly.
pub fn patchify_latents(latents: &Array) -> Result<Array> {
    let sh = latents.shape();
    if sh.len() != 4 {
        return Err(Error::Msg(format!(
            "flux2 patchify: expected [B,C,H,W], got {sh:?}"
        )));
    }
    let (b, c, h, w) = (sh[0], sh[1], sh[2], sh[3]);
    if h % 2 != 0 || w % 2 != 0 {
        return Err(Error::Msg(format!(
            "flux2 patchify: H and W must be even, got {h}x{w}"
        )));
    }
    let x = latents.reshape(&[b, c, h / 2, 2, w / 2, 2])?;
    let x = x.transpose_axes(&[0, 1, 3, 5, 2, 4])?;
    Ok(x.reshape(&[b, c * 4, h / 2, w / 2])?)
}

/// Pack spatial latents `[B, C, H, W]` into transformer tokens `[B, H·W, C]`
/// (the fork's `pack_latents`).
pub fn pack_latents(latents: &Array) -> Result<Array> {
    let sh = latents.shape();
    if sh.len() != 4 {
        return Err(Error::Msg(format!(
            "flux2 pack: expected [B,C,H,W], got {sh:?}"
        )));
    }
    let (b, c, h, w) = (sh[0], sh[1], sh[2], sh[3]);
    Ok(latents
        .reshape(&[b, c, h * w])?
        .transpose_axes(&[0, 2, 1])?)
}

/// Unpack transformer tokens `[B, seq, C]` back to spatial latents `[B, C, lat_h, lat_w]`,
/// where `lat_h = height/16`, `lat_w = width/16` (the fork's `unpack_latents`).
pub fn unpack_latents(latents: &Array, width: u32, height: u32) -> Result<Array> {
    let sh = latents.shape();
    if sh.len() != 3 {
        return Err(Error::Msg(format!(
            "flux2 unpack: expected packed [B,seq,C], got {sh:?}"
        )));
    }
    let (b, seq, c) = (sh[0], sh[1], sh[2]);
    let lat_h = (height / 16) as i32;
    let lat_w = (width / 16) as i32;
    if lat_h * lat_w != seq {
        return Err(Error::Msg(format!(
            "flux2 unpack: seq {seq} != {lat_h}x{lat_w} for {width}x{height}"
        )));
    }
    Ok(latents
        .reshape(&[b, lat_h, lat_w, c])?
        .transpose_axes(&[0, 3, 1, 2])?)
}

/// Build the latent grid ids `[1, lat_h·lat_w, 4]` with coordinate `[t_coord, h, w, 0]`
/// (the fork's `prepare_grid_ids`). Row-major over `(h, w)` to match the packed token order.
pub fn prepare_grid_ids(lat_h: usize, lat_w: usize, t_coord: i32) -> Array {
    let mut ids: Vec<i32> = Vec::with_capacity(lat_h * lat_w * 4);
    for h in 0..lat_h {
        for w in 0..lat_w {
            ids.push(t_coord);
            ids.push(h as i32);
            ids.push(w as i32);
            ids.push(0);
        }
    }
    Array::from_slice(&ids, &[1, (lat_h * lat_w) as i32, 4])
}

/// Build the text ids `[1, seq, 4]` with coordinate `[0, 0, 0, token_index]`
/// (the fork's `prepare_text_ids`).
pub fn prepare_text_ids(seq: usize) -> Array {
    let mut ids: Vec<i32> = Vec::with_capacity(seq * 4);
    for token in 0..seq {
        ids.push(0);
        ids.push(0);
        ids.push(0);
        ids.push(token as i32);
    }
    Array::from_slice(&ids, &[1, seq as i32, 4])
}

/// Seeded txt2img latent noise, packed: `[1, (height/16)·(width/16), in_channels]`.
/// Mirrors `Flux2LatentCreator.prepare_packed_latents` — sample at `[1, in_channels, lat_h,
/// lat_w]` then pack — so the seeded RNG and token order match the fork (verified e2e in S4).
pub fn create_noise(seed: u64, width: u32, height: u32, in_channels: usize) -> Result<Array> {
    validate_multiple_of_16(width, height)?;
    let key = random::key(seed)?;
    let lat_h = (height / 16) as i32;
    let lat_w = (width / 16) as i32;
    let shape = [1, in_channels as i32, lat_h, lat_w];
    let latents = random::normal::<f32>(&shape[..], None, None, Some(&key))?;
    pack_latents(&latents)
}

/// The FLUX.2 denoising schedule: flow-match Euler with the empirical-mu time-shift. FLUX.2's
/// `requires_sigma_shift` path is exactly the core `FlowMatchEuler::for_image` (empirical mu
/// from the latent seq length, exponential time-shift, no terminal stretch) — proven against
/// the fork's `get_timesteps_and_sigmas`.
pub fn schedule(num_steps: usize, width: u32, height: u32) -> FlowMatchEuler {
    FlowMatchEuler::for_image(num_steps, width, height)
}

/// The timestep values fed to the transformer's time embedding: `shifted_sigma · 1000` for each
/// denoising step (the fork's `timesteps = sigmas[:n] · num_train_timesteps`). Distinct from the
/// `1 - sigma` convention other mlx-gen DiTs use — FLUX.2 passes the scaled sigma directly.
pub fn timesteps_x1000(schedule: &FlowMatchEuler) -> Vec<f32> {
    schedule.sigmas[..schedule.num_steps()]
        .iter()
        .map(|s| s * 1000.0)
        .collect()
}

/// Preprocess a reference image for the edit path: PIL-LANCZOS resize to `target_width ×
/// target_height` (no-op when already sized), normalize `[0,255] → [-1,1]`, lay out as **NHWC**
/// `[1, H, W, 3]` f32 — the input the FLUX.2 VAE encoder expects. Mirrors the fork's
/// `ImageUtil.scale_to_dimensions` + `to_array` (`2·x − 1`).
pub fn preprocess_ref_image(image: &Image, target_width: u32, target_height: u32) -> Result<Array> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    let (tw, th) = (target_width as usize, target_height as usize);
    if image.pixels.len() != iw * ih * 3 {
        return Err(Error::Msg(format!(
            "flux2 ref image pixel buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    let resized: Vec<f32> = if (ih, iw) == (th, tw) {
        image.pixels.iter().map(|&p| p as f32).collect()
    } else {
        resize_lanczos_u8(&image.pixels, ih, iw, th, tw)
    };
    let norm: Vec<f32> = resized.iter().map(|&v| 2.0 * (v / 255.0) - 1.0).collect();
    Ok(Array::from_slice(&norm, &[1, th as i32, tw as i32, 3]))
}

/// img2img start step (the fork's `Config.init_time_step`): `max(1, floor(num_steps · strength))`
/// for a positive strength clamped to `[0, 1]`, else `0` (pure txt2img). The denoise loop runs
/// `start_step..num_steps`, and the init image is blended in at `sigmas[start_step]`.
pub fn init_time_step(num_steps: usize, strength: Option<f32>) -> usize {
    match strength {
        Some(s) if s > 0.0 => {
            let s = s.clamp(0.0, 1.0);
            // Python `int(num_steps * strength)` truncates toward zero == floor for s >= 0.
            ((num_steps as f32 * s) as usize).max(1)
        }
        _ => 0,
    }
}

/// img2img / edit noise blend: `(1 - sigma)·clean + sigma·noise` at the start sigma.
/// Mirrors `LatentCreator.add_noise_by_interpolation`. (Used by sc-2644 img2img / S5 edit.)
pub fn add_noise_by_interpolation(clean: &Array, noise: &Array, sigma: f32) -> Result<Array> {
    let one_minus = Array::from_slice(&[1.0 - sigma], &[1]);
    let s = Array::from_slice(&[sigma], &[1]);
    Ok(add(&multiply(clean, &one_minus)?, &multiply(noise, &s)?)?)
}

fn validate_multiple_of_16(width: u32, height: u32) -> Result<()> {
    if !width.is_multiple_of(16) || !height.is_multiple_of(16) {
        return Err(Error::Msg(format!(
            "flux2: width and height must be multiples of 16, got {width}x{height}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_time_step_matches_fork_floor_and_clamp() {
        // None / zero / negative strength → pure txt2img (start at 0).
        assert_eq!(init_time_step(4, None), 0);
        assert_eq!(init_time_step(4, Some(0.0)), 0);
        assert_eq!(init_time_step(4, Some(-0.5)), 0);
        // floor(steps · strength), with a floor of 1 for any positive strength.
        assert_eq!(init_time_step(4, Some(0.1)), 1); // floor(0.4)=0 → clamped up to 1
        assert_eq!(init_time_step(4, Some(0.6)), 2); // floor(2.4)=2
        assert_eq!(init_time_step(20, Some(0.6)), 12); // floor(12.0)=12
                                                       // strength clamped to 1.0.
        assert_eq!(init_time_step(4, Some(2.0)), 4);
    }
}
