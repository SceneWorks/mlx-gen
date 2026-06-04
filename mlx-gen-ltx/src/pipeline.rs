//! S5 — the **2-stage distilled T2V pipeline**: the denoise loop + the stage transition (2× spatial
//! upsample + re-noise) + video output. Port of the `mlx_video` reference `generate_av.py` video path
//! (`denoise_av` with audio disabled + the `generate_video_with_audio` stage orchestration).
//!
//! The shipped `base_q8` is a **unified split-weight** checkpoint (`split_model.json` `format:
//! "split"`), so the reference takes its `legacy_unified_sampler` branch: **fixed distilled sigmas**
//! ([`STAGE1_SIGMAS`] 8 steps + [`STAGE2_SIGMAS`] 3 steps) and the **legacy dtype-preserving Euler**
//! update. The distilled 2.3 model bakes in guidance, so `effective_cfg_scale = 1.0` → **no CFG**
//! (single forward per step, no negative prompt). T2V has no I2V conditioning state.
//!
//! Flow ([`generate_t2v`]): random noise → stage-1 denoise at half-res → [`upsample_latents`] 2× →
//! re-noise (`noise·σ₂₀ + latent·(1−σ₂₀)`) → stage-2 denoise at full-res → VAE decode →
//! `(x+1)/2·255` uint8 frames `[F, H, W, 3]` (the consuming app muxes these to MP4 — matching the
//! Wan sibling, MP4 encoding is out of the crate).
//!
//! **Precision (S5 gate).** Run in the **f32** regime (latents f32, transformer [`Precision::F32Q8`],
//! upsampler/VAE f32) to gate the pipeline *math* — the 2-stage orchestration, the legacy Euler, the
//! re-noise, the flatten/unflatten — bit-tight, isolated from bf16 rounding (consistent with the S3b
//! DiT gate). The bf16-**production** end-to-end px>8 verdict is S6, which wires the real text encoder
//! and the public `generate()`. Honors "divergence is not rounding": the parity test localizes any
//! gap (per-stage latents + decoded frames) rather than writing it off.

use mlx_rs::ops::{add, broadcast_to, divide, maximum, minimum, multiply, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::media::AudioTrack;
use mlx_gen::Result;

use crate::audio_vae::AudioDecoder;
use crate::transformer::{to_denoised, AvDiT, LtxDiT};
use crate::upsampler::{upsample_latents, LatentUpsampler};
use crate::vae::LtxVideoVae;
use crate::vocoder::LtxVocoder;

/// Distilled stage-1 sigmas (`DEFAULT_STAGE_1_SIGMAS`, 8 denoise steps).
pub const STAGE1_SIGMAS: [f32; 9] = [
    1.0, 0.993_75, 0.987_5, 0.981_25, 0.975, 0.909_375, 0.725, 0.421_875, 0.0,
];
/// Distilled stage-2 sigmas (`DEFAULT_STAGE_2_SIGMAS`, 3 denoise steps). `STAGE2_SIGMAS[0]` is the
/// stage-transition re-noise scale.
pub const STAGE2_SIGMAS: [f32; 4] = [0.909_375, 0.725, 0.421_875, 0.0];

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// Force a logically-contiguous copy (see `vae.rs`): host reads (`as_slice`) return the *physical*
/// buffer, so an array left strided by the `(F,H,W,C)` transpose reads scrambled.
fn contiguous(x: &Array) -> Result<Array> {
    let shape = x.shape().to_vec();
    Ok(x.reshape(&[-1])?.reshape(&shape)?)
}

/// The legacy dtype-preserving Euler update (the `use_legacy_euler` branch): for `σ_next > 0`,
/// `x' = denoised + σ_next·(x − denoised)/σ`; at the final step (`σ_next = 0`), `x' = denoised`.
/// Computed in `x`'s dtype (the σ scalars are cast to it) — algebraically `x + (σ_next − σ)·v` but
/// kept in the reference's exact op order/dtype for bit-parity.
pub fn euler_step(x: &Array, denoised: &Array, sigma: f32, sigma_next: f32) -> Result<Array> {
    if sigma_next <= 0.0 {
        return Ok(denoised.clone());
    }
    let dt = x.dtype();
    let sn = scalar(sigma_next).as_dtype(dt)?;
    let sg = scalar(sigma).as_dtype(dt)?;
    let step = divide(&multiply(&sn, &subtract(x, denoised)?)?, &sg)?;
    Ok(add(denoised, &step)?)
}

/// One stage's denoise loop. T2V distilled: **no CFG**, **legacy Euler**, no I2V state.
///
/// * `latents` — `(B, 128, F, H, W)` NCFHW, the stage's dtype (f32 here, S5 gate).
/// * `context` — `(B, ctx, inner)` text embeddings (the connector output / S6's text encoder).
/// * `positions` — `(B, 3, S, 2)` position grid for this stage's latent dims.
/// * `sigmas` — the stage schedule; `sigmas.len() − 1` denoise steps.
/// * `on_step` — progress callback, fired once per completed step.
pub fn denoise(
    dit: &LtxDiT,
    latents: &Array,
    context: &Array,
    positions: &Array,
    sigmas: &[f32],
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    let dt = latents.dtype();
    let sh = latents.shape();
    let (b, c, f, h, w) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
    let num_tokens = f * h * w;
    let mut lat = latents.clone();

    for i in 0..sigmas.len() - 1 {
        let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);
        // (B, C, F, H, W) → (B, C, S) → (B, S, C) packed tokens.
        let flat = lat.reshape(&[b, c, -1])?.transpose_axes(&[0, 2, 1])?;
        // Per-token timesteps = σ (uniform for T2V), shape (B, num_tokens) — matches the reference.
        let ts = broadcast_to(&scalar(sigma).as_dtype(dt)?, &[b, num_tokens])?;
        let velocity = dit.forward(&flat, &ts, context, None, positions)?;
        // (B, S, C) → (B, C, S) → (B, C, F, H, W).
        let velocity = velocity
            .transpose_axes(&[0, 2, 1])?
            .reshape(&[b, c, f, h, w])?;
        let sig = scalar(sigma).as_dtype(dt)?;
        let denoised = to_denoised(&lat, &velocity, &sig)?;
        lat = euler_step(&lat, &denoised, sigma, sigma_next)?;
        mlx_rs::transforms::eval([&lat])?;
        on_step(i + 1);
    }
    Ok(lat)
}

/// Stage-transition re-noise: `noise·scale + latent·(1 − scale)`, dtype-preserving. `1 − scale` is
/// computed in `latent`'s dtype (`array(1) − array(scale)`), matching the reference exactly.
pub fn renoise(latents: &Array, noise: &Array, noise_scale: f32) -> Result<Array> {
    let dt = latents.dtype();
    let s = scalar(noise_scale).as_dtype(dt)?;
    let one_minus = subtract(&scalar(1.0).as_dtype(dt)?, &s)?;
    Ok(add(&multiply(noise, &s)?, &multiply(latents, &one_minus)?)?)
}

/// VAE-decode latents `(B, 128, F, H, W)` → `(F, H, W, 3)` uint8 frames. Reference order:
/// squeeze batch → `(F, H, W, 3)` → `clip((x+1)/2, 0, 1)·255` → uint8.
pub fn decode_to_frames(vae: &LtxVideoVae, latents: &Array) -> Result<Array> {
    to_uint8_frames(&vae.decode(latents)?)
}

/// `(B=1, 3, F, H, W)` video in ~[-1, 1] → `(F, H, W, 3)` uint8. The reference clips `(x+1)/2` to
/// `[0, 1]` *before* scaling by 255, so the result saturates at 255 (truncating cast).
pub fn to_uint8_frames(video: &Array) -> Result<Array> {
    let sh = video.shape(); // (1, 3, F, H, W)
    let (c, f, h, w) = (sh[1], sh[2], sh[3], sh[4]);
    let dt = video.dtype();
    let chw = video
        .reshape(&[c, f, h, w])?
        .transpose_axes(&[1, 2, 3, 0])?; // (F, H, W, 3)
    let half = divide(
        &add(&chw, &scalar(1.0).as_dtype(dt)?)?,
        &scalar(2.0).as_dtype(dt)?,
    )?;
    let clipped = minimum(
        &maximum(&half, &scalar(0.0).as_dtype(dt)?)?,
        &scalar(1.0).as_dtype(dt)?,
    )?;
    let scaled = multiply(&clipped, &scalar(255.0).as_dtype(dt)?)?;
    contiguous(&scaled.as_dtype(Dtype::Uint8)?)
}

/// The full 2-stage distilled T2V latent pipeline: stage-1 denoise → 2× upsample → re-noise →
/// stage-2 denoise. `stage1_noise`/`stage2_noise` are the (injected) initial + re-noise samples,
/// `context` the shared text embeddings, `*_positions` each stage's grid, `latent_{mean,std}` the VAE
/// `per_channel_statistics`. Returns the final full-res latents `(B, 128, F, H, W)`.
#[allow(clippy::too_many_arguments)]
pub fn generate_t2v_latents(
    dit: &LtxDiT,
    upsampler: &LatentUpsampler,
    stage1_noise: &Array,
    stage1_positions: &Array,
    stage2_noise: &Array,
    stage2_positions: &Array,
    context: &Array,
    latent_mean: &Array,
    latent_std: &Array,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    let lat = denoise(
        dit,
        stage1_noise,
        context,
        stage1_positions,
        &STAGE1_SIGMAS,
        on_step,
    )?;
    let lat = upsample_latents(&lat, upsampler, latent_mean, latent_std)?;
    let lat = renoise(&lat, stage2_noise, STAGE2_SIGMAS[0])?;
    denoise(
        dit,
        &lat,
        context,
        stage2_positions,
        &STAGE2_SIGMAS,
        on_step,
    )
}

/// [`generate_t2v_latents`] + VAE decode → uint8 frames `(F, H, W, 3)`.
#[allow(clippy::too_many_arguments)]
pub fn generate_t2v(
    dit: &LtxDiT,
    upsampler: &LatentUpsampler,
    vae: &LtxVideoVae,
    stage1_noise: &Array,
    stage1_positions: &Array,
    stage2_noise: &Array,
    stage2_positions: &Array,
    context: &Array,
    latent_mean: &Array,
    latent_std: &Array,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    let latents = generate_t2v_latents(
        dit,
        upsampler,
        stage1_noise,
        stage1_positions,
        stage2_noise,
        stage2_positions,
        context,
        latent_mean,
        latent_std,
        on_step,
    )?;
    decode_to_frames(vae, &latents)
}

// ===================================================================================================
// AudioVideo pipeline (sc-2684) — the joint `denoise_av` loop + audio decode → waveform.
// ===================================================================================================

/// One stage's **joint** video+audio denoise loop (`denoise_av`). Distilled T2V+A: no CFG, legacy
/// Euler, audio always enabled (the cross-modal attention couples the streams every step). Audio init
/// is pure noise (no audio I2V).
///
/// * `video` — `(B, 128, F, H, W)` NCFHW; `audio` — `(B, 8, T, 16)` NCTF.
/// * `*_ctx` — the video (4096) / audio (2048) text embeddings.
/// * `*_pos` — the video `(B,3,Sv,2)` / audio `(B,1,Ta,2)` position grids.
#[allow(clippy::too_many_arguments)]
pub fn denoise_av(
    dit: &AvDiT,
    video: &Array,
    audio: &Array,
    video_ctx: &Array,
    audio_ctx: &Array,
    video_pos: &Array,
    audio_pos: &Array,
    sigmas: &[f32],
    on_step: &mut dyn FnMut(usize),
) -> Result<(Array, Array)> {
    let dt = video.dtype();
    let v = video.shape();
    let (vb, vc, vf, vh, vw) = (v[0], v[1], v[2], v[3], v[4]);
    let v_tokens = vf * vh * vw;
    let a = audio.shape();
    let (ab, ac, at, af) = (a[0], a[1], a[2], a[3]);

    let mut vlat = video.clone();
    let mut alat = audio.clone();
    for i in 0..sigmas.len() - 1 {
        let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);
        // Flatten: video (B,C,F,H,W)→(B,Sv,C); audio (B,C,T,F)→(B,T,C·F).
        let vflat = vlat.reshape(&[vb, vc, -1])?.transpose_axes(&[0, 2, 1])?;
        let aflat = alat
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[ab, at, ac * af])?;
        let vts = broadcast_to(&scalar(sigma).as_dtype(dt)?, &[vb, v_tokens])?;
        let ats = broadcast_to(&scalar(sigma).as_dtype(dt)?, &[ab, at])?;
        let (vvel, avel) = dit.forward(
            &vflat, &vts, video_ctx, None, video_pos, &aflat, &ats, audio_ctx, None, audio_pos,
        )?;
        let vvel = vvel
            .transpose_axes(&[0, 2, 1])?
            .reshape(&[vb, vc, vf, vh, vw])?;
        let avel = avel
            .reshape(&[ab, at, ac, af])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let sig = scalar(sigma).as_dtype(dt)?;
        let vden = to_denoised(&vlat, &vvel, &sig)?;
        let aden = to_denoised(&alat, &avel, &sig)?;
        vlat = euler_step(&vlat, &vden, sigma, sigma_next)?;
        alat = euler_step(&alat, &aden, sigma, sigma_next)?;
        mlx_rs::transforms::eval([&vlat, &alat])?;
        on_step(i + 1);
    }
    Ok((vlat, alat))
}

/// The full 2-stage **AudioVideo** latent pipeline: joint stage-1 denoise → 2× upsample the **video**
/// (audio is not upsampled) → re-noise both → joint stage-2 denoise. Returns `(video_latents (B,128,
/// F,H,W), audio_latents (B,8,T,16))`.
#[allow(clippy::too_many_arguments)]
pub fn generate_av_latents(
    dit: &AvDiT,
    upsampler: &LatentUpsampler,
    video_s1_noise: &Array,
    video_pos1: &Array,
    video_s2_noise: &Array,
    video_pos2: &Array,
    audio_s1_noise: &Array,
    audio_s2_noise: &Array,
    audio_pos: &Array,
    video_ctx: &Array,
    audio_ctx: &Array,
    latent_mean: &Array,
    latent_std: &Array,
    on_step: &mut dyn FnMut(usize),
) -> Result<(Array, Array)> {
    let (v, a) = denoise_av(
        dit,
        video_s1_noise,
        audio_s1_noise,
        video_ctx,
        audio_ctx,
        video_pos1,
        audio_pos,
        &STAGE1_SIGMAS,
        on_step,
    )?;
    let v = upsample_latents(&v, upsampler, latent_mean, latent_std)?;
    let v = renoise(&v, video_s2_noise, STAGE2_SIGMAS[0])?;
    let a = renoise(&a, audio_s2_noise, STAGE2_SIGMAS[0])?;
    denoise_av(
        dit,
        &v,
        &a,
        video_ctx,
        audio_ctx,
        video_pos2,
        audio_pos,
        &STAGE2_SIGMAS,
        on_step,
    )
}

/// Decode audio latents → an interleaved-PCM [`AudioTrack`]: `audio_decoder` → mel `(B,2,T',64)` →
/// `vocoder` → waveform `(B,2,samples)` → interleaved `f32`. `decode_audio = audio_decoder → vocoder`.
pub fn decode_audio_track(
    decoder: &AudioDecoder,
    vocoder: &LtxVocoder,
    audio_latents: &Array,
    sample_rate: u32,
) -> Result<AudioTrack> {
    let mel = decoder.decode(audio_latents)?;
    let wav = vocoder.forward(&mel)?; // (B, channels, samples)
    let sh = wav.shape();
    let (channels, samples) = (sh[1] as usize, sh[2]);
    // (1, C, S) → (S, C) → interleaved.
    let interleaved = contiguous(
        &wav.reshape(&[channels as i32, samples])?
            .transpose_axes(&[1, 0])?,
    )?
    .as_dtype(Dtype::Float32)?;
    Ok(AudioTrack {
        samples: interleaved.as_slice::<f32>().to_vec(),
        sample_rate,
        channels: channels as u16,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(v: &[f32], shape: &[i32]) -> Array {
        Array::from_slice(v, shape)
    }

    #[test]
    fn euler_step_matches_reference_formula() {
        // x' = denoised + σ_next·(x − denoised)/σ.
        let x = arr(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let den = arr(&[0.5, 1.0, 1.5, 2.0], &[4]);
        let (sigma, sigma_next) = (0.5_f32, 0.25_f32);
        let got = euler_step(&x, &den, sigma, sigma_next).unwrap();
        let want: Vec<f32> = (0..4)
            .map(|i| {
                let (xv, dv) = (x.as_slice::<f32>()[i], den.as_slice::<f32>()[i]);
                dv + sigma_next * (xv - dv) / sigma
            })
            .collect();
        for (g, w) in got.as_slice::<f32>().iter().zip(&want) {
            assert!((g - w).abs() < 1e-6, "euler {g} vs {w}");
        }
    }

    #[test]
    fn euler_step_final_is_denoised() {
        let x = arr(&[1.0, 2.0], &[2]);
        let den = arr(&[9.0, 8.0], &[2]);
        let got = euler_step(&x, &den, 0.42, 0.0).unwrap();
        assert_eq!(got.as_slice::<f32>(), den.as_slice::<f32>());
    }

    #[test]
    fn renoise_matches_reference_formula() {
        // noise·scale + latent·(1−scale).
        let lat = arr(&[1.0, 2.0, 3.0], &[3]);
        let noise = arr(&[0.0, 1.0, -1.0], &[3]);
        let scale = 0.909_375_f32;
        let got = renoise(&lat, &noise, scale).unwrap();
        let want: Vec<f32> = (0..3)
            .map(|i| {
                let (nv, lv) = (noise.as_slice::<f32>()[i], lat.as_slice::<f32>()[i]);
                nv * scale + lv * (1.0 - scale)
            })
            .collect();
        for (g, w) in got.as_slice::<f32>().iter().zip(&want) {
            assert!((g - w).abs() < 1e-6, "renoise {g} vs {w}");
        }
    }

    #[test]
    fn to_uint8_frames_clips_and_scales() {
        // (1,3,1,1,2): values spanning below/within/above [-1,1]. Channel values chosen so
        // (x+1)/2·255 lands on exact integers (no trunc-vs-round ambiguity).
        let video = arr(&[-2.0, -1.0, 0.2, 1.0, 0.6, 2.0], &[1, 3, 1, 1, 2]);
        let frames = to_uint8_frames(&video).unwrap();
        assert_eq!(frames.shape(), &[1, 1, 2, 3]); // (F,H,W,3)
        assert_eq!(frames.dtype(), Dtype::Uint8);
        // Layout (F,H,W,C): channels at w=0 are x=[-2,0.2,0.6]; at w=1 x=[-1,1,2].
        // (x+1)/2 clip[0,1] ·255: -2→0, 0.2→153, 0.6→204, -1→0, 1→255, 2→255.
        let got = frames.as_slice::<u8>();
        assert_eq!(got, &[0, 153, 204, 0, 255, 255]);
    }
}
