//! Float32 RoPE **position grid** in pixel space — port of `generate.py::create_position_grid`
//! (identical in `generate_av.py`).
//!
//! For latent dims `(num_frames, height, width)` and patch size `(1,1,1)`, each latent token gets
//! `[start, end)` bounds on three axes (frame, height, width), scaled from latent to pixel space by
//! the VAE factors (temporal 8×, spatial 32×). Two LTX-specific corrections on the **frame** axis:
//! a **causal first-frame fix** (the VAE's first-frame temporal stride is 1, not `temporal_scale`)
//! and **fps division** (frame index → time in seconds). Always f32 — the reference warns that
//! bf16 position grids degrade RoPE quality (sc-2679 keeps positions f32).
//!
//! Output shape `(batch, 3, num_frames·height·width, 2)`, C-order, where the last axis is
//! `[start, end]`. The token order is C-major over `(frame, height, width)` (`meshgrid(indexing="ij")`).

use mlx_rs::Array;

/// LTX-2 VAE factors + sampling defaults used by the T2V pipeline.
pub const TEMPORAL_SCALE: i64 = 8;
pub const SPATIAL_SCALE: i64 = 32;
pub const DEFAULT_FPS: f32 = 24.0;

/// Build the position grid with the LTX-2.3 defaults (temporal 8×, spatial 32×, 24 fps, causal fix).
pub fn create_position_grid(
    batch_size: usize,
    num_frames: usize,
    height: usize,
    width: usize,
) -> Array {
    create_position_grid_with(
        batch_size,
        num_frames,
        height,
        width,
        TEMPORAL_SCALE,
        SPATIAL_SCALE,
        DEFAULT_FPS,
        true,
    )
}

/// Build the position grid with explicit VAE scale factors / fps / causal-fix toggle.
///
/// Mirrors the reference op order exactly: integer `latent · scale` (exact), cast to f32, then the
/// frame-axis causal fix `max(0, px + 1 − temporal_scale)` and `÷ fps` are applied in f32 — so the
/// only rounding is the final `÷ fps`, matching numpy under NEP 50 (f32 array ÷ python float stays f32).
#[allow(clippy::too_many_arguments)]
pub fn create_position_grid_with(
    batch_size: usize,
    num_frames: usize,
    height: usize,
    width: usize,
    temporal_scale: i64,
    spatial_scale: i64,
    fps: f32,
    causal_fix: bool,
) -> Array {
    let hw = height * width;
    let num_patches = num_frames * hw;
    // C-order (batch, 3, num_patches, 2).
    let mut data = vec![0f32; batch_size * 3 * num_patches * 2];

    for p in 0..num_patches {
        let t = (p / hw) as i64;
        let rem = p % hw;
        let h = (rem / width) as i64;
        let w = (rem % width) as i64;

        for e in 0..2i64 {
            // frame axis (d=0): pixel = (t + e) · temporal_scale, then causal fix + fps.
            let frame_pix = (t + e) * temporal_scale;
            let mut frame_f = frame_pix as f32;
            if causal_fix {
                frame_f = (frame_f + 1.0 - temporal_scale as f32).max(0.0);
            }
            frame_f /= fps;

            // height axis (d=1) and width axis (d=2): pixel = (coord + e) · spatial_scale.
            let height_f = ((h + e) * spatial_scale) as f32;
            let width_f = ((w + e) * spatial_scale) as f32;

            for b in 0..batch_size {
                let base = ((b * 3) * num_patches + p) * 2 + e as usize;
                data[base] = frame_f; // d = 0
                data[base + num_patches * 2] = height_f; // d = 1
                data[base + 2 * num_patches * 2] = width_f; // d = 2
            }
        }
    }

    Array::from_slice(&data, &[batch_size as i32, 3, num_patches as i32, 2])
}

// --- Audio (sc-2684) ---------------------------------------------------------------------------

/// Audio VAE internal sample rate (`AUDIO_LATENT_SAMPLE_RATE`).
pub const AUDIO_LATENT_SAMPLE_RATE: i64 = 16000;
/// Mel hop length (`AUDIO_HOP_LENGTH`).
pub const AUDIO_HOP_LENGTH: i64 = 160;
/// Latent temporal downsample factor (`AUDIO_LATENT_DOWNSAMPLE_FACTOR`).
pub const AUDIO_LATENT_DOWNSAMPLE_FACTOR: i64 = 4;
/// Audio latent channels before patchifying (`AUDIO_LATENT_CHANNELS`).
pub const AUDIO_LATENT_CHANNELS: i64 = 8;
/// Audio latent mel bins (`AUDIO_MEL_BINS`).
pub const AUDIO_MEL_BINS: i64 = 16;
/// `AUDIO_LATENT_SAMPLE_RATE / AUDIO_HOP_LENGTH / AUDIO_LATENT_DOWNSAMPLE_FACTOR` = 25.
pub const AUDIO_LATENTS_PER_SECOND: f64 = 25.0;

/// Python `round()` (round-half-to-even) — matches `compute_audio_frames`'s `round(...)`.
fn py_round(x: f64) -> i64 {
    let f = x.floor();
    let diff = x - f;
    if diff < 0.5 {
        f as i64
    } else if diff > 0.5 {
        f as i64 + 1
    } else {
        let fi = f as i64;
        if fi % 2 == 0 {
            fi
        } else {
            fi + 1
        }
    }
}

/// Audio latent-frame count for a video duration — port of `compute_audio_frames`
/// (`round(num_video_frames / fps · AUDIO_LATENTS_PER_SECOND)`). Computed in f64 (Python floats).
pub fn compute_audio_frames(num_video_frames: usize, fps: f64) -> usize {
    let duration = num_video_frames as f64 / fps;
    py_round(duration * AUDIO_LATENTS_PER_SECOND).max(0) as usize
}

/// Build the audio RoPE position grid — port of `generate_av.py::create_audio_position_grid`.
///
/// Audio positions are **timestamps in seconds**, shape `(batch, 1, T, 2)` where the last axis is
/// `[start, end]`. For latent frame `t`: `mel = clip(t·4 + 1 − 4, 0)` (start) / `clip((t+1)·4 + 1 −
/// 4, 0)` (end), `time = mel · hop_length / sample_rate` — the causal-aligned mel→second map
/// (`_get_audio_latent_time_in_sec`). Always f32 (RoPE precision; the reference warns bf16 degrades).
pub fn create_audio_position_grid(batch_size: usize, audio_frames: usize) -> Array {
    create_audio_position_grid_with(
        batch_size,
        audio_frames,
        AUDIO_LATENT_SAMPLE_RATE,
        AUDIO_HOP_LENGTH,
        AUDIO_LATENT_DOWNSAMPLE_FACTOR,
        true,
    )
}

/// [`create_audio_position_grid`] with explicit rates / causal toggle.
pub fn create_audio_position_grid_with(
    batch_size: usize,
    audio_frames: usize,
    sample_rate: i64,
    hop_length: i64,
    downsample_factor: i64,
    is_causal: bool,
) -> Array {
    // `mel · hop_length / sample_rate` in the reference's f32 op order (mult then divide).
    let time = |latent: i64| -> f32 {
        let mut mel = (latent * downsample_factor) as f32;
        if is_causal {
            mel = (mel + 1.0 - downsample_factor as f32).max(0.0);
        }
        (mel * hop_length as f32) / sample_rate as f32
    };

    let t = audio_frames;
    // C-order (batch, 1, T, 2).
    let mut data = vec![0f32; batch_size * t * 2];
    for f in 0..t {
        let start = time(f as i64); // latent indices 0..T
        let end = time(f as i64 + 1); // latent indices 1..T+1
        for b in 0..batch_size {
            let base = (b * t + f) * 2;
            data[base] = start;
            data[base + 1] = end;
        }
    }
    Array::from_slice(&data, &[batch_size as i32, 1, t as i32, 2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_and_first_frame_causal_fix() {
        // num_frames=2, h=3, w=4 → 24 patches.
        let g = create_position_grid(1, 2, 3, 4);
        assert_eq!(g.shape(), &[1, 3, 24, 2]);
        let v: Vec<f32> = g.as_slice::<f32>().to_vec();
        // C-order index helper for (b=0, d, p, e): ((d)*24 + p)*2 + e.
        let at = |d: usize, p: usize, e: usize| v[(d * 24 + p) * 2 + e];

        // Patch p=0 → (t=0,h=0,w=0). Frame axis: start clip(0+1-8)=0 → /24 = 0;
        // end clip(8+1-8)=1 → /24.
        assert!((at(0, 0, 0) - 0.0).abs() < 1e-9);
        assert!((at(0, 0, 1) - (1.0 / 24.0)).abs() < 1e-7);
        // Height axis at p=0: start 0, end 32.
        assert_eq!(at(1, 0, 0), 0.0);
        assert_eq!(at(1, 0, 1), 32.0);
        // Width axis at p=0: start 0, end 32.
        assert_eq!(at(2, 0, 0), 0.0);
        assert_eq!(at(2, 0, 1), 32.0);

        // Patch p=12 → (t=1,h=0,w=0): frame start clip(8+1-8)=1 → /24, end clip(16+1-8)=9 → /24.
        assert!((at(0, 12, 0) - (1.0 / 24.0)).abs() < 1e-7);
        assert!((at(0, 12, 1) - (9.0 / 24.0)).abs() < 1e-7);
        // Patch p=5 → (t=0,h=1,w=1): height start 32, end 64; width start 32, end 64.
        assert_eq!(at(1, 5, 0), 32.0);
        assert_eq!(at(1, 5, 1), 64.0);
        assert_eq!(at(2, 5, 0), 32.0);
        assert_eq!(at(2, 5, 1), 64.0);
    }

    #[test]
    fn audio_position_grid_matches_reference() {
        // 4 audio latent frames → (1, 1, 4, 2). Causal mel→sec: start mel=clip(4t-3,0),
        // end mel=4t+1; time = mel·160/16000.
        let g = create_audio_position_grid(1, 4);
        assert_eq!(g.shape(), &[1, 1, 4, 2]);
        let v: Vec<f32> = g.as_slice::<f32>().to_vec();
        let sec = |mel: f32| (mel * 160.0) / 16000.0;
        // (start, end) per latent frame t.
        let want = [
            (sec(0.0), sec(1.0)),  // t=0
            (sec(1.0), sec(5.0)),  // t=1
            (sec(5.0), sec(9.0)),  // t=2
            (sec(9.0), sec(13.0)), // t=3
        ];
        for (t, (s, e)) in want.iter().enumerate() {
            assert!(
                (v[t * 2] - s).abs() < 1e-9,
                "start t={t}: {} vs {s}",
                v[t * 2]
            );
            assert!(
                (v[t * 2 + 1] - e).abs() < 1e-9,
                "end t={t}: {} vs {e}",
                v[t * 2 + 1]
            );
        }
    }

    #[test]
    fn compute_audio_frames_matches_reference() {
        // round(num_frames / fps · 25). 33f@24fps: 33/24·25 = 34.375 → 34.
        assert_eq!(compute_audio_frames(33, 24.0), 34);
        // 9f@24fps: 9/24·25 = 9.375 → 9. 1f@24: 1/24·25 = 1.04 → 1.
        assert_eq!(compute_audio_frames(9, 24.0), 9);
        assert_eq!(compute_audio_frames(1, 24.0), 1);
        // 121f@24fps: 121/24·25 = 126.04 → 126.
        assert_eq!(compute_audio_frames(121, 24.0), 126);
    }
}
