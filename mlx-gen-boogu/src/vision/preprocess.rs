//! E7b-1: Qwen3-VL image preprocessing — packed patch pixels + `grid_thw` for [`super::VisionTower`].
//!
//! Port of the Qwen3-VL image processor (single-image path Boogu edit needs): [`smart_resize`] snaps
//! the image to `(h, w)` divisible by `factor = patch·merge` (32) with the area clamped to
//! `[min_pixels, max_pixels]`; [`pack_patches`] lays patches out in the **merge-grouped** order the
//! tower (and `get_vision_position_ids`) consume. Normalization is the Qwen3 default `mean = std =
//! 0.5` (→ `[-1, 1]`), and the patch size is **16** (vs the Qwen2.5 tower's 14).
//!
//! Mirrors the structure of `mlx-gen-bernini`'s `vit_preprocess` (proven against Qwen2.5-VL); only the
//! geometry constants and the normalization differ. The resize uses the `image` crate (Catmull-Rom),
//! which is not bit-identical to PIL bicubic — `grid_thw` is exact, only the resampled pixels differ
//! slightly (fine for a semantic vision tower).

use image::{imageops::FilterType, RgbImage};
use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

use mlx_gen::Result;

/// Qwen3-VL image norm (`mllm/preprocessor_config.json`: `image_mean = image_std = 0.5`).
pub const IMAGE_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
pub const IMAGE_STD: [f32; 3] = [0.5, 0.5, 0.5];

/// Patch geometry (Qwen3-VL vision processor).
pub const PATCH_SIZE: i64 = 16;
pub const TEMPORAL_PATCH_SIZE: i64 = 2;
pub const MERGE_SIZE: i64 = 2;
/// `patch · merge` — the dimension-divisibility factor.
pub const FACTOR: i64 = PATCH_SIZE * MERGE_SIZE;

/// Pixel-area bounds (`size = {shortest_edge, longest_edge}` in the processor config).
pub const MIN_PIXELS: i64 = 65_536;
pub const MAX_PIXELS: i64 = 16_777_216;

/// Python `round` (round half to **even**) — banker's rounding.
pub(crate) fn py_round(x: f64) -> i64 {
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

/// Qwen-VL `smart_resize`: snap `(height, width)` to multiples of `factor`, keeping aspect ratio
/// while clamping total pixels into `[min_pixels, max_pixels]`.
pub fn smart_resize(
    height: i64,
    width: i64,
    factor: i64,
    min_pixels: i64,
    max_pixels: i64,
) -> (i64, i64) {
    let (hf, wf, ff) = (height as f64, width as f64, factor as f64);
    let mut h_bar = py_round(hf / ff) * factor;
    let mut w_bar = py_round(wf / ff) * factor;
    if h_bar * w_bar > max_pixels {
        let beta = ((hf * wf) / max_pixels as f64).sqrt();
        h_bar = factor.max((hf / beta / ff).floor() as i64 * factor);
        w_bar = factor.max((wf / beta / ff).floor() as i64 * factor);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (hf * wf)).sqrt();
        h_bar = (hf * beta / ff).ceil() as i64 * factor;
        w_bar = (wf * beta / ff).ceil() as i64 * factor;
    }
    (h_bar, w_bar)
}

/// Pack normalized frames `[F, C, H, W]` into `pixel_values [seq, C·T·patch²]` + `grid_thw (t, h, w)`,
/// in the merge-grouped order `(grid_t, h/m, w/m, m, m, C, T, ph, pw)`. `H`/`W` must be multiples of
/// `patch·merge`.
pub fn pack_patches(
    frames: &Array,
    patch: i64,
    temporal: i64,
    merge: i64,
) -> Result<(Array, [i32; 3])> {
    let s = frames.shape();
    let (f, c, h, w) = (s[0] as i64, s[1] as i64, s[2] as i64, s[3] as i64);

    // Temporal-pad to a multiple of `temporal` by repeating the last frame.
    let frames = if f % temporal != 0 {
        let pad = temporal - (f % temporal);
        let idx: Vec<i32> = vec![(f - 1) as i32; pad as usize];
        let last = frames.take_axis(Array::from_slice(&idx, &[pad as i32]), 0)?;
        concatenate_axis(&[frames, &last], 0)?
    } else {
        frames.clone()
    };

    let fp = frames.shape()[0] as i64;
    let grid_t = fp / temporal;
    let grid_h = h / patch;
    let grid_w = w / patch;
    let (gh, gw) = (grid_h / merge, grid_w / merge);
    let i = |x: i64| x as i32;

    let reshaped = frames.reshape(&[
        i(grid_t),
        i(temporal),
        i(c),
        i(gh),
        i(merge),
        i(patch),
        i(gw),
        i(merge),
        i(patch),
    ])?;
    // (grid_t, gh, gw, m, m, C, T, ph, pw)
    let perm = reshaped.transpose_axes(&[0, 3, 6, 4, 7, 2, 1, 5, 8])?;
    let seq = grid_t * grid_h * grid_w;
    let row = c * temporal * patch * patch;
    let pixel_values = perm.reshape(&[i(seq), i(row)])?;
    Ok((pixel_values, [i(grid_t), i(grid_h), i(grid_w)]))
}

/// `[1, 3, h, w]` f32 from channels-last RGB8 bytes, rescaled (1/255) + normalized `(x - mean)/std`.
fn normalized_frame(pixels_hwc: &[u8], h: i64, w: i64, mean: [f32; 3], std: [f32; 3]) -> Array {
    let (hu, wu) = (h as usize, w as usize);
    let mut data = vec![0f32; 3 * hu * wu];
    for c in 0..3usize {
        let (m, sd) = (mean[c], std[c]);
        for y in 0..hu {
            for x in 0..wu {
                let u = pixels_hwc[(y * wu + x) * 3 + c] as f32;
                data[(c * hu + y) * wu + x] = (u / 255.0 - m) / sd;
            }
        }
    }
    Array::from_slice(&data, &[1, 3, h as i32, w as i32])
}

/// Full Qwen3-VL preprocessing of one RGB image → `pixel_values [seq, 1536]` + `grid_thw`.
pub fn preprocess_image(img: &RgbImage) -> Result<(Array, [i32; 3])> {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let (rh, rw) = smart_resize(h, w, FACTOR, MIN_PIXELS, MAX_PIXELS);
    let resized = if (rh, rw) == (h, w) {
        img.clone()
    } else {
        image::imageops::resize(img, rw as u32, rh as u32, FilterType::CatmullRom)
    };
    let frame = normalized_frame(resized.as_raw(), rh, rw, IMAGE_MEAN, IMAGE_STD);
    pack_patches(&frame, PATCH_SIZE, TEMPORAL_PATCH_SIZE, MERGE_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_factor_32() {
        // 512×512 is already a multiple of 32 and within [min,max] → unchanged.
        assert_eq!(
            smart_resize(512, 512, 32, MIN_PIXELS, MAX_PIXELS),
            (512, 512)
        );
    }

    #[test]
    fn pack_shapes_qwen3() {
        // 512×512 → grid [1, 32, 32]; seq = 1·32·32 = 1024; row = 3·2·16·16 = 1536.
        let frame = Array::zeros::<f32>(&[1, 3, 512, 512]).unwrap();
        let (pv, grid) = pack_patches(&frame, 16, 2, 2).unwrap();
        assert_eq!(grid, [1, 32, 32]);
        assert_eq!(pv.shape(), &[1024, 1536]);
    }
}
