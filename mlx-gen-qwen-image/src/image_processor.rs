//! Qwen2-VL image processor — port of the fork's hand-rolled `QwenImageProcessor`
//! (`models/qwen/tokenizer/qwen_image_processor.py`), used by Qwen-Image-Edit's reference
//! flow. Pipeline: `smart_resize` → PIL-compatible BICUBIC resize → `/255` → CLIP normalize
//! → temporal-repeat → 9-D patchify → `(N, 1176)` pixel_values + `(1, 3)` grid_thw.
//!
//! Parity (tests/qwen_image_processor.rs): no-resize and upscale are **bit-exact**; the
//! antialiased downscale path matches PIL `Image.BICUBIC` to a measured max of 1/255 (one
//! uint8 quantization level — PIL's fixed-point resampler isn't bit-reproduced, but the
//! Keys kernel, antialias support scaling, and clip8 rounding are). That's well below any
//! meaningful threshold for vision-encoder input, so the float impl stands.

use mlx_rs::Array;

use mlx_gen::Result;

pub const OPENAI_CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
pub const OPENAI_CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// RGB uint8 image in HWC layout.
pub struct ImageInput<'a> {
    pub data: &'a [u8],
    pub height: usize,
    pub width: usize,
}

/// Patchified output ready for the vision encoder.
pub struct ProcessedImage {
    /// `(grid_t·grid_h·grid_w, channel·temporal·patch·patch)` = `(N, 1176)`, f32.
    pub pixel_values: Array,
    /// `(1, 3)` int32: `[grid_t, grid_h, grid_w]`.
    pub grid_thw: Array,
}

#[derive(Debug, Clone)]
pub struct QwenImageProcessor {
    pub min_pixels: i64,
    pub max_pixels: i64,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl Default for QwenImageProcessor {
    fn default() -> Self {
        Self {
            min_pixels: 56 * 56,
            max_pixels: 28 * 28 * 1280,
            patch_size: 14,
            temporal_patch_size: 2,
            merge_size: 2,
            image_mean: OPENAI_CLIP_MEAN,
            image_std: OPENAI_CLIP_STD,
        }
    }
}

impl QwenImageProcessor {
    /// Integer target dims: round each side to a multiple of `patch_size*merge_size`,
    /// then clamp the pixel count into `[min_pixels, max_pixels]`. Uses Python's round-half-
    /// to-even (`round_ties_even`) so `.5` cases match the fork exactly.
    pub fn smart_resize(&self, height: usize, width: usize) -> (usize, usize) {
        let factor = (self.patch_size * self.merge_size) as f64;
        let (h, w) = (height as f64, width as f64);
        let (minp, maxp) = (self.min_pixels as f64, self.max_pixels as f64);

        let mut h_bar = (h / factor).round_ties_even() * factor;
        let mut w_bar = (w / factor).round_ties_even() * factor;
        if h_bar * w_bar > maxp {
            let beta = ((h * w) / maxp).sqrt();
            h_bar = factor.max((h / beta / factor).floor() * factor);
            w_bar = factor.max((w / beta / factor).floor() * factor);
        } else if h_bar * w_bar < minp {
            let beta = (minp / (h * w)).sqrt();
            h_bar = (h * beta / factor).ceil() * factor;
            w_bar = (w * beta / factor).ceil() * factor;
        }
        (h_bar as usize, w_bar as usize)
    }

    pub fn preprocess(&self, image: ImageInput) -> Result<ProcessedImage> {
        let (rh, rw) = self.smart_resize(image.height, image.width);

        // Resize on the uint8 image (matching PIL), then convert to normalized f32 CHW.
        let resized: Vec<f32> = if (image.height, image.width) == (rh, rw) {
            image.data.iter().map(|&p| p as f32).collect()
        } else {
            resize_bicubic_u8(image.data, image.height, image.width, rh, rw)
        };

        // /255, CLIP-normalize, and lay out as CHW; then duplicate across temporal_patch_size
        // (a single frame is repeated, mirroring the fork's `np.repeat` of the last frame).
        let (c, t) = (3usize, self.temporal_patch_size);
        let plane = rh * rw;
        let mut chw = vec![0f32; t * c * plane];
        for ch in 0..c {
            let (mean, std) = (self.image_mean[ch], self.image_std[ch]);
            for y in 0..rh {
                for x in 0..rw {
                    let v = (resized[(y * rw + x) * c + ch] / 255.0 - mean) / std;
                    let chw_idx = ch * plane + y * rw + x;
                    for frame in 0..t {
                        chw[frame * c * plane + chw_idx] = v;
                    }
                }
            }
        }

        // Patchify: (grid_t, temporal, channel, gh/m, m, patch, gw/m, m, patch)
        //   -> transpose (0,3,6,4,7,2,1,5,8) -> (grid_t·grid_h·grid_w, channel·temporal·patch²).
        let p = self.patch_size as i32;
        let m = self.merge_size as i32;
        let (gh, gw) = ((rh / self.patch_size) as i32, (rw / self.patch_size) as i32);
        let nine = Array::from_slice(&chw, &[1, t as i32, c as i32, gh / m, m, p, gw / m, m, p]);
        let patched = nine
            .transpose_axes(&[0, 3, 6, 4, 7, 2, 1, 5, 8])?
            .reshape(&[gh * gw, (c as i32) * (t as i32) * p * p])?;

        let grid_thw = Array::from_slice(&[1i32, gh, gw], &[1, 3]);
        Ok(ProcessedImage {
            pixel_values: patched,
            grid_thw,
        })
    }
}

/// PIL `bicubic_filter` (Keys cubic, a = -0.5).
fn cubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Per-output-pixel resampling coefficients for a 1-D axis resize, matching PIL's
/// `precompute_coeffs`: antialias by scaling the filter support when downscaling, clamp the
/// window to the input bounds, and renormalize the (possibly truncated) weights to sum to 1.
fn precompute_coeffs(in_size: usize, out_size: usize) -> Vec<(usize, Vec<f64>)> {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale; // BICUBIC support = 2.0
    let mut out = Vec::with_capacity(out_size);
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let xmin = ((center - support + 0.5).floor() as i64).max(0) as usize;
        let xmax = ((center + support + 0.5).floor() as i64).min(in_size as i64) as usize;
        let mut weights = Vec::with_capacity(xmax - xmin);
        let mut total = 0.0;
        for x in xmin..xmax {
            let w = cubic((x as f64 - center + 0.5) / filterscale);
            weights.push(w);
            total += w;
        }
        if total != 0.0 {
            for w in &mut weights {
                *w /= total;
            }
        }
        out.push((xmin, weights));
    }
    out
}

/// Two-pass (horizontal then vertical) bicubic resize of a uint8 HWC image, with clip8
/// rounding between/after passes — the structure of PIL's `ImagingResample`. Returns f32
/// HWC with integer-valued samples in `[0, 255]` (i.e. `np.array(img.resize(...))` as float).
pub(crate) fn resize_bicubic_u8(
    src: &[u8],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<f32> {
    let c = 3usize;
    let clip8 = |v: f64| (v.round().clamp(0.0, 255.0)) as f32;

    // Horizontal pass: (in_h, in_w) -> (in_h, out_w).
    let hcoeffs = precompute_coeffs(in_w, out_w);
    let mut horiz = vec![0f32; in_h * out_w * c];
    for y in 0..in_h {
        for (xx, (xmin, w)) in hcoeffs.iter().enumerate() {
            for ch in 0..c {
                let mut acc = 0.0;
                for (k, &wk) in w.iter().enumerate() {
                    acc += src[(y * in_w + xmin + k) * c + ch] as f64 * wk;
                }
                horiz[(y * out_w + xx) * c + ch] = clip8(acc);
            }
        }
    }

    // Vertical pass: (in_h, out_w) -> (out_h, out_w).
    let vcoeffs = precompute_coeffs(in_h, out_h);
    let mut out = vec![0f32; out_h * out_w * c];
    for (yy, (ymin, w)) in vcoeffs.iter().enumerate() {
        for x in 0..out_w {
            for ch in 0..c {
                let mut acc = 0.0;
                for (k, &wk) in w.iter().enumerate() {
                    acc += horiz[((ymin + k) * out_w + x) * c + ch] as f64 * wk;
                }
                out[(yy * out_w + x) * c + ch] = clip8(acc);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_matches_reference_cases() {
        let p = QwenImageProcessor::default();
        assert_eq!(p.smart_resize(56, 84), (56, 84)); // already aligned -> no-op
        assert_eq!(p.smart_resize(200, 150), (196, 140)); // downscale
        assert_eq!(p.smart_resize(20, 20), (56, 56)); // upscale to min_pixels
    }
}
