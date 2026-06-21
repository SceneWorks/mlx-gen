//! Qwen2-VL image processor — port of the fork's hand-rolled `QwenImageProcessor`
//! (`models/qwen/tokenizer/qwen_image_processor.py`), used by Qwen-Image-Edit's reference
//! flow. Pipeline: `smart_resize` → PIL-compatible BICUBIC resize → `/255` → CLIP normalize
//! → temporal-repeat → 9-D patchify → `(N, 1176)` pixel_values + `(1, 3)` grid_thw.
//!
//! The PIL-exact resamplers (`resize_bicubic_u8` / `resize_lanczos_u8`) live in core
//! ([`mlx_gen::image`]); they're re-exported here so this module and the VL tokenizer keep
//! importing them from `crate::image_processor`.

use mlx_rs::Array;

use mlx_gen::{Error, Result};

pub(crate) use mlx_gen::image::{resize_bicubic_u8, resize_lanczos_u8};

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
}

impl Default for QwenImageProcessor {
    fn default() -> Self {
        Self {
            min_pixels: 56 * 56,
            max_pixels: 28 * 28 * 1280,
            patch_size: 14,
            temporal_patch_size: 2,
            merge_size: 2,
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
        // The pub re-exported processor indexes `data` as `h*w*3` below; reject a mismatched buffer
        // up front (the registered edit path validates upstream, a direct caller does not) (F-020/L-A).
        let expected = image.height * image.width * 3;
        if image.data.len() != expected {
            return Err(Error::Msg(format!(
                "qwen image processor: input buffer {} bytes != {}x{}x3 ({expected})",
                image.data.len(),
                image.width,
                image.height
            )));
        }
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
            let (mean, std) = (OPENAI_CLIP_MEAN[ch], OPENAI_CLIP_STD[ch]);
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
