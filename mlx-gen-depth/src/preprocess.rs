//! Host-side image (de)serialization for the depth estimator: an arbitrary RGB8 image →
//! the model's normalized NHWC input, and a raw depth map → a normalized single-channel
//! depth-control image.
//!
//! Matches `depth-anything/Depth-Anything-V2-Small-hf`'s `preprocessor_config.json`:
//! resize the longest tier to a fixed **518×518** square (the default DA-V2 inference size,
//! keeping the pos-embed grid at 37×37 — no interpolation), rescale to `[0,1]`, then ImageNet
//! normalize (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`).
//!
//! The estimator is a *preprocessor*: the produced depth map is min/max-normalized to `[0,1]`
//! and emitted as a grayscale-broadcast RGB control image — the standard ControlNet depth
//! convention (near = bright). The host worker resizes it back to the generation resolution.

use mlx_rs::Array;

use mlx_gen::Result;

/// The fixed model input side (`preprocessor_config.json` `size` = 518; multiple of `patch_size` 14).
pub const INPUT_SIZE: i32 = 518;

/// ImageNet normalization mean (per RGB channel).
pub const IMAGE_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// ImageNet normalization std (per RGB channel).
pub const IMAGE_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Nearest-neighbor-free **bilinear** resample of an RGB8 HWC byte image to `out·out`, in `[0,1]`
/// float, returned planar-free as interleaved HWC `f32` (length `out·out·3`). align_corners=False
/// (the torch/PIL pixel-center convention DA-V2's `BICUBIC`-tier resize approximates; bilinear is a
/// faithful, cheaper preprocessing resize for a depth hint).
fn resize_rgb8_to_unit(rgb: &[u8], in_h: usize, in_w: usize, out: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; out * out * 3];
    let sx = in_w as f32 / out as f32;
    let sy = in_h as f32 / out as f32;
    for oy in 0..out {
        let fy = ((oy as f32 + 0.5) * sy - 0.5).max(0.0);
        let y0 = (fy.floor() as usize).min(in_h - 1);
        let y1 = (y0 + 1).min(in_h - 1);
        let wy = fy - y0 as f32;
        for ox in 0..out {
            let fx = ((ox as f32 + 0.5) * sx - 0.5).max(0.0);
            let x0 = (fx.floor() as usize).min(in_w - 1);
            let x1 = (x0 + 1).min(in_w - 1);
            let wx = fx - x0 as f32;
            for c in 0..3 {
                let p = |y: usize, x: usize| rgb[(y * in_w + x) * 3 + c] as f32;
                let top = p(y0, x0) * (1.0 - wx) + p(y0, x1) * wx;
                let bot = p(y1, x0) * (1.0 - wx) + p(y1, x1) * wx;
                let v = (top * (1.0 - wy) + bot * wy) / 255.0;
                buf[(oy * out + ox) * 3 + c] = v;
            }
        }
    }
    buf
}

/// Arbitrary RGB8 HWC image (`width`·`height`·3 bytes) → the model's NHWC f32 input
/// `[1, 518, 518, 3]`, resized + rescaled + ImageNet-normalized. Uses the default [`INPUT_SIZE`].
pub fn rgb8_to_input(rgb: &[u8], width: u32, height: u32) -> Result<Array> {
    rgb8_to_input_sized(rgb, width, height, INPUT_SIZE)
}

/// [`rgb8_to_input`] at an explicit square `size` (must be a multiple of the backbone patch size so
/// the token grid matches the loaded `position_embeddings`). The default model is square-518; this
/// is the size-parametric form the estimator calls with `config().image_size`.
pub fn rgb8_to_input_sized(rgb: &[u8], width: u32, height: u32, size: i32) -> Result<Array> {
    let out = size as usize;
    let mut unit = resize_rgb8_to_unit(rgb, height as usize, width as usize, out);
    // ImageNet normalize, in place.
    for (i, v) in unit.iter_mut().enumerate() {
        let c = i % 3;
        *v = (*v - IMAGE_MEAN[c]) / IMAGE_STD[c];
    }
    Ok(Array::from_slice(&unit, &[1, size, size, 3]))
}

/// A raw depth map `[H, W]` (f32, model units) → an RGB8 HWC depth-control image (`H·W·3` bytes),
/// min/max-normalized to `[0,255]` and broadcast across the three channels (near = bright, the
/// standard ControlNet depth convention). A degenerate (flat) map yields a uniform mid-gray.
pub fn depth_to_control_rgb8(depth: &[f32], h: usize, w: usize) -> Vec<u8> {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &d in depth {
        if d.is_finite() {
            lo = lo.min(d);
            hi = hi.max(d);
        }
    }
    let span = hi - lo;
    let mut out = vec![0u8; h * w * 3];
    for (i, &d) in depth.iter().enumerate() {
        let norm = if span > f32::EPSILON {
            ((d - lo) / span).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let v = (norm * 255.0).round() as u8;
        out[i * 3] = v;
        out[i * 3 + 1] = v;
        out[i * 3 + 2] = v;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::indexing::IndexOp;

    #[test]
    fn input_shape_and_normalization() {
        // A 4×4 mid-gray image → [1,518,518,3]; gray 128/255 ≈ 0.502 normalized per channel.
        let rgb = vec![128u8; 4 * 4 * 3];
        let inp = rgb8_to_input(&rgb, 4, 4).unwrap();
        assert_eq!(inp.shape(), &[1, INPUT_SIZE, INPUT_SIZE, 3]);
        let expected_r = (128.0 / 255.0 - IMAGE_MEAN[0]) / IMAGE_STD[0];
        let got: f32 = inp.index((0, 0, 0, 0)).item();
        assert!(
            (got - expected_r).abs() < 1e-4,
            "channel-0 normalize: got {got}, want {expected_r}"
        );
    }

    #[test]
    fn depth_normalizes_to_full_range() {
        // A linear ramp 0..1 over 4 pixels → min maps to 0, max to 255.
        let depth = vec![0.0f32, 0.33, 0.66, 1.0];
        let out = depth_to_control_rgb8(&depth, 1, 4);
        assert_eq!(out.len(), 4 * 3);
        assert_eq!(out[0], 0, "min depth → 0");
        assert_eq!(out[(3) * 3], 255, "max depth → 255");
        // Grayscale broadcast: r==g==b.
        assert!(out.chunks(3).all(|p| p[0] == p[1] && p[1] == p[2]));
    }

    #[test]
    fn flat_depth_is_mid_gray() {
        let depth = vec![7.0f32; 9];
        let out = depth_to_control_rgb8(&depth, 3, 3);
        assert!(
            out.iter().all(|&b| b == 128),
            "a degenerate (flat) depth map must be a uniform mid-gray, got {out:?}"
        );
    }
}
