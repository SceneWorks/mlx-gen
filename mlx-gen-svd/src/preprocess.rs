//! Faithful port of diffusers' SVD CLIP-image preprocessing — `_resize_with_antialiasing`
//! (`pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py`): a separable gaussian
//! blur (reflect-padded, depthwise) followed by
//! `torch.nn.functional.interpolate(mode="bicubic", align_corners=True)` to 224×224.
//!
//! The reference runs this resize in the `[-1, 1]` normalized space — diffusers normalizes the
//! image *before* resizing and un-normalizes *after* (see `_encode_image`) — so it cannot reuse the
//! core u8 fixed-point PIL resamplers (those are integer `[0, 255]` and use a different cubic kernel
//! `A = -0.5` plus PIL's own antialias model). Here:
//!   * sigma / kernel-size follow skimage/diffusers (`sigma = max((factor − 1)/2, 0.001)`,
//!     `ks = int(max(4·sigma, 3))` forced odd); sigma is computed in f64 then cast to f32 for the
//!     gaussian, matching diffusers (`torch.tensor([sigma], dtype=input.dtype)`);
//!   * the gaussian is `exp(−x²/(2σ²))` normalized, computed in f32;
//!   * the interpolation is torch's cubic-convolution (`A = −0.75`) with align-corners source
//!     coordinates and clamped (replicate) borders.
//!
//! This replaces the earlier core PIL-bicubic approximation in the CLIP path (sc-3412); the VAE
//! preprocess uses PIL lanczos and stays on the core resampler.

/// Reflect-101 index map (torch `pad(mode="reflect")`): mirrors without repeating the edge sample,
/// e.g. for `n = 5` the position `-1 → 1` and `5 → 3`.
fn reflect_index(j: isize, n: isize) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let mut k = j.rem_euclid(period);
    if k >= n {
        k = period - k;
    }
    k as usize
}

/// diffusers `_gaussian(window, sigma)` in f32 (`sigma.dtype == input.dtype == f32`): the normalized
/// `exp(−x²/(2σ²))` over `x = i − window//2` (`+0.5` shift only for even windows, which we never hit).
fn gaussian_kernel(window: usize, sigma: f32) -> Vec<f32> {
    let half = (window / 2) as f32;
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut g = Vec::with_capacity(window);
    let mut sum = 0.0f32;
    for i in 0..window {
        let mut x = i as f32 - half;
        if window.is_multiple_of(2) {
            x += 0.5;
        }
        let v = (-(x * x) / two_sigma_sq).exp();
        g.push(v);
        sum += v;
    }
    for v in &mut g {
        *v /= sum;
    }
    g
}

/// diffusers kernel size from a (f64) sigma: `int(max(2.0·2·sigma, 3))`, forced odd. `as usize`
/// truncates toward zero, matching Python `int()` for the non-negative argument.
fn kernel_size(sigma: f64) -> usize {
    let mut k = (4.0 * sigma).max(3.0) as usize;
    if k.is_multiple_of(2) {
        k += 1;
    }
    k
}

/// torch bicubic cubic-convolution weights (`A = −0.75`) for one output coordinate under
/// `align_corners = True`. Returns `(base, [w₋₁, w₀, w₁, w₂])`; the four taps sit at `base−1..=base+2`
/// and are clamped to `[0, in−1]` by the caller (torch's bounded fetch / replicate border).
fn cubic_weights(out_idx: usize, in_size: usize, out_size: usize) -> (isize, [f32; 4]) {
    const A: f64 = -0.75;
    let cubic1 = |x: f64| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let cubic2 = |x: f64| (((A * x) - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    let scale = if out_size > 1 {
        (in_size as f64 - 1.0) / (out_size as f64 - 1.0)
    } else {
        0.0
    };
    let real = scale * out_idx as f64;
    let base = real.floor();
    let t = real - base;
    let w = [
        cubic2(t + 1.0) as f32,
        cubic1(t) as f32,
        cubic1(1.0 - t) as f32,
        cubic2(2.0 - t) as f32,
    ];
    (base as isize, w)
}

/// Horizontal separable pass over a planar `[3, h, w]` buffer applying `kernel` with reflect pad.
fn blur_horizontal(inp: &[f32], h: usize, w: usize, kernel: &[f32]) -> Vec<f32> {
    let pad = (kernel.len() - 1) / 2;
    let mut out = vec![0.0f32; 3 * h * w];
    for c in 0..3 {
        let plane = c * h * w;
        for y in 0..h {
            let row = plane + y * w;
            for x in 0..w {
                let mut acc = 0.0f32;
                for (k, &wk) in kernel.iter().enumerate() {
                    let sx = reflect_index(x as isize + k as isize - pad as isize, w as isize);
                    acc += wk * inp[row + sx];
                }
                out[row + x] = acc;
            }
        }
    }
    out
}

/// Vertical separable pass over a planar `[3, h, w]` buffer applying `kernel` with reflect pad.
fn blur_vertical(inp: &[f32], h: usize, w: usize, kernel: &[f32]) -> Vec<f32> {
    let pad = (kernel.len() - 1) / 2;
    let mut out = vec![0.0f32; 3 * h * w];
    for c in 0..3 {
        let plane = c * h * w;
        for y in 0..h {
            let row = plane + y * w;
            for x in 0..w {
                let mut acc = 0.0f32;
                for (k, &wk) in kernel.iter().enumerate() {
                    let sy = reflect_index(y as isize + k as isize - pad as isize, h as isize);
                    acc += wk * inp[plane + sy * w + x];
                }
                out[row + x] = acc;
            }
        }
    }
    out
}

/// Horizontal bicubic resize `in_w → out_w` over a planar `[3, h, in_w]` buffer.
fn resize_horizontal(inp: &[f32], h: usize, in_w: usize, out_w: usize) -> Vec<f32> {
    let coeffs: Vec<(isize, [f32; 4])> =
        (0..out_w).map(|o| cubic_weights(o, in_w, out_w)).collect();
    let mut out = vec![0.0f32; 3 * h * out_w];
    for c in 0..3 {
        let in_plane = c * h * in_w;
        let out_plane = c * h * out_w;
        for y in 0..h {
            let in_row = in_plane + y * in_w;
            let out_row = out_plane + y * out_w;
            for (o, (base, w)) in coeffs.iter().enumerate() {
                let mut acc = 0.0f32;
                for (m, &wm) in w.iter().enumerate() {
                    let idx =
                        (base - 1 + m as isize).clamp(0, in_w as isize - 1) as usize;
                    acc += wm * inp[in_row + idx];
                }
                out[out_row + o] = acc;
            }
        }
    }
    out
}

/// Vertical bicubic resize `in_h → out_h` over a planar `[3, in_h, w]` buffer.
fn resize_vertical(inp: &[f32], in_h: usize, w: usize, out_h: usize) -> Vec<f32> {
    let coeffs: Vec<(isize, [f32; 4])> =
        (0..out_h).map(|o| cubic_weights(o, in_h, out_h)).collect();
    let mut out = vec![0.0f32; 3 * out_h * w];
    for c in 0..3 {
        let in_plane = c * in_h * w;
        let out_plane = c * out_h * w;
        for (o, (base, wts)) in coeffs.iter().enumerate() {
            let out_row = out_plane + o * w;
            for x in 0..w {
                let mut acc = 0.0f32;
                for (m, &wm) in wts.iter().enumerate() {
                    let iy = (base - 1 + m as isize).clamp(0, in_h as isize - 1) as usize;
                    acc += wm * inp[in_plane + iy * w + x];
                }
                out[out_row + x] = acc;
            }
        }
    }
    out
}

/// Diffusers `_encode_image` CLIP preprocessing through the antialiased resize: an RGB8 HWC image
/// `[in_h, in_w, 3]` → the resized unit image HWC `[out_h, out_w, 3]` in `[0, 1]`, ready for the
/// CLIP mean/std normalize. Mirrors `image·2−1 → _resize_with_antialiasing → (image+1)/2`.
pub fn resize_with_antialiasing_unit(
    rgb8: &[u8],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<f32> {
    // RGB8 HWC → planar [3, in_h, in_w] in [-1, 1].
    let mut buf = vec![0.0f32; 3 * in_h * in_w];
    for y in 0..in_h {
        for x in 0..in_w {
            for c in 0..3 {
                let v = rgb8[(y * in_w + x) * 3 + c] as f32 / 255.0;
                buf[c * in_h * in_w + y * in_w + x] = v * 2.0 - 1.0;
            }
        }
    }

    // sigma in f64 (diffusers computes factors/sigmas in python float, then casts to the f32 tensor).
    let factor_h = in_h as f64 / out_h as f64;
    let factor_w = in_w as f64 / out_w as f64;
    let sigma_h = ((factor_h - 1.0) / 2.0).max(0.001);
    let sigma_w = ((factor_w - 1.0) / 2.0).max(0.001);
    let ky = kernel_size(sigma_h);
    let kx = kernel_size(sigma_w);
    let kernel_y = gaussian_kernel(ky, sigma_h as f32);
    let kernel_x = gaussian_kernel(kx, sigma_w as f32);

    // _gaussian_blur2d: horizontal (kernel_x) then vertical (kernel_y), reflect-padded.
    let blurred = blur_horizontal(&buf, in_h, in_w, &kernel_x);
    let blurred = blur_vertical(&blurred, in_h, in_w, &kernel_y);

    // interpolate(bicubic, align_corners=True): horizontal then vertical.
    let resized = resize_horizontal(&blurred, in_h, in_w, out_w);
    let resized = resize_vertical(&resized, in_h, out_w, out_h);

    // planar [-1,1] → HWC [0,1].
    let mut hwc = vec![0.0f32; out_h * out_w * 3];
    for c in 0..3 {
        let plane = c * out_h * out_w;
        for y in 0..out_h {
            for x in 0..out_w {
                let v = resized[plane + y * out_w + x];
                hwc[(y * out_w + x) * 3 + c] = (v + 1.0) * 0.5;
            }
        }
    }
    hwc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_index_mirrors_without_repeating_edge() {
        // n=5: ... 2 1 | 0 1 2 3 4 | 3 2 ...
        assert_eq!(reflect_index(-1, 5), 1);
        assert_eq!(reflect_index(-2, 5), 2);
        assert_eq!(reflect_index(0, 5), 0);
        assert_eq!(reflect_index(4, 5), 4);
        assert_eq!(reflect_index(5, 5), 3);
        assert_eq!(reflect_index(6, 5), 2);
        assert_eq!(reflect_index(7, 1), 0); // degenerate axis
    }

    #[test]
    fn cubic_weights_are_a_delta_at_integer_coordinates() {
        // align_corners maps out 0 and out=in-1 exactly onto input grid points → passthrough.
        let (base, w) = cubic_weights(0, 10, 4);
        assert_eq!(base, 0);
        assert!((w[1] - 1.0).abs() < 1e-6);
        assert!(w[0].abs() < 1e-6 && w[2].abs() < 1e-6 && w[3].abs() < 1e-6);
    }

    #[test]
    fn gaussian_kernel_is_normalized_and_symmetric() {
        let k = gaussian_kernel(5, 1.0);
        let sum: f32 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum {sum}");
        assert!((k[0] - k[4]).abs() < 1e-7 && (k[1] - k[3]).abs() < 1e-7);
        assert!(k[2] > k[1] && k[1] > k[0]); // peaked at the center
    }

    #[test]
    fn kernel_size_matches_diffusers_formula() {
        assert_eq!(kernel_size(0.001), 3); // max(0.004, 3) → 3
        assert_eq!(kernel_size(0.5), 3); // max(2, 3) → 3
        assert_eq!(kernel_size(1.0), 5); // int(4) = 4 → odd 5
        assert_eq!(kernel_size(1.2857), 5); // int(5.14) = 5
        assert_eq!(kernel_size(1.786), 7); // int(7.14) = 7
    }
}
