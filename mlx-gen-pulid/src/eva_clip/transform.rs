//! EVA-CLIP input transform (sc-3073 durable nugget, folded here): the `face_features_image`
//! (512² aligned, background-whitened grayscale, NHWC f32 in [0,1] from `mlx-gen-face`) is resized
//! to 336² and normalized with the OpenAI/EVA mean/std before the ViT.
//!
//! The reference is torchvision `resize(t, 336, BICUBIC)` on a **float** tensor — antialiased
//! (downscale) Keys-cubic (a=-0.5), computed in float (NO u8 quantization, NO clamp). This is a
//! distinct path from core `resize_bicubic_u8` (PIL's 8-bit fixed-point), so the float resize lives
//! here. The downstream gate is ArcFace-cosine (cross-encoder, not bit-exact), so a faithful float
//! bicubic — not byte-parity — is what's required.

use mlx_rs::ops::{divide, subtract};
use mlx_rs::Array;

use mlx_gen::Result;

/// OpenAI/EVA normalization constants (`eva_clip/constants.py`).
pub const EVA_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
pub const EVA_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// Keys cubic (a = -0.5), support 2.0 — the bicubic filter (matches PIL/torchvision).
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

/// torchvision/PIL `precompute_coeffs`: antialias by scaling the filter support when downscaling,
/// clamp the window, renormalize. Returns `(window_start, weights)` per output pixel (f64).
fn coeffs(in_size: usize, out_size: usize, antialias: bool) -> Vec<(usize, Vec<f64>)> {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = if antialias { scale.max(1.0) } else { 1.0 };
    let support = 2.0 * filterscale;
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

/// Separable float bicubic resize of an HWC f32 image (3 channels), accumulated in f64. No
/// quantization or clamp — torchvision's float `antialias=True` bicubic.
pub fn resize_bicubic_f32(
    src: &[f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<f32> {
    let c = 3usize;
    // Horizontal: (in_h, in_w) -> (in_h, out_w)
    let hc = coeffs(in_w, out_w, true);
    let mut horiz = vec![0f32; in_h * out_w * c];
    for y in 0..in_h {
        for (xx, (xmin, w)) in hc.iter().enumerate() {
            for ch in 0..c {
                let mut acc = 0.0f64;
                for (k, &wk) in w.iter().enumerate() {
                    acc += src[(y * in_w + xmin + k) * c + ch] as f64 * wk;
                }
                horiz[(y * out_w + xx) * c + ch] = acc as f32;
            }
        }
    }
    // Vertical: (in_h, out_w) -> (out_h, out_w)
    let vc = coeffs(in_h, out_h, true);
    let mut out = vec![0f32; out_h * out_w * c];
    for (yy, (ymin, w)) in vc.iter().enumerate() {
        for x in 0..out_w {
            for ch in 0..c {
                let mut acc = 0.0f64;
                for (k, &wk) in w.iter().enumerate() {
                    acc += horiz[((ymin + k) * out_w + x) * c + ch] as f64 * wk;
                }
                out[(yy * out_w + x) * c + ch] = acc as f32;
            }
        }
    }
    out
}

/// Full EVA transform: NHWC `[1, H, W, 3]` f32 in [0,1] → resized to `size²` (bicubic) and
/// normalized `(x - mean) / std` per channel. Returns NHWC `[1, size, size, 3]`.
pub fn eva_transform(ffi_nhwc: &Array, size: i32) -> Result<Array> {
    let sh = ffi_nhwc.shape();
    let (b, in_h, in_w) = (sh[0], sh[1] as usize, sh[2] as usize);
    assert_eq!(b, 1, "eva_transform handles a single image");
    let flat = ffi_nhwc.as_dtype(mlx_rs::Dtype::Float32)?.reshape(&[-1])?;
    let src: Vec<f32> = flat.as_slice::<f32>().to_vec();
    let resized = resize_bicubic_f32(&src, in_h, in_w, size as usize, size as usize);
    let resized = Array::from_slice(&resized, &[1, size, size, 3]);
    normalize(&resized)
}

/// Per-channel `(x - mean) / std` over an NHWC array (channels last).
pub fn normalize(x: &Array) -> Result<Array> {
    let mean = Array::from_slice(&EVA_MEAN, &[1, 1, 1, 3]);
    let std = Array::from_slice(&EVA_STD, &[1, 1, 1, 3]);
    Ok(divide(&subtract(x, &mean)?, &std)?)
}
