//! Shared leaf helpers (mirrors `mlx-gen-sam3`'s `util`): weight-key joining, torch→MLX
//! conv-weight permutes, and a NHWC bilinear resize (the DPT neck/head upsamples).

use mlx_rs::ops::{add, multiply};
use mlx_rs::Array;

use mlx_gen::Result;

/// `"{prefix}.{leaf}"` (or just `leaf` when `prefix` is empty).
pub(crate) fn join(prefix: &str, leaf: &str) -> String {
    if prefix.is_empty() {
        leaf.to_string()
    } else {
        format!("{prefix}.{leaf}")
    }
}

/// Permute a torch conv weight `[out, in, kH, kW]` (OIHW) → MLX `[out, kH, kW, in]` (OHWI).
pub(crate) fn conv_w_ohwi(w: &Array) -> Result<Array> {
    Ok(w.transpose_axes(&[0, 2, 3, 1])?)
}

/// Permute a torch transposed-conv weight `[in, out, kH, kW]` (IOHW) → MLX `[out, kH, kW, in]` (OHWI).
pub(crate) fn conv_transpose_w(w: &Array) -> Result<Array> {
    Ok(w.transpose_axes(&[1, 2, 3, 0])?)
}

/// Build a 1-D bilinear gather: for an axis of length `in_len` resized to `out_len`, return the two
/// integer source indices (`lo`, `hi`) and the fractional weight `frac` (= weight on `hi`) per output
/// position, following torch `interpolate(mode="bilinear")`.
///
/// `align_corners == true` maps output `i` to source `i * (in-1)/(out-1)`; `false` maps to the
/// pixel-center convention `(i + 0.5) * in/out - 0.5` (clamped to `[0, in-1]`).
fn bilinear_axis(in_len: i32, out_len: i32, align_corners: bool) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let mut lo = Vec::with_capacity(out_len as usize);
    let mut hi = Vec::with_capacity(out_len as usize);
    let mut frac = Vec::with_capacity(out_len as usize);
    let last = in_len - 1;
    for i in 0..out_len {
        let src = if align_corners {
            if out_len == 1 {
                0.0
            } else {
                i as f32 * (in_len - 1) as f32 / (out_len - 1) as f32
            }
        } else {
            let s = (i as f32 + 0.5) * in_len as f32 / out_len as f32 - 0.5;
            s.max(0.0)
        };
        let l = src.floor() as i32;
        let l = l.clamp(0, last);
        let h = (l + 1).min(last);
        lo.push(l);
        hi.push(h);
        frac.push((src - l as f32).clamp(0.0, 1.0));
    }
    (lo, hi, frac)
}

/// Resample one spatial axis (`axis` ∈ {1=H, 2=W} of an NHWC tensor) from its current length to
/// `out_len` by bilinear interpolation. Implemented as a gather of the two bracketing rows/cols and
/// a fractional blend, so it runs entirely in MLX (no host loop over pixels).
fn resample_axis(x: &Array, axis: i32, out_len: i32, align_corners: bool) -> Result<Array> {
    let in_len = x.shape()[axis as usize];
    if in_len == out_len {
        return Ok(x.clone());
    }
    let (lo, hi, frac) = bilinear_axis(in_len, out_len, align_corners);
    let lo = Array::from_slice(&lo, &[out_len]);
    let hi = Array::from_slice(&hi, &[out_len]);
    // Broadcast the per-output weight over the gathered tensor: shape [1,…,out_len,…,1].
    let mut wshape = vec![1i32; x.shape().len()];
    wshape[axis as usize] = out_len;
    let w_hi = Array::from_slice(&frac, &wshape);
    let ones = Array::from_slice(&vec![1.0f32; out_len as usize], &wshape);
    let w_lo = mlx_rs::ops::subtract(&ones, &w_hi)?;

    let g_lo = x.take_axis(&lo, axis)?;
    let g_hi = x.take_axis(&hi, axis)?;
    Ok(add(&multiply(&g_lo, &w_lo)?, &multiply(&g_hi, &w_hi)?)?)
}

/// NHWC bilinear resize `[B, H, W, C]` → `[B, out_h, out_w, C]` (torch `interpolate(mode="bilinear")`).
/// Separable: resample H then W. `align_corners` matches the torch flag used at the call site.
pub(crate) fn bilinear_resize(
    x: &Array,
    out_h: i32,
    out_w: i32,
    align_corners: bool,
) -> Result<Array> {
    let y = resample_axis(x, 1, out_h, align_corners)?;
    resample_axis(&y, 2, out_w, align_corners)
}
