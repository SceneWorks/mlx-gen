//! SCAIL-2 conditioning preprocessing — tensor ops that turn user inputs into the latents the DiT
//! consumes (sc-5443). This module owns the **28-channel color-coded mask** build
//! (`extract_and_compress_mask_to_latent`, upstream `wan/utils/scail_utils.py`); the VAE-encode of the
//! reference / pose latents reuses [`mlx_gen_wan::WanVae`], and the CLIP image encode is
//! [`crate::clip::ScailClip`].

use mlx_gen::array::scalar;
use mlx_gen::Result;
use mlx_rs::ops::{concatenate_axis, multiply, split, subtract};
use mlx_rs::{Array, Dtype};

/// A normalized pixel is "on" when the original `[0,255]` value is ≥ 225, i.e. `(225-127.5)/127.5` in
/// the `[-1,1]` mask space (upstream `_ON_THRESH`).
const ON_THRESH: f32 = (225.0 - 127.5) / 127.5;

/// Default temporal-compression stride (the z16 VAE temporal stride): 4 frames → 1 latent frame,
/// packed into the channel axis (×7 colors = 28).
pub const TEMPORAL_STRIDE: usize = 4;

fn f32(x: &Array) -> Result<Array> {
    Ok(x.as_dtype(Dtype::Float32)?)
}

/// `1 - x`.
fn one_minus(x: &Array) -> Result<Array> {
    Ok(subtract(scalar(1.0), x)?)
}

/// `a · b · c`.
fn mul3(a: &Array, b: &Array, c: &Array) -> Result<Array> {
    Ok(multiply(&multiply(a, b)?, c)?)
}

/// Convert a 3-channel RGB color-coded segmentation mask `(3, T, H, W)` in `[-1, 1]` into the
/// 28-channel binary mask latent `(28, T_latent, H/8, W/8)` the DiT's `patch_embedding_mask` consumes
/// — **no VAE**, matching upstream `extract_and_compress_mask_to_latent(additional_spatial_downsample=1)`.
///
/// Pipeline: threshold each channel at [`ON_THRESH`] → the **7 exclusive color classes**
/// (white/red/green/blue/yellow/magenta/cyan as R/G/B AND-products) → **8× area downsample** (exact
/// 8×8 average pool; `H` and `W` must be divisible by 8) → **temporal pack** by `temporal_stride`
/// (frame 0 repeated `stride` times for the lead latent frame; the `stride` frames of each latent step
/// stacked into the channel axis, 7·stride = 28).
pub fn extract_and_compress_mask_to_latent(mask: &Array, temporal_stride: usize) -> Result<Array> {
    let t = mask.shape()[1];
    let h = mask.shape()[2];
    let w = mask.shape()[3];
    // Request-derived mask dims: reject as a typed error rather than abort the worker (F-020/L-A).
    if h % 8 != 0 || w % 8 != 0 {
        return Err(mlx_gen::Error::Msg(format!(
            "scail2 mask: H,W must be divisible by 8 (got {h}x{w})"
        )));
    }

    // (3, T, H, W) → (T, 3, H, W), threshold each channel to {0,1}.
    let m = f32(&mask.transpose_axes(&[1, 0, 2, 3])?)?;
    let chans = split(&m, 3, 1)?; // 3 × (T, 1, H, W)
    let r = f32(&chans[0].gt(scalar(ON_THRESH))?)?;
    let g = f32(&chans[1].gt(scalar(ON_THRESH))?)?;
    let b = f32(&chans[2].gt(scalar(ON_THRESH))?)?;
    let (nr, ng, nb) = (one_minus(&r)?, one_minus(&g)?, one_minus(&b)?);

    // 7 exclusive color classes (T, 7, H, W).
    let white = mul3(&r, &g, &b)?;
    let red = mul3(&r, &ng, &nb)?;
    let green = mul3(&nr, &g, &nb)?;
    let blue = mul3(&nr, &ng, &b)?;
    let yellow = mul3(&r, &g, &nb)?;
    let magenta = mul3(&r, &ng, &b)?;
    let cyan = mul3(&nr, &g, &b)?;
    let binary7 = concatenate_axis(&[&white, &red, &green, &blue, &yellow, &magenta, &cyan], 1)?;

    // 8× area downsample = exact 8×8 average pool: (T,7,H,W) → (T,7,H/8,8,W/8,8) → mean over the blocks.
    let (hl, wl) = (h / 8, w / 8);
    let pooled = binary7
        .reshape(&[t, 7, hl, 8, wl, 8])?
        .mean_axes(&[3, 5], None)?; // (T, 7, hl, wl)

    // Temporal pack: lead latent frame repeats frame 0 `stride` times; T_latent groups of `stride`
    // frames stack into the channel axis → 7·stride channels.
    let stride = temporal_stride as i32;
    let t_lat = (t - 1) / stride + 1;
    let frame0 = pooled.take_axis(Array::from_slice(&[0i32], &[1]), 0)?;
    let lead: Vec<&Array> = (0..stride).map(|_| &frame0).collect();
    let lead = concatenate_axis(&lead, 0)?; // (stride, 7, hl, wl)
    let rest = pooled.take_axis(
        Array::from_slice(&(1..t).collect::<Vec<i32>>(), &[t - 1]),
        0,
    )?;
    let padded = concatenate_axis(&[&lead, &rest], 0)?; // (T_latent·stride, 7, hl, wl)

    Ok(padded
        .reshape(&[t_lat, stride * 7, hl, wl])?
        .transpose_axes(&[1, 0, 2, 3])?) // (28, T_latent, hl, wl)
}
