//! Image helpers split across the backend boundary (epic 3720, Appendix A):
//!
//! - The PIL-compatible resampling ([`resize_bicubic_u8`] & friends) and the host-side mask /
//!   geometry ops ([`contain_box`], [`outpaint_border_mask`], [`union_masks`]) are pure and now live
//!   in [`gen_core::imageops`]; they are re-exported here so the historical `mlx_gen::image::…`
//!   paths keep resolving.
//! - [`decoded_to_image`] — the VAE-decoded-tensor → [`Image`] denormalize/quantize step — operates
//!   on an `mlx_rs::Array` and stays here.

use mlx_rs::ops::{add, maximum, minimum, multiply, round};
use mlx_rs::Array;

use crate::array::scalar;
use crate::media::Image;
use crate::{Error, Result};

pub use gen_core::imageops::*;

/// Denormalize a VAE-decoded tensor to an RGB8 [`Image`]: `clip(x·0.5 + 0.5, 0, 1)` → drop the
/// singleton temporal axis (5-D → 4-D) → NCHW→NHWC → `(x·255).round()` → `u8` (batch must be 1).
/// Identical across the Z-Image and Qwen-Image pipelines (the decoded tensor must already be f32).
pub fn decoded_to_image(decoded: &Array) -> Result<Image> {
    // Rank + dtype hardening (F-064): the readback below is `as_slice::<f32>()`, which reinterprets
    // the raw buffer — a bf16/f16 tensor (one missed `.as_dtype(Float32)` in any of ~20 provider
    // decode paths) would mis-read the bytes and abort the process. Reject a wrong rank up front, and
    // defensively cast to f32 before the arithmetic (a no-op for the f32 tensors every caller passes).
    let rank = decoded.shape().len();
    if rank != 4 && rank != 5 {
        return Err(Error::Msg(format!(
            "decoded_to_image: expected a 4-D (NCHW) or 5-D (NCTHW) tensor, got rank {rank}"
        )));
    }
    let decoded = decoded.as_dtype(mlx_rs::Dtype::Float32)?;
    let half = scalar(0.5);
    // denormalize: clip(x*0.5 + 0.5, 0, 1)
    let x = add(&multiply(&decoded, &half)?, &half)?;
    let x = minimum(&maximum(&x, scalar(0.0))?, scalar(1.0))?;
    // drop the singleton temporal axis if present (5-D → 4-D)
    let x = if x.shape().len() == 5 {
        x.squeeze_axes(&[2])?
    } else {
        x
    };
    // NCHW → NHWC
    let x = x.transpose_axes(&[0, 2, 3, 1])?;
    // (x*255).round() to integer pixel values.
    let x = round(&multiply(&x, scalar(255.0))?, 0)?;

    let sh = x.shape();
    // Batch-correct + overflow-safe (F-082): this path emits a single image, so reject B>1 (the
    // per-image pipelines call with B==1) rather than silently keeping only batch 0, and size in
    // usize / flatten via -1 to avoid the u32/i32 product overflow at large resolutions.
    if sh[0] != 1 {
        return Err(Error::Msg(format!(
            "decoded_to_image: expected batch size 1, got {}",
            sh[0]
        )));
    }
    let (h, w, c) = (sh[1] as usize, sh[2] as usize, sh[3] as usize);
    let n = h * w * c;
    // `transpose_axes` yields a strided view; `reshape` re-materializes in C-order, so the slice is
    // logical NHWC.
    let flat = x.reshape(&[-1])?;
    let pixels: Vec<u8> = flat.as_slice::<f32>()[..n]
        .iter()
        .map(|&v| v as u8)
        .collect();
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

/// Reject generation dimensions that aren't multiples of 16 — the latent-pack requirement shared by
/// the FLUX-family pipelines (FLUX.1 / FLUX.2). `family` names the model in the error message so the
/// single shared check reads like the per-crate copies it replaces (F-083).
pub fn validate_multiple_of_16(width: u32, height: u32, family: &str) -> Result<()> {
    if !width.is_multiple_of(16) || !height.is_multiple_of(16) {
        return Err(Error::Msg(format!(
            "{family}: width and height must be multiples of 16, got {width}x{height}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `resize_u8` (now in gen-core) must be **bit-identical** to PIL `Image.BICUBIC` (the
    /// fixed-point integer path), not merely close — this is what gives the conditioning images
    /// pixel-parity with the fork (sc-2465: an f64-coefficient resampler diverged ±1–2 ULP at
    /// gradient cliffs → 24% e2e px>8). The golden tensor is read via `crate::weights::Weights`
    /// (MLX), so this test stays in mlx-gen. Golden via `tools/dump_pil_resize_golden.py`.
    #[test]
    #[ignore = "needs tools/golden/pil_resize_golden.safetensors (run tools/dump_pil_resize_golden.py)"]
    fn resize_bicubic_matches_pil_512_to_384() {
        let g = crate::weights::Weights::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tools/golden/pil_resize_golden.safetensors"
        ))
        .unwrap();
        // Sawtooth: `(x+y)%256`, ×2, ×3 — sharp 255→0 cliffs where bicubic implementations diverge.
        let mut saw = Vec::with_capacity(512 * 512 * 3);
        let mut smo = Vec::with_capacity(512 * 512 * 3);
        for y in 0..512u32 {
            for x in 0..512u32 {
                let b = (y + x) % 256;
                saw.push(b as u8);
                saw.push(((b * 2) % 256) as u8);
                saw.push(((b * 3) % 256) as u8);
                let v = ((x + y) / 4).min(255) as u8;
                smo.push(v);
                smo.push(v);
                smo.push(v);
            }
        }
        let cmp = |got: &[f32], pil: &[i32]| -> (usize, i32) {
            assert_eq!(got.len(), pil.len(), "len");
            got.iter()
                .zip(pil)
                .fold((0usize, 0i32), |(n, m), (&gv, &pv)| {
                    let d = (gv as i32 - pv).abs();
                    (n + (d != 0) as usize, m.max(d))
                })
        };
        let (saw_diff, saw_max) = cmp(
            &resize_bicubic_u8(&saw, 512, 512, 384, 384).unwrap(),
            g.require("pil384").unwrap().as_slice::<i32>(),
        );
        let (smo_diff, smo_max) = cmp(
            &resize_bicubic_u8(&smo, 512, 512, 384, 384).unwrap(),
            g.require("pil384_smooth").unwrap().as_slice::<i32>(),
        );
        println!("vs PIL 512->384: sawtooth {saw_diff} diff (max {saw_max}), smooth {smo_diff} diff (max {smo_max})");
        assert_eq!(
            saw_diff, 0,
            "resize_u8 must bit-match PIL BICUBIC on the cliff gradient"
        );
        assert_eq!(
            smo_diff, 0,
            "resize_u8 must bit-match PIL BICUBIC on the smooth ramp"
        );
    }
}
