//! `mlx-gen-depth` — native-MLX **Depth Anything V2** monocular depth estimator for mlx-gen
//! (epic 8236, sc-8242).
//!
//! A plain utility *preprocessor* (not a generation-registry provider): an arbitrary RGB image →
//! a normalized single-channel depth-control image, with **no Python / torch**. It is the auto
//! depth source for the Fun-Controlnet-Union depth tier — the sibling of the host-side canny / pose
//! preprocessors — but, unlike those pure-raster ones, depth needs real neural inference, so it runs
//! on MLX (Apple-silicon).
//!
//! ## Architecture (port of the HF `transformers` `DepthAnythingForDepthEstimation`)
//! * [`backbone::Dinov2Backbone`] — DINOv2 ViT-S/14 encoder; returns the four `out_indices`
//!   ([3,6,9,12]) hidden states.
//! * [`neck::DptNeck`] — DPT reassemble (per-level 1×1 projection + factor resize) + 3×3 projection
//!   (`convs`) + RefineNet feature-fusion stage.
//! * [`head::DepthHead`] — `conv1` → ×N bilinear upsample → `conv2`+ReLU → `conv3`+ReLU → `[B,H,W]`.
//!
//! ## Variant / weights
//! Default is **Small** (ViT-S/14): `depth-anything/Depth-Anything-V2-Small-hf`
//! (apache-2.0, **ungated**, ships standard `model.safetensors` — no re-host needed). The Base/Large
//! `-hf` checkpoints share the module graph and plug in via [`config::DepthAnythingConfig`].
//!
//! ## Public API
//! [`DepthAnythingV2::from_dir`] / [`DepthAnythingV2::from_weights`] load the model;
//! [`DepthAnythingV2::estimate_control_rgb8`] takes an arbitrary RGB8 image and returns a
//! min/max-normalized grayscale-broadcast RGB depth-control image (same `width`·`height`).

pub mod backbone;
pub mod config;
pub mod head;
pub mod neck;
pub mod preprocess;
mod util;

use std::path::Path;

use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

pub use config::DepthAnythingConfig;

/// The loaded Depth Anything V2 estimator (backbone + neck + head).
pub struct DepthAnythingV2 {
    backbone: backbone::Dinov2Backbone,
    neck: neck::DptNeck,
    head: head::DepthHead,
    cfg: DepthAnythingConfig,
}

impl DepthAnythingV2 {
    /// Load from a directory holding the transformers checkpoint (`model.safetensors` + `config.json`)
    /// at the **Small** default config.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let w = Weights::from_dir(dir)?;
        Self::from_weights(&w, DepthAnythingConfig::small())
    }

    /// Load from already-read [`Weights`] with an explicit config (for Base/Large or testing).
    pub fn from_weights(w: &Weights, cfg: DepthAnythingConfig) -> Result<Self> {
        let backbone = backbone::Dinov2Backbone::from_weights(w, "backbone", cfg.clone())?;
        let neck = neck::DptNeck::from_weights(w, "neck", &cfg)?;
        let head = head::DepthHead::from_weights(w, "head", &cfg)?;
        Ok(Self {
            backbone,
            neck,
            head,
            cfg,
        })
    }

    /// The loaded configuration.
    pub fn config(&self) -> &DepthAnythingConfig {
        &self.cfg
    }

    /// Run the model on a normalized NHWC input `[1, 518, 518, 3]` → a depth map `[H, W]` (f32,
    /// model units; relative depth). Exposed for parity/testing; most callers want
    /// [`estimate_control_rgb8`](Self::estimate_control_rgb8).
    pub fn forward(&self, pixel_values: &Array) -> Result<Array> {
        let grid = self.cfg.grid();
        let hidden = self.backbone.forward(pixel_values)?;
        if hidden.len() != 4 {
            return Err(Error::Msg(format!(
                "depth backbone produced {} captured states (expected 4)",
                hidden.len()
            )));
        }
        let fused = self.neck.forward(&hidden, grid, self.cfg.hidden_size)?;
        let depth = self.head.forward(&fused, grid)?; // [1, H, W]
        let sh = depth.shape();
        Ok(depth.reshape(&[sh[1], sh[2]])?)
    }

    /// Arbitrary RGB8 HWC image (`width`·`height`·3 bytes) → a depth-control RGB8 image of the SAME
    /// `width`·`height` (min/max-normalized, grayscale broadcast; near = bright). The model runs at
    /// its native 518² and the result is bilinearly resized back to the input dimensions on the host.
    pub fn estimate_control_rgb8(&self, rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
        let expected = width as usize * height as usize * 3;
        if rgb.len() != expected {
            return Err(Error::Msg(format!(
                "depth input buffer is {} bytes, expected {expected} ({width}×{height}×3)",
                rgb.len()
            )));
        }
        let input = preprocess::rgb8_to_input_sized(rgb, width, height, self.cfg.image_size)?;
        let depth = self.forward(&input)?; // [image_size, image_size]
        depth.eval()?;
        let (dh, dw) = (depth.shape()[0] as usize, depth.shape()[1] as usize);
        let depth_vals: Vec<f32> = depth.as_slice::<f32>().to_vec();
        // Normalize at native resolution, then resize the control image back to input dims.
        let native = preprocess::depth_to_control_rgb8(&depth_vals, dh, dw);
        Ok(resize_control_rgb8(
            &native,
            dh,
            dw,
            height as usize,
            width as usize,
        ))
    }
}

/// Bilinear resize of an RGB8 HWC control image back to the host generation resolution. (The model's
/// native depth is 518²; the control image must match the requested `out_h·out_w`.)
fn resize_control_rgb8(
    rgb: &[u8],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<u8> {
    if in_h == out_h && in_w == out_w {
        return rgb.to_vec();
    }
    let mut out = vec![0u8; out_h * out_w * 3];
    let sx = in_w as f32 / out_w as f32;
    let sy = in_h as f32 / out_h as f32;
    for oy in 0..out_h {
        let fy = ((oy as f32 + 0.5) * sy - 0.5).max(0.0);
        let y0 = (fy.floor() as usize).min(in_h - 1);
        let y1 = (y0 + 1).min(in_h - 1);
        let wy = fy - y0 as f32;
        for ox in 0..out_w {
            let fx = ((ox as f32 + 0.5) * sx - 0.5).max(0.0);
            let x0 = (fx.floor() as usize).min(in_w - 1);
            let x1 = (x0 + 1).min(in_w - 1);
            let wx = fx - x0 as f32;
            for c in 0..3 {
                let p = |y: usize, x: usize| rgb[(y * in_w + x) * 3 + c] as f32;
                let top = p(y0, x0) * (1.0 - wx) + p(y0, x1) * wx;
                let bot = p(y1, x0) * (1.0 - wx) + p(y1, x1) * wx;
                out[(oy * out_w + ox) * 3 + c] =
                    (top * (1.0 - wy) + bot * wy).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_control_identity_when_same_dims() {
        let rgb = vec![10u8, 20, 30, 40, 50, 60]; // 1×2×3
        let out = resize_control_rgb8(&rgb, 1, 2, 1, 2);
        assert_eq!(out, rgb);
    }

    #[test]
    fn resize_control_changes_dims() {
        let rgb = vec![0u8; 4 * 4 * 3];
        let out = resize_control_rgb8(&rgb, 4, 4, 8, 8);
        assert_eq!(out.len(), 8 * 8 * 3);
    }
}
