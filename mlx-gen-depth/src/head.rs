//! DPT depth-estimation head — port of `DepthAnythingDepthEstimationHead` for the `head.*` weights.
//!
//! `conv1` (3×3 pad-1, fusion_hidden_size→fusion_hidden_size/2) → bilinear upsample (align_corners
//! true) to the full `patch_grid · patch_size` resolution → `conv2` (3×3 pad-1 → head_hidden_size)
//! → ReLU → `conv3` (1×1 → 1) → ReLU (DA-V2 is a *relative*-depth model: the final activation is
//! ReLU, no sigmoid / max-depth scaling). Output `[B, H, W]` single-channel depth.

use mlx_rs::Array;

use mlx_gen::nn::conv2d;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::DepthAnythingConfig;
use crate::util::{bilinear_resize, conv_w_ohwi, join};

pub struct DepthHead {
    conv1_w: Array,
    conv1_b: Array,
    conv2_w: Array,
    conv2_b: Array,
    conv3_w: Array,
    conv3_b: Array,
    patch_size: i32,
}

impl DepthHead {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &DepthAnythingConfig) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        Ok(Self {
            conv1_w: conv_w_ohwi(w.require(&p("conv1.weight"))?)?,
            conv1_b: w.require(&p("conv1.bias"))?.clone(),
            conv2_w: conv_w_ohwi(w.require(&p("conv2.weight"))?)?,
            conv2_b: w.require(&p("conv2.bias"))?.clone(),
            conv3_w: conv_w_ohwi(w.require(&p("conv3.weight"))?)?,
            conv3_b: w.require(&p("conv3.bias"))?.clone(),
            patch_size: cfg.patch_size,
        })
    }

    /// `fused`: the neck's fused NHWC map `[B, h, w, fusion_hidden]`. `patch_grid` is the backbone
    /// token-grid side (37 at the default size) — the head upsamples to `patch_grid · patch_size`
    /// (the input resolution). Returns `[B, H, W]`.
    pub fn forward(&self, fused: &Array, patch_grid: i32) -> Result<Array> {
        let x = conv2d(fused, &self.conv1_w, Some(&self.conv1_b), 1, 1)?;
        let full = patch_grid * self.patch_size;
        let x = bilinear_resize(&x, full, full, true)?;
        let x = conv2d(&x, &self.conv2_w, Some(&self.conv2_b), 1, 1)?;
        let x = relu(&x)?;
        let x = conv2d(&x, &self.conv3_w, Some(&self.conv3_b), 1, 0)?;
        let x = relu(&x)?; // relative-depth: ReLU output, channel dim = 1.
        let sh = x.shape();
        // [B, H, W, 1] → [B, H, W].
        Ok(x.reshape(&[sh[0], sh[1], sh[2]])?)
    }
}

fn relu(x: &Array) -> Result<Array> {
    Ok(mlx_rs::ops::maximum(x, Array::from_f32(0.0))?)
}
