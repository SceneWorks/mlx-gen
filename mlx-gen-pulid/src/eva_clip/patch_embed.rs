//! EVA `PatchEmbed`: `Conv2d(in→embed, kernel=stride=patch)` over NHWC pixels. Port of the
//! `eva_vit_model.py PatchEmbed`. The conv weight ships OIHW and is transposed to OHWI (MLX
//! channels-last) by the converter, so here it is a straight `conv2d`.

use mlx_rs::Array;

use mlx_gen::nn::conv2d;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::eva_clip::join;

pub struct PatchEmbed {
    proj_w: Array, // [embed, patch, patch, in] (OHWI)
    proj_b: Array, // [embed]
    patch: i32,
    embed_dim: i32,
}

impl PatchEmbed {
    pub fn from_weights(w: &Weights, prefix: &str, patch: i32, embed_dim: i32) -> Result<Self> {
        Ok(Self {
            proj_w: w.require(&join(prefix, "proj.weight"))?.clone(),
            proj_b: w.require(&join(prefix, "proj.bias"))?.clone(),
            patch,
            embed_dim,
        })
    }

    /// `pixel_values`: NHWC `[B, H, W, in]` (H=W=image_size) → `[B, grid², embed]` (row-major,
    /// matching torch `flatten(2).transpose(1,2)`).
    pub fn forward(&self, pixel_values: &Array) -> Result<Array> {
        let sh = pixel_values.shape();
        let b = sh[0];
        let y = conv2d(
            pixel_values,
            &self.proj_w,
            Some(&self.proj_b),
            self.patch,
            0,
        )?; // [B, g, g, embed]
        let g = y.shape()[1];
        Ok(y.reshape(&[b, g * g, self.embed_dim])?)
    }
}
