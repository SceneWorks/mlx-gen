//! `VisionPatchEmbed`: a bias-less `Conv3d(in→embed, kernel=stride=[temporal, patch, patch])` over
//! reshaped `pixel_values`. Port of the fork's `qwen_vision_patch_embed.py`.

use mlx_rs::Array;

use mlx_gen::nn::conv3d;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::text_encoder::join;

pub struct VisionPatchEmbed {
    /// MLX `Conv3d` weight `[embed, kD=temporal, kH=patch, kW=patch, in]` (already channels-last).
    proj_w: Array,
    in_channels: i32,
    temporal: i32,
    patch: i32,
    embed_dim: i32,
}

impl VisionPatchEmbed {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        in_channels: i32,
        temporal: i32,
        patch: i32,
        embed_dim: i32,
    ) -> Result<Self> {
        Ok(Self {
            proj_w: w.require(&join(prefix, "proj.weight"))?.clone(),
            in_channels,
            temporal,
            patch,
            embed_dim,
        })
    }

    /// `pixel_values`: `[num_patches, in·temporal·patch·patch]` (1176) → `[num_patches, embed]`.
    pub fn forward(&self, pixel_values: &Array) -> Result<Array> {
        let n = pixel_values.shape()[0];
        // [n, in, temporal, patch, patch] (channels-first) → NDHWC for the conv.
        let x =
            pixel_values.reshape(&[n, self.in_channels, self.temporal, self.patch, self.patch])?;
        let x = x.transpose_axes(&[0, 2, 3, 4, 1])?;
        // kernel == stride == full window → one output voxel per patch → [n,1,1,1,embed].
        let y = conv3d(
            &x,
            &self.proj_w,
            None,
            (self.temporal, self.patch, self.patch),
            (0, 0, 0),
        )?;
        Ok(y.reshape(&[n, self.embed_dim])?)
    }
}
