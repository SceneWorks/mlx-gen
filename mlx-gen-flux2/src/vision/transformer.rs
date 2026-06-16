//! Pixtral vision tower: per-image bias-less patch Conv2d → `ln_pre` RMSNorm → 24 pre-norm blocks
//! (block-diagonal attention + 2-D RoPE) → image features. Port of `PixtralVisionModel`.
//!
//! There is **no** post-layernorm: `vision_feature_layer = -1` selects the last block's raw hidden
//! state (the reference returns `last_hidden_state` directly), which the [`super::projector`]
//! consumes.

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::rope_2d::{cu_seqlens, rope_2d};
use super::{PatchConv, PixtralBlock, PixtralVisionConfig};
use crate::text_encoder::join;

pub struct PixtralVisionTower {
    patch_conv: PatchConv,
    ln_pre: Array,
    layers: Vec<PixtralBlock>,
    cfg: PixtralVisionConfig,
}

impl PixtralVisionTower {
    /// Load from the `vision_tower.*` subtree (`patch_conv.weight`, `ln_pre.weight`,
    /// `transformer.layers.{i}.…`).
    pub fn from_weights(w: &Weights, prefix: &str, cfg: PixtralVisionConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(PixtralBlock::from_weights(
                w,
                &join(prefix, &format!("transformer.layers.{i}")),
                &cfg,
            )?);
        }
        Ok(Self {
            patch_conv: PatchConv::from_weights(w, &join(prefix, "patch_conv"), cfg.patch_size)?,
            ln_pre: w.require(&join(prefix, "ln_pre.weight"))?.clone(),
            layers,
            cfg,
        })
    }

    /// `images`: one NHWC `[1, H_i, W_i, num_channels]` per reference image; `grids`: each image's
    /// `(gh, gw) = (H_i/patch, W_i/patch)`. Returns image features `[Σ gh·gw, hidden]`. Block-diagonal
    /// attention keeps each image's patches separate.
    pub fn forward(&self, images: &[&Array], grids: &[(i32, i32)]) -> Result<Array> {
        let mut parts = Vec::with_capacity(images.len());
        for image in images {
            parts.push(self.patch_conv.forward(image)?);
        }
        let refs: Vec<&Array> = parts.iter().collect();
        let mut x = if parts.len() == 1 {
            parts[0].clone()
        } else {
            concatenate_axis(&refs, 0)?
        };
        x = rms_norm(&x, &self.ln_pre, self.cfg.rms_norm_eps)?;

        let (cos, sin) = rope_2d(grids, self.cfg.head_dim, self.cfg.rope_theta);
        let cu = cu_seqlens(grids);
        for layer in &self.layers {
            x = layer.forward(&x, &cos, &sin, &cu)?;
        }
        Ok(x)
    }
}
