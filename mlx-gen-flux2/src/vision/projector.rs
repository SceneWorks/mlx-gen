//! Mistral3 multimodal projector: `norm` (RMSNorm) → `patch_merger` (2×2 spatial merge →
//! `merging_layer`) → `linear_1` → **gelu** → `linear_2`. Maps Pixtral image features
//! `[Σ gh·gw, vision_hidden]` into the Mistral token-embedding space `[Σ (gh/s)·(gw/s), text_hidden]`
//! (s = `spatial_merge_size`). All Linears are bias-less. Port of `Mistral3MultiModalProjector` +
//! `Mistral3PatchMerger`.
//!
//! The projected tokens are then scattered into the Mistral input embeddings where
//! `input_ids == image_token_index` (sc-5919, the edit generate path) — this module stops at the
//! projected tokens.

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::{concatenate_axis, matmul};
use mlx_rs::Array;

use mlx_gen::nn::gelu_exact;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::text_encoder::join;

pub struct Mistral3Projector {
    /// RMSNorm over the vision hidden, applied **before** merging. eps = the text-config eps.
    norm: Array,
    /// `merging_layer` `[vision_hidden, vision_hidden·s²]` — the 2×2-merged patch → vision_hidden.
    merging_w: Array,
    /// `linear_1` `[text_hidden, vision_hidden]`.
    linear_1_w: Array,
    /// `linear_2` `[text_hidden, text_hidden]`.
    linear_2_w: Array,
    eps: f32,
    spatial_merge: i32,
}

impl Mistral3Projector {
    /// Load from the `multi_modal_projector.*` subtree. `eps` is the **text-config** `rms_norm_eps`
    /// (the projector's norm uses the language eps, not the vision one); `spatial_merge` is the
    /// config's `spatial_merge_size` (dev: 2).
    pub fn from_weights(w: &Weights, prefix: &str, spatial_merge: i32, eps: f32) -> Result<Self> {
        Ok(Self {
            norm: w.require(&join(prefix, "norm.weight"))?.clone(),
            merging_w: w
                .require(&join(prefix, "patch_merger.merging_layer.weight"))?
                .clone(),
            linear_1_w: w.require(&join(prefix, "linear_1.weight"))?.clone(),
            linear_2_w: w.require(&join(prefix, "linear_2.weight"))?.clone(),
            eps,
            spatial_merge,
        })
    }

    /// `image_features`: `[Σ gh·gw, vision_hidden]` from the tower; `grids`: each image's `(gh, gw)`
    /// (gh, gw must be multiples of `spatial_merge`). Returns `[Σ (gh/s)·(gw/s), text_hidden]`.
    pub fn forward(&self, image_features: &Array, grids: &[(i32, i32)]) -> Result<Array> {
        // norm first (over the vision hidden), then per-image 2×2 patch merge.
        let normed = rms_norm(image_features, &self.norm, self.eps)?;
        let merged = self.patch_merge(&normed, grids)?;
        // merging_layer → linear_1 → gelu → linear_2 (all bias-less).
        let x = matmul(&merged, self.merging_w.t())?;
        let x = matmul(&x, self.linear_1_w.t())?;
        let x = gelu_exact(&x)?;
        Ok(matmul(&x, self.linear_2_w.t())?)
    }

    /// Per-image `nn.functional.unfold(kernel=stride=s)` over the `[gh, gw, d]` patch grid: each
    /// `s×s` block → a single `d·s²`-wide token, channel-major then `(row, col)` within the block
    /// (the reference's unfold index `c·s² + ki·s + kj`). Images concatenated in order.
    fn patch_merge(&self, x: &Array, grids: &[(i32, i32)]) -> Result<Array> {
        let s = self.spatial_merge;
        let d = x.shape()[1];
        let mut outs = Vec::with_capacity(grids.len());
        let mut cur = 0;
        for &(gh, gw) in grids {
            if gh % s != 0 || gw % s != 0 {
                return Err(Error::Msg(format!(
                    "flux2 projector: patch grid {gh}x{gw} not divisible by spatial_merge {s}"
                )));
            }
            let n = gh * gw;
            let idx = Array::from_slice(&(cur..cur + n).collect::<Vec<i32>>(), &[n]);
            let img = x.take_axis(&idx, 0)?; // [gh·gw, d], row-major
                                             // [gh, gw, d] → split gh→(gh/s, ki=s), gw→(gw/s, kj=s) → gather (d, ki, kj) last →
                                             // [(gh/s)·(gw/s), d·s²] with the flat index `c·s² + ki·s + kj`.
            let block = img
                .reshape(&[gh / s, s, gw / s, s, d])?
                .transpose_axes(&[0, 2, 4, 1, 3])?
                .reshape(&[(gh / s) * (gw / s), d * s * s])?;
            outs.push(block);
            cur += n;
        }
        let refs: Vec<&Array> = outs.iter().collect();
        Ok(concatenate_axis(&refs, 0)?)
    }
}
