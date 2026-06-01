//! `PatchMerger`: RMSNorm(`ln_q`) → group each `spatial_merge²` patches into one row → `mlp_0` →
//! exact GELU → `mlp_1`, mapping `embed → out_hidden`. Port of the fork's `qwen_patch_merger.py`.
//!
//! The fork reshapes per image (`[t·h·w, embed] → [-1, embed·merge²]`); because the windowed
//! hidden states are already grouped in `merge²`-sized, image-contiguous blocks, a single global
//! `[-1, embed·merge²]` reshape is identical and avoids threading `grid_thw` through.

use mlx_rs::fast::rms_norm;
use mlx_rs::nn::gelu;
use mlx_rs::Array;

use mlx_gen::nn::linear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::text_encoder::join;

const EPS: f32 = 1e-6;

pub struct PatchMerger {
    ln_q: Array,
    mlp0_w: Array,
    mlp0_b: Array,
    mlp1_w: Array,
    mlp1_b: Array,
    hidden_merged: i32,
}

impl PatchMerger {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        embed_dim: i32,
        spatial_merge_size: i32,
    ) -> Result<Self> {
        Ok(Self {
            ln_q: w.require(&join(prefix, "ln_q.weight"))?.clone(),
            mlp0_w: w.require(&join(prefix, "mlp_0.weight"))?.clone(),
            mlp0_b: w.require(&join(prefix, "mlp_0.bias"))?.clone(),
            mlp1_w: w.require(&join(prefix, "mlp_1.weight"))?.clone(),
            mlp1_b: w.require(&join(prefix, "mlp_1.bias"))?.clone(),
            hidden_merged: embed_dim * spatial_merge_size * spatial_merge_size,
        })
    }

    /// `x`: `[seq, embed]` (window-reordered) → `[seq/merge², out_hidden]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let x = rms_norm(x, &self.ln_q, EPS)?.reshape(&[-1, self.hidden_merged])?;
        let x = linear(&x, &self.mlp0_w, &self.mlp0_b)?;
        let x = gelu(&x)?;
        linear(&x, &self.mlp1_w, &self.mlp1_b)
    }
}
