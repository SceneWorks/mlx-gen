//! EVA02-CLIP-L-14-336 **visual** tower (sc-3070) — the face-identity ViT PuLID-FLUX feeds the
//! background-removed aligned face crop through. Port of `eva_clip/eva_vit_model.py`
//! `EVAVisionTransformer` (the `.visual` submodule only; no text tower).
//!
//! Pipeline: `Conv2d` patch-embed → prepend CLS + add learned abs `pos_embed` → 24 sub-LN blocks
//! (interleaved 2-D RoPE on patch tokens, full SDPA, SwiGLU) → final LayerNorm → take CLS token →
//! `head` projection. Returns `id_cond_vit` (768-d) plus the 5 hidden states captured at the
//! **input** of blocks {4,8,12,16,20} (1024-d each) that the IDFormer (sc-3071) consumes.
//!
//! NOTE this checkpoint has `use_mean_pooling=False` (`visual.norm.*` present, no `visual.fc_norm.*`)
//! ⇒ the pooled feature is `norm(x)[:, 0]` (CLS), not `fc_norm(x.mean(1))`.

mod attention;
mod block;
mod mlp;
mod patch_embed;
pub mod rope;
pub mod transform;

use mlx_rs::fast::layer_norm;
use mlx_rs::ops::{add, broadcast_to, concatenate_axis};
use mlx_rs::Array;

use mlx_gen::nn::linear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use block::Block;
use patch_embed::PatchEmbed;
use rope::VisionRope;

/// EVA LayerNorm epsilon (`model.py:123` `partial(LayerNorm, eps=1e-6)`).
pub(crate) const EPS: f32 = 1e-6;

/// Join a dotted weight-key prefix with a leaf (`"" + leaf` ⇒ `leaf`).
pub(crate) fn join(prefix: &str, leaf: &str) -> String {
    if prefix.is_empty() {
        leaf.to_string()
    } else {
        format!("{prefix}.{leaf}")
    }
}

/// EVA02-CLIP-L-14-336 visual-tower config.
#[derive(Clone, Debug)]
pub struct EvaConfig {
    pub image_size: i32,
    pub patch: i32,
    pub embed_dim: i32, // width = 1024
    pub depth: i32,
    pub num_heads: i32,
    pub proj_dim: i32, // head out (num_classes) = 768
    pub pt_seq_len: i32,
    pub rope_theta: f64,
    pub hidden_capture: Vec<i32>,
}

impl Default for EvaConfig {
    fn default() -> Self {
        Self {
            image_size: 336,
            patch: 14,
            embed_dim: 1024,
            depth: 24,
            num_heads: 16,
            proj_dim: 768,
            pt_seq_len: 16,
            rope_theta: 10000.0,
            hidden_capture: vec![4, 8, 12, 16, 20],
        }
    }
}

impl EvaConfig {
    pub fn head_dim(&self) -> i32 {
        self.embed_dim / self.num_heads
    }
    pub fn grid(&self) -> i32 {
        self.image_size / self.patch
    }
}

/// EVA visual tower output: the projected id feature + the captured intermediate hidden states.
pub struct EvaOutput {
    /// `[B, proj_dim]` (768) — the `head`-projected pooled feature (pre-L2-norm).
    pub id_cond_vit: Array,
    /// 5 × `[B, grid²+1, embed_dim]` (577×1024) — inputs of blocks {4,8,12,16,20}.
    pub hidden: Vec<Array>,
}

pub struct EvaVisionTransformer {
    patch_embed: PatchEmbed,
    cls_token: Array,
    pos_embed: Array,
    blocks: Vec<Block>,
    norm_w: Array,
    norm_b: Array,
    head_w: Array,
    head_b: Array,
    rope: VisionRope,
    cfg: EvaConfig,
}

impl EvaVisionTransformer {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: EvaConfig) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        let (nh, hd) = (cfg.num_heads, cfg.head_dim());
        let blocks = (0..cfg.depth)
            .map(|i| Block::from_weights(w, &p(&format!("blocks.{i}")), nh, hd))
            .collect::<Result<Vec<_>>>()?;
        let rope = VisionRope::build(hd, cfg.grid(), cfg.pt_seq_len, cfg.rope_theta)?;
        Ok(Self {
            patch_embed: PatchEmbed::from_weights(w, &p("patch_embed"), cfg.patch, cfg.embed_dim)?,
            cls_token: w.require(&p("cls_token"))?.clone(),
            pos_embed: w.require(&p("pos_embed"))?.clone(),
            blocks,
            norm_w: w.require(&p("norm.weight"))?.clone(),
            norm_b: w.require(&p("norm.bias"))?.clone(),
            head_w: w.require(&p("head.weight"))?.clone(),
            head_b: w.require(&p("head.bias"))?.clone(),
            rope,
            cfg,
        })
    }

    /// Recomputed RoPE table (exposed for the weight-free construction gate vs `rope.freqs_*`).
    pub fn rope(&self) -> &VisionRope {
        &self.rope
    }

    /// The tower's loaded [`EvaConfig`]. Consumers (e.g. PuLID's uncond-embedding builder) derive the
    /// EVA token geometry (`grid²+1` sequence length, `embed_dim`, hidden-capture count) from this
    /// rather than re-hardcoding the default-tower constants (F-082).
    pub fn config(&self) -> &EvaConfig {
        &self.cfg
    }

    /// `pixel_values`: NHWC `[B, image_size, image_size, 3]`, EVA-normalized.
    pub fn forward(&self, pixel_values: &Array) -> Result<EvaOutput> {
        let mut x = self.patch_embed.forward(pixel_values)?; // [B, grid², embed]
        let b = x.shape()[0];
        let cls = broadcast_to(&self.cls_token, &[b, 1, self.cfg.embed_dim])?;
        x = concatenate_axis(&[&cls, &x], 1)?; // [B, grid²+1, embed]
        x = add(&x, &self.pos_embed)?;

        let mut hidden = Vec::with_capacity(self.cfg.hidden_capture.len());
        for (idx, blk) in self.blocks.iter().enumerate() {
            if self.cfg.hidden_capture.contains(&(idx as i32)) {
                hidden.push(x.clone());
            }
            x = blk.forward(&x, &self.rope)?;
        }

        x = layer_norm(&x, Some(&self.norm_w), Some(&self.norm_b), EPS)?;
        // CLS token → head projection.
        let cls_tok = x
            .take_axis(Array::from_slice(&[0i32], &[1]), 1)?
            .reshape(&[b, self.cfg.embed_dim])?;
        let id_cond_vit = linear(&cls_tok, &self.head_w, &self.head_b)?;
        Ok(EvaOutput {
            id_cond_vit,
            hidden,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// F-082: the EVA token geometry PuLID's uncond builder derives from the config must reproduce the
    /// previously-hardcoded default-tower constants (577×1024 over 5 captures) AND track a non-default
    /// `image_size`, so the constructor's "any EvaConfig" contract is honored, not silently assumed.
    #[test]
    fn derived_token_geometry_matches_and_tracks_config() {
        let cfg = EvaConfig::default();
        assert_eq!(cfg.grid() * cfg.grid() + 1, 577, "grid²+1 = 577 for 336/14");
        assert_eq!(cfg.embed_dim, 1024);
        assert_eq!(cfg.hidden_capture.len(), 5);

        // A larger square input (one more patch per side) shifts the derived sequence length.
        let bigger = EvaConfig {
            image_size: cfg.image_size + cfg.patch,
            ..cfg.clone()
        };
        assert_eq!(bigger.grid(), cfg.grid() + 1);
        assert_eq!(
            bigger.grid() * bigger.grid() + 1,
            (cfg.grid() + 1) * (cfg.grid() + 1) + 1
        );
    }
}
