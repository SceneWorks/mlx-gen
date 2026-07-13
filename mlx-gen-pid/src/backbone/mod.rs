//! `PixDiT_T2I` backbone forward — the base text-to-image PixelDiT that `PidNet` (the LQ
//! super-resolution variant, ported separately) inherits. Dual-stream MMDiT patch blocks + per-pixel
//! PiT blocks, 2-D NTK image RoPE + 1-D text RoPE, sinusoidal timestep conditioning, unfold/fold
//! patchify. Faithful port of `PixDiT_T2I.forward` (the released SR students set `enable_ed=False`,
//! `repa_encoder_index` only affects a training side-output, so the inference forward is this clean
//! no-encoder-decoder path).
//!
//! Runs f32 activations (the parity target and the dense-16-bit-GEMM-safe path, matching the other
//! MMDiT stacks in this workspace). Projections are [`mlx_gen::adapters::AdaptableLinear`] so quant /
//! LoRA can hang off them later with no separate code path.

mod blocks;
mod layers;
mod rope;

use mlx_rs::ops::{add, split_sections};
use mlx_rs::Array;

use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::PidConfig;
use crate::memo::{memo, TableCache};
use blocks::{MMDiTBlockT2I, PiTBlock};
use layers::{
    fold_patches, unfold_patches, FinalLayer, PatchTokenEmbedder, PixelTokenEmbedder,
    TimestepConditioner,
};
use rope::{rope_1d_text, rope_2d_ntk};

// The pure host-side positional math is exposed so parity tests can gate it directly (tightly),
// independent of the cross-backend matmul floor in the full forward.
pub use layers::sincos_2d_pos;
pub use rope::{rope_1d_text as text_rope_table, rope_2d_ntk as image_rope_table};

const ROPE_THETA: f32 = 10000.0;
const ROPE_SCALE: f32 = 16.0;

/// A hook called before each patch block with `(block_idx, s_main)`, returning the (possibly gated)
/// `s_main`. `PidNet`'s sigma-aware LQ adapter implements this to inject the controlnet-style gate
/// between patch blocks (the reference's `_run_patch_blocks` loop); the base T2I forward passes none.
pub trait PatchInjector {
    fn inject(&self, block_idx: i32, s_main: &Array) -> Result<Array>;
}

/// The `PixDiT_T2I` backbone.
pub struct PixDiT {
    pixel_embedder: PixelTokenEmbedder,
    s_embedder: PatchTokenEmbedder,
    t_embedder: TimestepConditioner,
    y_embedder: PatchTokenEmbedder,
    y_pos_embedding: Array,
    patch_blocks: Vec<MMDiTBlockT2I>,
    pixel_blocks: Vec<PiTBlock>,
    final_layer: FinalLayer,
    cfg: PidConfig,
    /// Per-decode caches for the step-invariant RoPE tables (F-153). Keyed on the only inputs that vary
    /// within a decode — the patch grid `(hs, ws)` for the two 2-D image/pixel RoPE tables and the text
    /// length `ltxt` for the 1-D text RoPE. head_dim / θ / scale / ref-grid are per-decode config
    /// constants, so `(hs, ws)` / `ltxt` fully key each table. Image and pixel RoPE use different
    /// head_dims → separate caches (never collide on the shared `(hs, ws)` key).
    img_rope_cache: TableCache<(i32, i32), (Array, Array)>,
    pixel_rope_cache: TableCache<(i32, i32), (Array, Array)>,
    text_rope_cache: TableCache<i32, (Array, Array)>,
}

/// Slice the `[B, S, …]` axis-1 prefix `[:, :n]` (no-op when `S == n`). A zero-copy strided split at
/// the fixed `n` boundary, vs. an arange `take_axis` gather (F-152).
fn prefix_axis1(a: &Array, n: i32) -> Result<Array> {
    if a.shape()[1] == n {
        return Ok(a.clone());
    }
    Ok(split_sections(a, &[n], 1)?.swap_remove(0))
}

impl PixDiT {
    /// `prefix` is `""` for a bare-key fixture or `"net."` for the released checkpoint's nesting.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &PidConfig) -> Result<Self> {
        let patch_blocks = (0..cfg.patch_depth)
            .map(|i| {
                MMDiTBlockT2I::from_weights(
                    w,
                    &format!("{prefix}patch_blocks.{i}"),
                    cfg.hidden_size,
                    cfg.num_groups,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let pixel_blocks = (0..cfg.pixel_depth)
            .map(|i| {
                PiTBlock::from_weights(
                    w,
                    &format!("{prefix}pixel_blocks.{i}"),
                    cfg.pixel_hidden_size,
                    cfg.pixel_attn_hidden_size,
                    cfg.pixel_num_groups,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            pixel_embedder: PixelTokenEmbedder::from_weights(
                w,
                &format!("{prefix}pixel_embedder"),
                cfg.pixel_hidden_size,
            )?,
            s_embedder: PatchTokenEmbedder::from_weights(w, &format!("{prefix}s_embedder"))?,
            t_embedder: TimestepConditioner::from_weights(w, &format!("{prefix}t_embedder"))?,
            y_embedder: PatchTokenEmbedder::from_weights(w, &format!("{prefix}y_embedder"))?,
            y_pos_embedding: w.require(&format!("{prefix}y_pos_embedding"))?.clone(),
            patch_blocks,
            pixel_blocks,
            final_layer: FinalLayer::from_weights(w, &format!("{prefix}final_layer"))?,
            cfg: cfg.clone(),
            img_rope_cache: TableCache::default(),
            pixel_rope_cache: TableCache::default(),
            text_rope_cache: TableCache::default(),
        })
    }

    /// `x`: `[B, 3, H, W]`; `t`: `[B]`; `y`: `[B, Ltxt, txt_embed_dim]` (caption embeddings).
    /// Returns the predicted pixel tensor `[B, 3, H, W]`.
    pub fn forward(&self, x: &Array, t: &Array, y: &Array) -> Result<Array> {
        self.forward_inner(x, t, y, None)
    }

    /// Like [`Self::forward`] but with a per-patch-block injection hook — `PidNet` passes its
    /// sigma-aware LQ adapter here to gate `s_main` between blocks.
    pub fn forward_with(
        &self,
        x: &Array,
        t: &Array,
        y: &Array,
        injector: &dyn PatchInjector,
    ) -> Result<Array> {
        self.forward_inner(x, t, y, Some(injector))
    }

    fn forward_inner(
        &self,
        x: &Array,
        t: &Array,
        y: &Array,
        injector: Option<&dyn PatchInjector>,
    ) -> Result<Array> {
        let cfg = &self.cfg;
        let patch = cfg.patch_size;
        let sh = x.shape();
        let (b, h, w) = (sh[0], sh[2], sh[3]);
        let (hs, ws) = (h / patch, w / patch);
        let l = hs * ws;

        let x_patches = unfold_patches(x, patch)?;
        let t_emb = self
            .t_embedder
            .forward(t)?
            .reshape(&[b, 1, cfg.hidden_size])?;

        let ltxt = y.shape()[1].min(cfg.txt_max_length);
        let y = prefix_axis1(y, ltxt)?;
        let y_emb = self.y_embedder.forward(&y)?;
        let y_pos = prefix_axis1(&self.y_pos_embedding, ltxt)?.as_dtype(y_emb.dtype())?;
        let mut y_emb = add(&y_emb, &y_pos)?;

        let condition = silu(&t_emb)?;
        // Step-invariant RoPE tables — memoized per `(hs, ws)` / `ltxt` so the 4 sampler steps (and
        // same-sized tiles) reuse one host build + H2D upload instead of rebuilding it each forward
        // (F-153). Byte-identical: the cache returns the exact `rope_*` output for that key.
        let (cos_img, sin_img) = memo(&self.img_rope_cache, (hs, ws), || {
            rope_2d_ntk(
                cfg.head_dim(),
                hs,
                ws,
                cfg.rope_ref_grid_h(),
                cfg.rope_ref_grid_w(),
                ROPE_THETA,
                ROPE_SCALE,
            )
        });
        let (cos_txt, sin_txt) = memo(&self.text_rope_cache, ltxt, || {
            rope_1d_text(cfg.head_dim(), ltxt, cfg.text_rope_theta)
        });

        let mut s_main = self.s_embedder.forward(&x_patches)?;
        for (i, blk) in self.patch_blocks.iter().enumerate() {
            if let Some(inj) = injector {
                s_main = inj.inject(i as i32, &s_main)?;
            }
            let (sx, sy) = blk.forward(
                &s_main, &y_emb, &condition, &cos_img, &sin_img, &cos_txt, &sin_txt,
            )?;
            s_main = sx;
            y_emb = sy;
        }
        let s = silu(&add(&t_emb, &s_main)?)?;
        let s_cond = s.reshape(&[b * l, cfg.hidden_size])?;

        let mut x_pixels = self.pixel_embedder.forward(x, h, w, patch)?;
        let (cos_pix, sin_pix) = memo(&self.pixel_rope_cache, (hs, ws), || {
            rope_2d_ntk(
                cfg.pixel_head_dim(),
                hs,
                ws,
                cfg.rope_ref_grid_h(),
                cfg.rope_ref_grid_w(),
                ROPE_THETA,
                ROPE_SCALE,
            )
        });
        for blk in &self.pixel_blocks {
            x_pixels = blk.forward(&x_pixels, &s_cond, &cos_pix, &sin_pix, b, l)?;
        }
        let x_pixels = self.final_layer.forward(&x_pixels)?;

        // [B*L, P2, C_out] -> [B, L, P2, C_out] -> [B, C_out, P2, L] -> fold -> [B, C_out, H, W]
        let c_out = cfg.in_channels;
        let p2 = patch * patch;
        let xp = x_pixels
            .reshape(&[b, l, p2, c_out])?
            .transpose_axes(&[0, 3, 2, 1])?;
        fold_patches(&xp, c_out, hs, ws, patch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{abs, max, subtract};

    /// F-152: `prefix_axis1`'s strided split is element-identical to the arange `take_axis` gather it
    /// replaced (for a true prefix), and a no-op clone when `n == S`.
    #[test]
    fn prefix_axis1_matches_take_axis_gather() {
        let a = Array::from_iter(0..(2 * 6 * 3), &[2, 6, 3])
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap();
        let n = 4;
        let got = prefix_axis1(&a, n).unwrap();
        let idx = Array::from_slice(&(0..n).collect::<Vec<i32>>(), &[n]);
        let want = a.take_axis(&idx, 1).unwrap();
        assert_eq!(got.shape(), want.shape());
        let d = max(abs(subtract(&got, &want).unwrap()).unwrap(), None)
            .unwrap()
            .item::<f32>();
        assert_eq!(d, 0.0, "prefix split == gather");
        // Full-length prefix is the identity.
        let whole = prefix_axis1(&a, 6).unwrap();
        assert_eq!(whole.shape(), a.shape());
    }
}
