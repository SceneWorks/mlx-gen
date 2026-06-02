//! `VisionTransformer`: the full Qwen2.5-VL vision tower. Port of the fork's
//! `qwen_vision_transformer.py` `__call__`:
//!
//! 1. `patch_embed(pixel_values)` → `[seq, embed]`.
//! 2. Build the 2-D RoPE table + window/full `cu_seqlens` ([`super::grid`]).
//! 3. **Window-reorder** hidden states + RoPE (group rows by `window_index`).
//! 4. 32 blocks: windowed attention, except `fullatt_block_indexes` which attend per-image.
//! 5. `merger` → `[seq/merge², out_hidden]`.
//! 6. **Reverse** the window reorder (inverse permutation) → original group order.

use mlx_rs::ops::{concatenate_axis, cos, sin};
use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::grid::{cu_seqlens, rot_pos_emb, window_index, Grid, VisionGridConfig};
use super::{PatchMerger, VisionBlock, VisionPatchEmbed};
use crate::text_encoder::join;

/// Vision-transformer config. `mlp_hidden = round(embed_dim · mlp_ratio)` and
/// `out_hidden_size` (the merger output) are pre-resolved here.
#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub patch_size: i32,
    pub temporal_patch_size: i32,
    pub in_channels: i32,
    pub embed_dim: i32,
    pub depth: i32,
    pub num_heads: i32,
    pub mlp_hidden: i32,
    pub out_hidden_size: i32,
    pub spatial_merge_size: i32,
    pub window_size: i32,
    pub fullatt_block_indexes: Vec<i32>,
    pub rope_theta: f32,
}

impl VisionConfig {
    /// The Qwen-Image-Edit-2509 `vision_config` (depth 32, embed 1280, 16 heads × 80,
    /// mlp_ratio 2.671875 → 3420, out 3584, window 112, full-attn at `[7,15,23,31]`).
    pub fn qwen_image_edit() -> Self {
        Self {
            patch_size: 14,
            temporal_patch_size: 2,
            in_channels: 3,
            embed_dim: 1280,
            depth: 32,
            num_heads: 16,
            mlp_hidden: 3420,
            out_hidden_size: 3584,
            spatial_merge_size: 2,
            window_size: 112,
            fullatt_block_indexes: vec![7, 15, 23, 31],
            rope_theta: 10000.0,
        }
    }

    pub fn head_dim(&self) -> i32 {
        self.embed_dim / self.num_heads
    }

    fn grid_config(&self) -> VisionGridConfig {
        VisionGridConfig {
            patch_size: self.patch_size,
            spatial_merge_size: self.spatial_merge_size,
            window_size: self.window_size,
            rope_dim: self.head_dim() / 2,
            rope_theta: self.rope_theta,
        }
    }
}

pub struct VisionTransformer {
    patch_embed: VisionPatchEmbed,
    blocks: Vec<VisionBlock>,
    merger: PatchMerger,
    cfg: VisionConfig,
}

impl VisionTransformer {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &VisionConfig) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let patch_embed = VisionPatchEmbed::from_weights(
            w,
            &join(prefix, "patch_embed"),
            cfg.in_channels,
            cfg.temporal_patch_size,
            cfg.patch_size,
            cfg.embed_dim,
        )?;
        let blocks = (0..cfg.depth)
            .map(|i| {
                VisionBlock::from_weights(
                    w,
                    &join(prefix, &format!("blocks.{i}")),
                    cfg.num_heads,
                    head_dim,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let merger = PatchMerger::from_weights(
            w,
            &join(prefix, "merger"),
            cfg.embed_dim,
            cfg.spatial_merge_size,
        )?;
        Ok(Self {
            patch_embed,
            blocks,
            merger,
            cfg: cfg.clone(),
        })
    }

    /// `pixel_values`: `[seq_patches, in·temporal·patch·patch]`; `grids`: one `(t, grid_h, grid_w)`
    /// per image (patch units). Returns vision embeds `[seq_patches/merge², out_hidden]`.
    pub fn forward(&self, pixel_values: &Array, grids: &[Grid]) -> Result<Array> {
        self.forward_impl(pixel_values, grids, None)
    }

    /// Debug: same as [`forward`](Self::forward) but returns per-stage intermediates for parity
    /// bisection against the fork (`conv_input`, `patch_embed`, `reordered`, `block0`,
    /// `blocks_all`, `merger`, `final`). Not used in production — only the `*_real_weights` tests.
    /// Delegates to the same [`forward_impl`](Self::forward_impl) as [`forward`](Self::forward), so
    /// it can no longer silently drift from the real forward path.
    pub fn forward_capture(
        &self,
        pixel_values: &Array,
        grids: &[Grid],
    ) -> Result<Vec<(&'static str, Array)>> {
        let mut caps = Vec::new();
        self.forward_impl(pixel_values, grids, Some(&mut caps))?;
        Ok(caps)
    }

    /// Shared implementation of [`forward`](Self::forward) and
    /// [`forward_capture`](Self::forward_capture). When `capture` is `Some`, per-stage
    /// intermediates are recorded for bisection; the returned tensor is **identical** either way —
    /// capture is a pure side-channel of `.clone()`s (and the debug-only `conv_input` recompute),
    /// so disabling it leaves the production op sequence untouched.
    fn forward_impl(
        &self,
        pixel_values: &Array,
        grids: &[Grid],
        mut capture: Option<&mut Vec<(&'static str, Array)>>,
    ) -> Result<Array> {
        let cfg = &self.cfg;
        let grid_cfg = cfg.grid_config();
        let embed = cfg.embed_dim;
        let rope_dim = cfg.head_dim() / 2;
        let merge_unit = cfg.spatial_merge_size * cfg.spatial_merge_size;

        // Debug-only: the patch-embed conv input (post reshape+transpose to NDHWC), to isolate
        // reshape vs conv. Computed only when capturing.
        if let Some(caps) = capture.as_mut() {
            let n = pixel_values.shape()[0];
            let conv_in = pixel_values
                .reshape(&[
                    n,
                    cfg.in_channels,
                    cfg.temporal_patch_size,
                    cfg.patch_size,
                    cfg.patch_size,
                ])?
                .transpose_axes(&[0, 2, 3, 4, 1])?;
            caps.push(("conv_input", conv_in));
        }

        let hidden = self.patch_embed.forward(pixel_values)?; // [seq, embed]
        if let Some(caps) = capture.as_mut() {
            caps.push(("patch_embed", hidden.clone()));
        }
        let seq = hidden.shape()[0];
        let num_groups = seq / merge_unit;

        let rope = rot_pos_emb(grids, &grid_cfg)?; // [seq, rope_dim]
        let (wi, cu_window) = window_index(grids, &grid_cfg);
        let cu_full = cu_seqlens(grids);
        let wi_arr = Array::from_slice(&wi, &[num_groups]);

        // Window-reorder hidden + rope at the merge-group level.
        let hidden = hidden
            .reshape(&[num_groups, merge_unit, embed])?
            .take_axis(&wi_arr, 0)?
            .reshape(&[seq, embed])?;
        if let Some(caps) = capture.as_mut() {
            caps.push(("reordered", hidden.clone()));
        }
        let rope = rope
            .reshape(&[num_groups, merge_unit, rope_dim])?
            .take_axis(&wi_arr, 0)?
            .reshape(&[seq, rope_dim])?;

        // position_embeddings = (cos(emb), sin(emb)), emb = [rope ‖ rope] → [seq, head_dim].
        let emb = concatenate_axis(&[&rope, &rope], 1)?;
        let cos_emb = cos(&emb)?;
        let sin_emb = sin(&emb)?;

        let mut h = hidden;
        for (i, block) in self.blocks.iter().enumerate() {
            let cu = if cfg.fullatt_block_indexes.contains(&(i as i32)) {
                &cu_full
            } else {
                &cu_window
            };
            h = block.forward(&h, &cos_emb, &sin_emb, cu)?;
            if i == 0 {
                if let Some(caps) = capture.as_mut() {
                    caps.push(("block0", h.clone()));
                }
            }
        }
        if let Some(caps) = capture.as_mut() {
            caps.push(("blocks_all", h.clone()));
        }

        let h = self.merger.forward(&h)?; // [num_groups, out_hidden]
        if let Some(caps) = capture.as_mut() {
            caps.push(("merger", h.clone()));
        }

        // Reverse the window reorder: inverse permutation of window_index.
        let mut reverse = vec![0i32; num_groups as usize];
        for (k, &g) in wi.iter().enumerate() {
            reverse[g as usize] = k as i32;
        }
        let reverse_arr = Array::from_slice(&reverse, &[num_groups]);
        let out = h.take_axis(&reverse_arr, 0)?;
        if let Some(caps) = capture.as_mut() {
            caps.push(("final", out.clone()));
        }
        Ok(out)
    }
}
