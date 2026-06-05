//! Vision-transformer grid math (Qwen2.5-VL): window partitioning + 2-D vision RoPE position ids.
//!
//! These are **pure functions** of `grid_thw` + config — no model weights — so they are the
//! error-prone *and* most easily verifiable core of the windowed attention. They are unit-tested
//! byte-exact against the fork golden (`tools/dump_qwen_vision_golden.py`, gate 1) before any of
//! the weight-bearing modules are ported. Everything downstream (block-diagonal SDPA, the window
//! reorder/reverse, the merger grouping) depends on these indices being correct.
//!
//! Ports `VisionTransformer.{get_window_index, rot_pos_emb}` and the inline `cu_seqlens` from the
//! fork's `qwen_vision_transformer.py`.

use mlx_rs::Array;

use mlx_gen::Result;

/// `(t, grid_h, grid_w)` for one image, in **patch** units (`image_px / patch_size`).
pub type Grid = [i32; 3];

/// Vision config knobs that drive the grid math (the Qwen-Image-Edit-2511 `vision_config`).
#[derive(Clone, Copy, Debug)]
pub struct VisionGridConfig {
    pub patch_size: i32,
    pub spatial_merge_size: i32,
    pub window_size: i32,
    /// Vision RoPE dim = `head_dim / 2` = 40.
    pub rope_dim: i32,
    pub rope_theta: f32,
}

impl Default for VisionGridConfig {
    fn default() -> Self {
        Self {
            patch_size: 14,
            spatial_merge_size: 2,
            window_size: 112,
            rope_dim: 40,
            rope_theta: 10000.0,
        }
    }
}

/// Port of `VisionTransformer.get_window_index` **plus** the consecutive-dedup the fork applies in
/// `__call__`. Returns `(window_index, cu_window_seqlens)`:
///
/// - `window_index`: a permutation over the `sum_t(t·llm_h·llm_w)` merge-groups that gathers each
///   `merger_window²` block contiguously (padding entries, marked `-100` in the fork, are dropped).
///   Used to reorder hidden states + RoPE before the blocks and to reverse it after the merger.
/// - `cu_window_seqlens`: cumulative **patch** counts at window boundaries (deduped), for the
///   block-diagonal SDPA of the windowed blocks.
///
/// `merger_window = window_size / patch_size / spatial_merge_size` (= 4 for the default config).
pub fn window_index(grids: &[Grid], cfg: &VisionGridConfig) -> (Vec<i32>, Vec<i32>) {
    let merge = cfg.spatial_merge_size;
    let merge_unit = merge * merge; // patches per merge-group (4)
    let vmw = cfg.window_size / cfg.patch_size / merge; // merger-space window edge (4)

    let mut window_index: Vec<i32> = Vec::new();
    let mut cu_window: Vec<i32> = vec![0];
    let mut window_index_id = 0i32;

    for &[t, gh, gw] in grids {
        let llm_h = gh / merge;
        let llm_w = gw / merge;
        // The fork pads to a multiple of `vmw`; note `pad == vmw` when already a multiple (an extra,
        // entirely-padded window row/col whose zero-length entries are dropped + deduped below).
        let pad_h = vmw - llm_h % vmw;
        let pad_w = vmw - llm_w % vmw;
        let nwh = (llm_h + pad_h) / vmw;
        let nww = (llm_w + pad_w) / vmw;

        // Iterate windows in (t, window-row, window-col) order, then (row, col) within the window —
        // matching the fork's reshape to (t, nwh, vmw, nww, vmw) + transpose (0,1,3,2,4). The merger
        // index of an unpadded cell (i, j) in plane `ti` is `ti·llm_h·llm_w + i·llm_w + j`.
        for ti in 0..t {
            let plane = ti * llm_h * llm_w;
            for wh in 0..nwh {
                for ww in 0..nww {
                    let mut seqlen = 0i32;
                    for r in 0..vmw {
                        for c in 0..vmw {
                            let i = wh * vmw + r;
                            let j = ww * vmw + c;
                            if i < llm_h && j < llm_w {
                                window_index.push(window_index_id + plane + i * llm_w + j);
                                seqlen += 1;
                            }
                        }
                    }
                    // cumsum(seqlens)·merge_unit + running base: each push uses the prior cumulative.
                    let last = *cu_window.last().unwrap();
                    cu_window.push(last + seqlen * merge_unit);
                }
            }
        }
        window_index_id += t * llm_h * llm_w;
    }

    // Dedup consecutive-equal (drops the all-padding windows' zero-length contributions).
    let mut cu_dedup = vec![cu_window[0]];
    for &v in &cu_window[1..] {
        if v != *cu_dedup.last().unwrap() {
            cu_dedup.push(v);
        }
    }
    (window_index, cu_dedup)
}

/// Full-attention cumulative seqlens: `[0, cumulative t·h·w per image]` (patch units). The
/// full-attention blocks (`fullatt_block_indexes`) attend within each image; the windowed blocks
/// use [`window_index`]'s `cu_window_seqlens` instead.
pub fn cu_seqlens(grids: &[Grid]) -> Vec<i32> {
    let mut out = vec![0];
    let mut offset = 0;
    for &[t, h, w] in grids {
        offset += t * h * w;
        out.push(offset);
    }
    out
}

/// Port of `VisionTransformer.rot_pos_emb`: the 2-D vision RoPE position table
/// `[seq_patches, rope_dim]` (each row `[h_freqs(rope_dim/2) ‖ w_freqs(rope_dim/2)]`) in the
/// spatial-merge layout, **before** the window reorder. `seq_patches = sum_t(t·grid_h·grid_w)`.
///
/// `inv_freq[k] = 1/θ^(2k/rope_dim)`; `freq = pos · inv_freq` (the `VisionRotaryEmbedding` table
/// gathered at the h/w position ids). Built in f32 in plain Rust — it is exact integer-driven math.
pub fn rot_pos_emb(grids: &[Grid], cfg: &VisionGridConfig) -> Result<Array> {
    let merge = cfg.spatial_merge_size;
    let half = (cfg.rope_dim / 2) as usize; // 20
    let inv_freq: Vec<f32> = (0..half)
        .map(|k| 1.0 / cfg.rope_theta.powf((2 * k) as f32 / cfg.rope_dim as f32))
        .collect();

    let mut data: Vec<f32> = Vec::new();
    let mut seq = 0i32;
    for &[t, h, w] in grids {
        let merge_h = h / merge;
        let merge_w = w / merge;
        // The fork builds h/w pos ids on the [h, w] grid, reshapes to (merge_h, merge, merge_w,
        // merge), transposes (0,2,1,3), flattens -> iterate (a, c, b, d) with hpos = a·merge + b,
        // wpos = c·merge + d. Tiled over t.
        for _ti in 0..t {
            for a in 0..merge_h {
                for c in 0..merge_w {
                    for b in 0..merge {
                        for d in 0..merge {
                            let hpos = (a * merge + b) as f32;
                            let wpos = (c * merge + d) as f32;
                            for &f in &inv_freq {
                                data.push(hpos * f);
                            }
                            for &f in &inv_freq {
                                data.push(wpos * f);
                            }
                        }
                    }
                }
            }
        }
        seq += t * h * w;
    }
    Ok(Array::from_slice(&data, &[seq, cfg.rope_dim]))
}
