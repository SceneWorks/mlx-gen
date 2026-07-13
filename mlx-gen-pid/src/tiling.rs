//! Spatial tiling for the PiD pixel-space decode (sc-10087).
//!
//! At large output resolutions (≥~6144²) a single whole-image `PidNet::forward` overflows the
//! practical memory + command-buffer envelope: on Metal the one long fused-attention command buffer
//! trips the IOGPU watchdog (~100 s abort); on CUDA the true peak exhausts VRAM → silent sysmem-fallback
//! paging. The dominant peak term is the per-forward O(area) pixel + patch-stream activations, so we
//! attack it directly: split the pixel grid into overlapping tiles and run the **velocity** forward one
//! tile at a time, feather-blending the per-tile predictions back into a full-resolution `v`.
//!
//! ## Fidelity note — this is an *approximation*, not a transparent split
//! The `PixDiT` is globally self-attentive at both stages (the MMDiT patch stream attends over all
//! `L = Hs·Ws` image tokens; the PiT pixel stream compresses each patch to one token and attends over
//! all `L`). Tiling therefore drops cross-tile attention within a step. PiD's global structure comes
//! from the small, whole **LQ latent** conditioning (each tile still receives its aligned latent slice),
//! so the pixel diffusion is mostly local detail synthesis super-resolving from that latent — which is
//! why overlap + feather blend is expected to be near-seamless. Validated empirically by the real-weight
//! A/B (tiled vs whole-image at 1024-native → 4096²) before this path is committed.
//!
//! ## Design
//! - The 4-step SDE loop stays **whole-image** (see [`crate::sampler::Sampler::run_tiled`]): `x`, the
//!   seeded `noise`, and each ε remain full-resolution `[B,3,H,W]`, so the sampler math and the
//!   launch-portable RNG sequence (sc-3673) are byte-for-byte unchanged. Only the per-step *forward* is
//!   tiled. Peak ≈ (full-res `x`/`noise`/ε, fixed) + one tile's activations.
//! - Tiles are aligned to `F = H / zH` (the pixel→latent factor, 32 for the 4× qwenimage student:
//!   vae_compression 8 × sr_scale 4). `F` is a multiple of `patch_size` (16), so every tile is also
//!   patch-aligned and its LQ-latent slice `lq[:, :, y0/F:y1/F, x0/F:x1/F]` is integer and consistent
//!   with the tile's patch grid.
//! - Blend mirrors the seedvr2 spatial-tiler (sc-5201): `feather_weight` tapers edges that abut a
//!   neighbor; accumulate `acc += pad(v_tile·w)`, `wsum += pad(w)`, return `acc/wsum`; `eval` per tile so
//!   activations don't stack.

use mlx_rs::ops::{add, divide, multiply, pad, split_sections};
use mlx_rs::transforms::eval;
use mlx_rs::Array;

use mlx_gen::Result;

use crate::lq::PidNet;
use crate::memo::{memo, TableCache};

/// Per-decode cache of the separable feather-weight tables (F-153), keyed by a tile's
/// `(th, tw, fade_top, fade_bottom, fade_left, fade_right, overlap)`. The tile plan is deterministic
/// across the 4 sampler steps, so each distinct tile shape's feather is built once (host loop + H2D)
/// rather than per tile per step. Holds the raw f32 table; the per-tile dtype cast stays per use.
pub type FeatherCache = TableCache<(i32, i32, bool, bool, bool, bool, i32), Array>;

/// A spatial tile's pixel-space extent `[y0,y1) × [x0,x1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpatialTile {
    pub y0: i32,
    pub y1: i32,
    pub x0: i32,
    pub x1: i32,
}

/// Tile a length-`n` axis into full-`tile`-size windows at stride `tile-overlap`; the final window's
/// start is clamped so it stays full size and ends at `n` (overlapping its neighbor a little more).
/// One window covers `n <= tile`. With `n`/`tile`/`overlap` all multiples of `align`, every start/end
/// is too. Mirrors `mlx-gen-seedvr2::video::tile_axis`.
fn tile_axis(n: i32, tile: i32, overlap: i32) -> Vec<(i32, i32)> {
    if n <= tile {
        return vec![(0, n)];
    }
    let stride = (tile - overlap).max(1);
    let mut out = Vec::new();
    let mut s = 0;
    loop {
        let e = (s + tile).min(n);
        let start = (e - tile).max(0); // keep the tile full size
        if out.last() != Some(&(start, e)) {
            out.push((start, e));
        }
        if e >= n {
            break;
        }
        s += stride;
    }
    out
}

/// The grid of overlapping tiles covering an `h × w` grid. `tile`/`overlap` are rounded **down** to a
/// multiple of `align` (the pixel→latent factor) so each tile's LQ slice and patch grid stay integer;
/// `tile` is floored to at least `align`.
pub fn plan_tiles(h: i32, w: i32, tile: i32, overlap: i32, align: i32) -> Vec<SpatialTile> {
    let align = align.max(1);
    let tile = (tile / align * align).max(align);
    let overlap = (overlap / align * align).clamp(0, tile - align);
    let ys = tile_axis(h, tile, overlap);
    let xs = tile_axis(w, tile, overlap);
    let mut out = Vec::with_capacity(ys.len() * xs.len());
    for &(y0, y1) in &ys {
        for &(x0, x1) in &xs {
            out.push(SpatialTile { y0, y1, x0, x1 });
        }
    }
    out
}

/// Linear taper along one axis: ramp up over the first `overlap` px when `fade_start`, down over the
/// last `overlap` when `fade_end`, 1 in between; floored positive so the accumulated weight is never
/// zero. Mirrors `mlx-gen-seedvr2::video::axis_ramp`.
fn axis_ramp(len: i32, fade_start: bool, fade_end: bool, overlap: i32) -> Vec<f32> {
    let ov = overlap.max(1);
    (0..len)
        .map(|i| {
            let mut w = 1.0f32;
            if fade_start && i < ov {
                w = w.min((i as f32 + 1.0) / (ov as f32 + 1.0));
            }
            if fade_end && i >= len - ov {
                w = w.min((len - i) as f32 / (ov as f32 + 1.0));
            }
            w.max(1e-4)
        })
        .collect()
}

/// Separable per-pixel feather weights `(th·tw)` for a tile, tapering to ~0 over `overlap` px on each
/// edge abutting a neighbor (`fade_*`) and staying 1 at outer image edges. `w = ry·rx`. Assembly divides
/// by the accumulated weight, so exact partition-of-unity isn't required. Mirrors seedvr2.
fn feather_weight(
    th: i32,
    tw: i32,
    fade_top: bool,
    fade_bottom: bool,
    fade_left: bool,
    fade_right: bool,
    overlap: i32,
) -> Vec<f32> {
    let ry = axis_ramp(th, fade_top, fade_bottom, overlap);
    let rx = axis_ramp(tw, fade_left, fade_right, overlap);
    let mut out = vec![0f32; (th * tw) as usize];
    for y in 0..th as usize {
        for x in 0..tw as usize {
            out[y * tw as usize + x] = ry[y] * rx[x];
        }
    }
    out
}

/// Slice `a`'s axis `axis` to `[start, end)` with a zero-copy strided split at the fixed tile
/// boundaries, vs. an arange `take_axis` gather of a full-res tensor per tile per step (F-152).
fn slice_axis(a: &Array, axis: i32, start: i32, end: i32) -> Result<Array> {
    let len = a.shape()[axis as usize];
    // Cut only at the boundaries that are interior to the axis; the wanted segment is the one that
    // begins at `start` — index 1 when there is a leading `[0..start)` piece, else index 0.
    let mut cuts = Vec::with_capacity(2);
    if start > 0 {
        cuts.push(start);
    }
    if end < len {
        cuts.push(end);
    }
    if cuts.is_empty() {
        return Ok(a.clone()); // whole axis
    }
    let want = if start > 0 { 1 } else { 0 };
    Ok(split_sections(a, &cuts, axis)?.swap_remove(want))
}

/// One tiled **velocity** forward: compute `v = net(x, t, …)` for the whole `[B,3,H,W]` grid by running
/// the net on overlapping pixel tiles (each with its aligned `lq_latent` slice) and feather-blending the
/// per-tile predictions. `overlap` is the feather width (px). `eval`s per tile so tile activations don't
/// stack — which is also what keeps each Metal command buffer short enough to dodge the watchdog.
#[allow(clippy::too_many_arguments)]
pub fn forward_tiled(
    net: &PidNet,
    x: &Array,
    t_scaled: &Array,
    caption: &Array,
    lq_latent: &Array,
    sigma: &Array,
    tile: i32,
    overlap: i32,
    feather_cache: &FeatherCache,
) -> Result<Array> {
    let sh = x.shape();
    let (h, w) = (sh[2], sh[3]);
    let z_h = lq_latent.shape()[2];
    // Pixel→latent factor (32 for the 4× qwenimage student). Tiles align to this so LQ slices are
    // integer and each tile's patch grid matches its LQ projection.
    let f = (h / z_h).max(1);
    let plan = plan_tiles(h, w, tile, overlap, f);

    // Whole-image single tile → just the plain forward (no blend overhead, exact).
    if plan.len() == 1 {
        return net.forward(x, t_scaled, caption, lq_latent, sigma);
    }

    let mut acc: Option<Array> = None; // [B,3,H,W]
    let mut wsum: Option<Array> = None; // [1,1,H,W]
    for tl in &plan {
        let (th, tw) = (tl.y1 - tl.y0, tl.x1 - tl.x0);
        let x_tile = slice_axis(&slice_axis(x, 2, tl.y0, tl.y1)?, 3, tl.x0, tl.x1)?;
        let lq_tile = slice_axis(
            &slice_axis(lq_latent, 2, tl.y0 / f, tl.y1 / f)?,
            3,
            tl.x0 / f,
            tl.x1 / f,
        )?;
        let v_tile = net.forward(&x_tile, t_scaled, caption, &lq_tile, sigma)?; // [B,3,th,tw]

        // Feather table memoized per tile geometry + fade pattern (F-153) — the plan repeats across the
        // 4 steps, so each shape is built once; the dtype cast stays per use (byte-identical weight).
        let (fade_top, fade_bottom, fade_left, fade_right) =
            (tl.y0 > 0, tl.y1 < h, tl.x0 > 0, tl.x1 < w);
        let key = (
            th,
            tw,
            fade_top,
            fade_bottom,
            fade_left,
            fade_right,
            overlap,
        );
        let weight = memo(feather_cache, key, || {
            let wvec = feather_weight(
                th,
                tw,
                fade_top,
                fade_bottom,
                fade_left,
                fade_right,
                overlap,
            );
            Array::from_slice(&wvec, &[1, 1, th, tw])
        })
        .as_dtype(v_tile.dtype())?;
        let pad_spec = [(0, 0), (0, 0), (tl.y0, h - tl.y1), (tl.x0, w - tl.x1)];
        let wv = pad(&multiply(&v_tile, &weight)?, &pad_spec[..], None, None)?;
        let wp = pad(&weight, &pad_spec[..], None, None)?;
        acc = Some(match acc {
            Some(a) => add(&a, &wv)?,
            None => wv,
        });
        wsum = Some(match wsum {
            Some(a) => add(&a, &wp)?,
            None => wp,
        });
        // Materialize so this tile's activations (and prior graph) are freed before the next tile, and
        // so each Metal command buffer stays short (watchdog).
        eval([acc.as_ref().unwrap(), wsum.as_ref().unwrap()])?;
    }
    Ok(divide(acc.expect("≥1 tile"), wsum.expect("≥1 tile"))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_axis_covers_without_gaps_and_stays_full_size() {
        let ax = tile_axis(4096, 2048, 256);
        assert_eq!(ax.first().unwrap().0, 0);
        assert_eq!(ax.last().unwrap().1, 4096);
        for &(s, e) in &ax {
            assert_eq!(e - s, 2048, "every window is full tile size");
        }
        // consecutive windows overlap (no gaps)
        for pair in ax.windows(2) {
            assert!(pair[1].0 < pair[0].1, "windows overlap");
        }
    }

    #[test]
    fn single_window_when_grid_fits() {
        assert_eq!(tile_axis(2048, 2048, 256), vec![(0, 2048)]);
        assert_eq!(tile_axis(1024, 2048, 256), vec![(0, 1024)]);
    }

    #[test]
    fn plan_tiles_aligns_to_factor() {
        // tile/overlap rounded down to multiples of align=32; all starts/ends divisible by 32.
        let plan = plan_tiles(4096, 4096, 2000, 250, 32);
        assert!(!plan.is_empty());
        for t in &plan {
            for v in [t.y0, t.y1, t.x0, t.x1] {
                assert_eq!(v % 32, 0, "tile edge {v} aligned to 32");
            }
        }
    }

    #[test]
    fn whole_image_is_one_tile() {
        assert_eq!(plan_tiles(4096, 4096, 8192, 256, 32).len(), 1);
    }

    /// F-152: the strided-split `slice_axis` is element-for-element identical to the arange `take_axis`
    /// gather it replaced, for a prefix (`start==0`), an interior slice, a suffix (`end==len`), and the
    /// whole axis — on every axis of a small tensor.
    #[test]
    fn slice_axis_matches_take_axis_gather() {
        use mlx_rs::ops::{abs, max, subtract};
        let a = Array::from_iter(0..(2 * 3 * 4 * 5), &[2, 3, 4, 5])
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap();
        let gather = |axis: i32, start: i32, end: i32| {
            let idx = Array::from_slice(&(start..end).collect::<Vec<i32>>(), &[end - start]);
            a.take_axis(&idx, axis).unwrap()
        };
        for (axis, len) in [(0, 2), (1, 3), (2, 4), (3, 5)] {
            for (start, end) in [(0, len), (0, len - 1), (1, len), (1, len - 1)] {
                if start >= end {
                    continue;
                }
                let got = slice_axis(&a, axis, start, end).unwrap();
                let want = gather(axis, start, end);
                assert_eq!(got.shape(), want.shape(), "axis {axis} [{start},{end})");
                let d = max(abs(subtract(&got, &want).unwrap()).unwrap(), None)
                    .unwrap()
                    .item::<f32>();
                assert_eq!(
                    d, 0.0,
                    "axis {axis} [{start},{end}) element-equal to gather"
                );
            }
        }
    }

    #[test]
    fn feather_tapers_only_abutting_edges() {
        // interior tile: fades on all four edges → corners near 0, center 1.
        let w = feather_weight(64, 64, true, true, true, true, 16);
        assert!(w[0] < 0.1, "top-left corner tapered");
        assert!(w[(32 * 64 + 32) as usize] > 0.9, "center ~1");
        // outer edge (no top fade): first row stays 1 along the non-abutting edge.
        let e = feather_weight(64, 64, false, true, false, true, 16);
        assert!(e[0] > 0.9, "outer corner not tapered");
    }
}
