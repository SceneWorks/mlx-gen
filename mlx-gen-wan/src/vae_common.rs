//! Layout-agnostic VAE scaffolding shared between the z16 [`vae`](crate::vae) (NCTHW) and the z48
//! [`vae22`](crate::vae22) (channels-last NTHWC) Wan VAEs. Only the pieces that are genuinely
//! byte-identical across the two layouts live here; the per-file conv/norm leaves (which carry the
//! channel axis at different positions) stay with their respective modules.

use mlx_gen::tiling::TilePlan;
use mlx_gen::{CancelFlag, Error, Result};
use mlx_rs::ops::{add, divide, maximum, multiply, pad};
use mlx_rs::Array;

/// A length-1 `f32` array, used as a broadcastable scalar operand in MLX ops.
pub(crate) fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// Force a logically-contiguous copy. mlx-rs host reads (`as_slice`) return the *physical* buffer,
/// so an array left strided by a `transpose` is read scrambled. A reshape round-trip materializes
/// logical order. Internal mlx ops are stride-aware, so this is only needed at the host-read
/// boundary (the public decode/encode output).
pub(crate) fn contiguous(x: &Array) -> Result<Array> {
    let shape = x.shape().to_vec();
    Ok(x.reshape(&[-1])?.reshape(&shape)?)
}

/// Gather the contiguous range `[start, end)` along `axis` (mlx-rs has no slice op). Layout-agnostic:
/// the z16 VAE slices the temporal/spatial axes at 2/3/4, the z48 VAE (channels-last) at 1/2/3.
pub(crate) fn slice_axis(x: &Array, axis: i32, start: i32, end: i32) -> Result<Array> {
    let idx: Vec<i32> = (start..end).collect();
    Ok(x.take_axis(Array::from_slice(&idx, &[end - start]), axis)?)
}

/// Last `n` frames along the temporal axis `t_axis`: the reference `x[…, -n:, …]`. The temporal axis
/// differs by layout — z16 (NCTHW) at 2, z48 (channels-last NTHWC) at 1 — so each VAE binds its axis.
pub(crate) fn last_t_axis(x: &Array, n: i32, t_axis: i32) -> Result<Array> {
    let t = x.shape()[t_axis as usize];
    let idx: Vec<i32> = (t - n..t).collect();
    Ok(x.take_axis(Array::from_slice(&idx, &[n]), t_axis)?)
}

/// Bound the lazy computation graph + peak memory by forcing materialization here (the reference's
/// per-chunk / per-tile `mx.eval`). Used by the z48 VAE's chunked encode/decode to keep the wider
/// feature maps from accumulating into one unbounded graph.
pub(crate) fn eval(x: &Array) -> Result<()> {
    mlx_rs::transforms::eval([x])?;
    Ok(())
}

/// Per-conv last-frames cache threaded through the chunked encode. `idx` resets to 0 each chunk and
/// advances once per cache-bearing conv (in the fixed traversal order), so slots stay aligned.
pub(crate) struct FeatCache {
    pub(crate) slots: Vec<Option<Array>>,
    pub(crate) idx: usize,
}
impl FeatCache {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            slots: vec![None; n],
            idx: 0,
        }
    }
}

/// `[1; 5]` with `len` placed at `axis` — a 1-D blend mask reshaped to broadcast along its own axis.
fn axis_shape(axis: i32, len: i32) -> [i32; 5] {
    let mut s = [1i32; 5];
    s[axis as usize] = len;
    s
}

/// The trapezoidally-blended tile-accumulate loop shared by both VAEs' `decode_tiled`. Slices each
/// overlapping tile out of `denorm`, decodes it via the layout-specific `decode_tile` closure
/// (conv2 → decoder → optional unpatchify → clamp), trapezoidally blends along the three tiled axes,
/// and accumulates into the full output. `axes` are the `[t, h, w]` axis indices for the layout
/// (`[2, 3, 4]` for NCTHW z16, `[1, 2, 3]` for channels-last z48); the mask shapes and pad
/// placements derive from those indices, so the only per-layout input is the closure.
///
/// `denorm` is the already-denormalized latent; `plan` comes from
/// [`TilingConfig::plan`](mlx_gen::tiling::TilingConfig::plan). The reference's per-tile `mx.eval`
/// (bounding the lazy graph + peak memory) is preserved.
///
/// `cancel` is the cooperative cancellation handle (F-014): the z48 decode is ~95% of a Lightning
/// render's wall-clock (sc-4998), so a cancel is checked between tiles and returns [`Error::Canceled`].
/// The per-tile `eval` already forces materialization, so the check observes the trip promptly.
pub(crate) fn tile_decode_accumulate(
    denorm: &Array,
    plan: &TilePlan,
    axes: [i32; 3],
    cancel: Option<&CancelFlag>,
    decode_tile: impl Fn(&Array) -> Result<Array>,
) -> Result<Array> {
    let [t_ax, h_ax, w_ax] = axes;
    let mut output: Option<Array> = None;
    let mut weights: Option<Array> = None;
    for t in &plan.t {
        for hh in &plan.h {
            for ww in &plan.w {
                if cancel.is_some_and(CancelFlag::is_cancelled) {
                    return Err(Error::Canceled);
                }
                let tile = slice_axis(denorm, t_ax, t.start, t.end)?;
                let tile = slice_axis(&tile, h_ax, hh.start, hh.end)?;
                let tile = slice_axis(&tile, w_ax, ww.start, ww.end)?;
                let dec = decode_tile(&tile)?;

                let ds = dec.shape();
                let at = ds[t_ax as usize].min(t.out_stop - t.out_start);
                let ah = ds[h_ax as usize].min(hh.out_stop - hh.out_start);
                let aw = ds[w_ax as usize].min(ww.out_stop - ww.out_start);

                // 1-D masks → outer product, each broadcasting along its own (t/h/w) axis.
                let tm = Array::from_slice(&t.mask[..at as usize], &axis_shape(t_ax, at));
                let hm = Array::from_slice(&hh.mask[..ah as usize], &axis_shape(h_ax, ah));
                let wm = Array::from_slice(&ww.mask[..aw as usize], &axis_shape(w_ax, aw));
                let blend = multiply(&multiply(&tm, &hm)?, &wm)?;

                let dec = slice_axis(&dec, t_ax, 0, at)?;
                let dec = slice_axis(&dec, h_ax, 0, ah)?;
                let dec = slice_axis(&dec, w_ax, 0, aw)?;
                let weighted = multiply(&dec, &blend)?;

                // Place at the (out_start) offsets by zero-padding to the full output shape.
                let mut pads = [(0, 0); 5];
                pads[t_ax as usize] = (t.out_start, plan.out_f - (t.out_start + at));
                pads[h_ax as usize] = (hh.out_start, plan.out_h - (hh.out_start + ah));
                pads[w_ax as usize] = (ww.out_start, plan.out_w - (ww.out_start + aw));
                let weighted_full = pad(&weighted, &pads[..], None, None)?;
                let blend_full = pad(&blend, &pads[..], None, None)?;

                output = Some(match output {
                    None => weighted_full,
                    Some(acc) => add(&acc, &weighted_full)?,
                });
                weights = Some(match weights {
                    None => blend_full,
                    Some(acc) => add(&acc, &blend_full)?,
                });
                // Bound the lazy graph + peak memory (the reference's per-tile `mx.eval`).
                output.as_ref().unwrap().eval()?;
                weights.as_ref().unwrap().eval()?;
            }
        }
    }

    let output =
        output.ok_or_else(|| Error::Msg("wan vae: tile-decode plan had no tiles".into()))?;
    let weights =
        weights.ok_or_else(|| Error::Msg("wan vae: tile-decode plan had no tiles".into()))?;
    contiguous(&divide(&output, &maximum(&weights, scalar(1e-8))?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::tiling::{
        AxisTile, SpatialTiling, TemporalTiling, TilePlan, TilingConfig, VaeTiling,
    };

    /// Two non-overlapping tiles along the temporal axis with all-ones masks and an identity decode
    /// must exactly reconstruct the input — exercising slice/mask/pad placement and accumulation for
    /// a given axis layout. Returns the round-tripped values for comparison against the input.
    fn roundtrip(denorm: &Array, axes: [i32; 3], t_full: i32) -> Vec<f32> {
        let half = t_full / 2;
        let tile = |start, out_start| AxisTile {
            start,
            end: start + half,
            out_start,
            out_stop: out_start + half,
            mask: vec![1.0; half as usize],
        };
        let unit = AxisTile {
            start: 0,
            end: 2,
            out_start: 0,
            out_stop: 2,
            mask: vec![1.0; 2],
        };
        let plan = TilePlan {
            t: vec![tile(0, 0), tile(half, half)],
            h: vec![unit.clone()],
            w: vec![unit],
            out_f: t_full,
            out_h: 2,
            out_w: 2,
        };
        let out =
            tile_decode_accumulate(denorm, &plan, axes, None, |tile| Ok(tile.clone())).unwrap();
        out.eval().unwrap();
        out.as_slice::<f32>().to_vec()
    }

    #[test]
    fn identity_roundtrip_ncthw() {
        // [1, 1, 4, 2, 2] — channel axis at 1, tiled axes [2, 3, 4].
        let vals: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let denorm = Array::from_slice(&vals, &[1, 1, 4, 2, 2]);
        assert_eq!(roundtrip(&denorm, [2, 3, 4], 4), vals);
    }

    #[test]
    fn identity_roundtrip_channels_last() {
        // [1, 4, 2, 2, 1] — channel axis last, tiled axes [1, 2, 3].
        let vals: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let denorm = Array::from_slice(&vals, &[1, 4, 2, 2, 1]);
        assert_eq!(roundtrip(&denorm, [1, 2, 3], 4), vals);
    }

    /// F-014: a pre-tripped [`CancelFlag`] makes the tiled decode return [`Error::Canceled`] at the
    /// first tile boundary instead of decoding every tile of a dominant-cost VAE pass.
    #[test]
    fn tile_decode_honors_pretripped_cancel() {
        let vals: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let denorm = Array::from_slice(&vals, &[1, 1, 4, 2, 2]);
        let half = 2;
        let tile = |start, out_start| AxisTile {
            start,
            end: start + half,
            out_start,
            out_stop: out_start + half,
            mask: vec![1.0; half as usize],
        };
        let unit = AxisTile {
            start: 0,
            end: 2,
            out_start: 0,
            out_stop: 2,
            mask: vec![1.0; 2],
        };
        let plan = TilePlan {
            t: vec![tile(0, 0), tile(half, half)],
            h: vec![unit.clone()],
            w: vec![unit],
            out_f: 4,
            out_h: 2,
            out_w: 2,
        };
        let cancel = CancelFlag::new();
        cancel.cancel();
        let res =
            tile_decode_accumulate(&denorm, &plan, [2, 3, 4], Some(&cancel), |t| Ok(t.clone()));
        assert!(
            matches!(res, Err(Error::Canceled)),
            "a pre-tripped cancel must abort the tiled decode with Error::Canceled"
        );
    }

    /// Block (nearest-neighbour) upsample by `scales` along `axes` — `repeat_axis` along each. This is
    /// **tile-consistent**: upsampling a contiguous latent slice `[begin, end)` yields exactly the
    /// `[begin·s, end·s)` slab of the full upsample, with no seam, so overlapping tiles agree wherever
    /// they overlap. A correct partition-of-unity blend must therefore reconstruct the full upsample
    /// exactly — which isolates the slice/mask/pad/accumulate/normalize machinery from any real-decoder
    /// seam residual.
    fn upsample(x: &Array, axes: [i32; 3], scales: [i32; 3]) -> Array {
        let mut y = x.clone();
        for (&ax, &s) in axes.iter().zip(scales.iter()) {
            if s > 1 {
                y = Array::repeat_axis::<f32>(y, s, ax).unwrap();
            }
        }
        y
    }

    /// sc-5690 regression: when a plan tiles **two or three axes at once** (the case the production
    /// `auto` path emits for high-res long video), `tile_decode_accumulate` must still reconstruct a
    /// tile-consistent decode **exactly**. Drives the real [`TilingConfig::plan`] geometry (scaled
    /// trapezoidal masks, overlap, ragged last tiles, `out_start` placement) with a tile-consistent
    /// `upsample` decode and asserts the blended output equals the single full upsample. Covers the
    /// asymmetric shape from the bug report (temporal + one spatial axis, the other a single tile) for
    /// which the symmetric `tiling_parity` golden had no coverage.
    fn assert_combined_reconstructs(
        shape: [i32; 5],
        axes: [i32; 3],
        cfg: &TilingConfig,
        label: &str,
    ) {
        // Wan z16 (non-causal): temporal ×4, spatial ×8 — these match `upsample`'s block factors so a
        // tiled decode reconstructs the full upsample. (`axes` selects the channel-position layout;
        // the per-axis geometry is layout-independent.)
        let vae = VaeTiling::WAN;
        let scales = [vae.temporal_scale, vae.spatial_scale, vae.spatial_scale];
        let (f, h, w) = (
            shape[axes[0] as usize],
            shape[axes[1] as usize],
            shape[axes[2] as usize],
        );
        assert!(
            cfg.needs_tiling(vae, f, h, w),
            "{label}: config must actually tile [{f},{h},{w}]"
        );
        let plan = cfg.plan(vae, f, h, w);
        let n: i32 = shape.iter().product();
        let vals: Vec<f32> = (0..n).map(|i| (i as f32 * 0.137).sin()).collect();
        let denorm = Array::from_slice(&vals, &shape);

        let expected = upsample(&denorm, axes, scales);
        let got = tile_decode_accumulate(&denorm, &plan, axes, None, |t| {
            Ok(upsample(t, axes, scales))
        })
        .unwrap();
        got.eval().unwrap();
        assert_eq!(
            got.shape(),
            expected.shape(),
            "{label}: tiled output shape (t={} h={} w={} tiles)",
            plan.t.len(),
            plan.h.len(),
            plan.w.len()
        );
        let (g, e) = (got.as_slice::<f32>(), expected.as_slice::<f32>());
        let max = g
            .iter()
            .zip(e)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max < 1e-4,
            "{label}: combined-axis blend did not reconstruct (t={} h={} w={} tiles) max|Δ|={max:.3e}",
            plan.t.len(),
            plan.h.len(),
            plan.w.len()
        );
    }

    /// Small config that tiles a z16 latent: spatial 64 px → 8 latent / 32 px → 4 overlap; temporal
    /// 16 f → 4 latent / 8 f → 2 overlap.
    fn small_cfg() -> TilingConfig {
        TilingConfig {
            spatial: Some(SpatialTiling {
                tile_px: 64,
                overlap_px: 32,
            }),
            temporal: Some(TemporalTiling {
                tile_frames: 16,
                overlap_frames: 8,
            }),
        }
    }

    #[test]
    fn combined_axes_reconstruct_ncthw() {
        // NCTHW (z16 layout), channel axis at 1, tiled axes [2, 3, 4].
        let cfg = small_cfg();
        // The bug-report shape class: temporal tiles + ONE spatial axis tiles, the other is a single
        // tile (h = 8 latent ≤ the 8-latent tile). This asymmetric combined plan was untested.
        assert_combined_reconstructs([1, 2, 6, 8, 12], [2, 3, 4], &cfg, "asym t+w (h single)");
        assert_combined_reconstructs([1, 2, 6, 12, 8], [2, 3, 4], &cfg, "asym t+h (w single)");
        // Fully symmetric (all three axes tile).
        assert_combined_reconstructs([1, 2, 6, 12, 12], [2, 3, 4], &cfg, "symmetric t+h+w");
        // Ragged last tile on every axis (3 tiles each, last shorter).
        assert_combined_reconstructs([1, 2, 7, 13, 13], [2, 3, 4], &cfg, "ragged 3×3×3");
    }

    #[test]
    fn combined_axes_reconstruct_channels_last() {
        // Channels-last (z48 vae22 layout), channel axis last, tiled axes [1, 2, 3].
        let cfg = small_cfg();
        assert_combined_reconstructs([1, 6, 8, 12, 2], [1, 2, 3], &cfg, "cl asym t+w (h single)");
        assert_combined_reconstructs([1, 6, 12, 12, 2], [1, 2, 3], &cfg, "cl symmetric t+h+w");
        assert_combined_reconstructs([1, 7, 13, 13, 2], [1, 2, 3], &cfg, "cl ragged 3×3×3");
    }
}
