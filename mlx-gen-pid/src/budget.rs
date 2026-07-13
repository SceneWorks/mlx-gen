//! Memory + command-buffer budget for the PiD super-resolving decode (F-013 sc-9095, tiling sc-10087).
//!
//! PiD denoises directly in **high-resolution pixel space**: `noise`/`x`/each per-step ε is a full
//! `[B,3,H,W]` f32 tensor at the *super-resolved output* resolution, and on top of those the patch
//! stream runs `(H/patch)·(W/patch)` tokens through the MMDiT blocks. A `max_size`-legal 1536²/2048²
//! request super-resolves 4× to 6144²/8192², where a single whole-image forward overflows the envelope
//! **two ways** (sc-10087): on Metal the one long fused-attention command buffer trips the IOGPU
//! watchdog (~100 s abort); on CUDA the true peak exhausts VRAM → silent sysmem-fallback paging.
//!
//! The fix is [spatial tiling](crate::tiling): run the per-step velocity forward on overlapping tiles.
//! This module sizes that — it splits the estimated peak into the two terms tiling treats differently:
//!   * **resident full-res buffers** — `x` + `noise` + the per-step ε + the blend accumulators, all at
//!     the *output* resolution and alive across the whole decode. Tiling does **not** shrink these.
//!   * **per-forward activations** — the pixel-space transients + the patch-stream working set, which
//!     scale with the *forward* grid. Tiling shrinks these to one tile.
//!
//! [`plan_tile_edge`] picks the largest tile whose per-forward term fits `safe_gib` **and** stays under
//! the Metal watchdog's proven-safe forward edge; [`guard`] refuses only when even the resident buffers
//! plus a minimum tile won't fit (a genuinely too-small machine), instead of the old blanket refusal.

use mlx_gen::{Error, Result};

use crate::config::PidConfig;

/// 1 GiB in bytes (`1024³`, matching MLX's `metal::malloc` GiB reporting / the core `memory` module).
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Bytes per f32 element (PiD runs its pixel/patch tensors in f32).
const F32_BYTES: f64 = 4.0;

/// Full-resolution `[B,3,H,W]` f32 tensors held **resident** across the whole decode, at the *output*
/// resolution — the running `x`, the seeded `noise`, the per-step SDE ε (3 for the 4-step distill), and
/// (under tiling) the `acc` blend accumulator. `6` is deliberately conservative (over-counts by ~1 for
/// the 1-channel `wsum` + margin). **Tiling does not reduce this term** — it is the true floor, and the
/// reason the guard still exists as a backstop even with tiling. (sc-10087: the pre-tiling model folded
/// these into a single `9×` pixel count that conflated resident buffers with per-forward transients.)
const RESIDENT_FULL_TENSORS: f64 = 6.0;

/// Pixel-space transient `[B,3,h,w]` tensors live during a **single forward** at the *forward* grid
/// (`pixel_embedder` output, the running `x_pixels`, the final layer) — shrinks to one tile under tiling.
const FWD_PIXEL_TENSORS: f64 = 3.0;

/// Multiplier on the single-forward patch-token working set (`tokens · hidden`) — the several activation
/// temporaries an MMDiT block holds live (QKV, the MLP 4× expansion, the flash-attention working set)
/// times a small factor for concurrently-live blocks. Scales with the *forward* grid (tiled → one tile).
const PATCH_ACT_BLOCK_MULT: f64 = 8.0;

/// Tile edges are chosen as multiples of this (also the feather/overlap granularity). A multiple of the
/// pixel→latent factor (32 for the 4× qwenimage student) and of `patch_size` (16), so tiles stay
/// LQ-slice- and patch-aligned.
pub const TILE_ALIGN: i32 = 512;

/// Smallest tile edge the planner will drop to. Below this the per-tile fixed overheads dominate and the
/// approximation degrades; if even this won't fit, [`guard`] refuses.
pub const MIN_TILE_EDGE: i32 = 1024;

/// Default feather overlap (px) between tiles — matches the seedvr2 spatial-tiler default (sc-5201) and
/// the sc-10087 A/B (2048 px tiles / 256 px overlap → no measurable seam).
pub const DEFAULT_TILE_OVERLAP: i32 = 256;

/// Largest whole-image output edge left **untiled** — proven to complete under the Metal IOGPU watchdog
/// (sc-10087 A/B: a 4096² whole-image forward completes ~43 s/step; 6144² trips the ~100 s watchdog).
/// At or below this (and in budget) the decode takes the exact whole-image path; above it, tiling engages
/// **regardless of how much RAM is free** — the Metal failure is command-buffer *time*, not memory. (On
/// CUDA there is no watchdog; the candle mirror keys untiling on memory alone.)
pub const WATCHDOG_SAFE_EDGE: i32 = 4096;

/// Preferred tile edge when tiling **is** engaged, on a machine with memory to spare. The sc-10087 A/B
/// validated 2048 px tiles (seamless) — and because attention is superlinear, many small forwards are
/// *faster* than few large ones (a 2048² tile forward ≈ 3.9 s vs a 4096² ≈ 43 s), so we prefer this over
/// the larger watchdog-max tile. Shrunk further only when memory forces it (see [`plan_tile_edge`]).
pub const PREFERRED_TILE_EDGE: i32 = 2048;

/// GiB of the resident full-res buffers for a `[b,3,th,tw]` output — the term tiling can't shrink.
fn resident_full_gib(b: f64, th: f64, tw: f64) -> f64 {
    RESIDENT_FULL_TENSORS * b * 3.0 * th * tw * F32_BYTES / GIB
}

/// GiB of a single forward's activations over an `fh × fw` grid (the whole output, or one tile).
fn forward_activations_gib(b: f64, fh: f64, fw: f64, patch_size: i32, hidden: i32) -> f64 {
    let pixels = FWD_PIXEL_TENSORS * b * 3.0 * fh * fw * F32_BYTES;
    let tokens = (fh / patch_size as f64) * (fw / patch_size as f64);
    let patch_act = PATCH_ACT_BLOCK_MULT * b * tokens * hidden as f64 * F32_BYTES;
    (pixels + patch_act) / GIB
}

/// Estimated concurrent peak (GiB) of a **whole-image** (untiled) PiD decode producing `[b,3,th,tw]`:
/// the resident full-res buffers plus a single forward over the whole output grid. Pure (no device
/// query) so it is unit-testable at fixed sizes.
pub fn estimated_decode_peak_gib(b: i32, th: i32, tw: i32, patch_size: i32, hidden: i32) -> f64 {
    let (b, th, tw) = (b as f64, th as f64, tw as f64);
    resident_full_gib(b, th, tw) + forward_activations_gib(b, th, tw, patch_size, hidden)
}

/// Estimated concurrent peak (GiB) of a **tiled** decode of `[b,3,th,tw]` with square `tile`-px tiles:
/// the same resident full-res buffers (unchanged by tiling) plus one forward over a single tile (clamped
/// to the output when the output is smaller than a tile).
pub fn tiled_peak_gib(b: i32, th: i32, tw: i32, tile: i32, patch_size: i32, hidden: i32) -> f64 {
    let (bf, thf, twf) = (b as f64, th as f64, tw as f64);
    let fh = tile.min(th) as f64;
    let fw = tile.min(tw) as f64;
    resident_full_gib(bf, thf, twf) + forward_activations_gib(bf, fh, fw, patch_size, hidden)
}

/// A chosen tiling plan for a decode of output `th × tw`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TilePlan {
    /// Square tile edge (px, multiple of [`TILE_ALIGN`]).
    pub edge: i32,
    /// Feather overlap (px) between neighbouring tiles.
    pub overlap: i32,
    /// `true` when the whole output fits in one tile → decode the exact whole-image path (no blend).
    pub whole_fits: bool,
}

/// Pick a tiling plan for a `[b,3,th,tw]` decode under `safe_gib`. The edge is the largest multiple of
/// [`TILE_ALIGN`] that (a) keeps the per-forward term within budget once the resident buffers are set
/// aside, and (b) stays ≤ [`WATCHDOG_SAFE_EDGE`] (Metal command-buffer safety), floored at
/// [`MIN_TILE_EDGE`]. `whole_fits` is `true` only when that edge already covers the output **and** the
/// output edge is watchdog-safe — i.e. the untiled path is both in-budget and watchdog-safe. Pure.
pub fn plan_tile_edge(
    b: i32,
    th: i32,
    tw: i32,
    patch_size: i32,
    hidden: i32,
    safe_gib: f64,
) -> TilePlan {
    let out_edge = th.max(tw);
    // Untile when the whole output is both watchdog-safe (Metal) and in budget — the exact whole-image
    // path, byte-identical to the pre-tiling decode.
    let whole_fits = out_edge <= WATCHDOG_SAFE_EDGE
        && estimated_decode_peak_gib(b, th, tw, patch_size, hidden) <= safe_gib;
    if whole_fits {
        return TilePlan {
            edge: out_edge,
            overlap: DEFAULT_TILE_OVERLAP,
            whole_fits: true,
        };
    }
    // Tiling engaged. Prefer the validated/fast [`PREFERRED_TILE_EDGE`], shrinking only if memory forces
    // it: the per-forward term must fit what's left after the (un-shrinkable) resident buffers.
    let resident = resident_full_gib(b as f64, th as f64, tw as f64);
    let avail = (safe_gib - resident).max(0.0);
    let per_px2 = forward_activations_gib(b as f64, 1.0, 1.0, patch_size, hidden); // GiB at 1×1
    let mem_edge = if per_px2 > 0.0 {
        (avail / per_px2).max(0.0).sqrt() as i32
    } else {
        out_edge
    };
    let edge = (mem_edge
        .min(PREFERRED_TILE_EDGE)
        .min(out_edge)
        .max(MIN_TILE_EDGE)
        / TILE_ALIGN
        * TILE_ALIGN)
        .max(TILE_ALIGN);
    TilePlan {
        edge,
        overlap: DEFAULT_TILE_OVERLAP,
        whole_fits: false,
    }
}

/// The PiD decode budget backstop (F-013 + sc-10087): with tiling, a large output is *tiled* rather than
/// refused, so this refuses only when even a [`MIN_TILE_EDGE`] tile plus the resident full-res buffers
/// exceeds `safe_gib` — a machine too small to hold the output-resolution `x`/`noise`/ε at all. `model_id`
/// only labels the error. Pure (budget injected) so it is unit-testable without a device query.
///
/// Prices a **single** decode (`B=1`): every PiD consumer decodes one latent at a time — the minted
/// decoder is shared across a request's `count` loop, but each `decode` call holds one output-resolution
/// buffer set — so the concurrent peak never scales with `count` (F-150). Pricing `count` full-resolution
/// buffer sets falsely refused / needlessly shrank tiles for multi-image requests.
pub fn guard(
    model_id: &str,
    width: u32,
    height: u32,
    scale: i32,
    cfg: &PidConfig,
    safe_gib: f64,
) -> Result<()> {
    let tw = (width * scale as u32) as i32;
    let th = (height * scale as u32) as i32;
    let floor = tiled_peak_gib(1, th, tw, MIN_TILE_EDGE, cfg.patch_size, cfg.hidden_size);
    if floor > safe_gib {
        return Err(Error::Msg(format!(
            "{model_id}: a PiD decode at {tw}×{th} (super-resolved {scale}× from {width}×{height}) \
             needs ~{floor:.0} GB even tiled at the minimum {MIN_TILE_EDGE}px tile — the output-resolution \
             noise/latent buffers alone exceed this machine's ~{safe_gib:.0} GB safe budget. Reduce the \
             resolution or disable PiD for this request."
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // sr4x backbone dims (patch_size 16, hidden 1536) — the released students' geometry.
    const PATCH: i32 = 16;
    const HIDDEN: i32 = 1536;

    #[test]
    fn whole_peak_grows_with_area() {
        let base = estimated_decode_peak_gib(1, 512, 512, PATCH, HIDDEN);
        let dbl = estimated_decode_peak_gib(1, 1024, 1024, PATCH, HIDDEN);
        // Both terms scale with area → doubling each edge quadruples the peak.
        assert!((dbl - 4.0 * base).abs() / base < 1e-9);
    }

    #[test]
    fn resident_buffers_are_counted_in_the_peak() {
        // The resident term alone must be non-trivial (the pre-tiling bug was under-counting it): at
        // 6144² it is ~2.5 GiB, so the whole peak must exceed that even before activations.
        let resident = resident_full_gib(1.0, 6144.0, 6144.0);
        assert!(resident > 2.0, "6144² resident buffers ~{resident:.1} GiB");
        let whole = estimated_decode_peak_gib(1, 6144, 6144, PATCH, HIDDEN);
        assert!(whole > resident);
    }

    #[test]
    fn tiling_cuts_the_forward_term() {
        // A 6144² decode: tiling to 2048 px tiles must land well under the whole-image peak (the whole
        // point) while keeping the same resident floor.
        let whole = estimated_decode_peak_gib(1, 6144, 6144, PATCH, HIDDEN);
        let tiled = tiled_peak_gib(1, 6144, 6144, 2048, PATCH, HIDDEN);
        assert!(
            tiled < whole * 0.5,
            "tiled {tiled:.1} vs whole {whole:.1} GiB"
        );
        assert!(tiled > resident_full_gib(1.0, 6144.0, 6144.0));
    }

    #[test]
    fn plan_untiles_small_watchdog_safe_output() {
        // 512²×4 = 2048² output: in-budget and watchdog-safe → whole-image path.
        let plan = plan_tile_edge(1, 2048, 2048, PATCH, HIDDEN, 96.0);
        assert!(plan.whole_fits, "2048² should not tile: {plan:?}");
    }

    #[test]
    fn plan_tiles_6144_even_with_ample_ram() {
        // 1536²×4 = 6144² output on a big-RAM Mac (memory fits!) must STILL tile, because 6144 > the
        // Metal watchdog-safe forward edge — the exact sc-10087 repro.
        let plan = plan_tile_edge(1, 6144, 6144, PATCH, HIDDEN, 100.0);
        assert!(
            !plan.whole_fits,
            "6144² must tile even with RAM to spare: {plan:?}"
        );
        // With memory to spare it uses the preferred (fast, validated) tile, not the watchdog max.
        assert_eq!(plan.edge, PREFERRED_TILE_EDGE);
        assert_eq!(plan.edge % TILE_ALIGN, 0);
    }

    #[test]
    fn plan_shrinks_tile_below_preferred_on_a_small_gpu() {
        // Ample RAM → the preferred tile; a tight budget forces a smaller one.
        let big = plan_tile_edge(1, 8192, 8192, PATCH, HIDDEN, 100.0).edge;
        assert_eq!(big, PREFERRED_TILE_EDGE);
        let small = plan_tile_edge(1, 8192, 8192, PATCH, HIDDEN, 5.0).edge;
        assert!(
            small < big,
            "tighter budget → smaller tile ({small} < {big})"
        );
        assert!(small >= MIN_TILE_EDGE);
    }

    #[test]
    fn guard_admits_large_output_that_tiling_can_handle() {
        // The pre-tiling guard REFUSED 2048²×4=8192² under 14 GiB. With tiling it must ADMIT it (tiled),
        // as long as the resident buffers + a min tile fit.
        let cfg = PidConfig::sr4x();
        assert!(
            guard("qwenimage", 2048, 2048, 4, &cfg, 14.0).is_ok(),
            "8192² should tile, not refuse, under 14 GiB"
        );
    }

    #[test]
    fn guard_refuses_when_even_a_min_tile_wont_fit() {
        // A tiny budget that can't even hold the output-res resident buffers + a MIN_TILE_EDGE forward.
        let cfg = PidConfig::sr4x();
        let err = guard("qwenimage", 2048, 2048, 4, &cfg, 2.0)
            .expect_err("8192² must refuse under a 2 GiB budget");
        let msg = err.to_string();
        assert!(matches!(err, Error::Msg(_)), "typed refusal, got {err:?}");
        assert!(msg.contains("minimum"), "names the min-tile floor: {msg}");
    }

    #[test]
    fn guard_prices_a_single_decode_not_the_request_count() {
        // F-150: the guard prices ONE decode (B=1) — every consumer decodes one latent at a time, so a
        // large `count` must NOT inflate the resident-buffer floor. A budget that admits a 2048²-native
        // (8192² output) single decode must admit it regardless of how many images the request asks for;
        // the pre-fix `guard` multiplied the resident floor by `count` and falsely refused.
        let cfg = PidConfig::sr4x();
        // Sits between a 1× floor and a would-be 16× floor: only the per-decode (B=1) pricing admits it.
        let one = tiled_peak_gib(1, 8192, 8192, MIN_TILE_EDGE, PATCH, HIDDEN);
        let sixteen = tiled_peak_gib(16, 8192, 8192, MIN_TILE_EDGE, PATCH, HIDDEN);
        let budget = (one + sixteen) / 2.0;
        assert!(
            one < budget && budget < sixteen,
            "budget straddles 1× vs 16×"
        );
        assert!(
            guard("qwenimage", 2048, 2048, 4, &cfg, budget).is_ok(),
            "a single 8192² decode fits {budget:.1} GiB regardless of request count"
        );
    }
}
