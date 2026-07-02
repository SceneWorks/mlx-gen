//! Memory budget guard for the PiD super-resolving decode (F-013, sc-9095).
//!
//! PiD denoises directly in **high-resolution pixel space**: `noise`/`x`/each per-step ε is a full
//! `[B, 3, H, W]` f32 tensor at the *super-resolved output* resolution (`H = zH · vae_compression ·
//! scale`), and on top of those the patch stream runs `(H/patch)·(W/patch)` tokens through 14 MMDiT
//! blocks. A `max_size`-legal 2048² request decodes at 8192² (`scale=4`) — ~262 k patch tokens at
//! width 1536 through 14 blocks *plus* the full-grid pixel tensors → an **uncatchable** MLX OOM /
//! SIGKILL on a smaller Mac.
//!
//! This mirrors the shared budgeted-decode convention the rest of the workspace already deploys —
//! wan's `preflight_denoise_memory_guard` (sc-4986) and z48 vae22 tiling (sc-4998), seedvr2's
//! `plan_chunk_size` (sc-8135/sc-8261) — all of which compare an estimated concurrent GPU peak
//! against [`mlx_gen::memory::safe_budget_gib`] (`min(MLX_limit × 0.85, maxBufferLength)`) and refuse
//! before the OS kills the process. PiD's super-res pixel decode has no clean spatial-tiling seam (a
//! single global-attention PixDiT forward over the whole pixel grid, not a stack of local convs like
//! the video VAEs), so the correct minimum here is the **typed over-budget refusal**, raised at the
//! [`crate::resolve_pid_decoder`] seam *before* the caption encode + `PidNet` build.

use mlx_gen::{Error, Result};

use crate::config::PidConfig;

/// 1 GiB in bytes (`1024³`, matching MLX's `metal::malloc` GiB reporting / the core `memory` module).
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Bytes per f32 element (PiD runs its pixel/patch tensors in f32).
const F32_BYTES: f64 = 4.0;

/// Number of full `[B, 3, H, W]` pixel-space tensors held **concurrently** at the decode's peak.
///
/// The sampler keeps the running `x`, draws the initial `noise` and pre-materializes every per-step
/// SDE ε (4-step distill ⇒ 3 interior ε), and a single forward's transient pixel-space activations add
/// roughly two more full-grid tensors. `9` is a deliberately conservative count so the estimate
/// over-shoots rather than under-shoots (what a guard wants) — the same bias wan/seedvr2's cost models
/// take. Not fit to a hardware measurement; PiD's footprint has not been profiled, so the guard errs
/// toward refusing early.
const CONCURRENT_FULL_GRID_TENSORS: f64 = 9.0;

/// Multiplier on the single-block patch-token working set (`tokens · hidden`), standing in for the
/// several activation temporaries a MMDiT block holds live at once (QKV projections, the MLP 4×
/// expansion, attention scores kept in the fused-attention working set) times a small factor for the
/// blocks whose activations overlap. Conservative on the high side for the same over-shoot reason.
const PATCH_ACT_BLOCK_MULT: f64 = 8.0;

/// Estimated concurrent GPU peak (GiB) of a PiD decode producing `[b, 3, th, tw]` pixels, for a
/// backbone with `patch_size` / `hidden` (patch-stream width). Two terms:
///   • the full-resolution pixel-space f32 tensors alive at once
///     ([`CONCURRENT_FULL_GRID_TENSORS`] × `b·3·th·tw·4 B`), and
///   • the patch-stream activation working set — `tokens = (th/patch)·(tw/patch)` at width `hidden`,
///     scaled by [`PATCH_ACT_BLOCK_MULT`] for the concurrently-live block temporaries.
///
/// Pure (no device query) so it is unit-testable against fixed sizes.
pub fn estimated_decode_peak_gib(b: i32, th: i32, tw: i32, patch_size: i32, hidden: i32) -> f64 {
    let (b, th, tw) = (b as f64, th as f64, tw as f64);
    let pixels = CONCURRENT_FULL_GRID_TENSORS * b * 3.0 * th * tw * F32_BYTES;
    let tokens = (th / patch_size as f64) * (tw / patch_size as f64);
    let patch_act = PATCH_ACT_BLOCK_MULT * b * tokens * hidden as f64 * F32_BYTES;
    (pixels + patch_act) / GIB
}

/// Decide whether a PiD decode of `[b, 3, th, tw]` (backbone `patch_size`/`hidden`) fits under
/// `safe_gib`. Pure so the refusal logic is unit-testable without a device query. Returns the
/// estimated peak when it exceeds the budget (⇒ refuse), `None` when it fits.
pub fn over_budget_peak_gib(
    b: i32,
    th: i32,
    tw: i32,
    patch_size: i32,
    hidden: i32,
    safe_gib: f64,
) -> Option<f64> {
    let peak = estimated_decode_peak_gib(b, th, tw, patch_size, hidden);
    (peak > safe_gib).then_some(peak)
}

/// The PiD decode memory-budget guard (F-013): given the request's native `width × height`, image
/// `count`, the engine's super-res `scale`, the backbone `cfg` (for `patch_size`/`hidden_size`), and a
/// `safe_gib` ceiling, return a typed over-budget [`Error`] when the super-resolved decode's estimated
/// peak exceeds the budget, else `Ok(())`.
///
/// `model_id` only labels the error. Pure (the budget is injected) so the seam's refusal is
/// unit-testable at any resolution/budget without loading a [`crate::PidEngine`] or querying the
/// device — the [`crate::resolve_pid_decoder`] seam calls it with the live
/// [`mlx_gen::memory::safe_budget_gib`].
pub fn guard(
    model_id: &str,
    count: u32,
    width: u32,
    height: u32,
    scale: i32,
    cfg: &PidConfig,
    safe_gib: f64,
) -> Result<()> {
    let tw = (width * scale as u32) as i32;
    let th = (height * scale as u32) as i32;
    let b = count.max(1) as i32;
    if let Some(peak) = over_budget_peak_gib(b, th, tw, cfg.patch_size, cfg.hidden_size, safe_gib) {
        return Err(Error::Msg(format!(
            "{model_id}: a PiD decode at {tw}×{th} (super-resolved {scale}× from {width}×{height}) \
             needs ~{peak:.0} GB of concurrent pixel-space + patch-stream tensors, exceeding this \
             machine's ~{safe_gib:.0} GB safe budget (min(MLX limit × 0.85, maxBufferLength)). \
             Unmitigated this OOM/SIGKILLs the worker (F-013). Reduce the resolution or disable PiD \
             for this request."
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
    fn peak_grows_super_linearly_with_edge() {
        // Both terms scale with area, so doubling each edge quadruples the peak.
        let base = estimated_decode_peak_gib(1, 512, 512, PATCH, HIDDEN);
        let dbl = estimated_decode_peak_gib(1, 1024, 1024, PATCH, HIDDEN);
        assert!((dbl - 4.0 * base).abs() / base < 1e-9);
    }

    #[test]
    fn eightk_decode_exceeds_a_small_mac_budget() {
        // A max_size-legal 2048² request super-resolves to 8192² (scale=4): the exact OOM the guard
        // exists to catch. The two-term peak (pixel grid + 262 k patch tokens × 14-block working set)
        // lands well above a smaller Mac's safe budget (~13.6 GiB on a 16 GB machine).
        let peak = estimated_decode_peak_gib(1, 8192, 8192, PATCH, HIDDEN);
        assert!(
            peak > 14.0,
            "8192² peak should exceed a small-Mac budget, got {peak}"
        );
    }

    #[test]
    fn guard_refuses_max_size_legal_request_that_super_resolves_over_budget() {
        // A max_size-legal 2048² request at scale=4 decodes at 8192² → over a 14 GiB budget: the guard
        // must return a typed over-budget error naming both the super-res target and the budget.
        let cfg = PidConfig::sr4x();
        let err = guard("qwenimage", 1, 2048, 2048, 4, &cfg, 14.0)
            .expect_err("2048²×4 super-res must be refused under a 14 GiB budget");
        let msg = err.to_string();
        assert!(matches!(err, Error::Msg(_)), "typed refusal, got {err:?}");
        assert!(msg.contains("8192×8192"), "names super-res target: {msg}");
        assert!(msg.contains("safe budget"), "names the budget: {msg}");
    }

    #[test]
    fn guard_admits_in_budget_request() {
        // A 512² request at scale=4 (2048² decode) fits comfortably under a 14 GiB budget → Ok.
        let cfg = PidConfig::sr4x();
        assert!(guard("qwenimage", 1, 512, 512, 4, &cfg, 14.0).is_ok());
    }

    #[test]
    fn guard_counts_the_batch_toward_the_peak() {
        // A per-image size that fits alone can tip over-budget once `count` multiplies the concurrent
        // tensors — the guard must scale with count.
        let cfg = PidConfig::sr4x();
        let safe = 14.0;
        assert!(guard("qwenimage", 1, 1024, 1024, 4, &cfg, safe).is_ok());
        assert!(
            guard("qwenimage", 8, 1024, 1024, 4, &cfg, safe).is_err(),
            "count=8 should exceed the budget the single image fits under"
        );
    }
}
