//! MLX device memory limits for budget-aware tiling.
//!
//! The shared budget convention across the engine crates (seedvr2, wan, …) is "this machine's MLX
//! memory limit × [`SAFE_FRAC`]". That works for the *OS-kill* / MLX-backpressure bound — but it
//! ignores Metal's **per-single-buffer** cap [`maxBufferLength`]: MLX's metal allocator throws at
//! `allocator.cpp` when any one allocation exceeds it (the `"[metal::malloc] … maximum allowed buffer
//! size"` OOM). On a high-RAM Mac with no user GPU cap, the MLX memory limit defaults to ≈ 1.5× the
//! device working set, so `limit × 0.85` can sit *above* `maxBufferLength` — and a budget-sized tile
//! then admits a single allocation that blows past the per-buffer cap (sc-8135: SeedVR2 4× → 4096²
//! OOMs). [`safe_budget_gib`] clamps the budget by the cap so tiling always keeps every single
//! allocation under it, regardless of total RAM / MLX limit / user cap.
//!
//! [`maxBufferLength`]: https://developer.apple.com/documentation/metal/mtldevice/3043259-maxbufferlength

use mlx_rs::memory::get_memory_limit;

/// 1 GiB in bytes (`1024³`, matching MLX's `metal::malloc` GiB reporting).
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Fraction of the MLX memory limit treated as a safe peak (15 % headroom for the OS / other
/// processes / MLX backpressure). Matches the seedvr2 `SAFE_FRAC` / wan `× 0.85` convention.
pub const SAFE_FRAC: f64 = 0.85;

/// Fallback budget (≈ a small Apple-Silicon tier) when the MLX limit is unset/unknown — conservative
/// so the sizer tiles more rather than risking an OOM on an unknown machine.
const UNKNOWN_BUDGET_GIB: f64 = 8.0;

/// The device's Metal per-single-buffer cap — `MTLDevice.maxBufferLength`, the hard limit MLX's metal
/// allocator enforces on any single allocation — in GiB, or `None` when it can't be queried (non-Metal
/// backend, MLX `device_info` unavailable, or the property is 0/unset). This is the exact value MLX
/// compares against in its `"[metal::malloc] … maximum allowed buffer size"` check, queried through
/// MLX's own `device_info` C API so it reflects the *same* device MLX allocates on.
pub fn max_buffer_length_gib() -> Option<f64> {
    /// MLX `device_info` key for `MTLDevice.maxBufferLength`
    /// (`mlx/backend/metal/device_info.cpp`).
    const KEY: &[u8] = b"max_buffer_length\0";
    // SAFETY: `mlx_device_new`/`mlx_get_default_device`/`mlx_device_info_*` are plain C value-query
    // functions: they allocate opaque handles we free below, write through out-pointers, and read the
    // device's static property map. No aliasing of Rust-owned memory; the key is a static NUL-terminated
    // C string. We free both handles on every path.
    unsafe {
        use std::os::raw::c_char;
        let mut dev = mlx_sys::mlx_device_new();
        let got_default = mlx_sys::mlx_get_default_device(&mut dev) == 0;
        let mut cap: Option<f64> = None;
        if got_default {
            let mut info = mlx_sys::mlx_device_info_new();
            if mlx_sys::mlx_device_info_get(&mut info, dev) == 0 {
                let mut val: usize = 0;
                if mlx_sys::mlx_device_info_get_size(&mut val, info, KEY.as_ptr() as *const c_char)
                    == 0
                    && val > 0
                {
                    cap = Some(val as f64 / GIB);
                }
            }
            mlx_sys::mlx_device_info_free(info);
        }
        mlx_sys::mlx_device_free(dev);
        cap
    }
}

/// MLX's current memory limit (the backpressure soft cap; default ≈ 1.5× the device working set,
/// lowered by a user GPU cap) in GiB.
fn memory_limit_gib() -> f64 {
    get_memory_limit() as f64 / GIB
}

/// A raw (pre-cap) safe budget: the MLX memory limit × [`SAFE_FRAC`], falling back to
/// [`UNKNOWN_BUDGET_GIB`] when the limit is unset (< 1 GiB). Pure — no device query — so the cap clamp
/// is unit-testable in isolation via [`clamp_budget_to_cap`].
fn raw_safe_budget_gib() -> f64 {
    let lim = memory_limit_gib();
    let lim = if lim < 1.0 { UNKNOWN_BUDGET_GIB } else { lim };
    lim * SAFE_FRAC
}

/// Clamp a raw peak-GB budget by the per-single-buffer cap so no single allocation the budget admits
/// can exceed `maxBufferLength`. A `None`/non-positive cap (couldn't be queried) leaves the budget
/// unchanged. Pure for unit testing.
pub fn clamp_budget_to_cap(raw_gib: f64, cap_gib: Option<f64>) -> f64 {
    match cap_gib {
        Some(cap) if cap > 0.0 => raw_gib.min(cap),
        _ => raw_gib,
    }
}

/// The safe peak-GB budget for a single tiling pass: `min(MLX_limit × 0.85, maxBufferLength)`.
///
/// The cost model a tiling sizer uses is `peak ≈ weights + activations`, and the single largest Metal
/// buffer within that peak is one *activation* tensor — strictly smaller than the aggregate activation
/// term (which bundles many temporaries). So `peak ≤ budget` ⟹ the largest single buffer `< budget`.
/// Clamping the budget to `maxBufferLength` therefore guarantees every single allocation stays under
/// the per-buffer cap, while the `× 0.85` half preserves the OS/MLX-backpressure headroom. See the
/// module docs for why the clamp is necessary on high-RAM Macs (sc-8135).
pub fn safe_budget_gib() -> f64 {
    clamp_budget_to_cap(raw_safe_budget_gib(), max_buffer_length_gib())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_uses_cap_when_raw_exceeds_it() {
        // 150 GiB MLX limit × 0.85 = 127.5, but a 80 GiB per-buffer cap clamps it down.
        assert_eq!(clamp_budget_to_cap(127.5, Some(80.0)), 80.0);
    }

    #[test]
    fn clamp_noop_when_cap_above_raw() {
        // A user GPU cap already keeps the budget small → cap (80) above raw (54.4) → unchanged.
        assert_eq!(clamp_budget_to_cap(54.4, Some(80.0)), 54.4);
    }

    #[test]
    fn clamp_noop_when_cap_unknown() {
        // Non-Metal / query failure → cap None → leave the budget as-is.
        assert_eq!(clamp_budget_to_cap(127.5, None), 127.5);
    }

    #[test]
    fn clamp_noop_when_cap_nonpositive() {
        assert_eq!(clamp_budget_to_cap(127.5, Some(0.0)), 127.5);
    }

    /// `safe_budget_gib` must never exceed the device's per-single-buffer cap (when queryable). Runs
    /// only where MLX/Metal is available (the macOS test lane); the contract lane builds gen-core, not
    /// this crate.
    #[test]
    fn safe_budget_never_exceeds_per_buffer_cap() {
        let safe = safe_budget_gib();
        if let Some(cap) = max_buffer_length_gib() {
            assert!(
                safe <= cap + 1e-9,
                "safe_budget_gib {safe} GiB exceeds maxBufferLength {cap} GiB — the sc-8135 clamp regressed"
            );
        }
    }

    /// On the macOS test lane MLX/Metal is available, so the `device_info` query must return a real,
    /// sane per-buffer cap. This guards the FFI plumbing + the `"max_buffer_length"` key against a
    /// silent `None` that would make the clamp a no-op (and the bug live).
    #[test]
    fn max_buffer_length_queried_on_metal() {
        let cap = max_buffer_length_gib().expect(
            "max_buffer_length_gib returned None on a Metal machine — FFI/key mismatch (sc-8135)",
        );
        // Apple-Silicon `maxBufferLength` ranges from a few GiB (small tiers) to a few hundred GiB.
        assert!(
            (1.0..=4096.0).contains(&cap),
            "implausible maxBufferLength {cap} GiB"
        );
    }
}
