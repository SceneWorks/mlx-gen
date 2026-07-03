//! Shared test helpers (F-118). The `snapshot()` / `snapshot_opt()` resolvers were copy-pasted
//! byte-for-byte into ~20 of this crate's `#[ignore]`d real-weight test files; they live here now,
//! keyed off the single canonical `SDXL_SNAPSHOT` env var (falling back to the standard HF cache
//! `stabilityai/stable-diffusion-xl-base-1.0` snapshots dir). Behavior is byte-identical to the
//! per-file copies — the panicking `snapshot()` and the `Option`-returning `snapshot_opt()` are the
//! two shapes that existed inline.
//!
//! `tests/common/mod.rs` is compiled once into each integration-test binary that declares `mod
//! common;`; every binary uses only a subset, so `#![allow(dead_code)]` suppresses the otherwise
//! unavoidable dead-code warnings under `-D warnings`.
#![allow(dead_code)]

use std::path::PathBuf;

/// The `SDXL_SNAPSHOT` override, else the newest HF-cache snapshot dir; **panics** if neither exists.
/// The form used by the parity/real-weight tests that unconditionally need weights.
pub fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("SDXL_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

/// The `SDXL_SNAPSHOT` override, else the newest HF-cache snapshot dir, or `None` if unavailable —
/// the graceful `skip:` form used by the smoke/perf tests that self-skip when weights are absent.
pub fn snapshot_opt() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SDXL_SNAPSHOT") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots");
    std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
}
