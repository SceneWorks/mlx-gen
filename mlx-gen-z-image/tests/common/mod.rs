//! Shared helpers for the Z-Image integration tests (F-045): the HF-snapshot discovery and the
//! relative-error metric, previously copy-pasted across many test files (and already drifting — the
//! `perf` bench wanted an `Option`-returning variant while the rest panicked). Included per test
//! binary via `mod common;`. `#![allow(dead_code)]` because no single test uses every helper.
#![allow(dead_code)]

use std::path::PathBuf;

use mlx_rs::{Array, Dtype};

/// The `Tongyi-MAI/Z-Image-Turbo` snapshot directory: the `ZIMAGE_SNAPSHOT` override if set, else the
/// first snapshot under the HF hub cache. Returns `None` if neither is present — for tests/benches
/// that skip rather than panic when real weights are absent.
pub fn snapshot_opt() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ZIMAGE_SNAPSHOT") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots");
    std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
}

/// [`snapshot_opt`] but panicking with a clear message when no snapshot is found — for the
/// `#[ignore]`d weight-gated tests that need real weights to run at all.
pub fn snapshot() -> PathBuf {
    snapshot_opt()
        .expect("a Z-Image-Turbo snapshot (set ZIMAGE_SNAPSHOT or populate the HF hub cache)")
}

/// `(max|a-b| / peak|b|, mean|a-b| / mean|b|)` over the full tensors (cast to f32, flattened) — the
/// peak- and mean-relative error used by the parity gates.
pub fn rel(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let (xs, ys) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = ys.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let mabs = (ys.iter().map(|y| y.abs()).sum::<f32>() / ys.len() as f32).max(1e-12);
    let max_diff = xs
        .iter()
        .zip(ys)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let mean_diff = xs.iter().zip(ys).map(|(x, y)| (x - y).abs()).sum::<f32>() / xs.len() as f32;
    (max_diff / peak, mean_diff / mabs)
}
