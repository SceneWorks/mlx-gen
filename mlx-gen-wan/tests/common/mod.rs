//! Shared test helpers (F-118). The elementwise `max_abs(&[f32], &[f32])` reducer was copy-pasted
//! byte-for-byte into several of this crate's parity tests; it lives here now. Byte-identical to the
//! per-file copies (the plain no-assert fold form). The length-asserting / `Array`-typed variants in
//! `wanvace_cond_parity` / `s0_parity` / `compile_micro` are deliberately different (they assert or
//! take on-device tensors) and keep their own local definitions.
//!
//! `tests/common/mod.rs` is compiled once into each integration-test binary that declares `mod
//! common;`; a binary may use only a subset, so `#![allow(dead_code)]` suppresses the otherwise
//! unavoidable dead-code warnings under `-D warnings`.
#![allow(dead_code)]

/// Max absolute elementwise difference between two equal-length host slices.
pub fn max_abs(got: &[f32], exp: &[f32]) -> f32 {
    got.iter()
        .zip(exp.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0f32, f32::max)
}
