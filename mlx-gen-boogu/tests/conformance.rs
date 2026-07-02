//! Real-weight gen-core **contract conformance** for `boogu_image_turbo` (epic 3720, sc-4481
//! standard; wired here by sc-9098/F-102 — the crate declared the testkit dev-dep without ever
//! invoking it).
//!
//! Drives the assembled MLX engine (Qwen3-VL text tower + single-stream DiT + FLUX.1 16-ch VAE)
//! through the backend-neutral checks — capability honesty, progress monotonicity, typed
//! cancellation, seed determinism, registry round-trip — the same guarantees `z_image_turbo` and
//! `krea_2_turbo` are held to. Turbo is the cheapest checkpoint (few-step DMD student, CFG-free).
//! `#[ignore]` because it needs the converted Boogu turnkey snapshot; run on a populated dev box:
//! ```sh
//! BOOGU_TURBO_DIR=<converted turbo snapshot> \
//!   cargo test -p mlx-gen-boogu --release --test conformance -- --ignored --nocapture
//! ```
//!
//! The weights-free descriptor-level sweep for all three Boogu ids runs by default in
//! `tests/descriptor_conformance.rs`.

use std::path::PathBuf;

// Force-link the provider so its `inventory::submit!` registrations survive the linker (this test
// references no other boogu symbol); the worker does the same `as _` import per model crate.
use mlx_gen_boogu as _;

use gen_core_testkit::Profile;
use mlx_gen::{LoadSpec, WeightsSource};

/// The converted Boogu Turbo snapshot: `BOOGU_TURBO_DIR` (the crate's real-weight test convention,
/// see `tests/generator.rs`). Panics with a clear message when absent.
fn snapshot() -> PathBuf {
    std::env::var("BOOGU_TURBO_DIR")
        .map(PathBuf::from)
        .expect("set BOOGU_TURBO_DIR to the converted Boogu Turbo snapshot")
}

#[test]
#[ignore = "needs the converted Boogu Turbo snapshot (BOOGU_TURBO_DIR); macos-mlx / dev box only"]
fn boogu_image_turbo_satisfies_gen_core_contract() {
    let snap = snapshot();
    gen_core_testkit::conformance(
        || {
            // The turnkey's own packing (Q8-pre-packed or bf16) — the suite exercises the contract,
            // not quantization.
            let spec = LoadSpec::new(WeightsSource::Dir(snap.clone()));
            mlx_gen::load("boogu_image_turbo", &spec).expect("load boogu_image_turbo")
        },
        // 256² / 2 steps — the minimum valid Boogu config (min_size 256, multiple-of-16); the Turbo
        // DMD loop resolves `req.steps` verbatim, so Step.total == 2.
        &Profile::cheap(),
    );
}
