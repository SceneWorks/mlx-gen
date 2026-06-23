//! Real-weight gen-core **contract conformance** for `krea_2_turbo` (epic 3720, sc-4481 standard).
//!
//! Drives the assembled MLX engine (tokenizer + Qwen3-VL-4B TE + single-stream DiT + Qwen-Image VAE)
//! through the backend-neutral checks — capability honesty, progress monotonicity, typed cancellation
//! ([`mlx_gen::Error::Canceled`] round-tripping to `gen_core::Error::Canceled`), seed determinism,
//! registry round-trip — the same guarantees a candle provider will be held to (sc-7580). `#[ignore]`
//! because it needs the real `krea/Krea-2-Turbo` weights; run on the macos-mlx lane / a dev box:
//! ```sh
//! KREA_TURBO_DIR=~/.cache/huggingface/hub/models--krea--Krea-2-Turbo/snapshots/<rev> \
//!   cargo test -p mlx-gen-krea --release --test conformance -- --ignored --nocapture
//! ```

use std::path::PathBuf;

// Force-link the provider so its `inventory::submit!` registration survives the linker (this test
// references no other krea symbol); the worker does the same `as _` import per model crate.
use mlx_gen_krea as _;

use gen_core_testkit::Profile;
use mlx_gen::{LoadSpec, WeightsSource};

/// The `krea/Krea-2-Turbo` snapshot: `KREA_TURBO_DIR` if set, else the first snapshot under the HF hub
/// cache. Panics with a clear message when absent (the `#[ignore]` gate needs real weights to run).
fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("KREA_TURBO_DIR") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").expect("HOME");
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--krea--Krea-2-Turbo/snapshots");
    std::fs::read_dir(&snaps)
        .ok()
        .and_then(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| p.is_dir())
        })
        .expect("set KREA_TURBO_DIR or populate the HF hub cache for krea/Krea-2-Turbo")
}

#[test]
#[ignore = "needs real Krea 2 Turbo weights (KREA_TURBO_DIR or HF cache); macos-mlx / dev box only"]
fn krea_2_turbo_satisfies_gen_core_contract() {
    let snap = snapshot();
    gen_core_testkit::conformance(
        || {
            // Dense bf16 — the cheapest load; the suite exercises the contract, not quantization.
            let spec = LoadSpec::new(WeightsSource::Dir(snap.clone()));
            mlx_gen::load("krea_2_turbo", &spec).expect("load krea_2_turbo")
        },
        // 256² / few-step — the minimum valid Krea config (min_size 256, multiple-of-16).
        &Profile::cheap(),
    );
}
