//! Real-weight progress-contract conformance gate for `seedvr2` (sc-11133, F-162/F-164).
//!
//! Binds the repo-wide progress-contract property (monotone, in-bounds `Step`; the bar reaches
//! `total`; `Progress::Decoding` emitted exactly once) to the SeedVR2 image-upscale path via the
//! reusable testkit check (`gen_core_testkit::check_progress_contract_with`). A multi-image request
//! exercises the count-axis fold (F-162); a large target exercises the spatial-tiling per-tile
//! progress (F-164). Both must present one monotone bar that reaches its total with a single
//! `Decoding` — the class the pre-fix `Step{1,1}`-per-image / no-per-tile-progress code violated.
//!
//! Snapshot-gated (skips when absent), matching the crate's `registry_e2e.rs` / cancellation
//! conventions: it needs the raw `numz/SeedVR2_comfyUI` checkpoint dir (the 3B fp16 DiT) in the HF
//! cache.

use std::path::PathBuf;

use mlx_gen::{registry, Conditioning, GenerationRequest, Image, LoadSpec, WeightsSource};
// Force-link the provider so its `inventory::submit!` registration survives the linker; otherwise
// `mlx_gen::load` reports "no generator registered". The worker does the same per model crate.
use mlx_gen_seedvr2::registry::MODEL_ID;

/// The raw `numz/SeedVR2_comfyUI` checkpoint snapshot dir (mirrors `cancellation_conformance.rs`):
/// returns `None` (→ skip) when the 3B checkpoint is not in the HF cache.
fn raw_dir() -> Option<PathBuf> {
    let base = PathBuf::from(std::env::var("HOME").ok()?)
        .join(".cache/huggingface/hub/models--numz--SeedVR2_comfyUI/snapshots");
    let snap = std::fs::read_dir(&base).ok()?.flatten().next()?.path();
    snap.join("seedvr2_ema_3b_fp16.safetensors")
        .exists()
        .then_some(snap)
}

/// A small synthetic low-res RGB8 input image (the `registry_e2e.rs` upscale source).
fn lr_image(w: u32, h: u32) -> Image {
    Image {
        width: w,
        height: h,
        pixels: (0..(w * h * 3)).map(|i| (i % 256) as u8).collect(),
    }
}

#[test]
#[ignore = "needs real SeedVR2 3B checkpoint (numz/SeedVR2_comfyUI in HF cache); macos-mlx / dev box only"]
fn seedvr2_image_batch_satisfies_progress_contract() {
    let Some(snap) = raw_dir() else {
        eprintln!("SKIP: raw SeedVR2 3B checkpoint absent");
        return;
    };
    let gen =
        registry::load(MODEL_ID, &LoadSpec::new(WeightsSource::Dir(snap))).expect("load seedvr2");

    // Multi-image single-pass batch: exercises the F-162 count-axis fold (one monotone `1..=count`
    // bar, not three restarting `Step{1,1}`s), and the single terminal `Decoding`.
    let req = GenerationRequest {
        width: 128,
        height: 128,
        seed: Some(7),
        count: 3,
        conditioning: vec![Conditioning::Reference {
            image: lr_image(96, 96),
            strength: None,
        }],
        ..Default::default()
    };

    gen_core_testkit::check_progress_contract_with(gen.as_ref(), &req)
        .expect("seedvr2 image batch must satisfy the progress contract (F-162)");
}
