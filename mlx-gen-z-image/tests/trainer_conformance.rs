//! Real-weight gen-core **Trainer contract** conformance for `z_image_turbo` (epic 3720, sc-4895).
//!
//! The trainer half of the "one real provider per contract" AC: it drives the actual
//! `ZImageTurboTrainer` through the backend-neutral checks (capability honesty, `TrainingProgress`
//! monotonicity, typed cancellation before any step, registry round-trip) — the guarantees a candle
//! trainer will be held to identically. `#[ignore]` because it needs the real
//! `Tongyi-MAI/Z-Image-Turbo` weights (`ZIMAGE_SNAPSHOT` or the HF cache); run on the self-hosted
//! Apple-Silicon runner or a populated dev box:
//!   cargo test -p mlx-gen-z-image --release --test trainer_conformance -- --ignored --nocapture
//!
//! `trainer_conformance` constructs a fresh trainer per train()-invoking check (the cancellation
//! paths + the progress run), because `train` is `&mut self` and the trainer is effectively
//! single-use once it casts/mutates its base model. The cheap profile keeps each cheap: a 2-item
//! 64px dataset, 2 steps.

use std::path::Path;

// Force-link the provider so its `inventory::submit!` trainer registration survives (this test
// references `MODEL_ID`, but keep the `as _` discipline explicit for the registry round-trip).
use mlx_gen_z_image as _;

use gen_core_testkit::TrainerProfile;
use mlx_gen::{LoadSpec, TrainingItem, WeightsSource};

mod common;
use common::snapshot;

/// Two solid-colour swatch PNGs + captions in `dir` (mirrors the trainer e2e dataset).
fn make_dataset(dir: &Path) -> Vec<TrainingItem> {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let mut items = Vec::new();
    for (i, color) in [[200u8, 40, 40], [40, 80, 200]].iter().enumerate() {
        let mut img = image::RgbImage::new(96, 96);
        for px in img.pixels_mut() {
            *px = image::Rgb(*color);
        }
        let path = dir.join(format!("img{i}.png"));
        img.save(&path).unwrap();
        items.push(TrainingItem {
            image_path: path,
            caption: format!("a solid colour swatch number {i}"),
        });
    }
    items
}

#[test]
#[ignore = "needs real Z-Image-Turbo weights (ZIMAGE_SNAPSHOT or HF cache); macos-mlx / dev box only"]
fn z_image_turbo_trainer_satisfies_gen_core_contract() {
    assert_eq!(mlx_gen_z_image::MODEL_ID, "z_image_turbo");
    let tmp = std::env::temp_dir().join("z_image_trainer_conformance");
    let items = make_dataset(&tmp.join("data"));
    let profile = TrainerProfile::cheap(items, tmp.join("out"));
    let snap = snapshot();

    gen_core_testkit::trainer_conformance(
        || {
            let spec = LoadSpec::new(WeightsSource::Dir(snap.clone()));
            mlx_gen::load_trainer("z_image_turbo", &spec).expect("load z_image_turbo trainer")
        },
        &profile,
    );
}
