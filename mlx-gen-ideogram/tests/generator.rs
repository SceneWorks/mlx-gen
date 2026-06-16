//! sc-5988 — Ideogram 4 end-to-end through the `gen_core` **registry + `Generator`** surface
//! (the path the SceneWorks worker uses): resolve `"ideogram_4"` by id → `Generator::generate`
//! over a `GenerationRequest` → an `Image`, exercising cooperative cancellation and `Progress`
//! streaming. The `model.rs` unit tests cover this wiring weightlessly; this proves it actually
//! generates on real weights.
//!
//! `#[ignore]` — needs the converted snapshot (~53 GB). Run:
//!   IDEOGRAM4_MLX=~/.cache/ideogram4-mlx-convert \
//!     cargo test -p mlx-gen-ideogram --test generator -- --ignored --nocapture

mod common;

use std::path::PathBuf;

use common::CAPTION_JSON;
use mlx_gen::gen_core::registry;
use mlx_gen::{CancelFlag, GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource};
use mlx_gen_ideogram::MODEL_ID;

fn snapshot_dir() -> PathBuf {
    std::env::var("IDEOGRAM4_MLX")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME")).join(".cache/ideogram4-mlx-convert")
        })
}

#[test]
#[ignore = "needs converted weights (~53 GB)"]
fn generates_via_registry() {
    // Resolve by id through the registry (proves link-time self-registration on real weights).
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot_dir()));
    let g = registry::load(MODEL_ID, &spec).expect("registry load ideogram_4");
    assert_eq!(g.descriptor().id, "ideogram_4");

    // Cooperative cancellation: a pre-cancelled request bails out (no full denoise).
    let cancelled = GenerationRequest {
        prompt: CAPTION_JSON.into(),
        width: 256,
        height: 256,
        cancel: {
            let c = CancelFlag::new();
            c.cancel();
            c
        },
        ..Default::default()
    };
    let err = g
        .generate(&cancelled, &mut |_| {})
        .expect_err("a cancelled request must error");
    assert!(
        err.to_string().to_lowercase().contains("cancel"),
        "expected a cancellation error, got: {err}"
    );

    // Full generate via the Generator surface, collecting Progress.
    let envn = |k: &str, d: u32| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let res = envn("IDEOGRAM4_SMOKE_RES", 256);
    let steps = envn("IDEOGRAM4_SMOKE_STEPS", 50);
    let req = GenerationRequest {
        prompt: CAPTION_JSON.into(),
        width: res,
        height: res,
        steps: Some(steps),
        guidance: Some(7.0),
        seed: Some(0),
        ..Default::default()
    };

    let mut max_step = 0u32;
    let mut decoding_seen = false;
    let out = g
        .generate(&req, &mut |p| match p {
            Progress::Step { current, total } => {
                assert!(current >= 1 && current <= total, "{current}/{total}");
                max_step = max_step.max(current);
            }
            Progress::Decoding => decoding_seen = true,
        })
        .expect("generate");

    let imgs = match out {
        GenerationOutput::Images(v) => v,
        other => panic!("expected Images, got {other:?}"),
    };
    assert_eq!(imgs.len(), 1, "count defaults to 1");
    let im = &imgs[0];
    assert_eq!((im.width, im.height), (res, res), "image dims");
    assert_eq!(
        im.pixels.len(),
        (res * res * 3) as usize,
        "RGB8 pixel count"
    );
    let (mn, mx) = (
        *im.pixels.iter().min().unwrap(),
        *im.pixels.iter().max().unwrap(),
    );
    assert!(mx > mn, "degenerate (constant) image");
    assert_eq!(max_step, steps, "progress reached the final step");
    assert!(decoding_seen, "a Decoding progress event was emitted");

    let out_path = std::env::temp_dir().join("ideogram4_generator.png");
    image::RgbImage::from_raw(res, res, im.pixels.clone())
        .unwrap()
        .save(&out_path)
        .unwrap();
    println!(
        "wrote {} — {steps} steps (max progress {max_step}), decoding={decoding_seen}",
        out_path.display()
    );
}
