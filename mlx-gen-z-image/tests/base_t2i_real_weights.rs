//! sc-8320: maintainer's on-device gate for the **base** (non-Turbo, full-CFG) `z_image` engine.
//!
//! `#[ignore]`d — needs the real `Tongyi-MAI/Z-Image` snapshot (the 19 GB base weights, distinct from
//! the `Z-Image-Turbo` snapshot the other real-weight tests use). Run with:
//!   cargo test -p mlx-gen-z-image --release --test base_t2i_real_weights -- --ignored --nocapture
//!
//! Unlike the Turbo golden tests (which compare against a fork dump), the base has no fork golden, so
//! this is a **coherence smoke**: drive the public `load("z_image", spec).generate(req)` API at the
//! base recipe (50 steps, shift 6.0, CFG guidance 4.0 + a negative prompt) and assert it returns a
//! correctly sized, non-degenerate image (not flat / not pure noise). The maintainer eyeballs the saved
//! PNG for quality. A second pass with `guidance = 1.0` confirms the CFG-off (single-forward) path.

use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};
use mlx_gen_z_image as _;
use std::path::PathBuf;

/// Resolve the **base** `Tongyi-MAI/Z-Image` snapshot: the `BASE_ZIMAGE_SNAPSHOT` override if set, else
/// the first snapshot under the HF hub cache. `None` when neither is present (skip rather than fail).
fn base_snapshot() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BASE_ZIMAGE_SNAPSHOT") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--Tongyi-MAI--Z-Image/snapshots");
    std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
}

/// Render `req` through the public base generator and assert the image is the requested size and is
/// non-degenerate (some spread in the pixel values — neither a flat fill nor pure noise).
fn render_and_check(spec: &LoadSpec, req: &GenerationRequest, tag: &str) {
    let generator = mlx_gen::load("z_image", spec).expect("base z_image loads from the snapshot");
    let out = generator
        .generate(req, &mut |_| {})
        .expect("base generate succeeds");
    let img = match out {
        GenerationOutput::Images(mut v) => {
            assert_eq!(v.len(), 1, "count=1 -> one image");
            v.pop().unwrap()
        }
        other => panic!("expected Images, got {other:?}"),
    };
    assert_eq!(
        (img.width, img.height),
        (req.width, req.height),
        "image size"
    );

    // Coherence: a real render has structure — its luma spans a meaningful range. A collapsed
    // (all-black / all-grey) or saturated render would have near-zero spread.
    let min = *img.pixels.iter().min().unwrap();
    let max = *img.pixels.iter().max().unwrap();
    let mean = img.pixels.iter().map(|&p| p as u64).sum::<u64>() as f64 / img.pixels.len() as f64;
    println!(
        "✓ base z_image [{tag}]: {}x{} render; px min={min} max={max} mean={mean:.1}",
        img.width, img.height
    );
    assert!(
        max as i32 - min as i32 > 32,
        "[{tag}] degenerate render: pixel range {min}..={max} is too flat to be a coherent image"
    );

    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../tools/golden/base_z_image_{tag}.png"));
    let _ = image::save_buffer(
        &out_path,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    );
    println!("  saved {}", out_path.display());
}

#[test]
#[ignore = "needs the real Tongyi-MAI/Z-Image base snapshot (set BASE_ZIMAGE_SNAPSHOT or populate the HF cache)"]
fn base_t2i_cfg_renders_coherent_image() {
    let Some(snap) = base_snapshot() else {
        eprintln!(
            "skip base_t2i_cfg_renders_coherent_image: no Tongyi-MAI/Z-Image snapshot \
             (set BASE_ZIMAGE_SNAPSHOT or populate the HF hub cache)"
        );
        return;
    };
    let spec = LoadSpec::new(WeightsSource::Dir(snap));

    // Full base recipe: undistilled, CFG on (guidance 4.0 + a negative prompt). Steps default to 50
    // when unset, but pin a smaller count here so the smoke is tractable while still exercising CFG.
    let req = GenerationRequest {
        prompt: "a red fox sitting in a snowy forest, photorealistic, sharp focus".into(),
        negative_prompt: Some("blurry, low quality, distorted".into()),
        guidance: Some(4.0),
        width: 1024,
        height: 1024,
        steps: Some(28),
        seed: Some(42),
        ..Default::default()
    };
    render_and_check(&spec, &req, "cfg");
}

#[test]
#[ignore = "needs the real Tongyi-MAI/Z-Image base snapshot (set BASE_ZIMAGE_SNAPSHOT or populate the HF cache)"]
fn base_t2i_cfg_no_negative_renders_coherent_image() {
    // Regression (sc-8958, MLX twin of candle sc-8646): CFG on (guidance 4.0) with an **unset**
    // negative prompt. The uncond branch
    // encodes the empty string; before the fix this errored `z_image: negative conditioning tokenized
    // to an empty sequence` because gen-core's tokenizer short-circuits an empty prompt before the chat
    // template is applied. Must render coherently now (the empty uncond goes through the template).
    let Some(snap) = base_snapshot() else {
        eprintln!(
            "skip base_t2i_cfg_no_negative_renders_coherent_image: no Tongyi-MAI/Z-Image snapshot"
        );
        return;
    };
    let spec = LoadSpec::new(WeightsSource::Dir(snap));

    let req = GenerationRequest {
        prompt: "a red fox sitting in a snowy forest, photorealistic, sharp focus".into(),
        negative_prompt: None,
        guidance: Some(4.0),
        width: 1024,
        height: 1024,
        steps: Some(28),
        seed: Some(42),
        ..Default::default()
    };
    render_and_check(&spec, &req, "cfg_no_negative");
}

#[test]
#[ignore = "needs the real Tongyi-MAI/Z-Image base snapshot (set BASE_ZIMAGE_SNAPSHOT or populate the HF cache)"]
fn base_t2i_cfg_off_renders_coherent_image() {
    let Some(snap) = base_snapshot() else {
        eprintln!("skip base_t2i_cfg_off_renders_coherent_image: no Tongyi-MAI/Z-Image snapshot");
        return;
    };
    let spec = LoadSpec::new(WeightsSource::Dir(snap));

    // guidance = 1.0 turns CFG off → a single cond forward per step (Turbo-equivalent compute path).
    let req = GenerationRequest {
        prompt: "a red fox sitting in a snowy forest, photorealistic, sharp focus".into(),
        guidance: Some(1.0),
        width: 1024,
        height: 1024,
        steps: Some(28),
        seed: Some(42),
        ..Default::default()
    };
    render_and_check(&spec, &req, "cfg_off");
}
