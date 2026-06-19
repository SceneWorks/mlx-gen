//! Boogu-Image-0.1 end-to-end through the `gen_core` **registry + `Generator`** surface — the path
//! the SceneWorks worker uses: resolve a model by id → `Generator::generate` over a
//! `GenerationRequest` → an `Image`, exercising cooperative cancellation and `Progress` streaming.
//! The `model.rs` unit tests cover this wiring weightlessly; this proves it actually generates on
//! real weights.
//!
//! `#[ignore]` — needs the converted snapshots (128 GB Mac). Run:
//!   BOOGU_BASE_DIR=<base snapshot> [BOOGU_TURBO_DIR=<turbo>] [BOOGU_EDIT_DIR=<edit>] \
//!     CARGO_TARGET_DIR=~/Repos/mlx-gen/target \
//!     cargo test -p mlx-gen-boogu --test generator -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::gen_core::registry;
use mlx_gen::{
    CancelFlag, Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, Progress,
    WeightsSource,
};
use mlx_gen_boogu::{BOOGU_IMAGE_EDIT_ID, BOOGU_IMAGE_ID, BOOGU_IMAGE_TURBO_ID};

fn dir(var: &str) -> Option<PathBuf> {
    std::env::var(var).ok().map(PathBuf::from)
}

/// A small non-degenerate synthetic RGB reference for the edit path (a diagonal gradient). Real
/// edit coherence is validated in `pipeline_e2e::edit_smoke`; this only proves the registry edit
/// path runs and consumes the `Reference`.
fn gradient(res: u32) -> Image {
    let mut pixels = Vec::with_capacity((res * res * 3) as usize);
    for y in 0..res {
        for x in 0..res {
            pixels.push((x * 255 / res) as u8);
            pixels.push((y * 255 / res) as u8);
            pixels.push(128);
        }
    }
    Image {
        width: res,
        height: res,
        pixels,
    }
}

fn assert_image(out: GenerationOutput, res: u32, max_step: u32, steps: u32, decoding: bool) {
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
    assert!(decoding, "a Decoding progress event was emitted");
}

/// Collect a `Generator::generate` run into (max step seen, decoding seen).
fn run(g: &dyn mlx_gen::Generator, req: &GenerationRequest) -> (GenerationOutput, u32, bool) {
    let mut max_step = 0u32;
    let mut decoding = false;
    let out = g
        .generate(req, &mut |p| match p {
            Progress::Step { current, total } => {
                assert!(current >= 1 && current <= total, "{current}/{total}");
                max_step = max_step.max(current);
            }
            Progress::Decoding => decoding = true,
        })
        .expect("generate");
    (out, max_step, decoding)
}

#[test]
#[ignore = "needs real Base weights (128 GB Mac): set BOOGU_BASE_DIR"]
fn base_generates_via_registry() {
    let Some(root) = dir("BOOGU_BASE_DIR") else {
        eprintln!("skipping: set BOOGU_BASE_DIR");
        return;
    };
    let g = registry::load(BOOGU_IMAGE_ID, &LoadSpec::new(WeightsSource::Dir(root)))
        .expect("registry load boogu_image");
    assert_eq!(g.descriptor().id, "boogu_image");

    // Cooperative cancellation: a pre-cancelled request bails out (no full denoise).
    let cancelled = GenerationRequest {
        prompt: "a red apple on a wooden table".into(),
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

    let (res, steps) = (256u32, 8u32);
    let req = GenerationRequest {
        prompt: "a red apple on a wooden table".into(),
        width: res,
        height: res,
        steps: Some(steps),
        guidance: Some(4.0),
        seed: Some(0),
        ..Default::default()
    };
    let (out, max_step, decoding) = run(g.as_ref(), &req);
    assert_image(out, res, max_step, steps, decoding);
    println!("boogu_image: {steps} steps, decoding={decoding} — OK");
}

#[test]
#[ignore = "needs real Turbo weights (128 GB Mac): set BOOGU_TURBO_DIR"]
fn turbo_generates_via_registry() {
    let Some(root) = dir("BOOGU_TURBO_DIR") else {
        eprintln!("skipping: set BOOGU_TURBO_DIR");
        return;
    };
    let g = registry::load(
        BOOGU_IMAGE_TURBO_ID,
        &LoadSpec::new(WeightsSource::Dir(root)),
    )
    .expect("registry load boogu_image_turbo");
    assert_eq!(g.descriptor().id, "boogu_image_turbo");
    assert!(
        !g.descriptor().capabilities.supports_guidance,
        "turbo is CFG-free"
    );

    let (res, steps) = (256u32, 4u32);
    // CFG-free: no guidance (the floor rejects it on turbo).
    let req = GenerationRequest {
        prompt: "a red apple on a wooden table".into(),
        width: res,
        height: res,
        steps: Some(steps),
        seed: Some(0),
        ..Default::default()
    };
    let (out, max_step, decoding) = run(g.as_ref(), &req);
    assert_image(out, res, max_step, steps, decoding);
    println!("boogu_image_turbo: {steps} steps — OK");
}

#[test]
#[ignore = "needs real Edit weights (128 GB Mac): set BOOGU_EDIT_DIR"]
fn edit_generates_via_registry() {
    let Some(root) = dir("BOOGU_EDIT_DIR") else {
        eprintln!("skipping: set BOOGU_EDIT_DIR");
        return;
    };
    let g = registry::load(
        BOOGU_IMAGE_EDIT_ID,
        &LoadSpec::new(WeightsSource::Dir(root)),
    )
    .expect("registry load boogu_image_edit");
    assert_eq!(g.descriptor().id, "boogu_image_edit");

    // Edit with no reference must error (the source is required).
    let no_ref = GenerationRequest {
        prompt: "make the apple green".into(),
        width: 256,
        height: 256,
        ..Default::default()
    };
    assert!(
        g.generate(&no_ref, &mut |_| {}).is_err(),
        "edit without a reference must error"
    );

    let (res, steps) = (256u32, 8u32);
    let req = GenerationRequest {
        prompt: "make the apple green".into(),
        width: res,
        height: res,
        steps: Some(steps),
        guidance: Some(4.0),
        seed: Some(0),
        conditioning: vec![Conditioning::Reference {
            image: gradient(res),
            strength: None,
        }],
        ..Default::default()
    };
    let (out, max_step, decoding) = run(g.as_ref(), &req);
    assert_image(out, res, max_step, steps, decoding);
    println!("boogu_image_edit: {steps} steps — OK");
}
