//! sc-8251: maintainer's on-device gate for the **base** (non-Turbo, full-CFG) `z_image_control`
//! Fun-Controlnet-Union engine.
//!
//! `#[ignore]`d — needs the real `Tongyi-MAI/Z-Image` base snapshot (the 19 GB base weights) **and**
//! the base control checkpoint `alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1` in the HF cache. Run with:
//!   cargo test -p mlx-gen-z-image --release --test base_control_real_weights -- --ignored --nocapture
//!
//! Unlike the Turbo control golden tests (which compare against a fork dump), the base control variant
//! has no fork golden, so this is a **steering smoke**: drive the public
//! `load("z_image_control", spec).generate(req)` API at the base recipe (shift 6.0, CFG guidance 4.0 +
//! a negative prompt) with a `Conditioning::Control`, and assert (a) the render is correctly sized +
//! non-degenerate and (b) the **control image actually steers** the output — two renders that differ
//! ONLY in the control image (same seed/prompt/dims) produce meaningfully different pixels, while a
//! repeat with the identical control image is deterministic. Runs per control kind (pose, canny, depth)
//! since the Fun-Union family is input-agnostic (the kind only labels the preprocessor's output). The
//! maintainer eyeballs the saved PNGs for quality.

use mlx_gen::{
    Conditioning, ControlKind, GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource,
};
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

/// Resolve the **base** Fun-Controlnet-Union checkpoint: the `BASE_CONTROL_WEIGHTS` override if set,
/// else the `Z-Image-Fun-Controlnet-Union-2.1.safetensors` (the full Union ckpt, not the `-lite`) under
/// the `alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1` HF cache. `None` when absent (skip).
fn base_control_source() -> Option<WeightsSource> {
    if let Ok(p) = std::env::var("BASE_CONTROL_WEIGHTS") {
        return Some(WeightsSource::File(PathBuf::from(p)));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home).join(
        ".cache/huggingface/hub/models--alibaba-pai--Z-Image-Fun-Controlnet-Union-2.1/snapshots",
    );
    let file = std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .flat_map(|d| {
            std::fs::read_dir(d)
                .into_iter()
                .flatten()
                .filter_map(|e| e.ok())
                .map(|e| e.path())
        })
        // Prefer the full Union ckpt; explicitly skip the `-lite` and the Tile variants.
        .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.contains("Union") && !n.contains("lite"))
                .unwrap_or(false)
        })?;
    Some(WeightsSource::File(file))
}

/// A synthetic high-contrast structural control image: an off-center filled rectangle on a black field.
/// `shift` slides the rectangle so two calls produce **different** structure (to prove steering). Not a
/// realistic pose/canny/depth map, but a valid 33-ch-VAE-encodable control signal with clear spatial
/// structure the control branch can latch onto.
fn synthetic_control(width: u32, height: u32, shift: u32) -> Image {
    let (w, h) = (width as usize, height as usize);
    let mut pixels = vec![0u8; w * h * 3];
    let x0 = (w / 4 + shift as usize).min(w.saturating_sub(2));
    let x1 = (3 * w / 4 + shift as usize).min(w);
    let y0 = h / 4;
    let y1 = 3 * h / 4;
    for y in y0..y1 {
        for x in x0..x1 {
            let i = (y * w + x) * 3;
            pixels[i] = 255;
            pixels[i + 1] = 255;
            pixels[i + 2] = 255;
        }
    }
    Image {
        width,
        height,
        pixels,
    }
}

/// Render `req` through the public base control generator → one image, asserting size + non-degeneracy.
fn render(spec: &LoadSpec, req: &GenerationRequest) -> Image {
    let generator =
        mlx_gen::load("z_image_control", spec).expect("z_image_control loads from base + control");
    let out = generator
        .generate(req, &mut |_| {})
        .expect("base control generate succeeds");
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
    let min = *img.pixels.iter().min().unwrap();
    let max = *img.pixels.iter().max().unwrap();
    assert!(
        max as i32 - min as i32 > 32,
        "degenerate render: pixel range {min}..={max} is too flat to be a coherent image"
    );
    img
}

/// Mean absolute per-pixel difference between two equally sized RGB8 images, in `[0, 255]`.
fn mean_abs_diff(a: &Image, b: &Image) -> f64 {
    assert_eq!(a.pixels.len(), b.pixels.len(), "comparable images");
    let sum: u64 = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
        .sum();
    sum as f64 / a.pixels.len() as f64
}

fn save(img: &Image, tag: &str) {
    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../tools/golden/base_z_image_control_{tag}.png"));
    let _ = image::save_buffer(
        &out_path,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    );
    println!("  saved {}", out_path.display());
}

/// Drive one control `kind`: render with control image A, render again with A (determinism), and render
/// with a shifted control image B. A must equal A (same inputs ⇒ same pixels) and B must differ from A
/// by a meaningful margin (the control image steers the output).
fn steering_run(spec: &LoadSpec, kind: ControlKind, tag: &str) {
    let (w, h) = (1024u32, 1024u32);
    let base = || GenerationRequest {
        prompt: "a single object on a plain background, studio lighting, sharp focus".into(),
        negative_prompt: Some("blurry, low quality, distorted".into()),
        guidance: Some(4.0),
        width: w,
        height: h,
        steps: Some(28),
        seed: Some(7),
        ..Default::default()
    };
    let ctrl = |shift: u32, scale: f32| Conditioning::Control {
        image: synthetic_control(w, h, shift),
        kind: kind.clone(),
        scale,
    };

    let mut req_a = base();
    req_a.conditioning = vec![ctrl(0, 0.8)];
    let img_a = render(spec, &req_a);
    save(&img_a, &format!("{tag}_a"));

    // Determinism: identical inputs ⇒ identical pixels.
    let img_a2 = render(spec, &req_a);
    let det = mean_abs_diff(&img_a, &img_a2);
    println!("✓ base z_image_control [{tag}]: determinism mean|Δ|={det:.3}");
    assert!(
        det < 0.5,
        "[{tag}] non-deterministic render: mean|Δ|={det:.3}"
    );

    // Steering: a different control image (shifted structure) ⇒ a meaningfully different render.
    let mut req_b = base();
    req_b.conditioning = vec![ctrl(w / 4, 0.8)];
    let img_b = render(spec, &req_b);
    save(&img_b, &format!("{tag}_b"));
    let steer = mean_abs_diff(&img_a, &img_b);
    println!("✓ base z_image_control [{tag}]: steering mean|Δ|={steer:.3} (A vs shifted B)");
    assert!(
        steer > 2.0,
        "[{tag}] control image did not steer the render: mean|Δ|={steer:.3} between two distinct \
         control images is too small — the control branch appears inert"
    );
}

/// Skip-or-resolve the `(base snapshot, base control)` pair; returns the assembled `LoadSpec`.
fn spec_or_skip(test: &str) -> Option<LoadSpec> {
    let Some(snap) = base_snapshot() else {
        eprintln!("skip {test}: no Tongyi-MAI/Z-Image base snapshot (set BASE_ZIMAGE_SNAPSHOT)");
        return None;
    };
    let Some(control) = base_control_source() else {
        eprintln!(
            "skip {test}: no Z-Image-Fun-Controlnet-Union-2.1 checkpoint (set BASE_CONTROL_WEIGHTS)"
        );
        return None;
    };
    Some(LoadSpec::new(WeightsSource::Dir(snap)).with_control(control))
}

#[test]
#[ignore = "needs the real Tongyi-MAI/Z-Image base snapshot + Z-Image-Fun-Controlnet-Union-2.1 control ckpt"]
fn base_control_pose_steers_render() {
    let Some(spec) = spec_or_skip("base_control_pose_steers_render") else {
        return;
    };
    steering_run(&spec, ControlKind::Pose, "pose");
}

#[test]
#[ignore = "needs the real Tongyi-MAI/Z-Image base snapshot + Z-Image-Fun-Controlnet-Union-2.1 control ckpt"]
fn base_control_canny_steers_render() {
    let Some(spec) = spec_or_skip("base_control_canny_steers_render") else {
        return;
    };
    steering_run(&spec, ControlKind::Canny, "canny");
}

#[test]
#[ignore = "needs the real Tongyi-MAI/Z-Image base snapshot + Z-Image-Fun-Controlnet-Union-2.1 control ckpt"]
fn base_control_depth_steers_render() {
    let Some(spec) = spec_or_skip("base_control_depth_steers_render") else {
        return;
    };
    steering_run(&spec, ControlKind::Depth, "depth");
}
