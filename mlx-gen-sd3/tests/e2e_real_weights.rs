//! SD3.5-Large end-to-end real-weight smoke (E5, sc-7864) — the de-facto correctness gate for the
//! whole native port. E3's MMDiT forward could not be numerically A/B'd vs diffusers (no torch env),
//! so a COHERENT real-weight 1024² image out of this pipeline is the proof the forward + sampler +
//! CFG + VAE de-norm are all correct.
//!
//! `#[ignore]`d — needs the real `stabilityai/stable-diffusion-3.5-large` snapshot in the HF cache
//! (or `SD3_LARGE_SNAPSHOT`) and Metal. Run with:
//!   cargo test -p mlx-gen-sd3 --release --test e2e_real_weights -- --ignored --nocapture
//!
//! The smoke drives the PUBLIC registry path (`mlx_gen::load("sd3_5_large", spec).generate(req)`),
//! saves the PNG, and reports sanity signals (the coordinator can't view the image): per-channel
//! mean/std (NOT constant, NOT pure-noise), a histogram spread, and a crude spatial-coherence signal
//! (mean absolute neighbor gradient — low-but-nonzero = structured content, not white noise).

use std::path::PathBuf;

use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, Progress, Quant, WeightsSource};

// Force the linker to keep `mlx-gen-sd3`'s `inventory::submit!` registration static (it is otherwise
// dropped as unreferenced, since this test reaches the generator only through the `mlx_gen::load`
// registry — the CLAUDE.md "Linkage gotcha"). Asserting the id keeps the import honest.
use mlx_gen_sd3 as sd3;

/// Resolve the SD3.5-Large snapshot dir: `SD3_LARGE_SNAPSHOT` override, else the first snapshot in the
/// HF hub cache.
fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("SD3_LARGE_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").expect("HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--stabilityai--stable-diffusion-3.5-large/snapshots");
    std::fs::read_dir(&snaps)
        .unwrap_or_else(|_| panic!("no SD3.5-Large snapshots under {snaps:?}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("set SD3_LARGE_SNAPSHOT or populate the HF hub cache")
}

/// Per-channel mean + std over an RGB8 image — flat/constant output collapses std≈0; pure noise has a
/// near-uniform histogram (std≈74 for U[0,255]) with no spatial structure.
fn channel_stats(pixels: &[u8], w: u32, h: u32) -> [(f32, f32); 3] {
    let n = (w * h) as usize;
    let mut out = [(0.0f32, 0.0f32); 3];
    for (c, stat) in out.iter_mut().enumerate() {
        let mut sum = 0.0f64;
        for i in 0..n {
            sum += pixels[i * 3 + c] as f64;
        }
        let mean = (sum / n as f64) as f32;
        let mut var = 0.0f64;
        for i in 0..n {
            let d = pixels[i * 3 + c] as f64 - mean as f64;
            var += d * d;
        }
        let std = (var / n as f64).sqrt() as f32;
        *stat = (mean, std);
    }
    out
}

/// Mean absolute horizontal+vertical neighbor gradient on the luma plane — a crude spatial-coherence
/// signal. White noise has a HIGH gradient (~85/255 for U[0,255]); a flat image ≈0; a coherent photo
/// is LOW-but-nonzero (smooth regions + edges), typically ~5–25. So a coherent render lands well below
/// the noise floor and well above flat.
fn mean_neighbor_gradient(pixels: &[u8], w: u32, h: u32) -> f32 {
    let (wi, hi) = (w as usize, h as usize);
    let luma = |x: usize, y: usize| -> f32 {
        let i = (y * wi + x) * 3;
        0.299 * pixels[i] as f32 + 0.587 * pixels[i + 1] as f32 + 0.114 * pixels[i + 2] as f32
    };
    let mut sum = 0.0f64;
    let mut cnt = 0u64;
    for y in 0..hi {
        for x in 0..wi {
            if x + 1 < wi {
                sum += (luma(x, y) - luma(x + 1, y)).abs() as f64;
                cnt += 1;
            }
            if y + 1 < hi {
                sum += (luma(x, y) - luma(x, y + 1)).abs() as f64;
                cnt += 1;
            }
        }
    }
    (sum / cnt as f64) as f32
}

/// Drive the public load→generate path at a given resolution / steps / quant and assert the output is
/// a coherent image (not constant, not pure noise). Saves the PNG and prints the sanity signals.
fn run_smoke(label: &str, w: u32, h: u32, steps: u32, quant: Quant, guidance: f32) {
    let snap = snapshot();
    eprintln!(
        "\n=== SD3.5-Large e2e smoke [{label}] {w}x{h} steps={steps} quant={quant:?} \
         guidance={guidance} ===\nsnapshot: {snap:?}"
    );

    // Reference a crate symbol so the generator's `inventory::submit!` static is linked (see the
    // `use mlx_gen_sd3 as sd3` note above).
    assert_eq!(sd3::MODEL_ID, "sd3_5_large");

    let spec = LoadSpec::new(WeightsSource::Dir(snap)).with_quant(quant);
    let t_load = std::time::Instant::now();
    let generator = mlx_gen::load(sd3::MODEL_ID, &spec).expect("load sd3_5_large");
    eprintln!("loaded in {:.1}s", t_load.elapsed().as_secs_f32());

    let req = GenerationRequest {
        prompt: "a photograph of a red fox sitting in a green meadow, sharp focus, daylight".into(),
        negative_prompt: Some("blurry, low quality, distorted".into()),
        width: w,
        height: h,
        count: 1,
        seed: Some(7),
        steps: Some(steps),
        guidance: Some(guidance),
        ..Default::default()
    };

    let t_gen = std::time::Instant::now();
    let mut last_step = 0u32;
    let out = generator
        .generate(&req, &mut |p| {
            if let Progress::Step { current, total } = p {
                last_step = current;
                if current == 1 || current == total || current % 4 == 0 {
                    eprintln!(
                        "  step {current}/{total} ({:.1}s)",
                        t_gen.elapsed().as_secs_f32()
                    );
                }
            }
        })
        .expect("generate");
    let gen_secs = t_gen.elapsed().as_secs_f32();
    assert_eq!(last_step, steps, "expected {steps} denoise-step events");

    let img = match out {
        GenerationOutput::Images(mut v) => {
            assert_eq!(v.len(), 1, "count=1 -> one image");
            v.pop().unwrap()
        }
        other => panic!("expected Images, got {other:?}"),
    };
    assert_eq!((img.width, img.height), (w, h), "image size");

    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../tools/golden/sd3_5_large_{label}.png"));
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    image::save_buffer(
        &out_path,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .expect("save png");

    let stats = channel_stats(&img.pixels, w, h);
    let grad = mean_neighbor_gradient(&img.pixels, w, h);
    eprintln!(
        "generated in {gen_secs:.1}s ({:.2}s/step)",
        gen_secs / steps as f32
    );
    eprintln!("saved: {out_path:?}");
    eprintln!(
        "channel mean/std: R {:.1}/{:.1}  G {:.1}/{:.1}  B {:.1}/{:.1}",
        stats[0].0, stats[0].1, stats[1].0, stats[1].1, stats[2].0, stats[2].1
    );
    eprintln!("mean neighbor gradient (luma): {grad:.2} (white-noise≈85, flat≈0, coherent≈5–25)");

    // --- coherence assertions (honest gates) ---------------------------------------------------
    // 1. NOT constant: at least one channel has meaningful contrast.
    let max_std = stats.iter().fold(0.0f32, |m, s| m.max(s.1));
    assert!(
        max_std > 8.0,
        "[{label}] output looks flat/constant (max channel std {max_std:.2}); a real image has \
         contrast"
    );
    // 2. NOT pure noise: a coherent image's neighbor gradient is far below the white-noise floor.
    assert!(
        grad < 60.0,
        "[{label}] neighbor gradient {grad:.2} near the white-noise floor (~85) — the render is \
         noise, NOT a coherent image. This means a bug in the MMDiT forward / sampler / CFG / VAE \
         de-norm."
    );
    // 3. NOT flat: there IS spatial structure.
    assert!(
        grad > 1.0,
        "[{label}] neighbor gradient {grad:.2} ≈ 0 — the render is essentially flat, not an image"
    );
    eprintln!("[{label}] PASS: coherent image (contrast {max_std:.1}, gradient {grad:.2})");
}

/// PRIMARY ACCEPTANCE: a real-weight 1024²/28-step true-CFG render. Q8 to fit the 8.1B model + 3 TEs
/// comfortably. This is the AC gate.
#[test]
#[ignore = "needs the SD3.5-Large snapshot (set SD3_LARGE_SNAPSHOT) + Metal"]
fn e2e_large_1024_q8() {
    run_smoke("1024_q8", 1024, 1024, 28, Quant::Q8, 3.5);
}

/// Faster correctness pre-check at a smaller resolution / fewer steps (Q4) — proves the path before
/// the full 1024²/28 run. Kept as a separate `#[ignore]` test so either can be run alone.
#[test]
#[ignore = "needs the SD3.5-Large snapshot (set SD3_LARGE_SNAPSHOT) + Metal"]
fn e2e_large_512_q4_quick() {
    run_smoke("512_q4", 512, 512, 20, Quant::Q4, 3.5);
}
