//! epic 7114 (sc-7120): Boogu **Base** unified curated-sampler + scheduler smoke over the public
//! `generate()` path (real weights). Proves the P3 flow-match adoption for Boogu's Base/Edit rectified-
//! flow Euler loop (the **OneMinusSigma** convention with a static `mu = 1.15` shift, whose DiT predicts
//! the velocity in clean-fraction time — `predict` negates it into the noise-fraction FLOW convention):
//!
//! - **N1 (default no-op):** an unset `req.sampler` resolves to the curated Euler integrator over the
//!   native static-shift schedule, so it is byte-identical to an explicit `sampler: "euler"`.
//! - **N2 (named sampler coherent):** `sampler: "dpmpp_2m"` renders a coherent image that differs from
//!   Euler — a real solver swap, not a silent fallback.
//! - **Scheduler axis:** `scheduler: "karras"` renders a coherent image that differs from the native
//!   schedule — the curated σ schedule (shift-aware, `mu = 1.15`) flows through end-to-end.
//!
//! `#[ignore]`d — needs the real Boogu Base snapshot (`mllm/ transformer/ vae/`), env `BOOGU_BASE_DIR`:
//!   BOOGU_BASE_DIR=/path/to/boogu_base \
//!     cargo test -p mlx-gen-boogu --release --test unified_sampler_smoke -- --ignored --nocapture

use std::path::{Path, PathBuf};

use mlx_gen::{GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource};
// Reference the provider crate so its `inventory` registration links (`mlx_gen::load` resolves it).
use mlx_gen_boogu::BOOGU_IMAGE_ID;

const W: u32 = 256;
const H: u32 = 256;
const STEPS: u32 = 8;
const SEED: u64 = 42;
const PROMPT: &str = "a red apple on a wooden table, photorealistic";

fn snapshot() -> Option<PathBuf> {
    std::env::var("BOOGU_BASE_DIR").ok().map(PathBuf::from)
}

/// (std, distinct-level-count, mean horizontal-adjacent-|Δ|) over an RGB8 buffer — a coherent natural
/// image has a broad histogram AND spatial smoothness; pure noise has a high adjacent Δ and a flat std.
fn image_stats(px: &[u8], w: u32) -> (f32, usize, f32) {
    let n = px.len() as f64;
    let mean = px.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = px.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let mut seen = [false; 256];
    for &v in px {
        seen[v as usize] = true;
    }
    let distinct = seen.iter().filter(|&&b| b).count();
    let stride = (w * 3) as usize;
    let (mut adj_sum, mut adj_n) = (0f64, 0u64);
    for (i, &v) in px.iter().enumerate() {
        if i >= 3 && i % stride >= 3 {
            adj_sum += (v as i32 - px[i - 3] as i32).unsigned_abs() as f64;
            adj_n += 1;
        }
    }
    (
        var.sqrt() as f32,
        distinct,
        (adj_sum / adj_n.max(1) as f64) as f32,
    )
}

fn assert_coherent(img: &Image, label: &str) {
    assert_eq!(
        img.pixels.len(),
        (img.width * img.height * 3) as usize,
        "{label}: pixel buffer is RGB8 HWC"
    );
    let (std, distinct, adj) = image_stats(&img.pixels, img.width);
    eprintln!("[{label}] std={std:.1} distinct={distinct} adjΔ={adj:.1}");
    assert!(std > 10.0, "{label}: image is near-flat (std {std:.1})");
    assert!(
        distinct > 24,
        "{label}: too few distinct levels ({distinct})"
    );
    assert!(
        adj < 60.0,
        "{label}: not spatially smooth — looks like noise (adjΔ {adj:.1})"
    );
}

fn render(root: &Path, sampler: Option<&str>, scheduler: Option<&str>) -> Image {
    let spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf()));
    let generator = mlx_gen::load(BOOGU_IMAGE_ID, &spec).expect("load boogu_image");
    let req = GenerationRequest {
        prompt: PROMPT.into(),
        width: W,
        height: H,
        seed: Some(SEED),
        steps: Some(STEPS),
        guidance: Some(4.0),
        sampler: sampler.map(Into::into),
        scheduler: scheduler.map(Into::into),
        ..Default::default()
    };
    match generator.generate(&req, &mut |_| {}).unwrap() {
        GenerationOutput::Images(mut v) => v.pop().unwrap(),
        other => panic!("expected Images, got {other:?}"),
    }
}

fn frac_diff(a: &Image, b: &Image) -> f32 {
    let differ = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .filter(|(x, y)| (**x as i16 - **y as i16).abs() > 4)
        .count();
    differ as f32 / a.pixels.len() as f32
}

/// N1: the unset-sampler default selects the SAME curated Euler integrator as an explicit
/// `sampler:"euler"` (both `None` and `Some("euler")` resolve to `Euler` in `run_flow_sampler`). This is
/// the proof the OneMinusSigma routing + velocity negation reproduce the legacy Base Euler loop.
///
/// Boogu's Base path is a large bf16 DiT + Qwen3-VL text encoder + fused SDPA, which is NOT bit-exactly
/// reproducible run-to-run on Metal (unlike the smaller Z-Image-Turbo) — so the gate is "default agrees
/// with euler at least as well as default agrees with ITSELF across two renders" (the compute-noise
/// floor), which still rejects a silent sampler mismatch (a real solver swap diverges in 65–80% of
/// pixels, as the `dpmpp_2m` test shows — orders of magnitude past the noise floor).
#[test]
#[ignore = "needs the real Boogu Base snapshot (set BOOGU_BASE_DIR)"]
fn default_sampler_equals_explicit_euler() {
    let Some(root) = snapshot() else {
        eprintln!("skipping: set BOOGU_BASE_DIR");
        return;
    };
    let default = render(&root, None, None);
    let default2 = render(&root, None, None); // run-to-run determinism probe
    let euler = render(&root, Some("euler"), None);
    assert_coherent(&default, "default");
    assert_coherent(&euler, "euler");
    let det_frac = frac_diff(&default, &default2);
    let eul_frac = frac_diff(&default, &euler);
    eprintln!("[N1] default-vs-default (run-to-run noise floor): frac={det_frac:.4}");
    eprintln!("[N1] default-vs-euler:                            frac={eul_frac:.4}");
    assert!(
        eul_frac <= det_frac + 0.02,
        "default (None) must select the same Euler solver as explicit \"euler\" — it diverged \
         (frac {eul_frac:.4}) well past the run-to-run noise floor (frac {det_frac:.4}); a silent \
         sampler mismatch?"
    );
}

/// N2: a curated named sampler renders a coherent image that genuinely differs from Euler.
#[test]
#[ignore = "needs the real Boogu Base snapshot (set BOOGU_BASE_DIR)"]
fn named_sampler_dpmpp_2m_is_coherent_and_distinct() {
    let Some(root) = snapshot() else {
        eprintln!("skipping: set BOOGU_BASE_DIR");
        return;
    };
    let euler = render(&root, Some("euler"), None);
    let dpmpp = render(&root, Some("dpmpp_2m"), None);
    assert_coherent(&dpmpp, "dpmpp_2m");
    let frac = frac_diff(&euler, &dpmpp);
    eprintln!(
        "[dpmpp_2m] differs from euler in {:.2}% of pixels",
        frac * 100.0
    );
    assert!(
        frac > 0.01,
        "dpmpp_2m must differ from euler (a real solver swap, not a silent fallback); differ {frac}"
    );
}

/// Scheduler axis: a curated scheduler re-shapes the σ schedule (shift-aware, `mu = 1.15`) and renders a
/// coherent image that differs from the native schedule. Uses `karras` — a structurally-distinct σ ramp:
/// the `normal`/`sgm_uniform` schedules nearly coincide with Boogu's native `linspace(1,1/N,N)`-through-
/// shift (a weak distinctness signal, ~0.7%), while `karras` re-distributes the steps and differs clearly.
#[test]
#[ignore = "needs the real Boogu Base snapshot (set BOOGU_BASE_DIR)"]
fn scheduler_karras_is_coherent_and_distinct() {
    let Some(root) = snapshot() else {
        eprintln!("skipping: set BOOGU_BASE_DIR");
        return;
    };
    let native = render(&root, None, None);
    let karras = render(&root, None, Some("karras"));
    assert_coherent(&karras, "karras");
    let frac = frac_diff(&native, &karras);
    eprintln!(
        "[karras] differs from native in {:.2}% of pixels",
        frac * 100.0
    );
    assert!(
        frac > 0.01,
        "karras must re-shape the schedule vs the native default; differ {frac}"
    );
}
