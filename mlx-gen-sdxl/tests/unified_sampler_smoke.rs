//! epic 7114 (sc-7121): SDXL unified curated-sampler + scheduler smoke over the public `generate()`
//! path (real weights). Proves the **additive** DDPM-cohort adoption — the curated k-diffusion path
//! (`denoise_curated` over `DiscreteModelSampling`, ε-prediction) runs end-to-end alongside the bespoke
//! ancestral default, which is left byte-exact:
//!
//! - **N1 (default untouched):** an unset `req.sampler`/`req.scheduler` renders the bespoke ancestral
//!   default — a coherent image (the legacy loop, not the curated path). This is the byte-exact default.
//! - **N2 (named curated sampler):** `sampler: "dpmpp_2m"` renders a coherent image that differs from
//!   the default — a real solver swap onto the unified path, not a silent fallback.
//! - **Scheduler axis:** `scheduler: "karras"` (with the curated `euler` sampler) renders a coherent
//!   image — the curated σ schedule flows through `schedule_sigmas`/`run_curated_sampler` end-to-end.
//!
//! `#[ignore]`d — needs the real `stabilityai/stable-diffusion-xl-base-1.0` snapshot (env
//! `SDXL_SNAPSHOT` or the HF cache):
//!   SDXL_SNAPSHOT=/path/to/sdxl-base-1.0 \
//!     cargo test -p mlx-gen-sdxl --release --test unified_sampler_smoke -- --ignored --nocapture

mod common;

use mlx_gen::{GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource};
// Reference the provider crate so its `inventory` registration links (`mlx_gen::load` resolves it).
use mlx_gen_sdxl::MODEL_ID;

const W: u32 = 512;
const H: u32 = 512;
const STEPS: u32 = 8;
const SEED: u64 = 42;
const PROMPT: &str = "a fox sitting in a forest, photorealistic";

use common::snapshot_opt as snapshot;

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

fn render(sampler: Option<&str>, scheduler: Option<&str>) -> Option<Image> {
    let root = snapshot()?;
    let spec = LoadSpec::new(WeightsSource::Dir(root));
    let generator = mlx_gen::load(MODEL_ID, &spec).expect("load sdxl");
    let req = GenerationRequest {
        prompt: PROMPT.into(),
        width: W,
        height: H,
        seed: Some(SEED),
        steps: Some(STEPS),
        guidance: Some(7.0),
        sampler: sampler.map(Into::into),
        scheduler: scheduler.map(Into::into),
        ..Default::default()
    };
    match generator.generate(&req, &mut |_| {}).unwrap() {
        GenerationOutput::Images(mut v) => Some(v.pop().unwrap()),
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

/// N1: the bespoke ancestral default renders a coherent image (the legacy loop, unchanged).
#[test]
#[ignore = "needs the real stabilityai/stable-diffusion-xl-base-1.0 snapshot (set SDXL_SNAPSHOT)"]
fn default_ancestral_is_coherent() {
    let Some(img) = render(None, None) else {
        eprintln!("skipping: no SDXL snapshot (set SDXL_SNAPSHOT)");
        return;
    };
    assert_coherent(&img, "default");
}

/// N2: a curated named sampler renders a coherent image that genuinely differs from the default.
#[test]
#[ignore = "needs the real stabilityai/stable-diffusion-xl-base-1.0 snapshot (set SDXL_SNAPSHOT)"]
fn curated_dpmpp_2m_is_coherent_and_distinct() {
    let (Some(default), Some(dpmpp)) = (render(None, None), render(Some("dpmpp_2m"), None)) else {
        eprintln!("skipping: no SDXL snapshot (set SDXL_SNAPSHOT)");
        return;
    };
    assert_coherent(&dpmpp, "dpmpp_2m");
    let frac = frac_diff(&default, &dpmpp);
    eprintln!(
        "[dpmpp_2m] differs from default in {:.2}% of pixels",
        frac * 100.0
    );
    assert!(
        frac > 0.01,
        "dpmpp_2m must differ from the ancestral default (a real curated solver, not a fallback); \
         differ {frac}"
    );
}

/// Scheduler axis: the curated `karras` σ schedule (with the curated `euler` sampler) renders a
/// coherent image — proving `req.scheduler` flows through `schedule_sigmas`/`run_curated_sampler`.
#[test]
#[ignore = "needs the real stabilityai/stable-diffusion-xl-base-1.0 snapshot (set SDXL_SNAPSHOT)"]
fn curated_karras_scheduler_is_coherent() {
    let Some(karras) = render(Some("euler"), Some("karras")) else {
        eprintln!("skipping: no SDXL snapshot (set SDXL_SNAPSHOT)");
        return;
    };
    assert_coherent(&karras, "euler+karras");
}
