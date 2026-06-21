//! epic 7114 (sc-7120): FLUX.2-klein unified curated-sampler smoke over the public `generate()` path
//! (real weights). Proves the P3 flow-match adoption for FLUX.2 (the `σ · 1000` timestep convention):
//!
//! - **N1 (default no-op):** an unset `req.sampler` resolves to the curated Euler integrator over the
//!   resolution-shifted flow schedule — the same schedule the legacy inline loop drove — so it is
//!   byte-identical to an explicit `sampler: "euler"`.
//! - **N2 (named sampler coherent):** `sampler: "dpmpp_2m"` renders a coherent natural image that
//!   genuinely differs from Euler.
//!
//! `#[ignore]`d — needs the real `black-forest-labs/FLUX.2-klein-9b` snapshot (env `FLUX2_SNAPSHOT`
//! or the HF cache):
//!   cargo test -p mlx-gen-flux2 --test unified_sampler_smoke -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::{GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource};
// Reference the provider crate so its `inventory` registration links (`mlx_gen::load` resolves it).
use mlx_gen_flux2::FLUX2_KLEIN_9B_ID;

const W: u32 = 256;
const H: u32 = 256;
const STEPS: u32 = 4;
const SEED: u64 = 42;
const PROMPT: &str = "a fox sitting in a forest, photorealistic";

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("FLUX2_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-9b/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

/// (std, distinct-level-count, mean horizontal-adjacent-|Δ|) over an RGB8 buffer — a coherent natural
/// image has a broad histogram AND spatial smoothness; pure noise has a high adjacent Δ, a flat fill
/// std≈0.
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

fn render(sampler: Option<&str>) -> Image {
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    let generator = mlx_gen::load(FLUX2_KLEIN_9B_ID, &spec).expect("load flux2_klein_9b");
    let req = GenerationRequest {
        prompt: PROMPT.into(),
        width: W,
        height: H,
        seed: Some(SEED),
        steps: Some(STEPS),
        sampler: sampler.map(Into::into),
        ..Default::default()
    };
    match generator.generate(&req, &mut |_| {}).unwrap() {
        GenerationOutput::Images(mut v) => v.pop().unwrap(),
        other => panic!("expected Images, got {other:?}"),
    }
}

/// N1: the unset-sampler default IS the curated Euler integrator — byte-identical to `sampler:"euler"`.
#[test]
#[ignore = "needs the real black-forest-labs/FLUX.2-klein-9b snapshot"]
fn default_sampler_equals_explicit_euler() {
    let default = render(None);
    let euler = render(Some("euler"));
    assert_coherent(&default, "default");
    assert_eq!(
        default.pixels, euler.pixels,
        "the unset-sampler default must be byte-identical to explicit euler (epic 7114 N1)"
    );
}

/// N2: a curated named sampler renders a coherent image that genuinely differs from Euler.
#[test]
#[ignore = "needs the real black-forest-labs/FLUX.2-klein-9b snapshot"]
fn named_sampler_dpmpp_2m_is_coherent_and_distinct() {
    let euler = render(Some("euler"));
    let dpmpp = render(Some("dpmpp_2m"));
    assert_coherent(&dpmpp, "dpmpp_2m");
    let differ = euler
        .pixels
        .iter()
        .zip(&dpmpp.pixels)
        .filter(|(a, b)| (**a as i16 - **b as i16).abs() > 4)
        .count();
    let frac = differ as f32 / euler.pixels.len() as f32;
    eprintln!(
        "[dpmpp_2m] differs from euler in {:.2}% of pixels",
        frac * 100.0
    );
    assert!(
        frac > 0.01,
        "dpmpp_2m must differ from euler (a real solver swap, not a silent fallback); differ {frac}"
    );
}
