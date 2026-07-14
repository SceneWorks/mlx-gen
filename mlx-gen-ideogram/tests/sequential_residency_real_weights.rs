//! sc-10840 (epic 10834): the `Sequential` component-residency A/B on real Ideogram 4 weights.
//!
//! `#[ignore]`d — needs a converted Ideogram 4 MLX snapshot (`IDEOGRAM4_MLX`, else
//! `~/.cache/ideogram4-mlx-convert`) and Metal. Run:
//!   cargo test -p mlx-gen-ideogram --release --test sequential_residency_real_weights -- --ignored --nocapture
//!
//! Same two claims as the SD3 / SANA / Lens A/Bs: (1) `Sequential` peaks LOWER than `Resident` because
//! the Qwen3-VL text encoder is dropped (+ `clear_cache()`) before the two DiTs + VAE materialize, and
//! (2) the output is BYTE-IDENTICAL. A repeat-job check confirms nothing stays resident across jobs.
//! `IDEOGRAM4_SEQ_STEPS` / `IDEOGRAM4_SEQ_SIZE` tune the probe.

use std::path::PathBuf;

use mlx_gen::{GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, WeightsSource};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

// Force-link the provider crate so its `inventory` generator registration runs.
use mlx_gen_ideogram as _;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Resolve the converted Ideogram 4 MLX snapshot dir (`IDEOGRAM4_MLX` override, else the convert cache).
fn snapshot() -> PathBuf {
    std::env::var("IDEOGRAM4_MLX")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME")).join(".cache/ideogram4-mlx-convert")
        })
}

fn probe_request() -> GenerationRequest {
    // Asymmetric CFG (the quality id runs the uncond DiT) exercises the full heavy phase. A fixed seed
    // makes the byte-identity assertion meaningful; quality is irrelevant here. The prompt is the
    // model's native JSON caption.
    let size = env_u32("IDEOGRAM4_SEQ_SIZE", 768);
    GenerationRequest {
        prompt: r#"{"high_level_description":"a red fox in a snowy forest, photograph"}"#.into(),
        guidance: Some(7.0),
        width: size,
        height: size,
        seed: Some(1234),
        steps: Some(env_u32("IDEOGRAM4_SEQ_STEPS", 12)),
        ..Default::default()
    }
}

fn render_measured(policy: OffloadPolicy, req: &GenerationRequest) -> (Vec<u8>, usize) {
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot())).with_offload_policy(policy);
    let model = mlx_gen::load("ideogram_4", &spec).expect("load ideogram_4");
    reset_peak_memory();
    let out = model.generate(req, &mut |_| {}).expect("generate");
    let peak = get_peak_memory();
    let img = match out {
        GenerationOutput::Images(mut v) => {
            assert_eq!(v.len(), 1, "expected a single image");
            v.pop().unwrap()
        }
        other => panic!("expected Images, got {other:?}"),
    };
    let Image { pixels, .. } = img;
    drop(model);
    clear_cache();
    (pixels, peak)
}

#[test]
#[ignore = "needs a converted Ideogram 4 MLX snapshot (IDEOGRAM4_MLX or the convert cache)"]
fn sequential_bounds_peak_and_is_byte_identical() {
    let req = probe_request();
    let (pixels_resident, peak_resident) = render_measured(OffloadPolicy::Resident, &req);
    let (pixels_sequential, peak_sequential) = render_measured(OffloadPolicy::Sequential, &req);

    println!(
        "Ideogram-4 {}x{} @ {} steps:\n  Resident   peak = {:.3} GiB\n  Sequential peak = {:.3} GiB\n  saved = {:.3} GiB ({:.1}%)",
        req.width,
        req.height,
        req.steps.unwrap(),
        peak_resident as f64 / GIB,
        peak_sequential as f64 / GIB,
        (peak_resident.saturating_sub(peak_sequential)) as f64 / GIB,
        100.0 * (peak_resident.saturating_sub(peak_sequential)) as f64 / peak_resident as f64,
    );

    let diff = pixels_resident
        .iter()
        .zip(&pixels_sequential)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        diff,
        0,
        "Sequential residency changed the output: {diff}/{} bytes differ (must be byte-identical)",
        pixels_resident.len()
    );
    assert!(
        peak_sequential < peak_resident,
        "Sequential peak {:.3} GiB was not below Resident {:.3} GiB — the Qwen3-VL TE drop did not \
         reduce peak",
        peak_sequential as f64 / GIB,
        peak_resident as f64 / GIB,
    );
}

#[test]
#[ignore = "needs a converted Ideogram 4 MLX snapshot (IDEOGRAM4_MLX or the convert cache)"]
fn sequential_repeat_job_stays_bounded() {
    let req = probe_request();
    let (_p1, peak1) = render_measured(OffloadPolicy::Sequential, &req);
    let (_p2, peak2) = render_measured(OffloadPolicy::Sequential, &req);
    println!(
        "Ideogram-4 Sequential repeat-job peaks: job1 = {:.3} GiB, job2 = {:.3} GiB",
        peak1 as f64 / GIB,
        peak2 as f64 / GIB,
    );
    let slop = peak1 / 10;
    assert!(
        peak2 <= peak1 + slop,
        "repeat Sequential job peaked higher ({:.3} vs {:.3} GiB) — a component stayed resident",
        peak2 as f64 / GIB,
        peak1 as f64 / GIB,
    );
}
