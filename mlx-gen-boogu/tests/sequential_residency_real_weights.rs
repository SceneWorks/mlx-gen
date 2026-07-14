//! sc-10840 (epic 10834): the `Sequential` component-residency A/B on real Boogu-Image weights.
//!
//! `#[ignore]`d — needs a converted Boogu Base snapshot (`BOOGU_BASE_DIR`, the crate's real-weight
//! convention — see `tests/generator.rs`). Run:
//!   BOOGU_BASE_DIR=<base snapshot> \
//!     cargo test -p mlx-gen-boogu --release --test sequential_residency_real_weights -- --ignored --nocapture
//!
//! Same two claims as the SD3 / Z-Image A/Bs: (1) `Sequential` peaks LOWER than `Resident` because the
//! ~17.5 GB Qwen3-VL `mllm/` encoder is dropped (+ `clear_cache()`) before the ~20.6 GB DiT + VAE
//! materialize, and (2) the output is BYTE-IDENTICAL (the render phase — including the img2img/edit VAE
//! encodes — is independent of the mllm, so dropping it changes nothing). A repeat-job check confirms
//! nothing stays resident across jobs. Set `BOOGU_SEQ_Q8=1` for the Q8 case, `BOOGU_SEQ_STEPS`/
//! `BOOGU_SEQ_SIZE` to tune.

use std::path::PathBuf;

use mlx_gen::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, Quant, WeightsSource,
};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

// Force-link the provider crate so its `inventory` generator registration runs.
use mlx_gen_boogu as _;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The converted Boogu Base snapshot dir (`BOOGU_BASE_DIR`).
fn snapshot() -> PathBuf {
    std::env::var("BOOGU_BASE_DIR")
        .map(PathBuf::from)
        .expect("set BOOGU_BASE_DIR to the converted Boogu Base snapshot")
}

fn probe_request() -> GenerationRequest {
    // True CFG (pos + drop-instruction encode) exercises the seam's cond+uncond materialize/drop path.
    // A fixed seed makes the byte-identity assertion meaningful; quality is irrelevant here.
    let size = env_u32("BOOGU_SEQ_SIZE", 768);
    GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        guidance: Some(4.0),
        width: size,
        height: size,
        seed: Some(1234),
        steps: Some(env_u32("BOOGU_SEQ_STEPS", 12)),
        ..Default::default()
    }
}

fn base_spec() -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    if std::env::var("BOOGU_SEQ_Q8").is_ok() {
        spec = spec.with_quant(Quant::Q8);
    }
    spec
}

fn render_measured(policy: OffloadPolicy, req: &GenerationRequest) -> (Vec<u8>, usize) {
    let spec = base_spec().with_offload_policy(policy);
    let model = mlx_gen::load("boogu_image", &spec).expect("load boogu_image");
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
#[ignore = "needs a converted Boogu Base snapshot (BOOGU_BASE_DIR); macos-mlx / dev box only"]
fn sequential_bounds_peak_and_is_byte_identical() {
    let req = probe_request();
    let (pixels_resident, peak_resident) = render_measured(OffloadPolicy::Resident, &req);
    let (pixels_sequential, peak_sequential) = render_measured(OffloadPolicy::Sequential, &req);

    println!(
        "Boogu Base {}x{} @ {} steps{}:\n  Resident   peak = {:.3} GiB\n  Sequential peak = {:.3} GiB\n  saved = {:.3} GiB ({:.1}%)",
        req.width,
        req.height,
        req.steps.unwrap(),
        if std::env::var("BOOGU_SEQ_Q8").is_ok() { " (Q8)" } else { "" },
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
        "Sequential peak {:.3} GiB was not below Resident {:.3} GiB — the mllm drop did not reduce peak",
        peak_sequential as f64 / GIB,
        peak_resident as f64 / GIB,
    );
}

#[test]
#[ignore = "needs a converted Boogu Base snapshot (BOOGU_BASE_DIR); macos-mlx / dev box only"]
fn sequential_repeat_job_stays_bounded() {
    let req = probe_request();
    let (_p1, peak1) = render_measured(OffloadPolicy::Sequential, &req);
    let (_p2, peak2) = render_measured(OffloadPolicy::Sequential, &req);
    println!(
        "Boogu Base Sequential repeat-job peaks: job1 = {:.3} GiB, job2 = {:.3} GiB",
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
