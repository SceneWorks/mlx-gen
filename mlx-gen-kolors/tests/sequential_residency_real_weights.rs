//! sc-10840 (epic 10834): the `Sequential` component-residency A/B on real Kolors weights.
//!
//! `#[ignore]`d — needs a `Kwai-Kolors/Kolors-diffusers` snapshot (`KOLORS_SNAPSHOT`, else the HF
//! cache; the materialized `tokenizer/tokenizer.json` must be present) and Metal. Run:
//!   cargo test -p mlx-gen-kolors --release --test sequential_residency_real_weights -- --ignored --nocapture
//!
//! Same two claims as the SD3 / SANA / Lens A/Bs: (1) `Sequential` peaks LOWER than `Resident` because
//! the 6B ChatGLM3 encoder is dropped (+ `clear_cache()`) before the U-Net + VAE materialize, and (2)
//! the output is BYTE-IDENTICAL. A repeat-job check confirms nothing stays resident across jobs.
//! `KOLORS_SEQ_STEPS` / `KOLORS_SEQ_SIZE` tune the probe.

mod common;

use mlx_gen::{GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, WeightsSource};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

// Force-link the provider crate so its `inventory` generator registration runs.
use mlx_gen_kolors as _;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn probe_request() -> GenerationRequest {
    // CFG (guidance 5.0) exercises the seam's cond+uncond ChatGLM3 encode/materialize/drop path. A
    // fixed seed makes the byte-identity assertion meaningful; quality is irrelevant here.
    let size = env_u32("KOLORS_SEQ_SIZE", 768);
    GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        negative_prompt: Some("blurry, low quality".into()),
        guidance: Some(5.0),
        width: size,
        height: size,
        seed: Some(1234),
        steps: Some(env_u32("KOLORS_SEQ_STEPS", 12)),
        ..Default::default()
    }
}

fn render_measured(policy: OffloadPolicy, req: &GenerationRequest) -> (Vec<u8>, usize) {
    let spec = LoadSpec::new(WeightsSource::Dir(common::snapshot())).with_offload_policy(policy);
    let model = mlx_gen::load("kolors", &spec).expect("load kolors");
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
#[ignore = "needs a real Kwai-Kolors/Kolors-diffusers snapshot (KOLORS_SNAPSHOT or the HF cache)"]
fn sequential_bounds_peak_and_is_byte_identical() {
    let req = probe_request();
    let (pixels_resident, peak_resident) = render_measured(OffloadPolicy::Resident, &req);
    let (pixels_sequential, peak_sequential) = render_measured(OffloadPolicy::Sequential, &req);

    println!(
        "Kolors {}x{} @ {} steps:\n  Resident   peak = {:.3} GiB\n  Sequential peak = {:.3} GiB\n  saved = {:.3} GiB ({:.1}%)",
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
        "Sequential peak {:.3} GiB was not below Resident {:.3} GiB — the ChatGLM3 drop did not \
         reduce peak",
        peak_sequential as f64 / GIB,
        peak_resident as f64 / GIB,
    );
}

#[test]
#[ignore = "needs a real Kwai-Kolors/Kolors-diffusers snapshot (KOLORS_SNAPSHOT or the HF cache)"]
fn sequential_repeat_job_stays_bounded() {
    let req = probe_request();
    let (_p1, peak1) = render_measured(OffloadPolicy::Sequential, &req);
    let (_p2, peak2) = render_measured(OffloadPolicy::Sequential, &req);
    println!(
        "Kolors Sequential repeat-job peaks: job1 = {:.3} GiB, job2 = {:.3} GiB",
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
