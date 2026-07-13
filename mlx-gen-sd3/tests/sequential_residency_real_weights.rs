//! sc-10840 (epic 10834): the `Sequential` component-residency A/B on real SD3.5 weights.
//!
//! `#[ignore]`d — needs a real `stabilityai/stable-diffusion-3.5-large` snapshot (`SD3_LARGE_SNAPSHOT`,
//! else the HF cache). Run:
//!   cargo test -p mlx-gen-sd3 --release --test sequential_residency_real_weights -- --ignored --nocapture
//!
//! Same two claims as the SDXL / Z-Image A/Bs: (1) `Sequential` peaks LOWER than `Resident` because the
//! TRIPLE text encoder (CLIP-L + CLIP-G + T5-XXL) is dropped (+ `clear_cache()`) before the MMDiT + VAE
//! materialize, and (2) the output is BYTE-IDENTICAL. SD3's T5-XXL alone is the biggest TE-drop in the
//! group, so the saving is proportionally large. A repeat-job check confirms nothing stays resident
//! across jobs. Set `SD3_SEQ_Q8=1` for the Q8 case, `SD3_SEQ_STEPS`/`SD3_SEQ_SIZE` to tune.

use std::path::PathBuf;

use mlx_gen::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, Quant, WeightsSource,
};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

// Force-link the provider crate so its `inventory` generator registration runs (this test only calls
// `mlx_gen::load("sd3_5_large", …)`).
use mlx_gen_sd3 as _;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

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

fn probe_request() -> GenerationRequest {
    // Real CFG (pos + neg encode) to exercise the seam's triple-TE cond+uncond materialize/drop path.
    // A fixed seed makes the byte-identity assertion meaningful; quality is irrelevant here.
    let size = env_u32("SD3_SEQ_SIZE", 768);
    GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        negative_prompt: Some("blurry, low quality".into()),
        guidance: Some(4.5),
        width: size,
        height: size,
        seed: Some(1234),
        steps: Some(env_u32("SD3_SEQ_STEPS", 12)),
        ..Default::default()
    }
}

fn base_spec() -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    if std::env::var("SD3_SEQ_Q8").is_ok() {
        spec = spec.with_quant(Quant::Q8);
    }
    spec
}

fn render_measured(policy: OffloadPolicy, req: &GenerationRequest) -> (Vec<u8>, usize) {
    let spec = base_spec().with_offload_policy(policy);
    let model = mlx_gen::load("sd3_5_large", &spec).expect("load sd3_5_large");
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
#[ignore = "needs a real SD3.5-Large snapshot (SD3_LARGE_SNAPSHOT or the HF cache)"]
fn sequential_bounds_peak_and_is_byte_identical() {
    let req = probe_request();
    let (pixels_resident, peak_resident) = render_measured(OffloadPolicy::Resident, &req);
    let (pixels_sequential, peak_sequential) = render_measured(OffloadPolicy::Sequential, &req);

    println!(
        "SD3.5-Large {}x{} @ {} steps{}:\n  Resident   peak = {:.3} GiB\n  Sequential peak = {:.3} GiB\n  saved = {:.3} GiB ({:.1}%)",
        req.width,
        req.height,
        req.steps.unwrap(),
        if std::env::var("SD3_SEQ_Q8").is_ok() { " (Q8)" } else { "" },
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
        "Sequential peak {:.3} GiB was not below Resident {:.3} GiB — the triple-TE drop did not \
         reduce peak",
        peak_sequential as f64 / GIB,
        peak_resident as f64 / GIB,
    );
}

#[test]
#[ignore = "needs a real SD3.5-Large snapshot (SD3_LARGE_SNAPSHOT or the HF cache)"]
fn sequential_repeat_job_stays_bounded() {
    let req = probe_request();
    let (_p1, peak1) = render_measured(OffloadPolicy::Sequential, &req);
    let (_p2, peak2) = render_measured(OffloadPolicy::Sequential, &req);
    println!(
        "SD3.5-Large Sequential repeat-job peaks: job1 = {:.3} GiB, job2 = {:.3} GiB",
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
