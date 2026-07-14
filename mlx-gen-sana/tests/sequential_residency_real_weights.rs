//! sc-10840 (epic 10834): the `Sequential` component-residency A/B on real SANA weights.
//!
//! `#[ignore]`d — needs a real `Sana_1600M_1024px_diffusers`-shaped snapshot (`SANA_PIPELINE_WEIGHTS`).
//! Run:
//!   SANA_PIPELINE_WEIGHTS=/path/Sana_1600M_1024px_diffusers \
//!     cargo test -p mlx-gen-sana --release --test sequential_residency_real_weights -- --ignored --nocapture
//!
//! Same two claims as the SDXL / Z-Image A/Bs: (1) `Sequential` peaks LOWER than `Resident` because the
//! Gemma-2 CHI text encoder is dropped (+ `clear_cache()`) before the Linear-DiT trunk + DC-AE
//! materialize, and (2) the output is BYTE-IDENTICAL. SANA's Gemma encoder is comparable to (often ≥)
//! the DiT, so the saving is proportionally large. A repeat-job check confirms nothing stays resident
//! across jobs. Set `SANA_SEQ_STEPS` / `SANA_SEQ_SIZE` to tune; `SANA_SPRINT=1` drives the Sprint id.

use std::path::PathBuf;

use mlx_gen::{GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, WeightsSource};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

// Force-link the provider crate so its `inventory` generator registration runs (this test only calls
// `mlx_gen::load("sana_1600m", …)`).
use mlx_gen_sana as _;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn snapshot() -> Option<PathBuf> {
    std::env::var("SANA_PIPELINE_WEIGHTS")
        .ok()
        .map(PathBuf::from)
}

fn is_sprint() -> bool {
    std::env::var("SANA_SPRINT").is_ok()
}

fn model_id() -> &'static str {
    if is_sprint() {
        "sana_sprint_1600m"
    } else {
        "sana_1600m"
    }
}

fn probe_request() -> GenerationRequest {
    // Base SANA is true-CFG (pos + neg encode) — exercises the seam's cond+uncond materialize/drop path.
    // Sprint is CFG-free (cond only). A fixed seed makes the byte-identity assertion meaningful.
    let size = env_u32("SANA_SEQ_SIZE", 1024);
    let (guidance, negative, steps) = if is_sprint() {
        (None, None, env_u32("SANA_SEQ_STEPS", 2))
    } else {
        (
            Some(4.5),
            Some("blurry, low quality".to_string()),
            env_u32("SANA_SEQ_STEPS", 12),
        )
    };
    GenerationRequest {
        prompt: "a red panda on a mossy log in a misty forest, photograph".into(),
        negative_prompt: negative,
        guidance,
        width: size,
        height: size,
        seed: Some(1234),
        steps: Some(steps),
        ..Default::default()
    }
}

fn render_measured(
    policy: OffloadPolicy,
    snap: PathBuf,
    req: &GenerationRequest,
) -> (Vec<u8>, usize) {
    let spec = LoadSpec::new(WeightsSource::Dir(snap)).with_offload_policy(policy);
    let model = mlx_gen::load(model_id(), &spec).expect("load sana");
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
#[ignore = "needs a Sana_1600M_1024px_diffusers snapshot; set SANA_PIPELINE_WEIGHTS"]
fn sequential_bounds_peak_and_is_byte_identical() {
    let Some(snap) = snapshot() else {
        eprintln!("skipping: set SANA_PIPELINE_WEIGHTS to run the SANA residency A/B");
        return;
    };
    let req = probe_request();
    let (pixels_resident, peak_resident) =
        render_measured(OffloadPolicy::Resident, snap.clone(), &req);
    let (pixels_sequential, peak_sequential) =
        render_measured(OffloadPolicy::Sequential, snap, &req);

    println!(
        "SANA ({}) {}x{} @ {} steps:\n  Resident   peak = {:.3} GiB\n  Sequential peak = {:.3} GiB\n  saved = {:.3} GiB ({:.1}%)",
        model_id(),
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
        "Sequential peak {:.3} GiB was not below Resident {:.3} GiB — the Gemma-TE drop did not \
         reduce peak",
        peak_sequential as f64 / GIB,
        peak_resident as f64 / GIB,
    );
}

#[test]
#[ignore = "needs a Sana_1600M_1024px_diffusers snapshot; set SANA_PIPELINE_WEIGHTS"]
fn sequential_repeat_job_stays_bounded() {
    let Some(snap) = snapshot() else {
        eprintln!("skipping: set SANA_PIPELINE_WEIGHTS to run the SANA residency A/B");
        return;
    };
    let req = probe_request();
    let (_p1, peak1) = render_measured(OffloadPolicy::Sequential, snap.clone(), &req);
    let (_p2, peak2) = render_measured(OffloadPolicy::Sequential, snap, &req);
    println!(
        "SANA Sequential repeat-job peaks: job1 = {:.3} GiB, job2 = {:.3} GiB",
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
