//! sc-10839 (epic 10834 Phase 1): the `Sequential` component-residency A/B on real Z-Image weights.
//!
//! `#[ignore]`d — needs a real Z-Image-Turbo snapshot (`ZIMAGE_SNAPSHOT`, else the HF cache). Run:
//!   cargo test -p mlx-gen-z-image --release --test sequential_residency_real_weights -- --ignored --nocapture
//!
//! Same two claims as the SDXL A/B (see that file): (1) `Sequential` peaks LOWER than `Resident`
//! because the Qwen text encoder is dropped (+ `clear_cache()`) before the DiT materializes, and
//! (2) the output is BYTE-IDENTICAL. Z-Image's Qwen encoder is comparable to the DiT, so the saving
//! is proportionally larger than SDXL's. A repeat-job check confirms nothing stays resident across
//! jobs. Set `ZIMAGE_SEQ_Q8=1` for the Q8 case, `ZIMAGE_SEQ_STEPS`/`ZIMAGE_SEQ_SIZE` to tune.

mod common;

// Force-link the provider crate so its `inventory` generator registration runs (this test only
// calls `mlx_gen::load("z_image_turbo", …)`).
use mlx_gen_z_image as _;

use common::snapshot;
use mlx_gen::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, Quant, WeightsSource,
};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};
use std::path::PathBuf;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn probe_request() -> GenerationRequest {
    // Turbo is guidance-distilled (no CFG / negative prompt), 4-step default. A fixed seed makes the
    // byte-identity assertion meaningful; quality is irrelevant (Resident vs Sequential, not a golden).
    let size = env_u32("ZIMAGE_SEQ_SIZE", 768);
    GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        width: size,
        height: size,
        seed: Some(1234),
        steps: Some(env_u32("ZIMAGE_SEQ_STEPS", 4)),
        ..Default::default()
    }
}

/// A base-`z_image` probe (F-172, sc-11124): the base is undistilled, so it runs **real CFG** — a
/// negative prompt + guidance — which exercises the seam's pos+neg encode/materialize/drop path (the
/// Turbo probe only encodes a single cond). Small step count keeps the ignored test tractable.
fn base_probe_request() -> GenerationRequest {
    let size = env_u32("ZIMAGE_SEQ_SIZE", 768);
    GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        negative_prompt: Some("blurry, low quality".into()),
        guidance: Some(4.0),
        width: size,
        height: size,
        seed: Some(1234),
        steps: Some(env_u32("ZIMAGE_SEQ_STEPS", 8)),
        ..Default::default()
    }
}

/// The base `Tongyi-MAI/Z-Image` snapshot, from `ZIMAGE_BASE_SNAPSHOT` (a distinct repo from the Turbo
/// snapshot). Returns `None` so the base A/B skips rather than panics when only the Turbo snapshot is
/// available.
fn base_model_snapshot_opt() -> Option<PathBuf> {
    std::env::var("ZIMAGE_BASE_SNAPSHOT")
        .ok()
        .map(PathBuf::from)
}

fn spec_for(snapshot: PathBuf) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot));
    if std::env::var("ZIMAGE_SEQ_Q8").is_ok() {
        spec = spec.with_quant(Quant::Q8);
    }
    spec
}

fn base_spec() -> LoadSpec {
    spec_for(snapshot())
}

fn render_measured(policy: OffloadPolicy, req: &GenerationRequest) -> (Vec<u8>, usize) {
    render_measured_id("z_image_turbo", base_spec(), policy, req)
}

/// The generalized A/B render: load `model_id` from `spec` under `policy`, measure peak, return the
/// single output image's bytes + peak. Shared by the Turbo flagship probe and the base `z_image`
/// sibling probe (sc-11124) so both exercise the identical shared [`mlx_gen::Residency`] seam.
fn render_measured_id(
    model_id: &str,
    spec: LoadSpec,
    policy: OffloadPolicy,
    req: &GenerationRequest,
) -> (Vec<u8>, usize) {
    let spec = spec.with_offload_policy(policy);
    let model = mlx_gen::load(model_id, &spec).expect("load model");
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
#[ignore = "needs a real Z-Image-Turbo snapshot (ZIMAGE_SNAPSHOT or the HF cache)"]
fn sequential_bounds_peak_and_is_byte_identical() {
    let req = probe_request();
    let (pixels_resident, peak_resident) = render_measured(OffloadPolicy::Resident, &req);
    let (pixels_sequential, peak_sequential) = render_measured(OffloadPolicy::Sequential, &req);

    println!(
        "Z-Image {}x{} @ {} steps{}:\n  Resident   peak = {:.3} GiB\n  Sequential peak = {:.3} GiB\n  saved = {:.3} GiB ({:.1}%)",
        req.width,
        req.height,
        req.steps.unwrap(),
        if std::env::var("ZIMAGE_SEQ_Q8").is_ok() { " (Q8)" } else { "" },
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
        "Sequential peak {:.3} GiB was not below Resident {:.3} GiB — the text-encoder drop did not \
         reduce peak",
        peak_sequential as f64 / GIB,
        peak_resident as f64 / GIB,
    );
}

#[test]
#[ignore = "needs a real Z-Image-Turbo snapshot (ZIMAGE_SNAPSHOT or the HF cache)"]
fn sequential_repeat_job_stays_bounded() {
    let req = probe_request();
    let (_p1, peak1) = render_measured(OffloadPolicy::Sequential, &req);
    let (_p2, peak2) = render_measured(OffloadPolicy::Sequential, &req);
    println!(
        "Z-Image Sequential repeat-job peaks: job1 = {:.3} GiB, job2 = {:.3} GiB",
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

#[test]
#[ignore = "needs a real base Z-Image snapshot (set ZIMAGE_BASE_SNAPSHOT)"]
fn base_z_image_sequential_bounds_peak_and_is_byte_identical() {
    // F-172 (sc-11124): the base `z_image` sibling now honors `offload_policy` via the SAME shared
    // `mlx_gen::Residency` seam as the Turbo flagship — before the fix it ignored the policy and always
    // loaded `Resident` (silently OOMing a fit-gated Sequential request). This is the base analog of
    // `sequential_bounds_peak_and_is_byte_identical`, gated on a distinct base snapshot; it runs REAL
    // CFG (pos+neg encode), the harder seam path. The two control siblings share this same seam via
    // `load_control_residency` (weight-free routing covered by their unit tests); a control real-weight
    // A/B is a follow-up needing the base + control checkpoints (overlaps sc-11126 F-180).
    let Some(snap) = base_model_snapshot_opt() else {
        eprintln!("skipping: set ZIMAGE_BASE_SNAPSHOT to run the base z_image residency A/B");
        return;
    };
    let req = base_probe_request();
    let (pixels_resident, peak_resident) = render_measured_id(
        "z_image",
        spec_for(snap.clone()),
        OffloadPolicy::Resident,
        &req,
    );
    let (pixels_sequential, peak_sequential) =
        render_measured_id("z_image", spec_for(snap), OffloadPolicy::Sequential, &req);

    println!(
        "base z_image {}x{} @ {} steps (CFG):\n  Resident   peak = {:.3} GiB\n  Sequential peak = {:.3} GiB",
        req.width,
        req.height,
        req.steps.unwrap(),
        peak_resident as f64 / GIB,
        peak_sequential as f64 / GIB,
    );

    let diff = pixels_resident
        .iter()
        .zip(&pixels_sequential)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        diff,
        0,
        "base z_image Sequential residency changed the output: {diff}/{} bytes differ (must be \
         byte-identical)",
        pixels_resident.len()
    );
    assert!(
        peak_sequential < peak_resident,
        "base z_image Sequential peak {:.3} GiB was not below Resident {:.3} GiB — the text-encoder \
         drop did not reduce peak",
        peak_sequential as f64 / GIB,
        peak_resident as f64 / GIB,
    );
}
