//! sc-10840 (epic 10834): the `Sequential` component-residency A/B on real Anima weights.
//!
//! `#[ignore]`d — needs the licensed `circlestone-labs/Anima` snapshot in the HF cache (its
//! `split_files/` tree) and Metal. Run:
//!   cargo test -p mlx-gen-anima --release --test sequential_residency_real_weights -- --ignored --nocapture
//!
//! Same two claims as the SD3 / SANA / Lens A/Bs: (1) `Sequential` peaks LOWER than `Resident` because
//! the Qwen3-0.6B text encoder is dropped (+ `clear_cache()`) before the DiT + bundled conditioner +
//! VAE materialize, and (2) the output is BYTE-IDENTICAL. A repeat-job check confirms nothing stays
//! resident across jobs. `ANIMA_SEQ_STEPS` / `ANIMA_SEQ_SIZE` tune the probe.

use std::path::PathBuf;

use mlx_gen::{GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, WeightsSource};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

// Force-link the provider crate so its `inventory` generator registration runs.
use mlx_gen_anima as _;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Resolve the Anima `split_files/` dir from `ANIMA_SNAPSHOT`, else the HF hub cache (no hardcoded sha).
fn split_files() -> PathBuf {
    if let Ok(p) = std::env::var("ANIMA_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").expect("HOME");
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--circlestone-labs--Anima/snapshots");
    std::fs::read_dir(&base)
        .unwrap_or_else(|_| panic!("no Anima snapshots under {base:?}"))
        .filter_map(|e| e.ok())
        .find_map(|e| {
            let p = e.path().join("split_files");
            p.join("diffusion_models").is_dir().then_some(p)
        })
        .expect("set ANIMA_SNAPSHOT or populate the HF hub cache")
}

fn probe_request() -> GenerationRequest {
    // CFG (base variant) to exercise the seam's cond+uncond conditioner-input materialize/drop path. A
    // fixed seed makes the byte-identity assertion meaningful; quality is irrelevant here.
    let size = env_u32("ANIMA_SEQ_SIZE", 768);
    GenerationRequest {
        prompt: "an anime girl with silver hair, detailed, masterpiece".into(),
        negative_prompt: Some("blurry, low quality".into()),
        guidance: Some(4.5),
        width: size,
        height: size,
        seed: Some(1234),
        steps: Some(env_u32("ANIMA_SEQ_STEPS", 8)),
        ..Default::default()
    }
}

fn render_measured(policy: OffloadPolicy, req: &GenerationRequest) -> (Vec<u8>, usize) {
    let spec = LoadSpec::new(WeightsSource::Dir(split_files())).with_offload_policy(policy);
    let model = mlx_gen::load("anima_base", &spec).expect("load anima_base");
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
#[ignore = "needs a real circlestone-labs/Anima snapshot (ANIMA_SNAPSHOT or the HF cache)"]
fn sequential_bounds_peak_and_is_byte_identical() {
    let req = probe_request();
    let (pixels_resident, peak_resident) = render_measured(OffloadPolicy::Resident, &req);
    let (pixels_sequential, peak_sequential) = render_measured(OffloadPolicy::Sequential, &req);

    println!(
        "Anima-base {}x{} @ {} steps:\n  Resident   peak = {:.3} GiB\n  Sequential peak = {:.3} GiB\n  saved = {:.3} GiB ({:.1}%)",
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
        "Sequential peak {:.3} GiB was not below Resident {:.3} GiB — the Qwen3-TE drop did not \
         reduce peak",
        peak_sequential as f64 / GIB,
        peak_resident as f64 / GIB,
    );
}

#[test]
#[ignore = "needs a real circlestone-labs/Anima snapshot (ANIMA_SNAPSHOT or the HF cache)"]
fn sequential_repeat_job_stays_bounded() {
    let req = probe_request();
    let (_p1, peak1) = render_measured(OffloadPolicy::Sequential, &req);
    let (_p2, peak2) = render_measured(OffloadPolicy::Sequential, &req);
    println!(
        "Anima-base Sequential repeat-job peaks: job1 = {:.3} GiB, job2 = {:.3} GiB",
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
