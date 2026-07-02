//! Real-weight gen-core **contract conformance** for the two Bernini ids (sc-9098, F-038):
//! `bernini_renderer` and the full planner+renderer `bernini`.
//!
//! The renderer families cannot run the generic profile-based `gen_core_testkit::conformance`
//! suite — the text-only cheap request would fall back to the 81-frame video default, and the seed
//! check's three full renders are prohibitive on the ~56 GB dual-expert stack — so this mirrors
//! the request-supplied pattern the testkit provides for them (`check_*_with`, the SVD/SeedVR2/
//! scail2 shape): capability honesty + registry round-trip (cheap), plus the exact 1..=total
//! progress contract on a minimal 1-frame t2i. Typed cancellation is covered by
//! `tests/cancellation_conformance.rs`; the weights-free descriptor sweep for both ids runs by
//! default in `tests/descriptor_conformance.rs`.
//!
//! The full-pipeline progress check pins the F-038 fix: the multi-minute MAR planner stage
//! (`planning_step` = 25 steps × 3 Qwen2.5-VL-7B forwards) folds into the `Progress::Step` bar
//! ahead of the renderer denoise, and both stages report 1-based steps reaching `total`.
//!
//! `#[ignore]` because each test assembles + loads a multi-10-GB snapshot (the crate's e2e gating
//! convention; see `tests/render_real.rs` / `tests/bernini_e2e.rs` for the snapshot prerequisites).

use std::path::PathBuf;

use gen_core_testkit::{check_progress_with, check_registry_roundtrip, check_validate_honesty};
use mlx_gen::{registry, GenerationRequest, LoadSpec, WeightsSource};
use mlx_gen_wan::convert::assemble_bernini_renderer_snapshot;
// Force-link the provider (registers both `bernini_renderer` and `bernini`).
use mlx_gen_bernini as _;

fn hf_snapshot(repo: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots");
    std::fs::read_dir(snaps)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.is_dir())
}

/// Assemble the converted **renderer** snapshot once (mirrors
/// `cancellation_conformance.rs::ensure_snapshot`).
fn ensure_renderer_snapshot() -> PathBuf {
    let home = PathBuf::from(std::env::var("HOME").unwrap());
    let snapshot = home.join(".cache/mlx-gen-models/bernini_renderer_mlx_bf16");
    if !snapshot.join("high_noise_model.safetensors").is_file() {
        let pkg = hf_snapshot("ByteDance/Bernini-Diffusers")
            .expect("ByteDance/Bernini-Diffusers snapshot in the HF cache");
        let base = home.join(".cache/mlx-gen-models/wan2_2_t2v_a14b_mlx_bf16");
        assert!(
            base.join("high_noise_model.safetensors").is_file(),
            "converted base Wan2.2-T2V-A14B snapshot required at {}",
            base.display()
        );
        assemble_bernini_renderer_snapshot(&snapshot, &pkg, &base, None, true).expect("assemble");
    }
    snapshot
}

/// The combined **full-Bernini** (planner+renderer) snapshot once (mirrors
/// `bernini_e2e.rs::ensure_snapshot`).
fn ensure_full_snapshot() -> PathBuf {
    let home = PathBuf::from(std::env::var("HOME").unwrap());
    let snapshot = home.join(".cache/mlx-gen-models/bernini_full_mlx_bf16");
    let complete = snapshot.join("qwen2_5_vl.safetensors").is_file()
        && snapshot.join("high_noise_model.safetensors").is_file();
    if !complete {
        let pkg = hf_snapshot("ByteDance/Bernini-Diffusers")
            .expect("ByteDance/Bernini-Diffusers snapshot in the HF cache");
        let base = home.join(".cache/mlx-gen-models/wan2_2_t2v_a14b_mlx_bf16");
        assert!(
            base.join("high_noise_model.safetensors").is_file(),
            "converted base Wan2.2-T2V-A14B snapshot required at {}",
            base.display()
        );
        mlx_gen_bernini::convert::assemble_bernini_snapshot(&snapshot, &pkg, &base, true)
            .expect("assemble full snapshot");
    }
    snapshot
}

/// Minimal valid 1-frame t2i request (the `render_real.rs`/`bernini_e2e.rs` smoke shape) with the
/// given denoise step count.
fn t2i_request(mode: &str, steps: u32) -> GenerationRequest {
    GenerationRequest {
        prompt: "a red apple on a wooden table, studio lighting".into(),
        width: 256,
        height: 256,
        frames: Some(1),
        steps: Some(steps),
        seed: Some(0),
        video_mode: Some(mode.into()),
        ..Default::default()
    }
}

#[test]
#[ignore = "real weights: assembles + loads the ~56 GB Bernini renderer snapshot, runs a short denoise"]
fn bernini_renderer_satisfies_gen_core_contract() {
    let gen = registry::load(
        "bernini_renderer",
        &LoadSpec::new(WeightsSource::Dir(ensure_renderer_snapshot())),
    )
    .expect("load bernini_renderer");
    let g = gen.as_ref();

    // Cheap (validate-only) capability honesty + registry round-trip.
    check_validate_honesty(g, &gen_core_testkit::Profile::cheap()).expect("validate honesty");
    check_registry_roundtrip(g).expect("registry round-trip");

    // Exact 1..=total progress on a 3-step 1-frame t2i (the F-038 1-based fix: the renderer's
    // Step.current must reach total). total == req.steps for the renderer-only pipeline.
    check_progress_with(g, &t2i_request("t2v_apg", 3), Some(3)).expect("progress contract");
}

#[test]
#[ignore = "real weights: assembles + loads the full Bernini (planner+renderer) snapshot, runs the MAR loop + denoise"]
fn bernini_full_pipeline_satisfies_gen_core_contract() {
    let gen = registry::load(
        "bernini",
        &LoadSpec::new(WeightsSource::Dir(ensure_full_snapshot())),
    )
    .expect("load bernini");
    let g = gen.as_ref();

    check_validate_honesty(g, &gen_core_testkit::Profile::cheap()).expect("validate honesty");
    check_registry_roundtrip(g).expect("registry round-trip");

    // Typed cancellation, tripped at the first *planner* Step — proves the MAR stage is both
    // progress-visible and cancellable without ever reaching the heavy renderer denoise.
    let req = t2i_request("t2i", 2);
    gen_core_testkit::check_cancellation_with(g, &req).expect("typed cancellation (planner stage)");

    // Folded progress bar (F-038): planning_step (25) MAR steps + 2 renderer steps, exactly
    // 1..=27 with a constant total. Update the expected total if `FullDefaults::PLANNING_STEP`
    // ever changes.
    check_progress_with(g, &t2i_request("t2i", 2), Some(25 + 2)).expect("progress contract");
}
