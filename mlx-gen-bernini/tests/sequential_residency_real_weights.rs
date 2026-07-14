//! sc-10840 (epic 10834): Bernini's staged-residency peak scaffold on real weights.
//!
//! `#[ignore]`d — assembles + loads the full ~56 GB Bernini snapshot (see `bernini_e2e.rs`). Run:
//!   cargo test -p mlx-gen-bernini --release --test sequential_residency_real_weights -- --ignored --nocapture
//!
//! **Why no Resident-vs-Sequential A/B.** Unlike the image engines wired onto the two-phase
//! [`mlx_gen::Residency`] seam (SD3 / Qwen-Image / Boogu), Bernini is **structurally always-staged**:
//! its generator holds NO component weights, and `generate_impl` loads per generate in phase order —
//! planner (Qwen2.5-VL-7B) → drop → UMT5-XXL T5 → drop → the two co-resident MoE experts + z16 VAE —
//! dropping BOTH encoders (+ `clear_cache()`, sc-10840) before the experts load. There is no
//! Resident-warm mode to toggle, so there is no A/B baseline to compare against. What sc-10840 added is
//! the `clear_cache()` discipline at the two encoder-drop boundaries, which is **output-neutral** (it
//! only returns freed buffer-cache pages to the OS) — the coherence smokes in `bernini_e2e.rs` already
//! guard the output. This scaffold measures the staged peak and asserts it stays well below the naive
//! whole-model resident sum (planner + T5 + both experts + VAE), i.e. the encoders really did free
//! before the experts.

use std::path::PathBuf;

use mlx_gen::media::Image;
use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};
use mlx_gen_bernini::convert::assemble_bernini_snapshot;
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

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

/// Assemble the combined full-Bernini snapshot once (reused across reruns), returning its dir.
fn ensure_snapshot() -> PathBuf {
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
        assemble_bernini_snapshot(&snapshot, &pkg, &base, true).expect("assemble full snapshot");
    }
    snapshot
}

#[test]
#[ignore = "real weights: assembles + loads the ~56 GB full Bernini snapshot, runs a staged denoise"]
fn staged_peak_bounds_below_whole_model_sum() {
    let model =
        mlx_gen_bernini::bernini::load(&LoadSpec::new(WeightsSource::Dir(ensure_snapshot())))
            .expect("load bernini");
    // Tiny t2i (1 frame, 256², 4 steps) — the whole staged stack: planner load + MAR loop + drop +
    // clear_cache → T5 encode + drop + clear_cache → two experts + APG denoise → VAE decode.
    let req = GenerationRequest {
        prompt: "a red apple on a wooden table, studio lighting".into(),
        width: 256,
        height: 256,
        frames: Some(1),
        steps: Some(4),
        seed: Some(0),
        video_mode: Some("t2i".into()),
        ..Default::default()
    };
    reset_peak_memory();
    let out = model.generate(&req, &mut |_| {}).expect("generate");
    let peak = get_peak_memory();
    let img = match out {
        GenerationOutput::Images(mut v) => {
            assert_eq!(v.len(), 1, "1-frame t2i yields one image");
            v.pop().unwrap()
        }
        GenerationOutput::Video { .. } => panic!("expected Images for a 1-frame request"),
    };
    // Output stays coherent (the sc-10840 clear_cache calls are memory-only, not compute).
    let Image {
        width,
        height,
        pixels,
    } = &img;
    assert_eq!((*width, *height), (256, 256));
    assert!(
        pixels.iter().any(|&p| p != 0) && pixels.iter().any(|&p| p != 255),
        "decoded image must not be uniformly black/white"
    );

    // Bernini bf16 whole-model resident sum ≈ planner(~15) + T5(~11) + 2 experts(~56) + VAE ≈ 80+ GiB.
    // The staged peak is dominated by the two-expert phase (~56 GiB bf16) because the encoders freed
    // (+ clear_cache) before the experts loaded, so it must sit well under the naive sum. A generous
    // 72 GiB ceiling makes this a regression tripwire (a lost drop/flush would blow past it), not a
    // tight fit. Prints the measured peak for the record.
    println!(
        "Bernini full t2i 256² @ 4 steps: staged peak = {:.3} GiB",
        peak as f64 / GIB,
    );
    assert!(
        (peak as f64 / GIB) < 72.0,
        "staged peak {:.3} GiB approached the whole-model resident sum — an encoder drop / \
         clear_cache regressed",
        peak as f64 / GIB,
    );
    drop(model);
    clear_cache();
}
