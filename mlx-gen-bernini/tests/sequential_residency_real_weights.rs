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
    let snapshot = ensure_snapshot();
    let model =
        mlx_gen_bernini::bernini::load(&LoadSpec::new(WeightsSource::Dir(snapshot.clone())))
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

    // Self-calibrating tripwire (sc-10840). The old fixed 72 GiB ceiling sat BETWEEN the ~56 GiB clean
    // staged peak and the ~80 GiB whole-model sum, so losing ONE of the two `clear_cache()` flushes —
    // which re-admits the ~11 GiB T5 into the expert phase (~67 GiB) — still passed (false-green). Derive
    // the bound from the real on-disk expert bytes instead: a clean run peaks at the two co-resident bf16
    // experts (+ z16 VAE) because BOTH encoders (planner Qwen2.5-VL ~15 GiB, UMT5-XXL T5 ~11 GiB) are
    // dropped + `clear_cache()`d before the experts load. A lost flush lingers ~11-15 GiB of encoder into
    // that phase and blows past `experts + VAE + HEADROOM`, which sits well below a single-flush loss.
    let file_gib = |name: &str| {
        std::fs::metadata(snapshot.join(name))
            .map(|m| m.len() as f64 / GIB)
            .unwrap_or(0.0)
    };
    let expert_phase_gib = file_gib("low_noise_model.safetensors")
        + file_gib("high_noise_model.safetensors")
        + file_gib("vae.safetensors");
    // Denoise activations for the tiny 256² × 1-frame × 4-step run are well under this headroom; the
    // point is to sit below `expert_phase + smaller_encoder (T5 ~11 GiB)` so a single lost flush trips.
    const HEADROOM_GIB: f64 = 6.0;
    let ceiling = expert_phase_gib + HEADROOM_GIB;
    println!(
        "Bernini full t2i 256² @ 4 steps: staged peak = {:.3} GiB (ceiling {:.3} GiB = experts+VAE \
         {:.3} + {:.1} headroom)",
        peak as f64 / GIB,
        ceiling,
        expert_phase_gib,
        HEADROOM_GIB,
    );
    assert!(
        (peak as f64 / GIB) < ceiling,
        "staged peak {:.3} GiB exceeded experts+VAE + {:.0} GiB headroom ({:.3} GiB) — an encoder drop \
         / clear_cache regressed and a freed encoder lingered into the expert phase",
        peak as f64 / GIB,
        HEADROOM_GIB,
        ceiling,
    );
    drop(model);
    clear_cache();
}
