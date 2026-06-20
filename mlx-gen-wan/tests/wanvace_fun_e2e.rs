//! Wan2.2 **VACE-Fun A14B** dual-expert end-to-end real-weight smoke (epic 3456 / sc-6604) —
//! **weight-gated** (`#[ignore]`, very heavy: loads BOTH 14B experts + 11 GB UMT5 + z16 VAE).
//!
//! The dual-expert sibling of `wanvace_e2e.rs`. Exercises the full native VACE-Fun path — UMT5 text
//! encode → 96-ch control latent (clip + mask + refs) → boundary-switched **dual-expert** VACE
//! denoise (`denoise_vace_moe`, high-noise `transformer/` ≥ boundary 0.875, low-noise `transformer_2/`
//! below) → z16-VAE decode → RGB frames — on a real assembled `wan2_2_vace_fun_14b` snapshot. Its job
//! is the **integration**: that both experts load with the right dims, the boundary swap fires, and
//! the staged pipeline yields a coherent, correctly-shaped video. (No diffusers golden — VACE-Fun has
//! no committed parity fixture; the transformer math is the same validated `WanVaceTransformer` the
//! single-expert `wanvace_real_parity.rs` locks at 14B-class dims.)
//!
//! Q4 keeps the steady-state footprint safe on a 128 GB Mac (the dense-bf16 LOAD peak ≈ both
//! transformer shard sets ≈ 70 GB, then each expert quantizes in place). Frame count is the engine's
//! documented non-causal z16 decode (n=13 → t_lat 4 → 16 frames), shared with base Wan + single-expert
//! VACE (see `wanvace_e2e.rs`).
//!
//! Snapshot: assembled by `mlx_gen_wan::convert::assemble_wan_vace_fun_snapshot` (the diffusers
//! VACE-Fun `transformer/` + `transformer_2/` + a converted base-Wan snapshot's shared
//! `t5_encoder.safetensors`/`vae.safetensors`/`tokenizer.json`). Point at an explicit assembled dir
//! with `WANVACE_FUN_DIR=/path`, or let the test assemble from the local HF cache
//! (`linoyts/Wan2.2-VACE-Fun-14B-diffusers` + a cached `SceneWorks/wan2.2-{t2v,i2v}-a14b-mlx` base).
//! Run: `cargo test -p mlx-gen-wan --release --test wanvace_fun_e2e -- --ignored --nocapture`.

use std::path::PathBuf;

use mlx_gen::{
    registry, Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, Progress, Quant,
    ReplacementMode, WeightsSource,
};
use mlx_gen_wan::convert::assemble_wan_vace_fun_snapshot;
use mlx_gen_wan::MODEL_ID_VACE_FUN;

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}

/// The one snapshot dir under a HF-cache repo's `snapshots/<hash>/`.
fn hf_snapshot(repo_dir: &str) -> Option<PathBuf> {
    let base = home()
        .join(".cache/huggingface/hub")
        .join(repo_dir)
        .join("snapshots");
    std::fs::read_dir(&base)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.is_dir())
}

/// Resolve the assembled dual-expert snapshot: `WANVACE_FUN_DIR` if set, else assemble one from the
/// local HF cache (linoyts `transformer/` + `transformer_2/`) + a cached base-Wan snapshot (the shared
/// UMT5/z16-VAE/tokenizer). Returns `None` (→ skip) when an ingredient is missing locally.
fn resolve_or_assemble_snapshot() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("WANVACE_FUN_DIR") {
        return Some(PathBuf::from(d));
    }
    let vace = hf_snapshot("models--linoyts--Wan2.2-VACE-Fun-14B-diffusers")?;
    let high = vace.join("transformer");
    let low = vace.join("transformer_2");
    if !high.join("config.json").is_file() || !low.join("config.json").is_file() {
        return None;
    }
    // Any converted base-Wan 14B snapshot carries the shared native t5/vae/tokenizer.
    let base = [
        "models--SceneWorks--wan2.2-t2v-a14b-mlx",
        "models--SceneWorks--wan2.2-i2v-a14b-mlx",
    ]
    .into_iter()
    .filter_map(hf_snapshot)
    .find(|p| p.join("t5_encoder.safetensors").is_file())?;
    let out = home().join(".cache/mlx-gen-models/wan2_2_vace_fun_14b_mlx");
    assemble_wan_vace_fun_snapshot(&out, &high, &low, &base, true).ok()
}

/// A solid mid-gray RGB frame.
fn frame(w: u32, h: u32) -> Image {
    Image {
        width: w,
        height: h,
        pixels: vec![128u8; (w * h * 3) as usize],
    }
}

/// A white (fully-active) mask frame — VACE regenerates the whole frame.
fn mask_frame(w: u32, h: u32) -> Image {
    Image {
        width: w,
        height: h,
        pixels: vec![255u8; (w * h * 3) as usize],
    }
}

#[test]
#[ignore = "needs a VACE-Fun snapshot — set WANVACE_FUN_DIR or cache linoyts/Wan2.2-VACE-Fun-14B-diffusers + a base-Wan mlx snapshot"]
fn wan_vace_fun_dual_expert_generate_smoke() {
    let Some(dir) = resolve_or_assemble_snapshot() else {
        eprintln!(
            "skipping: no VACE-Fun snapshot (set WANVACE_FUN_DIR or populate the HF cache with \
             linoyts transformers + a SceneWorks/wan2.2-*-a14b-mlx base)"
        );
        return;
    };
    eprintln!("wan_vace_fun e2e: snapshot = {}", dir.display());

    // Q4 keeps the dual-14B steady-state safe on 128 GB. The engine loads both experts, switches at
    // boundary 0.875 across the schedule, and decodes.
    let g = registry::load(
        MODEL_ID_VACE_FUN,
        &LoadSpec::new(WeightsSource::Dir(dir)).with_quant(Quant::Q4),
    )
    .expect("load wan2_2_vace_fun_14b");

    let (w, h, n) = (256u32, 256u32, 13usize); // n=13 = 1 + 4·3 → t_lat = 4
    let req = GenerationRequest {
        prompt: "a person walking through a city street".into(),
        width: w,
        height: h,
        frames: Some(n as u32),
        steps: Some(8),
        conditioning: vec![Conditioning::ControlClip {
            frames: (0..n).map(|_| frame(w, h)).collect(),
            mask: (0..n).map(|_| mask_frame(w, h)).collect(),
            masking_strength: 1.0,
            start_frame: 0,
            mode: ReplacementMode::FaceOnly,
        }],
        ..Default::default()
    };

    let mut on_progress = |_p: Progress| {};
    let out = g
        .generate(&req, &mut on_progress)
        .expect("vace-fun generate");
    let GenerationOutput::Video { frames, .. } = out else {
        panic!("expected Video output, got {out:?}");
    };

    // Engine non-causal z16 decode: t_lat = (n-1)/4 + 1, output = 4·t_lat (16 for n=13).
    let t_lat = (n - 1) / 4 + 1;
    let expected = t_lat * 4;
    assert_eq!(
        frames.len(),
        expected,
        "expected {expected} non-causal output frames"
    );
    for (i, f) in frames.iter().enumerate() {
        assert_eq!((f.width, f.height), (w, h), "frame {i} size");
        assert_eq!(f.pixels.len(), (w * h * 3) as usize, "frame {i} buffer");
    }

    // Coherence: a real decode, not a degenerate constant / pinned extreme.
    let all: Vec<u8> = frames
        .iter()
        .flat_map(|f| f.pixels.iter().copied())
        .collect();
    let mean = all.iter().map(|&b| b as f64).sum::<f64>() / all.len() as f64;
    assert!(
        (2.0..=253.0).contains(&mean),
        "decoded pixel mean {mean:.1} is pinned at an extreme — VAE decode looks degenerate"
    );
    let identical = frames.windows(2).all(|p| p[0].pixels == p[1].pixels);
    assert!(
        !identical,
        "all {} frames byte-identical — temporal decode degenerate",
        frames.len()
    );

    eprintln!(
        "wan_vace_fun dual-expert e2e ok: {} frames @ {w}x{h}, pixel mean {mean:.1}",
        frames.len()
    );
}
