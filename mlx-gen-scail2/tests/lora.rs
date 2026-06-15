//! SCAIL-2 inference LoRA gates (sc-5451 residual path + sc-5684 diff-patch / lightning path).
//!
//! Real-weight `#[ignore]` tests (they need the assembled snapshot and, for the apply smokes, a LoRA
//! file):
//!   * `adaptable_paths_resolve_on_real_dit` — load the real DiT and assert every
//!     [`AdaptableHost::adaptable_paths`] entry resolves through [`AdaptableHost::adaptable_mut`].
//!     This is the drift guard the trait contract requires (the CI unit tests in `model.rs` cover the
//!     path-set shape; this proves the paths actually address a Linear on the real model).
//!   * `lora_apply_smoke` — install a LoRA (`SCAIL2_LORA`) onto the (optionally Q4) base and run the
//!     full pipeline. Works for BOTH a pure low-rank file (the residual path) AND the raw lightx2v
//!     **diff-patch** lightning file (merged in place into the dense weights, sc-5684). Driven by env:
//!       - lightning recipe: the defaults — `SCAIL2_LORA_GUIDE=1.0` (CFG **off**, single DiT
//!         forward/step), `SCAIL2_LORA_STEPS=8`, `SCAIL2_LORA_SHIFT=1.0`.
//!       - Bias-Aware DPO refinement: run with `SCAIL2_LORA_GUIDE=5.0` (CFG on). NOTE the DPO LoRA
//!         ships as a torch pickle (`bias-aware-dpo-lora.pt`) — point `SCAIL2_LORA` at a safetensors
//!         conversion (the snapshot-assembly / SceneWorks-bundling step), since the loader reads
//!         safetensors.
//!   * `diff_patch_merge_report` — merge the raw lightx2v file (`SCAIL2_DIFF_PATCH_LORA`) into the
//!     real dense DiT weight map and assert exactly the right targets merged and that the in_dim-36
//!     vanilla-Wan `patch_embedding` is the lone cross-architecture skip (the precise sc-5684 contract,
//!     without a full generate).
//!   * `diff_patch_rejected_on_prequantized` — the same file against a *pre-quantized-on-disk* snapshot
//!     (`SCAIL2_Q4_SNAPSHOT_DIR`) must error loudly (the dense delta can't fold into packed u32 weights).
//!
//! Run on macOS against the assembled snapshot:
//! ```text
//! SCAIL2_SNAPSHOT_DIR=~/.cache/scail2-mlx-convert \
//! SCAIL2_DIFF_PATCH_LORA="$HOME/.cache/huggingface/hub/models--lightx2v--Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v/snapshots/*/loras/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors" \
//!   cargo test -p mlx-gen-scail2 --test lora -- --ignored --nocapture
//! # full lightning e2e: SCAIL2_LORA=<same file> SCAIL2_LORA_QUANT=bf16 … lora_apply_smoke
//! ```

use std::path::PathBuf;

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterKind, AdapterSpec, Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec,
    Quant, ReplacementMode, WeightsSource,
};
use mlx_gen_scail2::pipeline::MODEL_ID;
use mlx_gen_scail2::{Scail2Config, Scail2Dit};

fn snapshot_dir() -> PathBuf {
    std::env::var("SCAIL2_SNAPSHOT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/scail2-mlx-convert")
        })
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// A deterministic gradient image (stands in for a real reference / driving frame).
fn gradient(w: usize, h: usize, phase: usize) -> Image {
    let mut pixels = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            pixels.extend_from_slice(&[
                ((x + phase) % 256) as u8,
                ((y + phase) % 256) as u8,
                ((x + y + phase) % 256) as u8,
            ]);
        }
    }
    Image {
        width: w as u32,
        height: h as u32,
        pixels,
    }
}

/// A two-region color-coded mask (left white, right red) with a moving split.
fn color_mask(w: usize, h: usize, split: usize) -> Image {
    let mut pixels = Vec::with_capacity(w * h * 3);
    for _y in 0..h {
        for x in 0..w {
            let rgb = if x < split {
                [255u8, 255, 255]
            } else {
                [255u8, 0, 0]
            };
            pixels.extend_from_slice(&rgb);
        }
    }
    Image {
        width: w as u32,
        height: h as u32,
        pixels,
    }
}

/// Load the real DiT and prove every advertised adaptable path resolves to a Linear (the trait
/// contract `adaptable_paths` ⊆ resolvable-by-`adaptable_mut`; works on both the bf16 and the Q4
/// packed snapshot — the Linear exists either way).
#[test]
#[ignore = "real snapshot; run with --ignored on macOS (see module doc)"]
fn adaptable_paths_resolve_on_real_dit() {
    let root = snapshot_dir();
    let dit_path = root.join("dit.safetensors");
    assert!(
        dit_path.exists(),
        "missing snapshot at {} — assemble it first (sc-5445)",
        root.display()
    );
    let cfg = Scail2Config::from_model_dir(&root).unwrap();
    let w = Weights::from_file(&dit_path).unwrap();
    let mut dit = Scail2Dit::from_weights(&w, &cfg).unwrap();

    let paths = dit.adaptable_paths();
    assert_eq!(paths.len(), 11 + cfg.wan.num_layers * 12);
    for p in &paths {
        let parts: Vec<&str> = p.split('.').collect();
        assert!(
            dit.adaptable_mut(&parts).is_some(),
            "advertised adaptable path `{p}` does not resolve to a Linear"
        );
    }
    println!("resolved {} adaptable SCAIL-2 LoRA targets", paths.len());
}

/// Install a real LoRA and run the full pipeline. `SCAIL2_LORA` is the safetensors path; the recipe
/// knobs default to the lightx2v lightning recipe (CFG off, 8 steps, shift 1.0). Skips cleanly when
/// `SCAIL2_LORA` is unset so `--ignored` runs without the optional file don't fail.
#[test]
#[ignore = "real snapshot + a LoRA file; run with --ignored on macOS (see module doc)"]
fn lora_apply_smoke() {
    let Ok(lora) = std::env::var("SCAIL2_LORA") else {
        eprintln!("SCAIL2_LORA unset — skipping the LoRA apply smoke");
        return;
    };
    let lora_path = PathBuf::from(&lora);
    assert!(lora_path.exists(), "SCAIL2_LORA not found: {lora}");

    let size = env_usize("SCAIL2_SMOKE_SIZE", 256);
    let w = env_usize("SCAIL2_SMOKE_W", size);
    let h = env_usize("SCAIL2_SMOKE_H", size);
    let n_frames = env_usize("SCAIL2_SMOKE_FRAMES", 13);
    // Lightning defaults (lightx2v cfg-step-distill): CFG off, few steps, shift 1.0.
    let guide = env_f32("SCAIL2_LORA_GUIDE", 1.0);
    let steps = env_usize("SCAIL2_LORA_STEPS", 8);
    let shift = env_f32("SCAIL2_LORA_SHIFT", 1.0);
    let scale = env_f32("SCAIL2_LORA_ALPHA", 1.0);
    // Default Q4 (the SceneWorks worker default) — proves LoRA-over-quantized composition (sc-5445).
    let quant = match std::env::var("SCAIL2_LORA_QUANT").as_deref() {
        Ok("q8") | Ok("Q8") => Some(Quant::Q8),
        Ok("bf16") | Ok("none") => None,
        _ => Some(Quant::Q4),
    };

    let root = snapshot_dir();
    assert!(
        root.join("dit.safetensors").exists(),
        "missing snapshot at {} — assemble it first (sc-5445)",
        root.display()
    );

    let reference = gradient(w, h, 0);
    let ref_mask = color_mask(w, h, w / 2);
    let driving: Vec<Image> = (0..n_frames).map(|i| gradient(w, h, i * 7)).collect();
    let masks: Vec<Image> = (0..n_frames)
        .map(|i| color_mask(w, h, w / 4 + (i % (w / 2))))
        .collect();

    let req = GenerationRequest {
        prompt: "a person dancing, cinematic".into(),
        negative_prompt: Some("blurry, low quality".into()),
        width: w as u32,
        height: h as u32,
        steps: Some(steps as u32),
        guidance: Some(guide),
        scheduler_shift: Some(shift),
        seed: Some(7),
        fps: Some(16),
        conditioning: vec![
            Conditioning::Reference {
                image: reference,
                strength: None,
            },
            Conditioning::Mask { image: ref_mask },
            Conditioning::ControlClip {
                frames: driving,
                mask: masks,
                masking_strength: 1.0,
                start_frame: 0,
                mode: ReplacementMode::default(),
            },
        ],
        ..Default::default()
    };

    let mut spec = LoadSpec::new(WeightsSource::Dir(root)).with_adapters(vec![AdapterSpec::new(
        lora_path,
        scale,
        AdapterKind::Lora,
    )]);
    if let Some(q) = quant {
        spec = spec.with_quant(q);
    }
    let gen = mlx_gen::registry::load(MODEL_ID, &spec).expect("load scail2 provider with LoRA");

    let mut last_step = 0u32;
    let out = gen
        .generate(&req, &mut |p| {
            if let mlx_gen::Progress::Step { current, .. } = p {
                last_step = current;
            }
        })
        .expect("generate with LoRA must succeed");

    let GenerationOutput::Video { frames, .. } = out else {
        panic!("expected a Video output");
    };
    assert!(!frames.is_empty(), "no frames produced");
    assert_eq!(last_step as usize, steps, "all denoise steps ran");
    let (mut lo, mut hi) = (255u8, 0u8);
    for f in &frames {
        for &p in &f.pixels {
            lo = lo.min(p);
            hi = hi.max(p);
        }
    }
    assert!(
        hi > lo,
        "decoded video is a single flat value ({lo}) — dead pipeline"
    );
    println!(
        "LoRA smoke: {} frames @ {w}x{h}, guide {guide}, {steps} steps, shift {shift}, byte range [{lo},{hi}]",
        frames.len()
    );
}

/// Merge the **real** lightx2v diff-patch lightning LoRA (`SCAIL2_DIFF_PATCH_LORA`) into the real
/// dense DiT weight map and assert the precise sc-5684 contract: every compatible dim-5120 target
/// (the 40 blocks' projections + their qk/`norm3` norm `.diff`s + the dim-5120 globals + `img_emb` +
/// `head.head`) merges, and the in_dim-36 vanilla-Wan `patch_embedding` is the lone cross-architecture
/// skip. The merge builds lazy graphs only (no 32 GB materialization), so this is cheap. Skips when the
/// env var is unset.
#[test]
#[ignore = "needs the raw lightx2v diff-patch LoRA file + dense snapshot; run with --ignored on macOS"]
fn diff_patch_merge_report() {
    let Ok(lora) = std::env::var("SCAIL2_DIFF_PATCH_LORA") else {
        eprintln!("SCAIL2_DIFF_PATCH_LORA unset — skipping the diff-patch merge test");
        return;
    };
    let lora_path = PathBuf::from(&lora);
    assert!(
        lora_path.exists(),
        "SCAIL2_DIFF_PATCH_LORA not found: {lora}"
    );
    assert!(
        mlx_gen_scail2::has_diff_patch_keys(&lora_path).unwrap(),
        "the file must be a diff-patch (.diff/.diff_b) LoRA"
    );

    let root = snapshot_dir();
    let dit_path = root.join("dit.safetensors");
    assert!(
        dit_path.exists(),
        "missing dense snapshot at {}",
        root.display()
    );
    let mut w = Weights::from_file(&dit_path).unwrap();
    let spec = AdapterSpec::new(lora_path, 1.0, AdapterKind::Lora);
    let report = mlx_gen_scail2::merge_diff_patch_adapters(&mut w, &[&spec]).unwrap();
    mlx_gen_scail2::lora::report_outcome(&report, MODEL_ID).expect("matched real SCAIL-2 modules");

    println!(
        "diff-patch merge: {} weights, {} biases merged; cross-arch skips {:?}; unmatched {:?}",
        report.merged_weights,
        report.merged_biases,
        report.skipped_cross_arch,
        report.skipped_unmatched
    );
    // 40 blocks × (12 lora projections + 6 norm `.diff`s = 18) = 720, + dim-5120 globals with a weight
    // delta (text_embedding.0/.2, time_embedding.0/.2, time_projection.1, img_emb.proj.1/.3 = lora;
    // img_emb.proj.0/.4 = diff; head.head = diff) = 10 → 730.
    assert_eq!(
        report.merged_weights, 730,
        "every compatible weight delta merges"
    );
    // Every `.diff_b` except patch_embedding's (531 total − 1 skipped with the cross-arch module) = 530.
    assert_eq!(
        report.merged_biases, 530,
        "every compatible bias delta merges"
    );
    assert_eq!(
        report.skipped_cross_arch,
        vec!["patch_embedding".to_string()],
        "the in_dim-36 vanilla-Wan patch_embedding is the only cross-architecture skip"
    );
    assert!(
        report.skipped_unmatched.is_empty(),
        "no lightning target should be unmatched"
    );
}

/// The same diff-patch file against a **pre-quantized-on-disk** snapshot (`SCAIL2_Q4_SNAPSHOT_DIR`,
/// e.g. `~/.cache/scail2-mlx-q4`) must error loudly: a dense delta can't fold into packed u32 weights,
/// so the loader directs the user to the dense bf16 snapshot rather than silently dropping the patch.
/// Skips when the env var is unset.
#[test]
#[ignore = "needs the raw lightx2v file + a pre-quantized snapshot; run with --ignored on macOS"]
fn diff_patch_rejected_on_prequantized() {
    let (Ok(lora), Ok(q4_dir)) = (
        std::env::var("SCAIL2_DIFF_PATCH_LORA"),
        std::env::var("SCAIL2_Q4_SNAPSHOT_DIR"),
    ) else {
        eprintln!("SCAIL2_DIFF_PATCH_LORA / SCAIL2_Q4_SNAPSHOT_DIR unset — skipping");
        return;
    };
    let (lora_path, root) = (PathBuf::from(&lora), PathBuf::from(&q4_dir));
    let (w, h, n) = (64usize, 64usize, 5usize);
    let req = GenerationRequest {
        prompt: "x".into(),
        width: w as u32,
        height: h as u32,
        steps: Some(2),
        conditioning: vec![
            Conditioning::Reference {
                image: gradient(w, h, 0),
                strength: None,
            },
            Conditioning::Mask {
                image: color_mask(w, h, w / 2),
            },
            Conditioning::ControlClip {
                frames: (0..n).map(|i| gradient(w, h, i)).collect(),
                mask: (0..n).map(|i| color_mask(w, h, w / 4 + i)).collect(),
                masking_strength: 1.0,
                start_frame: 0,
                mode: ReplacementMode::default(),
            },
        ],
        ..Default::default()
    };
    let spec = LoadSpec::new(WeightsSource::Dir(root)).with_adapters(vec![AdapterSpec::new(
        lora_path,
        1.0,
        AdapterKind::Lora,
    )]);
    let gen = mlx_gen::registry::load(MODEL_ID, &spec).expect("load scail2 provider");
    let err = gen
        .generate(&req, &mut |_| {})
        .expect_err("a diff-patch LoRA on a pre-quantized snapshot must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("DENSE") || msg.contains("dense"),
        "expected a dense-snapshot-required error, got: {msg}"
    );
    println!("diff-patch on pre-quantized snapshot correctly rejected: {msg}");
}
