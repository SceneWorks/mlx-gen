//! sc-10977 (epic 10975 — MLX **video**-lane sequential component residency): the residency proof for
//! SCAIL-2 (Wan2.1-14B I2V character animation) on real weights.
//!
//! **What was already true.** Unlike the image lane (and unlike LTX's pre-sc-10976 resident struct),
//! SCAIL-2 has staged its heavy components since the original pipeline commit (sc-5443): the `Scail2`
//! provider is `root`-only, and `crate::generate` loads the UMT5-XXL text encoder and the open-CLIP
//! tower in scoped blocks (load → encode → `eval` → drop at scope exit) BEFORE the `Scail2Dit`
//! materializes. So the ~11 GB UMT5 and the DiT never co-reside in active memory. This is the exact
//! body of the story's named template, `mlx_gen_wan::text_encoder::encode_text_staged`.
//!
//! **So sc-10977 is TEST-ONLY** — this regression gate; no `generate.rs` change. The story's suggested
//! `clear_cache()` after the TE/CLIP drops was implemented and then **measured to be a no-op** (bf16
//! tier, staged peak 35.05 GiB with vs 35.05 GiB without): `get_peak_memory` tracks ACTIVE memory, and
//! the DiT weight allocation reuses the freed UMT5/CLIP buffer cache, so clearing it early changes
//! nothing. Only a phase that allocates FRESH scratch it can't reuse benefits — the VAE decode
//! (`generate.rs`, sc-5681) — not a weight load, so it was dropped per minimum-code.
//!
//! **Why this is NOT the image-lane `OffloadPolicy::Resident`↔`Sequential` A/B.** Per epic 10975 the
//! video lane stages **unconditionally** (Wan-style, no `offload_policy`/fit-gate). There is no
//! production "Resident" mode to flip, so — as with the LTX test — this bounds the staged `generate`
//! peak below a **co-residence estimate** = (measured UMT5 resident peak) + (the DiT's on-disk
//! `dit.safetensors` bytes). A naive impl that held the UMT5 resident through the denoise would peak at
//! ≥ that estimate; the staged path holds at most one giant at a time. This is the residency **regression
//! gate**: it fails loudly if a future refactor lets the UMT5 leak into the DiT phase.
//!
//! **The SCAIL-2 picture** (measured on the `SceneWorks/scail2-mlx` root = the Q4 tier: `config.json`
//! `quantization.bits = 4`, so the DiT builds packed with NO dense bf16 transient — `config.rs`): the
//! UMT5-XXL is ~11 GB, the Q4 DiT ~8.9 GB on disk. The UMT5 is the LARGER phase (like LTX), so the staged
//! peak floor is the **text phase**, and the win is dropping the ~9 GB Q4 DiT out of co-residence with
//! the UMT5. (A bf16-DiT tier would invert this — there the ~28 GB dense DiT is the peak and the ~11 GB
//! UMT5 is the drop; the invariant `staged < UMT5 + DiT` holds either way.)
//!
//! **Output correctness** is owned by the existing parity gates (`dit_parity`/`dit_real_parity`,
//! `clip_parity`, `mask_parity`, `generate_smoke`): staging changes only WHEN each component is
//! built/freed, not the encode/denoise/decode math. This test owns the MEMORY invariant + a
//! non-degenerate-output sanity check.
//!
//! `#[ignore]`d — needs the real snapshot. Defaults to the HF cache `SceneWorks/scail2-mlx` (the root
//! snapshot dir = the Q4 tier); override with `SCAIL2_MODEL_DIR`. Run: `cargo test -p mlx-gen-scail2
//! --release --test sequential_residency_real_weights -- --ignored --nocapture`.

// Force-link the provider crate so its `inventory` generator registration runs.
use mlx_gen_scail2 as _;

use std::path::{Path, PathBuf};

use mlx_gen::weights::Weights;
use mlx_gen::{
    Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, ReplacementMode,
    WeightsSource,
};
use mlx_gen_scail2::config::Scail2Config;
use mlx_gen_scail2::pipeline::MODEL_ID;
use mlx_gen_wan::{load_tokenizer, Umt5Encoder};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap())
}

/// First snapshot dir under an HF-cache `models--…` entry.
fn hf_snapshot(model: &str) -> Option<PathBuf> {
    let snaps = home()
        .join(".cache/huggingface/hub")
        .join(model)
        .join("snapshots");
    std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
}

/// The SCAIL-2 model dir — `SCAIL2_MODEL_DIR`, else the HF-cache `SceneWorks/scail2-mlx` root snapshot
/// (which ships the Q4-packed `dit.safetensors` + `config.json` with `quantization.bits = 4`).
fn model_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("SCAIL2_MODEL_DIR") {
        return Some(PathBuf::from(d));
    }
    hf_snapshot("models--SceneWorks--scail2-mlx")
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// A deterministic gradient image (stands in for a real reference / driving frame). Mirrors
/// `generate_smoke.rs` so the synthetic job drives the same preprocessing path.
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

/// A two-region color-coded mask (left white / right red) with a moving split.
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

/// A small deterministic single-character animation job: gradient reference + 2-color ref mask + a
/// short driving clip with per-frame color masks. `n_frames` stays in one VAE-aligned window
/// (`1 + 4·k`), so the whole staged text → denoise → decode path runs without multi-segment history.
fn request(w: usize, h: usize, n_frames: usize, steps: usize) -> GenerationRequest {
    let reference = gradient(w, h, 0);
    let ref_mask = color_mask(w, h, w / 2);
    let driving: Vec<Image> = (0..n_frames).map(|i| gradient(w, h, i * 7)).collect();
    let masks: Vec<Image> = (0..n_frames)
        .map(|i| color_mask(w, h, w / 4 + (i % (w / 2))))
        .collect();
    GenerationRequest {
        prompt: "a person dancing, cinematic".into(),
        negative_prompt: Some("blurry, low quality".into()),
        width: w as u32,
        height: h as u32,
        steps: Some(steps as u32),
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
    }
}

/// Measure the resident footprint of the UMT5-XXL text phase ALONE, exactly as `crate::generate`'s
/// scoped block builds it: load the tokenizer + raw weights, then (after resetting the peak so this
/// captures the encoder's materialization + encode) build `Umt5Encoder`, run a real `encode`, and
/// `eval` so every layer's weights are forced resident. Returns the peak bytes.
fn umt5_resident_peak(root: &Path) -> usize {
    let cfg = Scail2Config::from_model_dir(root).expect("Scail2Config::from_model_dir");
    let tok =
        load_tokenizer(root.join("tokenizer.json"), cfg.wan.text_len).expect("load_tokenizer");
    let w = Weights::from_file(root.join("t5_encoder.safetensors")).expect("t5_encoder weights");
    reset_peak_memory();
    let enc = Umt5Encoder::from_weights(&w, &cfg.wan).expect("build Umt5Encoder");
    let context = enc
        .encode(&tok, "a person dancing, cinematic")
        .expect("encode");
    mlx_rs::transforms::eval([&context]).expect("eval");
    let peak = get_peak_memory();
    drop(enc);
    drop(w);
    clear_cache();
    peak
}

/// Run the real staged `generate` through the registered provider, returning the video frames + the
/// process peak unified memory. No `LoadSpec::quantize` — the root `config.json` already declares Q4,
/// so the DiT loads packed off disk (the production residency path).
fn staged_generate(root: &Path, req: &GenerationRequest) -> (Vec<Image>, usize) {
    let spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf()));
    let gen = mlx_gen::registry::load(MODEL_ID, &spec).expect("load scail2 provider");
    reset_peak_memory();
    let out = gen
        .generate(req, &mut |_| {})
        .expect("generate must succeed");
    let peak = get_peak_memory();
    let GenerationOutput::Video { frames, .. } = out else {
        panic!("expected a Video output");
    };
    drop(gen);
    clear_cache();
    (frames, peak)
}

#[test]
#[ignore = "needs SceneWorks/scail2-mlx (SCAIL2_MODEL_DIR or the HF cache); ~13 GB+ unified memory"]
fn scail2_staged_peak_stays_below_umt5_plus_dit_coresidence() {
    let Some(root) = model_dir() else {
        eprintln!("skip: no SCAIL2_MODEL_DIR and no SceneWorks/scail2-mlx in the HF cache");
        return;
    };
    if !root.join("dit.safetensors").exists() || !root.join("t5_encoder.safetensors").exists() {
        eprintln!(
            "skip: missing dit.safetensors / t5_encoder.safetensors under {}",
            root.display()
        );
        return;
    }

    let w = env_usize("SCAIL2_RES_SIZE", 256);
    let h = w;
    let n_frames = env_usize("SCAIL2_RES_FRAMES", 9); // 1 + 4·2 → one VAE-aligned window
    let steps = env_usize("SCAIL2_RES_STEPS", 6);
    let req = request(w, h, n_frames, steps);

    // The DiT's resident weight proxy: the on-disk `dit.safetensors` bytes (follow symlink).
    let dit_bytes = std::fs::metadata(root.join("dit.safetensors"))
        .expect("stat dit.safetensors")
        .len() as usize;

    // Staged production path first, then the UMT5-alone resident peak (each brackets its own
    // reset/clear so neither inflates the other).
    let (frames, staged_peak) = staged_generate(&root, &req);
    let umt5_peak = umt5_resident_peak(&root);

    let coresident_estimate = umt5_peak + dit_bytes;
    let saved = coresident_estimate.saturating_sub(staged_peak);

    println!(
        "\nSCAIL-2 sequential residency ({w}×{h}, {n_frames} frames, {steps} steps):\n  \
         UMT5 resident peak (text phase) = {:.2} GiB\n  \
         DiT weights (dit.safetensors, Q4-packed) = {:.2} GiB\n  \
         co-resident estimate (UMT5 + DiT) = {:.2} GiB\n  \
         staged generate peak = {:.2} GiB\n  \
         saved vs co-residence ≈ {:.2} GiB ({:.1}%)",
        umt5_peak as f64 / GIB,
        dit_bytes as f64 / GIB,
        coresident_estimate as f64 / GIB,
        staged_peak as f64 / GIB,
        saved as f64 / GIB,
        100.0 * saved as f64 / coresident_estimate as f64,
    );

    // (1) Non-degenerate output: a non-empty clip whose frames are each the requested size, and frame 0
    // is not a flat single-color buffer (the staged denoise + decode actually produced pixels). The
    // exact output frame count is NOT pinned here — the z16 VAE temporally expands the driving clip
    // (9 driving frames → 12 out), and `generate_smoke` / the parity gates own the count/values.
    assert!(!frames.is_empty(), "no video frames produced");
    for (i, f) in frames.iter().enumerate() {
        assert_eq!(
            f.pixels.len(),
            w * h * 3,
            "frame {i} is {}×{} — wrong pixel count",
            f.width,
            f.height
        );
    }
    let f0 = &frames[0];
    assert!(
        f0.pixels.iter().any(|&p| p != f0.pixels[0]),
        "frame 0 is a flat single-color buffer — the staged denoise/decode produced no image"
    );

    // (2) The residency invariant: the staged generate NEVER reaches UMT5+DiT co-residence. If staging
    // regressed (UMT5 held through the denoise), staged_peak ≈ UMT5 + DiT + smalls > this estimate.
    assert!(
        staged_peak < coresident_estimate,
        "staged peak {:.2} GiB was NOT below the UMT5+DiT co-residence estimate {:.2} GiB — the UMT5 \
         drop before the DiT did not bound peak (staging regressed?)",
        staged_peak as f64 / GIB,
        coresident_estimate as f64 / GIB,
    );
    // (3) Tripwire: the smaller giant really left co-residence — the win should be multiple GiB (the Q4
    // DiT ≈ 9 GiB), well above measurement noise. A tiny/zero saving means both stayed resident.
    assert!(
        saved as f64 / GIB > 2.0,
        "saved only {:.2} GiB — expected several GiB (≈ the Q4 DiT dropped out of co-residence); \
         staging may not be freeing the UMT5 / DiT",
        saved as f64 / GIB,
    );
}
