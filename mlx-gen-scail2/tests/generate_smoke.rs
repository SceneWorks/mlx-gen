//! SCAIL-2 end-to-end `generate()` smoke (sc-5443).
//!
//! Two tiers:
//!   * `missing_reference_errors` (CI) — drives the registered provider with an empty request and
//!     checks the conditioning-extraction wiring rejects it cleanly (no weights touched).
//!   * `generate_{animation,replacement}_smoke` (`#[ignore]`, real-weight) — run the full
//!     preprocessing → plain-CFG denoise → VAE-decode pipeline per driving mode on synthetic inputs at
//!     a small resolution / few steps and assert a sane video comes out. They prove the loop *runs*
//!     end-to-end; per-step numeric parity vs. the upstream forward is the `dit_real_parity` gate
//!     (sc-5446).
//!
//! Run the real-weight smoke on macOS against the assembled snapshot:
//! `cargo test -p mlx-gen-scail2 --test generate_smoke -- --ignored --nocapture`
//! (env: `SCAIL2_SNAPSHOT_DIR`, `SCAIL2_SMOKE_SIZE`=256, `SCAIL2_SMOKE_FRAMES`=13,
//! `SCAIL2_SMOKE_STEPS`=8).

use std::path::PathBuf;

use mlx_gen::{
    Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, Quant, ReplacementMode,
    WeightsSource,
};
// Referencing the crate forces the linker to include its `inventory::submit!` registration.
use mlx_gen_scail2::pipeline::MODEL_ID;

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

/// A deterministic gradient image (stands in for a real reference / driving frame).
fn gradient(w: usize, h: usize, phase: usize) -> Image {
    let mut pixels = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let r = ((x + phase) % 256) as u8;
            let g = ((y + phase) % 256) as u8;
            let b = ((x + y + phase) % 256) as u8;
            pixels.extend_from_slice(&[r, g, b]);
        }
    }
    Image {
        width: w as u32,
        height: h as u32,
        pixels,
    }
}

/// A two-region color-coded mask: left half white (visible), right half red (a character region),
/// with a moving vertical split so per-frame masks differ.
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

#[test]
fn missing_reference_errors() {
    // load() only needs an existing dir (config.json is optional → defaults). The conditioning
    // extraction runs before any weight load, so an empty request fails fast.
    let spec = LoadSpec::new(WeightsSource::Dir(std::env::temp_dir()));
    let gen = mlx_gen::registry::load(MODEL_ID, &spec).expect("load scail2 provider");
    let req = GenerationRequest {
        prompt: "a person dancing".into(),
        ..Default::default()
    };
    let err = gen
        .generate(&req, &mut |_| {})
        .expect_err("empty conditioning must error");
    let msg = format!("{err}");
    assert!(
        msg.contains("Reference"),
        "expected a Reference-required error, got: {msg}"
    );
}

/// Run the full pipeline once for one driving mode (`replace=false` → animation,
/// `replace=true` → cross-identity replacement via `video_mode`) at an optional load-time quant
/// (`None` = bf16, `Some(Q4/Q8)` = the sc-5445 load-time DiT quantization) and assert a sane video.
fn run_mode(label: &str, replace: bool, quant: Option<Quant>) {
    // `SCAIL2_SMOKE_SIZE` sets a square default; `SCAIL2_SMOKE_W` / `_H` override per-axis so the
    // real (non-square) production buckets like 832x480 can be measured for the sc-5445 minMemoryGb.
    let size = env_usize("SCAIL2_SMOKE_SIZE", 256);
    let w = env_usize("SCAIL2_SMOKE_W", size);
    let h = env_usize("SCAIL2_SMOKE_H", size);
    let n_frames = env_usize("SCAIL2_SMOKE_FRAMES", 13);
    let steps = env_usize("SCAIL2_SMOKE_STEPS", 8);

    let root = snapshot_dir();
    assert!(
        root.join("dit.safetensors").exists(),
        "missing snapshot at {} — assemble it first (sc-5445)",
        root.display()
    );

    // Synthetic single-character job: gradient reference, a 2-color ref mask, a short driving clip
    // with per-frame color masks.
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
        seed: Some(7),
        fps: Some(16),
        video_mode: replace.then(|| "replacement".to_string()),
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

    let mut spec = LoadSpec::new(WeightsSource::Dir(root));
    if let Some(q) = quant {
        spec = spec.with_quant(q);
    }
    let gen = mlx_gen::registry::load(MODEL_ID, &spec).expect("load scail2 provider");

    let mut last_step = 0u32;
    let out = gen
        .generate(&req, &mut |p| {
            if let mlx_gen::Progress::Step { current, .. } = p {
                last_step = current;
            }
        })
        .expect("generate must succeed");

    let GenerationOutput::Video { frames, fps, .. } = out else {
        panic!("expected a Video output");
    };
    assert_eq!(fps, 16, "fps passthrough");
    assert!(!frames.is_empty(), "no frames produced");
    assert_eq!(last_step as usize, steps, "all denoise steps ran");
    for (i, f) in frames.iter().enumerate() {
        assert_eq!(f.width as usize, w, "frame {i} width");
        assert_eq!(f.height as usize, h, "frame {i} height");
        assert_eq!(f.pixels.len(), w * h * 3, "frame {i} pixel buffer size");
    }
    // Sanity: the decoded video is not a single flat color (would signal a dead pipeline).
    let (mut lo, mut hi) = (255u8, 0u8);
    for f in &frames {
        for &p in &f.pixels {
            lo = lo.min(p);
            hi = hi.max(p);
        }
    }
    assert!(hi > lo, "decoded video is a single flat value ({lo})");
    println!(
        "{label}: {} frames @ {w}x{h}, {steps} steps, byte range [{lo},{hi}]",
        frames.len()
    );
}

#[test]
#[ignore = "real ~46 GB snapshot; run with --ignored on macOS (see module doc)"]
fn generate_animation_smoke() {
    run_mode("animation/bf16", false, None);
}

#[test]
#[ignore = "real ~46 GB snapshot; run with --ignored on macOS (see module doc)"]
fn generate_replacement_smoke() {
    run_mode("replacement/bf16", true, None);
}

// Load-time DiT quantization (sc-5445): the same e2e path with Q4 (the SceneWorks worker default)
// and Q8 applied to the DiT's attention + FFN Linears. Proves the quantize wiring runs end-to-end
// and the quantized forward still produces a sane (non-flat) video. Q4 is the validated default.
#[test]
#[ignore = "real ~46 GB snapshot; run with --ignored on macOS (see module doc)"]
fn generate_animation_q4_smoke() {
    run_mode("animation/Q4", false, Some(Quant::Q4));
}

#[test]
#[ignore = "real ~46 GB snapshot; run with --ignored on macOS (see module doc)"]
fn generate_replacement_q4_smoke() {
    run_mode("replacement/Q4", true, Some(Quant::Q4));
}

#[test]
#[ignore = "real ~46 GB snapshot; run with --ignored on macOS (see module doc)"]
fn generate_animation_q8_smoke() {
    run_mode("animation/Q8", false, Some(Quant::Q8));
}
