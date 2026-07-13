//! sc-5145: the **full Bernini** pipeline — registry wiring (CI) + real-weight end-to-end coherence
//! smokes (`#[ignore]`). The smokes assemble the combined planner+renderer snapshot from the cached
//! `ByteDance/Bernini-Diffusers` package and drive `mlx_gen::load("bernini")` through the whole stack:
//! preprocess → 3 planner streams → MAR semantic-planning loop → 4 renderer prompt streams (+T5) →
//! ViT-conditioned dual-expert APG denoise → z16 VAE decode.
//!
//! - `t2i`: text-only image — planner MAR loop with no input visuals + `vae_txt_vit_wapg`.
//! - `r2v`: reference-image → video — the multi-image conditioning path (ViT + VAE encode of refs,
//!   source-id RoPE) + `rv2v_wapg`.
//!
//! Per the established bar (full-trajectory pixel parity is cross-backend-chaos-limited; the per-module
//! parity suites validate the components + early steps) these assert **coherence**, not bit-parity.

use std::path::PathBuf;

use mlx_gen::media::Image;
use mlx_gen::{
    registry, Conditioning, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource,
};
use mlx_gen_bernini::convert::assemble_bernini_snapshot;

const MODEL_ID: &str = "bernini";

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
    // Presence of both a planner component and a renderer DiT marks a complete combined snapshot.
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

/// A deterministic non-uniform RGB8 image (diagonal gradient) for a conditioning reference.
fn synthetic_image(w: u32, h: u32) -> Image {
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.push((x % 256) as u8);
            pixels.push((y % 256) as u8);
            pixels.push(((x + y) % 256) as u8);
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

fn assert_coherent_image(img: &Image, w: u32, h: u32) {
    assert_eq!((img.width, img.height), (w, h));
    assert_eq!(img.pixels.len(), (w * h * 3) as usize, "RGB8 buffer");
    assert!(
        img.pixels.iter().any(|&p| p != 0) && img.pixels.iter().any(|&p| p != 255),
        "decoded image must not be uniformly black/white"
    );
}

/// The full pipeline self-registers under `bernini`: a registry `load` with a bad dir dispatches to
/// the Bernini loader (which fails on the missing snapshot), proving it is wired — not "unknown model".
#[test]
fn registers_in_model_registry() {
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent/bernini".into()));
    let err = registry::load(MODEL_ID, &spec)
        .err()
        .expect("load of a missing dir must error");
    let msg = format!("{err}").to_lowercase();
    assert!(
        !msg.contains("no generator") && !msg.contains("unknown"),
        "expected the Bernini loader to dispatch, got: {msg}"
    );
}

#[test]
#[ignore = "real weights: assembles + loads the full Bernini (planner+renderer) snapshot, runs the MAR loop + denoise"]
fn t2i_real_weight_smoke() {
    let model =
        mlx_gen_bernini::bernini::load(&LoadSpec::new(WeightsSource::Dir(ensure_snapshot())))
            .expect("load bernini");
    // Tiny t2i (1 frame, 256², 4 steps, short planning): planner backbone + connector + clip-diff MAR
    // loop (no input visuals) → 4 streams + T5 → vae_txt_vit_wapg dual-expert denoise → VAE decode.
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
    let mut on_progress = |_p| {};
    match model.generate(&req, &mut on_progress).expect("generate") {
        GenerationOutput::Images(imgs) => {
            assert_eq!(imgs.len(), 1, "1-frame t2i yields one image");
            assert_coherent_image(&imgs[0], 256, 256);
        }
        GenerationOutput::Video { .. } => panic!("expected Images for a 1-frame request"),
    }
}

#[test]
#[ignore = "real weights: reference-image → video full pipeline (ViT+VAE encode of a ref, src-id RoPE, rv2v_wapg)"]
fn r2v_real_weight_smoke() {
    let model =
        mlx_gen_bernini::bernini::load(&LoadSpec::new(WeightsSource::Dir(ensure_snapshot())))
            .expect("load bernini");
    // r2v with one synthetic reference image, a short (5-frame) clip — exercises the image-source path
    // the t2i smoke skips: ViT-encode of the ref (planner conditioning), VAE-encode (renderer source),
    // source-id RoPE on a real source, and the rv2v_wapg chain.
    let req = GenerationRequest {
        prompt: "the subject riding a bicycle".into(),
        width: 256,
        height: 256,
        frames: Some(5),
        steps: Some(4),
        seed: Some(0),
        video_mode: Some("r2v".into()),
        conditioning: vec![Conditioning::MultiReference {
            images: vec![synthetic_image(256, 256)],
        }],
        ..Default::default()
    };
    let mut on_progress = |_p| {};
    match model.generate(&req, &mut on_progress).expect("generate") {
        GenerationOutput::Video { frames, .. } => {
            assert!(!frames.is_empty(), "r2v yields video frames");
            assert_coherent_image(&frames[0], 256, 256);
        }
        GenerationOutput::Images(_) => panic!("expected Video for a multi-frame request"),
    }
}
