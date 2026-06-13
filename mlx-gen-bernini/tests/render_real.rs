//! sc-4706: Bernini renderer provider — registry wiring (CI) + real-weight end-to-end generation
//! smokes (`#[ignore]`): one image mode (t2i via `t2v_apg`, text-only) and one reference mode
//! (r2v via `r2v_apg`, which adds VAE-encode of a source image + source-id RoPE on a real source +
//! the chained APG). Both drive the sc-4705-assembled snapshot through load → APG denoise → decode.

use std::path::PathBuf;

use mlx_gen::media::Image;
use mlx_gen::{
    registry, Conditioning, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource,
};
use mlx_gen_wan::convert::assemble_bernini_renderer_snapshot;

const MODEL_ID: &str = "bernini_renderer";

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

/// Assemble the converted renderer snapshot once (reused across reruns), returning its dir.
fn ensure_snapshot() -> PathBuf {
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

/// A deterministic non-uniform RGB8 image (diagonal gradient) for a conditioning source.
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

/// The provider self-registers under `bernini_renderer`: a registry `load` with a bad dir dispatches
/// to the Bernini loader (fails on the missing snapshot), proving it is wired — not "unknown model".
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
#[ignore = "real weights: assembles + loads the ~56 GB Bernini renderer snapshot, runs a denoise"]
fn t2i_real_weight_smoke() {
    let model =
        mlx_gen_bernini::pipeline::load(&LoadSpec::new(WeightsSource::Dir(ensure_snapshot())))
            .expect("load bernini_renderer");
    // Tiny t2i (1 frame, 256², 4 steps) — load + UMT5 + src-id RoPE + packed forward + expert switch
    // + APG (t2v_apg) + VAE decode end-to-end.
    let req = GenerationRequest {
        prompt: "a red apple on a wooden table, studio lighting".into(),
        width: 256,
        height: 256,
        frames: Some(1),
        steps: Some(4),
        seed: Some(0),
        video_mode: Some("t2v_apg".into()),
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
#[ignore = "real weights: reference-conditioned r2v_apg (VAE-encode + source-id RoPE + chained APG)"]
fn r2v_real_weight_smoke() {
    let model =
        mlx_gen_bernini::pipeline::load(&LoadSpec::new(WeightsSource::Dir(ensure_snapshot())))
            .expect("load bernini_renderer");
    // r2v_apg with one synthetic reference image — exercises the conditioning path the t2i smoke
    // skips: VAE-encode of the source, source-id RoPE on a real source (id 1), the packed
    // multi-segment forward (target + 1 source), and the chained APG (two momentum buffers).
    let req = GenerationRequest {
        prompt: "the subject riding a bicycle".into(),
        width: 256,
        height: 256,
        frames: Some(1),
        steps: Some(4),
        seed: Some(0),
        video_mode: Some("r2v_apg".into()),
        conditioning: vec![Conditioning::Reference {
            image: synthetic_image(256, 256),
            strength: None,
        }],
        ..Default::default()
    };
    let mut on_progress = |_p| {};
    match model.generate(&req, &mut on_progress).expect("generate") {
        GenerationOutput::Images(imgs) => {
            assert_eq!(imgs.len(), 1);
            assert_coherent_image(&imgs[0], 256, 256);
        }
        GenerationOutput::Video { .. } => panic!("expected Images for a 1-frame request"),
    }
}
