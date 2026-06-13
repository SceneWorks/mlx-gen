//! sc-4706: Bernini renderer provider — registry wiring (CI) + a real-weight end-to-end generation
//! smoke (`#[ignore]`, drives the sc-4705-assembled snapshot through load → APG denoise → VAE decode).

use std::path::PathBuf;

use mlx_gen::{registry, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};
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

/// The provider self-registers under `bernini_renderer`: a registry `load` with a bad dir dispatches
/// to the Bernini loader (fails on the missing snapshot), proving it is wired — not "unknown model".
#[test]
fn registers_in_model_registry() {
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent/bernini".into()));
    let err = registry::load(MODEL_ID, &spec)
        .err()
        .expect("load of a missing dir must error");
    let msg = format!("{err}");
    assert!(
        !msg.to_lowercase().contains("no generator") && !msg.to_lowercase().contains("unknown"),
        "expected the Bernini loader to dispatch, got: {msg}"
    );
}

#[test]
#[ignore = "real weights: assembles + loads the ~56 GB Bernini renderer snapshot, runs a denoise"]
fn t2i_real_weight_smoke() {
    let home = PathBuf::from(std::env::var("HOME").unwrap());
    let snapshot = home.join(".cache/mlx-gen-models/bernini_renderer_mlx_bf16");
    // Assemble once (reuse on rerun).
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

    let spec = LoadSpec::new(WeightsSource::Dir(snapshot));
    let model = mlx_gen_bernini::pipeline::load(&spec).expect("load bernini_renderer");

    // Tiny t2i (1 frame, 256², 4 steps) — exercises load + UMT5 + src-id RoPE + packed forward +
    // expert switch + APG (t2v_apg) + VAE decode end-to-end.
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
    let out = model.generate(&req, &mut on_progress).expect("generate");
    match out {
        GenerationOutput::Images(imgs) => {
            assert_eq!(imgs.len(), 1, "1-frame t2i yields one image");
            let img = &imgs[0];
            assert_eq!((img.width, img.height), (256, 256));
            assert_eq!(img.pixels.len(), 256 * 256 * 3, "RGB8 buffer");
            assert!(
                img.pixels.iter().any(|&p| p != 0) && img.pixels.iter().any(|&p| p != 255),
                "decoded image must not be uniformly black/white"
            );
        }
        GenerationOutput::Video { .. } => panic!("expected Images for a 1-frame request"),
    }
}
