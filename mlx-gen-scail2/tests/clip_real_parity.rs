//! SCAIL-2 CLIP **real-weight** parity gate (sc-5445 / sc-5446).
//!
//! Loads the real de-prefixed `clip.safetensors` (the stock open-CLIP XLM-RoBERTa ViT-H/14 visual
//! tower, 32 layers / 1280-dim) and runs the full CLIP path ([`mlx_gen_scail2::clip_preprocess`] →
//! [`mlx_gen_scail2::ScailClip::encode`]) against the upstream `VisionTransformer.visual(use_31_block
//! =True)` reference. The tiny-seeded `clip_parity.rs` proves the algorithm; this proves the real
//! weights load + run.
//!
//! Generate the snapshot's `clip.safetensors` + the reference on the Mac:
//!
//! ```text
//! SCAIL2_PARITY_DIR=~/.cache/scail2-parity \
//!   ~/mlx-flux-venv/bin/python _vendor/scail2/_gen_clip_realweight_fixtures.py
//! ```
//!
//! `#[ignore]` (needs the ~2.5 GB weights + fixtures, off CI). Run with
//! `cargo test -p mlx-gen-scail2 --test clip_real_parity -- --ignored --nocapture`.
//! `SCAIL2_SNAPSHOT_DIR` overrides the snapshot dir (default `~/.cache/scail2-mlx-convert`).

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_scail2::{clip_preprocess, ClipVisionConfig, ScailClip};
use mlx_rs::{Array, Dtype};

fn home_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap())
}

fn parity_dir() -> PathBuf {
    std::env::var("SCAIL2_PARITY_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| home_dir().join(".cache/scail2-parity"))
}

fn snapshot_dir() -> PathBuf {
    std::env::var("SCAIL2_SNAPSHOT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| home_dir().join(".cache/scail2-mlx-convert"))
}

fn cosine(a: &Array, b: &Array) -> f32 {
    let va = a
        .reshape(&[-1])
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let vb = b
        .reshape(&[-1])
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    assert_eq!(va.len(), vb.len());
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in va.iter().zip(vb.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

#[test]
#[ignore = "real ~2.5 GB ViT-H/14 weights + fixtures (see module doc); run with --ignored on macOS"]
fn clip_real_parity() {
    let clip_path = snapshot_dir().join("clip.safetensors");
    let io_path = parity_dir().join("clip_real/io.safetensors");
    assert!(
        clip_path.exists() && io_path.exists(),
        "missing real CLIP weights/fixtures — generate with \
         `~/mlx-flux-venv/bin/python _vendor/scail2/_gen_clip_realweight_fixtures.py` \
         ({} / {})",
        clip_path.display(),
        io_path.display(),
    );

    let w = Weights::from_file(&clip_path).unwrap();
    let clip = ScailClip::from_weights(&w, &ClipVisionConfig::vit_h_14()).unwrap();

    let io = Weights::from_file(&io_path).unwrap();
    let pixel = clip_preprocess(io.require("image").unwrap(), 224).unwrap();
    let out = clip.encode(&pixel).unwrap();
    let reference = io.require("output").unwrap();
    assert_eq!(out.shape(), reference.shape(), "real CLIP feature shape");
    let cos = cosine(&out, reference);
    println!("real ViT-H/14 penultimate (clip_preprocess → encode): cosine {cos:.7}");
    // f32-vs-f32 over 31 layers; residual is MLX Metal matmul reduced precision.
    assert!(cos > 0.999, "real CLIP cosine {cos} below 0.999");
}
