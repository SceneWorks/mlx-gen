//! sc-5986 — real-weight parity for the Ideogram 4 single-stream DiT against the upstream torch
//! `Ideogram4Transformer` (f32, independent graph). Exercises the interleaved 3D MRoPE, the AdaLN
//! sandwich-norm blocks, per-head q/k RMSNorm, the segment mask, the `[text;image]` token
//! composition, and the affine-less final layer.
//!
//! `#[ignore]` — needs the converted snapshot + the golden (`tools/dump_ideogram4_dit_golden.py`):
//!   CARGO_TARGET_DIR=~/Repos/mlx-gen/target \
//!     cargo test -p mlx-gen-ideogram --test dit_parity -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_ideogram::load_transformer;
use mlx_rs::ops::{multiply, sqrt, sum};
use mlx_rs::{Array, Dtype};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/ideogram4_dit.safetensors"
);

fn snapshot_dir() -> PathBuf {
    std::env::var("IDEOGRAM4_MLX")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME")).join(".cache/ideogram4-mlx-convert")
        })
}

fn cosine(a: &Array, b: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    let dot = sum(multiply(&a, &b).unwrap(), false).unwrap();
    let na = sqrt(sum(multiply(&a, &a).unwrap(), false).unwrap()).unwrap();
    let nb = sqrt(sum(multiply(&b, &b).unwrap(), false).unwrap()).unwrap();
    (dot / (na * nb)).item::<f32>()
}

#[test]
#[ignore = "needs converted weights + generated golden"]
fn dit_matches_reference() {
    let g = Weights::from_file(GOLDEN).expect("golden — run tools/dump_ideogram4_dit_golden.py");
    let dit = load_transformer(&snapshot_dir()).expect("load converted transformer");
    let out = dit
        .forward(
            g.require("llm_features").unwrap(),
            g.require("x").unwrap(),
            g.require("t").unwrap(),
            g.require("position_ids").unwrap(),
            g.require("segment_ids").unwrap(),
            g.require("indicator").unwrap(),
        )
        .unwrap();
    let want = g.require("golden").unwrap();
    assert_eq!(out.shape(), want.shape(), "velocity shape");

    let c = cosine(&out, want);
    println!("Ideogram 4 DiT parity cosine = {c:.7}");
    assert!(
        c > 0.999,
        "DiT parity cosine {c} too low — structural mismatch"
    );
}
