//! sc-5985 — real-weight parity for the Ideogram 4 Qwen3-VL text encoder against the
//! **transformers** Qwen3-VL forward (an independent graph). Exercises bias-less GQA, per-head
//! q/k RMSNorm, HF half-split RoPE, the causal mask, and the 13-layer hidden-state concat.
//!
//! `#[ignore]` — needs the converted snapshot (`tools/convert_ideogram4_to_mlx.py`) and the golden
//! (`tools/dump_ideogram4_te_golden.py`). Run:
//!   tools/dump_ideogram4_te_golden.py   # writes tools/golden/ideogram4_te.safetensors
//!   CARGO_TARGET_DIR=~/Repos/mlx-gen/target \
//!     cargo test -p mlx-gen-ideogram --test te_parity -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_ideogram::load_text_encoder;
use mlx_rs::ops::{multiply, sqrt, sum};
use mlx_rs::{Array, Dtype};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/ideogram4_te.safetensors"
);

/// Converted-snapshot dir: `$IDEOGRAM4_MLX` or `~/.cache/ideogram4-mlx-convert`.
fn snapshot_dir() -> PathBuf {
    std::env::var("IDEOGRAM4_MLX")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME")).join(".cache/ideogram4-mlx-convert")
        })
}

/// Cosine similarity over all elements (robust to bf16 precision/scale): `⟨a,b⟩ / (‖a‖·‖b‖)`.
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
fn te_matches_transformers_reference() {
    let g = Weights::from_file(GOLDEN).expect("golden — run tools/dump_ideogram4_te_golden.py");
    let te = load_text_encoder(&snapshot_dir()).expect("load converted text encoder");
    let out = te
        .prompt_embeds(
            g.require("input_ids").unwrap(),
            g.require("attention_mask").unwrap(),
        )
        .unwrap();
    let want = g.require("golden").unwrap();
    assert_eq!(out.shape(), want.shape(), "prompt_embeds shape");

    let c = cosine(&out, want);
    println!("Ideogram 4 TE parity cosine = {c:.7}");
    assert!(
        c > 0.999,
        "TE parity cosine {c} too low — structural mismatch"
    );
}
