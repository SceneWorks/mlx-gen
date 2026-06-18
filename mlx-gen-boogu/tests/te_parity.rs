//! E2 — real-weight parity for the Boogu Qwen3-VL-8B condition encoder against the reference
//! `last_hidden_state` captured by `tools/golden_dump.py`.
//!
//! `#[ignore]` — needs the Base snapshot (`mllm/`) and the golden file. Run:
//!   BOOGU_BASE_DIR=<snapshot> BOOGU_GOLDEN=<...>/boogu_golden.safetensors \
//!     CARGO_TARGET_DIR=~/Repos/mlx-gen/target \
//!     cargo test -p mlx-gen-boogu --test te_parity -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_boogu::load_text_encoder;
use mlx_rs::ops::{multiply, sqrt, sum};
use mlx_rs::{Array, Dtype};

/// Cosine similarity over all elements (robust to bf16 precision): `⟨a,b⟩ / (‖a‖·‖b‖)`.
fn cosine(a: &Array, b: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    let dot = sum(multiply(&a, &b).unwrap(), false).unwrap();
    let na = sqrt(sum(multiply(&a, &a).unwrap(), false).unwrap()).unwrap();
    let nb = sqrt(sum(multiply(&b, &b).unwrap(), false).unwrap()).unwrap();
    (dot / (na * nb)).item::<f32>()
}

fn snapshot_dir() -> PathBuf {
    PathBuf::from(std::env::var("BOOGU_BASE_DIR").expect("set BOOGU_BASE_DIR to the snapshot root"))
}

fn golden_path() -> PathBuf {
    std::env::var("BOOGU_GOLDEN").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(std::env::var("HOME").expect("HOME"))
            .join("Repos/mlx-gen-wt-boogu/reference/goldens/boogu_golden.safetensors")
    })
}

#[test]
#[ignore = "needs real weights + golden (tools/golden_dump.py)"]
fn te_matches_reference_last_hidden() {
    let g = Weights::from_file(golden_path()).expect("golden — run tools/golden_dump.py");
    let te = load_text_encoder(snapshot_dir()).expect("load Qwen3-VL condition encoder");
    let out = te
        .last_hidden(
            g.require("tok_input_ids").unwrap(),
            g.require("tok_attention_mask").unwrap(),
        )
        .unwrap();
    let want = g.require("instruction_hidden_states").unwrap();
    assert_eq!(out.shape(), want.shape(), "last_hidden_state shape");

    let c = cosine(&out, want);
    println!("Boogu Qwen3-VL TE parity cosine = {c:.7}");
    assert!(c > 0.999, "TE parity cosine {c} too low — structural mismatch");
}
