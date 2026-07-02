//! sc-3172 — Lens gpt-oss encoder **Q4/Q8** parity vs the bf16 reference.
//!
//! Loads the `text_encoder` with the MoE experts quantized to MLX Q8 / Q4 (the `~12 GB` path —
//! [`LensTextEncoder::from_weights_quant`]) and asserts the captured hidden states still match the
//! **bf16** reference captures in `tools/golden/lens_encoder_golden.safetensors` (the same golden the
//! dense sc-3171 gate uses, dumped from the vendor `LensGptOssEncoder`). The dense bf16 encoder's own
//! worst cosine vs this golden is **0.9971** (sc-3171, the mlx-Metal-vs-torch bf16 floor); quantizing
//! the experts must stay near that floor — Q8 near-lossless, Q4 coherent — confirming the quantized
//! weights are wired correctly (a wrong pack / transpose / group axis would collapse the cosine).
//!
//! Each `#[ignore]`d test loads **only** its quantized encoder, so the process peaks at the `~12 GB`
//! it advertises (run under `/usr/bin/time -l` to confirm the peak RSS) rather than the `~40 GB`
//! dense bf16 stack — the whole point of the story.
//!
//! Run: `cargo test -p mlx-gen-lens --test encoder_quant_parity -- --ignored --nocapture`

use mlx_rs::ops::{multiply, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::Quant;
use mlx_gen_lens::config::GptOssConfig;
use mlx_gen_lens::text_encoder::encoder::{LensTextEncoder, DEFAULT_SELECTED_LAYERS};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/lens_encoder_golden.safetensors"
);

/// The dense bf16 encoder's worst cosine vs this golden (sc-3171) — the cross-build floor quant sits
/// just under.
const DENSE_FLOOR_COS: f32 = 0.9971;

fn text_encoder_dir() -> std::path::PathBuf {
    let base = std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/huggingface/hub/models--microsoft--Lens-Turbo/snapshots");
    let snap = std::fs::read_dir(&base)
        .unwrap_or_else(|_| panic!("snapshot dir {}", base.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .max()
        .expect("a snapshot");
    snap.join("text_encoder")
}

/// Cosine similarity over the flattened tensors.
fn cosine(got: &Array, want: &Array) -> f32 {
    let dot = sum(multiply(got, want).unwrap(), None)
        .unwrap()
        .item::<f32>();
    let na = sum(multiply(got, got).unwrap(), None)
        .unwrap()
        .item::<f32>()
        .sqrt();
    let nb = sum(multiply(want, want).unwrap(), None)
        .unwrap()
        .item::<f32>()
        .sqrt();
    dot / (na * nb).max(1e-12)
}

/// Run the quantized encoder over the golden prompts; return the worst (prompt, layer) cosine vs the
/// bf16 reference captures.
fn worst_cosine_for(quant: Quant) -> f32 {
    let g = Weights::from_file(GOLDEN).expect("encoder golden");
    let n: usize = g.metadata("n_prompts").unwrap().parse().unwrap();
    let selected: Vec<usize> = g
        .metadata("selected_layers")
        .unwrap()
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(selected, DEFAULT_SELECTED_LAYERS.to_vec());

    let cfg = GptOssConfig::lens();
    eprintln!(
        "loading text_encoder @ {quant:?} (MXFP4 → per-layer Q{} pack)…",
        quant.bits()
    );
    let w = Weights::from_dir(text_encoder_dir()).expect("load text_encoder shards");
    let encoder = LensTextEncoder::from_weights_quant(&w, &cfg, Dtype::Bfloat16, Some(quant))
        .expect("build quantized encoder");

    let mut worst = 1f32;
    for i in 0..n {
        let ids = g.require(&format!("ids_{i}")).unwrap().clone(); // [1, L] i32
        let captured = encoder.encode(&ids, None).expect("encode");
        for (j, layer_idx) in selected.iter().enumerate() {
            let want = g.require(&format!("cap_{i}_{j}")).unwrap();
            let got = captured[j].as_dtype(Dtype::Float32).unwrap();
            let cos = cosine(&got, want);
            eprintln!("  prompt {i} layer {layer_idx}: cosine {cos:.5}");
            worst = worst.min(cos);
        }
    }
    eprintln!("worst cosine @ {quant:?}: {worst:.5}  (dense bf16 floor {DENSE_FLOOR_COS})");
    worst
}

#[test]
#[ignore = "needs tools/golden/lens_encoder_golden.safetensors + the Lens-Turbo text_encoder snapshot (~12GB Q8 load)"]
fn encoder_q8_matches_reference() {
    // Q8 is near-lossless: it must hold essentially the dense bf16 floor (a hair under, for the added
    // 8-bit group rounding).
    let worst = worst_cosine_for(Quant::Q8);
    assert!(
        worst > 0.995,
        "Q8 worst cosine {worst:.5} ≤ 0.995 — quant degraded well past the bf16 floor {DENSE_FLOOR_COS}"
    );
    eprintln!("ALL PASS");
}

#[test]
#[ignore = "needs tools/golden/lens_encoder_golden.safetensors + the Lens-Turbo text_encoder snapshot (~12GB Q4 load)"]
fn encoder_q4_matches_reference() {
    // Q4 (the ~12 GB target) is lossier but must stay coherent — the captures still track the bf16
    // reference, not collapse.
    let worst = worst_cosine_for(Quant::Q4);
    assert!(
        worst > 0.95,
        "Q4 worst cosine {worst:.5} ≤ 0.95 — not a coherent quantization"
    );
    eprintln!("ALL PASS");
}
