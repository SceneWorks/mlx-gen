//! S1 Gemma-3-12B text-encoder parity vs the reference (sc-2679 S1) + the **TE-quant** gate (sc-2686).
//!
//! `#[ignore]`d: needs the Gemma shards (~13–24 GB) + the golden from `tools/dump_ltx_gemma_golden.py`
//! (gitignored, real-weights). Runs the Rust `GemmaModel` on the golden's input_ids/attention_mask and
//! checks all 49 hidden states reproduce the reference.
//!
//!  - `gemma_hidden_states_match_reference` — the dense bf16 path (default `…-bf16` snapshot).
//!  - `gemma_quant_hidden_states_match_reference` — the **quantized** path: point `LTX_GEMMA_Q8_DIR`
//!    at a quantized Gemma snapshot (e.g. `mlx-community/gemma-3-12b-it-8bit`), dump its golden with
//!    `LTX_GEMMA_DIR=… tools/dump_ltx_gemma_golden.py`, and this gates the Rust `GemmaLinear::Quant`
//!    consumption (the reference `apply_quantization` path) end-to-end through all 48 layers.
//!
//! Run: `cargo test -p mlx-gen-ltx --test gemma_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max, subtract};
use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen_ltx::gemma::{GemmaConfig, GemmaModel, GemmaQuant};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/ltx_gemma_golden.safetensors"
);
const GOLDEN_Q8: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/ltx_gemma_golden_q8.safetensors"
);

fn newest_snapshot(repo_dir: &str) -> std::path::PathBuf {
    let base = std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join(format!(".cache/huggingface/hub/{repo_dir}/snapshots"));
    std::fs::read_dir(&base)
        .unwrap_or_else(|_| panic!("snapshot dir {}", base.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .max()
        .expect("a snapshot")
}

fn gemma_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("LTX_GEMMA_DIR") {
        return d.into();
    }
    newest_snapshot("models--mlx-community--gemma-3-12b-it-bf16")
}

fn gemma_q8_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("LTX_GEMMA_Q8_DIR") {
        return d.into();
    }
    newest_snapshot("models--mlx-community--gemma-3-12b-it-8bit")
}

fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(got, want).unwrap()).unwrap();
    let denom = max(abs(want).unwrap(), None).unwrap().item::<f32>();
    max(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

/// Load `gemma_dir` (dense bf16 or, with `quant`, the quantized snapshot), run the 49-state forward on
/// the golden's input, and assert the worst per-layer `peak_rel` is below `tol`.
fn run_gemma_gate(gemma_dir: &std::path::Path, golden: &str, quant: Option<GemmaQuant>, tol: f32) {
    let w = Weights::from_dir(gemma_dir).expect("load gemma shards");
    let model =
        GemmaModel::from_weights(&w, GemmaConfig::gemma_3_12b(), quant).expect("build gemma");

    let g = Weights::from_file(golden).expect("gemma golden");
    let input_ids = g.require("input_ids").unwrap();
    let attention_mask = g.require("attention_mask").unwrap();

    let hiddens = model
        .forward(input_ids, attention_mask)
        .expect("gemma forward");
    let num: usize = g.metadata("num_hidden").unwrap().parse().unwrap();
    assert_eq!(hiddens.len(), num, "expected {num} hidden states");

    let mut worst = 0f32;
    let mut worst_i = 0;
    let mut profile = Vec::new();
    for (i, h) in hiddens.iter().enumerate() {
        let want = g.require(&format!("h_{i:02}")).unwrap();
        let pr = peak_rel(h, want);
        profile.push(pr);
        if pr > worst {
            worst = pr;
            worst_i = i;
        }
    }
    // Per-layer profile: monotonic growth with depth (near-exact embedding) => bf16/quant accumulation,
    // not a structural bug. Print a sampled trace.
    let trace: Vec<String> = [0usize, 1, 2, 8, 16, 24, 32, 40, 47, 48]
        .iter()
        .filter(|&&i| i < profile.len())
        .map(|&i| format!("h{i}={:.2e}", profile[i]))
        .collect();
    eprintln!("gemma per-layer peak_rel: {}", trace.join(" "));
    eprintln!("gemma: worst peak_rel {worst:.3e} at hidden {worst_i}");
    assert!(
        worst < tol,
        "gemma hidden-state peak_rel {worst:.3e} at {worst_i} exceeds tol {tol:.1e}"
    );
}

#[test]
#[ignore = "needs gemma-3-12b-it-bf16 (~24 GB) + tools/golden/ltx_gemma_golden.safetensors"]
fn gemma_hidden_states_match_reference() {
    // bf16 Rust vs bf16 reference (mlx 0.31.1 == reference 0.31.0); bf16 drift over 48 layers.
    run_gemma_gate(&gemma_dir(), GOLDEN, None, 2e-2);
}

#[test]
#[ignore = "needs gemma-3-12b-it-8bit (~13 GB) + tools/golden/ltx_gemma_golden_q8.safetensors"]
fn gemma_quant_hidden_states_match_reference() {
    // The TE-quant gate (sc-2686): the Rust `GemmaLinear::Quant` consumption reproduces the reference
    // `apply_quantization` forward. Quant geometry is carried in the golden metadata (== the snapshot
    // config.json). Tolerance matches the dense gate — quant adds no structural error, only its own
    // group-affine rounding atop the bf16 drift.
    let g = Weights::from_file(GOLDEN_Q8).expect("gemma q8 golden");
    let bits: i32 = g.metadata("bits").unwrap().parse().unwrap();
    let group: i32 = g.metadata("group").unwrap().parse().unwrap();
    assert!(bits > 0 && group > 0, "golden must carry quant geometry");
    run_gemma_gate(
        &gemma_q8_dir(),
        GOLDEN_Q8,
        Some(GemmaQuant { group, bits }),
        2e-2,
    );
}
