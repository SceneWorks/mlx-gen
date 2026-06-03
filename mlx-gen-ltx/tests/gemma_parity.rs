//! S1 Gemma-3-12B text-encoder parity vs the reference (sc-2679 S1).
//!
//! `#[ignore]`d: needs the `mlx-community/gemma-3-12b-it-bf16` shards (~24 GB) + the golden from
//! `tools/dump_ltx_gemma_golden.py` (gitignored, real-weights). Runs the Rust `GemmaModel` in bf16
//! (matching the reference) on the golden's input_ids/attention_mask and checks all 49 hidden
//! states reproduce the reference.
//!
//! Run: `cargo test -p mlx-gen-ltx --test gemma_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max, subtract};
use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen_ltx::gemma::{GemmaConfig, GemmaModel};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/ltx_gemma_golden.safetensors"
);

fn gemma_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("LTX_GEMMA_DIR") {
        return d.into();
    }
    let base = std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/huggingface/hub/models--mlx-community--gemma-3-12b-it-bf16/snapshots");
    // newest snapshot dir
    std::fs::read_dir(&base)
        .expect("gemma snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .max()
        .expect("a gemma snapshot")
}

fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(got, want).unwrap()).unwrap();
    let denom = max(abs(want).unwrap(), None).unwrap().item::<f32>();
    max(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

#[test]
#[ignore = "needs gemma-3-12b-it-bf16 (~24 GB) + tools/golden/ltx_gemma_golden.safetensors"]
fn gemma_hidden_states_match_reference() {
    let w = Weights::from_dir(gemma_dir()).expect("load gemma shards");
    let model = GemmaModel::from_weights(&w, GemmaConfig::gemma_3_12b()).expect("build gemma");

    let g = Weights::from_file(GOLDEN).expect("gemma golden");
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
    // Per-layer profile: monotonic growth with depth (near-exact embedding) => bf16 accumulation,
    // not a structural bug. Print a sampled trace.
    let trace: Vec<String> = [0usize, 1, 2, 8, 16, 24, 32, 40, 47, 48]
        .iter()
        .filter(|&&i| i < profile.len())
        .map(|&i| format!("h{i}={:.2e}", profile[i]))
        .collect();
    eprintln!("gemma per-layer peak_rel: {}", trace.join(" "));
    let final_pr = peak_rel(
        &hiddens[num - 1],
        g.require(&format!("h_{:02}", num - 1)).unwrap(),
    );
    eprintln!(
        "gemma: worst peak_rel {worst:.3e} at hidden {worst_i}; final norm peak_rel {final_pr:.3e}"
    );
    // bf16 Rust vs bf16 reference (mlx 0.31.1 == reference 0.31.0); bf16 drift over 48 layers.
    assert!(
        worst < 2e-2,
        "gemma hidden-state peak_rel {worst:.3e} at {worst_i} too high"
    );
}
