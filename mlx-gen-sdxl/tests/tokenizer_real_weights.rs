//! sc-2400 S1: byte-identical token-id parity for the SDXL CLIP-BPE tokenizer vs the vendored
//! Apple `_vendor/mlx_sd` reference.
//!
//! `#[ignore]`d — needs the real `stabilityai/stable-diffusion-xl-base-1.0` snapshot in the HF
//! cache and the golden from `tools/dump_sdxl_tokenizer_golden.py` (gitignored, local). Run with:
//!   cargo test -p mlx-gen-sdxl --release --test tokenizer_real_weights -- --ignored --nocapture
//!
//! The tokenizer must reproduce the vendored ids *exactly* (not "close") — any divergence shifts
//! the conditioning sequence and breaks downstream parity.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_sdxl::ClipBpeTokenizer;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/sdxl_tokenizer_golden.safetensors"
);

/// Locate the SDXL-base-1.0 snapshot dir (env override, else the HF cache).
fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("SDXL_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

#[test]
#[ignore = "needs the real SDXL snapshot + tokenizer golden"]
fn tokenizer_ids_match_vendored() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let tok = ClipBpeTokenizer::from_dir(snapshot().join("tokenizer")).unwrap();

    let n: usize = g.metadata("n_prompts").unwrap().parse().unwrap();
    for i in 0..n {
        let prompt = g.metadata(&format!("prompt_{i}")).unwrap();
        let golden = g.require(&format!("ids_{i}")).unwrap().as_slice::<i32>();
        let got = tok.tokenize(prompt).unwrap();
        assert_eq!(
            got,
            golden,
            "tokenizer ids diverge from the vendored reference for prompt {i}: {prompt:?}\n  got:    {got:?}\n  golden: {golden:?}"
        );
        println!("✓ [{i}] {prompt:?} -> {} ids match", got.len());
    }
    println!("✓ all {n} prompts tokenize byte-identically to the vendored Apple tokenizer");
}

#[test]
#[ignore = "needs the real SDXL snapshot + tokenizer golden"]
fn tokenizer_cfg_batch_padding_matches_vendored() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let tok = ClipBpeTokenizer::from_dir(snapshot().join("tokenizer")).unwrap();

    let prompt = g.metadata("batch_prompt").unwrap();
    let negative = g.metadata("negative").unwrap();
    let golden = g.require("batch_prompt_neg").unwrap();
    let (rows, cols) = (golden.shape()[0], golden.shape()[1]);

    let batch = tok.tokenize_batch(prompt, Some(negative)).unwrap();
    assert_eq!(
        batch.shape(),
        &[rows, cols],
        "batch shape (rows x padded len)"
    );
    assert_eq!(
        batch.as_slice::<i32>(),
        golden.as_slice::<i32>(),
        "CFG batch padding (pad id 0, batch-max length) diverges from the vendored _tokenize"
    );
    println!("✓ CFG batch [{rows}x{cols}] (prompt+negative, pad-0) matches the vendored reference");
}
