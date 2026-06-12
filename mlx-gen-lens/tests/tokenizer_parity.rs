//! sc-3167 — Lens harmony tokenizer parity vs `LensPipeline._build_chat_inputs`.
//!
//! Loads the snapshot `tokenizer.json` + the golden reference ids
//! (`tools/golden/lens_tokenizer_golden.safetensors`, dumped by
//! `tools/dump_lens_tokenizer_golden.py`) and asserts the Rust `LensTokenizer` reproduces the
//! reference `input_ids` **byte-for-byte** for every prompt (using the golden's recorded date for the
//! harmony preamble), and that the preamble is exactly `TXT_OFFSET` tokens.
//!
//! Run: `cargo test -p mlx-gen-lens --test tokenizer_parity -- --ignored --nocapture`

use mlx_gen::weights::Weights;
use mlx_gen_lens::text::{LensTokenizer, TXT_OFFSET};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/lens_tokenizer_golden.safetensors"
);

fn newest_snapshot() -> std::path::PathBuf {
    let base = std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/huggingface/hub/models--microsoft--Lens-Turbo/snapshots");
    std::fs::read_dir(&base)
        .unwrap_or_else(|_| panic!("snapshot dir {}", base.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .max()
        .expect("a snapshot")
}

#[test]
#[ignore = "needs tools/golden/lens_tokenizer_golden.safetensors + the Lens-Turbo tokenizer snapshot"]
fn lens_tokenizer_matches_reference() {
    let g = Weights::from_file(GOLDEN).expect("tokenizer golden");
    let date = g.metadata("current_date").expect("current_date meta");
    let n: usize = g.metadata("n_prompts").unwrap().parse().unwrap();
    let off: usize = g.metadata("txt_offset").unwrap().parse().unwrap();
    assert_eq!(
        off, TXT_OFFSET,
        "golden txt_offset {off} != crate TXT_OFFSET {TXT_OFFSET}"
    );

    let tok = LensTokenizer::from_file(newest_snapshot().join("tokenizer/tokenizer.json"))
        .expect("load lens tokenizer");

    for i in 0..n {
        let prompt = g.metadata(&format!("prompt_{i}")).expect("prompt meta");
        let want: Vec<i32> = g
            .require(&format!("ids_{i}"))
            .unwrap()
            .as_slice::<i32>()
            .to_vec();
        let got = tok.encode(prompt, date).unwrap().ids;
        assert!(
            got.len() >= TXT_OFFSET,
            "prompt {i}: {} tokens < preamble {TXT_OFFSET}",
            got.len()
        );
        assert_eq!(
            got,
            want,
            "prompt {i} ({prompt:?}): ids diverge ({} vs {} tokens)",
            got.len(),
            want.len()
        );
        eprintln!(
            "prompt {i}: {} tokens, byte-exact (caption tokens [{TXT_OFFSET}..])",
            got.len()
        );
    }
}
