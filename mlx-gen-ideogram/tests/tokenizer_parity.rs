//! sc-5988 — Ideogram 4 native tokenizer parity vs the reference `_tokenize`.
//!
//! Proves the native Rust path (`ChatTemplate::QwenInstruct` + `encode_chat_ids(.., false)`)
//! reproduces the Python `apply_chat_template(messages, add_generation_prompt=True)` +
//! `tokenizer(text, add_special_tokens=False)` ids for the model's native JSON caption — so the
//! pipeline no longer depends on Python-dumped `input_ids`.
//!
//! `#[ignore]` — needs the ~11 MB Qwen3-VL `tokenizer.json` (too large to commit); it lives in the
//! converted snapshot's `tokenizer/` dir. Golden: `tools/golden/ideogram4_prompt_ids.safetensors`
//! ← `tools/dump_ideogram4_prompt_ids.py` (the same `CAPTION_JSON`). Run:
//!   IDEOGRAM4_MLX=~/.cache/ideogram4-mlx-convert \
//!     cargo test -p mlx-gen-ideogram --test tokenizer_parity -- --ignored --nocapture

mod common;

use std::path::PathBuf;

use common::CAPTION_JSON;
use mlx_gen::array::host_i32;
use mlx_gen::weights::Weights;
use mlx_gen_ideogram::load_tokenizer;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/ideogram4_prompt_ids.safetensors"
);

fn snapshot_dir() -> PathBuf {
    std::env::var("IDEOGRAM4_MLX")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME")).join(".cache/ideogram4-mlx-convert")
        })
}

#[test]
#[ignore = "needs the ~11MB Qwen3-VL tokenizer.json (snapshot tokenizer/ dir)"]
fn native_tokenizer_matches_reference() {
    let tok = load_tokenizer(&snapshot_dir()).expect("load tokenizer (snapshot tokenizer/ dir)");
    let ids = tok
        .encode_chat_ids(CAPTION_JSON, false)
        .expect("encode JSON caption");

    let g =
        Weights::from_file(GOLDEN).expect("run tools/dump_ideogram4_prompt_ids.py for the golden");
    let want = host_i32(g.require("input_ids").unwrap()).unwrap();

    assert_eq!(
        ids.len(),
        want.len(),
        "token count diverged: native {} vs reference {}",
        ids.len(),
        want.len()
    );
    assert_eq!(
        ids, want,
        "native ids diverged from the reference _tokenize"
    );
    println!("native tokenizer parity OK — {} tokens", ids.len());
}
