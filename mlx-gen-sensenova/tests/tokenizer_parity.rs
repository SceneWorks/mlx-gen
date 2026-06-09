//! sc-3186: the `neo1_0` template matches the reference strings (fast), and the tokenizer encodes
//! identically to the reference (`#[ignore]`, needs the snapshot's `tokenizer.json`).
//!
//! `tools/build_sensenova_tokenizer.py` materializes `tokenizer.json` into the snapshot and dumps a
//! small fixture: golden `input_ids` per test string, plus the reference `neo1_0` query strings (in
//! metadata). The fast test validates the template against those strings without any model; the
//! ignored test validates real encoding.
//!
//! Run the encoding check: `cargo test -p mlx-gen-sensenova --test tokenizer_parity -- --ignored`

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_sensenova::{build_neo1_query, load_tokenizer, SYSTEM_MESSAGE_FOR_GEN};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/tokenizer_golden.safetensors"
);

/// The prompt the golden was built with (`tools/build_sensenova_tokenizer.py`).
const PROMPT: &str = "a red fox sitting in a snowy forest, photorealistic";

#[test]
fn neo1_template_matches_reference_strings() {
    let w = Weights::from_file(FIXTURE).expect("fixture");
    // The reference query strings live in the fixture metadata.
    assert_eq!(
        build_neo1_query(PROMPT, SYSTEM_MESSAGE_FOR_GEN),
        w.metadata("str.query_gen").expect("str.query_gen"),
        "neo1_0 query (with SYSTEM_MESSAGE_FOR_GEN) must match the reference verbatim"
    );
    assert_eq!(
        build_neo1_query(PROMPT, ""),
        w.metadata("str.query_empty").expect("str.query_empty"),
        "neo1_0 query (empty system) must match the reference verbatim"
    );
}

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("SENSENOVA_U1_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--sensenova--SenseNova-U1-8B-MoT/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir (set SENSENOVA_U1_SNAPSHOT)")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

#[test]
#[ignore = "needs the snapshot's materialized tokenizer.json (tools/build_sensenova_tokenizer.py)"]
fn tokenizer_encodes_like_reference() {
    let w = Weights::from_file(FIXTURE).expect("fixture");
    let tok = load_tokenizer(snapshot()).expect("load tokenizer");

    for name in ["plain", "query_gen", "query_empty", "specials"] {
        let s = w.metadata(&format!("str.{name}")).unwrap();
        let golden: Vec<i32> = w
            .require(&format!("ids.{name}"))
            .unwrap()
            .as_slice::<i32>()
            .to_vec();
        let got = tok.encode_ids(s, false).unwrap();
        assert_eq!(
            got,
            golden,
            "encoding mismatch for {name} ({} tokens)",
            golden.len()
        );
        println!("{name:>12}: {} tokens OK", golden.len());
    }
}
