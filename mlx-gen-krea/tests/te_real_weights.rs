//! sc-7569 — real-weight parity for the Krea 2 Qwen3-VL-4B text encoder + tokenizer against the
//! transformers reference loaded with the published `krea/Krea-2-Turbo` `text_encoder/` weights.
//!
//! `#[ignore]` — needs the real snapshot + the golden (`tools/dump_krea_te_real_golden.py`):
//! ```sh
//! KREA_TURBO_DIR=~/.cache/huggingface/hub/models--krea--Krea-2-Turbo/snapshots/<rev> \
//!   cargo test -p mlx-gen-krea --release --test te_real_weights -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_krea::{KreaTeConfig, KreaTextEncoder, KreaTokenizer};
use mlx_rs::ops::{array_eq, multiply, sqrt, sum};
use mlx_rs::{Array, Dtype};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/krea_te_real.safetensors"
);
const PROMPT: &str =
    "A medium-shot photograph of a red fox sitting in a snowy forest at golden hour.";

fn snapshot() -> PathBuf {
    PathBuf::from(std::env::var("KREA_TURBO_DIR").expect("set KREA_TURBO_DIR to the snapshot root"))
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
#[ignore = "needs real weights (KREA_TURBO_DIR) + golden (tools/dump_krea_te_real_golden.py)"]
fn te_matches_real_reference() {
    let g = Weights::from_file(GOLDEN)
        .expect("golden — run tools/dump_krea_te_real_golden.py with KREA_TURBO_DIR set");
    let root = snapshot();
    let cfg = KreaTeConfig::from_snapshot(&root).unwrap();
    assert_eq!(cfg.rope_theta, 5_000_000.0, "qwen3_vl_text default θ");
    assert_eq!(cfg.select_hidden.len(), 12, "12 stacked select-layers");

    let w = Weights::from_dir(root.join("text_encoder")).expect("load real text_encoder/");
    let te = KreaTextEncoder::from_weights(&w, "language_model", &cfg).unwrap();

    let hiddens = te
        .forward(
            g.require("in.input_ids").unwrap(),
            g.require("in.attention_mask").unwrap(),
        )
        .unwrap();
    let want = g.require("out.hiddens").unwrap();
    assert_eq!(hiddens.shape(), want.shape(), "stacked-context shape");

    let c = cosine(&hiddens, want);
    println!("Krea 2 real-weight TE parity cosine = {c:.7}");
    assert!(c > 0.98, "real-weight TE cosine {c:.7} <= 0.98");
}

#[test]
#[ignore = "needs the real snapshot tokenizer (KREA_TURBO_DIR) + golden input_ids"]
fn tokenizer_matches_reference() {
    let root = snapshot();
    let tok = KreaTokenizer::from_snapshot(&root).expect("load tokenizer.json");

    // The system-instruction prefix must tokenize to exactly the slice the encoder drops.
    assert_eq!(
        tok.prefix_len().unwrap(),
        34,
        "prefix template token count (encoder slices this many)"
    );

    // The full templated prompt must reproduce the reference `input_ids` byte-for-byte.
    let g = Weights::from_file(GOLDEN).expect("golden input_ids");
    let want = g.require("in.input_ids").unwrap(); // [1, L] int32
    let ids = tok.ids(PROMPT).unwrap();
    let mine = Array::from_slice(&ids, &[1, ids.len() as i32]);
    assert_eq!(mine.shape(), want.shape(), "tokenized length");
    assert!(
        array_eq(&mine, want, false).unwrap().item::<bool>(),
        "tokenizer ids diverge from the reference"
    );
}
