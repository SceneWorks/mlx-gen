//! sc-3190: real-weight (35GB) interleaved-generation (Document Studio) e2e — `#[ignore]`.
//!
//! Loads the actual checkpoint and runs [`T2iModel::interleave_gen`] (think-mode) on a fixed prompt,
//! comparing to the reference `interleave_gen` (`tools/dump_sensenova_interleave_realweight.py`).
//!
//! Long greedy rollouts are NOT bit-stable across builds — a single near-tie argmax flip (the f32
//! port vs the bf16 reference, or just the deep-backbone f32 matmul floor) cascades — so the full
//! token stream can't match. What's validated here: the **deterministic prefix** (the think block
//! agrees with the reference for a substantial run, confirming the system message + prefill + greedy
//! decode are correct), and that the orchestration produces a **coherent interleaved document**
//! (think block + ≥1 generated image via the text↔image↔re-encode loop) end to end on real weights.
//! The append/image numerics are validated deterministically by `interleave_parity` (the synthetic
//! `append_generated_image` golden) + the sc-3188/3189 image-gen parity.
//!
//! Run: `cargo test -p mlx-gen-sensenova --test interleave_realweight -- --ignored --nocapture`

use std::path::PathBuf;

use mlx_gen_sensenova::{
    loader::load_raw, text::load_tokenizer, NeoChatConfig, T2iModel, T2iOptions,
    INTERLEAVE_SYSTEM_MESSAGE,
};

const DEFAULT_SNAPSHOT: &str = concat!(
    env!("HOME"),
    "/.cache/huggingface/hub/models--sensenova--SenseNova-U1-8B-MoT/snapshots/\
     bfa9b436503cb8aed4f2bc60e3236710cc77468d"
);

fn snapshot_dir() -> PathBuf {
    std::env::var("SENSENOVA_SNAPSHOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_SNAPSHOT))
}

fn fixture() -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/interleave_realweight_golden.safetensors"
    ))
}

/// Length (bytes) of the common prefix of two strings.
fn common_prefix_len(a: &str, b: &str) -> usize {
    a.chars()
        .zip(b.chars())
        .take_while(|(x, y)| x == y)
        .map(|(x, _)| x.len_utf8())
        .sum()
}

#[test]
#[ignore = "needs the local 35GB checkpoint + dumped golden; run with --ignored"]
fn interleave_realweight_matches_reference() {
    let snap = snapshot_dir();
    let fix = fixture();
    if !snap.exists() || !fix.exists() {
        eprintln!("skipping: snapshot or golden missing — see tools/dump_sensenova_interleave_realweight.py");
        return;
    }

    let golden = mlx_gen::weights::Weights::from_file(&fix).expect("load golden");
    let prompt = golden.metadata("prompt").unwrap();
    let width: i32 = golden.metadata("width").unwrap().parse().unwrap();
    let height: i32 = golden.metadata("height").unwrap().parse().unwrap();
    let num_steps: usize = golden.metadata("num_steps").unwrap().parse().unwrap();
    let cfg_scale: f32 = golden.metadata("cfg").unwrap().parse().unwrap();
    let img_cfg_scale: f32 = golden.metadata("img_cfg").unwrap().parse().unwrap();
    let timestep_shift: f32 = golden.metadata("timestep_shift").unwrap().parse().unwrap();
    let ref_text = golden.metadata("text").unwrap().to_string();

    println!("loading checkpoint {} …", snap.display());
    let cfg = NeoChatConfig::from_dir(&snap).expect("config");
    let weights = load_raw(&snap).expect("weights");
    let tokenizer = load_tokenizer(&snap).expect("tokenizer");
    let model = T2iModel::from_weights(&weights, &cfg).expect("build T2iModel");

    let opts = T2iOptions {
        cfg_scale,
        img_cfg_scale,
        num_steps,
        timestep_shift,
        think_mode: true,
        ..Default::default()
    };
    // Generous budget so the rollout reaches images on the port's own (possibly divergent) path.
    let out = model
        .interleave_gen(
            &tokenizer,
            prompt,
            &[],
            width,
            height,
            &opts,
            INTERLEAVE_SYSTEM_MESSAGE,
            512,
            10,
            None,
        )
        .expect("interleave_gen");

    println!("got {} image(s)", out.images.len());
    println!("got text : {:?}", out.text);

    // Coherent interleaved document.
    assert!(out.text.contains("<think>"), "think-mode block missing");
    assert!(
        out.text.contains("<image>"),
        "no <image> placeholder in text"
    );
    assert!(!out.images.is_empty(), "interleave produced no images");
    for img in &out.images {
        let n = img.shape().iter().product::<i32>();
        let v = img.reshape(&[n]).unwrap();
        assert!(
            v.as_slice::<f32>().iter().all(|x| x.is_finite()),
            "image has non-finite values"
        );
    }

    // Deterministic prefix: agrees with the reference think block before the first near-tie flip.
    let common = common_prefix_len(&out.text, &ref_text);
    println!("common prefix: {common} bytes");
    assert!(
        common >= 40,
        "deterministic think-text prefix agrees only {common} bytes with the reference"
    );
}
