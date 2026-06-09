//! sc-3191: real-weight (35GB) VQA end-to-end parity — `#[ignore]`, run locally.
//!
//! Loads the actual checkpoint and runs [`T2iModel::vqa`] (greedy) on a fixed image + question,
//! comparing the answer token stream to the reference `chat`/`generate` greedy decode
//! (`tools/dump_sensenova_vqa_realweight.py`). Also asserts the prompt query encodes to the
//! reference's condition ids. Greedy argmax on the understanding path is robust, so a long agreeing
//! prefix is expected (a late bf16-vs-f32 near-tie flip is tolerated).
//!
//! Run: `cargo test -p mlx-gen-sensenova --test vqa_realweight -- --ignored --nocapture`

use std::path::PathBuf;

use mlx_gen_sensenova::{loader::load_raw, text::load_tokenizer, NeoChatConfig, Sampler, T2iModel};

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
        "/tests/fixtures/vqa_realweight_golden.safetensors"
    ))
}

#[test]
#[ignore = "needs the local 35GB checkpoint + dumped golden; run with --ignored"]
fn vqa_realweight_matches_reference() {
    let snap = snapshot_dir();
    let fix = fixture();
    if !snap.exists() || !fix.exists() {
        eprintln!(
            "skipping: snapshot or golden missing — see tools/dump_sensenova_vqa_realweight.py"
        );
        return;
    }

    let golden = mlx_gen::weights::Weights::from_file(&fix).expect("load golden");
    let question = golden.metadata("question").unwrap();
    let max_new: usize = golden.metadata("max_new").unwrap().parse().unwrap();
    let eos_id: i32 = golden.metadata("eos_id").unwrap().parse().unwrap();
    let want_ids: Vec<i32> = golden
        .require("answer_ids")
        .unwrap()
        .as_slice::<i32>()
        .to_vec();
    let want_cond_ids: Vec<i32> = golden
        .require("cond_input_ids")
        .unwrap()
        .as_slice::<i32>()
        .to_vec();
    let src = golden.require("src").unwrap().clone();

    println!("loading checkpoint {} …", snap.display());
    let cfg = NeoChatConfig::from_dir(&snap).expect("config");
    let weights = load_raw(&snap).expect("weights");
    let tokenizer = load_tokenizer(&snap).expect("tokenizer");
    let model = T2iModel::from_weights(&weights, &cfg).expect("build T2iModel");

    // Query ids must match the reference.
    let (pix, (gh, gw)) = model.preprocess_image(&src).expect("preprocess");
    let n = (gh / 2) * (gw / 2);
    let q = mlx_gen_sensenova::build_neo1_query(&format!("<image>\n{question}"), "").replacen(
        "<image>",
        &format!("<img>{}</img>", "<IMG_CONTEXT>".repeat(n as usize)),
        1,
    );
    let got_cond_ids = tokenizer.encode_ids(&q, true).unwrap();
    assert_eq!(got_cond_ids, want_cond_ids, "condition query ids mismatch");

    // Greedy decode the answer token stream.
    let (mut cache, first_logits, t_idx) = model
        .prefill_it2i_logits(&got_cond_ids, Some(&pix), &[(gh, gw)])
        .expect("prefill");
    let got = model
        .decode_text(
            &first_logits,
            &mut cache,
            t_idx,
            &[eos_id],
            max_new,
            Sampler::Greedy,
        )
        .expect("decode");

    let agree = got
        .iter()
        .zip(&want_ids)
        .take_while(|(a, b)| a == b)
        .count();
    println!("vqa answer: agree {agree}/{} tokens", want_ids.len());
    println!("  got : {got:?}");
    println!("  want: {want_ids:?}");
    println!(
        "  decoded(got) : {:?}",
        tokenizer
            .decode(&got.iter().map(|&i| i as u32).collect::<Vec<_>>(), true)
            .unwrap()
    );
    println!(
        "  decoded(want): {:?}",
        tokenizer
            .decode(
                &want_ids.iter().map(|&i| i as u32).collect::<Vec<_>>(),
                true
            )
            .unwrap()
    );

    // Greedy argmax on the understanding path is robust; require a long agreeing prefix
    // (bf16-vs-f32 may flip a late near-tie).
    let need = (want_ids.len() * 3 / 4).max(8);
    assert!(
        agree >= need,
        "greedy stream agrees only {agree}/{} (need {need})",
        want_ids.len()
    );

    // The composed `vqa` entry runs end to end and returns coherent text.
    let answer = model
        .vqa(
            &tokenizer,
            question,
            std::slice::from_ref(&src),
            max_new,
            Sampler::Greedy,
        )
        .expect("vqa");
    println!("  vqa(): {answer:?}");
    assert!(!answer.is_empty(), "vqa returned empty answer");
}
