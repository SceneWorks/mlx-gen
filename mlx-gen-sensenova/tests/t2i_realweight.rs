//! sc-3188: real-weight (35GB) T2I end-to-end parity — `#[ignore]`, run locally.
//!
//! Loads the actual `sensenova/SenseNova-U1-8B-MoT` checkpoint into MLX, runs
//! [`T2iModel::generate`] for a fixed prompt with the reference's **injected** initial noise, and
//! checks directional/structural similarity to the reference final image (dumped by
//! `tools/dump_sensenova_t2i_realweight.py`). e2e is cross-build (MLX-Metal bf16 vs torch bf16 over
//! the denoise steps), so the gate is cosine + a loose peak-rel, not bit parity. Also independently
//! asserts the tokenizer encodes the prompt query to the same ids as the reference.
//!
//! Requires the local checkpoint + the dumped golden; neither is in CI. Run:
//!   cargo test -p mlx-gen-sensenova --test t2i_realweight -- --ignored --nocapture
//! Override the snapshot dir with `SENSENOVA_SNAPSHOT=/path/to/snapshot`.

use std::path::PathBuf;

use mlx_gen_sensenova::{
    loader::load_raw, text::load_tokenizer, NeoChatConfig, T2iModel, T2iOptions,
};
use mlx_rs::Array;

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
        "/tests/fixtures/t2i_realweight_golden.safetensors"
    ))
}

fn flat(a: &Array) -> Vec<f32> {
    let n = a.shape().iter().product::<i32>();
    a.reshape(&[n]).unwrap().as_slice::<f32>().to_vec()
}

#[test]
#[ignore = "needs the local 35GB checkpoint + dumped golden; run with --ignored"]
fn t2i_realweight_matches_reference() {
    let snap = snapshot_dir();
    let fix = fixture();
    if !snap.exists() || !fix.exists() {
        eprintln!(
            "skipping: snapshot ({}) or golden ({}) missing — regenerate with \
             tools/dump_sensenova_t2i_realweight.py",
            snap.display(),
            fix.display()
        );
        return;
    }

    let golden = mlx_gen::weights::Weights::from_file(&fix).expect("load golden");
    let prompt = golden.metadata("prompt").unwrap();
    let width: i32 = golden.metadata("width").unwrap().parse().unwrap();
    let height: i32 = golden.metadata("height").unwrap().parse().unwrap();
    let num_steps: usize = golden.metadata("num_steps").unwrap().parse().unwrap();
    let want_ids: Vec<i32> = golden
        .require("input_ids")
        .unwrap()
        .as_slice::<i32>()
        .to_vec();
    let raw_noise = golden.require("raw_noise").unwrap().clone();
    let want_image = golden.require("final_image").unwrap().clone();

    println!("loading checkpoint {} …", snap.display());
    let cfg = NeoChatConfig::from_dir(&snap).expect("config");
    let weights = load_raw(&snap).expect("weights");
    let tokenizer = load_tokenizer(&snap).expect("tokenizer");
    let model = T2iModel::from_weights(&weights, &cfg).expect("build T2iModel");

    // Independent tokenizer check: the generate query must encode to the reference ids.
    let query = format!(
        "{}<think>\n\n</think>\n\n<img>",
        mlx_gen_sensenova::build_neo1_query(prompt, mlx_gen_sensenova::SYSTEM_MESSAGE_FOR_GEN)
    );
    let got_ids = tokenizer.encode_ids(&query, true).unwrap();
    assert_eq!(
        got_ids, want_ids,
        "prompt query token ids must match the reference"
    );

    let opts = T2iOptions {
        cfg_scale: 1.0,
        num_steps,
        timestep_shift: 1.0,
        enable_timestep_shift: true,
        t_eps: 0.02,
        ..Default::default()
    };
    let out = model
        .generate(
            &tokenizer,
            prompt,
            width,
            height,
            &opts,
            Some(&raw_noise),
            None,
        )
        .expect("generate");

    let got = flat(&out.image);
    let want = flat(&want_image);
    assert_eq!(got.len(), want.len());
    assert!(
        got.iter().all(|v| v.is_finite()),
        "image has non-finite values"
    );

    // Directional (cosine) + structural (peak-rel) similarity.
    let dot: f64 = got
        .iter()
        .zip(&want)
        .map(|(&a, &b)| a as f64 * b as f64)
        .sum();
    let na: f64 = got.iter().map(|&a| (a as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = want.iter().map(|&b| (b as f64).powi(2)).sum::<f64>().sqrt();
    let cosine = dot / (na * nb + 1e-12);
    let peak = want.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-6);
    let max_abs = got
        .iter()
        .zip(&want)
        .fold(0f32, |m, (&a, &b)| m.max((a - b).abs()));
    let peak_rel = max_abs / peak;
    let mean_got = got.iter().sum::<f32>() / got.len() as f32;
    let mean_want = want.iter().sum::<f32>() / want.len() as f32;
    println!(
        "real-weight e2e: cosine={cosine:.5}  peak-rel={peak_rel:.3e}  mean got={mean_got:.4} want={mean_want:.4}"
    );

    assert!(
        cosine > 0.99,
        "directional similarity {cosine:.5} below 0.99"
    );
    assert!(peak_rel < 0.1, "peak-rel {peak_rel:.3e} exceeds 0.10");
}
