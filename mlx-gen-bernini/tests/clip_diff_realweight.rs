//! sc-5139: real-weight load + forward smoke for the connector + clip-diff head (`#[ignore]`).
//!
//! Loads the converted `connector.safetensors` (12 tensors) and `vit_decoder.safetensors` (140) —
//! proving the sc-5144 converter keys match the modules — and runs `for_gen`/`for_vit` + a tiny
//! triple-CFG `sample()` at the real dims (hidden 3584, gen 4096, width 4096, 16 res blocks),
//! asserting finite outputs. Numeric correctness is covered by the f32 synthetic golden.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_bernini::clip_diff::DiffLossFm;
use mlx_gen_bernini::connector::MlpConnector;
use mlx_rs::{random, Array, Dtype};

fn snapshot() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/mlx-gen-models/bernini_planner_mlx_bf16")
}

fn randn(shape: &[i32], seed: u64) -> Array {
    let key = random::key(seed).unwrap();
    random::normal::<f32>(shape, None, None, Some(&key))
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap()
}

fn finite_max(a: &Array) -> f32 {
    a.abs().unwrap().max(None).unwrap().item::<f32>()
}

#[test]
#[ignore = "real weights: loads the connector + 16-block clip-diff head and runs forwards"]
fn clip_diff_real_weight_smoke() {
    let snap = snapshot();

    let cw = Weights::from_file(snap.join("connector.safetensors")).expect("connector weights");
    let conn = MlpConnector::from_weights(&cw, "").expect("connector");
    let x = randn(&[2, 3584], 0);
    let gen = conn.for_gen(&x).unwrap();
    let vit = conn.for_vit(&x).unwrap();
    assert_eq!(gen.shape(), &[2, 4096]);
    assert_eq!(vit.shape(), &[2, 3584]);
    assert!(finite_max(&gen).is_finite() && finite_max(&vit).is_finite());

    let vw = Weights::from_file(snap.join("vit_decoder.safetensors")).expect("vit_decoder weights");
    // depth 16, in/z channels 3584, shift 2.0 (clip_diff_cfg).
    let mut head = DiffLossFm::from_weights(&vw, "net", 16, 3584, 2.0).expect("clip-diff head");
    // triple-CFG sample: cond z tiled x3, base noise [N, 3584], 3 denoise steps.
    let n = 2i32;
    let z = randn(&[3 * n, 3584], 1);
    let noise = randn(&[n, 3584], 2);
    let sample = head.sample(&z, 1.4, 3, Some(1.2), &noise).expect("sample");
    assert_eq!(sample.shape(), &[3 * n, 3584]);
    let m = finite_max(&sample);
    assert!(
        m.is_finite() && m > 0.0,
        "sampled vit embed finite & non-trivial (max {m})"
    );
    println!(
        "clip-diff real-weight ok: for_gen {:?} sample {:?} max|·|={m:.4}",
        gen.shape(),
        sample.shape()
    );
}
