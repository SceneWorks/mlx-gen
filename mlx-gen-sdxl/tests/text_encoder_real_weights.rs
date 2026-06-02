//! sc-2400 S2: dual-CLIP text-encoder parity vs the vendored Apple reference (f32).
//!
//! `#[ignore]`d — needs the real SDXL snapshot + the golden from
//! `tools/dump_sdxl_text_encoder_golden.py`. Run with:
//!   cargo test -p mlx-gen-sdxl --release --test text_encoder_real_weights -- --ignored --nocapture
//!
//! Validates the SDXL conditioning exactly as the reference builds it:
//! `concat(te1.hidden_states[-2], te2.hidden_states[-2])` (penultimate layer, before final LN) and
//! `te2.pooled` (projected EOS). Reference + Rust both f32, so tolerances are tight — this isolates
//! the encoder math (the production fp16 rounding is absorbed into the e2e gate, S5).

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_sdxl::{load_text_encoder_1, load_text_encoder_2};
use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/sdxl_text_encoder_golden.safetensors"
);

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

/// Peak-relative error `max|a-b| / max|b|`.
fn peak_rel(a: &Array, b: &Array) -> f32 {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    max_diff / peak
}

#[test]
#[ignore = "needs the real SDXL snapshot + text-encoder golden"]
fn dual_clip_conditioning_matches_vendored() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let input_ids = g.require("input_ids").unwrap();
    let snap = snapshot();

    let te1 = load_text_encoder_1(&snap).unwrap();
    let te2 = load_text_encoder_2(&snap).unwrap();
    let o1 = te1.forward(input_ids).unwrap();
    let o2 = te2.forward(input_ids).unwrap();

    // hidden_states[-2] (penultimate encoder layer, before final LN).
    let h1 = &o1.hidden_states[o1.hidden_states.len() - 2];
    let h2 = &o2.hidden_states[o2.hidden_states.len() - 2];
    let pr_h1 = peak_rel(h1, g.require("te1_hidden_m2").unwrap());
    let pr_h2 = peak_rel(h2, g.require("te2_hidden_m2").unwrap());
    println!("te1 hidden[-2] peak_rel={pr_h1:.3e} (768-wide, 12L)");
    println!("te2 hidden[-2] peak_rel={pr_h2:.3e} (1280-wide, 32L)");
    assert!(pr_h1 < 5e-4, "te1 hidden[-2] diverged: {pr_h1:.3e}");
    assert!(pr_h2 < 5e-4, "te2 hidden[-2] diverged: {pr_h2:.3e}");

    // Full SDXL conditioning = concat([h1, h2], -1) -> [1, N, 2048].
    let cond = concatenate_axis(&[h1, h2], 2).unwrap();
    let golden_cond = g.require("conditioning").unwrap();
    assert_eq!(cond.shape(), golden_cond.shape(), "conditioning shape");
    let pr_cond = peak_rel(&cond, golden_cond);
    println!("conditioning {:?} peak_rel={pr_cond:.3e}", cond.shape());
    assert!(pr_cond < 5e-4, "conditioning diverged: {pr_cond:.3e}");

    // Pooled (te2.pooled_output, projected EOS) -> [1, 1280].
    let golden_pooled = g.require("pooled").unwrap();
    assert_eq!(o2.pooled.shape(), golden_pooled.shape(), "pooled shape");
    let pr_pool = peak_rel(&o2.pooled, golden_pooled);
    println!("pooled {:?} peak_rel={pr_pool:.3e}", o2.pooled.shape());
    assert!(pr_pool < 5e-4, "pooled diverged: {pr_pool:.3e}");

    println!("✓ dual-CLIP conditioning + pooled match the vendored reference (f32)");
}
