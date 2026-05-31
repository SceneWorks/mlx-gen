//! sc-2348 slice 1: Qwen-Image causal-Conv3d VAE parity vs the frozen fork.
//!
//! `#[ignore]`d — needs the local golden from `tools/dump_qwen_vae_golden.py` (gitignored): the
//! fork's f32 VAE weights (keyed by the internal module tree) + fixed inputs + fork encode/decode
//! outputs. Both sides run f32, so this isolates VAE *math* parity (the disk-snapshot key remapping
//! lands with the full-model assembly). Run:
//!   cargo test -p mlx-gen-qwen-image --release --test vae_real_weights -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_gen_qwen_image::QwenVae;
use mlx_rs::Array;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/qwen_vae_golden.safetensors"
);

/// Peak- and mean-relative error vs the fork golden. Peak `max|a-b|/max|b|`; mean
/// `mean|a-b|/mean|b|`. Peak ≫ mean ⇒ localized (suspect a bug); peak ≈ mean ⇒ distributed
/// (f32 reduction-order accumulation, expected to grow with conv-net depth/size).
fn rel_errors(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let sum_abs_b: f64 = b.iter().map(|&v| v.abs() as f64).sum();
    let sum_abs_diff: f64 = a.iter().zip(b).map(|(&x, &y)| (x - y).abs() as f64).sum();
    (max_diff / peak, (sum_abs_diff / sum_abs_b) as f32)
}

#[test]
#[ignore = "needs real Qwen-Image VAE weights + local golden"]
fn vae_decode_matches_fork_golden() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let vae = QwenVae::from_weights(&g).unwrap();
    let out = vae.decode(g.require("dec_in").unwrap()).unwrap();
    let want = g.require("dec_out").unwrap();
    assert_eq!(out.shape(), want.shape(), "decode output shape");
    let (peak, mean) = rel_errors(&out, want);
    println!("VAE decode: peak-rel = {peak:.3e}, mean-rel = {mean:.3e}");
    // mean-rel is the structural-correctness gate (matches encode's ~1.1e-3); the looser peak
    // bound tolerates the few image-border pixels that diverge in f32 after upsample+conv.
    assert!(mean < 2e-3, "VAE decode mean-rel regressed: {mean:.3e}");
    assert!(peak < 1.5e-2, "VAE decode peak-rel regressed: {peak:.3e}");
}

#[test]
#[ignore = "needs real Qwen-Image VAE weights + local golden"]
fn vae_encode_matches_fork_golden() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let vae = QwenVae::from_weights(&g).unwrap();
    let out = vae.encode(g.require("enc_in").unwrap()).unwrap();
    let want = g.require("enc_out").unwrap();
    assert_eq!(out.shape(), want.shape(), "encode output shape");
    let (peak, mean) = rel_errors(&out, want);
    println!("VAE encode: peak-rel = {peak:.3e}, mean-rel = {mean:.3e}");
    assert!(mean < 2e-3, "VAE encode mean-rel regressed: {mean:.3e}");
    assert!(peak < 5e-3, "VAE encode peak-rel regressed: {peak:.3e}");
}
