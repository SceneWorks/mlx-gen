//! S2 parity gate: the Wan 2.1 `WanVae` (z16) must reproduce the `mlx_video` reference's decode +
//! chunked encode.
//!
//! The z16 WanVAE's production weights are not on disk (only the 5B's z48 vae22 is), so — unlike the
//! S1/S3 11-GB gates — this runs against a **self-contained committed fixture**: a tiny `dim=4`
//! instance with seeded random weights + reference decode/encode IO (`tools/dump_s2_fixtures.py`,
//! ~1.2 MB). The architecture is dimension-parametric, so this exercises every path (causal 3-D
//! conv, channel-L2 norm, per-frame attention, temporal up/down `time_conv`, the chunked-encode
//! `feat_cache`, mean/std denorm). It runs on Metal in CI — no `#[ignore]`.
//!
//! Honors "divergence is not rounding": the reference runs the VAE in f32; this port does too. The
//! only expected gap is the float-summation ordering between mlx `conv3d` and the reference's
//! conv2d-per-time-window decomposition of the same convolution — bounded and root-caused below.

use mlx_gen::weights::Weights;
use mlx_gen_wan::WanVae;
use mlx_rs::Array;

fn fixture() -> Weights {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/s2_vae.safetensors"
    );
    Weights::from_file(path)
        .unwrap_or_else(|e| panic!("read {path}: {e} (run tools/dump_s2_fixtures.py)"))
}

/// `(max|Δ|, Σ|Δ| / Σ|ref|)` over two equal-length f32 slices.
fn diff(got: &[f32], exp: &[f32]) -> (f32, f64) {
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut sum_ref = 0f64;
    for (g, e) in got.iter().zip(exp.iter()) {
        let d = (g - e).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
        sum_ref += e.abs() as f64;
    }
    (max_abs, sum_abs / sum_ref.max(1e-9))
}

#[test]
fn wan_vae_decode_matches_reference() {
    let w = fixture();
    let vae = WanVae::from_weights(&w).expect("build WanVae");

    let dec_in = w.require("dec_in").expect("dec_in");
    let exp = w.require("dec_out").expect("dec_out");
    let got = vae.decode(dec_in).expect("decode");
    assert_eq!(got.shape(), exp.shape(), "decode output shape");

    let (max_abs, mean_rel) = diff(got.as_slice::<f32>(), exp.as_slice::<f32>());
    println!(
        "[decode] shape={:?} max|Δ|={max_abs:.3e} mean_rel={mean_rel:.3e}",
        got.shape()
    );
    assert!(
        mean_rel < 1e-3,
        "decode diverged from reference: mean_rel={mean_rel:.3e} max|Δ|={max_abs:.3e}"
    );
}

#[test]
fn wan_vae_encode_matches_reference() {
    let w = fixture();
    let vae = WanVae::from_weights(&w).expect("build WanVae");

    let enc_in = w.require("enc_in").expect("enc_in");
    let exp = w.require("enc_out").expect("enc_out");
    let got = vae.encode(enc_in).expect("encode");
    assert_eq!(got.shape(), exp.shape(), "encode output shape");

    let (max_abs, mean_rel) = diff(got.as_slice::<f32>(), exp.as_slice::<f32>());
    println!(
        "[encode] shape={:?} max|Δ|={max_abs:.3e} mean_rel={mean_rel:.3e}",
        got.shape()
    );
    assert!(
        mean_rel < 1e-3,
        "encode diverged from reference: mean_rel={mean_rel:.3e} max|Δ|={max_abs:.3e}"
    );
}

/// `encode_sample` (the `.sample()` path used by Bernini's video `get_vae_features`) reduces to
/// `encode` (the `.mode()` path) when the injected Gaussian noise is zero — both are
/// `normalize(mean)`. This gates that the sample path shares the (already-parity'd) chunked encoder
/// and that the reparameterize plumbing is a no-op at `eps = 0`. The clamp/exp/std·eps formula itself
/// is unit-tested in `vae.rs` (`reparameterize_matches_closed_form`).
#[test]
fn wan_vae_encode_sample_eps0_equals_mode() {
    let w = fixture();
    let vae = WanVae::from_weights(&w).expect("build WanVae");

    let enc_in = w.require("enc_in").expect("enc_in");
    let mode = vae.encode(enc_in).expect("encode");
    let eps = Array::zeros::<f32>(mode.shape()).expect("zeros eps");
    let sampled = vae.encode_sample(enc_in, &eps).expect("encode_sample");

    let (max_abs, mean_rel) = diff(sampled.as_slice::<f32>(), mode.as_slice::<f32>());
    println!("[encode_sample eps=0] max|Δ|={max_abs:.3e} mean_rel={mean_rel:.3e}");
    assert!(
        max_abs < 1e-6,
        "encode_sample(eps=0) must equal encode (mode): max|Δ|={max_abs:.3e}"
    );
}
