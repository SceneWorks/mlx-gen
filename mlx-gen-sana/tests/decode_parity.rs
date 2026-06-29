//! DC-AE decoder **parity** gate vs diffusers `AutoencoderDC` (spike sc-8486 GO/NO-GO).
//!
//! `#[ignore]`d: needs the real `dc-ae-f32c32-sana-1.0` weights (~1.25 GB) and a golden produced by
//! `tools/dump_dcae_golden.py` (latent + reference raw-decoder image). This test decodes the SAME
//! latent through the Rust port and checks it reproduces the diffusers output.
//!
//! "Divergence is not rounding": the decoder is dense f32, so the only expected gap is Metal's
//! reduced-precision matmul (~1e-3 relative) compounded over depth. A large gap = a real port bug.
//!
//! Run:
//!   SANA_DCAE_WEIGHTS=/path/diffusion_pytorch_model.safetensors \
//!   SANA_DCAE_GOLDEN=/path/dcae_golden.safetensors \
//!   cargo test -p mlx-gen-sana --test decode_parity -- --ignored --nocapture

use mlx_rs::ops::{abs, max as max_op, subtract, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_sana::{DcAeConfig, DcAeDecoder};

fn f32(x: &Array) -> Array {
    x.as_dtype(Dtype::Float32).unwrap()
}

/// `max|Δ| / max|ref|`.
fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(f32(got), f32(want)).unwrap()).unwrap();
    let denom = max_op(abs(f32(want)).unwrap(), None).unwrap().item::<f32>();
    max_op(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

/// `Σ|Δ| / Σ|ref|`.
fn mean_rel(got: &Array, want: &Array) -> f32 {
    let num = sum(abs(subtract(f32(got), f32(want)).unwrap()).unwrap(), None).unwrap();
    let den = sum(abs(f32(want)).unwrap(), None).unwrap();
    num.item::<f32>() / den.item::<f32>().max(1e-12)
}

#[test]
#[ignore = "needs dc-ae-f32c32-sana-1.0 weights + dump_dcae_golden.py golden"]
fn decode_matches_diffusers() {
    let weights_path = std::env::var("SANA_DCAE_WEIGHTS").expect("set SANA_DCAE_WEIGHTS");
    let golden_path = std::env::var("SANA_DCAE_GOLDEN").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/dcae_decode_golden.safetensors"
        )
        .into()
    });

    let golden = Weights::from_file(&golden_path).expect("load golden");
    let latent = golden.require("latent").expect("golden latent"); // [1,32,32,32] NCHW
    let want_nchw = golden.require("image").expect("golden image"); // [1,3,1024,1024] NCHW
    let want = want_nchw.transpose_axes(&[0, 2, 3, 1]).unwrap(); // → NHWC

    let weights = Weights::from_file(&weights_path).expect("load weights");
    let decoder = DcAeDecoder::from_weights(&weights, DcAeConfig::sana_f32c32()).expect("build");
    let got = decoder.decode(latent).expect("decode"); // [1,1024,1024,3] NHWC

    assert_eq!(got.shape(), want.shape(), "shape");
    let peak = peak_rel(&got, &want);
    let mean = mean_rel(&got, &want);
    println!("DC-AE decode parity vs diffusers: peak_rel={peak:.5}  mean_rel={mean:.5}");

    // Dense f32 path. `mean_rel` is the faithfulness gate — a port bug (wrong transpose/op order/
    // layout) wrecks the mean; rounding does not. Measured mean_rel ≈ 0.005.
    //
    // `peak_rel` is looser by design: the error-map analysis (tools/dump_dcae_golden.py +
    // tests/dump_got diagnostic, sc-8486) showed the worst-case is ~0.07 absolute at <0.01% of
    // pixels (2 of 3.1M > 0.06), interior-not-border, located where the linear-attention 1/(Σ+eps)
    // normalizer amplifies Metal's reduced-precision (tf32-like) matmul noise. That is precision,
    // not a bug, so the peak ceiling tracks that ceiling rather than the mean.
    assert!(
        mean < 1e-2,
        "mean_rel {mean} too high — that IS a port bug, not rounding"
    );
    assert!(
        peak < 0.10,
        "peak_rel {peak} above the attention-normalizer precision ceiling"
    );
}
