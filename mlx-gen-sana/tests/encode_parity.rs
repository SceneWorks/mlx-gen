//! DC-AE **encoder** parity gate vs diffusers `AutoencoderDC.encode` (img2img, sc-10190).
//!
//! `#[ignore]`d: needs the real `dc-ae-f32c32-sana-1.0` weights (~1.25 GB) and a golden produced by
//! `tools/dump_dcae_encode_golden.py` (a fixed input image + its raw encoder latent). This test
//! encodes the SAME image through the Rust `DcAeEncoder` and checks it reproduces the diffusers latent.
//!
//! "Divergence is not rounding": the encoder is dense f32, so the only expected gap is Metal's
//! reduced-precision matmul (~1e-3 relative) compounded over depth. A large `mean_rel` = a real port
//! bug — a wrong pixel-unshuffle channel packing, a missing out-shortcut, a stride-2-conv-vs-unshuffle
//! downsample mixup, or a swapped encoder `layers_per_block`.
//!
//! Run:
//!   SANA_DCAE_WEIGHTS=/path/vae/diffusion_pytorch_model.safetensors \
//!   cargo test -p mlx-gen-sana --test encode_parity -- --ignored --nocapture

use mlx_rs::ops::{abs, max as max_op, subtract, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_sana::{DcAeConfig, DcAeEncoder};

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
#[ignore = "needs dc-ae-f32c32-sana-1.0 weights + dump_dcae_encode_golden.py golden"]
fn encode_matches_diffusers() {
    let weights_path = std::env::var("SANA_DCAE_WEIGHTS").expect("set SANA_DCAE_WEIGHTS");
    let golden_path = std::env::var("SANA_DCAE_ENCODE_GOLDEN").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/dcae_encode_golden.safetensors"
        )
        .into()
    });

    let golden = Weights::from_file(&golden_path).expect("load golden");
    let image = golden.require("image").expect("golden image"); // [1,3,H,W] NCHW, [-1,1]
    let want = golden.require("latent").expect("golden latent"); // [1,32,H/32,W/32] NCHW (raw)

    let weights = Weights::from_file(&weights_path).expect("load weights");
    let encoder = DcAeEncoder::from_weights(&weights, DcAeConfig::sana_f32c32()).expect("build");
    let got = encoder.encode(image, &Default::default()).expect("encode"); // [1,32,H/32,W/32] NCHW (raw, pre-scale)

    assert_eq!(got.shape(), want.shape(), "shape");
    let peak = peak_rel(&got, want);
    let mean = mean_rel(&got, want);
    println!("DC-AE encode parity vs diffusers: peak_rel={peak:.5}  mean_rel={mean:.5}");

    // Same convention as `decode_parity.rs`: `mean_rel` is the faithfulness gate (a port bug wrecks
    // the mean; Metal's reduced-precision matmul does not), `peak_rel` is the looser
    // attention-normalizer precision ceiling.
    assert!(
        mean < 1e-2,
        "mean_rel {mean} too high — that IS a port bug, not rounding"
    );
    assert!(
        peak < 0.10,
        "peak_rel {peak} above the attention-normalizer precision ceiling"
    );
}
