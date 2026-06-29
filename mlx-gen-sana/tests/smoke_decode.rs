//! Real-weight **smoke** test for the DC-AE decoder (spike sc-8486).
//!
//! `#[ignore]`d: needs the real `dc-ae-f32c32-sana-1.0` `diffusion_pytorch_model.safetensors`
//! (~1.25 GB, f32). This is NOT a parity gate — it proves the ported decoder *runs* on the real
//! weights on Metal end-to-end (correct shapes, finite output, sane range) before the torch golden
//! lands. The numeric parity gate (vs diffusers `AutoencoderDC.decode`) is `decode_parity.rs`.
//!
//! Run:
//!   SANA_DCAE_WEIGHTS=/path/diffusion_pytorch_model.safetensors \
//!   cargo test -p mlx-gen-sana --test smoke_decode -- --ignored --nocapture

use mlx_rs::ops::{max as max_op, min as min_op, sum};
use mlx_rs::random::normal;

use mlx_gen::weights::Weights;
use mlx_gen_sana::{DcAeConfig, DcAeDecoder};

#[test]
#[ignore = "needs dc-ae-f32c32-sana-1.0 safetensors (~1.25 GB); set SANA_DCAE_WEIGHTS"]
fn decoder_runs_on_real_weights() {
    let path = std::env::var("SANA_DCAE_WEIGHTS").expect("set SANA_DCAE_WEIGHTS");
    let weights = Weights::from_file(&path).expect("load weights");
    let decoder = DcAeDecoder::from_weights(&weights, DcAeConfig::sana_f32c32()).expect("build");

    // A 1024px image decodes from a [B=1, C=32, 32, 32] latent (32× spatial compression).
    let key = mlx_rs::random::key(0).unwrap();
    let latent = normal::<f32>(&[1, 32, 32, 32], None, None, &key).expect("latent");

    let img = decoder.decode(&latent).expect("decode");

    assert_eq!(img.shape(), &[1, 1024, 1024, 3], "decoded image shape");

    let lo = min_op(&img, None).unwrap().item::<f32>();
    let hi = max_op(&img, None).unwrap().item::<f32>();
    let total = sum(&img, None).unwrap().item::<f32>();
    println!("decoded 1024² image: min={lo:.4} max={hi:.4} sum={total:.1}");

    assert!(
        lo.is_finite() && hi.is_finite() && total.is_finite(),
        "non-finite output"
    );
    // DC-AE reconstructs into image space; a random latent won't be a natural image but values must
    // stay in a bounded range (the reference clamps to [-1,1] downstream). Guard against blow-up.
    assert!(
        hi - lo > 1e-3,
        "output is constant — graph likely degenerate"
    );
    assert!(
        lo > -50.0 && hi < 50.0,
        "output range exploded: [{lo}, {hi}]"
    );
}
