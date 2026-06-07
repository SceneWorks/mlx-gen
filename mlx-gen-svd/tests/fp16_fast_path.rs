//! SVD fp16 fast-path validation (epic 3040 follow-up to sc-3054). The dense provider path runs the
//! UNet + image encoder in **fp16** (the production `torch_dtype=float16` dtype; `Precision::Fp32`
//! selects the f32 quality path, the VAE always stays f32). These gates load the *same* source
//! weights cast to f16 vs f32 and run one forward each, so the only difference is compute precision —
//! confirming the fp16 path stays within the fp16 rounding floor of the parity-validated f32 path
//! (no dtype-flow bug, no instability). Needs the SVD checkpoint locally → `--ignored`.
//!
//! Run: `cargo test -p mlx-gen-svd --test fp16_fast_path -- --ignored --nocapture`

use mlx_rs::ops::{abs, divide, max as max_op, sqrt, square, subtract, sum};
use mlx_rs::{random, Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_svd::{ImageEncoderConfig, SvdImageEncoder, SvdUnet, UnetConfig};

fn snapshot() -> std::path::PathBuf {
    let cache = std::env::var("HF_HUB_CACHE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/huggingface/hub")
        });
    let snaps = cache
        .join("models--stabilityai--stable-video-diffusion-img2vid-xt")
        .join("snapshots");
    std::fs::read_dir(&snaps)
        .expect("svd snapshot dir")
        .next()
        .unwrap()
        .unwrap()
        .path()
}

/// rel-L2 = ‖a − b‖₂ / ‖b‖₂ (b = the f32 reference).
fn rel_l2(a: &Array, b: &Array) -> f32 {
    let diff = subtract(a, b).unwrap();
    let num = sqrt(sum(square(&diff).unwrap(), None).unwrap()).unwrap();
    let den = sqrt(sum(square(b).unwrap(), None).unwrap()).unwrap();
    divide(&num, &den).unwrap().item::<f32>()
}

#[test]
#[ignore = "needs the SVD checkpoint in the HF cache"]
fn fp16_unet_forward_within_f32_floor() {
    let snap = snapshot();
    let path = snap.join("unet/diffusion_pytorch_model.safetensors");

    let mut w32 = Weights::from_file(&path).expect("unet w");
    w32.cast_all(Dtype::Float32).unwrap();
    let unet_f32 = SvdUnet::from_weights(&w32, &UnetConfig::default()).unwrap();

    let mut w16 = Weights::from_file(&path).expect("unet w");
    w16.cast_all(Dtype::Float16).unwrap();
    let unet_f16 = SvdUnet::from_weights(&w16, &UnetConfig::default()).unwrap();

    // Deterministic CFG-batched input: [B=2, F=2, 16, 16, 8] sample + conditioning.
    let (b, f, hw) = (2, 2, 16);
    let key = random::key(3054).unwrap();
    let sample = random::normal::<f32>(&[b, f, hw, hw, 8], None, None, Some(&key)).unwrap();
    let key2 = random::key(3055).unwrap();
    let image_embeds = random::normal::<f32>(&[b, 1, 1024], None, None, Some(&key2)).unwrap();
    let added_time_ids = Array::from_slice(&[6.0f32, 127.0, 0.02, 6.0, 127.0, 0.02], &[b, 3]);
    let timestep = 0.25f32 * 7.0f32.ln(); // a representative model-timestep (0.25·ln σ)

    let out32 = unet_f32
        .forward(&sample, timestep, &image_embeds, &added_time_ids, f)
        .unwrap();
    let out16 = unet_f16
        .forward(&sample, timestep, &image_embeds, &added_time_ids, f)
        .unwrap();
    assert_eq!(out16.shape(), out32.shape());
    assert_eq!(out16.dtype(), Dtype::Float32, "forward must return f32");

    let rel = rel_l2(&out16, &out32);
    let max_abs = max_op(abs(subtract(&out16, &out32).unwrap()).unwrap(), None)
        .unwrap()
        .item::<f32>();
    println!("fp16 UNet forward vs f32: rel-L2 {rel}, max|Δ| {max_abs}");
    // fp16 carries ~3 decimal digits; one forward through the 4-stage spatiotemporal UNet stays at
    // the low-percent fp16 floor (the fused norm/SDPA reductions upcast internally).
    assert!(rel < 0.03, "fp16 UNet rel-L2 {rel} exceeds the fp16 floor");
}

#[test]
#[ignore = "needs the SVD checkpoint in the HF cache"]
fn fp16_image_encoder_within_f32_floor() {
    let snap = snapshot();
    let path = snap.join("image_encoder/model.safetensors");

    let mut w32 = Weights::from_file(&path).expect("enc w");
    w32.cast_all(Dtype::Float32).unwrap();
    let enc_f32 = SvdImageEncoder::from_weights(&w32, &ImageEncoderConfig::default()).unwrap();

    let mut w16 = Weights::from_file(&path).expect("enc w");
    w16.cast_all(Dtype::Float16).unwrap();
    let enc_f16 = SvdImageEncoder::from_weights(&w16, &ImageEncoderConfig::default()).unwrap();

    let key = random::key(3373).unwrap();
    let pixel_values = random::normal::<f32>(&[1, 224, 224, 3], None, None, Some(&key)).unwrap();

    let e32 = enc_f32.image_embeds(&pixel_values).unwrap();
    let e16 = enc_f16.image_embeds(&pixel_values).unwrap();
    assert_eq!(e16.dtype(), Dtype::Float32, "image_embeds must return f32");

    let rel = rel_l2(&e16, &e32);
    println!("fp16 image-encoder vs f32: rel-L2 {rel}");
    assert!(rel < 0.02, "fp16 encoder rel-L2 {rel} exceeds the fp16 floor");
}
