//! sc-2400 S4/S6: SDXL VAE decode + encode parity vs the vendored Apple reference (f32).
//!
//! `#[ignore]`d — needs the real SDXL snapshot + the golden from `tools/dump_sdxl_vae_golden.py`.
//! Run with:
//!   cargo test -p mlx-gen-sdxl --release --test vae_real_weights -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_sdxl::load_vae;
use mlx_rs::Array;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/sdxl_vae_golden.safetensors"
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
#[ignore = "needs the real SDXL snapshot + VAE golden"]
fn vae_decode_matches_vendored() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let vae = load_vae(&snapshot()).unwrap();

    let decoded = vae.decode(g.require("latents").unwrap()).unwrap();
    let golden = g.require("decoded").unwrap();
    assert_eq!(decoded.shape(), golden.shape(), "decoded shape");
    let pr = peak_rel(&decoded, golden);
    println!("vae decode {:?}: peak_rel={pr:.3e}", decoded.shape());
    assert!(pr < 5e-3, "VAE decode diverged: peak_rel {pr:.3e}");
    println!("✓ SDXL VAE decode matches the vendored reference (f32)");
}

#[test]
#[ignore = "needs the real SDXL snapshot + VAE golden"]
fn vae_encode_mean_matches_vendored() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let vae = load_vae(&snapshot()).unwrap();

    let mean = vae.encode_mean(g.require("image").unwrap()).unwrap();
    let golden = g.require("enc_mean").unwrap();
    assert_eq!(mean.shape(), golden.shape(), "enc_mean shape");
    let pr = peak_rel(&mean, golden);
    println!("vae encode_mean {:?}: peak_rel={pr:.3e}", mean.shape());
    assert!(pr < 5e-3, "VAE encode diverged: peak_rel {pr:.3e}");
    println!("✓ SDXL VAE encode (mean) matches the vendored reference (f32)");
}
