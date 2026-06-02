//! sc-2400 S3: SDXL U-Net single-forward parity vs the vendored Apple reference (f32).
//!
//! `#[ignore]`d — needs the real SDXL snapshot + the golden from `tools/dump_sdxl_unet_golden.py`.
//! Run with:
//!   cargo test -p mlx-gen-sdxl --release --test unet_real_weights -- --ignored --nocapture
//!
//! Feeds the golden's exact inputs (latents, timestep, dual-CLIP conditioning, pooled, the hardcoded
//! `[512,512,0,0,512,512]` micro-conditioning) and checks the predicted eps. Reference + Rust both
//! f32, so the tolerance is tight — this isolates the entire U-Net forward (down/mid/up, cross-attn,
//! the time + text_time embeddings, the skip connections).

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_sdxl::load_unet;
use mlx_rs::Array;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/sdxl_unet_golden.safetensors"
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
#[ignore = "needs the real SDXL snapshot + U-Net golden"]
fn unet_single_forward_matches_vendored() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let timestep: f32 = g.metadata("timestep").unwrap().parse().unwrap();
    let unet = load_unet(&snapshot()).unwrap();

    let eps = unet
        .forward(
            g.require("latents").unwrap(),
            timestep,
            g.require("conditioning").unwrap(),
            g.require("pooled").unwrap(),
            g.require("time_ids").unwrap(),
        )
        .unwrap();

    let golden = g.require("eps").unwrap();
    assert_eq!(eps.shape(), golden.shape(), "eps shape");
    let pr = peak_rel(&eps, golden);
    let mean_rel = {
        let n = golden.shape().iter().product::<i32>();
        let (a, b) = (eps.reshape(&[n]).unwrap(), golden.reshape(&[n]).unwrap());
        let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
        let mabs = b.iter().map(|v| v.abs()).sum::<f32>() / b.len() as f32;
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32 / mabs
    };
    println!(
        "unet eps {:?}: peak_rel={pr:.3e} mean_rel={mean_rel:.3e}",
        eps.shape()
    );
    assert!(pr < 5e-3, "U-Net forward diverged: peak_rel {pr:.3e}");
    println!("✓ SDXL U-Net single forward matches the vendored reference (f32)");
}
