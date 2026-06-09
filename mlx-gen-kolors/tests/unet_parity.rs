//! Kolors U-Net single-forward parity vs diffusers (sc-3093) — validates the ChatGLM3 conditioning
//! wiring: the auto-detected `encoder_hid_proj` (4096→2048 context projection) + the 5632-wide
//! `add_embedding` (pooled 4096 + 6·256 time-ids).
//!
//! `#[ignore]`d: needs the `Kwai-Kolors/Kolors-diffusers` `unet/` weights + the golden from
//! `tools/dump_kolors_unet_golden.py`. Feeds the golden's exact inputs (NHWC latents, timestep,
//! ChatGLM3 context [1,256,4096], pooled [1,4096], time_ids [1,6]) and checks the predicted eps
//! against the diffusers reference. f32 both sides; the floor is the torch(CPU)-vs-MLX(Metal)
//! cross-backend U-Net floor — a wiring bug (wrong projection, bad add_embedding width) diverges
//! orders of magnitude past it.
//!
//! Run: `cargo test -p mlx-gen-kolors --release --test unet_parity -- --ignored --nocapture`

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_kolors::unet::load_unet_kolors_dtype;
use mlx_rs::{Array, Dtype};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/kolors_unet_golden.safetensors"
);

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("KOLORS_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Kwai-Kolors--Kolors-diffusers/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn rel_stats(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let mabs = b.iter().map(|v| v.abs()).sum::<f32>() / b.len() as f32;
    let mean_diff = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32;
    (max_diff / peak, mean_diff / mabs.max(1e-12))
}

#[test]
#[ignore = "needs the Kolors snapshot unet/ + tools/golden/kolors_unet_golden.safetensors"]
fn kolors_unet_single_forward_matches_diffusers() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let timestep: f32 = g.metadata("timestep").unwrap().parse().unwrap();
    let unet = load_unet_kolors_dtype(&snapshot(), Dtype::Float32).expect("load Kolors unet");

    let eps = unet
        .forward(
            g.require("latents").unwrap(),
            timestep,
            g.require("conditioning").unwrap(), // ChatGLM3 context [1,256,4096] → encoder_hid_proj
            g.require("pooled").unwrap(),       // [1,4096] → 5632 add_embedding
            g.require("time_ids").unwrap(),     // [1,6]
        )
        .expect("kolors unet forward");

    let golden = g.require("eps").unwrap();
    assert_eq!(
        eps.shape(),
        golden.shape(),
        "eps shape {:?} vs {:?}",
        eps.shape(),
        golden.shape()
    );
    let (peak_rel, mean_rel) = rel_stats(&eps, golden);
    println!(
        "kolors unet eps {:?}: peak_rel={peak_rel:.3e} mean_rel={mean_rel:.3e}",
        eps.shape()
    );
    // Observed peak_rel ~5e-4 / mean_rel ~3.8e-4 (torch-CPU vs MLX-Metal f32 U-Net floor); ~10×
    // headroom for minor machine variance. A wiring bug is orders of magnitude past this.
    assert!(
        peak_rel < 5e-3,
        "Kolors U-Net forward diverged: peak_rel {peak_rel:.3e}"
    );
    assert!(
        mean_rel < 1e-3,
        "Kolors U-Net forward diverged: mean_rel {mean_rel:.3e}"
    );
    println!(
        "✓ Kolors U-Net single forward matches diffusers (encoder_hid_proj + 5632 add_embedding)"
    );
}
