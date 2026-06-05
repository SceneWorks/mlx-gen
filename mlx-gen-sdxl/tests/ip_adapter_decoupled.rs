//! sc-3059: SDXL IP-Adapter decoupled cross-attention engine checks (real weights).
//!
//! `#[ignore]`d — needs the SDXL base snapshot + `h94/IP-Adapter` (`ip-adapter-plus_sdxl_vit-h`).
//! Run: cargo test -p mlx-gen-sdxl --release --test ip_adapter_decoupled -- --ignored --nocapture
//!
//! Validates the injection primitive + the 70-layer walk-order remap WITHOUT a deep golden:
//!   1. Installing the real K/V pairs must succeed AND every cross-attn `forward_with_ip` reshape
//!      must line up — a wrong walk order maps a 640-d projection onto a 1280-d layer and panics.
//!   2. `forward_with_ip(scale = 0)` == plain `forward` **byte-for-byte** (the IP term is `0·o_ip`),
//!      proving the base path is untouched.
//!   3. `forward_with_ip(scale > 0)` != plain `forward` — the IP branch is actually wired in.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_sdxl::{load_ip_kv_pairs, load_unet_dtype, text_time_ids};
use mlx_rs::{Array, Dtype};

fn sdxl_snapshot() -> PathBuf {
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

fn ip_weights() -> Weights {
    let home = std::env::var("HOME").unwrap();
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--h94--IP-Adapter/snapshots");
    let dir = std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir for h94/IP-Adapter")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir");
    let mut w =
        Weights::from_file(dir.join("sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors")).unwrap();
    w.cast_all(Dtype::Float16).unwrap();
    w
}

fn randn(shape: &[i32], seed: u64) -> Array {
    mlx_rs::random::seed(seed).unwrap();
    mlx_rs::random::normal::<f32>(shape, None, None, None)
        .unwrap()
        .as_dtype(Dtype::Float16)
        .unwrap()
}

fn max_abs(a: &Array, b: &Array) -> f32 {
    let d = mlx_rs::ops::abs(mlx_rs::ops::subtract(a, b).unwrap()).unwrap();
    mlx_rs::ops::max(&d, None)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .item::<f32>()
}

#[test]
#[ignore = "needs the SDXL base snapshot + h94/IP-Adapter weights"]
fn ip_decoupled_attn_remap_and_scale_zero() {
    let mut unet = load_unet_dtype(&sdxl_snapshot(), Dtype::Float16).unwrap();
    let pairs = load_ip_kv_pairs(&ip_weights()).unwrap();
    assert_eq!(pairs.len(), 70, "SDXL IP-Adapter has 70 cross-attn layers");
    // Panics here if the walk order maps a projection onto a mismatched-dim layer.
    unet.install_ip_adapter(pairs).unwrap();

    // CFG-batched dummy inputs (B=2).
    let latents = randn(&[2, 64, 64, 4], 1);
    let cond = randn(&[2, 77, 2048], 2);
    let pooled = randn(&[2, 1280], 3);
    let time_ids = text_time_ids(2);
    let ip_tokens = randn(&[2, 16, 2048], 4);
    let t = 500.0;

    let eps_plain = unet
        .forward(&latents, t, &cond, &pooled, &time_ids)
        .unwrap();
    let eps_ip0 = unet
        .forward_with_ip(&latents, t, &cond, &pooled, &time_ids, (&ip_tokens, 0.0))
        .unwrap();
    let eps_ip = unet
        .forward_with_ip(&latents, t, &cond, &pooled, &time_ids, (&ip_tokens, 0.6))
        .unwrap();

    let d0 = max_abs(&eps_plain, &eps_ip0);
    let di = max_abs(&eps_plain, &eps_ip);
    println!("[ip-adapter] forward vs forward_with_ip(scale=0): max|Δ|={d0:.3e}");
    println!("[ip-adapter] forward vs forward_with_ip(scale=0.6): max|Δ|={di:.3e}");
    assert_eq!(d0, 0.0, "scale=0 must be byte-identical to plain forward");
    assert!(
        di > 1e-3,
        "scale>0 must change the prediction (IP branch wired)"
    );
}
