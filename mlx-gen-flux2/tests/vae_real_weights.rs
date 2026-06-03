//! sc-2346 S2: real-weights parity for the FLUX.2 VAE (decode_packed_latents + encode).
//! `#[ignore]`d — needs the real `black-forest-labs/FLUX.2-klein-9b` snapshot + the golden from
//! `tools/dump_flux2_vae_golden.py` (gitignored):
//!
//!   cd ~/repos/mflux && .venv/bin/python ~/repos/mlx-gen/tools/dump_flux2_vae_golden.py
//!   cargo test -p mlx-gen-flux2 --test vae_real_weights -- --ignored --nocapture
//!
//! The fork golden is dumped at **f32** (the Rust VAE's precision); the gate is tight. Golden
//! tensors are NCHW (fork-native); the Rust VAE is NHWC, so inputs/outputs are transposed here.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_flux2::load_vae;
use mlx_rs::{Array, Dtype};

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("MLX_GEN_FLUX2_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").expect("HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-9b/snapshots");
    std::fs::read_dir(&snaps)
        .expect("snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn golden() -> Weights {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/golden/flux2_vae.safetensors");
    Weights::from_file(&path).unwrap_or_else(|_| {
        panic!(
            "missing {} — run tools/dump_flux2_vae_golden.py",
            path.display()
        )
    })
}

fn nchw_to_nhwc(a: &Array) -> Array {
    a.as_dtype(Dtype::Float32)
        .unwrap()
        .transpose_axes(&[0, 2, 3, 1])
        .unwrap()
}

/// `(peak-relative, mean-relative)` error vs golden `b`.
fn rel(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let (xs, ys) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = ys.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let mabs = (ys.iter().map(|y| y.abs()).sum::<f32>() / ys.len() as f32).max(1e-12);
    let max_diff = xs
        .iter()
        .zip(ys)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let mean_diff = xs.iter().zip(ys).map(|(x, y)| (x - y).abs()).sum::<f32>() / xs.len() as f32;
    (max_diff / peak, mean_diff / mabs)
}

#[test]
#[ignore = "needs real FLUX.2-klein-9b snapshot + tools/golden/flux2_vae.safetensors"]
fn decode_packed_latents_matches_fork() {
    let vae = load_vae(&snapshot()).unwrap();
    let g = golden();
    let packed = nchw_to_nhwc(g.require("packed_in").unwrap()); // [1,4,4,128]
    let out = vae.decode_packed_latents(&packed).unwrap(); // [1,64,64,3]
    let want = nchw_to_nhwc(g.require("decode_out").unwrap()); // [1,64,64,3]
    assert_eq!(out.shape(), want.shape(), "decode shape");
    let (peak, mean) = rel(&out, &want);
    println!("flux2 VAE decode: peak_rel={peak:.5} mean_rel={mean:.5}");
    assert!(mean < 5e-3, "VAE decode diverged: mean_rel={mean}");
}

#[test]
#[ignore = "needs real FLUX.2-klein-9b snapshot + tools/golden/flux2_vae.safetensors"]
fn encode_matches_fork() {
    let vae = load_vae(&snapshot()).unwrap();
    let g = golden();
    let image = nchw_to_nhwc(g.require("image_in").unwrap()); // [1,64,64,3]
    let out = vae.encode_mean(&image).unwrap(); // [1,8,8,32]
    let want = nchw_to_nhwc(g.require("encode_out").unwrap()); // [1,8,8,32]
    assert_eq!(out.shape(), want.shape(), "encode shape");
    let (peak, mean) = rel(&out, &want);
    println!("flux2 VAE encode: peak_rel={peak:.5} mean_rel={mean:.5}");
    assert!(mean < 5e-3, "VAE encode diverged: mean_rel={mean}");
}
