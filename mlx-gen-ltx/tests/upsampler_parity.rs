//! S4 spatial-upsampler parity vs the reference `upsample_latents` (sc-2679 S4).
//!
//! `#[ignore]`d: needs the real `ltx_2_3_base_q8` `upsampler.safetensors` (~1 GB). The committed
//! golden (`tests/fixtures/ltx_upsampler_golden.safetensors`, from
//! `tools/dump_ltx_upsampler_golden.py`) holds the reference **bf16** `upsample_latents` I/O over a
//! synthetic latent; this test loads the SAME bf16 weights and checks the Rust `upsample_latents`
//! reproduces the output.
//!
//! The upsampler is pure dense (conv + group-norm, no quantized ops), run bf16 to match the
//! production path — every op is the same mlx op at the same dtype, so the gate is tight. Honors
//! "divergence is not rounding": a >1% gap here would be a real bug.
//!
//! Run: `LTX_BASE_DIR=… cargo test -p mlx-gen-ltx --test upsampler_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max as max_op, subtract, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_ltx::upsampler::{upsample_latents, LatentUpsampler};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_upsampler_golden.safetensors"
);

fn base_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("LTX_BASE_DIR") {
        return d.into();
    }
    let home = std::env::var("HOME").unwrap();
    std::path::PathBuf::from(home)
        .join("Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8")
}

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
#[ignore = "needs ltx_2_3_base_q8 upsampler.safetensors (~1 GB)"]
fn upsampler_matches_reference() {
    let dir = base_dir();
    let w = Weights::from_file(dir.join("upsampler.safetensors")).expect("upsampler.safetensors");
    let up = LatentUpsampler::from_weights(&w).expect("build LatentUpsampler");

    let g = Weights::from_file(GOLDEN).expect("golden (run tools/dump_ltx_upsampler_golden.py)");
    let latent = g.require("latent").unwrap();
    let mean = g.require("latent_mean").unwrap();
    let std = g.require("latent_std").unwrap();
    let want = g.require("output").unwrap();

    let got = upsample_latents(latent, &up, mean, std).expect("upsample_latents");
    assert_eq!(got.shape(), want.shape(), "output shape");
    let (pr, mr) = (peak_rel(&got, want), mean_rel(&got, want));
    eprintln!(
        "upsampler peak_rel = {pr:.3e} mean_rel = {mr:.3e} shape={:?}",
        got.shape()
    );
    assert!(mr < 5e-3, "upsampler mean_rel {mr:.3e} too high");
    assert!(pr < 1e-2, "upsampler peak_rel {pr:.3e} too high");
}
