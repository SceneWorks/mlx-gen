//! SCAIL-2 resize / CLIP-preprocess parity gate (sc-5443 / sc-5446).
//!
//! Compares the `F.interpolate`-faithful kernels ([`mlx_gen_scail2::interpolate`] bicubic/bilinear and
//! [`mlx_gen_scail2::clip_preprocess`]) against torch `F.interpolate(align_corners=False)`. Fixtures:
//!
//! ```text
//! SCAIL2_PARITY_DIR=~/.cache/scail2-parity \
//!   ~/mlx-flux-venv/bin/python _vendor/scail2/_gen_resize_parity_fixtures.py
//! ```
//!
//! `#[ignore]` (needs the fixtures). Run with `cargo test -p mlx-gen-scail2 -- --ignored`.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_scail2::{clip_preprocess, interpolate, Interp};
use mlx_rs::{Array, Dtype};

fn parity_dir() -> PathBuf {
    std::env::var("SCAIL2_PARITY_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/scail2-parity")
        })
}

fn flat(a: &Array) -> Vec<f32> {
    a.reshape(&[-1])
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

fn max_abs(a: &Array, b: &Array) -> f32 {
    let (va, vb) = (flat(a), flat(b));
    assert_eq!(va.len(), vb.len(), "shape mismatch");
    va.iter()
        .zip(vb.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

#[test]
#[ignore = "needs locally-generated fixtures (see module doc); run with --ignored on macOS"]
fn resize_parity() {
    let dir = parity_dir().join("resize");
    let io_path = dir.join("io.safetensors");
    assert!(
        io_path.exists(),
        "missing fixtures at {} — generate with \
         `SCAIL2_PARITY_DIR={} ~/mlx-flux-venv/bin/python _vendor/scail2/_gen_resize_parity_fixtures.py`",
        dir.display(),
        parity_dir().display(),
    );
    let io = Weights::from_file(&io_path).unwrap();

    // bicubic downscale (the CLIP-encode resize).
    let bic = interpolate(io.require("bicubic_in").unwrap(), 16, 16, Interp::Bicubic).unwrap();
    let bic_ref = io.require("bicubic_out").unwrap();
    assert_eq!(bic.shape(), bic_ref.shape());
    let d_bic = max_abs(&bic, bic_ref);

    // bilinear 0.5× (pose / mask half-resolution).
    let bil = interpolate(io.require("bilinear_in").unwrap(), 16, 24, Interp::Bilinear).unwrap();
    let bil_ref = io.require("bilinear_out").unwrap();
    assert_eq!(bil.shape(), bil_ref.shape());
    let d_bil = max_abs(&bil, bil_ref);

    // full CLIP preprocess (bicubic → [0,1] → normalize).
    let clip = clip_preprocess(io.require("clip_in").unwrap(), 24).unwrap();
    let clip_ref = io.require("clip_out").unwrap();
    assert_eq!(clip.shape(), clip_ref.shape());
    let d_clip = max_abs(&clip, clip_ref);

    println!("bicubic max|Δ| {d_bic:.2e}  bilinear max|Δ| {d_bil:.2e}  clip-preprocess max|Δ| {d_clip:.2e}");
    assert!(d_bic < 1e-5, "bicubic max|Δ| {d_bic} exceeds 1e-5");
    assert!(d_bil < 1e-5, "bilinear max|Δ| {d_bil} exceeds 1e-5");
    assert!(
        d_clip < 1e-5,
        "clip-preprocess max|Δ| {d_clip} exceeds 1e-5"
    );
}
