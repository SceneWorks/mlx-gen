//! sc-5988 — Ideogram 4 end-to-end smoke: load the full pipeline (2 DiTs + Qwen3-VL TE + VAE +
//! tokenizer) and generate a real image from the model's native JSON caption, tokenized **natively
//! in Rust** (no Python). Proves the engine runs end-to-end on Mac (not bit-parity — that's the
//! per-component tests; tokenizer parity is `tests/tokenizer_parity.rs`).
//!
//! `#[ignore]` — needs the converted snapshot (~53 GB). Run:
//!   IDEOGRAM4_MLX=~/.cache/ideogram4-mlx-convert \
//!     cargo test -p mlx-gen-ideogram --test smoke -- --ignored --nocapture

mod common;

use std::path::PathBuf;

use common::CAPTION_JSON;
use mlx_gen::array::host_i32;
use mlx_gen_ideogram::Ideogram4Pipeline;
use mlx_rs::Dtype;

fn snapshot_dir() -> PathBuf {
    std::env::var("IDEOGRAM4_MLX")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME")).join(".cache/ideogram4-mlx-convert")
        })
}

#[test]
#[ignore = "needs converted weights (~53 GB)"]
fn smoke_generates_image() {
    let pipe = Ideogram4Pipeline::load(&snapshot_dir()).expect("load pipeline");
    let ids = pipe.tokenize(CAPTION_JSON).expect("tokenize JSON caption");
    println!("native JSON caption → {} tokens; generating …", ids.len());

    let envn = |k: &str, d: u32| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let (h, w) = (
        envn("IDEOGRAM4_SMOKE_RES", 256),
        envn("IDEOGRAM4_SMOKE_RES", 256),
    );
    let steps = envn("IDEOGRAM4_SMOKE_STEPS", 8) as usize;
    let guidance = std::env::var("IDEOGRAM4_SMOKE_GUIDANCE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7.0f32);
    let img = pipe
        .generate(&ids, h, w, steps, guidance, 0.5, 0)
        .expect("generate");
    assert_eq!(img.shape(), &[h as i32, w as i32, 3], "image shape");

    // Host-extract the RGB and assert it isn't a constant/degenerate frame.
    let px = host_i32(&img.as_dtype(Dtype::Int32).unwrap()).unwrap();
    let (min, max) = (*px.iter().min().unwrap(), *px.iter().max().unwrap());
    let mean = px.iter().map(|&v| v as f64).sum::<f64>() / px.len() as f64;
    println!(
        "image px range [{min}, {max}], mean {mean:.1}, {} px @ {h}x{w}/{steps} steps",
        px.len()
    );
    assert!(
        max > min,
        "degenerate (constant) image — pipeline produced no signal"
    );
    assert!(min >= 0 && max <= 255, "px out of u8 range");

    let bytes: Vec<u8> = px.iter().map(|&v| v as u8).collect();
    let out = std::env::temp_dir().join("ideogram4_smoke.png");
    image::RgbImage::from_raw(w, h, bytes)
        .unwrap()
        .save(&out)
        .unwrap();
    println!("wrote {}", out.display());
}
