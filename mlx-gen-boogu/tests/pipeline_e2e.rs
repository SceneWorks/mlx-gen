//! E5 (sc-6393) — Boogu Base T2I pipeline: tokenizer parity (sc-6390) + a real-weight end-to-end
//! smoke that renders a coherent image.
//!
//! `tokenizer_matches_golden` needs the snapshot + the golden (`tools/golden_dump.py`); the e2e
//! smoke additionally needs ~all of a 128 GB Mac. Run:
//!   BOOGU_BASE_DIR=<snapshot> CARGO_TARGET_DIR=~/Repos/mlx-gen/target \
//!     cargo test -p mlx-gen-boogu --test pipeline_e2e -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_boogu::tokenizer::BooguTokenizer;
use mlx_gen_boogu::{BooguPipeline, GenerateOptions};

fn snapshot_dir() -> PathBuf {
    PathBuf::from(std::env::var("BOOGU_BASE_DIR").expect("set BOOGU_BASE_DIR to the snapshot root"))
}

fn golden_path() -> PathBuf {
    std::env::var("BOOGU_GOLDEN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME"))
                .join("Repos/mlx-gen-wt-boogu/reference/goldens/boogu_golden.safetensors")
        })
}

/// sc-6390 — the rendered T2I chat template + tokenizer reproduce the reference `processor`'s
/// `tok_input_ids` token-for-token for the golden prompt.
#[test]
#[ignore = "needs the mllm tokenizer + golden (tools/golden_dump.py)"]
fn tokenizer_matches_golden() {
    let g = Weights::from_file(golden_path()).expect("golden — run tools/golden_dump.py");
    let tok = BooguTokenizer::from_snapshot(snapshot_dir()).expect("load mllm tokenizer");

    let ids = tok.t2i_ids("a red apple on a wooden table").unwrap();
    let want: Vec<i32> = g
        .require("tok_input_ids")
        .unwrap()
        .as_dtype(mlx_rs::Dtype::Int32)
        .unwrap()
        .as_slice::<i32>()
        .to_vec();

    println!("boogu tokenizer: {} ids (golden {})", ids.len(), want.len());
    assert_eq!(
        ids, want,
        "T2I tokenization must match the reference processor"
    );
}

/// Real-weight end-to-end T2I smoke: render a small image and assert it is non-degenerate, saving a
/// PNG for visual inspection. (Coherence is judged by eye on the saved file.)
#[test]
#[ignore = "needs real weights (128 GB Mac): set BOOGU_BASE_DIR"]
fn t2i_smoke() {
    let pipe = BooguPipeline::from_snapshot(snapshot_dir()).expect("load Boogu pipeline");
    let opts = GenerateOptions {
        height: 512,
        width: 512,
        steps: 28,
        text_guidance_scale: 4.0,
        seed: 0,
    };
    let img = pipe
        .generate("a red apple on a wooden table", &opts)
        .expect("generate");

    assert_eq!(img.width, 512);
    assert_eq!(img.height, 512);
    assert_eq!(img.pixels.len(), 512 * 512 * 3);

    // Non-degenerate: a real render has spread across the 0..255 range, not a flat fill.
    let (mn, mx) = img
        .pixels
        .iter()
        .fold((255u8, 0u8), |(mn, mx), &p| (mn.min(p), mx.max(p)));
    let mean = img.pixels.iter().map(|&p| p as u64).sum::<u64>() / img.pixels.len() as u64;
    println!("render stats: min={mn} max={mx} mean={mean}");
    assert!(mx - mn > 32, "render looks degenerate (min={mn} max={mx})");

    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../reference/outputs/boogu_mlx_t2i_apple_512_s28.png");
    std::fs::create_dir_all(out.parent().unwrap()).unwrap();
    image::save_buffer(
        &out,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    println!("wrote {}", out.display());
}
