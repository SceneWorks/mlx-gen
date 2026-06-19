//! E9 (sc-6397) — assemble a pre-quantized **turnkey** snapshot and prove it loads + runs through the
//! exact `BooguPipeline::from_snapshot` path (the published `SceneWorks/boogu-image-mlx/<variant>`).
//!
//! `#[ignore]` (needs real weights + ~one variant's disk). Drive per variant via env:
//!   BOOGU_ASM_SRC=<dense snapshot root>   (defaults to BOOGU_BASE_DIR)
//!   BOOGU_ASM_OUT=<turnkey output dir>    (defaults to <tmp>/boogu_turnkey)
//!   BOOGU_ASM_BITS=8|4                    (defaults to 8 — the ship default)
//!   BOOGU_ASM_TURBO=1                     (use the Turbo sampler for the verify render)
//! e.g. assemble Base→Q8 + verify:
//!   BOOGU_ASM_SRC=<base> BOOGU_ASM_OUT=~/boogu-image-mlx/base CARGO_TARGET_DIR=~/Repos/mlx-gen/target \
//!     cargo test -p mlx-gen-boogu --test turnkey assemble_turnkey_loads -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen_boogu::convert::assemble_quantized_snapshot;
use mlx_gen_boogu::{BooguPipeline, GenerateOptions, TurboOptions};

fn env_or(key: &str, fallback: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| {
        std::env::var(fallback).unwrap_or_else(|_| panic!("set {key} or {fallback}"))
    })
}

/// Total bytes of a directory tree (for the manifest `estimatedSizeBytes`).
fn dir_size(dir: &std::path::Path) -> u64 {
    let mut total = 0;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                total += dir_size(&p);
            } else if let Ok(m) = p.metadata() {
                total += m.len();
            }
        }
    }
    total
}

#[test]
#[ignore = "needs real weights (128 GB Mac): set BOOGU_ASM_SRC/BOOGU_BASE_DIR"]
fn assemble_turnkey_loads() {
    let src = PathBuf::from(env_or("BOOGU_ASM_SRC", "BOOGU_BASE_DIR"));
    let dst = std::env::var("BOOGU_ASM_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("boogu_turnkey"));
    let bits: i32 = std::env::var("BOOGU_ASM_BITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let turbo = std::env::var("BOOGU_ASM_TURBO").is_ok();

    println!(
        "[E9] assembling Q{bits} turnkey  {}  →  {}",
        src.display(),
        dst.display()
    );
    let _ = std::fs::remove_dir_all(&dst); // start clean so no stale source shards linger
    assemble_quantized_snapshot(&src, &dst, bits).expect("assemble turnkey");
    println!("[E9] turnkey size = {:.2} GB", dir_size(&dst) as f64 / 1e9);

    // Load through the SHIPPED path + render a small image to prove it runs end-to-end.
    let pipe = BooguPipeline::from_snapshot(&dst).expect("from_snapshot on the turnkey");
    let img = if turbo {
        pipe.generate_turbo(
            "a red apple on a wooden table",
            &TurboOptions {
                height: 512,
                width: 512,
                steps: 4,
                seed: 0,
                conditioning_sigma: 0.001,
            },
        )
        .expect("turbo generate")
    } else {
        pipe.generate(
            "a red apple on a wooden table",
            &GenerateOptions {
                height: 512,
                width: 512,
                steps: 8,
                text_guidance_scale: 4.0,
                seed: 0,
            },
        )
        .expect("generate")
    };

    assert_eq!((img.width, img.height), (512, 512));
    let (mn, mx) = img
        .pixels
        .iter()
        .fold((255u8, 0u8), |(mn, mx), &p| (mn.min(p), mx.max(p)));
    println!("[E9] render stats: min={mn} max={mx}");
    assert!(
        mx - mn > 32,
        "turnkey render looks degenerate (min={mn} max={mx})"
    );
    println!(
        "[E9] OK — Q{bits} turnkey at {} loads + renders",
        dst.display()
    );
}
