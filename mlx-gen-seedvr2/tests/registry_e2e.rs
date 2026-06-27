//! Registry + real-weight load/run gate (sc-4813). Loads the SeedVR2 3B pipeline from the **raw**
//! `numz/SeedVR2_comfyUI` checkpoint dir (exercising the native converter + load path on real
//! weights), then: (a) the bundled neg-embed matches the reference; (b) the full model path matches
//! the golden `decoded`; (c) `generate` runs end-to-end on an image and returns the right size.
//! Needs the HF cache + e2e golden; skips otherwise.

use mlx_gen::weights::Weights;
use mlx_gen::Image;
use mlx_gen_seedvr2::config::DitConfig;
use mlx_gen_seedvr2::pipeline::Seedvr2Pipeline;
use mlx_rs::{Array, Dtype};

fn golden_dir() -> std::path::PathBuf {
    std::env::var("SEEDVR2_GOLDEN_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::Path::new(&std::env::var("HOME").unwrap())
                .join(".cache/mlx-gen-seedvr2-golden")
        })
}

fn raw_dir() -> Option<std::path::PathBuf> {
    let base = std::path::Path::new(&std::env::var("HOME").unwrap())
        .join(".cache/huggingface/hub/models--numz--SeedVR2_comfyUI/snapshots");
    let snap = std::fs::read_dir(&base).ok()?.flatten().next()?.path();
    snap.join("seedvr2_ema_3b_fp16.safetensors")
        .exists()
        .then_some(snap)
}

fn cosine(got: &Array, exp: &Array) -> f32 {
    let g = got
        .as_dtype(Dtype::Float32)
        .unwrap()
        .reshape(&[-1])
        .unwrap();
    let e = exp
        .as_dtype(Dtype::Float32)
        .unwrap()
        .reshape(&[-1])
        .unwrap();
    let (gs, es) = (g.as_slice::<f32>(), e.as_slice::<f32>());
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (a, b) in gs.iter().zip(es.iter()) {
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64).powi(2);
        nb += (*b as f64).powi(2);
    }
    (dot / (na.sqrt() * nb.sqrt()).max(1e-12)) as f32
}

#[test]
fn seedvr2_loads_and_runs_from_real_checkpoint() {
    let (Some(raw), gdir) = (raw_dir(), golden_dir()) else {
        eprintln!("SKIP: raw checkpoint absent");
        return;
    };
    if !gdir.join("e2e_io_f32.safetensors").exists() {
        eprintln!("SKIP: e2e golden absent");
        return;
    }
    // load f32 from the raw checkpoint (native convert + load), then check the model path.
    let pipe = Seedvr2Pipeline::load(
        &raw,
        "seedvr2_ema_3b_fp16.safetensors",
        &DitConfig::seedvr2_3b(),
        Dtype::Float32,
    )
    .expect("load from raw checkpoint");
    let io = Weights::from_file(gdir.join("e2e_io_f32.safetensors")).expect("e2e io");

    // (a) bundled neg-embed matches the reference
    let neg_cos = cosine(
        pipe.neg_embed().expect("neg-embed"),
        io.require("neg_embed").unwrap(),
    );
    eprintln!("neg_embed cosine = {neg_cos:.6}");
    assert!(neg_cos > 0.9999, "bundled neg-embed mismatch: {neg_cos}");

    // (b) full model path on real-converted weights vs golden decoded
    let decoded = pipe
        .run_model(
            io.require("processed").unwrap(),
            io.require("noise").unwrap(),
            io.require("neg_embed").unwrap(),
            io.require("timestep").unwrap(),
            256,
            256,
        )
        .expect("run_model");
    let cos = cosine(&decoded, io.require("decoded").unwrap());
    eprintln!("real-weight decoded cosine = {cos:.6}");
    assert!(cos > 0.999, "real-weight model path diverged: {cos}");

    // (c) full generate() smoke: a small synthetic LR image → 256×256 RGB8, no panic
    let lr = Image {
        width: 96,
        height: 96,
        pixels: (0..96 * 96 * 3).map(|i| (i % 256) as u8).collect(),
    };
    let out = pipe
        .generate(&lr, 256, 256, 42, 0.0, &mlx_gen::CancelFlag::new())
        .expect("generate");
    assert_eq!((out.width, out.height), (256, 256));
    assert_eq!(out.pixels.len(), 256 * 256 * 3);
    eprintln!(
        "generate ok: {}x{} ({} px)",
        out.width,
        out.height,
        out.pixels.len()
    );
}

/// 7B (sc-5197) real-weight smoke: load the 36-layer pixel-mode-RoPE variant from the raw checkpoint
/// (bf16, ~17 GB) and run `generate` end-to-end. The numeric pixel-RoPE parity is gated separately in
/// `dit_parity.rs::seedvr2_dit_7b_matches_reference`; this proves the full 7B pipeline loads + runs.
#[test]
fn seedvr2_7b_loads_and_generates() {
    let Some(snap) = raw_dir() else {
        eprintln!("SKIP: raw checkpoint absent");
        return;
    };
    if !snap.join("seedvr2_ema_7b_fp16.safetensors").exists() {
        eprintln!("SKIP: 7B checkpoint absent");
        return;
    }
    let pipe = Seedvr2Pipeline::load(
        &snap,
        "seedvr2_ema_7b_fp16.safetensors",
        &DitConfig::seedvr2_7b(),
        Dtype::Bfloat16,
    )
    .expect("load 7B from raw checkpoint");

    let lr = Image {
        width: 96,
        height: 96,
        pixels: (0..96 * 96 * 3).map(|i| (i % 256) as u8).collect(),
    };
    let out = pipe
        .generate(&lr, 128, 128, 7, 0.0, &mlx_gen::CancelFlag::new())
        .expect("7B generate");
    assert_eq!((out.width, out.height), (128, 128));
    assert_eq!(out.pixels.len(), 128 * 128 * 3);
    eprintln!("7B generate ok: {}x{}", out.width, out.height);
}

/// Q8 (sc-5198) real-weight smoke: load 3B, quantize the DiT Linears to Q8, and run `generate`
/// end-to-end (exercises the `quantized_matmul` forward path + the load-time quantize wiring). The
/// numeric near-losslessness is gated separately in `quant_parity.rs`.
#[test]
fn seedvr2_q8_loads_and_generates() {
    let Some(snap) = raw_dir() else {
        eprintln!("SKIP: raw checkpoint absent");
        return;
    };
    let mut pipe = Seedvr2Pipeline::load(
        &snap,
        "seedvr2_ema_3b_fp16.safetensors",
        &DitConfig::seedvr2_3b(),
        Dtype::Bfloat16,
    )
    .expect("load 3B from raw checkpoint");
    pipe.quantize(8).expect("quantize Q8");

    let lr = Image {
        width: 96,
        height: 96,
        pixels: (0..96 * 96 * 3).map(|i| (i % 256) as u8).collect(),
    };
    let out = pipe
        .generate(&lr, 128, 128, 7, 0.0, &mlx_gen::CancelFlag::new())
        .expect("Q8 generate");
    assert_eq!((out.width, out.height), (128, 128));
    assert_eq!(out.pixels.len(), 128 * 128 * 3);
    eprintln!("Q8 generate ok: {}x{}", out.width, out.height);
}

/// A deterministic sharp RGB image (`size²`): a fine high-frequency pattern so a faithful upscale
/// keeps lots of edge energy and the (sc-8228) decoder collapse is unmistakable. No external fixture.
fn sharp_image(size: u32) -> Image {
    let n = (size * size * 3) as usize;
    let mut px = Vec::with_capacity(n);
    for y in 0..size {
        for x in 0..size {
            // high-freq checker XOR a slow gradient → broadband content (not a degenerate checker).
            let checker = if ((x / 2) ^ (y / 2)) & 1 == 0 {
                220u8
            } else {
                35u8
            };
            let grad = ((x + y) * 255 / (2 * size)) as u8;
            px.push(checker);
            px.push(grad);
            px.push(checker ^ grad);
        }
    }
    Image {
        width: size,
        height: size,
        pixels: px,
    }
}

/// Mean + variance of a u8 buffer (used for a non-degenerate sanity floor on the output).
fn mean_var(px: &[u8]) -> (f64, f64) {
    let n = px.len() as f64;
    let mean = px.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = px.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    (mean, var)
}

/// sc-8262 / sc-8228 regression: at a target past the VAE decoder's correctness cap, `generate`
/// (the auto path) MUST route through spatial tiling — never the single full-resolution decode that
/// corrupts the output (sc-8228). We assert the auto render matches an explicitly force-tiled render
/// of the same input/seed (cosine high) and is non-degenerate. Pre-fix `generate` single-passed the
/// full 2048² decode (real-weight pixel cosine vs the tiled/reference render ≈ 0.95, lapvar collapsed);
/// post-fix it tiles (cosine ≥ 0.99). The 0.985 gate sits cleanly between. Real-weight, skips without
/// the checkpoint. (The pure budget/tile-size guard is `video::tests::spatial_tile_never_exceeds_vae_cap`.)
#[test]
fn seedvr2_generate_tiles_above_vae_cap() {
    let Some(snap) = raw_dir() else {
        eprintln!("SKIP: raw checkpoint absent");
        return;
    };
    let pipe = Seedvr2Pipeline::load(
        &snap,
        "seedvr2_ema_3b_fp16.safetensors",
        &DitConfig::seedvr2_3b(),
        Dtype::Bfloat16,
    )
    .expect("load 3B from raw checkpoint");

    let cap = mlx_gen_seedvr2::video::VAE_SAFE_DECODE_EDGE_PX;
    let target = cap + 512; // 2048 — comfortably past the cap so the auto path must tile.
    let lr = sharp_image((target / 2) as u32);
    let cancel = mlx_gen::CancelFlag::new();

    // Auto path (what the worker calls). Post-fix this tiles on the VAE cap.
    let auto = pipe
        .generate(&lr, target, target, 99, 0.0, &cancel)
        .expect("auto generate");
    assert_eq!((auto.width, auto.height), (target as u32, target as u32));

    // Cap-independent reference: a tiny memory budget forces the spatial tiler regardless of the cap.
    let tiled = pipe
        .generate_budgeted(&lr, target, target, 99, 0.0, 8.0, &cancel)
        .expect("forced-tiled generate");

    let auto_f = Array::from_slice(
        &auto.pixels.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        &[auto.pixels.len() as i32],
    );
    let tiled_f = Array::from_slice(
        &tiled.pixels.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        &[tiled.pixels.len() as i32],
    );
    let cos = cosine(&auto_f, &tiled_f);
    eprintln!("auto-vs-forced-tiled cosine @ {target}² = {cos:.4}");

    // Non-degenerate: a collapsed/uniform decode would have near-zero variance.
    let (_m, var) = mean_var(&auto.pixels);
    eprintln!("auto output pixel variance = {var:.1}");
    assert!(
        var > 100.0,
        "auto {target}² output looks degenerate (var {var:.1})"
    );
    assert!(
        cos > 0.985,
        "auto {target}² generate diverged from the tiled reference (cos {cos:.4}) — the VAE \
         correctness cap is not routing the large decode through tiling (sc-8228 regression)"
    );
}
