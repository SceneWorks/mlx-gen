//! HD spatial-tiling fidelity gate (sc-5201). Spatial tiling runs the full encode→DiT→decode path on
//! overlapping tiles and feather-blends them — it is **not** bit-exact to an untiled pass (the causal
//! VAE sees different padding at tile borders), but with overlap the blended result tracks the untiled
//! one closely. We force a small tile on a small frame (where both fit) and assert the tiled decode
//! matches the untiled `run_model_5d` decode. Weight-gated; skips without the raw checkpoint.

use mlx_gen::Image;
use mlx_gen_seedvr2::config::DitConfig;
use mlx_gen_seedvr2::pipeline::Seedvr2Pipeline;
use mlx_rs::{Array, Dtype};

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
fn seedvr2_spatial_tiling_tracks_untiled() {
    let Some(snap) = raw_dir() else {
        eprintln!("SKIP: raw checkpoint absent");
        return;
    };
    let pipe = Seedvr2Pipeline::load(
        &snap,
        "seedvr2_ema_3b_fp16.safetensors",
        &DitConfig::seedvr2_3b(),
        Dtype::Float32,
    )
    .expect("load 3B");
    let neg = pipe.neg_embed().expect("neg-embed").clone();

    // A 256×256 target; a SMOOTH LR gradient (realistic low-frequency content — a harsh high-frequency
    // pattern amplifies the per-tile VAE border-padding difference unrealistically).
    let (h, w) = (256, 256);
    let (lw, lh) = (96usize, 96usize);
    let mut pixels = Vec::with_capacity(lw * lh * 3);
    for y in 0..lh {
        for x in 0..lw {
            let g = ((x + y) * 255 / (lw + lh)) as u8;
            pixels.push(g);
            pixels.push(255 - g);
            pixels.push(((x * 255) / lw) as u8);
        }
    }
    let lr = Image {
        width: lw as u32,
        height: lh as u32,
        pixels,
    };
    // Compare the **decoded** tensors through the same code path: a single tile spanning the whole
    // frame (tile = full size, no overlap → no tiling/blend) vs a forced 160-px / 64-overlap 2×2 grid.
    let processed = pipe.preprocess_frame(&lr, w, h, 0.0).expect("preprocess");
    let untiled = pipe
        .run_frame_tiled(
            &processed,
            7,
            /*tile=*/ w.max(h),
            /*overlap=*/ 0,
            &neg,
        )
        .expect("single-tile (untiled) path");
    let tiled = pipe
        .run_frame_tiled(&processed, 7, 160, 64, &neg)
        .expect("tiled path");
    assert_eq!(tiled.shape(), untiled.shape(), "decoded shape");
    let cos = cosine(&tiled, &untiled);
    eprintln!(
        "spatial tiling vs untiled: cosine={cos:.6} (shape {:?})",
        tiled.shape()
    );
    // Not bit-exact (VAE border padding differs per tile); overlap-feather keeps it close.
    assert!(cos > 0.95, "tiled decode diverged from untiled: {cos}");
}

/// Cosine similarity over two equal-length RGB8 pixel buffers.
fn pixel_cosine(a: &[u8], b: &[u8]) -> f32 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    (dot / (na.sqrt() * nb.sqrt()).max(1e-12)) as f32
}

/// sc-6067: the **image** path (`Seedvr2Pipeline::generate`) must spatially tile when a single
/// full-resolution pass would exceed the memory budget — previously it ran the whole frame in one
/// pass with no budget check, so a large upscale tried a single allocation past Metal's max buffer
/// size and panicked the worker. We drive the budget-injectable `generate_budgeted`: a huge ceiling
/// → the one-pass still path; a tiny ceiling → forced over-budget → the feather-blended tiled path.
/// Both must return the requested dims, and the tiled result must track the one-pass result (the
/// tiler is the same parity-gated `run_frame_tiled` exercised above). Weight-gated; skips without the
/// checkpoint.
#[test]
fn seedvr2_image_path_tiles_when_over_budget() {
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
    .expect("load 3B");

    // 512² target from a smooth LR gradient — large enough that the forced-tiny budget tiles it into
    // an overlapping grid (tile floors at 256 px → a 3×3 grid over 512²).
    let (w, h) = (512, 512);
    let (lw, lh) = (160usize, 160usize);
    let mut pixels = Vec::with_capacity(lw * lh * 3);
    for y in 0..lh {
        for x in 0..lw {
            let g = ((x + y) * 255 / (lw + lh)) as u8;
            pixels.push(g);
            pixels.push(255 - g);
            pixels.push(((x * 255) / lw) as u8);
        }
    }
    let lr = Image {
        width: lw as u32,
        height: lh as u32,
        pixels,
    };

    // Huge ceiling → one-pass still path; tiny ceiling → OverBudget → spatial-tiled path.
    let single = pipe
        .generate_budgeted(&lr, w, h, 7, 0.0, 1.0e9)
        .expect("single-pass still path");
    let tiled = pipe
        .generate_budgeted(&lr, w, h, 7, 0.0, 1.0e-6)
        .expect("spatial-tiled path");

    assert_eq!(
        (single.width, single.height),
        (w as u32, h as u32),
        "single-pass output dims"
    );
    assert_eq!(
        (tiled.width, tiled.height),
        (w as u32, h as u32),
        "tiled output dims"
    );
    assert_eq!(
        tiled.pixels.len(),
        single.pixels.len(),
        "buffer sizes match"
    );

    // The tiled result tracks the one-pass result closely (feather-blend; not bit-exact — the causal
    // VAE sees different border padding per tile).
    let cos = pixel_cosine(&tiled.pixels, &single.pixels);
    eprintln!("image-path tiled vs single-pass: cosine={cos:.6}");
    assert!(cos > 0.95, "tiled image diverged from single-pass: {cos}");
}
