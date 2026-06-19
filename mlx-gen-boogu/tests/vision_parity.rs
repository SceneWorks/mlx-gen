//! E7b-1 (sc-6478) — real-weight parity for the Boogu Qwen3-VL **vision tower** + preprocessing.
//!
//! `vision_tower_matches_reference` feeds the reference processor's `pixel_values` + `grid_thw` into
//! the native tower ([`load_vision_tower`], which runs **f32** — see its doc) and checks the merged
//! `image_embeds` + the 3 deepstack features match the transformers `Qwen3VLVisionModel` (cosine vs
//! the f32 reference). `preprocess_matches_reference` checks the native image preprocessing reproduces
//! the reference `pixel_values` (exact for a multiple-of-32 image — no resampling). Goldens come from
//! `reference/dump_boogu_vision_golden.py`.
//!
//! `#[ignore]` — needs the Base snapshot (`mllm/`) + the golden + the test image:
//!   BOOGU_BASE_DIR=<snapshot> CARGO_TARGET_DIR=~/Repos/mlx-gen/target \
//!     cargo test -p mlx-gen-boogu --test vision_parity -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_boogu::load_vision_tower;
use mlx_gen_boogu::vision::preprocess::preprocess_image;
use mlx_rs::ops::{multiply, sqrt, sum};
use mlx_rs::{Array, Dtype};

fn cosine(a: &Array, b: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    let dot = sum(multiply(&a, &b).unwrap(), false).unwrap();
    let na = sqrt(sum(multiply(&a, &a).unwrap(), false).unwrap()).unwrap();
    let nb = sqrt(sum(multiply(&b, &b).unwrap(), false).unwrap()).unwrap();
    (dot / (na * nb)).item::<f32>()
}

fn snapshot_dir() -> PathBuf {
    PathBuf::from(std::env::var("BOOGU_BASE_DIR").expect("set BOOGU_BASE_DIR to the snapshot root"))
}

fn golden_path() -> PathBuf {
    std::env::var("BOOGU_VISION_GOLDEN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME"))
                .join("Repos/mlx-gen-wt-boogu/reference/goldens/boogu_vision.safetensors")
        })
}

fn grid_from_golden(g: &Weights) -> Vec<[i32; 3]> {
    let t = g
        .require("image_grid_thw")
        .unwrap()
        .as_dtype(Dtype::Int32)
        .unwrap();
    let v = t.as_slice::<i32>().to_vec();
    v.chunks(3).map(|c| [c[0], c[1], c[2]]).collect()
}

/// The native f32 vision tower reproduces the reference `Qwen3VLVisionModel` (merged image-embeds +
/// the 3 deepstack features + the pre-merger hidden) to cosine > 0.999.
#[test]
#[ignore = "needs real weights + golden (dump_boogu_vision_golden.py)"]
fn vision_tower_matches_reference() {
    let g = Weights::from_file(golden_path()).expect("golden — run dump_boogu_vision_golden.py");
    let tower = load_vision_tower(snapshot_dir()).expect("load Qwen3-VL vision tower (f32)");

    let grid = grid_from_golden(&g);
    let pv = g
        .require("pixel_values")
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let (embeds, deepstack, prenorm) = tower.forward_debug(&pv, &grid).expect("vision forward");

    // Compute all cosines first (so a failure reports where the drift starts), then assert.
    let c_pre = cosine(&prenorm, g.require("vision_prenorm_f32").unwrap());
    let c_emb = cosine(&embeds, g.require("image_embeds_f32").unwrap());
    let c_ds: Vec<f32> = (0..deepstack.len())
        .map(|i| {
            cosine(
                &deepstack[i],
                g.require(&format!("deepstack_f32_{i}")).unwrap(),
            )
        })
        .collect();
    println!("vision prenorm parity cosine      = {c_pre:.7}");
    println!("vision image_embeds parity cosine = {c_emb:.7}");
    for (i, c) in c_ds.iter().enumerate() {
        println!("vision deepstack[{i}] parity cosine  = {c:.7}");
    }

    assert_eq!(
        embeds.shape(),
        g.require("image_embeds_f32").unwrap().shape(),
        "merged image-embeds shape"
    );
    assert!(c_emb > 0.999, "image_embeds parity cosine {c_emb} too low");
    assert_eq!(deepstack.len(), 3, "3 deepstack features");
    for (i, c) in c_ds.iter().enumerate() {
        assert!(*c > 0.999, "deepstack[{i}] parity cosine {c} too low");
    }
}

#[test]
#[ignore = "needs the golden + test image (dump_boogu_vision_golden.py)"]
fn preprocess_matches_reference() {
    let g = Weights::from_file(golden_path()).expect("golden — run dump_boogu_vision_golden.py");
    let img_path = PathBuf::from(std::env::var("HOME").expect("HOME"))
        .join("Repos/mlx-gen-wt-boogu/reference/outputs/boogu_mlx_t2i_apple_512_s28.png");
    let img = image::open(&img_path)
        .unwrap_or_else(|e| panic!("open {}: {e}", img_path.display()))
        .to_rgb8();

    let (pv, grid) = preprocess_image(&img).expect("preprocess");
    let want = g.require("pixel_values").unwrap();
    assert_eq!(grid, [1, 32, 32], "grid_thw for a 512² image");
    assert_eq!(pv.shape(), want.shape(), "pixel_values shape");

    // 512 is a multiple of patch·merge (32) → smart_resize is identity, so this should be ~exact.
    let c = cosine(&pv, want);
    println!("preprocess pixel_values parity cosine = {c:.7}");
    assert!(c > 0.999, "preprocess parity cosine {c} too low");
}
