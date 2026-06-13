//! sc-5134: the Qwen2.5-VL vision tower matches the reference (near-bit, f32).
//!
//! Synthetic-fixture parity (the repo's weight-free golden pattern): a tiny
//! `Qwen2_5_VisionTransformerPretrainedModel` (4 blocks, fullatt at [1,3]) with random f32 weights,
//! dumped from the reference by `tools/dump_bernini_vision_tower_golden.py`. The grid
//! `[[1,6,6],[1,4,4]]` exercises window padding, multiple windows per image, **and** the
//! block-diagonal full-attention mask across two images. Tolerance reflects the MLX-Metal-vs-torch
//! f32 floor accumulated over the patch-embed matmul + 4 attention blocks (RoPE + f32 softmax) + the
//! merger.
//!
//! Run: `cargo test -p mlx-gen-bernini --test vision_tower_parity -- --nocapture`

use mlx_gen::weights::Weights;
use mlx_gen_bernini::vision::{VisionConfig, VisionTower};
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/vision_tower_golden.safetensors"
);

fn errors(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    (max_diff, max_diff / peak)
}

fn meta_i(w: &Weights, k: &str) -> i32 {
    w.metadata(k).unwrap().parse().unwrap()
}

#[test]
fn vision_tower_matches_reference_f32() {
    let w = Weights::from_file(FIXTURE).expect("load fixture");

    let fullatt: Vec<i32> = w
        .metadata("fullatt")
        .unwrap()
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let cfg = VisionConfig {
        hidden_size: meta_i(&w, "hidden"),
        num_heads: meta_i(&w, "heads"),
        intermediate_size: meta_i(&w, "intermediate"),
        depth: meta_i(&w, "depth"),
        fullatt_block_indexes: fullatt,
        spatial_merge_size: meta_i(&w, "spatial_merge"),
        window_size: meta_i(&w, "window"),
        patch_size: meta_i(&w, "patch"),
        temporal_patch_size: meta_i(&w, "temporal_patch"),
        in_channels: meta_i(&w, "in_chans"),
        out_hidden_size: meta_i(&w, "out_hidden"),
    };

    let tower = VisionTower::from_weights(&w, cfg, "visual").expect("vision tower");

    let pixel_values = w.require("io.pixel_values").unwrap().clone();
    let grid_arr = w.require("io.grid_thw").unwrap().clone();
    let rows = grid_arr.shape()[0];
    let g = grid_arr.as_slice::<i32>();
    let grid: Vec<[i32; 3]> = (0..rows as usize)
        .map(|i| [g[i * 3], g[i * 3 + 1], g[i * 3 + 2]])
        .collect();

    let got = tower.forward(&pixel_values, &grid).expect("forward");
    let want = w.require("out.tokens").unwrap();
    assert_eq!(got.shape(), want.shape(), "token shape");

    let (abs, rel) = errors(&got, want);
    println!(
        "vision tower: peak|Δ|={abs:.3e}  peak-rel={rel:.3e}  shape={:?}",
        got.shape()
    );
    // ~1e-3 f32 cross-backend floor (patch matmul + 4 RoPE/softmax blocks + merger); a wrong window
    // permutation / mask / merge is O(0.1+).
    assert!(rel < 5e-3, "vision tower peak-rel {rel:.3e} exceeds 5e-3");
}
