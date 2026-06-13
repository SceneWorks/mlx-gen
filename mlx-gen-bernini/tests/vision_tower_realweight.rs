//! sc-5134: real-weight load + forward smoke for the Qwen2.5-VL vision tower (`#[ignore]`).
//!
//! Loads the 390 `visual.*` tensors from the converted `qwen2_5_vl.safetensors` — proving the sc-5144
//! converter keys match the module — and runs a forward over a synthetic single-image `grid_thw` at
//! the real dims (hidden 1280, 16 heads, 32 blocks, out_hidden 3584), asserting a finite, correctly
//! shaped token tensor. Numeric correctness is covered by the f32 synthetic golden.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_bernini::vision::{VisionConfig, VisionTower};
use mlx_rs::{random, Array, Dtype};

fn snapshot() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/mlx-gen-models/bernini_planner_mlx_bf16")
}

fn finite_max(a: &Array) -> f32 {
    a.abs().unwrap().max(None).unwrap().item::<f32>()
}

#[test]
#[ignore = "real weights: loads the 32-block Qwen2.5-VL vision tower and runs a forward"]
fn vision_tower_real_weight_smoke() {
    let snap = snapshot();
    let cfg = VisionConfig::from_config_json(&snap.join("qwen2_5_vl_config.json"))
        .expect("vision config");
    assert_eq!(cfg.hidden_size, 1280);
    assert_eq!(cfg.depth, 32);
    assert_eq!(cfg.out_hidden_size, 3584);

    let w = Weights::from_file(snap.join("qwen2_5_vl.safetensors")).expect("qwen2_5_vl weights");
    let visual_keys = w.keys().filter(|k| k.starts_with("visual.")).count();
    assert_eq!(visual_keys, 390, "expected 390 visual.* tensors");

    let tower = VisionTower::from_weights(&w, cfg.clone(), "visual").expect("vision tower");

    // One 8×8-patch frame → 4×4 merged tokens = 16 tokens; in_dim = in·t·ph·pw = 3·2·14·14 = 1176.
    let grid = [[1i32, 8, 8]];
    let seq = grid[0][0] * grid[0][1] * grid[0][2];
    let in_dim = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
    let key = random::key(0).unwrap();
    let pixels = random::normal::<f32>(&[seq, in_dim], None, None, Some(&key))
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();

    let tokens = tower.forward(&pixels, &grid).expect("forward");
    let merged = (8 / cfg.spatial_merge_size) * (8 / cfg.spatial_merge_size);
    assert_eq!(tokens.shape(), &[merged, 3584]);
    let m = finite_max(&tokens);
    assert!(
        m.is_finite() && m > 0.0,
        "vit tokens finite & non-trivial (max {m})"
    );
    println!(
        "vision tower real-weight ok: tokens {:?} max|·|={m:.4}",
        tokens.shape()
    );
}
