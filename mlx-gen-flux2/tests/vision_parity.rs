//! sc-5918: parity for the FLUX.2-dev **Pixtral vision tower + Mistral3 projector** vs the PyTorch
//! reference (`transformers` `PixtralVisionModel` + `Mistral3MultiModalProjector`), on a TINY
//! synthetic config (committed fixture `tests/fixtures/pixtral_vision_golden.safetensors` ←
//! `tools/dump_flux2_dev_pixtral_vision_golden.py`).
//!
//! Exercises every path the port adds: bias-less split q/k/v/o + RMSNorm + SwiGLU under
//! block-diagonal attention, the 2-D Pixtral RoPE (θ=10000, `rotate_half`), and the projector's
//! `norm → 2×2 patch-merge (unfold) → linear_1 → gelu → linear_2`. A real structural bug (wrong
//! RoPE/theta, wrong patch order, wrong unfold layout, missing/extra norm) diverges by orders of
//! magnitude; f32 Metal matmul agrees to ~1e-2.

use mlx_gen::weights::Weights;
use mlx_gen_flux2::{Mistral3Projector, PixtralVisionConfig, PixtralVisionTower};
use mlx_rs::ops::all_close;
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/pixtral_vision_golden.safetensors"
);

fn close(a: &Array, b: &Array, rtol: f64, atol: f64) -> bool {
    all_close(a, b, rtol, atol, false).unwrap().item::<bool>()
}

/// Cosine similarity over the flattened tensors — a scale-free structural check alongside `all_close`.
fn cosine(a: &Array, b: &Array) -> f32 {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let dot: f32 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|&y| y * y).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

/// The tiny Pixtral vision config the dump script used (head_dim = 32/4 = 8, patch 2, θ=10000).
fn tiny_vision_config() -> PixtralVisionConfig {
    PixtralVisionConfig {
        hidden_size: 32,
        num_layers: 2,
        num_heads: 4,
        head_dim: 8,
        intermediate_size: 64,
        patch_size: 2,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
        num_channels: 3,
    }
}

#[test]
fn pixtral_vision_tower_and_projector_match_reference() {
    let w = Weights::from_file(FIXTURE).unwrap();

    let grid = w.require("grid_hw").unwrap().as_slice::<i32>().to_vec();
    let grids = vec![(grid[0], grid[1])]; // (gh, gw) = (4, 6)
    let pixel_values = w.require("pixel_values").unwrap().clone(); // NHWC [1, 8, 12, 3]

    // ---- vision tower ------------------------------------------------------------------------
    let tower = PixtralVisionTower::from_weights(&w, "vision_tower", tiny_vision_config()).unwrap();
    let feats = tower.forward(&[&pixel_values], &grids).unwrap();
    let want_feats = w.require("image_features").unwrap();
    println!(
        "image_features: shape={:?} cosine={:.6}",
        feats.shape(),
        cosine(&feats, want_feats)
    );
    assert_eq!(feats.shape(), want_feats.shape(), "image_features shape");
    assert!(
        close(&feats, want_feats, 1e-2, 1e-2),
        "Pixtral vision features diverged from the PyTorch reference"
    );

    // ---- multimodal projector ----------------------------------------------------------------
    let projector = Mistral3Projector::from_weights(&w, "multi_modal_projector", 2, 1e-5).unwrap();
    let got = projector.forward(&feats, &grids).unwrap();
    let want_proj = w.require("projected").unwrap();
    println!(
        "projected: shape={:?} cosine={:.6}",
        got.shape(),
        cosine(&got, want_proj)
    );
    assert_eq!(got.shape(), want_proj.shape(), "projected shape");
    assert!(
        close(&got, want_proj, 1e-2, 1e-2),
        "Mistral3 projector output diverged from the PyTorch reference"
    );
}
