//! sc-3836: the distilled-guidance Approximator + timestep/text-proj embeddings match the torch
//! `diffusers` reference. Golden = `tools/dump_chroma_golden.py` (tiny synthetic config, f32).
//!
//! `pooled_temb` is the end-to-end modulation tensor (`time_text_embed → distilled_guidance_layer`),
//! so matching it validates both the sinusoid `input_vec` build and the 5-layer SiLU residual MLP.

use mlx_gen::weights::Weights;
use mlx_gen_chroma::ChromaTransformer;
use mlx_gen_chroma::ChromaTransformerConfig;
use mlx_rs::ops::{abs, max, subtract};

/// The tiny config baked into `dump_chroma_golden.py`.
fn tiny_cfg() -> ChromaTransformerConfig {
    ChromaTransformerConfig {
        in_channels: 4,
        num_layers: 1,
        num_single_layers: 1,
        num_attention_heads: 2,
        attention_head_dim: 8,
        joint_attention_dim: 12,
        axes_dims_rope: [2, 2, 4],
        approximator_num_channels: 8,
        approximator_hidden_dim: 16,
        approximator_layers: 2,
    }
}

fn max_abs(a: &mlx_rs::Array) -> f32 {
    max(abs(a).unwrap(), None).unwrap().item::<f32>()
}

fn max_abs_diff(a: &mlx_rs::Array, b: &mlx_rs::Array) -> f32 {
    max_abs(&subtract(a, b).unwrap())
}

#[test]
fn pooled_temb_matches_diffusers() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let weights = Weights::from_file(format!("{dir}/chroma_tiny_weights.safetensors")).unwrap();
    let io = Weights::from_file(format!("{dir}/chroma_tiny_io.safetensors")).unwrap();

    let t = ChromaTransformer::from_weights(weights, tiny_cfg()).unwrap();

    let timestep = io.require("timestep").unwrap(); // [1], raw (unscaled)
    let golden = io.require("pooled_temb").unwrap(); // [1, 17, 16]

    // input_vec is pure sin/cos (no matmul) — gate it tightly to isolate the embedding build
    // (flip order, mod_proj index, *1000 scaling) from the cross-backend matmul floor.
    let iv_golden = io.require("input_vec").unwrap();
    let iv_got = t.input_vec_for_tests(timestep).unwrap();
    assert_eq!(iv_got.shape(), iv_golden.shape(), "input_vec shape");
    let iv_diff = max_abs_diff(&iv_got, iv_golden);
    assert!(
        iv_diff < 1e-5,
        "input_vec max|Δ| = {iv_diff} — embedding build diverges from diffusers"
    );

    let got = t.pooled_temb(timestep).unwrap();
    assert_eq!(got.shape(), golden.shape(), "pooled_temb shape");

    // Cross-backend gate: the golden is torch-CPU f32; the port runs mlx-Metal f32. Their matmuls
    // are NOT bit-identical — this codebase documents a ~2.4e-3 mlx-Metal-f32-matmul floor for
    // structural parity. A real structural bug (wrong activation/order/flip/eps) would be O(0.1+),
    // not sub-1e-3. We gate on peak-relative error against that floor.
    let d = max_abs_diff(&got, golden);
    let scale = max_abs(golden);
    let rel = d / scale;
    assert!(
        rel < 2e-3,
        "pooled_temb peak-rel = {rel} (max|Δ|={d}, scale={scale}) exceeds the mlx-Metal f32 floor"
    );
}
