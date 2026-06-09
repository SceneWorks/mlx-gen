//! sc-3837: the Chroma DiT forward matches the torch `diffusers` `ChromaTransformer2DModel`.
//! Golden = `tools/dump_chroma_golden.py` (tiny synthetic config, f32, with a real attention-mask 0
//! to exercise MMDiT masking). A correct forward matching end-to-end validates the double/single
//! blocks, pruned-adaLN slice offsets, RoPE, masking, and the pruned `norm_out`.

use mlx_gen::weights::Weights;
use mlx_gen_chroma::{ChromaTransformer, ChromaTransformerConfig};
use mlx_rs::ops::{abs, max, subtract};

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

#[test]
fn forward_matches_diffusers() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let weights = Weights::from_file(format!("{dir}/chroma_tiny_weights.safetensors")).unwrap();
    let io = Weights::from_file(format!("{dir}/chroma_tiny_io.safetensors")).unwrap();

    let t = ChromaTransformer::from_weights(weights, tiny_cfg()).unwrap();

    let got = t
        .forward(
            io.require("hidden").unwrap(),
            io.require("encoder").unwrap(),
            io.require("timestep").unwrap(),
            io.require("img_ids").unwrap(),
            io.require("txt_ids").unwrap(),
            Some(io.require("attention_mask").unwrap()),
        )
        .unwrap();

    let golden = io.require("output").unwrap();
    assert_eq!(got.shape(), golden.shape(), "output shape");

    // Cross-backend gate (torch-CPU f32 vs mlx-Metal f32). The 57-block chain accumulates the
    // ~1e-3 matmul floor; a structural bug (wrong slice offset / mask / RoPE / block wiring) would
    // be O(0.1+). Peak-relative against the documented floor.
    let d = max_abs(&subtract(&got, golden).unwrap());
    let scale = max_abs(golden);
    let rel = d / scale;
    assert!(
        rel < 5e-3,
        "transformer output peak-rel = {rel} (max|Δ|={d}, scale={scale}) exceeds the mlx-Metal f32 floor"
    );
}
