//! sc-3842: Chroma adapter key→module routing (diffusers/peft naming) + scale-0 ≡ base. Synthetic
//! (the tiny golden transformer), so it runs in CI without real weights.

use mlx_gen::adapters::{install_adapter, AdaptableHost, Adapter};
use mlx_gen::weights::Weights;
use mlx_gen_chroma::{ChromaTransformer, ChromaTransformerConfig};
use mlx_rs::ops::{abs, max, subtract};
use mlx_rs::Array;

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

fn tiny() -> ChromaTransformer {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let w = Weights::from_file(format!("{dir}/chroma_tiny_weights.safetensors")).unwrap();
    ChromaTransformer::from_weights(w, tiny_cfg()).unwrap()
}

fn dummy() -> Adapter {
    Adapter::Lora {
        a: Array::from_slice(&[0.0f32], &[1, 1]),
        b: Array::from_slice(&[0.0f32], &[1, 1]),
        scale: 0.0,
    }
}

#[test]
fn diffusers_paths_resolve() {
    let mut t = tiny();
    for p in [
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_k",
        "transformer_blocks.0.attn.to_v",
        "transformer_blocks.0.attn.add_q_proj",
        "transformer_blocks.0.attn.to_add_out",
        "transformer_blocks.0.attn.to_out.0",
        "transformer_blocks.0.ff.net.0.proj",
        "transformer_blocks.0.ff.net.2",
        "transformer_blocks.0.ff_context.net.0.proj",
        "single_transformer_blocks.0.attn.to_q",
        "single_transformer_blocks.0.proj_mlp",
        "single_transformer_blocks.0.proj_out",
        "x_embedder",
        "context_embedder",
        "proj_out",
        "distilled_guidance_layer.in_proj",
        "distilled_guidance_layer.out_proj",
        "distilled_guidance_layer.layers.0.linear_1",
    ] {
        assert!(
            install_adapter(&mut t, p, dummy()).is_ok(),
            "should resolve: {p}"
        );
    }
    for p in [
        "transformer_blocks.0.attn.to_out",          // missing the `.0`
        "transformer_blocks.9.attn.to_q",            // out-of-range block
        "transformer_blocks.0.norm1.linear",         // pruned-adaLN: no such linear
        "single_transformer_blocks.0.attn.to_out.0", // single attn is pre-only
        "nonsense",
    ] {
        assert!(
            install_adapter(&mut t, p, dummy()).is_err(),
            "should NOT resolve: {p}"
        );
    }
}

#[test]
fn adaptable_paths_all_resolve() {
    let mut t = tiny();
    let paths = t.adaptable_paths();
    assert_eq!(paths.len(), 12 + 5, "tiny: 1 double (12) + 1 single (5)");
    for p in paths {
        let segs: Vec<&str> = p.split('.').collect();
        assert!(
            t.adaptable_mut(&segs).is_some(),
            "adaptable_path must resolve: {p}"
        );
    }
}

#[test]
fn scale_zero_is_base_and_nonzero_differs() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let io = Weights::from_file(format!("{dir}/chroma_tiny_io.safetensors")).unwrap();
    let fwd = |t: &ChromaTransformer| -> Array {
        t.forward(
            io.require("hidden").unwrap(),
            io.require("encoder").unwrap(),
            io.require("timestep").unwrap(),
            io.require("img_ids").unwrap(),
            io.require("txt_ids").unwrap(),
            Some(io.require("attention_mask").unwrap()),
        )
        .unwrap()
    };
    let base = fwd(&tiny());

    // residual = (x · a) · b, so a is [in=16, r], b is [r, out=16], rank 4.
    let a = Array::from_slice(&vec![0.05f32; 16 * 4], &[16, 4]);
    let b = Array::from_slice(&vec![0.05f32; 4 * 16], &[4, 16]);

    let mut t0 = tiny();
    install_adapter(
        &mut t0,
        "transformer_blocks.0.attn.to_q",
        Adapter::Lora {
            a: a.clone(),
            b: b.clone(),
            scale: 0.0,
        },
    )
    .unwrap();
    let d0 = max(abs(subtract(fwd(&t0), &base).unwrap()).unwrap(), None)
        .unwrap()
        .item::<f32>();
    assert_eq!(d0, 0.0, "scale=0 LoRA must be a bit-exact no-op");

    let mut t1 = tiny();
    install_adapter(
        &mut t1,
        "transformer_blocks.0.attn.to_q",
        Adapter::Lora { a, b, scale: 1.0 },
    )
    .unwrap();
    let d1 = max(abs(subtract(fwd(&t1), &base).unwrap()).unwrap(), None)
        .unwrap()
        .item::<f32>();
    assert!(d1 > 1e-4, "scale=1 LoRA must change the output (got {d1})");
}
