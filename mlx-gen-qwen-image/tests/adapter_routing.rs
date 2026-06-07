//! sc-2528: Qwen adapter key→module routing for the targets whose trained-file (diffusers) naming
//! differs from the crate's internal fields — joint attention (`to_out.0`, the text-stream
//! `add_{q,k,v}_proj` → `add_{q,k,v}`) and the stream feed-forwards (`net.0.proj`/`net.2`). The
//! full 60-block routing is gated locally against real weights; this locks the translations in CI
//! with synthetic temp fixtures (no real weights).

use std::collections::HashMap;
use std::path::PathBuf;

use mlx_gen::adapters::{install_adapter, Adapter};
use mlx_gen::weights::Weights;
use mlx_gen::{AdapterKind, AdapterSpec};
use mlx_gen_qwen_image::apply_qwen_adapters;
use mlx_gen_qwen_image::transformer::{FeedForward, QwenJointAttention, QwenTransformerBlock};
use mlx_rs::Array;

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("mlx_gen_qwen_routing_test");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn dummy() -> Adapter {
    Adapter::Lora {
        a: Array::from_slice(&[0.0f32], &[1, 1]),
        b: Array::from_slice(&[0.0f32], &[1, 1]),
        scale: 0.0,
    }
}

fn write(path: &PathBuf, arrays: Vec<(&str, &Array)>) {
    Array::save_safetensors(arrays, None as Option<&HashMap<String, String>>, path).unwrap();
}

fn push_linear<'a>(
    tensors: &mut Vec<(&'a str, &'a Array)>,
    prefix: String,
    weight: &'a Array,
    bias: &'a Array,
) {
    tensors.push((
        Box::leak(format!("{prefix}.weight").into_boxed_str()),
        weight,
    ));
    tensors.push((Box::leak(format!("{prefix}.bias").into_boxed_str()), bias));
}

fn tiny_block() -> QwenTransformerBlock {
    let w8 = Array::from_slice(&vec![0.1f32; 64], &[8, 8]);
    let b8 = Array::from_slice(&[0.0f32; 8], &[8]);
    let n4 = Array::from_slice(&[1.0f32; 4], &[4]);
    let w48x8 = Array::from_slice(&vec![0.1f32; 48 * 8], &[48, 8]);
    let b48 = Array::from_slice(&[0.0f32; 48], &[48]);
    let w16x8 = Array::from_slice(&vec![0.1f32; 16 * 8], &[16, 8]);
    let b16 = Array::from_slice(&[0.0f32; 16], &[16]);
    let w8x16 = Array::from_slice(&vec![0.1f32; 8 * 16], &[8, 16]);
    let path = tmp("block_mod.safetensors");
    let mut t: Vec<(&str, &Array)> = Vec::new();

    push_linear(&mut t, "img_mod_linear".to_string(), &w48x8, &b48);
    push_linear(&mut t, "txt_mod_linear".to_string(), &w48x8, &b48);
    for p in [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
        "attn.attn_to_out.0",
        "attn.to_add_out",
    ] {
        push_linear(&mut t, p.to_string(), &w8, &b8);
    }
    for p in [
        "attn.norm_q",
        "attn.norm_k",
        "attn.norm_added_q",
        "attn.norm_added_k",
    ] {
        t.push((Box::leak(format!("{p}.weight").into_boxed_str()), &n4));
    }
    for stream in ["img_ff", "txt_ff"] {
        push_linear(&mut t, format!("{stream}.mlp_in"), &w16x8, &b16);
        push_linear(&mut t, format!("{stream}.mlp_out"), &w8x16, &b8);
    }
    write(&path, t);
    let w = Weights::from_file(&path).unwrap();
    QwenTransformerBlock::from_weights(&w, "", 2, 4).unwrap()
}

#[test]
fn attention_routes_diffusers_names() {
    // inner = num_heads*head_dim = 2*4 = 8; all 8 projections [8,8]+bias[8], norms [4].
    let w8 = Array::from_slice(&vec![0.1f32; 64], &[8, 8]);
    let b8 = Array::from_slice(&[0.0f32; 8], &[8]);
    let n4 = Array::from_slice(&[1.0f32; 4], &[4]);
    let path = tmp("attn.safetensors");
    let mut t: Vec<(&str, &Array)> = Vec::new();
    for p in [
        "to_q",
        "to_k",
        "to_v",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "attn_to_out.0",
        "to_add_out",
    ] {
        t.push((Box::leak(format!("{p}.weight").into_boxed_str()), &w8));
        t.push((Box::leak(format!("{p}.bias").into_boxed_str()), &b8));
    }
    for p in ["norm_q", "norm_k", "norm_added_q", "norm_added_k"] {
        t.push((Box::leak(format!("{p}.weight").into_boxed_str()), &n4));
    }
    write(&path, t);
    let w = Weights::from_file(&path).unwrap();
    let mut attn = QwenJointAttention::from_weights(&w, "", 2, 4).unwrap();

    // Trained-file (diffusers) naming resolves.
    for p in [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "to_add_out",
    ] {
        assert!(
            install_adapter(&mut attn, p, dummy()).is_ok(),
            "{p} should resolve"
        );
    }
    // Off-surface / internal names must not.
    for p in ["to_out", "add_q", "to_q.0", "to_add_out.0"] {
        assert!(
            install_adapter(&mut attn, p, dummy()).is_err(),
            "{p} must not resolve"
        );
    }
}

#[test]
fn feed_forward_routes_net_indices() {
    // mlp_in [16,8], mlp_out [8,16] + biases.
    let win = Array::from_slice(&vec![0.1f32; 128], &[16, 8]);
    let bin = Array::from_slice(&[0.0f32; 16], &[16]);
    let wout = Array::from_slice(&vec![0.1f32; 128], &[8, 16]);
    let bout = Array::from_slice(&[0.0f32; 8], &[8]);
    let path = tmp("ff.safetensors");
    write(
        &path,
        vec![
            ("mlp_in.weight", &win),
            ("mlp_in.bias", &bin),
            ("mlp_out.weight", &wout),
            ("mlp_out.bias", &bout),
        ],
    );
    let w = Weights::from_file(&path).unwrap();
    let mut ff = FeedForward::from_weights(&w, "").unwrap();

    // diffusers file naming: `net.0.proj` (in) / `net.2` (out).
    assert!(install_adapter(&mut ff, "net.0.proj", dummy()).is_ok());
    assert!(install_adapter(&mut ff, "net.2", dummy()).is_ok());
    // Internal field names + other indices must not resolve.
    assert!(install_adapter(&mut ff, "mlp_in", dummy()).is_err());
    assert!(install_adapter(&mut ff, "net.1", dummy()).is_err());
    assert!(install_adapter(&mut ff, "net.0", dummy()).is_err());
}

#[test]
fn block_routes_diffusers_modulation_linears() {
    // Minimal block: inner = num_heads*head_dim = 2*4 = 8. The modulation Linear is [6*8,8].
    let mut block = tiny_block();

    assert!(install_adapter(&mut block, "img_mod.1", dummy()).is_ok());
    assert!(install_adapter(&mut block, "txt_mod.1", dummy()).is_ok());
    assert!(install_adapter(&mut block, "img_mod", dummy()).is_err());
    assert!(install_adapter(&mut block, "txt_mod_linear", dummy()).is_err());
}

#[test]
fn strict_loader_applies_peft_modulation_loras() {
    let mut block = tiny_block();
    let r = 2i32;
    let down = Array::from_slice(&vec![0.01f32; (r * 8) as usize], &[r, 8]);
    let up = Array::from_slice(&vec![0.01f32; (48 * r) as usize], &[48, r]);
    let path = tmp("mod_lora.safetensors");
    write(
        &path,
        vec![
            ("transformer.img_mod.1.lora_A.weight", &down),
            ("transformer.img_mod.1.lora_B.weight", &up),
            ("transformer.txt_mod.1.lora_A.weight", &down),
            ("transformer.txt_mod.1.lora_B.weight", &up),
        ],
    );

    let report = apply_qwen_adapters(
        &mut block,
        &[AdapterSpec::new(path, 1.0, AdapterKind::Lora)],
    )
    .unwrap();
    assert_eq!(report.applied, 2);
    assert!(report.unmatched_paths.is_empty());
}
