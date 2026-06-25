//! SD3.5-**Medium** (MMDiT-X) converter + architecture validation (sc-7867 / M1).
//!
//! These exercise the Medium-specific layout — 24 blocks, hidden 1536, `pos_embed_max_size` 384, and
//! the first 13 blocks' `attn2` dual-attention + extended 9-chunk `norm1` AdaLN — against the real
//! `transformer/config.json` + safetensors arch (real-weight confirmed), with the dense numeric path
//! covered against a tiny synthetic fixture (the mlx-gen "test the remap without the real
//! checkpoint" pattern). The real on-disk header validation is `#[ignore]`d behind `SD3_MEDIUM_SNAPSHOT`.

use std::collections::HashMap;

use mlx_gen::weights::Weights;
use mlx_gen_sd3::{
    build_target_state_dict, expected_tensor_count, expected_transformer_tensors,
    quantize_sd3_transformer, validate_arch, Sd3Arch, Sd3Variant, MEDIUM_DUAL_ATTENTION_LAYERS,
    MEDIUM_HIDDEN, MEDIUM_NUM_LAYERS, MEDIUM_POS_EMBED_LEN, MEDIUM_POS_EMBED_MAX_SIZE,
    SD3_5_MEDIUM_ID,
};
use mlx_rs::Array;

/// A tiny but structurally-faithful MMDiT-X arch: 4 layers with the FIRST 2 dual-attention (so both
/// a dual and a plain block are exercised) and the LAST (block 3) context_pre_only. head_dim 4,
/// 2 heads ⇒ hidden 8, patch 2, 4-ch, joint 6, pooled 5, caption 8, pos_embed_max 3 ⇒ table len 9.
fn tiny_medium_arch() -> Sd3Arch {
    Sd3Arch {
        num_layers: 4,
        head_dim: 4,
        num_heads: 2,
        patch_size: 2,
        in_channels: 4,
        out_channels: 4,
        joint_attention_dim: 6,
        pooled_projection_dim: 5,
        caption_projection_dim: 8, // == hidden (2*4)
        pos_embed_max_size: 3,
        time_proj_dim: 7,
        dual_attention_layers: 2,
    }
}

fn synthetic_weights(arch: &Sd3Arch) -> Weights {
    let entries: Vec<(String, Array)> = expected_transformer_tensors(arch)
        .into_iter()
        .map(|e| {
            let dims: Vec<i32> = e.shape.iter().map(|&d| d as i32).collect();
            (e.key, Array::ones::<f32>(&dims).unwrap())
        })
        .collect();
    let path = std::env::temp_dir().join(format!(
        "mlx_gen_sd3_medium_synthetic_{}.safetensors",
        entries.len()
    ));
    Array::save_safetensors(
        entries.iter().map(|(k, v)| (k.as_str(), v)),
        None::<&HashMap<String, String>>,
        &path,
    )
    .unwrap();
    Weights::from_file(&path).unwrap()
}

// -------------------------------------------------------------------------------------------------
// Config / arch constants
// -------------------------------------------------------------------------------------------------

#[test]
fn medium_arch_constants_match_real_config() {
    let a = Sd3Arch::medium();
    assert_eq!(a.num_layers, 24);
    assert_eq!(a.head_dim, 64);
    assert_eq!(a.num_heads, 24);
    assert_eq!(a.hidden(), 1536); // 24 * 64
    assert_eq!(a.patch_size, 2);
    assert_eq!(a.in_channels, 16);
    assert_eq!(a.out_channels, 16);
    assert_eq!(a.joint_attention_dim, 4096);
    assert_eq!(a.pooled_projection_dim, 2048);
    assert_eq!(a.caption_projection_dim, 1536); // == hidden
    assert_eq!(a.pos_embed_max_size, 384);
    assert_eq!(a.pos_embed_len(), 384 * 384); // 147456
    assert_eq!(a.time_proj_dim, 256);
    assert_eq!(a.dual_attention_layers, 13);
    // Re-exported constants agree.
    assert_eq!(MEDIUM_NUM_LAYERS, 24);
    assert_eq!(MEDIUM_HIDDEN, 1536);
    assert_eq!(MEDIUM_POS_EMBED_MAX_SIZE, 384);
    assert_eq!(MEDIUM_POS_EMBED_LEN, 147456);
    assert_eq!(MEDIUM_DUAL_ATTENTION_LAYERS, 13);
}

#[test]
fn medium_variant_descriptor_and_schedule() {
    let v = Sd3Variant::Medium;
    assert_eq!(v.id(), SD3_5_MEDIUM_ID);
    assert_eq!(v.hf_model(), "stabilityai/stable-diffusion-3.5-medium");
    assert_eq!(v.arch(), Sd3Arch::medium());
    // Medium is true-CFG (negative prompt + guidance), unlike Turbo.
    assert!(v.supports_true_cfg());
    assert!(v.default_guidance() > 1.0);
    assert!(v.default_steps() >= 20);
    let d = v.descriptor();
    assert_eq!(d.id, SD3_5_MEDIUM_ID);
    assert!(d.capabilities.supports_negative_prompt);
    assert!(d.capabilities.supports_guidance);
    // pos_embed_max 384 → 1440²-capable; descriptor advertises max_size 1440.
    assert!(d.capabilities.max_size >= 1440);
}

#[test]
fn dual_attention_block_predicate() {
    let a = Sd3Arch::medium();
    // First 13 (0..=12) are dual; 13..=23 are plain.
    for i in 0..13 {
        assert!(a.is_dual_attention_block(i), "block {i} should be dual");
    }
    for i in 13..24 {
        assert!(!a.is_dual_attention_block(i), "block {i} should be plain");
    }
    // Large is plain MMDiT — no dual blocks.
    let lg = Sd3Arch::large();
    assert_eq!(lg.dual_attention_layers, 0);
    assert!(!lg.is_dual_attention_block(0));
}

// -------------------------------------------------------------------------------------------------
// Expected-tensor table: count + dual-attn presence/absence + shapes
// -------------------------------------------------------------------------------------------------

#[test]
fn medium_expected_tensor_count_matches_real_checkpoint() {
    // The real SD3.5-Medium transformer is 909 tensors (sc-7867, safetensors header audit).
    // Derive the same count from the arch table — the strongest single assertion that the per-block
    // accounting (13 dual blocks w/ attn2 + 9-chunk norm1, 11 plain blocks, last context_pre_only)
    // is correct.
    assert_eq!(expected_tensor_count(&Sd3Arch::medium()), 909);
}

/// Independently re-derive the 909 count from first principles, so a future edit to the table
/// formula that still happens to sum to 909 for the wrong reasons is less likely to slip through.
#[test]
fn medium_count_derivation_is_909() {
    let top = 17; // pos_embed table + proj(2) + 2 embedders*2 linears*2 (8) + context(2) + norm_out(2) + proj_out(2)
    let mut blocks = 0usize;
    for i in 0..24 {
        let is_last = i == 23;
        let is_dual = i <= 12;
        let mut c = 0;
        c += 2; // norm1.linear w+b
        c += 2; // norm1_context.linear w+b
        c += 8; // attn image: to_q/to_k/to_v/to_out.0 (4 linears * 2)
        c += 6; // attn text: add_q/add_k/add_v (3 * 2)
        if !is_last {
            c += 2; // attn.to_add_out w+b
        }
        c += 4; // qk norms: norm_q/norm_k/norm_added_q/norm_added_k (weight only)
        c += 4; // ff: net.0.proj + net.2 (2 * 2)
        if !is_last {
            c += 4; // ff_context: net.0.proj + net.2 (2 * 2)
        }
        if is_dual {
            c += 8; // attn2: to_q/to_k/to_v/to_out.0 (4 * 2)
            c += 2; // attn2 norm_q/norm_k (weight only)
        }
        blocks += c;
    }
    assert_eq!(top + blocks, 909);
    assert_eq!(top + blocks, expected_tensor_count(&Sd3Arch::medium()));
}

#[test]
fn dual_attention_tensors_present_on_0_to_12_absent_after() {
    let table = expected_transformer_tensors(&Sd3Arch::medium());
    let keys: std::collections::HashSet<&str> = table.iter().map(|e| e.key.as_str()).collect();

    for i in 0..13 {
        for suf in [
            "attn2.to_q.weight",
            "attn2.to_q.bias",
            "attn2.to_k.weight",
            "attn2.to_v.weight",
            "attn2.to_out.0.weight",
            "attn2.norm_q.weight",
            "attn2.norm_k.weight",
        ] {
            let k = format!("transformer_blocks.{i}.{suf}");
            assert!(keys.contains(k.as_str()), "missing {k}");
        }
        // attn2 must NOT carry added/text projections or norm_added_*.
        for suf in [
            "attn2.add_q_proj.weight",
            "attn2.to_add_out.weight",
            "attn2.norm_added_q.weight",
        ] {
            let k = format!("transformer_blocks.{i}.{suf}");
            assert!(!keys.contains(k.as_str()), "unexpected {k}");
        }
    }
    for i in 13..24 {
        let k = format!("transformer_blocks.{i}.attn2.to_q.weight");
        assert!(!keys.contains(k.as_str()), "block {i} must have no attn2");
    }
}

#[test]
fn norm1_chunk_size_dual_vs_plain() {
    let arch = Sd3Arch::medium();
    let h = arch.hidden() as i64;
    let table = expected_transformer_tensors(&arch);
    let find = |k: &str| {
        table
            .iter()
            .find(|e| e.key == k)
            .unwrap_or_else(|| panic!("missing {k}"))
            .shape
            .clone()
    };
    // Dual block (0): extended AdaLN-zero, 9 chunks. Real-weight: [13824, 1536] = 9*1536.
    assert_eq!(
        find("transformer_blocks.0.norm1.linear.weight"),
        vec![9 * h, h]
    );
    assert_eq!(
        find("transformer_blocks.0.norm1.linear.weight"),
        vec![13824, 1536]
    );
    // Plain block (13): AdaLN-zero, 6 chunks. [9216, 1536] = 6*1536.
    assert_eq!(
        find("transformer_blocks.13.norm1.linear.weight"),
        vec![6 * h, h]
    );
    assert_eq!(
        find("transformer_blocks.13.norm1.linear.weight"),
        vec![9216, 1536]
    );
    // norm1_context stays 6 chunks for non-last blocks (both dual and plain).
    assert_eq!(
        find("transformer_blocks.0.norm1_context.linear.weight"),
        vec![6 * h, h]
    );
    assert_eq!(
        find("transformer_blocks.13.norm1_context.linear.weight"),
        vec![6 * h, h]
    );
}

#[test]
fn medium_top_level_and_last_block_shapes() {
    let arch = Sd3Arch::medium();
    let table = expected_transformer_tensors(&arch);
    let find = |k: &str| table.iter().find(|e| e.key == k).unwrap().shape.clone();

    // Learned positional table [1, 384*384, 1536] = [1, 147456, 1536] — NO RoPE.
    assert_eq!(find("pos_embed.pos_embed"), vec![1, 147456, 1536]);
    // Patchify Conv2d [1536, 16, 2, 2].
    assert_eq!(find("pos_embed.proj.weight"), vec![1536, 16, 2, 2]);
    // context_embedder [1536, 4096].
    assert_eq!(find("context_embedder.weight"), vec![1536, 4096]);
    // text_embedder.linear_1 [1536, 2048].
    assert_eq!(
        find("time_text_embed.text_embedder.linear_1.weight"),
        vec![1536, 2048]
    );
    // proj_out [64, 1536].
    assert_eq!(find("proj_out.weight"), vec![64, 1536]);
    // norm_out AdaLN-continuous [2*1536, 1536] = [3072, 1536].
    assert_eq!(find("norm_out.linear.weight"), vec![3072, 1536]);

    // Last block (23) is context_pre_only: no to_add_out / ff_context; norm1_context is 2 chunks.
    let keys: std::collections::HashSet<&str> = table.iter().map(|e| e.key.as_str()).collect();
    assert!(!keys.contains("transformer_blocks.23.attn.to_add_out.weight"));
    assert!(!keys.contains("transformer_blocks.23.ff_context.net.0.proj.weight"));
    assert!(!keys.contains("transformer_blocks.23.attn2.to_q.weight")); // 23 is also plain
    assert_eq!(
        find("transformer_blocks.23.norm1_context.linear.weight"),
        vec![2 * 1536, 1536]
    );
    // …but block 23 keeps norm1 at 6 chunks (plain, not dual).
    assert_eq!(
        find("transformer_blocks.23.norm1.linear.weight"),
        vec![6 * 1536, 1536]
    );
}

#[test]
fn medium_expected_table_has_no_duplicate_keys() {
    let table = expected_transformer_tensors(&Sd3Arch::medium());
    let unique: std::collections::HashSet<&String> = table.iter().map(|e| &e.key).collect();
    assert_eq!(unique.len(), table.len(), "duplicate keys in Medium table");
}

// -------------------------------------------------------------------------------------------------
// Converter + validator over a tiny synthetic MMDiT-X fixture (dense numeric path)
// -------------------------------------------------------------------------------------------------

#[test]
fn build_target_state_dict_round_trips_medium_layout() {
    let arch = tiny_medium_arch();
    let src = synthetic_weights(&arch);
    let out = build_target_state_dict(&src, &arch).unwrap();

    assert_eq!(out.len(), expected_tensor_count(&arch));
    for e in expected_transformer_tensors(&arch) {
        let t = out
            .get(&e.key)
            .unwrap_or_else(|| panic!("missing {}", e.key));
        let got: Vec<i64> = t.shape().iter().map(|&d| d as i64).collect();
        assert_eq!(got, e.shape, "shape for {}", e.key);
    }
    // The dual blocks (0,1) carry attn2; the plain blocks (2,3) do not.
    assert!(out.contains_key("transformer_blocks.0.attn2.to_q.weight"));
    assert!(out.contains_key("transformer_blocks.1.attn2.to_q.weight"));
    assert!(!out.contains_key("transformer_blocks.2.attn2.to_q.weight"));
}

#[test]
fn validate_arch_accepts_exact_medium_set() {
    let arch = Sd3Arch::medium();
    let table = expected_transformer_tensors(&arch);
    let provided: Vec<(&str, &[i64])> = table
        .iter()
        .map(|e| (e.key.as_str(), e.shape.as_slice()))
        .collect();
    validate_arch(&arch, provided.iter().copied()).unwrap();
}

#[test]
fn validate_arch_rejects_medium_set_missing_attn2() {
    let arch = Sd3Arch::medium();
    let mut keyed: HashMap<String, Vec<i64>> = expected_transformer_tensors(&arch)
        .into_iter()
        .map(|e| (e.key, e.shape))
        .collect();
    // Drop a dual-attention tensor → must be reported missing.
    keyed.remove("transformer_blocks.0.attn2.to_q.weight");
    let provided: Vec<(&str, &[i64])> = keyed
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_slice()))
        .collect();
    let err = validate_arch(&arch, provided.iter().copied())
        .unwrap_err()
        .to_string();
    assert!(err.contains("missing"), "err: {err}");
    assert!(err.contains("attn2.to_q.weight"), "err: {err}");
}

#[test]
fn validate_arch_rejects_large_set_against_medium_arch() {
    // A Large checkpoint validated against the Medium arch must fail loudly (wrong-repo guard):
    // different layer count, hidden size, and the absence of attn2 on Large.
    let medium = Sd3Arch::medium();
    let large_table = expected_transformer_tensors(&Sd3Arch::large());
    let provided: Vec<(&str, &[i64])> = large_table
        .iter()
        .map(|e| (e.key.as_str(), e.shape.as_slice()))
        .collect();
    assert!(validate_arch(&medium, provided.iter().copied()).is_err());
}

#[test]
fn quantize_keeps_attn2_norms_dense_and_packs_attn2_linears() {
    // Build a tiny dense map and quantize; attn2 Linears must pack (gain .scales), attn2 norms must
    // stay dense (no .scales), mirroring the joint-attention norm handling. MLX `quantize` only
    // supports group sizes 32/64/128, so the fixture's Linear inner dim (hidden) must be ≥ and a
    // multiple of the group — head_dim 32 × 2 heads ⇒ hidden 64, quantized at group_size 32.
    let arch = Sd3Arch {
        num_layers: 3,
        head_dim: 32,
        num_heads: 2,
        patch_size: 2,
        in_channels: 4,
        out_channels: 4,
        joint_attention_dim: 64,
        pooled_projection_dim: 64,
        caption_projection_dim: 64, // == hidden (32*2)
        pos_embed_max_size: 3,
        time_proj_dim: 64,
        dual_attention_layers: 2,
    };
    let src = synthetic_weights(&arch);
    let dense = build_target_state_dict(&src, &arch).unwrap();
    let q = quantize_sd3_transformer(dense, 8, 32).unwrap();

    // A quantized Linear gains a sibling `.scales`; a dense norm does not.
    assert!(
        q.contains_key("transformer_blocks.0.attn2.to_q.scales"),
        "attn2.to_q should be quantized"
    );
    assert!(
        !q.contains_key("transformer_blocks.0.attn2.norm_q.scales"),
        "attn2.norm_q must stay dense"
    );
    // The learned pos_embed table stays dense too.
    assert!(!q.contains_key("pos_embed.pos_embed.scales"));
}

// -------------------------------------------------------------------------------------------------
// Real-weight header validation (gated; the Medium snapshot is cached but large)
// -------------------------------------------------------------------------------------------------

/// Validate the REAL on-disk SD3.5-Medium `transformer/` against [`Sd3Arch::medium`] using only the
/// safetensors header (no weight load). Set `SD3_MEDIUM_SNAPSHOT` to a snapshot root that contains a
/// `transformer/` directory, e.g. the HF cache snapshot dir for `stabilityai/stable-diffusion-3.5-medium`.
#[test]
#[ignore = "requires the SD3.5-Medium snapshot; set SD3_MEDIUM_SNAPSHOT"]
fn real_medium_transformer_header_validates() {
    let root =
        std::env::var("SD3_MEDIUM_SNAPSHOT").expect("set SD3_MEDIUM_SNAPSHOT to the snapshot root");
    let dir = std::path::Path::new(&root).join("transformer");
    mlx_gen_sd3::validate_transformer_dir(&Sd3Arch::medium(), &dir).unwrap();
}
