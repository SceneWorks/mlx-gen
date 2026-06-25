//! SD3.5 diffusers→MLX converter + architecture validation, exercised against a tiny synthetic
//! layout (no multi-GB weights — the mlx-gen "test the remap without the real checkpoint" pattern).

use std::collections::HashMap;
use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_sd3::{
    build_target_state_dict, expected_tensor_count, expected_transformer_tensors,
    safetensors_header_shapes, validate_arch, Sd3Arch,
};
use mlx_rs::Array;

/// A tiny but structurally-faithful arch: 3 layers (so the LAST, context_pre_only, block is
/// exercised), head_dim 4, 2 heads ⇒ hidden 8, patch 2, 4-ch, joint 6, pooled 5, caption 8,
/// pos_embed_max 3 ⇒ table len 9, time proj 7. Every shape stays split/compatible.
fn tiny_arch() -> Sd3Arch {
    Sd3Arch {
        num_layers: 3,
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
    }
}

/// Build an in-memory `Weights` carrying exactly the expected tensor set for `arch`, every tensor
/// filled with ones at its expected shape.
fn synthetic_weights(arch: &Sd3Arch) -> Weights {
    let entries: Vec<(String, Array)> = expected_transformer_tensors(arch)
        .into_iter()
        .map(|e| {
            let dims: Vec<i32> = e.shape.iter().map(|&d| d as i32).collect();
            (e.key, Array::ones::<f32>(&dims).unwrap())
        })
        .collect();
    let path = std::env::temp_dir().join(format!(
        "mlx_gen_sd3_synthetic_{}.safetensors",
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

#[test]
fn large_expected_tensor_count_matches_real_checkpoint() {
    // The real SD3.5-Large transformer is 1227 tensors (sc-7850). Derive the same count from the
    // arch table: this is the strongest single assertion that the per-block tensor accounting
    // (incl. the context_pre_only last-block drops) is correct.
    assert_eq!(expected_tensor_count(&Sd3Arch::large()), 1227);
}

#[test]
fn expected_table_has_no_duplicate_keys() {
    use std::collections::HashSet;
    let table = expected_transformer_tensors(&Sd3Arch::large());
    let unique: HashSet<&String> = table.iter().map(|e| &e.key).collect();
    assert_eq!(
        unique.len(),
        table.len(),
        "duplicate keys in expected table"
    );
}

#[test]
fn last_block_is_context_pre_only() {
    let arch = tiny_arch();
    let table = expected_transformer_tensors(&arch);
    let keys: Vec<&str> = table.iter().map(|e| e.key.as_str()).collect();

    // A non-final block (block 0) carries the text-stream output proj + ff_context + AdaLN-zero
    // (6·hidden) norm1_context.
    assert!(keys.contains(&"transformer_blocks.0.attn.to_add_out.weight"));
    assert!(keys.contains(&"transformer_blocks.0.ff_context.net.0.proj.weight"));
    let n1c0 = table
        .iter()
        .find(|e| e.key == "transformer_blocks.0.norm1_context.linear.weight")
        .unwrap();
    assert_eq!(
        n1c0.shape,
        vec![6 * arch.hidden() as i64, arch.hidden() as i64]
    );

    // The final block (block 2) drops to_add_out / ff_context, and its norm1_context is
    // AdaLN-continuous (2·hidden).
    assert!(!keys.contains(&"transformer_blocks.2.attn.to_add_out.weight"));
    assert!(!keys.contains(&"transformer_blocks.2.ff_context.net.0.proj.weight"));
    assert!(!keys.contains(&"transformer_blocks.2.ff_context.net.2.weight"));
    // …but it keeps the text-stream input projections + qk-norms.
    assert!(keys.contains(&"transformer_blocks.2.attn.add_q_proj.weight"));
    assert!(keys.contains(&"transformer_blocks.2.attn.norm_added_q.weight"));
    let n1c_last = table
        .iter()
        .find(|e| e.key == "transformer_blocks.2.norm1_context.linear.weight")
        .unwrap();
    assert_eq!(
        n1c_last.shape,
        vec![2 * arch.hidden() as i64, arch.hidden() as i64]
    );
}

#[test]
fn top_level_shapes_match_arch() {
    let arch = Sd3Arch::large();
    let table = expected_transformer_tensors(&arch);
    let find = |k: &str| table.iter().find(|e| e.key == k).unwrap().shape.clone();

    // Learned positional table [1, 192*192, 2432] — NO RoPE.
    assert_eq!(find("pos_embed.pos_embed"), vec![1, 36864, 2432]);
    // Patchify Conv2d [2432, 16, 2, 2].
    assert_eq!(find("pos_embed.proj.weight"), vec![2432, 16, 2, 2]);
    // context_embedder [2432, 4096].
    assert_eq!(find("context_embedder.weight"), vec![2432, 4096]);
    // timestep_embedder.linear_1 [2432, 256].
    assert_eq!(
        find("time_text_embed.timestep_embedder.linear_1.weight"),
        vec![2432, 256]
    );
    // text_embedder.linear_1 [2432, 2048].
    assert_eq!(
        find("time_text_embed.text_embedder.linear_1.weight"),
        vec![2432, 2048]
    );
    // proj_out [64, 2432].
    assert_eq!(find("proj_out.weight"), vec![64, 2432]);
    // norm_out AdaLN-continuous [2*2432, 2432].
    assert_eq!(find("norm_out.linear.weight"), vec![4864, 2432]);
}

#[test]
fn build_target_state_dict_is_identity_over_expected_keys() {
    let arch = tiny_arch();
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
}

#[test]
fn build_target_state_dict_drops_non_arch_tensors() {
    let arch = tiny_arch();
    let mut src = synthetic_weights(&arch);
    // A stray tensor a checkpoint might carry (e.g. an EMA / training artifact) — must not leak.
    src.insert(
        "some.stray.tensor".to_string(),
        Array::ones::<f32>(&[2, 2]).unwrap(),
    );
    let out = build_target_state_dict(&src, &arch).unwrap();
    assert!(!out.contains_key("some.stray.tensor"));
    assert_eq!(out.len(), expected_tensor_count(&arch));
}

#[test]
fn build_target_state_dict_errors_on_missing_tensor() {
    let arch = tiny_arch();
    // Drop one required tensor from the synthetic set.
    let mut entries: Vec<(String, Array)> = expected_transformer_tensors(&arch)
        .into_iter()
        .map(|e| {
            let dims: Vec<i32> = e.shape.iter().map(|&d| d as i32).collect();
            (e.key, Array::ones::<f32>(&dims).unwrap())
        })
        .collect();
    entries.retain(|(k, _)| k != "proj_out.weight");
    let path = std::env::temp_dir().join("mlx_gen_sd3_missing.safetensors");
    Array::save_safetensors(
        entries.iter().map(|(k, v)| (k.as_str(), v)),
        None::<&HashMap<String, String>>,
        &path,
    )
    .unwrap();
    let src = Weights::from_file(&path).unwrap();
    let err = build_target_state_dict(&src, &arch)
        .unwrap_err()
        .to_string();
    assert!(err.contains("proj_out.weight"), "err was: {err}");
}

#[test]
fn validate_arch_passes_for_exact_set() {
    let arch = tiny_arch();
    let table = expected_transformer_tensors(&arch);
    let provided: Vec<(&str, &[i64])> = table
        .iter()
        .map(|e| (e.key.as_str(), e.shape.as_slice()))
        .collect();
    validate_arch(&arch, provided.iter().copied()).unwrap();
}

#[test]
fn validate_arch_reports_missing_extra_and_bad_shape() {
    let arch = tiny_arch();
    let table = expected_transformer_tensors(&arch);

    // Build a perturbed provided set: drop proj_out.bias (missing), add a bogus key (extra),
    // and give context_embedder.weight a wrong shape (bad_shape).
    let mut keyed: HashMap<String, Vec<i64>> =
        table.into_iter().map(|e| (e.key, e.shape)).collect();
    keyed.remove("proj_out.bias");
    keyed.insert("bogus.extra".to_string(), vec![1]);
    keyed.insert("context_embedder.weight".to_string(), vec![1, 1]);

    let provided: Vec<(&str, &[i64])> = keyed
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_slice()))
        .collect();
    let err = validate_arch(&arch, provided.iter().copied())
        .unwrap_err()
        .to_string();
    assert!(err.contains("1 missing"), "err: {err}");
    assert!(err.contains("1 extra"), "err: {err}");
    assert!(err.contains("1 shape mismatch"), "err: {err}");
}

#[test]
fn safetensors_header_reads_shapes_without_body() {
    use std::io::Write;
    let header =
        br#"{"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"__metadata__":{"a":"b"}}"#;
    let mut buf = Vec::new();
    buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
    buf.extend_from_slice(header);
    buf.extend_from_slice(&[7u8; 24]); // body must not be needed
    let path = std::env::temp_dir().join("mlx_gen_sd3_hdr.safetensors");
    std::fs::File::create(&path)
        .unwrap()
        .write_all(&buf)
        .unwrap();

    let shapes = safetensors_header_shapes(&path).unwrap();
    assert_eq!(shapes.len(), 1, "__metadata__ is skipped");
    assert_eq!(shapes.get("w"), Some(&vec![2i64, 3]));

    // Round-trip: header shapes feed straight into the validator.
    let _ = PathBuf::from(&path);
    std::fs::remove_file(&path).ok();
}
