//! sc-3181: the canonical weight-map layout must exactly match the real
//! `sensenova/SenseNova-U1-8B-MoT` checkpoint.
//!
//! `#[ignore]`d — needs the real snapshot (env `SENSENOVA_U1_SNAPSHOT`, else the HF cache). This is
//! cheap: it parses `config.json` and the `model.safetensors.index.json` *key set* only — it does
//! **not** materialize the ~35 GB of tensors. It proves that [`expected_keys`] (the source of truth
//! the downstream module loaders share) accounts for every checkpoint tensor and invents none.
//!
//! Run:
//!   cargo test -p mlx-gen-sensenova --test weight_map_coverage -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen_sensenova::{check_coverage, NeoChatConfig};

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("SENSENOVA_U1_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--sensenova--SenseNova-U1-8B-MoT/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir (set SENSENOVA_U1_SNAPSHOT)")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

/// The tensor keys named in `model.safetensors.index.json` (`weight_map`).
fn checkpoint_keys(root: &std::path::Path) -> Vec<String> {
    let idx = root.join("model.safetensors.index.json");
    let text =
        std::fs::read_to_string(&idx).unwrap_or_else(|e| panic!("reading {}: {e}", idx.display()));
    let v: serde_json::Value = serde_json::from_str(&text).expect("parse index json");
    v["weight_map"]
        .as_object()
        .expect("weight_map object")
        .keys()
        .cloned()
        .collect()
}

#[test]
#[ignore = "needs the real SenseNova-U1-8B-MoT snapshot"]
fn canonical_keys_match_checkpoint_exactly() {
    let root = snapshot();
    let cfg = NeoChatConfig::from_dir(&root).expect("parse config.json");

    // Confirm we resolved the dense 8B-MoT, not the A3B MoE variant.
    assert!(
        !cfg.llm.is_moe(),
        "expected the dense 8B-MoT backbone (model_type {:?}, num_experts {:?})",
        cfg.llm.model_type,
        cfg.llm.num_experts,
    );

    let keys = checkpoint_keys(&root);
    println!("checkpoint tensors: {}", keys.len());

    let cov = check_coverage(keys.iter().map(String::as_str), &cfg);
    if !cov.is_complete() {
        panic!(
            "weight-map coverage mismatch:\n  missing ({}): {:#?}\n  unexpected ({}): {:#?}",
            cov.missing.len(),
            cov.missing,
            cov.unexpected.len(),
            cov.unexpected,
        );
    }
    assert_eq!(keys.len(), 1116, "8B-MoT ships 1116 tensors");
}
