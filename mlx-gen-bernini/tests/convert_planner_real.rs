//! sc-5144: Bernini planner-weights converter — real-weight conversion smoke (`#[ignore]`).
//!
//! Assembles the native MLX planner snapshot from a full `ByteDance/Bernini-Diffusers` package and
//! verifies the four component safetensors have the exact expected tensor counts and representative
//! shapes, that `mllm.lm_head` was dropped, and that the shared diffusers encoders + configs are
//! present. The package is ~168 GB F32 and lives outside CI, so the heavy test is `#[ignore]`d; the
//! key-routing / count logic is covered by the in-crate unit tests in `src/convert.rs`.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_bernini::convert::assemble_bernini_planner_snapshot;

fn hf_snapshot(repo: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots");
    std::fs::read_dir(snaps)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.is_dir())
}

/// Tensor count + the shape of one key, from a safetensors file.
fn count_and_shape(file: &PathBuf, key: &str) -> (usize, Vec<i32>) {
    let w = Weights::from_file(file).expect("open component safetensors");
    let n = w.keys().count();
    let shape = w.require(key).expect("representative key").shape().to_vec();
    (n, shape)
}

#[test]
#[ignore = "real weights: extracts the planner components from the ~168 GB Bernini-Diffusers index"]
fn assemble_real_bernini_planner() {
    let pkg = hf_snapshot("ByteDance/Bernini-Diffusers")
        .expect("ByteDance/Bernini-Diffusers snapshot in the HF cache");
    let out = PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/mlx-gen-models/bernini_planner_mlx_bf16");

    // Link the shared encoders (zero-copy) — this is the default the e2e pipeline (sc-5145) uses.
    let dir =
        assemble_bernini_planner_snapshot(&out, &pkg, true).expect("assemble planner snapshot");

    // Qwen2.5-VL: 728 tensors (729 − lm_head), embed_tokens is [vocab, hidden] = [152064, 3584].
    let (n, embed) = count_and_shape(
        &dir.join("qwen2_5_vl.safetensors"),
        "model.embed_tokens.weight",
    );
    assert_eq!(n, 728, "qwen2_5_vl tensor count (lm_head dropped)");
    assert_eq!(embed, vec![152064, 3584], "embed_tokens shape");
    // lm_head must be absent.
    let w = Weights::from_file(dir.join("qwen2_5_vl.safetensors")).unwrap();
    assert!(
        !w.keys().any(|k| k.contains("lm_head")),
        "lm_head must be dropped from the planner backbone"
    );

    // Connector: 12 tensors; proj_gen.0 is Linear(3584 → 4096) → weight [4096, 3584].
    let (n, proj) = count_and_shape(&dir.join("connector.safetensors"), "proj_gen.0.weight");
    assert_eq!(n, 12, "connector tensor count");
    assert_eq!(proj, vec![4096, 3584], "proj_gen.0 (gen branch in) shape");

    // ViT decoder: 140 tensors; final_layer.linear is Linear(width 4096 → target 3584).
    let (n, fin) = count_and_shape(
        &dir.join("vit_decoder.safetensors"),
        "net.final_layer.linear.weight",
    );
    assert_eq!(n, 140, "vit_decoder tensor count");
    assert_eq!(fin, vec![3584, 4096], "final_layer.linear shape");

    // Mask token: 1 tensor, [1, num_mask_token=4096, hidden=3584].
    let (n, mask) = count_and_shape(&dir.join("mask_tokens.safetensors"), "mask_tokens");
    assert_eq!(n, 1, "mask_tokens tensor count");
    assert_eq!(mask, vec![1, 4096, 3584], "mask_tokens shape");

    // bf16 native dtype for the backbone.
    assert_eq!(
        Weights::from_file(dir.join("qwen2_5_vl.safetensors"))
            .unwrap()
            .require("model.embed_tokens.weight")
            .unwrap()
            .dtype(),
        mlx_rs::Dtype::Bfloat16,
        "planner backbone saved bf16"
    );

    // Configs + shared encoders present.
    for f in [
        "qwen2_5_vl_config.json",
        "bernini_planner.json",
        "transformer_config.json",
        "transformer_2_config.json",
    ] {
        assert!(dir.join(f).is_file(), "config {f} present");
    }
    for d in [
        "t5_text_encoder",
        "t5_tokenizer",
        "vae",
        "scheduler",
        "mllm",
    ] {
        assert!(dir.join(d).exists(), "shared component {d} present");
    }
}
