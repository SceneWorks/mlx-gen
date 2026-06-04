//! Registry wiring + `load` rejection paths (sc-2679 S0 → S6).
//!
//! Verifies `ltx_2_3` self-registers into the `mlx-gen` model registry with the right descriptor,
//! that `load` rejects the not-yet-wired sibling features (quant / adapters / single-file source)
//! and an incomplete snapshot (S6 `load` assembles the full model — Gemma TE + transformer + VAE —
//! so a config-only dir no longer loads). The request-validation logic is unit-tested weight-free in
//! `model.rs` (`validate_request`).

use std::path::PathBuf;

use mlx_gen::{registry, LoadSpec, Modality, Quant, WeightsSource};

use mlx_gen_ltx::MODEL_ID;

const EROS_EMBEDDED_CONFIG: &str = r#"{
  "transformer": {
    "_class_name": "AVTransformer3DModel",
    "attention_head_dim": 128,
    "caption_channels": 3840,
    "cross_attention_dim": 4096,
    "in_channels": 128,
    "norm_eps": 1e-06,
    "num_attention_heads": 32,
    "num_layers": 48,
    "out_channels": 128,
    "audio_num_attention_heads": 32,
    "audio_attention_head_dim": 64,
    "audio_cross_attention_dim": 2048,
    "use_embeddings_connector": true,
    "connector_attention_head_dim": 128,
    "connector_num_attention_heads": 32,
    "connector_num_layers": 8,
    "connector_positional_embedding_max_pos": [4096],
    "connector_num_learnable_registers": 128,
    "use_middle_indices_grid": true,
    "apply_gated_attention": true,
    "connector_apply_gated_attention": true,
    "caption_projection_first_linear": false,
    "caption_projection_second_linear": false,
    "audio_connector_attention_head_dim": 64,
    "audio_connector_num_attention_heads": 32,
    "cross_attention_adaln": true,
    "rope_type": "split",
    "frequencies_precision": "float64",
    "positional_embedding_theta": 10000.0,
    "positional_embedding_max_pos": [20, 2048, 2048],
    "timestep_scale_multiplier": 1000
  }
}"#;

/// A throwaway model dir holding just `embedded_config.json` (S0 `load` only reads config).
fn temp_model_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ltx_s0_{}_{}", std::process::id(), tag));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("embedded_config.json"), EROS_EMBEDDED_CONFIG).unwrap();
    dir
}

#[test]
fn ltx_is_registered() {
    let reg = registry::generators()
        .find(|r| (r.descriptor)().id == MODEL_ID)
        .expect("ltx_2_3 not registered");
    let d = (reg.descriptor)();
    assert_eq!(d.id, "ltx_2_3");
    assert_eq!(d.family, "ltx");
    assert_eq!(d.modality, Modality::Video);
    // Distilled core: no guidance / negative prompt. LoRA in generate is wired (sc-2687); LoKr is the
    // sibling sc-2393, still off.
    assert!(!d.capabilities.supports_guidance);
    assert!(d.capabilities.supports_lora);
    assert!(!d.capabilities.supports_lokr);
    assert!(!d.capabilities.requires_sigma_shift);
}

#[test]
fn load_requires_full_model() {
    // S6: `load` assembles every component (Gemma TE + connector + transformer + upsampler + VAE),
    // so a config-only dir (no weight files) errors rather than returning a stub. (The full-model
    // load + generate is exercised by the real-weights `e2e_parity` gate; request validation is
    // unit-tested weight-free in `model.rs`.)
    let dir = temp_model_dir("load");
    assert!(
        registry::load(MODEL_ID, &LoadSpec::new(WeightsSource::Dir(dir.clone()))).is_err(),
        "config-only dir must not load (the full model's weight files are required)"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_rejects_unwired_features() {
    let dir = temp_model_dir("reject");
    // Single-file source.
    assert!(registry::load(
        MODEL_ID,
        &LoadSpec::new(WeightsSource::File(dir.join("embedded_config.json")))
    )
    .is_err());
    // Quantization (sibling slice — the transformer ships Q8 already).
    assert!(registry::load(
        MODEL_ID,
        &LoadSpec::new(WeightsSource::Dir(dir.clone())).with_quant(Quant::Q8)
    )
    .is_err());
    // (LoRA adapters are NO LONGER rejected — sc-2687 wires them; a LoKr file is rejected at apply
    // time, after the transformer loads, so it can't be exercised weight-free here. The full adapter
    // surface — routing, parity, per-pass, LoKr rejection — is gated by `lora_real_weights`.)

    std::fs::remove_dir_all(&dir).ok();
}
