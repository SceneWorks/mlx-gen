//! Registry wiring + config-driven `load` (S0).
//!
//! Verifies `wan2_2_ti2v_5b` self-registers into the `mlx-gen` model registry with the right
//! descriptor, that `load` reads the model's `config.json` (auto-detecting the 5B preset) and
//! returns a stub whose `generate` errors with an explicit "S1–S5 pending" message, and that the
//! not-yet-wired sibling features (quant / adapters / single-file source / precision override) are
//! rejected.

use std::path::PathBuf;

use mlx_gen::{
    registry, AdapterKind, AdapterSpec, GenerationRequest, LoadSpec, Modality, Precision, Quant,
    WeightsSource,
};

use mlx_gen_wan::MODEL_ID;

/// The 5B's serialized `config.json` (the `convert_wan.py` schema; model_type ti2v + dim 3072).
const TI2V_5B_CONFIG: &str = r#"{
  "model_type": "ti2v",
  "model_version": "2.2",
  "patch_size": [1, 2, 2],
  "in_dim": 48,
  "dim": 3072,
  "ffn_dim": 14336,
  "out_dim": 48,
  "num_heads": 24,
  "num_layers": 30,
  "vae_z_dim": 48,
  "vae_stride": [4, 16, 16],
  "dual_model": false,
  "sample_shift": 5.0,
  "sample_steps": 40,
  "sample_guide_scale": 5.0,
  "sample_fps": 24,
  "max_area": 901120
}"#;

/// A throwaway model dir holding just `config.json` (S0 `load` only reads config).
fn temp_model_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("wan_s0_{}_{}", std::process::id(), tag));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), TI2V_5B_CONFIG).unwrap();
    dir
}

#[test]
fn wan_is_registered() {
    let reg = registry::generators()
        .find(|r| (r.descriptor)().id == MODEL_ID)
        .expect("wan2_2_ti2v_5b not registered");
    let d = (reg.descriptor)();
    assert_eq!(d.id, "wan2_2_ti2v_5b");
    assert_eq!(d.family, "wan");
    assert_eq!(d.modality, Modality::Video);
    // 5B uses real CFG + negative prompt, advertises a single image reference (TI2V), KV cache.
    assert!(d.capabilities.supports_guidance);
    assert!(d.capabilities.supports_negative_prompt);
    assert!(d.capabilities.supports_kv_cache);
    assert!(!d.capabilities.supports_lora);
    assert!(d.capabilities.samplers.contains(&"unipc"));
}

#[test]
fn load_reads_config_and_stubs_generate() {
    let dir = temp_model_dir("load");
    let g = registry::load(MODEL_ID, &LoadSpec::new(WeightsSource::Dir(dir.clone())))
        .expect("load should succeed (reads config.json)");
    assert_eq!(g.descriptor().id, MODEL_ID);

    // validate accepts a 32-aligned request with 1+4k frames; rejects sub-tile + bad frame counts.
    let ok = GenerationRequest {
        width: 704,
        height: 1280,
        frames: Some(81),
        ..Default::default()
    };
    assert!(g.validate(&ok).is_ok());
    let bad_size = GenerationRequest {
        width: 16,
        height: 1280,
        ..Default::default()
    };
    assert!(g.validate(&bad_size).is_err());
    let bad_frames = GenerationRequest {
        width: 704,
        height: 1280,
        frames: Some(80),
        ..Default::default()
    };
    assert!(g.validate(&bad_frames).is_err());

    // generate is an explicit WIP error until S1–S5.
    let mut noop = |_p| {};
    let err = g.generate(&ok, &mut noop).unwrap_err().to_string();
    assert!(err.contains("S1"), "expected WIP message, got: {err}");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_rejects_unwired_features() {
    let dir = temp_model_dir("reject");
    // Single-file source.
    assert!(registry::load(
        MODEL_ID,
        &LoadSpec::new(WeightsSource::File(dir.join("config.json")))
    )
    .is_err());
    // Quantization (sc-2682).
    assert!(registry::load(
        MODEL_ID,
        &LoadSpec::new(WeightsSource::Dir(dir.clone())).with_quant(Quant::Q8)
    )
    .is_err());
    // Precision override (the dense path runs f32 activations).
    let mut spec = LoadSpec::new(WeightsSource::Dir(dir.clone()));
    spec.precision = Precision::Fp32;
    assert!(registry::load(MODEL_ID, &spec).is_err());
    // Adapters (sc-2683 / sc-2393).
    let adapters = vec![AdapterSpec {
        path: dir.join("x.safetensors"),
        scale: 1.0,
        kind: AdapterKind::Lora,
    }];
    assert!(registry::load(
        MODEL_ID,
        &LoadSpec::new(WeightsSource::Dir(dir.clone())).with_adapters(adapters)
    )
    .is_err());

    std::fs::remove_dir_all(&dir).ok();
}
