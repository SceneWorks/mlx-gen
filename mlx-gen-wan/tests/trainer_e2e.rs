//! sc-3046 e2e — the production `WanMoeTrainer` (the dual-expert A14B `Trainer` contract), driven
//! through the core registry exactly as the SceneWorks worker will.
//!
//! `#[ignore]`d — needs a converted Wan2.2-A14B MoE snapshot (`$WAN_A14B_MODEL_DIR` or the
//! `~/.cache/mlx-gen-models/wan2_2_t2v_a14b_mlx_bf16` cache: `low_noise_model` + `high_noise_model`
//! + `t5_encoder` + `vae` + `tokenizer.json`). Run:
//!   cargo test -p mlx-gen-wan --release --test trainer_e2e -- --ignored --nocapture
//!
//! Proves the full prepare→load→cache→train→save lifecycle: a tiny captioned PNG dataset is
//! VAE/UMT5-encoded and cached (then the TE freed), the two experts train on their alternating
//! noise bands, and TWO adapters (high_noise + low_noise) are written that reload through the REAL
//! Wan inference merge (`merge_wan_adapters`) per expert.

use std::path::{Path, PathBuf};

use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterKind, AdapterSpec, CancelFlag, LoadSpec, MoeExpert, NetworkType, TrainingConfig,
    TrainingItem, TrainingProgress, TrainingRequest, WeightsSource,
};
use mlx_gen_wan::merge_wan_adapters;

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("WAN_A14B_MODEL_DIR") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    PathBuf::from(home).join(".cache/mlx-gen-models/wan2_2_t2v_a14b_mlx_bf16")
}

/// Two solid-colour swatch PNGs + captions in `dir`.
fn make_dataset(dir: &Path) -> Vec<TrainingItem> {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let mut items = Vec::new();
    for (i, color) in [[200u8, 40, 40], [40, 80, 200]].iter().enumerate() {
        let mut img = image::RgbImage::new(128, 128);
        for px in img.pixels_mut() {
            *px = image::Rgb(*color);
        }
        let path = dir.join(format!("img{i}.png"));
        img.save(&path).unwrap();
        items.push(TrainingItem {
            image_path: path,
            caption: format!("a solid colour swatch number {i}"),
        });
    }
    items
}

/// Merge a trained per-expert LoRA file into a fresh expert weight map, asserting it applies.
fn assert_reloads(file: &Path, expert: MoeExpert, weights_file: &str, n_targets: usize) {
    let mut w = Weights::from_file(snapshot().join(weights_file)).unwrap();
    let spec = AdapterSpec {
        path: file.to_path_buf(),
        scale: 1.0,
        kind: AdapterKind::Lora,
        pass_scales: None,
        moe_expert: Some(expert),
    };
    let report = merge_wan_adapters(&mut w, std::slice::from_ref(&spec), expert)
        .expect("trained LoRA should merge through the inference path");
    assert_eq!(
        report.applied, n_targets,
        "every trained {expert:?} target should merge"
    );
    assert!(
        report.skipped.is_empty(),
        "no {expert:?} key should be skipped: {:?}",
        report.skipped
    );
}

#[test]
#[ignore = "needs the converted Wan2.2-A14B MoE checkpoint"]
fn wan_moe_trainer_trains_both_experts_and_reloads() {
    let tmp = std::env::temp_dir().join("wan_moe_trainer_e2e");
    let items = make_dataset(&tmp);
    // Link the provider crate's `inventory::submit!` registration into this test binary.
    assert_eq!(mlx_gen_wan::MODEL_ID_T2V_14B, "wan2_2_t2v_14b");

    let mut trainer = mlx_gen::load_trainer(
        "wan2_2_t2v_14b",
        &LoadSpec::new(WeightsSource::Dir(snapshot())),
    )
    .expect("wan2_2_t2v_14b trainer should be registered");

    let config = TrainingConfig {
        rank: 8,
        alpha: 8.0,
        learning_rate: 1e-4,
        steps: 24,       // 12 per expert (alternating)
        resolution: 256, // bucketed to 256 -> 16x16 latent
        save_every: 0,
        seed: 7,
        network_type: NetworkType::Lora,
        ..Default::default()
    };
    let req = TrainingRequest {
        items,
        config,
        output_dir: tmp.join("out"),
        file_name: "swatch.safetensors".to_string(),
        trigger_words: vec![],
        cancel: CancelFlag::new(),
    };

    let mut losses: Vec<f32> = Vec::new();
    let mut cached = 0u32;
    let out = trainer
        .train(&req, &mut |p| match p {
            TrainingProgress::Caching { current, .. } => cached = current,
            TrainingProgress::Training { loss, .. } => losses.push(loss),
            _ => {}
        })
        .expect("training should succeed");

    assert_eq!(cached, 2, "both dataset items should be cached");
    assert_eq!(out.steps, 24);
    assert_eq!(losses.len(), 24);
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "no NaN/Inf losses (per-expert functional autograd is sane)"
    );
    // Per-step loss mixes the two experts + σ variance; compare the first vs last quarter mean.
    let q = losses.len() / 4;
    let mean = |s: &[f32]| s.iter().sum::<f32>() / s.len() as f32;
    let (first_q, last_q) = (mean(&losses[..q]), mean(&losses[losses.len() - q..]));
    println!(
        "[wan-moe] cached {cached}; steps {}; loss-mean first-quarter {first_q:.5} -> last-quarter {last_q:.5}",
        out.steps
    );
    assert!(
        last_q < first_q * 0.95,
        "windowed loss-mean should fall on real data: {first_q:.5} -> {last_q:.5}"
    );

    // Two adapter files written: the high_noise (returned) + low_noise pair.
    let high = out.adapter_path.clone();
    let low = tmp.join("out").join("swatch.low_noise.safetensors");
    assert!(
        high.file_name()
            .unwrap()
            .to_string_lossy()
            .contains("high_noise"),
        "primary adapter should be the high_noise file: {}",
        high.display()
    );
    assert!(
        high.exists() && low.exists(),
        "both expert adapters should be written"
    );

    let wh = Weights::from_file(&high).unwrap();
    assert_eq!(wh.metadata("networkType"), Some("lora"));
    let n_targets = wh.keys().filter(|k| k.ends_with(".lora_A.weight")).count();
    assert!(n_targets > 0, "adapter should contain LoRA factors");
    assert!(
        wh.keys().any(|k| k.ends_with(".self_attn.q.lora_A.weight")),
        "adapter should carry native Wan attention keys"
    );

    // Round-trip: each expert's file merges into its own fresh weight map via the inference path.
    assert_reloads(
        &high,
        MoeExpert::High,
        "high_noise_model.safetensors",
        n_targets,
    );
    assert_reloads(
        &low,
        MoeExpert::Low,
        "low_noise_model.safetensors",
        n_targets,
    );
    println!("[wan-moe] e2e OK — {n_targets} targets/expert reload through merge_wan_adapters");
}
