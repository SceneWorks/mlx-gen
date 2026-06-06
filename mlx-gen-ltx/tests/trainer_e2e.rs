//! sc-3047 e2e — the production `LtxTrainer` (the `Trainer` contract realized on the LTX-2.3 video
//! DiT), driven through the core registry exactly as the SceneWorks worker will.
//!
//! `#[ignore]`d — needs a real LTX-2.3 split-weight snapshot (`$LTX_BASE_DIR` or the SceneWorks
//! `ltx_2_3_base_q8` cache) AND the Gemma-3-12B text-encoder snapshot (`$LTX_GEMMA_DIR` or the HF
//! cache). Run:
//!   cargo test -p mlx-gen-ltx --release --test trainer_e2e -- --ignored --nocapture
//!
//! Proves the full prepare→load→cache→train→save lifecycle: a tiny captioned PNG dataset is
//! VAE/Gemma-encoded and cached (then the 24 GB TE is freed), AdamW training drives the
//! rectified-flow loss down, and a LoRA is written that reloads through the REAL LTX inference path
//! (`apply_ltx_adapters`) onto a fresh DiT — applying to every trained target and forwarding finite.

use std::path::{Path, PathBuf};

use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterKind, AdapterSpec, CancelFlag, LoadSpec, NetworkType, TrainingConfig, TrainingItem,
    TrainingProgress, TrainingRequest, WeightsSource,
};
use mlx_gen_ltx::config::{LtxConfig, SplitModel};
use mlx_gen_ltx::pipeline::NUM_DENOISE_PASSES;
use mlx_gen_ltx::positions::create_position_grid;
use mlx_gen_ltx::transformer::Precision;
use mlx_gen_ltx::{apply_ltx_adapters, LtxDiT};

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("LTX_BASE_DIR") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    PathBuf::from(home)
        .join("Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8")
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

#[test]
#[ignore = "needs real LTX-2.3 + Gemma weights"]
fn ltx_trainer_trains_and_writes_lora_that_reloads() {
    let tmp = std::env::temp_dir().join("ltx_trainer_e2e");
    let items = make_dataset(&tmp);
    // Reference the provider crate so its `inventory::submit!` registration is linked into this test
    // binary (a consumer that links the crate gets it for free).
    assert_eq!(mlx_gen_ltx::MODEL_ID, "ltx_2_3");

    let mut trainer =
        mlx_gen::load_trainer("ltx_2_3", &LoadSpec::new(WeightsSource::Dir(snapshot())))
            .expect("ltx_2_3 trainer should be registered");

    // LTX is a deep 48-block DiT with the full attention surface adapted (384 residuals stacked
    // sequentially), so it is far more LR-sensitive than the shallower image DiTs — the reference LTX
    // trainer's default LR is 1e-4 (10× below the image trainers). At 1e-3 the growing `B` compounds
    // across 48 blocks and the velocity diverges; 1e-4 is the realistic, stable rate.
    let config = TrainingConfig {
        rank: 8,
        alpha: 8.0,
        learning_rate: 1e-4,
        steps: 32,
        resolution: 256, // bucketed to 256 -> 8x8 latent (64 tokens), fast
        save_every: 0,
        seed: 7,
        network_type: NetworkType::Lora,
        ..Default::default()
    };
    let req = TrainingRequest {
        items,
        config,
        output_dir: tmp.join("out"),
        file_name: "swatch_lora.safetensors".to_string(),
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
    assert_eq!(out.steps, 32, "all micro-steps should run");
    assert_eq!(losses.len(), 32);
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "no NaN/Inf losses (functional autograd over the Q8 base is sane)"
    );
    // Each step samples a fresh σ + noise, so per-step loss is dominated by σ variance; compare the
    // first vs last quarter mean (variance averages out).
    let q = losses.len() / 4;
    let mean = |s: &[f32]| s.iter().sum::<f32>() / s.len() as f32;
    let (first_q, last_q) = (mean(&losses[..q]), mean(&losses[losses.len() - q..]));
    println!(
        "[ltx-lora] cached {cached}; steps {}; loss-mean first-quarter {first_q:.5} -> last-quarter {last_q:.5}",
        out.steps
    );
    assert!(
        last_q < first_q * 0.9,
        "windowed loss-mean should fall on real data: {first_q:.5} -> {last_q:.5}"
    );

    // The produced adapter carries the reference LoRA keys + reload metadata.
    assert!(out.adapter_path.exists(), "adapter file should be written");
    let w = Weights::from_file(&out.adapter_path).unwrap();
    assert_eq!(w.metadata("networkType"), Some("lora"));
    let n_targets = w.keys().filter(|k| k.ends_with(".lora_A.weight")).count();
    assert!(n_targets > 0, "adapter should contain LoRA factors");
    assert!(
        w.keys().any(|k| k.ends_with(".attn1.to_q.lora_A.weight")),
        "adapter should carry attention LoRA keys"
    );

    // Round-trip: free the trainer's model, then reload the adapter through the REAL inference path
    // (`apply_ltx_adapters`) onto a fresh DiT, and confirm every trained target installs + forwards
    // finite (residual over the Q8 base, the same the trainer trained on).
    let adapter_path = out.adapter_path.clone();
    drop(trainer);
    let cfg = LtxConfig::from_model_dir(&snapshot()).unwrap();
    let split = SplitModel::from_model_dir(&snapshot()).unwrap();
    let tw = Weights::from_file(snapshot().join("transformer.safetensors")).unwrap();
    let mut dit =
        LtxDiT::from_weights(&tw, &cfg, Precision::quant_bf16(split.bits, split.group)).unwrap();
    let report = apply_ltx_adapters(
        &mut dit,
        &[AdapterSpec::new(adapter_path, 1.0, AdapterKind::Lora)],
        NUM_DENOISE_PASSES,
    )
    .expect("LoRA adapter should reload through the inference path");
    assert_eq!(
        report.applied, n_targets,
        "every trained LoRA target should reload"
    );
    assert!(
        report.skipped.is_empty(),
        "no LoRA key should be skipped: {:?}",
        report.skipped
    );

    // A forward with the reloaded LoRA installed produces a finite velocity (8x8 latent = 64 tokens).
    let mk = |shape: &[i32], seed: u64| {
        mlx_rs::random::normal::<f32>(shape, None, None, Some(&mlx_rs::random::key(seed).unwrap()))
            .unwrap()
    };
    let latent = mk(&[1, 64, 128], 1);
    let timestep = mlx_rs::Array::from_slice(&[0.5f32], &[1, 1]);
    let context = mk(&[1, 16, 4096], 2);
    let positions = create_position_grid(1, 1, 8, 8);
    let v = dit
        .forward(&latent, &timestep, &context, None, &positions)
        .unwrap();
    let s = v.sum(None).unwrap().item::<f32>();
    assert!(s.is_finite(), "reloaded-LoRA forward should be finite");
    println!("[ltx-lora] e2e OK — {n_targets} targets reload, forward finite");
}
