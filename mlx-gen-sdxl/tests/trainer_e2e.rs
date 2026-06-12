//! sc-3045 e2e — the production `SdxlTrainer` (the `Trainer` contract realized on the SDXL U-Net),
//! driven through the core registry exactly as the SceneWorks worker will.
//!
//! `#[ignore]`d — needs the real `stabilityai/stable-diffusion-xl-base-1.0` snapshot in the HF cache
//! (or `SDXL_SNAPSHOT`). Run:
//!   cargo test -p mlx-gen-sdxl --release --test trainer_e2e -- --ignored --nocapture
//!
//! Proves the full prepare→load→cache→train→save lifecycle: a tiny captioned PNG dataset is
//! VAE/dual-CLIP-encoded and cached, AdamW training drives the epsilon flow down, and an adapter is
//! written that reloads through the REAL SDXL inference path (`apply_sdxl_adapters[_with]`) onto a
//! fresh U-Net — merging into every trained target and forwarding finite.

use std::path::{Path, PathBuf};

use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterKind, AdapterSpec, CancelFlag, LoadSpec, NetworkType, TrainingConfig, TrainingItem,
    TrainingProgress, TrainingRequest, WeightsSource,
};
use mlx_gen_sdxl::{
    apply_sdxl_adapters, apply_sdxl_adapters_with, load_unet, text_time_ids, LoraCoverage,
};

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("SDXL_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

/// Two solid-colour swatch PNGs + captions in `dir`.
fn make_dataset(dir: &Path) -> Vec<TrainingItem> {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let mut items = Vec::new();
    for (i, color) in [[200u8, 40, 40], [40, 80, 200]].iter().enumerate() {
        let mut img = image::RgbImage::new(96, 96);
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

/// A tiny config: small rank, low resolution (bucketed to 64 → 8×8 latent), few steps.
fn config(network_type: NetworkType, gradient_checkpointing: bool) -> TrainingConfig {
    TrainingConfig {
        rank: 8,
        alpha: 8.0,
        learning_rate: 1e-3,
        steps: 16,
        resolution: 64,
        save_every: 0,
        seed: 7,
        network_type,
        decompose_factor: -1,
        gradient_checkpointing,
        ..Default::default()
    }
}

fn run(tmp: &Path, file_name: &str, network_type: NetworkType) -> (Vec<f32>, u32, PathBuf) {
    run_cfg(tmp, file_name, network_type, false)
}

fn run_cfg(
    tmp: &Path,
    file_name: &str,
    network_type: NetworkType,
    gradient_checkpointing: bool,
) -> (Vec<f32>, u32, PathBuf) {
    let items = make_dataset(tmp);
    // Reference the provider crate so its `inventory::submit!` registration is linked into this test
    // binary (a consumer that links the crate gets it for free; an integration test that names
    // nothing from the crate would otherwise have it dead-stripped).
    assert_eq!(mlx_gen_sdxl::MODEL_ID, "sdxl");

    let mut trainer = mlx_gen::load_trainer("sdxl", &LoadSpec::new(WeightsSource::Dir(snapshot())))
        .expect("sdxl trainer should be registered");

    let req = TrainingRequest {
        items,
        config: config(network_type, gradient_checkpointing),
        output_dir: tmp.join("out"),
        file_name: file_name.to_string(),
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
    assert_eq!(out.steps, 16, "all micro-steps should run");
    assert_eq!(losses.len(), 16);
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "no NaN/Inf losses (not diverging)"
    );
    // Each step samples a fresh integer timestep + noise, so per-step loss is dominated by timestep
    // variance, not a monotonic curve. Compare the first vs last quarter mean (variance averages out).
    let q = losses.len() / 4;
    let mean = |s: &[f32]| s.iter().sum::<f32>() / s.len() as f32;
    let (first_q, last_q) = (mean(&losses[..q]), mean(&losses[losses.len() - q..]));
    println!(
        "[sdxl-{network_type:?}] cached {cached}; steps {}; loss-mean first-quarter {first_q:.5} -> last-quarter {last_q:.5}",
        out.steps
    );
    assert!(
        last_q < first_q * 0.9,
        "windowed loss-mean should fall on real data: {first_q:.5} -> {last_q:.5}"
    );
    assert!(out.adapter_path.exists(), "adapter file should be written");
    (losses, out.steps, out.adapter_path)
}

#[test]
#[ignore = "needs real SDXL weights"]
fn sdxl_trainer_trains_and_writes_lora_that_reloads() {
    let tmp = std::env::temp_dir().join("sdxl_trainer_lora_e2e");
    let (_losses, _steps, adapter_path) = run(&tmp, "swatch_lora.safetensors", NetworkType::Lora);

    // The produced adapter carries PEFT keys under the SDXL prefix + reload metadata.
    let w = Weights::from_file(&adapter_path).unwrap();
    assert_eq!(w.metadata("networkType"), Some("lora"));
    let n_targets = w.keys().filter(|k| k.ends_with(".lora_A.weight")).count();
    assert!(n_targets > 0, "adapter should contain LoRA factors");
    assert!(
        w.keys()
            .any(|k| k.starts_with("base_model.model.unet.") && k.ends_with(".to_q.lora_A.weight")),
        "adapter should carry PEFT-prefixed attention LoRA keys"
    );

    // Round-trip: reload through the REAL inference path at the model::load default coverage
    // (Complete — which merges the down/mid/up attention the LoRA trained), onto a fresh f32 U-Net.
    let mut unet = load_unet(&snapshot()).unwrap();
    let report = apply_sdxl_adapters_with(
        &mut unet,
        &[AdapterSpec {
            path: adapter_path,
            scale: 1.0,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }],
        LoraCoverage::Complete,
    )
    .expect("LoRA adapter should reload through the inference path");
    assert_eq!(
        report.merged, n_targets,
        "every trained LoRA target should merge under Complete coverage"
    );
    assert_eq!(report.skipped_keys, 0, "no LoRA key should be skipped");
    forward_finite(&unet);
    println!("[sdxl-lora] e2e OK — {n_targets} targets reload + merge, forward finite");
}

#[test]
#[ignore = "needs real SDXL weights"]
fn sdxl_trainer_trains_and_writes_lokr_that_reloads() {
    let tmp = std::env::temp_dir().join("sdxl_trainer_lokr_e2e");
    let (_losses, _steps, adapter_path) = run(&tmp, "swatch_lokr.safetensors", NetworkType::Lokr);

    let w = Weights::from_file(&adapter_path).unwrap();
    assert_eq!(w.metadata("networkType"), Some("lokr"));
    assert!(w.metadata("decomposeFactor").is_some());
    let n_targets = w.keys().filter(|k| k.ends_with(".lokr_w1")).count();
    assert!(n_targets > 0, "adapter should contain LoKr factor keys");
    // LoKr targets the vendored attention surface (no mid_block — the SDXL LoKr loader keeps it out,
    // sc-2640), so every saved target reloads cleanly with nothing skipped.
    let mut unet = load_unet(&snapshot()).unwrap();
    let report = apply_sdxl_adapters(
        &mut unet,
        &[AdapterSpec {
            path: adapter_path,
            scale: 1.0,
            kind: AdapterKind::Lokr,
            pass_scales: None,
            moe_expert: None,
        }],
    )
    .expect("LoKr adapter should reload through the inference path");
    assert_eq!(
        report.merged, n_targets,
        "every trained LoKr target should merge"
    );
    assert_eq!(report.skipped_keys, 0, "no LoKr key should be skipped");
    forward_finite(&unet);
    println!("[sdxl-lokr] e2e OK — {n_targets} targets reload + merge, forward finite");
}

/// sc-4941 — the `gradient_checkpointing` path (per-block recompute) trains end-to-end: convergence
/// matches the dense path (the block-ckpt forward+grads are validated bit-close to dense by the
/// `block_ckpt_grads_match_dense` unit gate), and the adapter still saves + reloads.
#[test]
#[ignore = "needs real SDXL weights"]
fn sdxl_trainer_gradient_checkpointing_converges() {
    let tmp = std::env::temp_dir().join("sdxl_trainer_gc_e2e");
    let (losses, steps, adapter_path) = run_cfg(
        &tmp,
        "swatch_lora_gc.safetensors",
        NetworkType::Lora,
        true, // gradient_checkpointing ON → forward_block_checkpointed
    );
    assert_eq!(steps, 16);
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "no NaN/Inf under checkpointing"
    );
    let q = losses.len() / 4;
    let mean = |s: &[f32]| s.iter().sum::<f32>() / s.len() as f32;
    let (first_q, last_q) = (mean(&losses[..q]), mean(&losses[losses.len() - q..]));
    assert!(
        last_q < first_q * 0.9,
        "checkpointed training should converge like dense: {first_q:.5} -> {last_q:.5}"
    );
    let mut unet = load_unet(&snapshot()).unwrap();
    let report = apply_sdxl_adapters_with(
        &mut unet,
        &[AdapterSpec {
            path: adapter_path,
            scale: 1.0,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }],
        LoraCoverage::Complete,
    )
    .expect("checkpointed LoRA adapter should reload");
    assert!(report.merged > 0, "checkpointed adapter should merge");
    forward_finite(&unet);
    println!(
        "[sdxl-gc] e2e OK — checkpointed training converged {first_q:.5} -> {last_q:.5}, reloads"
    );
}

/// A forward over the adapted U-Net produces a finite eps (the reloaded adapter installs + runs).
fn forward_finite(unet: &mlx_gen_sdxl::UNet2DConditionModel) {
    let mk = |shape: &[i32], seed: u64| {
        mlx_rs::random::normal::<f32>(shape, None, None, Some(&mlx_rs::random::key(seed).unwrap()))
            .unwrap()
    };
    let x = mk(&[1, 8, 8, 4], 1);
    let cond = mk(&[1, 8, 2048], 2);
    let pooled = mk(&[1, 1280], 3);
    let time_ids = text_time_ids(1);
    let eps = unet.forward(&x, 500.0, &cond, &pooled, &time_ids).unwrap();
    let s = eps.sum(None).unwrap().item::<f32>();
    assert!(s.is_finite(), "reloaded-adapter forward should be finite");
}
