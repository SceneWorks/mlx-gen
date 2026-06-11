//! sc-4568 e2e — the production `KolorsTrainer` (the `Trainer` contract realized on the Kolors
//! U-Net), driven through the core registry exactly as the SceneWorks worker will.
//!
//! `#[ignore]`d — needs the real `Kwai-Kolors/Kolors-diffusers` snapshot in the HF cache (or
//! `KOLORS_SNAPSHOT`), with the materialized `tokenizer/tokenizer.json`. Run:
//!   cargo test -p mlx-gen-kolors --release --test trainer_e2e -- --ignored --nocapture
//!
//! Proves the full prepare→load→cache→train→save lifecycle: a tiny captioned PNG dataset is
//! VAE/ChatGLM3-encoded and cached, AdamW training drives the epsilon flow down, and an adapter is
//! written that reloads through the REAL SDXL inference adapter path (`apply_sdxl_adapters[_with]`)
//! onto a fresh Kolors-loaded U-Net (Kolors' U-Net == the SDXL `UNet2DConditionModel`) — merging into
//! every trained target and forwarding finite under Kolors conditioning shapes.

use std::path::{Path, PathBuf};

use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterKind, AdapterSpec, CancelFlag, LoadSpec, NetworkType, TrainingConfig, TrainingItem,
    TrainingProgress, TrainingRequest, WeightsSource,
};
use mlx_gen_sdxl::{
    apply_sdxl_adapters, apply_sdxl_adapters_with, load_unet_kolors_dtype, LoraCoverage,
};
use mlx_rs::{Array, Dtype};

/// The Kolors SDXL-style micro-conditioning `time_ids = (H, W, 0, 0, H, W)` (mirrors the crate's
/// `pub(crate) model::kolors_time_ids`, which an integration test can't reach).
fn kolors_time_ids(batch: i32, height: i32, width: i32) -> Array {
    let (h, w) = (height as f32, width as f32);
    let row = [h, w, 0.0, 0.0, h, w];
    let mut v = Vec::with_capacity(batch as usize * 6);
    for _ in 0..batch {
        v.extend_from_slice(&row);
    }
    Array::from_slice(&v, &[batch, 6])
}

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("KOLORS_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Kwai-Kolors--Kolors-diffusers/snapshots");
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
fn config(network_type: NetworkType) -> TrainingConfig {
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
        ..Default::default()
    }
}

fn run(tmp: &Path, file_name: &str, network_type: NetworkType) -> (Vec<f32>, u32, PathBuf) {
    let items = make_dataset(tmp);
    // Reference the provider crate so its `inventory::submit!` registration is linked into this test
    // binary (a consumer that links the crate gets it for free; an integration test that names
    // nothing from the crate would otherwise have it dead-stripped).
    assert_eq!(mlx_gen_kolors::MODEL_ID, "kolors");

    let mut trainer =
        mlx_gen::load_trainer("kolors", &LoadSpec::new(WeightsSource::Dir(snapshot())))
            .expect("kolors trainer should be registered");

    let req = TrainingRequest {
        items,
        config: config(network_type),
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
        "[kolors-{network_type:?}] cached {cached}; steps {}; loss-mean first-quarter {first_q:.5} -> last-quarter {last_q:.5}",
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
#[ignore = "needs real Kolors weights"]
fn kolors_trainer_trains_and_writes_lora_that_reloads() {
    let tmp = std::env::temp_dir().join("kolors_trainer_lora_e2e");
    let (_losses, _steps, adapter_path) = run(&tmp, "swatch_lora.safetensors", NetworkType::Lora);

    // The produced adapter carries PEFT keys under the diffusers-UNet prefix + reload metadata.
    let w = Weights::from_file(&adapter_path).unwrap();
    assert_eq!(w.metadata("networkType"), Some("lora"));
    let n_targets = w.keys().filter(|k| k.ends_with(".lora_A.weight")).count();
    assert!(n_targets > 0, "adapter should contain LoRA factors");
    assert!(
        w.keys()
            .any(|k| k.starts_with("base_model.model.unet.") && k.ends_with(".to_q.lora_A.weight")),
        "adapter should carry PEFT-prefixed attention LoRA keys"
    );

    // Round-trip: reload through the REAL SDXL inference adapter path at Complete coverage (which
    // merges the down/mid/up attention the LoRA trained), onto a fresh f32 Kolors U-Net.
    let mut unet = load_unet_kolors_dtype(&snapshot(), Dtype::Float32).unwrap();
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
    .expect("LoRA adapter should reload through the SDXL inference path onto the Kolors U-Net");
    assert_eq!(
        report.merged, n_targets,
        "every trained LoRA target should merge under Complete coverage"
    );
    assert_eq!(report.skipped_keys, 0, "no LoRA key should be skipped");
    forward_finite(&unet);
    println!("[kolors-lora] e2e OK — {n_targets} targets reload + merge, forward finite");
}

#[test]
#[ignore = "needs real Kolors weights"]
fn kolors_trainer_trains_and_writes_lokr_that_reloads() {
    let tmp = std::env::temp_dir().join("kolors_trainer_lokr_e2e");
    let (_losses, _steps, adapter_path) = run(&tmp, "swatch_lokr.safetensors", NetworkType::Lokr);

    let w = Weights::from_file(&adapter_path).unwrap();
    assert_eq!(w.metadata("networkType"), Some("lokr"));
    assert!(w.metadata("decomposeFactor").is_some());
    let n_targets = w.keys().filter(|k| k.ends_with(".lokr_w1")).count();
    assert!(n_targets > 0, "adapter should contain LoKr factor keys");
    // LoKr targets the vendored attention surface (no mid_block — the SDXL LoKr loader keeps it out,
    // sc-2640), so every saved target reloads cleanly with nothing skipped.
    let mut unet = load_unet_kolors_dtype(&snapshot(), Dtype::Float32).unwrap();
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
    .expect("LoKr adapter should reload through the SDXL inference path onto the Kolors U-Net");
    assert_eq!(
        report.merged, n_targets,
        "every trained LoKr target should merge"
    );
    assert_eq!(report.skipped_keys, 0, "no LoKr key should be skipped");
    forward_finite(&unet);
    println!("[kolors-lokr] e2e OK — {n_targets} targets reload + merge, forward finite");
}

/// A forward over the adapted Kolors U-Net produces a finite eps (the reloaded adapter installs +
/// runs) — under Kolors conditioning shapes: ChatGLM context `[1, N, 4096]`, pooled `[1, 4096]`, and
/// the real-resolution `time_ids = (H, W, 0, 0, H, W)`.
fn forward_finite(unet: &mlx_gen_sdxl::UNet2DConditionModel) {
    let mk = |shape: &[i32], seed: u64| {
        mlx_rs::random::normal::<f32>(shape, None, None, Some(&mlx_rs::random::key(seed).unwrap()))
            .unwrap()
    };
    let x = mk(&[1, 8, 8, 4], 1);
    let cond = mk(&[1, 8, 4096], 2);
    let pooled = mk(&[1, 4096], 3);
    let time_ids = kolors_time_ids(1, 64, 64);
    let eps = unet.forward(&x, 500.0, &cond, &pooled, &time_ids).unwrap();
    let s = eps.sum(None).unwrap().item::<f32>();
    assert!(s.is_finite(), "reloaded-adapter forward should be finite");
}
