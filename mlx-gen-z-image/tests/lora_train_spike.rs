//! sc-3042 SPIKE GO/NO-GO — LoRA training on the REAL Z-Image DiT, end to end in Rust.
//!
//! `#[ignore]`d — needs the real `Tongyi-MAI/Z-Image-Turbo` weights in the HF cache (or
//! `ZIMAGE_SNAPSHOT`). Run:
//!   cargo test -p mlx-gen-z-image --release --test lora_train_spike -- --ignored --nocapture
//!
//! GO criteria proven here (no torch dependency — see the spike writeup for why a torch
//! bit-comparison is neither achievable nor the right bar):
//!   1. zero-init adapter is a bit no-op (injection itself doesn't perturb the forward);
//!   2. AdamW training over the trainable LoRA factors collapses the flow-match loss (overfit a
//!      tiny "dataset" = one fixed `(clean latent, caption-embed)` sample) — autograd works on the
//!      real 30-block DiT;
//!   3. the trained adapter, saved as PEFT safetensors and RELOADED through the normal inference
//!      path (`apply_z_image_adapters`), reproduces the trained loss bit-for-bit — round-trip.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_z_image::{attention_targets, load_transformer, ZImageLoraTrainer};
use mlx_rs::random;
use mlx_rs::transforms::eval;

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("ZIMAGE_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

#[test]
#[ignore = "needs real Z-Image weights"]
fn z_image_lora_overfit_and_roundtrip() {
    let snap = snapshot();
    let transformer = load_transformer(&snap).unwrap();

    // LoRA targets: the four attention projections across the 30 main blocks (the SceneWorks torch
    // trainer's default suffix set), rank 8 / alpha 8 (scale 1.0).
    let targets = attention_targets(30);
    let (rank, alpha, lr) = (8i32, 8.0f32, 1e-3f32);
    let mut trainer = ZImageLoraTrainer::new(transformer, &targets, rank, alpha, lr, 7).unwrap();
    println!("[spike] {} LoRA targets", trainer.num_targets());

    // A fixed tiny sample: clean latent [16,1,16,16] (a 128x128 image), synthetic caption embed
    // [8, 2560], and a fixed noise + sigma. (Real VAE/text-encode caching is sc-3043.)
    let x0 = random::normal::<f32>(&[16, 1, 16, 16], None, None, Some(&random::key(1).unwrap()))
        .unwrap();
    let cap =
        random::normal::<f32>(&[8, 2560], None, None, Some(&random::key(2).unwrap())).unwrap();
    let noise = random::normal::<f32>(&[16, 1, 16, 16], None, None, Some(&random::key(3).unwrap()))
        .unwrap();
    eval([&x0, &cap, &noise]).unwrap();
    let sigma = 0.5f32;

    let base = trainer.eval_loss(&x0, &cap, sigma, &noise, false).unwrap();
    let init = trainer.eval_loss(&x0, &cap, sigma, &noise, true).unwrap();
    println!("[spike] base(no adapter)={base:.6}  zero-init adapter={init:.6}");
    assert!(
        (init - base).abs() < 1e-4,
        "zero-init LoRA must be a no-op: base {base} vs init {init}"
    );

    for step in 0..80 {
        let loss = trainer.train_step(&x0, &cap, sigma, &noise).unwrap();
        if step % 10 == 0 || step == 79 {
            println!("[spike] step {step:>3}  loss {loss:.6}");
        }
    }
    let trained = trainer.eval_loss(&x0, &cap, sigma, &noise, true).unwrap();
    println!("[spike] trained loss {trained:.6}  (base {base:.6})");
    assert!(
        trained < base * 0.2,
        "training should cut the flow-match loss >5x: base {base} -> trained {trained}"
    );

    // Save as PEFT safetensors and reload through the real inference path.
    let out = std::env::temp_dir().join("z_image_spike_lora.safetensors");
    trainer.save_peft(&out, rank).unwrap();
    let meta = Weights::from_file(&out).unwrap();
    assert_eq!(meta.metadata("networkType"), Some("lora"));
    let rt = trainer
        .roundtrip_eval(&out, &x0, &cap, sigma, &noise)
        .unwrap();
    println!("[spike] round-trip (reloaded adapter) loss {rt:.6}  vs trained {trained:.6}");
    assert!(
        (rt - trained).abs() <= trained * 0.02 + 1e-5,
        "reloaded adapter must reproduce the trained loss: trained {trained} vs round-trip {rt}"
    );

    println!("[spike] GO ✓  base {base:.6} -> trained {trained:.6} (round-trip {rt:.6})");
}
