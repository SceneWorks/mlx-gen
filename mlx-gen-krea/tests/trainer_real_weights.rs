//! sc-7577 — real-weight end-to-end smoke for the Krea 2 Raw LoRA trainer (epic 7565 P3). Weight-gated
//! (`#[ignore]`): drives the PUBLIC [`Trainer`] surface (`load_trainer` → `Trainer::train`) on the real
//! `krea/Krea-2-Raw` snapshot, a short run that must produce a **loadable PEFT adapter** — the story AC.
//!
//!   cargo test -p mlx-gen-krea --release --test trainer_real_weights -- --ignored --nocapture
//!
//! Set `KREA_RAW_DIR` to override the snapshot location (else the newest HF-cache `krea/Krea-2-Raw`).

use std::path::PathBuf;

use mlx_gen::{
    CancelFlag, LoadSpec, NetworkType, TrainingConfig, TrainingItem, TrainingProgress,
    TrainingRequest, WeightsSource,
};
use mlx_gen_krea::load_trainer;

/// Resolve the `krea/Krea-2-Raw` snapshot (the `KREA_RAW_DIR` override, else the newest HF-cache
/// snapshot with a `transformer/` tree).
fn raw_snapshot() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("KREA_RAW_DIR") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--krea--Krea-2-Raw/snapshots");
    std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir() && p.join("transformer").is_dir())
}

/// Write a synthetic 320×256 RGB PNG (a deterministic gradient — content is irrelevant; the trainer
/// center-crops + resizes it) so the smoke needs no external dataset fixture.
fn write_synth_image(path: &std::path::Path) {
    let mut img = image::RgbImage::new(320, 256);
    for (x, y, px) in img.enumerate_pixels_mut() {
        *px = image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8]);
    }
    img.save(path).expect("write synth png");
}

/// Parse a safetensors file's JSON header (the 8-byte LE length prefix + UTF-8 JSON) — enough to assert
/// the adapter is the expected PEFT shape without a safetensors-parser dependency.
fn safetensors_header(path: &std::path::Path) -> String {
    let bytes = std::fs::read(path).expect("read adapter");
    assert!(
        bytes.len() > 8,
        "adapter file too small: {} bytes",
        bytes.len()
    );
    let hlen = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    String::from_utf8(bytes[8..8 + hlen].to_vec()).expect("utf8 header")
}

/// Run a short LoRA training run on the real Raw weights and confirm it writes a loadable PEFT adapter.
#[test]
#[ignore = "needs real krea/Krea-2-Raw weights (~25 GB) + a Mac; run as its own process"]
fn short_train_produces_loadable_adapter() {
    let root = raw_snapshot().expect("krea/Krea-2-Raw snapshot (HF cache or KREA_RAW_DIR)");
    let tmp = std::env::temp_dir().join("krea_trainer_smoke");
    std::fs::create_dir_all(&tmp).unwrap();
    let img_path = tmp.join("swatch.png");
    write_synth_image(&img_path);

    let mut trainer = load_trainer(&LoadSpec::new(WeightsSource::Dir(root))).expect("load_trainer");

    let req = TrainingRequest {
        items: vec![TrainingItem {
            image_path: img_path,
            caption: "a vivid abstract color swatch".into(),
        }],
        config: TrainingConfig {
            rank: 4,
            alpha: 4.0,
            steps: 3,
            resolution: 256,
            save_every: 0,
            learning_rate: 1e-4,
            network_type: NetworkType::Lora,
            ..Default::default()
        },
        output_dir: tmp.clone(),
        file_name: "krea_smoke_lora.safetensors".into(),
        trigger_words: vec!["swatch".into()],
        cancel: CancelFlag::new(),
    };

    let mut saw_training = false;
    let mut last_step = 0u32;
    let out = trainer
        .train(&req, &mut |p| {
            if let TrainingProgress::Training { step, loss, .. } = p {
                saw_training = true;
                last_step = step;
                eprintln!("[sc-7577] step {step} loss {loss:.5}");
            }
        })
        .expect("train");

    assert_eq!(out.steps, 3, "all 3 micro-steps ran");
    assert!(
        saw_training && last_step == 3,
        "Training progress streamed to step 3"
    );
    assert!(
        out.final_loss.is_finite(),
        "final loss is finite: {}",
        out.final_loss
    );

    // The AC: a loadable PEFT adapter. Confirm the safetensors header carries the PEFT factor keys +
    // the reload-contract metadata (networkType/rank/alpha) the sc-7578 apply path reads.
    let header = safetensors_header(&out.adapter_path);
    assert!(
        header.contains(".lora_A.weight"),
        "PEFT lora_A keys: {header:.200}"
    );
    assert!(header.contains(".lora_B.weight"), "PEFT lora_B keys");
    assert!(
        header.contains("transformer_blocks."),
        "single-stream block targets"
    );
    assert!(
        header.contains("networkType") && header.contains("lora"),
        "reload metadata"
    );
    eprintln!(
        "[sc-7577] ✅ adapter {} ({} bytes)",
        out.adapter_path.display(),
        std::fs::metadata(&out.adapter_path).unwrap().len()
    );
}

/// The same short run with **gradient checkpointing** on — exercises the
/// `forward_with_blocks_checkpointed` path end-to-end through the public trainer (the OOM-mitigation
/// toggle), confirming it trains + writes a loadable adapter too.
#[test]
#[ignore = "needs real krea/Krea-2-Raw weights (~25 GB) + a Mac; run as its own process"]
fn short_train_checkpointed_produces_loadable_adapter() {
    let root = raw_snapshot().expect("krea/Krea-2-Raw snapshot (HF cache or KREA_RAW_DIR)");
    let tmp = std::env::temp_dir().join("krea_trainer_smoke_ckpt");
    std::fs::create_dir_all(&tmp).unwrap();
    let img_path = tmp.join("swatch.png");
    write_synth_image(&img_path);

    let mut trainer = load_trainer(&LoadSpec::new(WeightsSource::Dir(root))).expect("load_trainer");
    let req = TrainingRequest {
        items: vec![TrainingItem {
            image_path: img_path,
            caption: "a vivid abstract color swatch".into(),
        }],
        config: TrainingConfig {
            rank: 4,
            alpha: 4.0,
            steps: 2,
            resolution: 256,
            save_every: 0,
            gradient_checkpointing: true,
            network_type: NetworkType::Lora,
            ..Default::default()
        },
        output_dir: tmp.clone(),
        file_name: "krea_smoke_ckpt_lora.safetensors".into(),
        trigger_words: vec![],
        cancel: CancelFlag::new(),
    };

    let out = trainer
        .train(&req, &mut |_| {})
        .expect("train (checkpointed)");
    assert_eq!(out.steps, 2);
    let header = safetensors_header(&out.adapter_path);
    assert!(header.contains(".lora_A.weight") && header.contains(".lora_B.weight"));
    eprintln!(
        "[sc-7577] ✅ checkpointed adapter {}",
        out.adapter_path.display()
    );
}
