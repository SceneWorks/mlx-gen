//! sc-8669 (Group-B): maintainer's on-device proof that a **pre-quantized packed** FLUX.1 tier built
//! by [`mlx_gen_flux::convert::prequantize_turnkey`] loads directly via the packed-detect loaders
//! ([`mlx_gen_flux::quant`] + the local `TokenEmbedding` packed-detect + the shared Z-Image VAE) and
//! renders a coherent image — no dense bf16 transient, no in-app `.quantize` (epic 8506).
//!
//! FLUX.1 packs all four components: the DiT transformer, the CLIP (`text_encoder/`) + T5
//! (`text_encoder_2/`) encoders, and the VAE. Loading with `Quant::Q4` is honored but no-ops on the
//! already-packed weights — the property under test.
//!
//! `#[ignore]`d — needs a real `black-forest-labs/FLUX.1-dev` snapshot. Run with:
//!   cargo test -p mlx-gen-flux --release --test prequantize_real_weights -- --ignored --nocapture
//!
//! Env knobs: SC8670_SRC (source snapshot dir; default first HF-cache FLUX.1-dev snapshot),
//! SC8670_OUT (tier output dir), SC8670_BITS (4 default / 8), SC8670_KEEP (retain the built tier).

use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};
use mlx_gen_flux as flux;
use std::path::PathBuf;

fn flux_dev_snapshot() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SC8670_SRC") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots");
    std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
}

/// Build-only harness for producing **hostable** tiers (epic 8506 rollout): pack a tier from a FLUX.1
/// snapshot into `SC8670_OUT` and keep it — no load/generate. Run per tier:
///   SC8670_SRC=<snap> SC8670_OUT=<staging/q4> SC8670_BITS=4 \
///     cargo test -p mlx-gen-flux --release --test prequantize_real_weights -- --ignored build_tier_only --nocapture
#[test]
#[ignore = "build-only tier producer for hosting; set SC8670_SRC/OUT/BITS"]
fn build_tier_only() {
    let src =
        PathBuf::from(std::env::var("SC8670_SRC").expect("SC8670_SRC (source snapshot) required"));
    let out =
        PathBuf::from(std::env::var("SC8670_OUT").expect("SC8670_OUT (tier output dir) required"));
    let bits: i32 = std::env::var("SC8670_BITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    println!(
        "building Q{bits} tier: {} -> {}",
        src.display(),
        out.display()
    );
    flux::convert::prequantize_turnkey(&src, &out, bits).expect("prequantize_turnkey succeeds");
    for comp in ["transformer", "text_encoder", "text_encoder_2", "vae"] {
        let f = out.join(comp).join("model.safetensors");
        let sz = std::fs::metadata(&f)
            .unwrap_or_else(|_| panic!("missing {comp}/model.safetensors"))
            .len();
        println!("  {comp}/model.safetensors = {:.2} GB", sz as f64 / 1e9);
    }
    println!("✓ built {}", out.display());
}

#[test]
#[ignore = "needs a real FLUX.1-dev snapshot; builds a packed tier (set SC8670_SRC/OUT/BITS)"]
fn prequantize_turnkey_loads_packed_and_renders() {
    let Some(src) = flux_dev_snapshot() else {
        eprintln!("skip: no FLUX.1-dev snapshot (set SC8670_SRC or populate the HF cache)");
        return;
    };
    let bits: i32 = std::env::var("SC8670_BITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let quant = if bits >= 8 { Quant::Q8 } else { Quant::Q4 };
    let out = std::env::var("SC8670_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join(format!("flux1-dev-q{bits}")));

    println!(
        "building Q{bits} turnkey: {} -> {}",
        src.display(),
        out.display()
    );
    flux::convert::prequantize_turnkey(&src, &out, bits).expect("prequantize_turnkey succeeds");
    for comp in ["transformer", "text_encoder", "text_encoder_2", "vae"] {
        let f = out.join(comp).join("model.safetensors");
        assert!(f.is_file(), "missing packed {comp}/model.safetensors");
    }

    // Load DIRECTLY from the packed dir (packed-detect). `with_quant` is honored but the packed
    // weights are already quantized, so the post-load `.quantize` calls no-op.
    let spec = LoadSpec::new(WeightsSource::Dir(out.clone())).with_quant(quant);
    let generator = mlx_gen::load("flux1_dev", &spec).expect("packed flux1_dev loads");

    // Small/short — packed load-path proof, not a quality bench (the 12B DiT is slow). FLUX.1-dev is
    // guidance-distilled (embedded guidance, single forward/step).
    let req = GenerationRequest {
        prompt: "a red fox sitting in a snowy forest, photorealistic, sharp focus".into(),
        width: 512,
        height: 512,
        seed: Some(42),
        steps: Some(8),
        guidance: Some(3.5),
        ..Default::default()
    };
    let img = match generator
        .generate(&req, &mut |_| {})
        .expect("packed generate succeeds")
    {
        GenerationOutput::Images(mut v) => {
            assert_eq!(v.len(), 1);
            v.pop().unwrap()
        }
        other => panic!("expected Images, got {other:?}"),
    };
    assert_eq!((img.width, img.height), (512, 512), "image size");

    let min = *img.pixels.iter().min().unwrap();
    let max = *img.pixels.iter().max().unwrap();
    let mean = img.pixels.iter().map(|&p| p as u64).sum::<u64>() as f64 / img.pixels.len() as f64;
    println!("✓ packed Q{bits} flux1_dev: 512x512; px min={min} max={max} mean={mean:.1}");
    assert!(
        max as i32 - min as i32 > 32,
        "degenerate render: pixel range {min}..={max} too flat"
    );

    if std::env::var("SC8670_KEEP").is_err() {
        let _ = std::fs::remove_dir_all(&out);
        println!("  removed {} (set SC8670_KEEP to retain)", out.display());
    }
}
