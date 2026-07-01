//! sc-8746 (Group-B): maintainer's on-device proof that a **pre-quantized packed** SDXL tier built by
//! [`mlx_gen_sdxl::convert::prequantize_turnkey`] loads directly via the packed-detect loaders
//! ([`mlx_gen_sdxl::quant`] + the packed-aware `loader`) and renders a coherent image — no dense
//! fp16 transient, no in-app `.quantize` (epic 8506). This render is the completeness gate for the
//! loader packed-detect refactor: a missed quantized site loads u32 codes as dense floats → a
//! degenerate (flat) render, which the pixel-range assertion catches.
//!
//! SDXL packs **three** components (U-Net + both CLIP text encoders); the VAE stays dense (never
//! quantized). Loading with `Quant::Q4`/`Q8` is honored but no-ops on the already-packed weights —
//! the property under test. The `bf16` (dense) tier is the mirrored source (no `with_quant`).
//!
//! `#[ignore]`d — needs a real SDXL-family snapshot (RealVisXL_V5.0 by default). Run per tier:
//!   SC8746_SRC=<snap> SC8746_BITS=4 \
//!     cargo test -p mlx-gen-sdxl --release --test prequantize_real_weights -- --ignored --nocapture
//!
//! Env knobs: SC8746_SRC (source snapshot dir; default first HF-cache RealVisXL_V5.0 snapshot),
//! SC8746_OUT (tier output dir), SC8746_BITS (4 default / 8 / 0 = dense bf16 mirror), SC8746_KEEP
//! (retain the built tier).

use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};
use mlx_gen_sdxl as sdxl;
use std::path::PathBuf;

/// Resolve the source snapshot: `SC8746_SRC`, else the first cached RealVisXL_V5.0 snapshot.
fn sdxl_snapshot() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SC8746_SRC") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--SG161222--RealVisXL_V5.0/snapshots");
    std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
}

/// Build-only harness for producing **hostable** tiers (epic 8506 rollout): pack a tier from an SDXL
/// snapshot into `SC8746_OUT` and keep it — no load/generate. `SC8746_BITS=0` builds the dense bf16
/// tier (a verbatim mirror of the source). Run per tier:
///   SC8746_SRC=<snap> SC8746_OUT=<staging/q4> SC8746_BITS=4 \
///     cargo test -p mlx-gen-sdxl --release --test prequantize_real_weights -- --ignored build_tier_only --nocapture
#[test]
#[ignore = "build-only tier producer for hosting; set SC8746_SRC/OUT/BITS"]
fn build_tier_only() {
    let src =
        PathBuf::from(std::env::var("SC8746_SRC").expect("SC8746_SRC (source snapshot) required"));
    let out =
        PathBuf::from(std::env::var("SC8746_OUT").expect("SC8746_OUT (tier output dir) required"));
    let bits: i32 = std::env::var("SC8746_BITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    if bits == 0 {
        panic!(
            "SC8746_BITS=0 (dense bf16) is a verbatim mirror of the source — copy the snapshot dir \
             directly (deref symlinks) rather than running the packer"
        );
    }
    println!(
        "building Q{bits} tier: {} -> {}",
        src.display(),
        out.display()
    );
    sdxl::convert::prequantize_turnkey(&src, &out, bits).expect("prequantize_turnkey succeeds");
    for (comp, stem) in [
        ("unet", "diffusion_pytorch_model"),
        ("text_encoder", "model"),
        ("text_encoder_2", "model"),
    ] {
        let f = out.join(comp).join(format!("{stem}.safetensors"));
        let sz = std::fs::metadata(&f)
            .unwrap_or_else(|_| panic!("missing {comp}/{stem}.safetensors"))
            .len();
        println!("  {comp}/{stem}.safetensors = {:.3} GB", sz as f64 / 1e9);
    }
    println!("✓ built {}", out.display());
}

#[test]
#[ignore = "needs a real SDXL snapshot; builds a packed tier + renders (set SC8746_SRC/OUT/BITS)"]
fn prequantize_turnkey_loads_packed_and_renders() {
    let Some(src) = sdxl_snapshot() else {
        eprintln!("skip: no SDXL snapshot (set SC8746_SRC or populate the HF cache)");
        return;
    };
    let bits: i32 = std::env::var("SC8746_BITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let out = std::env::var("SC8746_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join(format!("sdxl-tier-q{bits}")));

    // Build the packed tier (Q4/Q8). For the dense bf16 tier the source snapshot IS the tier, so we
    // load `src` directly with no `with_quant`.
    let (load_root, quant): (PathBuf, Option<Quant>) = if bits == 0 {
        println!(
            "dense (bf16) tier: loading source snapshot directly {}",
            src.display()
        );
        (src.clone(), None)
    } else {
        println!(
            "building Q{bits} turnkey: {} -> {}",
            src.display(),
            out.display()
        );
        sdxl::convert::prequantize_turnkey(&src, &out, bits).expect("prequantize_turnkey succeeds");
        for (comp, stem) in [
            ("unet", "diffusion_pytorch_model"),
            ("text_encoder", "model"),
            ("text_encoder_2", "model"),
        ] {
            let f = out.join(comp).join(format!("{stem}.safetensors"));
            assert!(f.is_file(), "missing packed {comp}/{stem}.safetensors");
        }
        let q = if bits >= 8 { Quant::Q8 } else { Quant::Q4 };
        (out.clone(), Some(q))
    };

    // Load DIRECTLY from the tier dir. For a packed tier the loader packed-detects (no cast_all) and
    // the honored `with_quant` no-ops on the already-quantized bases.
    let mut spec = LoadSpec::new(WeightsSource::Dir(load_root));
    if let Some(q) = quant {
        spec = spec.with_quant(q);
    }
    let generator = mlx_gen::load("sdxl", &spec).expect("packed sdxl loads");

    // 768² / few-step — packed load-path proof, not a quality bench. RealVisXL renders well at
    // guidance 5–7, 20+ steps; keep it short for CI-adjacent runtime.
    let req = GenerationRequest {
        prompt: "a red fox sitting in a snowy forest, photorealistic, sharp focus".into(),
        negative_prompt: Some("blurry, lowres, deformed".into()),
        width: 768,
        height: 768,
        seed: Some(42),
        steps: Some(20),
        guidance: Some(6.0),
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
    assert_eq!((img.width, img.height), (768, 768), "image size");

    let min = *img.pixels.iter().min().unwrap();
    let max = *img.pixels.iter().max().unwrap();
    let mean = img.pixels.iter().map(|&p| p as u64).sum::<u64>() as f64 / img.pixels.len() as f64;
    let tier = if bits == 0 {
        "bf16".into()
    } else {
        format!("Q{bits}")
    };
    println!("✓ packed {tier} sdxl: 768x768; px min={min} max={max} mean={mean:.1}");
    assert!(
        max as i32 - min as i32 > 32,
        "degenerate render: pixel range {min}..={max} too flat (a missed packed site loads codes as \
         dense floats)"
    );
    assert!(
        img.pixels.iter().all(|&p| p != 0 || max > 0),
        "all-zero image"
    );

    if bits != 0 && std::env::var("SC8746_KEEP").is_err() {
        let _ = std::fs::remove_dir_all(&out);
        println!("  removed {} (set SC8746_KEEP to retain)", out.display());
    }
}
