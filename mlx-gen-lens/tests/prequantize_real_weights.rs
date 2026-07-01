//! sc-8763 (Group-B): maintainer's on-device proof that a **pre-quantized packed** Lens tier built by
//! [`mlx_gen_lens::convert::prequantize_turnkey`] loads directly via the packed-detect loaders
//! ([`mlx_gen_lens::quant`] wired into the DiT loaders + the gpt-oss encoder MoE) and renders a
//! coherent image — no dense transient, no in-app `.quantize`/MXFP4-requant (epic 8506). This render
//! is the completeness gate for the loader packed-detect refactor: a missed quantized site loads codes
//! as dense floats → a degenerate (flat) render, which the pixel-range assertion catches.
//!
//! Lens packs **two** components (the DiT + the gpt-oss encoder MoE experts); the VAE stays dense
//! (the shared Flux.2 decoder, never quantized). Loading with `Quant::Q4`/`Q8` is honored but the
//! encoder/DiT quant no-ops on the already-packed weights — the property under test. The `bf16`
//! (dense) tier is the mirrored source (loaded directly, no `with_quant`).
//!
//! `#[ignore]`d — needs a real Lens-family snapshot (`microsoft/Lens-Turbo` by default). Run per tier:
//!   SC8763_SRC=<snap> SC8763_BITS=4 \
//!     cargo test -p mlx-gen-lens --release --test prequantize_real_weights -- --ignored --nocapture
//!
//! Env knobs: SC8763_SRC (source snapshot dir; default first HF-cache Lens-Turbo snapshot),
//! SC8763_OUT (tier output dir), SC8763_BITS (4 default / 8 / 0 = dense bf16 mirror), SC8763_MODEL
//! (registry id: `lens_turbo` default / `lens`), SC8763_KEEP (retain the built tier).

use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};
use std::path::PathBuf;

/// Resolve the source snapshot: `SC8763_SRC`, else the first cached Lens-Turbo snapshot.
fn lens_snapshot() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SC8763_SRC") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--microsoft--Lens-Turbo/snapshots");
    std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
}

fn model_id() -> String {
    std::env::var("SC8763_MODEL").unwrap_or_else(|_| "lens_turbo".into())
}

/// Build-only harness for producing **hostable** tiers (epic 8506 rollout): pack a tier from a Lens
/// snapshot into `SC8763_OUT` and keep it — no load/generate. `SC8763_BITS=0` builds the dense bf16
/// tier (a verbatim mirror of the source). Run per tier:
///   SC8763_SRC=<snap> SC8763_OUT=<staging/q4> SC8763_BITS=4 \
///     cargo test -p mlx-gen-lens --release --test prequantize_real_weights -- --ignored build_tier_only --nocapture
#[test]
#[ignore = "build-only tier producer for hosting; set SC8763_SRC/OUT/BITS"]
fn build_tier_only() {
    let src =
        PathBuf::from(std::env::var("SC8763_SRC").expect("SC8763_SRC (source snapshot) required"));
    let out =
        PathBuf::from(std::env::var("SC8763_OUT").expect("SC8763_OUT (tier output dir) required"));
    let bits: i32 = std::env::var("SC8763_BITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    if bits == 0 {
        panic!(
            "SC8763_BITS=0 (dense bf16) is a verbatim mirror of the source — copy the snapshot dir \
             directly (deref symlinks) rather than running the packer"
        );
    }
    println!(
        "building Q{bits} tier: {} -> {}",
        src.display(),
        out.display()
    );
    mlx_gen_lens::convert::prequantize_turnkey(&src, &out, bits)
        .expect("prequantize_turnkey succeeds");
    for (comp, stem) in [
        ("transformer", "diffusion_pytorch_model"),
        ("text_encoder", "model"),
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
#[ignore = "needs a real Lens snapshot; builds a packed tier + renders (set SC8763_SRC/OUT/BITS)"]
fn prequantize_turnkey_loads_packed_and_renders() {
    let Some(src) = lens_snapshot() else {
        eprintln!("skip: no Lens snapshot (set SC8763_SRC or populate the HF cache)");
        return;
    };
    let bits: i32 = std::env::var("SC8763_BITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let out = std::env::var("SC8763_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join(format!("lens-tier-q{bits}")));
    let id = model_id();

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
        mlx_gen_lens::convert::prequantize_turnkey(&src, &out, bits)
            .expect("prequantize_turnkey succeeds");
        for (comp, stem) in [
            ("transformer", "diffusion_pytorch_model"),
            ("text_encoder", "model"),
        ] {
            let f = out.join(comp).join(format!("{stem}.safetensors"));
            assert!(f.is_file(), "missing packed {comp}/{stem}.safetensors");
        }
        let q = if bits >= 8 { Quant::Q8 } else { Quant::Q4 };
        (out.clone(), Some(q))
    };

    // Load DIRECTLY from the tier dir. For a packed tier the loaders packed-detect (no dense transient,
    // no MXFP4 requant) and the honored `with_quant` no-ops on the already-quantized bases.
    let mut spec = LoadSpec::new(WeightsSource::Dir(load_root));
    if let Some(q) = quant {
        spec = spec.with_quant(q);
    }
    let generator = mlx_gen::load(&id, &spec).expect("packed lens loads");

    // 512² / few-step — packed load-path proof, not a quality bench.
    let req = GenerationRequest {
        prompt: "a red fox sitting in a snowy forest, photorealistic, sharp focus".into(),
        negative_prompt: Some("blurry, lowres, deformed".into()),
        width: 512,
        height: 512,
        seed: Some(42),
        steps: Some(if id == "lens_turbo" { 4 } else { 12 }),
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
    // Sample standard deviation of the pixels (the non-degenerate gate — a missed packed site flattens
    // the render).
    let var = img
        .pixels
        .iter()
        .map(|&p| {
            let d = p as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / img.pixels.len() as f64;
    let std = var.sqrt();
    let tier = if bits == 0 {
        "bf16".into()
    } else {
        format!("Q{bits}")
    };
    println!("✓ packed {tier} {id}: 512x512; px min={min} max={max} mean={mean:.1} std={std:.1}");
    assert!(
        std > 20.0,
        "degenerate render: pixel std {std:.1} too flat (a missed packed site loads codes as dense \
         floats)"
    );
    assert!(
        max as i32 - min as i32 > 32,
        "degenerate render: pixel range {min}..={max} too flat"
    );

    if bits != 0 && std::env::var("SC8763_KEEP").is_err() {
        let _ = std::fs::remove_dir_all(&out);
        println!("  removed {} (set SC8763_KEEP to retain)", out.display());
    }
}
