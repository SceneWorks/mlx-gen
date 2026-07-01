//! sc-8771 (Group-B): maintainer's on-device proof that a **pre-quantized packed** SenseNova-U1 tier
//! built by [`mlx_gen_sensenova::convert::prequantize_turnkey`] loads directly via the packed-detect
//! loader ([`mlx_gen_sensenova::quant`] wired into [`mlx_gen_sensenova::qwen3`]'s backbone Linears)
//! and renders a coherent T2I image — no dense transient, no in-app `.quantize` (epic 8506). This
//! render is the completeness gate for the loader packed-detect refactor: a missed quantized site
//! loads u32 codes as dense floats → a degenerate (flat) render, which the pixel-range assertion
//! catches.
//!
//! SenseNova-U1 is a **unified** MoT model with no separate VAE or text encoder — the converter packs
//! the **backbone decoder-stack Linears** (attention + SwiGLU, both understanding & generation paths)
//! into one flat `model.safetensors`; the token embedding, `lm_head`, norms, vision convs, and the
//! flow-matching head stay dense. A packed tier is loaded with `Quant::None` (the loader packed-detects
//! via `{base}.scales`, so no in-app re-quantize is needed). The `bf16` (dense) tier is the mirrored
//! source, loaded directly.
//!
//! `#[ignore]`d — needs the real ~33 GB SenseNova-U1-8B-MoT snapshot. Run per tier:
//!   SC8771_SRC=<snap> SC8771_BITS=4 \
//!     cargo test -p mlx-gen-sensenova --release --test prequantize_real_weights -- --ignored --nocapture
//!
//! For `SC8771_MODEL=sensenova_u1_8b_fast` (sc-8775) the tier is built by
//! [`mlx_gen_sensenova::convert::prequantize_fast_turnkey`], which pre-merges the 8-step distill LoRA
//! into the generation path before packing and drops the `distill_merged.json` marker so
//! [`mlx_gen_sensenova::model::load_fast`] skips the load-time merge. Unlike the base bf16 tier (a
//! verbatim source mirror), the **fast** bf16 tier (`SC8771_BITS=0`) is a distinct MERGED checkpoint
//! and is built here too. The distill LoRA is resolved from `$SENSENOVA_DISTILL_LORA` / co-located /
//! the HF cache (`sensenova/SenseNova-U1-8B-MoT-LoRAs`).
//!
//! Env knobs: SC8771_SRC (source snapshot dir; default the cached SenseNova-U1-8B-MoT snapshot),
//! SC8771_OUT (tier output dir), SC8771_BITS (4 default / 8 / 0 = bf16 — mirror for base, merged build
//! for _fast), SC8771_MODEL (registry id: `sensenova_u1_8b` default / `sensenova_u1_8b_fast`),
//! SC8771_KEEP (retain the tier).

use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};
use mlx_gen_sensenova as _; // force-link the inventory registration for mlx_gen::load.
use std::path::PathBuf;

const DEFAULT_SNAPSHOT: &str = concat!(
    env!("HOME"),
    "/.cache/huggingface/hub/models--sensenova--SenseNova-U1-8B-MoT/snapshots/\
     bfa9b436503cb8aed4f2bc60e3236710cc77468d"
);

/// Resolve the source snapshot: `SC8771_SRC`, else the cached SenseNova-U1-8B-MoT snapshot.
fn sensenova_snapshot() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SC8771_SRC") {
        return Some(PathBuf::from(p));
    }
    let p = PathBuf::from(DEFAULT_SNAPSHOT);
    p.is_dir().then_some(p)
}

fn model_id() -> String {
    std::env::var("SC8771_MODEL").unwrap_or_else(|_| "sensenova_u1_8b".into())
}

fn bits_env() -> i32 {
    std::env::var("SC8771_BITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4)
}

/// Build-only harness for producing **hostable** tiers (epic 8506 rollout): pack a tier from a
/// SenseNova snapshot into `SC8771_OUT` and keep it — no load/generate. `SC8771_BITS=0` (dense bf16)
/// is a verbatim mirror of the source; copy the shards + assets directly rather than running the
/// packer. Run per tier:
///   SC8771_SRC=<snap> SC8771_OUT=<staging/q4> SC8771_BITS=4 \
///     cargo test -p mlx-gen-sensenova --release --test prequantize_real_weights -- --ignored build_tier_only --nocapture
#[test]
#[ignore = "build-only tier producer for hosting; set SC8771_SRC/OUT/BITS"]
fn build_tier_only() {
    let src =
        PathBuf::from(std::env::var("SC8771_SRC").expect("SC8771_SRC (source snapshot) required"));
    let out =
        PathBuf::from(std::env::var("SC8771_OUT").expect("SC8771_OUT (tier output dir) required"));
    let bits = bits_env();
    // The `_fast` tier (sc-8775) pre-merges the distill LoRA, so ALL of its tiers — including bf16 —
    // are distinct built checkpoints (bf16 = a MERGED dense checkpoint, not a source mirror). The base
    // bf16 tier, by contrast, is a verbatim source mirror (copy the shards directly, don't pack).
    let is_fast = model_id().ends_with("_fast");
    if bits == 0 && !is_fast {
        panic!(
            "SC8771_BITS=0 (dense bf16) is a verbatim mirror of the source — copy the snapshot dir \
             directly (deref symlinks) rather than running the packer (base tier). For the _fast \
             tier bf16 IS a distinct merged checkpoint; set SC8771_MODEL=sensenova_u1_8b_fast."
        );
    }
    let tier = if bits == 0 {
        "bf16 (merged)".to_string()
    } else {
        format!("Q{bits}")
    };
    println!(
        "building {tier} tier ({}): {} -> {}",
        model_id(),
        src.display(),
        out.display()
    );
    if is_fast {
        mlx_gen_sensenova::convert::prequantize_fast_turnkey(&src, &out, bits)
            .expect("prequantize_fast_turnkey succeeds");
    } else {
        mlx_gen_sensenova::convert::prequantize_turnkey(&src, &out, bits)
            .expect("prequantize_turnkey succeeds");
    }
    let f = out.join("model.safetensors");
    let sz = std::fs::metadata(&f)
        .expect("missing packed model.safetensors")
        .len();
    println!("  model.safetensors = {:.3} GB", sz as f64 / 1e9);
    for asset in ["config.json", "tokenizer.json"] {
        assert!(out.join(asset).is_file(), "missing {asset} in turnkey");
    }
    println!("✓ built {}", out.display());
}

#[test]
#[ignore = "needs the ~33GB SenseNova snapshot; builds a packed tier + renders (set SC8771_SRC/BITS)"]
fn prequantize_turnkey_loads_packed_and_renders() {
    let Some(src) = sensenova_snapshot() else {
        eprintln!("skip: no SenseNova snapshot (set SC8771_SRC or populate the HF cache)");
        return;
    };
    let bits = bits_env();
    let out = std::env::var("SC8771_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join(format!("sensenova-tier-q{bits}")));
    let id = model_id();
    let is_fast = id.ends_with("_fast");

    // Build the tier. The base bf16 tier is the source snapshot itself (loaded directly). Every other
    // tier is built into `out`: the base Q4/Q8 via `prequantize_turnkey`; ALL `_fast` tiers (q4/q8 AND
    // the MERGED bf16, sc-8775) via `prequantize_fast_turnkey`, which pre-merges the distill LoRA +
    // drops the marker so `load_fast` skips the load-time merge.
    let (load_root, built): (PathBuf, bool) = if bits == 0 && !is_fast {
        println!(
            "dense (bf16) base tier: loading source snapshot directly {}",
            src.display()
        );
        (src.clone(), false)
    } else {
        let kind = if bits == 0 {
            "merged bf16".to_string()
        } else {
            format!("Q{bits}")
        };
        println!(
            "building {kind} {id} turnkey: {} -> {}",
            src.display(),
            out.display()
        );
        if is_fast {
            mlx_gen_sensenova::convert::prequantize_fast_turnkey(&src, &out, bits)
                .expect("prequantize_fast_turnkey succeeds");
        } else {
            mlx_gen_sensenova::convert::prequantize_turnkey(&src, &out, bits)
                .expect("prequantize_turnkey succeeds");
        }
        assert!(
            out.join("model.safetensors").is_file(),
            "missing built model.safetensors"
        );
        (out.clone(), true)
    };

    // Load DIRECTLY from the tier dir. A packed tier packed-detects via `{base}.scales` (no dense
    // transient, no in-app re-quantize), so we load with `Quant::None`; the dense bf16 tier loads
    // dense the same way.
    let spec = LoadSpec::new(WeightsSource::Dir(load_root));
    let generator = mlx_gen::load(&id, &spec).expect("packed sensenova loads");

    // 256² / few-step — packed load-path proof, not a quality bench (an 8B multi-minute run).
    let req = GenerationRequest {
        prompt: "a red apple on a wooden table, studio lighting".into(),
        width: 256,
        height: 256,
        count: 1,
        seed: Some(42),
        steps: Some(8),
        guidance: Some(2.0),
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
    assert_eq!((img.width, img.height), (256, 256), "image size");

    let min = *img.pixels.iter().min().unwrap();
    let max = *img.pixels.iter().max().unwrap();
    let mean = img.pixels.iter().map(|&p| p as u64).sum::<u64>() as f64 / img.pixels.len() as f64;
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
    println!("✓ packed {tier} {id}: 256x256; px min={min} max={max} mean={mean:.1} std={std:.1}");
    assert!(
        std > 20.0,
        "degenerate render: pixel std {std:.1} too flat (a missed packed site loads codes as dense \
         floats)"
    );
    assert!(
        max as i32 - min as i32 > 32,
        "degenerate render: pixel range {min}..={max} too flat"
    );

    if built && std::env::var("SC8771_KEEP").is_err() {
        let _ = std::fs::remove_dir_all(&out);
        println!("  removed {} (set SC8771_KEEP to retain)", out.display());
    }
}
