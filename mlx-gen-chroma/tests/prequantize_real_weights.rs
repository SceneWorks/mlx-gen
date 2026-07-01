//! sc-8777 (Group-B): maintainer's on-device proof that a **pre-quantized packed** Chroma tier built
//! by [`mlx_gen_chroma::convert::prequantize_turnkey`] loads directly via the packed-detect loader
//! ([`mlx_gen_chroma::quant`] wired into [`mlx_gen_chroma::transformer::Lin::load`]) and renders a
//! coherent T2I image — no dense transient, no in-app `.quantize` (epic 8506). This render is the
//! completeness gate for the loader packed-detect refactor: a missed quantized site loads u32 codes
//! as dense floats → a degenerate (flat) render, which the pixel-std assertion catches.
//!
//! Chroma is a FLUX.1-schnell-derived DiT with a shared T5-XXL text encoder and FLUX.1 VAE. The
//! converter packs only the **DiT `transformer/` block Linears** (double blocks' attention + FFN,
//! single blocks' attention + `proj_mlp`/`proj_out`) into one flat
//! `transformer/diffusion_pytorch_model.safetensors`; the transformer embedders + Approximator, the
//! T5 encoder, and the VAE stay dense (mirrored). A packed tier is loaded with `Quant::None` (the
//! loader packed-detects via `{base}.scales`, so no in-app re-quantize is needed). The `bf16` (dense)
//! tier is the mirrored source, loaded directly.
//!
//! `#[ignore]`d — needs a real ~18GB Chroma diffusers snapshot. Run per tier:
//!   SC8777_SRC=<snap> SC8777_BITS=4 SC8777_MODEL=chroma1_base \
//!     cargo test -p mlx-gen-chroma --release --test prequantize_real_weights -- --ignored --nocapture
//!
//! Env knobs: SC8777_SRC (source snapshot dir; default the cached Chroma1-Base snapshot),
//! SC8777_OUT (tier output dir), SC8777_BITS (4 default / 8 / 0 = dense bf16 mirror), SC8777_MODEL
//! (registry id: `chroma1_base` default / `chroma1_hd` / `chroma1_flash`), SC8777_KEEP (retain the
//! tier).

use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};
use mlx_gen_chroma as _; // force-link the inventory registration for mlx_gen::load.
use std::path::PathBuf;

/// Resolve the cached HF snapshot dir for a `lodestones/<repo>` model, or `None` if absent.
fn cached_snapshot(repo: &str) -> Option<PathBuf> {
    let cache = std::env::var("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/huggingface/hub")
        });
    let snaps = cache.join(format!("models--lodestones--{repo}/snapshots"));
    std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir() && p.join("transformer").is_dir())
}

fn model_id() -> String {
    std::env::var("SC8777_MODEL").unwrap_or_else(|_| "chroma1_base".into())
}

/// The source HF repo for a registry id (the three variants ship distinct checkpoints).
fn repo_for(id: &str) -> &'static str {
    match id {
        "chroma1_hd" => "Chroma1-HD",
        "chroma1_flash" => "Chroma1-Flash",
        _ => "Chroma1-Base",
    }
}

/// Resolve the source snapshot: `SC8777_SRC`, else the cached snapshot for the selected model.
fn chroma_snapshot() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SC8777_SRC") {
        return Some(PathBuf::from(p));
    }
    cached_snapshot(repo_for(&model_id()))
}

fn bits_env() -> i32 {
    std::env::var("SC8777_BITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4)
}

/// Build-only harness for producing **hostable** tiers (epic 8506 rollout): pack a tier from a Chroma
/// snapshot into `SC8777_OUT` and keep it — no load/generate. `SC8777_BITS=0` (dense bf16) is a
/// verbatim mirror of the source; copy the snapshot dir directly rather than running the packer. Run:
///   SC8777_SRC=<snap> SC8777_OUT=<staging/q4> SC8777_BITS=4 \
///     cargo test -p mlx-gen-chroma --release --test prequantize_real_weights -- --ignored build_tier_only --nocapture
#[test]
#[ignore = "build-only tier producer for hosting; set SC8777_SRC/OUT/BITS"]
fn build_tier_only() {
    let src =
        PathBuf::from(std::env::var("SC8777_SRC").expect("SC8777_SRC (source snapshot) required"));
    let out =
        PathBuf::from(std::env::var("SC8777_OUT").expect("SC8777_OUT (tier output dir) required"));
    let bits = bits_env();
    if bits == 0 {
        panic!(
            "SC8777_BITS=0 (dense bf16) is a verbatim mirror of the source — copy the snapshot dir \
             directly (deref symlinks) rather than running the packer"
        );
    }
    println!(
        "building Q{bits} tier: {} -> {}",
        src.display(),
        out.display()
    );
    mlx_gen_chroma::convert::prequantize_turnkey(&src, &out, bits)
        .expect("prequantize_turnkey succeeds");
    let f = out
        .join("transformer")
        .join("diffusion_pytorch_model.safetensors");
    let sz = std::fs::metadata(&f)
        .expect("missing packed transformer safetensors")
        .len();
    println!(
        "  transformer/diffusion_pytorch_model.safetensors = {:.3} GB",
        sz as f64 / 1e9
    );
    for asset in ["model_index.json", "transformer/config.json"] {
        assert!(out.join(asset).is_file(), "missing {asset} in turnkey");
    }
    assert!(out.join("vae").is_dir(), "missing vae/ in turnkey");
    assert!(
        out.join("text_encoder").is_dir(),
        "missing text_encoder/ in turnkey"
    );
    println!("✓ built {}", out.display());
}

#[test]
#[ignore = "needs a ~18GB Chroma snapshot; builds a packed tier + renders (set SC8777_SRC/BITS/MODEL)"]
fn prequantize_turnkey_loads_packed_and_renders() {
    let Some(src) = chroma_snapshot() else {
        eprintln!("skip: no Chroma snapshot (set SC8777_SRC or populate the HF cache)");
        return;
    };
    let bits = bits_env();
    let out = std::env::var("SC8777_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join(format!("chroma-tier-q{bits}")));
    let id = model_id();

    // Build the packed tier (Q4/Q8). For the dense bf16 tier the source snapshot IS the tier, so we
    // load `src` directly.
    let load_root: PathBuf = if bits == 0 {
        println!(
            "dense (bf16) tier: loading source snapshot directly {}",
            src.display()
        );
        src.clone()
    } else {
        println!(
            "building Q{bits} turnkey: {} -> {}",
            src.display(),
            out.display()
        );
        mlx_gen_chroma::convert::prequantize_turnkey(&src, &out, bits)
            .expect("prequantize_turnkey succeeds");
        assert!(
            out.join("transformer")
                .join("diffusion_pytorch_model.safetensors")
                .is_file(),
            "missing packed transformer safetensors"
        );
        out.clone()
    };

    // Load DIRECTLY from the tier dir. A packed tier packed-detects via `{base}.scales` (no dense
    // transient, no in-app re-quantize), so we load with `Quant::None`; the dense bf16 tier loads
    // dense the same way.
    let spec = LoadSpec::new(WeightsSource::Dir(load_root));
    let generator = mlx_gen::load(&id, &spec).expect("packed chroma loads");

    // 256² / few-step — packed load-path proof, not a quality bench.
    let req = GenerationRequest {
        prompt: "a photograph of an astronaut riding a horse".into(),
        width: 256,
        height: 256,
        count: 1,
        seed: Some(42),
        steps: Some(8),
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

    if bits != 0 && std::env::var("SC8777_KEEP").is_err() {
        let _ = std::fs::remove_dir_all(&out);
        println!("  removed {} (set SC8777_KEEP to retain)", out.display());
    }
}
