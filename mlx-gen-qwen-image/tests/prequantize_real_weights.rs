//! sc-8670 (Group-B, sc-8669): maintainer's on-device proof that a **pre-quantized packed**
//! Qwen-Image tier built by [`mlx_gen_qwen_image::convert::prequantize_turnkey`] loads directly via
//! the packed-detect loader ([`mlx_gen_qwen_image::quant`]) and renders a coherent image — no dense
//! bf16 transformer transient, no in-app `transformer.quantize` (epic 8506).
//!
//! Qwen-Image quantizes the **transformer only**; the Qwen2.5-VL text encoder + VAE stay dense in
//! every tier. So the packed tier = a packed `transformer/model.safetensors` + the dense TE/VAE/
//! tokenizer copied through. Loading with `Quant::Q4` is honored but no-ops on the already-packed
//! transformer (and never touches the dense TE), which is exactly the property under test.
//!
//! `#[ignore]`d — needs a real `Qwen/Qwen-Image` snapshot (the ~40 GB bf16 transformer + ~15 GB
//! Qwen2.5-VL TE + VAE). Run with:
//!   cargo test -p mlx-gen-qwen-image --release --test prequantize_real_weights -- --ignored --nocapture
//!
//! Env knobs:
//!   SC8670_SRC    source snapshot dir (default: the first HF-cache Qwen-Image snapshot)
//!   SC8670_OUT    output dir for the packed tier (default: a scratch dir in the temp dir)
//!   SC8670_BITS   4 (Q4, default) or 8 (Q8)
//!   SC8670_KEEP   if set, keep the built tier (else it is removed after the render)

use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};
use mlx_gen_qwen_image as qwen;
use std::path::PathBuf;

/// Resolve a Qwen-Image snapshot: `SC8670_SRC` if set, else the first HF-cache snapshot.
fn qwen_image_snapshot() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SC8670_SRC") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots");
    std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
}

/// Build-only harness for producing **hostable** tiers (epic 8506 rollout): pack a tier from any
/// Qwen-Image-family snapshot (base T2I or Edit) into `SC8670_OUT` and keep it — no load/generate
/// (so it is variant-agnostic and fast). The hosted bf16 tier is the dense source mirrored directly
/// (no conversion); this harness produces the Q4/Q8 tiers. Run per tier:
///   SC8670_SRC=<snapshot> SC8670_OUT=<staging/q4> SC8670_BITS=4 \
///     cargo test -p mlx-gen-qwen-image --release --test prequantize_real_weights \
///     -- --ignored build_tier_only --nocapture
#[test]
#[ignore = "build-only tier producer for hosting; set SC8670_SRC/OUT/BITS"]
fn build_tier_only() {
    let src = PathBuf::from(
        std::env::var("SC8670_SRC").expect("SC8670_SRC (source snapshot dir) required"),
    );
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
    qwen::convert::prequantize_turnkey(&src, &out, bits).expect("prequantize_turnkey succeeds");
    let tf = out.join("transformer").join("model.safetensors");
    let sz = std::fs::metadata(&tf)
        .expect("packed transformer present")
        .len();
    println!(
        "✓ built {} ({:.2} GB packed transformer + dense TE/VAE)",
        out.display(),
        sz as f64 / 1e9
    );
}

#[test]
#[ignore = "needs a real Qwen/Qwen-Image snapshot; builds a packed tier (set SC8670_SRC/OUT/BITS)"]
fn prequantize_turnkey_loads_packed_and_renders() {
    let Some(src) = qwen_image_snapshot() else {
        eprintln!("skip: no Qwen/Qwen-Image snapshot (set SC8670_SRC or populate the HF cache)");
        return;
    };
    let bits: i32 = std::env::var("SC8670_BITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let quant = if bits >= 8 { Quant::Q8 } else { Quant::Q4 };
    let out = std::env::var("SC8670_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join(format!("qwen-image-q{bits}")));

    println!(
        "building Q{bits} turnkey: {} -> {}",
        src.display(),
        out.display()
    );
    qwen::convert::prequantize_turnkey(&src, &out, bits).expect("prequantize_turnkey succeeds");

    // The packed transformer must be a single `model.safetensors` (the loader globs `*.safetensors`).
    let tf = out.join("transformer").join("model.safetensors");
    assert!(tf.is_file(), "missing packed transformer/model.safetensors");
    let sz = std::fs::metadata(&tf).unwrap().len();
    println!(
        "  transformer/model.safetensors = {:.2} GB",
        sz as f64 / 1e9
    );
    // The TE stays dense (sharded) — a sanity check that we did NOT pack it.
    assert!(
        out.join("text_encoder").is_dir(),
        "missing dense text_encoder dir"
    );

    // Load DIRECTLY from the packed dir (packed-detect). `with_quant` is honored but the packed
    // transformer Linears are already quantized, so the post-load `quantize` calls no-op — this
    // proves the packed path, not an in-app quantize.
    let spec = LoadSpec::new(WeightsSource::Dir(out.clone())).with_quant(quant);
    let generator = mlx_gen::load("qwen_image", &spec).expect("packed qwen_image loads");

    // Small/short — this is a packed load-path proof, not a quality bench (the 20B MMDiT is slow).
    let req = GenerationRequest {
        prompt: "a red fox sitting in a snowy forest, photorealistic, sharp focus".into(),
        width: 512,
        height: 512,
        seed: Some(42),
        steps: Some(8),
        ..Default::default()
    };
    let out_img = generator
        .generate(&req, &mut |_| {})
        .expect("packed generate succeeds");
    let img = match out_img {
        GenerationOutput::Images(mut v) => {
            assert_eq!(v.len(), 1);
            v.pop().unwrap()
        }
        other => panic!("expected Images, got {other:?}"),
    };
    assert_eq!(
        (img.width, img.height),
        (req.width, req.height),
        "image size"
    );

    let min = *img.pixels.iter().min().unwrap();
    let max = *img.pixels.iter().max().unwrap();
    let mean = img.pixels.iter().map(|&p| p as u64).sum::<u64>() as f64 / img.pixels.len() as f64;
    println!(
        "✓ packed Q{bits} qwen_image: {}x{}; px min={min} max={max} mean={mean:.1}",
        img.width, img.height
    );
    assert!(
        max as i32 - min as i32 > 32,
        "degenerate render: pixel range {min}..={max} too flat for a coherent image"
    );

    let png = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../tools/golden/packed_qwen_image_q{bits}.png"));
    let _ = image::save_buffer(
        &png,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    );
    println!("  saved {}", png.display());

    if std::env::var("SC8670_KEEP").is_err() {
        let _ = std::fs::remove_dir_all(&out);
        println!("  removed {} (set SC8670_KEEP to retain)", out.display());
    }
}
