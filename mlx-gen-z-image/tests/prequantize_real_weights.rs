//! sc-8670: maintainer's on-device proof that a **pre-quantized packed** Z-Image tier built by
//! [`mlx_gen_z_image::convert::prequantize_turnkey`] loads directly via the packed-detect loaders
//! ([`mlx_gen_z_image::quant`]) and renders a coherent image — no dense bf16 transient, no in-app
//! `model.quantize` (the Group-B pilot, sc-8669 / epic 8506).
//!
//! `#[ignore]`d — needs the real `Tongyi-MAI/Z-Image-Turbo` snapshot (the 23 GB f32 transformer +
//! 7.5 GB bf16 text encoder + VAE). Run with:
//!   cargo test -p mlx-gen-z-image --release --test prequantize_real_weights -- --ignored --nocapture
//!
//! Env knobs:
//!   SC8670_SRC    source snapshot dir (default: the first HF-cache Z-Image-Turbo snapshot)
//!   SC8670_OUT    output dir for the packed tier (default: a scratch dir alongside the cache)
//!   SC8670_BITS   4 (Q4, default) or 8 (Q8)
//!   SC8670_KEEP   if set, keep the built tier (else it is removed after the render)

use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};
use mlx_gen_z_image as zi;
use std::path::PathBuf;

/// Resolve the Z-Image-**Turbo** snapshot: `SC8670_SRC` if set, else the first HF-cache snapshot.
fn turbo_snapshot() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SC8670_SRC") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots");
    std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
}

#[test]
#[ignore = "needs the real Tongyi-MAI/Z-Image-Turbo snapshot; builds a packed tier (set SC8670_SRC/OUT/BITS)"]
fn prequantize_turnkey_loads_packed_and_renders() {
    let Some(src) = turbo_snapshot() else {
        eprintln!(
            "skip: no Tongyi-MAI/Z-Image-Turbo snapshot (set SC8670_SRC or populate the HF cache)"
        );
        return;
    };
    let bits: i32 = std::env::var("SC8670_BITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let quant = if bits >= 8 { Quant::Q8 } else { Quant::Q4 };
    let out = std::env::var("SC8670_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join(format!("z-image-turbo-q{bits}")));

    // Build the packed turnkey (idempotent re-build: the transformer/text_encoder/vae dirs are
    // overwritten with a single packed `model.safetensors` each).
    println!(
        "building Q{bits} turnkey: {} -> {}",
        src.display(),
        out.display()
    );
    zi::convert::prequantize_turnkey(&src, &out, bits).expect("prequantize_turnkey succeeds");

    // The packed dir must NOT carry a dense source shard — the loader globs `*.safetensors`, and a
    // single packed `model.safetensors` per component is the whole weight set.
    for comp in ["transformer", "text_encoder", "vae"] {
        let f = out.join(comp).join("model.safetensors");
        assert!(f.is_file(), "missing packed {comp}/model.safetensors");
        let sz = std::fs::metadata(&f).unwrap().len();
        println!("  {comp}/model.safetensors = {:.2} GB", sz as f64 / 1e9);
    }

    // Load DIRECTLY from the packed dir (packed-detect). `with_quant` is honored but the packed
    // Linears are already quantized, so the post-load `quantize` calls no-op — this proves the
    // packed path, not an in-app quantize.
    let spec = LoadSpec::new(WeightsSource::Dir(out.clone())).with_quant(quant);
    let generator = mlx_gen::load("z_image_turbo", &spec).expect("packed z_image_turbo loads");

    // Turbo is guidance-distilled: 4 steps, no guidance / negative prompt.
    let req = GenerationRequest {
        prompt: "a red fox sitting in a snowy forest, photorealistic, sharp focus".into(),
        width: 1024,
        height: 1024,
        seed: Some(42),
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
        "✓ packed Q{bits} z_image_turbo: {}x{}; px min={min} max={max} mean={mean:.1}",
        img.width, img.height
    );
    assert!(
        max as i32 - min as i32 > 32,
        "degenerate render: pixel range {min}..={max} too flat for a coherent image"
    );

    let png = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../tools/golden/packed_z_image_turbo_q{bits}.png"));
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
