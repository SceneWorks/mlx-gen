//! sc-3329 regression: the **stock** SDXL Q8/Q4 quantized U-Net must render a coherent image at
//! 1024², not a flat/collapsed one. This started as a diagnostic probe (epic 3109 sc-3116) of an
//! InstantID 1024² Q8 collapse; it was root-caused to the base SDXL quant path — the resnet
//! `conv_shortcut` (a 1×1 conv stored as a Linear) was being quantized, injecting int8 error directly
//! into the residual stream, which at 1024² compounds across the denoise loop into a runaway latent
//! outlier that blows out the VAE. Fixed by keeping `conv_shortcut` dense (see
//! `ResnetBlock2D::quantize`). sc-2641 validated only ≤512², where the instability stays
//! sub-threshold, so it missed this; we now guard 1024² for both Q8 and Q4.
//!
//! A coherent SDXL render has high pixel variance; a collapsed one sits near-constant (std ~11).
//! `#[ignore]`d — needs the SDXL base snapshot. Run:
//!   cargo test -p mlx-gen-sdxl --release --test q8_1024_probe -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::{GenerationOutput, GenerationRequest, Image, LoadSpec, Quant, WeightsSource};
use mlx_gen_sdxl as _;

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("SDXL_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn std_dev(img: &Image) -> f32 {
    let n = img.pixels.len() as f64;
    let mean = img.pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
    let var = img
        .pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    var.sqrt() as f32
}

fn render(q: Quant, size: u32, steps: u32) -> Image {
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot())).with_quant(q);
    let model = mlx_gen::load("sdxl", &spec).unwrap();
    let req = GenerationRequest {
        prompt: "a portrait photo of a man, sharp focus, high detail".to_string(),
        negative_prompt: Some("lowres, blurry".to_string()),
        width: size,
        height: size,
        seed: Some(0),
        steps: Some(steps),
        guidance: Some(5.0),
        ..Default::default()
    };
    match model.generate(&req, &mut |_| {}).unwrap() {
        GenerationOutput::Images(mut v) => v.pop().unwrap(),
        other => panic!("expected Images, got {other:?}"),
    }
}

#[test]
#[ignore = "needs the SDXL base snapshot"]
fn base_sdxl_quant_coherent_at_512_and_1024() {
    for q in [Quant::Q8, Quant::Q4] {
        let s512 = std_dev(&render(q, 512, 12));
        let s1024 = std_dev(&render(q, 1024, 12));
        println!("[sdxl {q:?} probe] 512²: std = {s512:.1}; 1024²: std = {s1024:.1}");
        // Both resolutions must be coherent (high pixel variance). A collapsed/flat output sits
        // near-constant (std ~11); the 1024² check is the sc-3329 regression (conv_shortcut dense).
        assert!(
            s512 > 40.0,
            "base SDXL {q:?} collapsed at 512² (std {s512:.1})"
        );
        assert!(
            s1024 > 40.0,
            "base SDXL {q:?} collapsed at 1024² (std {s1024:.1}); 512² std {s512:.1} — the resnet \
             conv_shortcut must stay dense (sc-3329)"
        );
    }
}
