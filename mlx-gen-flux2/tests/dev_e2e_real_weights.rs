//! sc-2365: FLUX.2-**dev** end-to-end txt2img on real weights. `#[ignore]`d — needs the real
//! `black-forest-labs/FLUX.2-dev` snapshot (~60 GB DiT + ~45 GB TE):
//!
//!   cargo test -p mlx-gen-flux2 --release --test dev_e2e_real_weights -- --ignored --nocapture
//!
//! Proves the whole dev vertical end to end: assemble a pre-quantized Q4 snapshot (sc-5917 convert
//! for DiT + TE, VAE/tokenizer symlinked from the source), load it through the registry
//! (`mlx_gen::load("flux2_dev", …)` → `load_dev`), and render an image. This exercises the Mistral3
//! TE → the **embedded guidance** embedder (the one genuinely new dev piece) → the dev DiT denoise →
//! the (klein-shared) VAE decode. Rust runs f32 activations vs the BFL bf16 reference, so — like the
//! klein e2e — this is a coherence/quality floor (finite + non-degenerate render at the dev
//! defaults), not a bit-parity claim; a wiring bug (guidance dropped, TE mis-shaped, wrong DiT dims)
//! collapses the render to a flat field, which the variance gate catches.
//!
//! Env overrides for a faster local run: `MLX_GEN_FLUX2_DEV_SIZE` (default 1024), the acceptance
//! target; `MLX_GEN_FLUX2_DEV_STEPS` (default = the dev default, ~28); `MLX_GEN_FLUX2_DEV_PROMPT`.
//! Wrap the test binary in `/usr/bin/time -l` for the steady-state footprint (cf. sc-5917).

use std::path::PathBuf;

use mlx_gen::media::Image;
use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};
use mlx_gen_flux2::{quantize_flux2_dit, quantize_flux2_text_encoder_dir};

const BITS: i32 = 4;
const GROUP_SIZE: i32 = 64;

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("MLX_GEN_FLUX2_DEV_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").expect("HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots");
    std::fs::read_dir(&snaps)
        .expect("snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir under models--black-forest-labs--FLUX.2-dev/snapshots")
}

/// Assemble a complete pre-quantized Q4 dev snapshot at a stable temp path: pre-quantize the DiT +
/// Mistral TE (reusing a prior run's output if present, shared with `quant_prequantize_real_weights`),
/// and symlink the unchanged VAE + tokenizer from the source. Returns the assembled snapshot dir.
fn prequantized_dev_snapshot() -> PathBuf {
    let src = snapshot();
    let dst = std::env::temp_dir().join(format!("mlx_gen_flux2_dev_prequant_q{BITS}"));

    if !dst
        .join("transformer/diffusion_pytorch_model.safetensors")
        .exists()
    {
        println!("pre-quantizing dev DiT → Q{BITS}…");
        quantize_flux2_dit(
            &src.join("transformer"),
            &dst.join("transformer"),
            BITS,
            GROUP_SIZE,
        )
        .expect("pre-quantize dev DiT");
    }
    if !dst.join("text_encoder/model.safetensors").exists() {
        println!("pre-quantizing dev Mistral TE → Q{BITS}…");
        quantize_flux2_text_encoder_dir(
            &src.join("text_encoder"),
            &dst.join("text_encoder"),
            BITS,
            GROUP_SIZE,
        )
        .expect("pre-quantize dev TE");
    }
    // VAE (dense, identical to klein) + tokenizer: symlink straight from the source snapshot.
    for sub in ["vae", "tokenizer"] {
        let link = dst.join(sub);
        if !link.exists() {
            std::os::unix::fs::symlink(std::fs::canonicalize(src.join(sub)).unwrap(), &link)
                .expect("symlink component");
        }
    }
    dst
}

/// (mean, std) of the image's bytes — a coherent render has real spatial variance; a wiring bug that
/// drops guidance / mis-shapes the TE collapses it toward a flat field (std → 0).
fn mean_std(img: &Image) -> (f32, f32) {
    let n = img.pixels.len() as f32;
    let mean = img.pixels.iter().map(|&p| p as f32).sum::<f32>() / n;
    let var = img
        .pixels
        .iter()
        .map(|&p| (p as f32 - mean).powi(2))
        .sum::<f32>()
        / n;
    (mean, var.sqrt())
}

#[test]
#[ignore = "needs real FLUX.2-dev snapshot (~105 GB); assembles a Q4 snapshot in TMPDIR"]
fn dev_txt2img_renders_coherent_image() {
    let size: u32 = std::env::var("MLX_GEN_FLUX2_DEV_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let steps: Option<u32> = std::env::var("MLX_GEN_FLUX2_DEV_STEPS")
        .ok()
        .and_then(|s| s.parse().ok());
    let prompt = std::env::var("MLX_GEN_FLUX2_DEV_PROMPT")
        .unwrap_or_else(|_| "a red fox resting in fresh snow under soft winter light".into());

    let dst = prequantized_dev_snapshot();
    let gen = mlx_gen::load("flux2_dev", &LoadSpec::new(WeightsSource::Dir(dst)))
        .expect("dev loads through the registry");

    let req = GenerationRequest {
        prompt: prompt.clone(),
        width: size,
        height: size,
        count: 1,
        seed: Some(0),
        steps, // None ⇒ the dev default (~28)
        ..Default::default()
    };
    let mut last = 0u32;
    let out = gen
        .generate(&req, &mut |p| {
            if let mlx_gen::Progress::Step { current, total } = p {
                if current == 1 || current == total || current % 8 == 0 {
                    println!("  step {current}/{total}");
                }
                last = current;
            }
        })
        .expect("dev generate");
    let _ = last;

    let GenerationOutput::Images(images) = out else {
        panic!("expected images");
    };
    let img = &images[0];
    assert_eq!((img.width, img.height), (size, size), "output dimensions");
    assert_eq!(
        img.pixels.len(),
        (size * size * 3) as usize,
        "RGB8 pixel count"
    );

    let (mean, std) = mean_std(img);
    println!(
        "flux2-dev T2I OK: {size}² prompt={prompt:?} steps={steps:?} → mean={mean:.1} std={std:.1}"
    );
    // A coherent photo spans a wide tonal range; a degenerate/flat render (wiring bug) has std≈0
    // and a mean pinned at an extreme. Generous floor — this is a smoke, not an aesthetic judge.
    assert!(std > 10.0, "render looks degenerate (flat): std={std:.2}");
    assert!(
        mean > 2.0 && mean < 253.0,
        "render pinned to an extreme: mean={mean:.2}"
    );
}
