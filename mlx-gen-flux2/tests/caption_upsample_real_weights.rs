//! Real-weight smoke for FLUX.2-dev **caption upsampling** (sc-6030): load the dev Mistral3 tower
//! (packed Q4) + Pixtral vision tower + projector and run the actual `upsample_prompt` rewrite for
//! both the T2I (text-only) and I2I (image-conditioned) paths, asserting a non-empty rewrite that
//! differs from the input. This exercises the whole new surface end-to-end on real weights — the
//! multimodal merge, the KV-cached autoregressive Mistral decode, and the decode of the generated
//! tokens — without loading the 60 GB DiT (so it runs in ~15 GB).
//!
//! `#[ignore]`d — needs a FLUX.2-dev snapshot. Prefers the pre-quantized Q4 snapshot (sc-5917, the
//! `$TMPDIR/mlx_gen_flux2_dev_prequant_q4` the other dev tests reuse) so the Mistral tower loads
//! packed (~13 GB) instead of the ~45 GB bf16 transient; override with `MLX_GEN_FLUX2_DEV_PREQUANT`
//! or fall back to the stock HF snapshot (dense bf16) via `MLX_GEN_FLUX2_DEV_SNAPSHOT`.
//!
//!   cargo test -p mlx-gen-flux2 --test caption_upsample_real_weights -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::media::Image;
use mlx_gen::runtime::CancelFlag;
use mlx_gen_flux2::{
    load_multimodal_projector_dev, load_text_encoder_dev, load_tokenizer_dev,
    load_vision_tower_dev, upsample_prompt,
};

/// The dev snapshot to load from: the pre-quantized Q4 snapshot if present (fast, ~13 GB packed
/// Mistral tower), else the stock HF dev snapshot (dense bf16).
fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("MLX_GEN_FLUX2_DEV_PREQUANT") {
        return PathBuf::from(p);
    }
    let tmp = std::env::temp_dir().join("mlx_gen_flux2_dev_prequant_q4");
    if tmp.join("text_encoder").is_dir() {
        return tmp;
    }
    if let Ok(p) = std::env::var("MLX_GEN_FLUX2_DEV_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").expect("HOME");
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots");
    std::fs::read_dir(&base)
        .expect("a dev snapshot (set MLX_GEN_FLUX2_DEV_PREQUANT or MLX_GEN_FLUX2_DEV_SNAPSHOT)")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot under models--black-forest-labs--FLUX.2-dev/snapshots")
}

fn synthetic_image() -> Image {
    // A small deterministic gradient (RGB8, HWC). The upsampler only needs a real tensor through the
    // tower; the rewrite is sampled text, so exact pixels don't matter.
    let (w, h) = (96u32, 64u32);
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.push((x % 256) as u8);
            pixels.push((y % 256) as u8);
            pixels.push(((x + y) % 256) as u8);
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

#[test]
#[ignore = "needs a real FLUX.2-dev snapshot (prequant Q4 preferred)"]
fn t2i_and_i2i_upsample_produce_nonempty_rewrites() {
    let root = snapshot();
    let tokenizer = load_tokenizer_dev(&root).expect("dev tokenizer");
    let encoder = load_text_encoder_dev(&root).expect("dev Mistral tower + generation head");
    assert!(
        encoder.can_generate(),
        "load_text_encoder_dev must load the generation head (final norm + lm_head)"
    );
    let vision = load_vision_tower_dev(&root).expect("Pixtral vision tower");
    let projector = load_multimodal_projector_dev(&root).expect("Mistral3 projector");
    let cancel = CancelFlag::default();

    // T2I: text-only rewrite. Greedy (temperature 0) so the smoke is deterministic.
    let prompt = "a fox in the snow";
    let t2i = upsample_prompt(
        &tokenizer,
        &encoder,
        &vision,
        &projector,
        prompt,
        &[],
        0.0,
        64,
        0,
        &cancel,
    )
    .expect("t2i upsample");
    println!("T2I rewrite: {t2i}");
    assert!(!t2i.trim().is_empty(), "T2I rewrite must be non-empty");
    assert_ne!(
        t2i.trim(),
        prompt,
        "T2I rewrite should differ from the input prompt"
    );

    // I2I: image-conditioned rewrite (the vision tower + projector + multimodal splice path).
    let image = synthetic_image();
    let edit_prompt = "make it night";
    let i2i = upsample_prompt(
        &tokenizer,
        &encoder,
        &vision,
        &projector,
        edit_prompt,
        &[&image],
        0.0,
        64,
        0,
        &cancel,
    )
    .expect("i2i upsample");
    println!("I2I rewrite: {i2i}");
    assert!(!i2i.trim().is_empty(), "I2I rewrite must be non-empty");
}
