//! sc-2292: FLUX.2-**dev** strict-pose (Fun-Controlnet-Union) end-to-end on real weights. `#[ignore]`d
//! — needs the real `black-forest-labs/FLUX.2-dev` snapshot (~105 GB) **and** the
//! `alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union` checkpoint (~8 GB), both in the HF cache:
//!
//!   cargo test -p mlx-gen-flux2 --release --test control_real_weights -- --ignored --nocapture
//!
//! Proves the dev control vertical end to end: assemble a pre-quantized Q4 dev snapshot (shared with
//! `dev_e2e_real_weights`), load it through the registry as `flux2_dev_control` with the control
//! checkpoint overlaid (`spec.control`), and render a pose-conditioned image. This exercises the
//! Mistral3 TE → embedded guidance → the dev DiT with the VACE control branch (`control_img_in` +
//! the 4 control double blocks injecting hints at base blocks [0,2,4,6]) → VAE decode. No fork golden
//! for dev, so this is a coherence/quality floor (finite + non-degenerate render), not bit-parity; a
//! wiring bug (hints dropped, control context mis-shaped, wrong base-block injection) collapses the
//! render to a flat field, which the variance gate catches. At `control_context_scale = 0` the
//! `control_parity` unit test already proves the branch reduces to the base forward exactly.

use std::path::PathBuf;

use mlx_gen::media::Image;
use mlx_gen::{
    Conditioning, ControlKind, GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource,
};
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

/// The Fun-Controlnet-Union checkpoint (`-2602`, the CFG-distilled recommended one). Override with
/// `MLX_GEN_FLUX2_CONTROL_CHECKPOINT` to point at the non-distilled file.
fn control_checkpoint() -> PathBuf {
    if let Ok(p) = std::env::var("MLX_GEN_FLUX2_CONTROL_CHECKPOINT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").expect("HOME");
    let snaps = PathBuf::from(home).join(
        ".cache/huggingface/hub/models--alibaba-pai--FLUX.2-dev-Fun-Controlnet-Union/snapshots",
    );
    let snap = std::fs::read_dir(&snaps)
        .expect("control snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir under models--alibaba-pai--FLUX.2-dev-Fun-Controlnet-Union");
    snap.join("FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors")
}

/// Assemble a complete pre-quantized Q4 dev snapshot (shared TMPDIR with `dev_e2e_real_weights`):
/// pre-quantize the DiT + Mistral TE, symlink the unchanged VAE + tokenizer from the source.
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
    for sub in ["vae", "tokenizer"] {
        let link = dst.join(sub);
        if !link.exists() {
            std::os::unix::fs::symlink(std::fs::canonicalize(src.join(sub)).unwrap(), &link)
                .expect("symlink component");
        }
    }
    dst
}

/// A deterministic synthetic "pose" control image: a structured field (a stick-figure-ish set of
/// bright bars on a dark ground) — the smoke only needs a real, non-degenerate image to VAE-encode
/// into the control context; correctness is the coherence floor, not a pose match.
fn synthetic_pose(size: u32) -> Image {
    let mut pixels = vec![12u8; (size * size * 3) as usize];
    let s = size as i32;
    let put = |px: &mut [u8], x: i32, y: i32| {
        if x >= 0 && x < s && y >= 0 && y < s {
            let i = ((y * s + x) * 3) as usize;
            px[i] = 235;
            px[i + 1] = 235;
            px[i + 2] = 235;
        }
    };
    let cx = s / 2;
    // Vertical "spine" + two "arms" + two "legs" — thick bars so they survive the 8× VAE downsample.
    for y in (s / 6)..(5 * s / 6) {
        for dx in -2..=2 {
            put(&mut pixels, cx + dx, y);
        }
    }
    for t in 0..(s / 4) {
        for d in -1..=1 {
            put(&mut pixels, cx - t, s / 3 + t + d); // arm
            put(&mut pixels, cx + t, s / 3 + t + d); // arm
            put(&mut pixels, cx - t, 5 * s / 6 + d); // leg base spread
            put(&mut pixels, cx + t, 5 * s / 6 + d);
        }
    }
    Image {
        width: size,
        height: size,
        pixels,
    }
}

/// (mean, std) of the image's bytes — a coherent render has real spatial variance; a wiring bug
/// collapses it toward a flat field (std → 0).
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

/// Env-tunable for a fast local run: `MLX_GEN_FLUX2_CONTROL_SIZE` (default 512),
/// `MLX_GEN_FLUX2_CONTROL_STEPS` (default 8), `MLX_GEN_FLUX2_CONTROL_SCALE` (default 0.75 — the
/// README's 0.65–0.80 range), `MLX_GEN_FLUX2_CONTROL_PROMPT`.
#[test]
#[ignore = "needs real FLUX.2-dev (~105 GB) + Fun-Controlnet-Union (~8 GB); assembles a Q4 snapshot in TMPDIR"]
fn dev_control_renders_coherent_pose_image() {
    let size: u32 = std::env::var("MLX_GEN_FLUX2_CONTROL_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let steps: Option<u32> = Some(
        std::env::var("MLX_GEN_FLUX2_CONTROL_STEPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8),
    );
    let scale: f32 = std::env::var("MLX_GEN_FLUX2_CONTROL_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.75);
    let prompt = std::env::var("MLX_GEN_FLUX2_CONTROL_PROMPT")
        .unwrap_or_else(|_| "a person standing in a sunlit meadow, photorealistic".into());

    let dst = prequantized_dev_snapshot();
    let spec = LoadSpec::new(WeightsSource::Dir(dst))
        .with_control(WeightsSource::File(control_checkpoint()))
        .with_quant(Quant::Q4);
    let gen =
        mlx_gen::load("flux2_dev_control", &spec).expect("dev-control loads through the registry");

    let req = GenerationRequest {
        prompt: prompt.clone(),
        width: size,
        height: size,
        count: 1,
        seed: Some(0),
        steps,
        conditioning: vec![Conditioning::Control {
            image: synthetic_pose(size),
            kind: ControlKind::Pose,
            scale: Some(scale),
        }],
        ..Default::default()
    };
    let GenerationOutput::Images(images) = gen
        .generate(&req, &mut |p| {
            if let mlx_gen::Progress::Step { current, total } = p {
                if current == 1 || current == total || current % 8 == 0 {
                    println!("  step {current}/{total}");
                }
            }
        })
        .expect("dev-control generate")
    else {
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
        "flux2-dev CONTROL OK: {size}² scale={scale} steps={steps:?} prompt={prompt:?} → \
         mean={mean:.1} std={std:.1}"
    );
    assert!(
        std > 10.0,
        "control render looks degenerate (flat): std={std:.2}"
    );
    assert!(
        mean > 2.0 && mean < 253.0,
        "control render pinned to an extreme: mean={mean:.2}"
    );
}
