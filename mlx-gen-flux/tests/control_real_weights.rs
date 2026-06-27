//! sc-8238: FLUX.1-**dev** Fun-Controlnet-Union (Shakker `FLUX.1-dev-ControlNet-Union-Pro-2.0`)
//! end-to-end on real weights. `#[ignore]`d — needs the real `black-forest-labs/FLUX.1-dev` snapshot
//! **and** the Shakker control checkpoint, both in the HF cache, plus a Metal GPU:
//!
//!   FLUX1_DEV_SNAPSHOT=/path/to/FLUX.1-dev \
//!   FLUX1_CONTROL_CHECKPOINT=/path/to/diffusion_pytorch_model.safetensors \
//!     cargo test -p mlx-gen-flux --release --test control_real_weights -- --ignored --nocapture
//!
//! This is the **maintainer's on-device gate** (epic 8236): it proves the control vertical end to end —
//! load the dev snapshot through the registry as `flux1_dev_control` with the control checkpoint
//! overlaid (`spec.control`), and render WITH a structural control image. The assertion is a *measurable
//! steer*: the controlled render must differ from the matched control-free FLUX.1-dev render of the same
//! prompt + seed (the control residuals actually flow into the base double stream), AND stay coherent
//! (finite, real spatial variance — a wiring bug collapses it to a flat field). No fork golden for the
//! control path, so this is a steer + coherence floor, not bit-parity; the `control_residual_interval`
//! unit test already pins the injection-point math, and the diffusers zero-init `controlnet_blocks` mean
//! an unconditioned branch is a no-op (so any difference is the encoded control image steering).

use std::path::PathBuf;

use mlx_gen::media::Image;
use mlx_gen::{
    Conditioning, ControlKind, GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource,
};

/// Resolve the FLUX.1-dev snapshot dir: `FLUX1_DEV_SNAPSHOT`, else the newest HF-cache snapshot.
fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("FLUX1_DEV_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").expect("HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots");
    std::fs::read_dir(&snaps)
        .expect("snapshot dir under models--black-forest-labs--FLUX.1-dev/snapshots")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir under models--black-forest-labs--FLUX.1-dev/snapshots")
}

/// The Shakker `FLUX.1-dev-ControlNet-Union-Pro-2.0` checkpoint. Override with
/// `FLUX1_CONTROL_CHECKPOINT`; else the `diffusion_pytorch_model.safetensors` in the newest HF-cache
/// snapshot.
fn control_checkpoint() -> WeightsSource {
    if let Ok(p) = std::env::var("FLUX1_CONTROL_CHECKPOINT") {
        return WeightsSource::File(PathBuf::from(p));
    }
    let home = std::env::var("HOME").expect("HOME");
    let snaps = PathBuf::from(home).join(
        ".cache/huggingface/hub/models--Shakker-Labs--FLUX.1-dev-ControlNet-Union-Pro-2.0/snapshots",
    );
    let snap = std::fs::read_dir(&snaps)
        .expect("control snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir under models--Shakker-Labs--FLUX.1-dev-ControlNet-Union-Pro-2.0");
    WeightsSource::File(snap.join("diffusion_pytorch_model.safetensors"))
}

/// A deterministic synthetic structural control image (a stick-figure-ish set of bright bars on a dark
/// ground) — the smoke only needs a real, non-degenerate image to VAE-encode into the control latent;
/// correctness is the steer + coherence floor, not a pose match.
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
    for y in (s / 6)..(5 * s / 6) {
        for dx in -2..=2 {
            put(&mut pixels, cx + dx, y);
        }
    }
    for t in 0..(s / 4) {
        for d in -1..=1 {
            put(&mut pixels, cx - t, s / 3 + t + d);
            put(&mut pixels, cx + t, s / 3 + t + d);
            put(&mut pixels, cx - t, 5 * s / 6 + d);
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

/// Mean absolute per-byte difference between two equal-size renders (the steer magnitude).
fn mean_abs_diff(a: &Image, b: &Image) -> f32 {
    assert_eq!(a.pixels.len(), b.pixels.len());
    let n = a.pixels.len() as f32;
    a.pixels
        .iter()
        .zip(&b.pixels)
        .map(|(&x, &y)| (x as f32 - y as f32).abs())
        .sum::<f32>()
        / n
}

fn render(gen: &dyn mlx_gen::Generator, req: &GenerationRequest) -> Image {
    let GenerationOutput::Images(mut images) = gen
        .generate(req, &mut |p| {
            if let mlx_gen::Progress::Step { current, total } = p {
                if current == 1 || current == total || current % 8 == 0 {
                    println!("  step {current}/{total}");
                }
            }
        })
        .expect("flux1-dev-control generate")
    else {
        panic!("expected images");
    };
    images.swap_remove(0)
}

/// Env-tunable for a fast local run: `FLUX1_CONTROL_SIZE` (default 512), `FLUX1_CONTROL_STEPS`
/// (default 12), `FLUX1_CONTROL_SCALE` (default 0.7), `FLUX1_CONTROL_PROMPT`.
#[test]
#[ignore = "needs real FLUX.1-dev + Shakker ControlNet-Union-Pro-2.0 weights + Metal GPU"]
fn dev_control_measurably_steers_render() {
    let size: u32 = std::env::var("FLUX1_CONTROL_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let steps: Option<u32> = Some(
        std::env::var("FLUX1_CONTROL_STEPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(12),
    );
    let scale: f32 = std::env::var("FLUX1_CONTROL_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.7);
    let prompt = std::env::var("FLUX1_CONTROL_PROMPT")
        .unwrap_or_else(|_| "a person standing in a sunlit meadow, photorealistic".into());

    let root = snapshot();

    // (1) The matched control-FREE FLUX.1-dev render (same prompt + seed + dims) — the steer baseline.
    let base_spec = LoadSpec::new(WeightsSource::Dir(root.clone())).with_quant(Quant::Q4);
    let base_gen = mlx_gen::load("flux1_dev", &base_spec).expect("flux1_dev loads");
    let base_req = GenerationRequest {
        prompt: prompt.clone(),
        width: size,
        height: size,
        count: 1,
        seed: Some(0),
        steps,
        ..Default::default()
    };
    let base_img = render(base_gen.as_ref(), &base_req);

    // (2) The control render: identical request + a structural control image.
    let ctrl_spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_control(control_checkpoint())
        .with_quant(Quant::Q4);
    let ctrl_gen = mlx_gen::load("flux1_dev_control", &ctrl_spec)
        .expect("flux1_dev_control loads via registry");
    let ctrl_req = GenerationRequest {
        conditioning: vec![Conditioning::Control {
            image: synthetic_pose(size),
            kind: ControlKind::Pose,
            scale,
        }],
        ..base_req.clone()
    };
    let ctrl_img = render(ctrl_gen.as_ref(), &ctrl_req);

    assert_eq!((ctrl_img.width, ctrl_img.height), (size, size));
    let (mean, std) = mean_std(&ctrl_img);
    let steer = mean_abs_diff(&base_img, &ctrl_img);
    println!(
        "flux1-dev CONTROL OK: {size}² scale={scale} steps={steps:?} → \
         mean={mean:.1} std={std:.1} steer(meanAbsΔ vs control-free)={steer:.2}"
    );
    // Coherence floor: not a flat field, not pinned to an extreme.
    assert!(
        std > 10.0,
        "control render looks degenerate (flat): std={std:.2}"
    );
    assert!(
        mean > 2.0 && mean < 253.0,
        "control render pinned to an extreme: mean={mean:.2}"
    );
    // The load-bearing assertion: the control image MEASURABLY steers the render away from the matched
    // control-free baseline (the residuals reach the base double stream). A no-op injection would leave
    // the two renders byte-identical (steer ≈ 0).
    assert!(
        steer > 1.0,
        "control did not measurably steer the render (meanAbsΔ vs control-free = {steer:.3}); \
         residuals are not reaching the base double stream"
    );
}
