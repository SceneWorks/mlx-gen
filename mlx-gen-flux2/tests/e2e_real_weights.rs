//! sc-2346 S4: end-to-end real-weights parity for FLUX.2-klein txt2img. `#[ignore]`d — needs the
//! real snapshot + the f32 golden from `tools/dump_flux2_e2e_golden.py` (gitignored):
//!
//!   cd ~/repos/mflux && .venv/bin/python ~/repos/mlx-gen/tools/dump_flux2_e2e_golden.py
//!   cargo test -p mlx-gen-flux2 --test e2e_real_weights -- --ignored --nocapture
//!
//! Three gates, all f32 (golden forced `ModelConfig.precision=float32`, Rust runs f32):
//!  1. **RNG** — `create_noise` byte-matches the fork's seeded packed noise (prerequisite).
//!  2. **v0** — the step-0 transformer velocity on real weights (chaos-free: one forward, fed the
//!     fork's own noise/embeds/ids). This is the tight real-weights transformer+composition gate.
//!  3. **full generate** — the public `load().generate()` render vs the fork's decoded image
//!     (px>8). f32-vs-f32 across two MLX builds (NAX vs wheel) over a 4-step sampler still drifts,
//!     so this is a coherence/floor check, not a bit-parity claim (the FLUX.1 lesson).

use std::path::PathBuf;

use mlx_gen::image::decoded_to_image;
use mlx_gen::media::Image;
use mlx_gen::weights::Weights;
use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};
use mlx_gen_flux2::{create_noise, load_transformer};
use mlx_rs::{Array, Dtype};

const PROMPT: &str = "a red fox resting in fresh snow under soft winter light";

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("MLX_GEN_FLUX2_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").expect("HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-9b/snapshots");
    std::fs::read_dir(&snaps)
        .expect("snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn golden() -> Weights {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/golden/flux2_e2e.safetensors");
    Weights::from_file(&path).unwrap_or_else(|_| {
        panic!(
            "missing {} — run tools/dump_flux2_e2e_golden.py",
            path.display()
        )
    })
}

fn f32a(a: &Array) -> Array {
    a.as_dtype(Dtype::Float32).unwrap()
}

fn rel(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let a = f32a(a).reshape(&[n]).unwrap();
    let b = f32a(b).reshape(&[n]).unwrap();
    let (xs, ys) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = ys.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let mabs = (ys.iter().map(|y| y.abs()).sum::<f32>() / ys.len() as f32).max(1e-12);
    let max_d = xs
        .iter()
        .zip(ys)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let mean_d = xs.iter().zip(ys).map(|(x, y)| (x - y).abs()).sum::<f32>() / xs.len() as f32;
    (max_d / peak, mean_d / mabs)
}

fn px_gt8(a: &Image, b: &Image) -> f32 {
    assert_eq!(a.pixels.len(), b.pixels.len(), "image size mismatch");
    let n = a.pixels.len();
    let c = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .filter(|(x, y)| (**x as i32 - **y as i32).abs() > 8)
        .count();
    100.0 * c as f32 / n as f32
}

#[test]
#[ignore = "needs real FLUX.2-klein-9b snapshot + tools/golden/flux2_e2e.safetensors"]
fn seeded_noise_matches_fork() {
    let g = golden();
    let noise = create_noise(0, 256, 256, 128).unwrap();
    let (peak, mean) = rel(&noise, g.require("noise0").unwrap());
    println!("flux2 e2e RNG: peak_rel={peak:.2e} mean_rel={mean:.2e}");
    assert!(
        mean < 1e-5,
        "seeded noise diverged from the fork: mean_rel={mean}"
    );
}

#[test]
#[ignore = "needs real FLUX.2-klein-9b snapshot + tools/golden/flux2_e2e.safetensors"]
fn step0_velocity_matches_fork() {
    let g = golden();
    let t = load_transformer(&snapshot()).unwrap();
    // Feed the fork's own noise / prompt_embeds / ids → isolates the transformer (chaos-free).
    let v = t
        .forward(
            g.require("noise0").unwrap(),
            g.require("prompt_embeds").unwrap(),
            g.require("latent_ids").unwrap(),
            g.require("text_ids").unwrap(),
            1000.0,
        )
        .unwrap();
    let (peak, mean) = rel(&v, g.require("v0").unwrap());
    println!("flux2 e2e v0 (real-weights transformer): peak_rel={peak:.4} mean_rel={mean:.4}");
    assert!(mean < 1e-2, "step-0 velocity diverged: mean_rel={mean}");
}

#[test]
#[ignore = "needs real FLUX.2-klein-9b snapshot + tools/golden/flux2_e2e.safetensors"]
fn full_generate_matches_fork_composition() {
    let g = golden();
    let gen = mlx_gen::load(
        "flux2_klein_9b",
        &LoadSpec::new(WeightsSource::Dir(snapshot())),
    )
    .unwrap();
    let req = GenerationRequest {
        prompt: PROMPT.into(),
        width: 256,
        height: 256,
        count: 1,
        seed: Some(0),
        steps: Some(4),
        ..Default::default()
    };
    let out = gen.generate(&req, &mut |_| {}).unwrap();
    let GenerationOutput::Images(images) = out else {
        panic!("expected images");
    };
    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let px = px_gt8(&images[0], &gimg);
    println!("flux2 e2e full generate: {px:.2}% px>8 vs fork f32 (NAX-vs-wheel build delta over 4 steps)");
    // Wiring bug → ~100%; correct wiring + cross-build f32 drift over a 4-step sampler is bounded.
    assert!(
        px < 25.0,
        "full generate diverged from the fork composition: {px}% px>8"
    );
}
