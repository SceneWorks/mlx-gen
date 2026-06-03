//! sc-2346 S5: end-to-end real-weights parity for FLUX.2-klein single-reference EDIT. `#[ignore]`d
//! — needs the real snapshot + the f32 golden from `tools/dump_flux2_edit_golden.py`:
//!
//!   cd ~/repos/mflux && .venv/bin/python ~/repos/mlx-gen/tools/dump_flux2_edit_golden.py
//!   cargo test -p mlx-gen-flux2 --test edit_real_weights -- --ignored --nocapture
//!
//! Two gates (f32):
//!  1. **reference encoding** — the NEW edit chain (preprocess → VAE-encode → 2×2 patchify →
//!     BN-normalize → pack) reproduces the fork's `image_latents` (chaos-free, tight);
//!  2. **full edit generate** — `load("flux2_klein_9b_edit").generate(Reference)` render vs the
//!     fork's decoded image (px>8 coherence/floor, like the txt2img e2e).

use std::path::PathBuf;

use mlx_gen::image::decoded_to_image;
use mlx_gen::media::Image;
use mlx_gen::weights::Weights;
use mlx_gen::{Conditioning, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};
use mlx_gen_flux2::{load_vae, pack_latents, patchify_latents, preprocess_ref_image};
use mlx_rs::{Array, Dtype};

const PROMPT: &str = "make it look like a cold winter morning";

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
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/golden/flux2_edit.safetensors");
    Weights::from_file(&path).unwrap_or_else(|_| {
        panic!(
            "missing {} — run tools/dump_flux2_edit_golden.py",
            path.display()
        )
    })
}

/// Reconstruct the reference `Image` from the golden's `ref_u8` `[256,256,3]`.
fn ref_image(g: &Weights) -> Image {
    let a = g.require("ref_u8").unwrap().as_dtype(Dtype::Int32).unwrap();
    let sh = a.shape();
    let (h, w) = (sh[0] as u32, sh[1] as u32);
    let pixels: Vec<u8> = a
        .reshape(&[sh.iter().product::<i32>()])
        .unwrap()
        .as_slice::<i32>()
        .iter()
        .map(|&v| v as u8)
        .collect();
    Image {
        width: w,
        height: h,
        pixels,
    }
}

fn rel(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
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
#[ignore = "needs real FLUX.2-klein-9b snapshot + tools/golden/flux2_edit.safetensors"]
fn reference_encoding_matches_fork() {
    let g = golden();
    let vae = load_vae(&snapshot()).unwrap();
    let img = ref_image(&g);
    // The edit reference chain (256² → no LANCZOS resize), via the public pipeline + VAE APIs.
    let pre = preprocess_ref_image(&img, 256, 256).unwrap(); // NHWC [1,256,256,3]
    let enc = vae
        .encode_mean(&pre)
        .unwrap()
        .transpose_axes(&[0, 3, 1, 2])
        .unwrap(); // [1,32,32,32]
    let patchified = patchify_latents(&enc).unwrap(); // [1,128,16,16]
    let normed = vae.bn_normalize_nchw(&patchified).unwrap();
    let packed = pack_latents(&normed).unwrap(); // [1,256,128]
    let want = g.require("image_latents").unwrap();
    assert_eq!(packed.shape(), want.shape(), "image_latents shape");
    let (peak, mean) = rel(&packed, want);
    println!("flux2 edit reference encoding: peak_rel={peak:.4} mean_rel={mean:.4}");
    assert!(
        mean < 5e-3,
        "reference image_latents diverged: mean_rel={mean}"
    );
}

#[test]
#[ignore = "needs real FLUX.2-klein-9b snapshot + tools/golden/flux2_edit.safetensors"]
fn full_edit_generate_matches_fork() {
    let g = golden();
    let gen = mlx_gen::load(
        "flux2_klein_9b_edit",
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
        conditioning: vec![Conditioning::Reference {
            image: ref_image(&g),
            strength: None,
        }],
        ..Default::default()
    };
    let out = gen.generate(&req, &mut |_| {}).unwrap();
    let GenerationOutput::Images(images) = out else {
        panic!("expected images");
    };
    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let px = px_gt8(&images[0], &gimg);
    println!("flux2 edit full generate: {px:.2}% px>8 vs fork f32 (NAX-vs-wheel build delta)");
    assert!(
        px < 25.0,
        "edit generate diverged from the fork composition: {px}% px>8"
    );
}
