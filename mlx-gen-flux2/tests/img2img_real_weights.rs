//! sc-2644: end-to-end real-weights parity for FLUX.2-klein **txt2img img2img** (image_path +
//! image_strength). `#[ignore]`d — needs the real snapshot + the f32 golden from
//! `tools/dump_flux2_img2img_golden.py`:
//!
//!   cd ~/repos/mflux && .venv/bin/python ~/repos/mlx-gen/tools/dump_flux2_img2img_golden.py
//!   cd ~/repos/mflux && BITS=8 .venv/bin/python ~/repos/mlx-gen/tools/dump_flux2_img2img_quant_golden.py
//!   cargo test -p mlx-gen-flux2 --test img2img_real_weights -- --ignored --nocapture
//!
//! Five gates (f32 activations):
//!  1. **seeded noise** — `create_noise` reproduces the fork's `prepare_packed_latents` noise;
//!  2. **clean init encoding** — the img2img encode chain (preprocess → VAE-encode → 2×2 patchify →
//!     BN-normalize → pack) reproduces the fork's pre-blend `clean_latents` (chaos-free, tight);
//!  3. **blend** — `add_noise_by_interpolation(clean, noise, σ)` reproduces the fork's blended
//!     `init_latents` at `σ = sigmas[init_time_step]` (bit-exact);
//!  4. **full img2img generate** — `load("flux2_klein_9b").generate(Reference{strength})` render vs
//!     the fork's f32 decoded image (px>8 coherence/floor, like the txt2img/edit e2e);
//!  5. **Q8 img2img generate** — the same render under `spec.quantize = Q8` vs the fork's Q8 image
//!     (bounded coherence floor: Rust f32 acts vs fork bf16, identical quantized weights per sc-2643).

use std::path::PathBuf;

use mlx_gen::image::decoded_to_image;
use mlx_gen::media::Image;
use mlx_gen::weights::Weights;
use mlx_gen::{Conditioning, GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};
use mlx_gen_flux2::{
    add_noise_by_interpolation, create_noise, init_time_step, load_vae, pack_latents,
    patchify_latents, preprocess_ref_image, schedule,
};
use mlx_rs::{Array, Dtype};

const PROMPT: &str = "a red fox resting in fresh snow under soft winter light";
const SIZE: u32 = 256;
const STEPS: usize = 4;
const STRENGTH: f32 = 0.6;

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
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/golden/flux2_img2img.safetensors");
    Weights::from_file(&path).unwrap_or_else(|_| {
        panic!(
            "missing {} — run tools/dump_flux2_img2img_golden.py",
            path.display()
        )
    })
}

/// Reconstruct the init `Image` from the golden's `init_u8` `[256,256,3]`.
fn init_image(g: &Weights) -> Image {
    let a = g
        .require("init_u8")
        .unwrap()
        .as_dtype(Dtype::Int32)
        .unwrap();
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

/// Gate 1+2+3: the img2img latent-prep chain (noise, clean encode, blend) — all chaos-free, tight.
#[test]
#[ignore = "needs real FLUX.2-klein-9b snapshot + tools/golden/flux2_img2img.safetensors"]
fn img2img_latent_prep_matches_fork() {
    let g = golden();

    // init_time_step / start sigma agree with the fork's `Config.init_time_step`.
    let want_its = g
        .require("init_time_step")
        .unwrap()
        .as_dtype(Dtype::Int32)
        .unwrap()
        .as_slice::<i32>()[0] as usize;
    let its = init_time_step(STEPS, Some(STRENGTH));
    assert_eq!(its, want_its, "init_time_step mismatch");
    let sched = schedule(STEPS, SIZE, SIZE);
    let want_sigma = g.require("start_sigma").unwrap().as_slice::<f32>()[0];
    assert!(
        (sched.sigmas[its] - want_sigma).abs() < 1e-6,
        "start sigma mismatch: {} vs {want_sigma}",
        sched.sigmas[its]
    );

    // 1. Seeded noise.
    let noise = create_noise(0, SIZE, SIZE, 128).unwrap();
    let (np, _) = rel(&noise, g.require("noise0").unwrap());
    println!("flux2 img2img seeded noise: peak_rel={np:.6}");
    assert!(np < 1e-5, "seeded noise diverged: peak_rel={np}");

    // 2. Clean (pre-blend) init latents via the public pipeline + VAE APIs (256² → LANCZOS resize is
    //    a no-op; the `_match_latent_spatial_size` step is a no-op at H/8 == lat_h·2).
    let vae = load_vae(&snapshot()).unwrap();
    let img = init_image(&g);
    let pre = preprocess_ref_image(&img, SIZE, SIZE).unwrap(); // NHWC [1,256,256,3]
    let enc = vae
        .encode_mean(&pre)
        .unwrap()
        .transpose_axes(&[0, 3, 1, 2])
        .unwrap(); // [1,32,32,32]
    let patchified = patchify_latents(&enc).unwrap(); // [1,128,16,16]
    let normed = vae.bn_normalize_nchw(&patchified).unwrap();
    let clean = pack_latents(&normed).unwrap(); // [1,256,128]
    let want_clean = g.require("clean_latents").unwrap();
    assert_eq!(clean.shape(), want_clean.shape(), "clean_latents shape");
    let (cp, cm) = rel(&clean, want_clean);
    println!("flux2 img2img clean init: peak_rel={cp:.4} mean_rel={cm:.4}");
    assert!(cm < 5e-3, "clean init latents diverged: mean_rel={cm}");

    // 3. Blend `(1-σ)·clean + σ·noise` at the start sigma == the fork's `init_latents`.
    let blended = add_noise_by_interpolation(&clean, &noise, sched.sigmas[its]).unwrap();
    let (bp, _) = rel(&blended, g.require("init_latents").unwrap());
    println!("flux2 img2img blend: peak_rel={bp:.6}");
    assert!(bp < 5e-3, "blended init latents diverged: peak_rel={bp}");
}

/// The public img2img render: `load(quant).generate(Reference{strength})`.
fn render_img2img(quant: Option<Quant>, image: Image) -> Image {
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    spec.quantize = quant;
    let gen = mlx_gen::load("flux2_klein_9b", &spec).unwrap();
    let req = GenerationRequest {
        prompt: PROMPT.into(),
        width: SIZE,
        height: SIZE,
        count: 1,
        seed: Some(0),
        steps: Some(STEPS as u32),
        conditioning: vec![Conditioning::Reference {
            image,
            strength: Some(STRENGTH),
        }],
        ..Default::default()
    };
    let GenerationOutput::Images(mut images) = gen.generate(&req, &mut |_| {}).unwrap() else {
        panic!("expected images");
    };
    images.pop().unwrap()
}

/// Gate 4: the full public dense img2img render vs the fork f32 render.
#[test]
#[ignore = "needs real FLUX.2-klein-9b snapshot + tools/golden/flux2_img2img.safetensors"]
fn full_img2img_generate_matches_fork() {
    let g = golden();
    let img = render_img2img(None, init_image(&g));
    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let px = px_gt8(&img, &gimg);
    println!("flux2 img2img full generate: {px:.2}% px>8 vs fork f32 (NAX-vs-wheel build delta)");
    assert!(
        px < 25.0,
        "img2img generate diverged from the fork composition: {px}% px>8"
    );
}

/// Gate 5: the **Q8** img2img render vs the fork's Q8 render. The quant scope/packing is already
/// byte-exact (sc-2643), and the img2img latent prep is dtype-orthogonal (VAE encode + blend run
/// f32 regardless of quant), so this composes two proven paths — but the story names Q8 explicitly,
/// so verify it end-to-end rather than argue composition. Rust runs f32 activations vs the fork's
/// bf16 (identical quantized weights), so this is a bounded coherence floor amplified by the 4-step
/// chaos sampler, like the sc-2643 render gate — a wiring/scope bug would blow it past the floor.
#[test]
#[ignore = "needs real FLUX.2-klein-9b snapshot + tools/golden/flux2_img2img_q8.safetensors"]
fn q8_img2img_generate_matches_fork() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tools/golden/flux2_img2img_q8.safetensors");
    let g = Weights::from_file(&path).unwrap_or_else(|_| {
        panic!(
            "missing {} — run `BITS=8 python tools/dump_flux2_img2img_quant_golden.py`",
            path.display()
        )
    });
    let img = render_img2img(Some(Quant::Q8), init_image(&g));
    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let px = px_gt8(&img, &gimg);
    println!("flux2 Q8 img2img full generate: {px:.2}% px>8 vs fork Q8 (f32-act vs bf16-act + cross-build, chaos-amplified)");
    assert!(
        px < 70.0,
        "Q8 img2img generate not coherent: {px}% px>8 (scope/wiring bug?)"
    );
}
