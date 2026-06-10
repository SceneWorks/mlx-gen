//! sc-2533: real-weights validation of the Z-Image **img2img** port against the frozen fork.
//!
//! `#[ignore]`d — needs the real `Tongyi-MAI/Z-Image-Turbo` weights in the HF cache and the golden
//! produced by `tools/dump_z_image_img2img_golden.py` (gitignored, local). Run with:
//!   cargo test -p mlx-gen-z-image --release --test img2img_real_weights -- --ignored --nocapture
//!
//! Stage gates isolate each new piece — the flow-match schedule, the LANCZOS preprocess, the VAE
//! **encoder**, and the noise blend — then the final gate drives the **public**
//! `load(id, spec).generate(req)` API with a `Conditioning::Reference` and confirms the rendered
//! image matches the fork's img2img golden.

use mlx_gen::weights::Weights;
use mlx_gen::{
    Conditioning, FlowMatchEuler, GenerationOutput, GenerationRequest, Image, LoadSpec, Progress,
    WeightsSource,
};
use mlx_gen_z_image::{
    add_noise_by_interpolation, create_noise, decoded_to_image, encode_init_latents,
    init_time_step, load_vae, preprocess_init_image,
};
use mlx_rs::{Array, Dtype};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/z_image_img2img_golden.safetensors"
);

mod common;
use common::snapshot;

/// `(peak-relative, mean-relative)` error vs the golden tensor `b`.
fn rel_errors(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let mean_diff: f32 =
        a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).sum::<f32>() / a.len() as f32;
    (max_diff / peak, mean_diff / peak)
}

/// The synthetic init image, read back from the golden so both sides use byte-identical pixels.
fn init_image(g: &Weights) -> Image {
    let iw: u32 = g.metadata("iw").unwrap().parse().unwrap();
    let ih: u32 = g.metadata("ih").unwrap().parse().unwrap();
    let arr = g.require("init_image_u8").unwrap(); // int32 HWC
    let pixels: Vec<u8> = arr.as_slice::<i32>().iter().map(|&v| v as u8).collect();
    assert_eq!(pixels.len(), (iw * ih * 3) as usize, "init image size");
    Image {
        width: iw,
        height: ih,
        pixels,
    }
}

fn meta_u32(g: &Weights, k: &str) -> u32 {
    g.metadata(k).unwrap().parse().unwrap()
}

/// Gate A: the Rust flow-match schedule reproduces the fork's `config.scheduler.sigmas` (img2img
/// indexes `sigmas[init_time_step]`, so the schedule must match exactly), and `init_time_step`
/// matches the fork's `Config.init_time_step`.
#[test]
#[ignore = "needs real Z-Image weights + local img2img golden"]
fn img2img_schedule_and_init_step_match() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let steps = meta_u32(&g, "steps");
    let strength: f32 = g.metadata("strength").unwrap().parse().unwrap();
    let want_step: usize = g.metadata("init_time_step").unwrap().parse().unwrap();

    let got_step = init_time_step(steps as usize, Some(strength));
    assert_eq!(got_step, want_step, "init_time_step");

    // Z-Image-Turbo uses the static shift=3.0 schedule (sc-2536), not the empirical per-step mu.
    let sched = FlowMatchEuler::for_static_shift(steps as usize, 3.0);
    let mine = Array::from_slice(&sched.sigmas, &[sched.sigmas.len() as i32]);
    let (peak, _) = rel_errors(&mine, g.require("sigmas").unwrap());
    println!("img2img schedule: init_step={got_step} sigmas peak-rel={peak:.2e}");
    assert!(
        peak < 1e-4,
        "static-shift sigmas diverge from the fork: {peak:.2e}"
    );
}

/// Gate B: `preprocess_init_image` (PIL-LANCZOS scale → [-1,1] NCHW) vs the fork's
/// `ImageUtil.to_array(scale_to_dimensions(...))`. LANCZOS matches PIL to ~1/255, so this is a
/// near-bit comparison (the image lives in [-1,1]; 1/255 ≈ 0.008 per channel).
#[test]
#[ignore = "needs real Z-Image weights + local img2img golden"]
fn img2img_preprocess_matches_fork() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let (w, h) = (meta_u32(&g, "w"), meta_u32(&g, "h"));
    let pre = preprocess_init_image(&init_image(&g), w, h).unwrap();
    let want = g.require("image_nchw").unwrap();
    assert_eq!(pre.shape(), want.shape(), "preprocessed image shape");
    let (peak, mean) = rel_errors(&pre, want);
    println!("img2img preprocess: peak-rel {peak:.3e}  mean-rel {mean:.3e}");
    assert!(
        mean < 5e-3,
        "preprocess mean-rel {mean:.3e} (LANCZOS drift)"
    );
}

/// Gate C: the VAE **encoder** path — `encode_init_latents` (preprocess → encode → pack) vs the
/// fork's packed `clean` latents. This is the core new module for sc-2533.
#[test]
#[ignore = "needs real Z-Image weights + local img2img golden"]
fn img2img_encode_matches_fork() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let (w, h) = (meta_u32(&g, "w"), meta_u32(&g, "h"));
    let vae = load_vae(&snapshot()).unwrap();
    let clean = encode_init_latents(&vae, &init_image(&g), w, h).unwrap();
    let want = g.require("clean").unwrap();
    assert_eq!(clean.shape(), want.shape(), "clean latents shape");
    let (peak, mean) = rel_errors(&clean, want);
    println!("img2img encode (clean latents): peak-rel {peak:.3e}  mean-rel {mean:.3e}");
    assert!(mean < 3e-2, "VAE-encoder clean latents mean-rel {mean:.3e}");
}

/// Gate D: the noise blend `(1-σ)·clean + σ·noise` at σ = sigmas[init_step]. Blends the **fork's**
/// clean latents with the Rust seeded noise to isolate the blend + seeded-RNG parity from the
/// encoder error measured in Gate C.
#[test]
#[ignore = "needs real Z-Image weights + local img2img golden"]
fn img2img_blend_matches_fork() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let (steps, w, h) = (meta_u32(&g, "steps"), meta_u32(&g, "w"), meta_u32(&g, "h"));
    let seed: u64 = g.metadata("seed").unwrap().parse().unwrap();
    let init_step: usize = g.metadata("init_time_step").unwrap().parse().unwrap();

    let sched = FlowMatchEuler::for_static_shift(steps as usize, 3.0);
    let sigma = sched.sigmas[init_step];
    let noise = create_noise(seed, w, h)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let init = add_noise_by_interpolation(g.require("clean").unwrap(), &noise, sigma).unwrap();
    let want = g.require("init_latents").unwrap();
    assert_eq!(init.shape(), want.shape(), "init latents shape");
    let (peak, mean) = rel_errors(&init, want);
    println!("img2img blend (init latents): σ={sigma:.5} peak-rel {peak:.3e}  mean-rel {mean:.3e}");
    assert!(mean < 1e-2, "init-latent blend mean-rel {mean:.3e}");
}

/// The integration proof: the full img2img pipeline through the **public** Generator API with a
/// `Conditioning::Reference`, compared to the fork's img2img golden render.
#[test]
#[ignore = "needs real Z-Image weights + local img2img golden"]
fn img2img_full_pipeline_matches_fork() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let (steps, w, h) = (meta_u32(&g, "steps"), meta_u32(&g, "w"), meta_u32(&g, "h"));
    let seed: u64 = g.metadata("seed").unwrap().parse().unwrap();
    let strength: f32 = g.metadata("strength").unwrap().parse().unwrap();
    let prompt = g.metadata("prompt").unwrap().to_string();
    let init_step: usize = g.metadata("init_time_step").unwrap().parse().unwrap();
    let expected_progress = steps - init_step as u32;

    let spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    let generator = mlx_gen::load("z_image_turbo", &spec).unwrap();
    let req = GenerationRequest {
        prompt,
        width: w,
        height: h,
        seed: Some(seed),
        steps: Some(steps),
        conditioning: vec![Conditioning::Reference {
            image: init_image(&g),
            strength: Some(strength),
        }],
        ..Default::default()
    };
    let mut last_step = 0u32;
    let out = generator
        .generate(&req, &mut |p| {
            if let Progress::Step { current, total } = p {
                assert_eq!(
                    total, expected_progress,
                    "img2img runs steps-init_step iterations"
                );
                last_step = last_step.max(current);
            }
        })
        .unwrap();
    assert_eq!(last_step, expected_progress, "progress event count");

    let img = match out {
        GenerationOutput::Images(mut v) => {
            assert_eq!(v.len(), 1);
            v.pop().unwrap()
        }
        other => panic!("expected Images, got {other:?}"),
    };
    assert_eq!((img.width, img.height), (w, h), "image size");

    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tools/golden/rust_z_image_img2img.png");
    image::save_buffer(
        &out_path,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();

    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let differ = img
        .pixels
        .iter()
        .zip(&gimg.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count();
    let frac = differ as f32 / img.pixels.len() as f32;
    println!(
        "✓ img2img public generate: {}x{}; {differ}/{} px differ by >8 ({:.3}%); saved {}",
        img.width,
        img.height,
        img.pixels.len(),
        frac * 100.0,
        out_path.display()
    );
    assert!(
        frac < 0.05,
        "img2img image diverges from the fork: {differ} px (>8)"
    );
}
