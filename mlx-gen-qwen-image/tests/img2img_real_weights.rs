//! sc-2530: real-weights validation of the Qwen-Image **T2I img2img** port against the frozen fork.
//!
//! `#[ignore]`d — needs the real `Qwen/Qwen-Image` snapshot (env `QWEN_IMAGE_SNAPSHOT`, else the HF
//! cache) and the golden produced by `tools/dump_qwen_image_img2img_golden.py` (gitignored, local).
//! Run with:
//!   cargo test -p mlx-gen-qwen-image --release --test img2img_real_weights -- --ignored --nocapture
//!
//! Stage gates isolate each new piece — the schedule + `init_time_step`, the LANCZOS preprocess, the
//! VAE encode, and the noise blend — then the final gates drive the **public**
//! `load("qwen_image", spec).generate(req)` API with a `Conditioning::Reference` and confirm the
//! rendered image matches the fork's img2img golden, at both bf16 and Q8 (transformer-only quant,
//! the fork's scope — see `e2e_real_weights.rs`).

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::{
    Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, Progress, Quant,
    WeightsSource,
};
use mlx_gen_qwen_image::{
    add_noise_by_interpolation, create_noise, decoded_to_image, encode_init_latents,
    init_time_step, load_vae, preprocess_init_image, qwen_scheduler,
};
use mlx_rs::Array;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/qwen_image_img2img_golden.safetensors"
);
const Q8_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/qwen_image_img2img_q8_golden.safetensors"
);

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("QWEN_IMAGE_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

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

/// Gate A: the Rust `qwen_scheduler` reproduces the fork's `config.scheduler.sigmas` (img2img
/// indexes `sigmas[init_time_step]`, so the schedule must match exactly), and `init_time_step`
/// matches the fork's `Config.init_time_step`.
#[test]
#[ignore = "needs real Qwen-Image weights + local img2img golden"]
fn img2img_schedule_and_init_step_match() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let (steps, w, h) = (meta_u32(&g, "steps"), meta_u32(&g, "w"), meta_u32(&g, "h"));
    let strength: f32 = g.metadata("strength").unwrap().parse().unwrap();
    let want_step: usize = g.metadata("init_time_step").unwrap().parse().unwrap();

    let got_step = init_time_step(steps as usize, Some(strength));
    assert_eq!(got_step, want_step, "init_time_step");

    let sched = qwen_scheduler(steps as usize, w, h);
    let mine = Array::from_slice(&sched.sigmas, &[sched.sigmas.len() as i32]);
    let (peak, _) = rel_errors(&mine, g.require("sigmas").unwrap());
    println!("img2img schedule: init_step={got_step} sigmas peak-rel={peak:.2e}");
    assert!(peak < 1e-4, "qwen sigmas diverge from the fork: {peak:.2e}");
}

/// Gate B: `preprocess_init_image` (PIL-LANCZOS scale → [-1,1] NCHW) vs the fork's
/// `ImageUtil.to_array(scale_to_dimensions(...))`. LANCZOS matches PIL's fixed-point path, so this
/// is a near-bit comparison (the image lives in [-1,1]; 1/255 ≈ 0.008 per channel).
#[test]
#[ignore = "needs real Qwen-Image weights + local img2img golden"]
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

/// Gate C: the VAE encode path — `encode_init_latents` (preprocess → encode → pack) vs the fork's
/// packed `clean` latents. This is the core reuse of the slice-1 VAE encode for sc-2530.
#[test]
#[ignore = "needs real Qwen-Image weights + local img2img golden"]
fn img2img_encode_matches_fork() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let (w, h) = (meta_u32(&g, "w"), meta_u32(&g, "h"));
    let vae = load_vae(&snapshot()).unwrap();
    let clean = encode_init_latents(&vae, &init_image(&g), w, h).unwrap();
    let want = g.require("clean").unwrap();
    assert_eq!(clean.shape(), want.shape(), "clean latents shape");
    let (peak, mean) = rel_errors(&clean, want);
    println!("img2img encode (clean latents): peak-rel {peak:.3e}  mean-rel {mean:.3e}");
    assert!(mean < 3e-2, "VAE-encode clean latents mean-rel {mean:.3e}");
}

/// Gate D: the noise blend `(1-σ)·clean + σ·noise` at σ = sigmas[init_step]. Blends the **fork's**
/// clean latents with the Rust seeded noise (f32, as Qwen keeps it) to isolate the blend + seeded-
/// RNG parity from the encode error measured in Gate C.
#[test]
#[ignore = "needs real Qwen-Image weights + local img2img golden"]
fn img2img_blend_matches_fork() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let (steps, w, h) = (meta_u32(&g, "steps"), meta_u32(&g, "w"), meta_u32(&g, "h"));
    let seed: u64 = g.metadata("seed").unwrap().parse().unwrap();
    let init_step: usize = g.metadata("init_time_step").unwrap().parse().unwrap();

    let sched = qwen_scheduler(steps as usize, w, h);
    let sigma = sched.sigmas[init_step];
    let noise = create_noise(seed, w, h).unwrap(); // f32 (Qwen txt2img/img2img keeps noise f32)
    let init = add_noise_by_interpolation(g.require("clean").unwrap(), &noise, sigma).unwrap();
    let want = g.require("init_latents").unwrap();
    assert_eq!(init.shape(), want.shape(), "init latents shape");
    let (peak, mean) = rel_errors(&init, want);
    println!("img2img blend (init latents): σ={sigma:.5} peak-rel {peak:.3e}  mean-rel {mean:.3e}");
    assert!(mean < 1e-2, "init-latent blend mean-rel {mean:.3e}");
}

/// Drive the full img2img pipeline through the **public** Generator API with a
/// `Conditioning::Reference`, vs the fork's img2img golden render. Shared by the bf16 and Q8 gates.
fn full_pipeline_matches_fork(golden_path: &str, quant: Option<Quant>, max_px_frac: f32) {
    let g = Weights::from_file(golden_path).unwrap();
    let (steps, w, h) = (meta_u32(&g, "steps"), meta_u32(&g, "w"), meta_u32(&g, "h"));
    let seed: u64 = g.metadata("seed").unwrap().parse().unwrap();
    let strength: f32 = g.metadata("strength").unwrap().parse().unwrap();
    let guidance: f32 = g.metadata("guidance").unwrap().parse().unwrap();
    let prompt = g.metadata("prompt").unwrap().to_string();
    let init_step: usize = g.metadata("init_time_step").unwrap().parse().unwrap();
    let expected_progress = steps - init_step as u32;

    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    if let Some(q) = quant {
        spec = spec.with_quant(q);
    }
    let generator = mlx_gen::load("qwen_image", &spec).unwrap();
    let req = GenerationRequest {
        prompt,
        width: w,
        height: h,
        seed: Some(seed),
        steps: Some(steps),
        guidance: Some(guidance),
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

    let tag = quant.map(|_| "Q8 ").unwrap_or("");
    let want_img = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    assert_eq!(img.pixels.len(), want_img.pixels.len());
    let differ = img
        .pixels
        .iter()
        .zip(&want_img.pixels)
        .filter(|(a, b)| (**a as i16 - **b as i16).abs() > 8)
        .count();
    let frac = differ as f32 / img.pixels.len() as f32;
    println!(
        "✓ {tag}img2img public generate: {}x{}; {differ}/{} px differ by >8 ({:.3}%)",
        img.width,
        img.height,
        img.pixels.len(),
        frac * 100.0,
    );
    assert!(
        frac < max_px_frac,
        "img2img image diverges from the fork: {differ} px (>8), {:.3}%",
        frac * 100.0
    );
}

/// The integration proof (bf16): the full img2img pipeline through the public Generator API.
#[test]
#[ignore = "needs real Qwen-Image weights + local img2img golden"]
fn img2img_full_pipeline_matches_fork() {
    full_pipeline_matches_fork(GOLDEN, None, 0.05);
}

/// The Q8 integration proof: same public path with `spec.with_quant(Q8)` (transformer-only, the
/// fork's `quantize=8` scope) vs the fork's Q8 img2img golden.
#[test]
#[ignore = "needs real Qwen-Image weights + local Q8 img2img golden (QUANTIZE=8 dump_qwen_image_img2img_golden.py)"]
fn img2img_q8_full_pipeline_matches_fork() {
    full_pipeline_matches_fork(Q8_GOLDEN, Some(Quant::Q8), 0.05);
}
