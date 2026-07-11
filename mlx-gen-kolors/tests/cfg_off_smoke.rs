//! sc-9091 (F-005) — Kolors CFG-off (`guidance <= 1.0`) end-to-end smoke.
//!
//! The documented CFG-off path (`cfg <= 1` disables guidance, per the struct-API docs + capabilities)
//! never worked on any Kolors mode: every denoise assembly unconditionally built a B=2 `[pos, neg]`
//! conditioning batch, but the shared `mlx_gen_sdxl::denoise_core` only CFG-batches the latents to
//! B=2 when `cfg > 1.0`. So a `guidance: Some(1.0)` request fed the U-Net B=1 latents with B=2
//! conditioning and the attention reshape failed mid-denoise with an opaque MLX element-count error.
//!
//! The **default-run** proof of the fix lives in `model.rs`'s `cfg_conditioning_*` unit tests (they
//! assert the assembled batch dims are B=1 under CFG-off and B=2 under CFG-on, and that CFG-off needs
//! no negative). This file is the end-to-end complement: it drives the REAL registry `generate()` at
//! `guidance: Some(1.0)` for the base txt2img path **and** one conditioned mode (ControlNet-pose),
//! proving the whole denoise runs to a coherent image instead of erroring mid-loop. It is `#[ignore]`d
//! because — like every other Kolors integration test — it needs the real `Kwai-Kolors/Kolors-diffusers`
//! snapshot (and the ControlNet-Pose snapshot for the conditioned case); there is no synthetic
//! ChatGLM3-6B + SDXL U-Net + VAE fixture to run it default-green.
//!
//!   KOLORS_SNAPSHOT=<Kolors-diffusers dir> [KOLORS_CONTROLNET=<Kolors-ControlNet-Pose dir>] \
//!     cargo test -p mlx-gen-kolors --release --test cfg_off_smoke -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::{
    Conditioning, ControlKind, GenerationOutput, GenerationRequest, Image, LoadSpec, Precision,
    WeightsSource,
};
// Force-link the provider crate so its `inventory::submit!` registration is in this test binary.
use mlx_gen_kolors::MODEL_ID;

const SIZE: u32 = 512;
const STEPS: u32 = 6;
const SEED: u64 = 9091;
const PROMPT: &str = "a fox sitting in a sunlit forest, photorealistic, sharp focus";

fn snap_env(env: &str, repo: &str) -> Option<PathBuf> {
    if let Ok(p) = std::env::var(env) {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    let base = PathBuf::from(home).join(format!(".cache/huggingface/hub/{repo}/snapshots"));
    std::fs::read_dir(&base)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
}

fn base_snap() -> Option<PathBuf> {
    snap_env("KOLORS_SNAPSHOT", "models--Kwai-Kolors--Kolors-diffusers")
}
fn cn_snap() -> Option<PathBuf> {
    snap_env(
        "KOLORS_CONTROLNET",
        "models--Kwai-Kolors--Kolors-ControlNet-Pose",
    )
}

/// A deterministic synthetic pose/control image (no external asset needed — the smoke exercises the
/// CFG-off *wiring*, not pose adherence).
fn synthetic_image() -> Image {
    let (h, w) = (SIZE as usize, SIZE as usize);
    let mut px = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            px[i] = (x * 255 / (w - 1)) as u8;
            px[i + 1] = (y * 255 / (h - 1)) as u8;
            px[i + 2] = ((x ^ y) % 256) as u8;
        }
    }
    Image {
        width: w as u32,
        height: h as u32,
        pixels: px,
    }
}

fn spec(base: PathBuf, control: Option<PathBuf>) -> LoadSpec {
    LoadSpec {
        weights: WeightsSource::Dir(base),
        quantize: None,
        precision: Precision::Bf16,
        control: control.map(WeightsSource::Dir),
        ip_adapter: None,
        adapters: Vec::new(),
        extra_controls: Vec::new(),
        pid: None,
        identity: None,
        text_encoder: None,
        offload_policy: Default::default(),
    }
}

/// A `guidance: Some(1.0)` request (CFG off) with the given conditioning. A negative prompt IS set,
/// to prove it is *ignored* (never encoded) under CFG-off rather than required.
fn cfg_off_req(conditioning: Vec<Conditioning>) -> GenerationRequest {
    GenerationRequest {
        prompt: PROMPT.into(),
        negative_prompt: Some("lowres, blurry, deformed".into()),
        width: SIZE,
        height: SIZE,
        count: 1,
        steps: Some(STEPS),
        guidance: Some(1.0), // CFG off — the path F-005 says never worked.
        seed: Some(SEED),
        conditioning,
        ..Default::default()
    }
}

fn assert_coherent(img: &Image, label: &str) {
    assert_eq!(
        img.pixels.len(),
        (img.width * img.height * 3) as usize,
        "{label}: pixel buffer is RGB8 HWC"
    );
    let n = img.pixels.len() as f64;
    let mean = img.pixels.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = img
        .pixels
        .iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    let std = var.sqrt();
    let mut seen = [false; 256];
    for &v in &img.pixels {
        seen[v as usize] = true;
    }
    let distinct = seen.iter().filter(|&&b| b).count();
    println!("[cfg-off] {label:<24} std={std:.1} distinct={distinct}");
    assert!(std > 10.0, "{label}: image is near-flat (std {std:.1})");
    assert!(
        distinct > 24,
        "{label}: too few distinct levels ({distinct})"
    );
}

/// Base txt2img at `guidance = 1.0`: the CFG-off path must render a coherent image, NOT error mid
/// denoise on the B=2-vs-B=1 reshape (F-005).
#[test]
#[ignore = "needs the real Kwai-Kolors/Kolors-diffusers snapshot (set KOLORS_SNAPSHOT)"]
fn base_txt2img_cfg_off_runs() {
    let Some(base) = base_snap() else {
        eprintln!("skipping: no Kolors snapshot (set KOLORS_SNAPSHOT)");
        return;
    };
    let generator = mlx_gen::load(MODEL_ID, &spec(base, None)).expect("load kolors");
    let out = generator
        .generate(&cfg_off_req(vec![]), &mut |_| {})
        .expect("CFG-off base txt2img must not error mid-denoise (F-005)");
    match out {
        GenerationOutput::Images(mut v) => assert_coherent(&v.pop().unwrap(), "base txt2img"),
        other => panic!("expected Images, got {other:?}"),
    }
}

/// ControlNet-pose (one conditioned mode) at `guidance = 1.0`: the conditioned denoise assembly must
/// likewise build B=1 conditioning + control batch and run to a coherent image.
#[test]
#[ignore = "needs Kwai-Kolors/Kolors-diffusers + Kolors-ControlNet-Pose (KOLORS_SNAPSHOT, KOLORS_CONTROLNET)"]
fn controlnet_pose_cfg_off_runs() {
    let (Some(base), Some(cn)) = (base_snap(), cn_snap()) else {
        eprintln!("skipping: need KOLORS_SNAPSHOT + KOLORS_CONTROLNET");
        return;
    };
    let generator =
        mlx_gen::load(MODEL_ID, &spec(base, Some(cn))).expect("load kolors + controlnet");
    let req = cfg_off_req(vec![Conditioning::Control {
        image: synthetic_image(),
        kind: ControlKind::Pose,
        scale: Some(0.7),
    }]);
    let out = generator
        .generate(&req, &mut |_| {})
        .expect("CFG-off ControlNet-pose must not error mid-denoise (F-005)");
    match out {
        GenerationOutput::Images(mut v) => assert_coherent(&v.pop().unwrap(), "controlnet pose"),
        other => panic!("expected Images, got {other:?}"),
    }
}
