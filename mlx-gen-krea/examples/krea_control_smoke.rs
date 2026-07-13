//! Real-weight smoke for the Krea 2 pose-ControlNet path through the PRODUCTION registry seam
//! (sc-8465, epic 8459 S5). Loads the dense bf16 Krea 2 Turbo base + the converted MLX pose overlay via
//! `gen_core::registry::load("krea_2_turbo_control", …)` (the exact seam the SceneWorks worker's
//! `start_cached_gen_stream` uses), then renders the SAME pose skeleton at `control_scale = 0.6`
//! (pose-locked) and `control_scale = 0.0` (base passthrough) for an A/B. This is a MANUAL on-Metal
//! validation (a ~12B model), NOT a CI test.
//!
//! Paths default to the local HF cache / repo; override via env (`KREA_CTRL_BASE`, `KREA_CTRL_OVERLAY`,
//! `KREA_CTRL_POSE`, `KREA_CTRL_PROMPT`, `KREA_CTRL_OUT_DIR`, `KREA_CTRL_STEPS`, `KREA_CTRL_SIZE`,
//! `KREA_CTRL_SCALE`, `KREA_CTRL_SEED`).
//!
//! Run: `cargo run --release --example krea_control_smoke -p mlx-gen-krea`

use std::path::PathBuf;

use mlx_gen::gen_core::{
    CancelFlag, Conditioning, ControlKind, GenerationOutput, GenerationRequest, LoadSpec,
    WeightsSource,
};
use mlx_gen::media::Image;

// Force-link the crate so its `register_generators!` `inventory::submit!` for `krea_2_turbo_control`
// survives linker GC and `gen_core::registry::load` can resolve it — the worker's `use mlx_gen_krea
// as _;` anchor idiom (image_jobs.rs), reproduced here because this example references no other
// mlx_gen_krea symbol.
use mlx_gen_krea as _;

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// The env var if set, else a default resolved from the local HF cache (F-080: no hardcoded personal
/// `/Users/...` paths). See [`hf_snapshot`].
fn env_or_hf(key: &str, repo: &str, rel: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| hf_snapshot(repo, rel))
}

/// Resolve `rel` inside a snapshot of the HF-cache repo `repo`, derived from `$HF_HOME` (or
/// `$HOME/.cache/huggingface`) rather than a baked-in personal path. Best-effort: if the repo isn't
/// cached the constructed path simply won't exist and the caller's load errors clearly.
fn hf_snapshot(repo: &str, rel: &str) -> String {
    let snapshots = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/huggingface")
        })
        .join("hub")
        .join(repo)
        .join("snapshots");
    let snap = std::fs::read_dir(&snapshots)
        .ok()
        .and_then(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| p.is_dir())
        })
        .unwrap_or_else(|| snapshots.join("<snapshot>"));
    if rel.is_empty() {
        snap.to_string_lossy().into_owned()
    } else {
        snap.join(rel).to_string_lossy().into_owned()
    }
}

fn load_rgb(path: &str) -> Image {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("open pose skeleton {path}: {e}"))
        .to_rgb8();
    let (width, height) = img.dimensions();
    Image {
        width,
        height,
        pixels: img.into_raw(),
    }
}

fn save_png(img: &Image, path: &str) {
    let buf: image::RgbImage =
        image::ImageBuffer::from_raw(img.width, img.height, img.pixels.clone())
            .expect("output image buffer");
    buf.save(path)
        .unwrap_or_else(|e| panic!("save {path}: {e}"));
}

fn main() {
    let base = env_or_hf(
        "KREA_CTRL_BASE",
        "models--SceneWorks--krea-2-turbo-mlx",
        "bf16",
    );
    // The candle pose overlay, loaded DIRECTLY (RmsScale accepts the candle `*.weight_p1` convention —
    // no convert step). Defaults to the cached hosted checkpoint.
    let overlay = env_or_hf(
        "KREA_CTRL_OVERLAY",
        "models--SceneWorks--krea2-pose-controlnet-beta",
        "control_step5000.safetensors",
    );
    // The pose skeleton is NOT an HF-cache asset; default to a repo-relative path and require the env
    // override for anything else (no hardcoded personal `/Users/...` path).
    let pose = env_or("KREA_CTRL_POSE", "poses/tpose_01.png");
    let prompt = env_or(
        "KREA_CTRL_PROMPT",
        "a full-body studio photo of a person standing, plain grey background",
    );
    let out_dir = env_or("KREA_CTRL_OUT_DIR", "/tmp");
    let steps: u32 = env_or("KREA_CTRL_STEPS", "8").parse().expect("steps");
    let size: u32 = env_or("KREA_CTRL_SIZE", "512").parse().expect("size");
    let scale: f32 = env_or("KREA_CTRL_SCALE", "0.6").parse().expect("scale");
    let seed: u64 = env_or("KREA_CTRL_SEED", "1234").parse().expect("seed");

    // The worker's exact load seam: dense base dir + the overlay as the required control checkpoint,
    // resolved through the registry by engine id.
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&base)))
        .with_control(WeightsSource::File(PathBuf::from(&overlay)));
    eprintln!("[smoke] loading krea_2_turbo_control: base {base}");
    eprintln!("[smoke] overlay {overlay}");
    let generator = mlx_gen::gen_core::registry::load("krea_2_turbo_control", &spec)
        .expect("load krea_2_turbo_control generator");

    let skeleton = load_rgb(&pose);
    eprintln!(
        "[smoke] pose {}x{} → {size}x{size}, prompt '{prompt}' ({steps} steps, seed {seed})",
        skeleton.width, skeleton.height
    );

    // A/B: pose-locked (scale) vs base passthrough (0.0) — same prompt + seed, so only the control
    // residual differs. scale=0.0 must be the un-conditioned base image; scale>0 must follow the pose.
    for s in [scale, 0.0f32] {
        let request = GenerationRequest {
            prompt: prompt.clone(),
            width: size,
            height: size,
            count: 1,
            seed: Some(seed),
            steps: Some(steps),
            conditioning: vec![Conditioning::Control {
                image: skeleton.clone(),
                kind: ControlKind::Pose,
                scale: Some(s),
            }],
            cancel: CancelFlag::new(),
            ..Default::default()
        };
        let output = generator
            .generate(&request, &mut |_| {})
            .unwrap_or_else(|e| panic!("generate at scale {s}: {e}"));
        let img = match output {
            GenerationOutput::Images(mut images) => images.pop().expect("one image"),
            _ => panic!("control generator returned non-image output"),
        };
        let mn = *img.pixels.iter().min().unwrap();
        let mx = *img.pixels.iter().max().unwrap();
        let mean: f64 = img.pixels.iter().map(|&p| p as f64).sum::<f64>() / img.pixels.len() as f64;
        assert!(mx > mn, "degenerate (constant) output at scale {s}");
        let out = format!("{out_dir}/krea_control_s{s}.png");
        save_png(&img, &out);
        eprintln!(
            "[smoke] scale {s}: {}x{} px range [{mn},{mx}] mean {mean:.1} → {out}",
            img.width, img.height
        );
    }
}
