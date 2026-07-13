//! Minimal Krea 2 Raw t2i generate — produce a clean, face-detectable portrait to use as the PERSON
//! reference for the epic-10871 P4.2 identity-preservation scoring. NOT a CI test (a 12.9B model on
//! Metal). Env overrides: KREA_SNAPSHOT, KREA_T2I_PROMPT, KREA_T2I_OUT, KREA_T2I_STEPS,
//! KREA_T2I_GUIDANCE, KREA_T2I_SEED, KREA_T2I_W, KREA_T2I_H.
//!
//! Run: `cargo run --release --example krea_t2i -p mlx-gen-krea`

use std::path::PathBuf;

use mlx_gen::gen_core::{CancelFlag, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};
use mlx_gen::media::Image;
use mlx_gen_krea::model::load_raw;

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

fn save_png(img: &Image, path: &str) {
    let buf: image::RgbImage =
        image::ImageBuffer::from_raw(img.width, img.height, img.pixels.clone())
            .expect("output image buffer");
    buf.save(path)
        .unwrap_or_else(|e| panic!("save {path}: {e}"));
}

fn main() {
    let snapshot = env_or_hf("KREA_SNAPSHOT", "models--krea--Krea-2-Raw", "");
    let prompt = env_or(
        "KREA_T2I_PROMPT",
        "a photorealistic head-and-shoulders studio portrait of a woman with wavy auburn hair, green \
         eyes, light freckles, calm neutral expression, looking directly at the camera, soft even \
         lighting, plain light-grey seamless background, sharp focus, 85mm",
    );
    let out_path = env_or("KREA_T2I_OUT", "/tmp/krea_person_ref.png");
    let steps: u32 = env_or("KREA_T2I_STEPS", "30").parse().expect("steps");
    let guidance: f32 = env_or("KREA_T2I_GUIDANCE", "4.0")
        .parse()
        .expect("guidance");
    let seed: u64 = env_or("KREA_T2I_SEED", "7").parse().expect("seed");
    let width: u32 = env_or("KREA_T2I_W", "832").parse().expect("width");
    let height: u32 = env_or("KREA_T2I_H", "1152").parse().expect("height");

    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&snapshot)));
    eprintln!("[t2i] loading krea_2_raw from {snapshot}");
    let generator = load_raw(&spec).expect("load krea_2_raw generator");

    let request = GenerationRequest {
        prompt: prompt.clone(),
        negative_prompt: Some(String::new()),
        width,
        height,
        count: 1,
        seed: Some(seed),
        steps: Some(steps),
        guidance: Some(guidance),
        cancel: CancelFlag::new(),
        ..Default::default()
    };
    eprintln!("[t2i] '{prompt}' ({width}x{height}, {steps} steps, g={guidance}, seed={seed})");
    let output = generator
        .generate(&request, &mut |_| {})
        .expect("t2i generate");
    let out = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("t2i produced one image"),
        _ => panic!("raw generator returned non-image output"),
    };
    let mn = *out.pixels.iter().min().unwrap();
    let mx = *out.pixels.iter().max().unwrap();
    assert!(mx > mn, "degenerate (constant) t2i output");
    save_png(&out, &out_path);
    eprintln!("[t2i] wrote {out_path} ({}x{})", out.width, out.height);
}
