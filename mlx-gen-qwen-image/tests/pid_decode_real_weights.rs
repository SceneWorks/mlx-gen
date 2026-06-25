//! sc-7845 e2e: the **integrated** PiD decode path — load the real Qwen-Image model with a PiD
//! decoder overlay (`LoadSpec::with_pid`) and run `Generator::generate` twice (VAE baseline +
//! `use_pid`), proving the live denoised latent routes through the shared `decode_and_collect` seam
//! into a 4× super-resolved PiD image. This is the wiring this story adds; the PiD decode itself was
//! already real-weight validated against the CUDA reference in sc-7843 (`from_clean`/`from_ldm`).
//!
//! `#[ignore]`d — needs the real `Qwen/Qwen-Image` snapshot (env `QWEN_IMAGE_SNAPSHOT`, else the HF
//! cache), the converted PiD checkpoint (env `PID_QWEN_SAFETENSORS`, else
//! `tools/golden/pid/qwenimage_2kto4k.safetensors`), and a `gemma-2-2b-it` snapshot dir (env
//! `PID_GEMMA_DIR`, else the HF cache). Loads the full Qwen model **plus** PiD net + Gemma, so it is
//! memory-heavy; defaults to Q8 + 512² (→ 2048² PiD) + the few-step Lightning path to bound cost.
//!
//! ```sh
//! cargo test -p mlx-gen-qwen-image --release --test pid_decode_real_weights -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::{GenerationOutput, GenerationRequest, Image, LoadSpec, Quant, WeightsSource};
use mlx_gen_qwen_image::load;

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name).ok().map(PathBuf::from)
}

fn qwen_snapshot() -> PathBuf {
    if let Some(p) = env_path("QWEN_IMAGE_SNAPSHOT") {
        return p;
    }
    let home = std::env::var("HOME").unwrap();
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a Qwen-Image snapshot dir")
}

fn pid_checkpoint() -> PathBuf {
    env_path("PID_QWEN_SAFETENSORS").unwrap_or_else(|| {
        PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tools/golden/pid/qwenimage_2kto4k.safetensors"
        ))
    })
}

fn gemma_dir() -> PathBuf {
    if let Some(p) = env_path("PID_GEMMA_DIR") {
        return p;
    }
    let home = std::env::var("HOME").unwrap();
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Efficient-Large-Model--gemma-2-2b-it/snapshots");
    std::fs::read_dir(&base)
        .expect("gemma HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a gemma-2-2b-it snapshot dir")
}

fn quant_from_env() -> Option<Quant> {
    match std::env::var("QWEN_PID_QUANT").as_deref() {
        Ok("none") => None,
        Ok("q4") => Some(Quant::Q4),
        // Default Q8 to bound memory (full Qwen + PiD net + Gemma coexist in one process).
        _ => Some(Quant::Q8),
    }
}

fn size_from_env() -> u32 {
    std::env::var("QWEN_PID_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512)
}

fn stats(img: &Image) -> (u8, u8, f64) {
    let (mut lo, mut hi) = (255u8, 0u8);
    let mut sum = 0u64;
    for &p in &img.pixels {
        lo = lo.min(p);
        hi = hi.max(p);
        sum += p as u64;
    }
    (lo, hi, sum as f64 / img.pixels.len() as f64)
}

fn save_png(img: &Image, path: &str) {
    image::save_buffer(
        path,
        &img.pixels,
        img.width,
        img.height,
        image::ColorType::Rgb8,
    )
    .unwrap();
}

#[test]
#[ignore = "needs the Qwen-Image snapshot + converted PiD checkpoint + gemma-2-2b-it"]
fn qwen_image_pid_decode_vs_vae() {
    let size = size_from_env();
    let quant = quant_from_env();
    let mut spec = LoadSpec::new(WeightsSource::Dir(qwen_snapshot()));
    if let Some(q) = quant {
        spec = spec.with_quant(q);
    }
    spec = spec.with_pid(
        WeightsSource::File(pid_checkpoint()),
        WeightsSource::Dir(gemma_dir()),
    );

    eprintln!("loading Qwen-Image (+PiD overlay), quant={quant:?} size={size} ...");
    let t = Instant::now();
    let model = load(&spec).expect("load Qwen-Image + PiD");
    eprintln!("loaded in {:.1}s", t.elapsed().as_secs_f32());

    let base = GenerationRequest {
        prompt: "a mountain valley landscape at golden hour with a winding river and pine forest"
            .into(),
        width: size,
        height: size,
        count: 1,
        seed: Some(7),
        // Few-step CFG-off Lightning path keeps the denoise cheap; the decode is what we're testing.
        sampler: Some("lightning".into()),
        steps: Some(8),
        ..Default::default()
    };

    // --- VAE baseline ---
    let t = Instant::now();
    let vae_out = match model.generate(&base, &mut |_| {}).expect("vae generate") {
        GenerationOutput::Images(v) => v,
        _ => panic!("expected images"),
    };
    let vae_dt = t.elapsed().as_secs_f32();
    let vae_img = &vae_out[0];
    let (vlo, vhi, vmu) = stats(vae_img);
    eprintln!(
        "VAE: {}x{} in {vae_dt:.2}s  range [{vlo},{vhi}] mean {vmu:.1}",
        vae_img.width, vae_img.height
    );
    assert_eq!(vae_img.width, size, "VAE width == native");

    // --- PiD path (same request + use_pid) ---
    let pid_req = GenerationRequest {
        use_pid: true,
        ..base.clone()
    };
    let t = Instant::now();
    let pid_out = match model.generate(&pid_req, &mut |_| {}).expect("pid generate") {
        GenerationOutput::Images(v) => v,
        _ => panic!("expected images"),
    };
    let pid_dt = t.elapsed().as_secs_f32();
    let pid_img = &pid_out[0];
    let (plo, phi, pmu) = stats(pid_img);
    eprintln!(
        "PiD: {}x{} in {pid_dt:.2}s  range [{plo},{phi}] mean {pmu:.1}",
        pid_img.width, pid_img.height
    );

    // PiD super-resolves 4× (the released qwenimage student); the seam takes the output size from the
    // decoded tensor, so the image must be 4× the VAE-native side.
    assert_eq!(pid_img.width, size * 4, "PiD width == 4× native");
    assert_eq!(pid_img.height, size * 4, "PiD height == 4× native");
    // Non-degenerate, full-color (not the grayscale/flat failure mode).
    assert!(phi as i32 - plo as i32 > 40, "PiD output near-flat");

    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../tools/golden/pid");
    let _ = std::fs::create_dir_all(dir);
    save_png(vae_img, &format!("{dir}/qwen_vae_{}.png", vae_img.width));
    save_png(pid_img, &format!("{dir}/qwen_pid_{}.png", pid_img.width));
    eprintln!(
        "wrote {dir}/qwen_vae_{}.png + qwen_pid_{}.png  (PiD {:.1}× slower than VAE)",
        vae_img.width,
        pid_img.width,
        pid_dt / vae_dt.max(1e-3)
    );
}

#[test]
#[ignore = "needs the Qwen-Image snapshot (no PiD weights) — proves the error path"]
fn use_pid_without_loaded_pid_errors() {
    // Loading WITHOUT spec.pid, then requesting use_pid, must error clearly (not silently VAE-decode).
    let spec = LoadSpec::new(WeightsSource::Dir(qwen_snapshot())).with_quant(Quant::Q8);
    let model = load(&spec).expect("load Qwen-Image");
    let req = GenerationRequest {
        prompt: "a fox".into(),
        width: 512,
        height: 512,
        seed: Some(1),
        sampler: Some("lightning".into()),
        steps: Some(8),
        use_pid: true,
        ..Default::default()
    };
    let err = model
        .generate(&req, &mut |_| {})
        .expect_err("use_pid without loaded PiD must error")
        .to_string();
    assert!(err.contains("no PiD decoder is loaded"), "got: {err}");
}
