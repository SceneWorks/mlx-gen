//! sc-7846 e2e: the **integrated** PiD decode path for Z-Image-Turbo — load the real
//! `Tongyi-MAI/Z-Image-Turbo` model with a PiD decoder overlay (`LoadSpec::with_pid`) and run
//! `Generator::generate` once for the VAE baseline and once with `use_pid`, proving the live denoised
//! latent routes through `render_batch`'s shared `LatentDecoder` seam into a 4× super-resolved PiD
//! image. This is the highest-risk leg of the `flux`-backbone gate: Z-Image ships Flux1-dev's 16-ch
//! VAE (so it reuses the `flux` PiD student via the `zimage-turbo` alias), but its latents come from a
//! *different* upstream transformer than FLUX.1, so a flux-PiD "go" on FLUX.1 does not automatically
//! transfer here — this test is the per-model check.
//!
//! `#[ignore]`d — needs the `Tongyi-MAI/Z-Image-Turbo` snapshot (env `ZIMAGE_SNAPSHOT`, else the HF
//! cache), the converted flux PiD checkpoint (env `PID_FLUX_SAFETENSORS`, else
//! `tools/golden/pid/flux_2k.safetensors`), and a `gemma-2-2b-it` snapshot dir (env `PID_GEMMA_DIR`,
//! else the HF cache).
//!
//! ```sh
//! cargo test -p mlx-gen-z-image --release --test pid_decode_real_weights -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::{GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource};
use mlx_gen_z_image::load;

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name).ok().map(PathBuf::from)
}

fn first_snapshot_dir(repo: &str, what: &str) -> PathBuf {
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(repo)
        .join("snapshots");
    std::fs::read_dir(&snaps)
        .unwrap_or_else(|_| panic!("{what} HF cache snapshots dir: {}", snaps.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .unwrap_or_else(|| panic!("a {what} snapshot dir under {}", snaps.display()))
}

fn zimage_snapshot() -> PathBuf {
    env_path("ZIMAGE_SNAPSHOT")
        .unwrap_or_else(|| first_snapshot_dir("models--Tongyi-MAI--Z-Image-Turbo", "Z-Image-Turbo"))
}

fn pid_checkpoint() -> PathBuf {
    env_path("PID_FLUX_SAFETENSORS").unwrap_or_else(|| {
        PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tools/golden/pid/flux_2k.safetensors"
        ))
    })
}

fn gemma_dir() -> PathBuf {
    env_path("PID_GEMMA_DIR").unwrap_or_else(|| {
        first_snapshot_dir(
            "models--Efficient-Large-Model--gemma-2-2b-it",
            "gemma-2-2b-it",
        )
    })
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
#[ignore = "needs the Z-Image-Turbo snapshot + converted flux PiD checkpoint + gemma-2-2b-it"]
fn z_image_turbo_pid_decode_vs_vae() {
    let size: u32 = std::env::var("ZIMAGE_PID_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);

    let spec = LoadSpec::new(WeightsSource::Dir(zimage_snapshot())).with_pid(
        WeightsSource::File(pid_checkpoint()),
        WeightsSource::Dir(gemma_dir()),
    );

    eprintln!("loading Z-Image-Turbo (+PiD overlay), size={size} ...");
    let t = Instant::now();
    let model = load(&spec).expect("load Z-Image-Turbo + PiD");
    eprintln!("loaded in {:.1}s", t.elapsed().as_secs_f32());

    let base = GenerationRequest {
        prompt: "a red fox sitting in a snowy pine forest at dawn, photorealistic".into(),
        width: size,
        height: size,
        count: 1,
        seed: Some(7),
        ..Default::default()
    };

    // --- VAE baseline ---
    let t = Instant::now();
    let vae_img = match model.generate(&base, &mut |_| {}).expect("vae generate") {
        GenerationOutput::Images(v) => v.into_iter().next().unwrap(),
        _ => panic!("expected images"),
    };
    let vae_dt = t.elapsed().as_secs_f32();
    let (vlo, vhi, vmu) = stats(&vae_img);
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
    let pid_img = match model.generate(&pid_req, &mut |_| {}).expect("pid generate") {
        GenerationOutput::Images(v) => v.into_iter().next().unwrap(),
        _ => panic!("expected images"),
    };
    let pid_dt = t.elapsed().as_secs_f32();
    let (plo, phi, pmu) = stats(&pid_img);
    eprintln!(
        "PiD: {}x{} in {pid_dt:.2}s  range [{plo},{phi}] mean {pmu:.1}",
        pid_img.width, pid_img.height
    );

    // PiD super-resolves 4× (the flux student); the seam takes the output size from the decoded
    // tensor, so the image must be 4× the VAE-native side.
    assert_eq!(pid_img.width, size * 4, "PiD width == 4× native");
    assert_eq!(pid_img.height, size * 4, "PiD height == 4× native");
    assert!(phi as i32 - plo as i32 > 40, "PiD output near-flat");

    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../tools/golden/pid");
    let _ = std::fs::create_dir_all(dir);
    save_png(&vae_img, &format!("{dir}/zimage_vae_{}.png", vae_img.width));
    save_png(&pid_img, &format!("{dir}/zimage_pid_{}.png", pid_img.width));
    eprintln!(
        "wrote {dir}/zimage_vae_{}.png + zimage_pid_{}.png  (PiD {:.1}× slower than VAE)",
        vae_img.width,
        pid_img.width,
        pid_dt / vae_dt.max(1e-3)
    );
}

#[test]
#[ignore = "needs the Z-Image-Turbo snapshot (no PiD weights) — proves the error path"]
fn use_pid_without_loaded_pid_errors() {
    // Loading WITHOUT spec.pid, then requesting use_pid, must error clearly (not silently VAE-decode).
    let spec = LoadSpec::new(WeightsSource::Dir(zimage_snapshot()));
    let model = load(&spec).expect("load Z-Image-Turbo");
    let req = GenerationRequest {
        prompt: "a fox".into(),
        width: 512,
        height: 512,
        seed: Some(1),
        use_pid: true,
        ..Default::default()
    };
    let err = model
        .generate(&req, &mut |_| {})
        .expect_err("use_pid without loaded PiD must error")
        .to_string();
    assert!(err.contains("no PiD decoder is loaded"), "got: {err}");
}
