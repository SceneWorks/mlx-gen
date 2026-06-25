//! sc-7846 e2e: the **integrated** PiD decode path for Boogu — load the real Boogu **Turbo** turnkey
//! with a PiD decoder overlay (`LoadSpec::with_pid`) and run `Generator::generate` once for the VAE
//! baseline and once with `use_pid`, proving the live denoised latent routes through the pipeline's
//! shared `decode_latents`/`LatentDecoder` seam into a 4× super-resolved PiD image. Boogu's VAE is the
//! FLUX.1 16-ch `AutoencoderKL` (the shared `mlx_gen_z_image::vae::Vae`), so it reuses the `flux` PiD
//! student — and, like Z-Image, its latents come from a different upstream transformer than FLUX.1, so
//! this is a per-model leg of the `flux`-backbone gate. Uses the few-step CFG-free Turbo variant to
//! bound denoise cost (the decode is what's under test).
//!
//! `#[ignore]`d — needs the Boogu Turbo snapshot (env `BOOGU_TURBO_DIR`, else the published
//! `SceneWorks/boogu-image-mlx/turbo` in the HF cache), the converted flux PiD checkpoint (env
//! `PID_FLUX_SAFETENSORS`, else `tools/golden/pid/flux_2k.safetensors`), and a `gemma-2-2b-it`
//! snapshot dir (env `PID_GEMMA_DIR`, else the HF cache).
//!
//! ```sh
//! cargo test -p mlx-gen-boogu --release --test pid_decode_real_weights -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::{GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource};
use mlx_gen_boogu::load_turbo;

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name).ok().map(PathBuf::from)
}

fn boogu_turbo_dir() -> PathBuf {
    if let Some(p) = env_path("BOOGU_TURBO_DIR") {
        return p;
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--SceneWorks--boogu-image-mlx/snapshots");
    let snap = std::fs::read_dir(&snaps)
        .expect("boogu turnkey HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a boogu-image-mlx snapshot dir");
    // The published turnkey carries one `from_snapshot`-loadable root per variant.
    snap.join("turbo")
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
#[ignore = "needs the Boogu Turbo snapshot + converted flux PiD checkpoint + gemma-2-2b-it"]
fn boogu_turbo_pid_decode_vs_vae() {
    let size: u32 = std::env::var("BOOGU_PID_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);

    let spec = LoadSpec::new(WeightsSource::Dir(boogu_turbo_dir())).with_pid(
        WeightsSource::File(pid_checkpoint()),
        WeightsSource::Dir(gemma_dir()),
    );

    eprintln!("loading Boogu Turbo (+PiD overlay), size={size} ...");
    let t = Instant::now();
    let model = load_turbo(&spec).expect("load Boogu Turbo + PiD");
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

    assert_eq!(pid_img.width, size * 4, "PiD width == 4× native");
    assert_eq!(pid_img.height, size * 4, "PiD height == 4× native");
    assert!(phi as i32 - plo as i32 > 40, "PiD output near-flat");

    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../tools/golden/pid");
    let _ = std::fs::create_dir_all(dir);
    save_png(&vae_img, &format!("{dir}/boogu_vae_{}.png", vae_img.width));
    save_png(&pid_img, &format!("{dir}/boogu_pid_{}.png", pid_img.width));
    eprintln!(
        "wrote {dir}/boogu_vae_{}.png + boogu_pid_{}.png  (PiD {:.1}× slower than VAE)",
        vae_img.width,
        pid_img.width,
        pid_dt / vae_dt.max(1e-3)
    );
}
