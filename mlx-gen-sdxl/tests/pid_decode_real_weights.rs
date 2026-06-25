//! sc-7848 e2e: the **integrated** PiD decode path for the SDXL family — load a real SDXL-family
//! snapshot with a PiD decoder overlay (`LoadSpec::with_pid`) and run `Generator::generate` once for
//! the VAE baseline and once with `use_pid`, proving the live denoised latent routes through the
//! `decode_image` seam into a 4× super-resolved PiD image.
//!
//! SDXL is the **4-ch VAE latent** space (the largest latent→pixel ratio in the catalog) and the lone
//! **variance-preserving-frame** PiD student. The clean σ=0 decode we ship is frame-agnostic
//! (`add_noise(clean, σ=0)` is identity in either frame), so this exercises the same `sdxl` student
//! the registry resolves. The seam hands PiD the 0.13025-normalized latent unchanged (the exact
//! tensor `vae.decode` consumes), transposing NHWC↔NCHW around the student.
//!
//! The `sdxl` generator also backs **RealVisXL** (`realvisxl`, `realvisxl_lightning`) — same VAE,
//! same `decode_image` seam, only different U-Net weights — so pointing `SDXL_DIR` at a RealVisXL
//! snapshot exercises those legs through the identical code path.
//!
//! `#[ignore]`d — needs an SDXL-family snapshot (env `SDXL_DIR`, else the
//! `stabilityai/stable-diffusion-xl-base-1.0` HF cache), the converted SDXL PiD checkpoint (env
//! `PID_SDXL_SAFETENSORS`, else `tools/golden/pid/sdxl_2kto4k.safetensors`), and a `gemma-2-2b-it`
//! snapshot dir (env `PID_GEMMA_DIR`, else the HF cache). Dense fp16 (the SDXL production dtype) +
//! 512² (→ 2048² PiD) by default.
//!
//! ```sh
//! cargo test -p mlx-gen-sdxl --release --test pid_decode_real_weights -- --ignored --nocapture
//! # RealVisXL leg:
//! SDXL_DIR=~/.cache/huggingface/hub/models--SG161222--RealVisXL_V5.0/snapshots/<rev> \
//!   cargo test -p mlx-gen-sdxl --release --test pid_decode_real_weights sdxl_pid_decode_vs_vae -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::{GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource};
// Force the linker to keep this crate's `inventory::submit!` model registration (the CLAUDE.md
// linkage gotcha) so `mlx_gen::load("sdxl", …)` resolves from the test binary.
use mlx_gen_sdxl as _;

const MODEL_ID: &str = "sdxl";

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

fn sdxl_dir() -> PathBuf {
    env_path("SDXL_DIR").unwrap_or_else(|| {
        first_snapshot_dir(
            "models--stabilityai--stable-diffusion-xl-base-1.0",
            "stable-diffusion-xl-base-1.0",
        )
    })
}

fn pid_checkpoint() -> PathBuf {
    env_path("PID_SDXL_SAFETENSORS").unwrap_or_else(|| {
        PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tools/golden/pid/sdxl_2kto4k.safetensors"
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

fn one_image(out: GenerationOutput) -> Image {
    match out {
        GenerationOutput::Images(v) => v.into_iter().next().unwrap(),
        _ => panic!("expected images"),
    }
}

#[test]
#[ignore = "needs an SDXL-family snapshot + converted sdxl PiD checkpoint + gemma-2-2b-it"]
fn sdxl_pid_decode_vs_vae() {
    let size: u32 = std::env::var("SDXL_PID_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);

    let spec = LoadSpec::new(WeightsSource::Dir(sdxl_dir())).with_pid(
        WeightsSource::File(pid_checkpoint()),
        WeightsSource::Dir(gemma_dir()),
    );

    eprintln!(
        "loading {} (+PiD overlay), size={size} ...",
        sdxl_dir().display()
    );
    let t = Instant::now();
    let model = mlx_gen::load(MODEL_ID, &spec).expect("load sdxl + PiD");
    eprintln!("loaded in {:.1}s", t.elapsed().as_secs_f32());

    let base = GenerationRequest {
        prompt: "a red fox in a snowy pine forest at dawn, photorealistic".into(),
        width: size,
        height: size,
        count: 1,
        seed: Some(7),
        ..Default::default()
    };

    let t = Instant::now();
    let vae_img = one_image(model.generate(&base, &mut |_| {}).expect("vae generate"));
    let vae_dt = t.elapsed().as_secs_f32();
    let (vlo, vhi, vmu) = stats(&vae_img);
    eprintln!(
        "VAE: {}x{} in {vae_dt:.2}s  range [{vlo},{vhi}] mean {vmu:.1}",
        vae_img.width, vae_img.height
    );
    assert_eq!(vae_img.width, size, "VAE width == native");

    let pid_req = GenerationRequest {
        use_pid: true,
        ..base.clone()
    };
    let t = Instant::now();
    let pid_img = one_image(model.generate(&pid_req, &mut |_| {}).expect("pid generate"));
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
    save_png(&vae_img, &format!("{dir}/sdxl_vae_{}.png", vae_img.width));
    save_png(&pid_img, &format!("{dir}/sdxl_pid_{}.png", pid_img.width));
    eprintln!(
        "wrote {dir}/sdxl_vae_{}.png + sdxl_pid_{}.png  (PiD {:.1}× slower than VAE)",
        vae_img.width,
        pid_img.width,
        pid_dt / vae_dt.max(1e-3)
    );
}

#[test]
#[ignore = "needs an SDXL-family snapshot (no PiD weights) — proves the error path"]
fn use_pid_without_loaded_pid_errors() {
    // Loading WITHOUT spec.pid, then requesting use_pid, must error clearly (not silently VAE-decode).
    let spec = LoadSpec::new(WeightsSource::Dir(sdxl_dir()));
    let model = mlx_gen::load(MODEL_ID, &spec).expect("load sdxl");
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
