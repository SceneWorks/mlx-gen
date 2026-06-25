//! sc-7847 e2e: the **integrated** PiD decode path for the FLUX.2 family — load the real
//! `FLUX.2-klein-9b` snapshot with a PiD decoder overlay (`LoadSpec::with_pid`) and run
//! `Generator::generate` once for the VAE baseline and once with `use_pid`, proving the live denoised
//! **packed BN-normalized** latent routes through `generate_impl`'s `LatentDecoder` seam into a 4×
//! super-resolved PiD image. FLUX.2 is the highest-risk re-verify leg of epic 7840: unlike FLUX.1
//! (16-ch affine latent), the FLUX.2 student is fed the *packed 128-ch* BN latent at H/16 (the "32 vs
//! 128" registry note, resolved at wiring time — sc-7847). klein-9b is a distilled few-step model, so
//! the default step count is low.
//!
//! `#[ignore]`d — needs the `FLUX.2-klein-9b` snapshot (env `FLUX2_KLEIN_DIR`, else the HF cache), the
//! converted flux2 PiD checkpoint (env `PID_FLUX2_SAFETENSORS`, else `tools/golden/pid/flux2_2k.safetensors`),
//! and a `gemma-2-2b-it` snapshot dir (env `PID_GEMMA_DIR`, else the HF cache). Loads the full FLUX.2
//! model **plus** the PiD net + Gemma, so it is memory-heavy; defaults to Q8 + 512² (→ 2048² PiD).
//!
//! ```sh
//! cargo test -p mlx-gen-flux2 --release --test pid_decode_real_weights -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::{GenerationOutput, GenerationRequest, Image, LoadSpec, Quant, WeightsSource};
// Force the linker to keep this crate's `inventory::submit!` model registrations (the CLAUDE.md
// linkage gotcha) so `mlx_gen::load("flux2_klein_9b", …)` resolves from the test binary.
use mlx_gen_flux2 as _;

const MODEL_ID: &str = "flux2_klein_9b";

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

fn flux2_dir() -> PathBuf {
    env_path("FLUX2_KLEIN_DIR").unwrap_or_else(|| {
        first_snapshot_dir(
            "models--black-forest-labs--FLUX.2-klein-9b",
            "FLUX.2-klein-9b",
        )
    })
}

fn pid_checkpoint() -> PathBuf {
    env_path("PID_FLUX2_SAFETENSORS").unwrap_or_else(|| {
        PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tools/golden/pid/flux2_2k.safetensors"
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

fn quant_from_env() -> Option<Quant> {
    match std::env::var("FLUX2_PID_QUANT").as_deref() {
        Ok("none") => None,
        Ok("q4") => Some(Quant::Q4),
        // Default Q8 to bound memory (full FLUX.2 + PiD net + Gemma coexist in one process).
        _ => Some(Quant::Q8),
    }
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
#[ignore = "needs the FLUX.2-klein-9b snapshot + converted flux2 PiD checkpoint + gemma-2-2b-it"]
fn flux2_klein_pid_decode_vs_vae() {
    let size: u32 = std::env::var("FLUX2_PID_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);

    let mut spec = LoadSpec::new(WeightsSource::Dir(flux2_dir()));
    if let Some(q) = quant_from_env() {
        spec = spec.with_quant(q);
    }
    spec = spec.with_pid(
        WeightsSource::File(pid_checkpoint()),
        WeightsSource::Dir(gemma_dir()),
    );

    eprintln!("loading FLUX.2-klein-9b (+PiD overlay), size={size} ...");
    let t = Instant::now();
    let model = mlx_gen::load(MODEL_ID, &spec).expect("load FLUX.2-klein-9b + PiD");
    eprintln!("loaded in {:.1}s", t.elapsed().as_secs_f32());

    let base = GenerationRequest {
        prompt: "a mountain valley landscape at golden hour with a winding river and pine forest"
            .into(),
        width: size,
        height: size,
        count: 1,
        seed: Some(7),
        ..Default::default()
    };

    // --- VAE baseline ---
    let t = Instant::now();
    let vae_img = one_image(model.generate(&base, &mut |_| {}).expect("vae generate"));
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
    save_png(&vae_img, &format!("{dir}/flux2_vae_{}.png", vae_img.width));
    save_png(&pid_img, &format!("{dir}/flux2_pid_{}.png", pid_img.width));
    eprintln!(
        "wrote {dir}/flux2_vae_{}.png + flux2_pid_{}.png  (PiD {:.1}× slower than VAE)",
        vae_img.width,
        pid_img.width,
        pid_dt / vae_dt.max(1e-3)
    );
}

#[test]
#[ignore = "needs the FLUX.2-klein-9b snapshot (no PiD weights) — proves the error path"]
fn use_pid_without_loaded_pid_errors() {
    // Loading WITHOUT spec.pid, then requesting use_pid, must error clearly (not silently VAE-decode).
    let mut spec = LoadSpec::new(WeightsSource::Dir(flux2_dir()));
    if let Some(q) = quant_from_env() {
        spec = spec.with_quant(q);
    }
    let model = mlx_gen::load(MODEL_ID, &spec).expect("load FLUX.2-klein-9b");
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
