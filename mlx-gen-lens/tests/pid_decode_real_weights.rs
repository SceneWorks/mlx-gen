//! sc-7847 e2e: the **integrated** PiD decode path for Lens — load the real `microsoft/Lens-Turbo`
//! snapshot with a PiD decoder overlay (`LoadSpec::with_pid`) and run `Generator::generate` once for
//! the VAE baseline and once with `use_pid`, proving the live denoised packed latent routes through
//! the `vae::decode` seam into a 4× super-resolved PiD image. Lens is the FLUX.2 VAE latent space
//! (`mlx_gen_flux2::Flux2Vae`), so it reuses the `flux2` PiD student; the DiT's packed 128 channels
//! are already in the FLUX-canonical `c·4+p1·2+p2` order the student trained on (no re-pack needed).
//!
//! `#[ignore]`d — needs the `Lens-Turbo` snapshot (env `LENS_DIR`, else the HF cache), the converted
//! flux2 PiD checkpoint (env `PID_FLUX2_SAFETENSORS`, else `tools/golden/pid/flux2_2k.safetensors`),
//! and a `gemma-2-2b-it` snapshot dir (env `PID_GEMMA_DIR`, else the HF cache). The gpt-oss encoder is
//! ~20 B params, so this defaults to Q8 (quantizes the encoder's MoE experts at load).
//!
//! ```sh
//! cargo test -p mlx-gen-lens --release --test pid_decode_real_weights -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::{GenerationOutput, GenerationRequest, Image, LoadSpec, Quant, WeightsSource};
// Force the linker to keep this crate's `inventory::submit!` model registrations (the CLAUDE.md
// linkage gotcha) so `mlx_gen::load("lens_turbo", …)` resolves from the test binary.
use mlx_gen_lens as _;

const MODEL_ID: &str = "lens_turbo";

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

fn lens_dir() -> PathBuf {
    env_path("LENS_DIR")
        .unwrap_or_else(|| first_snapshot_dir("models--microsoft--Lens-Turbo", "Lens-Turbo"))
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
    match std::env::var("LENS_PID_QUANT").as_deref() {
        Ok("none") => None,
        Ok("q4") => Some(Quant::Q4),
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
#[ignore = "needs the Lens-Turbo snapshot + converted flux2 PiD checkpoint + gemma-2-2b-it"]
fn lens_turbo_pid_decode_vs_vae() {
    let size: u32 = std::env::var("LENS_PID_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);

    let mut spec = LoadSpec::new(WeightsSource::Dir(lens_dir()));
    if let Some(q) = quant_from_env() {
        spec = spec.with_quant(q);
    }
    spec = spec.with_pid(
        WeightsSource::File(pid_checkpoint()),
        WeightsSource::Dir(gemma_dir()),
    );

    eprintln!("loading Lens-Turbo (+PiD overlay), size={size} ...");
    let t = Instant::now();
    let model = mlx_gen::load(MODEL_ID, &spec).expect("load Lens-Turbo + PiD");
    eprintln!("loaded in {:.1}s", t.elapsed().as_secs_f32());

    let base = GenerationRequest {
        prompt: "a serene mountain lake at sunrise with mist over the water, photorealistic".into(),
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
    save_png(&vae_img, &format!("{dir}/lens_vae_{}.png", vae_img.width));
    save_png(&pid_img, &format!("{dir}/lens_pid_{}.png", pid_img.width));
    eprintln!(
        "wrote {dir}/lens_vae_{}.png + lens_pid_{}.png  (PiD {:.1}× slower than VAE)",
        vae_img.width,
        pid_img.width,
        pid_dt / vae_dt.max(1e-3)
    );
}
