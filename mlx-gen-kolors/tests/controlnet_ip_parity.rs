//! Kolors combined ControlNet-pose + IP-Adapter img2img parity (sc-5012).
//!
//! `#[ignore]`d: needs the Kolors snapshot (+ `tokenizer.json`), the `Kwai-Kolors/Kolors-ControlNet-Pose`
//! snapshot, and the `Kwai-Kolors/Kolors-IP-Adapter-Plus` snapshot — no dumped golden (synthetic
//! reference + control images). This is the worker-integration gate for the SceneWorks strict-pose
//! tier, where the pose ControlNet + the IP-Adapter identity + an img2img init run in ONE pass (torch
//! `KolorsDiffusersAdapter._run_pose`).
//!
//! Wiring invariant (f32): with `control_scale = 0` AND `ip_scale = 0` the combined pass
//! ([`Kolors::generate_controlnet_ip`]) is byte-identical to plain [`Kolors::img2img`] — both
//! injections vanish, so the only thing left is the img2img init + denoise (and the RNG order is
//! identical: the IP-token + VAE-encode steps draw no RNG, so the first draw after the seed is the
//! same noise). This proves the combined denoise reuses the validated img2img / ControlNet / IP
//! primitives without any spurious effect. Then with both scales > 0 the output perturbs vs the
//! img2img base and renders coherently. (f32 for the byte-exact invariant — the same bf16-chaos
//! caveat as the per-component ControlNet / IP gates.)
//!
//! Run: `cargo test -p mlx-gen-kolors --release --test controlnet_ip_parity -- --ignored --nocapture`

use std::path::PathBuf;

use mlx_gen::{Image, WeightsSource};
use mlx_gen_kolors::ip_adapter::load_kolors_ip_adapter;
use mlx_gen_kolors::unet::load_controlnet;
use mlx_gen_kolors::Kolors;
use mlx_rs::Dtype;

fn hf_snapshot(repo_dir: &str, what: &str) -> PathBuf {
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(repo_dir)
        .join("snapshots");
    std::fs::read_dir(&snaps)
        .unwrap_or_else(|_| panic!("{what} snapshots dir ({})", snaps.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .unwrap_or_else(|| panic!("a {what} snapshot dir"))
}

fn kolors_snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("KOLORS_SNAPSHOT") {
        return PathBuf::from(p);
    }
    hf_snapshot("models--Kwai-Kolors--Kolors-diffusers", "Kolors")
}

fn ip_snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("KOLORS_IP_ADAPTER") {
        return PathBuf::from(p);
    }
    hf_snapshot("models--Kwai-Kolors--Kolors-IP-Adapter-Plus", "IP-Adapter")
}

fn controlnet_snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("KOLORS_CONTROLNET") {
        return PathBuf::from(p);
    }
    hf_snapshot("models--Kwai-Kolors--Kolors-ControlNet-Pose", "ControlNet")
}

/// A deterministic synthetic RGB reference image (a smooth diagonal gradient with a centered
/// rectangle) — the IP-Adapter identity + img2img init only need a real image, not a real face, for
/// the wiring invariant (scale-0 == img2img) and the perturbation/coherence checks.
fn synthetic_reference(w: u32, h: u32) -> Image {
    let mut px = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            px[i] = ((x * 255) / w) as u8;
            px[i + 1] = ((y * 255) / h) as u8;
            px[i + 2] = (((x + y) * 255) / (w + h)) as u8;
        }
    }
    Image {
        width: w,
        height: h,
        pixels: px,
    }
}

/// A deterministic synthetic "skeleton" control image (a few bright strokes on black) — the content
/// is irrelevant at control_scale = 0 and only needs to be a valid image for the perturbation check.
fn synthetic_control(w: u32, h: u32) -> Image {
    let mut px = vec![0u8; (w * h * 3) as usize];
    for y in (h / 5)..(h / 5 + 12) {
        for x in (w / 6)..(5 * w / 6) {
            let i = ((y * w + x) * 3) as usize;
            px[i] = 255;
            px[i + 1] = 255;
            px[i + 2] = 255;
        }
    }
    Image {
        width: w,
        height: h,
        pixels: px,
    }
}

/// Mean absolute u8 difference between two same-sized images.
fn mean_abs_diff(a: &Image, b: &Image) -> f64 {
    assert_eq!(a.pixels.len(), b.pixels.len());
    a.pixels
        .iter()
        .zip(&b.pixels)
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as f64)
        .sum::<f64>()
        / a.pixels.len() as f64
}

#[test]
#[ignore = "needs the Kolors snapshot + tokenizer.json + Kolors-ControlNet-Pose + Kolors-IP-Adapter-Plus snapshots"]
fn kolors_controlnet_ip_scale0_matches_img2img_and_renders() {
    // f32 for the byte-exact invariant (same rationale as the per-component gates).
    let snap = kolors_snapshot();
    let mut kolors = Kolors::load(&snap, Dtype::Float32).expect("load Kolors");
    let cn = load_controlnet(&WeightsSource::Dir(controlnet_snapshot()), Dtype::Float32)
        .expect("load ControlNet");
    let (ip_encoder, pairs) = load_kolors_ip_adapter(&ip_snapshot(), Dtype::Float32).unwrap();
    kolors.install_ip_adapter(pairs).unwrap();
    let (h, w, steps, cfg, strength, seed) = (512i32, 512i32, 8usize, 5.0f32, 0.8f32, 11u64);
    let (prompt, negative) = ("a portrait of a person", "blurry, low quality");

    // The reference drives BOTH the IP identity AND the img2img init (torch `_run_pose`); the control
    // image is the pose skeleton (content irrelevant at control_scale = 0). Synthetic so the test runs
    // from the cached model weights alone (no dumped golden needed).
    let reference = synthetic_reference(w as u32, h as u32);
    let control = synthetic_control(w as u32, h as u32);

    // Plain img2img base (no control, no IP). Same seed/strength as the combined runs.
    let base = kolors
        .img2img(
            &reference, prompt, negative, steps, strength, cfg, seed, h, w,
        )
        .unwrap();

    // Combined pass with BOTH scales 0 → must equal plain img2img byte-for-byte (identical RNG order:
    // ip-tokens + VAE-encode draw no RNG, so the first post-seed draw is the same noise).
    let s0 = kolors
        .generate_controlnet_ip(
            &cn,
            &ip_encoder,
            &control,
            &reference,
            prompt,
            negative,
            steps,
            strength,
            cfg,
            0.0,
            0.0,
            seed,
            h,
            w,
        )
        .unwrap();
    let d0 = mean_abs_diff(&base, &s0);
    println!(
        "combined(control=0, ip=0) vs img2img: mean_abs_u8_diff={d0:.4} bytes_eq={}",
        base.pixels == s0.pixels
    );
    assert_eq!(
        base.pixels, s0.pixels,
        "control_scale=0 + ip_scale=0 (f32) must decode byte-identically to plain img2img (combined \
         injection not zero-clean)"
    );

    // Combined pass with both scales on → perturbs vs the img2img base + renders coherently.
    let s_on = kolors
        .generate_controlnet_ip(
            &cn,
            &ip_encoder,
            &control,
            &reference,
            prompt,
            negative,
            steps,
            strength,
            cfg,
            0.7,
            0.6,
            seed,
            h,
            w,
        )
        .unwrap();
    let don = mean_abs_diff(&base, &s_on);
    println!("combined(control=0.7, ip=0.6) vs img2img: mean_abs_u8_diff={don:.4}");
    assert!(
        don > 1.0,
        "control+ip on should visibly perturb the output vs the img2img base (mean_abs_u8_diff \
         {don:.4} too small)"
    );
    assert!(
        s_on.pixels.iter().any(|&p| p > 16) && s_on.pixels.iter().any(|&p| p < 239),
        "degenerate combined pose+IP render"
    );
    println!(
        "✓ Kolors combined ControlNet-pose + IP-Adapter img2img: scale-0 == img2img (wiring clean), \
         scales>0 perturb + render coherently"
    );
}
