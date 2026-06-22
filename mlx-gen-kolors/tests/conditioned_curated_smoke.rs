//! sc-7297 / epic 7114 — curated-sampler smoke for the Kolors **conditioned** sub-providers.
//!
//! Validates that routing the Kolors ControlNet-pose / IP-Adapter / combined-pose tiers through the
//! curated k-diffusion path (`Kolors::denoise_curated_latents` → `denoise_curated`, threading the
//! ControlNet residuals + IP decoupled-attn tokens) renders coherently — i.e. a curated solver does
//! NOT destabilize the strong conditioning these modes were originally fixed-sampler-locked for. This
//! is the gate the old registry guard's "would desync under a multi-eval solver" hypothesis demanded
//! before the lock could be lifted.
//!
//! Runs against the REAL weights (no torch, no `tools/golden`):
//!
//!   KOLORS_SNAPSHOT=<Kolors-diffusers snapshot dir> \
//!   KOLORS_CONTROLNET=<Kolors-ControlNet-Pose snapshot dir> \
//!   KOLORS_IP_ADAPTER=<Kolors-IP-Adapter-Plus snapshot dir> \
//!   [KOLORS_POSE=<pose/control image>] [KOLORS_REF=<reference/identity image>] \
//!   cargo test -p mlx-gen-kolors --release --test conditioned_curated_smoke -- --ignored --nocapture
//!
//! Gate (directional): for each conditioned mode and each curated solver —
//!   (1) the render is **coherent** (not collapsed to noise/flat — the destabilization failure mode), and
//!   (2) it **differs** from the bespoke `euler_discrete` default (a real solver swap, not a silent no-op).
//! The default path itself stays byte-exact (covered by the `*_parity` gates); here we prove the
//! curated route is wired through the conditioning and behaves.

use std::path::PathBuf;

use mlx_gen::{
    Conditioning, ControlKind, GenerationOutput, GenerationRequest, Image, LoadSpec, Precision,
    WeightsSource,
};
// Force-link the provider crate so its `inventory::submit!` registration is included in this test
// binary (else the linker dead-strips it and `mlx_gen::load("kolors", …)` finds no registration).
use mlx_gen_kolors::MODEL_ID;

const SIZE: u32 = 512;
const STEPS: u32 = 8;
const SEED: u64 = 7;
const PROMPT: &str = "a person standing in a sunlit park, photorealistic, sharp focus, high detail";
const NEGATIVE: &str = "lowres, blurry, deformed, disfigured, cartoon, painting";

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
fn ip_snap() -> Option<PathBuf> {
    snap_env(
        "KOLORS_IP_ADAPTER",
        "models--Kwai-Kolors--Kolors-IP-Adapter-Plus",
    )
}

/// An image from `env` (real photo / pose), or a deterministic synthetic fallback so the smoke still
/// exercises the wiring without curated assets. The conditioning *adherence* numbers are only
/// meaningful with real images; the coherence + distinctness gate holds either way.
fn image_from(env: &str) -> Image {
    if let Ok(p) = std::env::var(env) {
        let img = image::open(&p)
            .unwrap_or_else(|e| panic!("open {env}={p}: {e}"))
            .to_rgb8();
        return Image {
            width: img.width(),
            height: img.height(),
            pixels: img.into_raw(),
        };
    }
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

fn spec(base: PathBuf, control: Option<PathBuf>, ip: Option<PathBuf>) -> LoadSpec {
    LoadSpec {
        weights: WeightsSource::Dir(base),
        quantize: None,
        precision: Precision::Bf16,
        control: control.map(WeightsSource::Dir),
        ip_adapter: ip.map(WeightsSource::Dir),
        adapters: Vec::new(),
        extra_controls: Vec::new(),
    }
}

fn req(
    conditioning: Vec<Conditioning>,
    sampler: Option<&str>,
    scheduler: Option<&str>,
) -> GenerationRequest {
    GenerationRequest {
        prompt: PROMPT.into(),
        negative_prompt: Some(NEGATIVE.into()),
        width: SIZE,
        height: SIZE,
        count: 1,
        steps: Some(STEPS),
        guidance: Some(5.0),
        seed: Some(SEED),
        sampler: sampler.map(Into::into),
        scheduler: scheduler.map(Into::into),
        conditioning,
        ..Default::default()
    }
}

/// (std, distinct-level-count, mean horizontal-adjacent-|Δ|) over an RGB8 buffer.
fn image_stats(px: &[u8], w: u32) -> (f32, usize, f32) {
    let n = px.len() as f64;
    let mean = px.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = px.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let mut seen = [false; 256];
    for &v in px {
        seen[v as usize] = true;
    }
    let distinct = seen.iter().filter(|&&b| b).count();
    let stride = (w * 3) as usize;
    let (mut adj_sum, mut adj_n) = (0f64, 0u64);
    for (i, &v) in px.iter().enumerate() {
        if i >= 3 && i % stride >= 3 {
            adj_sum += (v as i32 - px[i - 3] as i32).unsigned_abs() as f64;
            adj_n += 1;
        }
    }
    (
        var.sqrt() as f32,
        distinct,
        (adj_sum / adj_n.max(1) as f64) as f32,
    )
}

fn assert_coherent(img: &Image, label: &str) {
    assert_eq!(
        img.pixels.len(),
        (img.width * img.height * 3) as usize,
        "{label}: pixel buffer is RGB8 HWC"
    );
    let (std, distinct, adj) = image_stats(&img.pixels, img.width);
    println!("[smoke] {label:<32} std={std:.1} distinct={distinct} adjΔ={adj:.1}");
    assert!(std > 10.0, "{label}: image is near-flat (std {std:.1})");
    assert!(
        distinct > 24,
        "{label}: too few distinct levels ({distinct})"
    );
    assert!(
        adj < 60.0,
        "{label}: not spatially smooth — looks like destabilized noise (adjΔ {adj:.1})"
    );
}

fn frac_diff(a: &Image, b: &Image) -> f32 {
    let differ = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .filter(|(x, y)| (**x as i16 - **y as i16).abs() > 4)
        .count();
    differ as f32 / a.pixels.len() as f32
}

fn one_image(out: GenerationOutput) -> Image {
    match out {
        GenerationOutput::Images(mut v) => v.pop().expect("one image"),
        other => panic!("expected Images, got {other:?}"),
    }
}

/// Load once, render the bespoke default + each curated solver over the SAME conditioning, and gate
/// each curated render on coherence + distinctness from the default.
fn run_mode(label: &str, spec: LoadSpec, conditioning: impl Fn() -> Vec<Conditioning>) {
    let gen = mlx_gen::load(MODEL_ID, &spec).expect("registry load");
    let default = one_image(
        gen.generate(&req(conditioning(), None, None), &mut |_| {})
            .expect("default generate"),
    );
    assert_coherent(&default, &format!("{label}/default(euler_discrete)"));

    for sampler in ["euler", "heun", "dpmpp_2m"] {
        let img = one_image(
            gen.generate(&req(conditioning(), Some(sampler), None), &mut |_| {})
                .expect("curated generate"),
        );
        let tag = format!("{label}/{sampler}");
        assert_coherent(&img, &tag);
        let frac = frac_diff(&default, &img);
        println!(
            "[smoke] {tag:<32} differs from default in {:.2}% of pixels",
            frac * 100.0
        );
        assert!(
            frac > 0.01,
            "{tag}: curated solver must differ from the euler_discrete default (real swap); differ {frac}"
        );
    }
}

/// ControlNet-pose, curated: the pose branch residuals thread the curated solver.
#[test]
#[ignore = "needs Kolors-diffusers + Kolors-ControlNet-Pose snapshots (set KOLORS_SNAPSHOT/KOLORS_CONTROLNET)"]
fn controlnet_curated_is_coherent_and_distinct() {
    let (Some(base), Some(cn)) = (base_snap(), cn_snap()) else {
        eprintln!("skipping: no Kolors base / ControlNet snapshot");
        return;
    };
    let pose = image_from("KOLORS_POSE");
    run_mode("controlnet", spec(base, Some(cn), None), || {
        vec![Conditioning::Control {
            image: pose.clone(),
            kind: ControlKind::Pose,
            scale: 0.7,
        }]
    });
}

/// IP-Adapter, curated: the decoupled-attn image tokens thread the curated solver.
#[test]
#[ignore = "needs Kolors-diffusers + Kolors-IP-Adapter-Plus snapshots (set KOLORS_SNAPSHOT/KOLORS_IP_ADAPTER)"]
fn ip_adapter_curated_is_coherent_and_distinct() {
    let (Some(base), Some(ip)) = (base_snap(), ip_snap()) else {
        eprintln!("skipping: no Kolors base / IP-Adapter snapshot");
        return;
    };
    let reference = image_from("KOLORS_REF");
    run_mode("ip_adapter", spec(base, None, Some(ip)), || {
        vec![Conditioning::Reference {
            image: reference.clone(),
            strength: None,
        }]
    });
}

/// Combined pose tier (sc-5012), curated: ControlNet residuals + IP tokens on an img2img init — the
/// exact composite the old registry guard refused to run under a multi-eval solver.
#[test]
#[ignore = "needs Kolors-diffusers + ControlNet-Pose + IP-Adapter-Plus snapshots"]
fn combined_pose_curated_is_coherent_and_distinct() {
    let (Some(base), Some(cn), Some(ip)) = (base_snap(), cn_snap(), ip_snap()) else {
        eprintln!("skipping: no Kolors base / ControlNet / IP-Adapter snapshot");
        return;
    };
    let pose = image_from("KOLORS_POSE");
    let reference = image_from("KOLORS_REF");
    run_mode("combined_pose", spec(base, Some(cn), Some(ip)), || {
        vec![
            Conditioning::Control {
                image: pose.clone(),
                kind: ControlKind::Pose,
                scale: 0.7,
            },
            Conditioning::Reference {
                image: reference.clone(),
                strength: None,
            },
        ]
    });
}
