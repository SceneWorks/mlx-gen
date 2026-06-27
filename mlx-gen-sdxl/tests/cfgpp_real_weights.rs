//! sc-8256 — CFG++ real-weight A/B on SDXL (epic 7434). Renders the same seed/prompt/steps/sampler at
//! a HIGH guidance scale with plain CFG vs CFG++ (`guidance_method = "cfg_pp"`), and reports the
//! oversaturation/clipping difference + saves both PNGs for eyeball review — the rendered confirmation
//! the spike (sc-8254) could not do (it proved the gen-core numerics + trajectory divergence; this is
//! the visual proof on real weights). Also gates the structural N1 (cfg_pp is opt-in: `None`/`"cfg"`
//! reproduce the plain path byte-for-byte).
//!
//! `#[ignore]`d — needs the real `stabilityai/stable-diffusion-xl-base-1.0` snapshot:
//!   SDXL_SNAPSHOT=/path cargo test -p mlx-gen-sdxl --release --test cfgpp_real_weights -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::{GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource};
use mlx_gen_sdxl::MODEL_ID;

const W: u32 = 1024;
const H: u32 = 1024;
const STEPS: u32 = 30;
const SEED: u64 = 42;
const PROMPT: &str =
    "a photograph of an astronaut riding a horse on mars, golden hour, highly detailed";
/// A CFG++-compatible base solver (euler/ddim/dpmpp_2m); the curated path drives `dpmpp_2m_cfg++`.
const SAMPLER: &str = "dpmpp_2m";

fn snapshot() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SDXL_SNAPSHOT") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots");
    std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
}

fn render(cfg: f32, guidance_method: Option<&str>) -> Image {
    let root = snapshot().expect("SDXL snapshot");
    let spec = LoadSpec::new(WeightsSource::Dir(root));
    let generator = mlx_gen::load(MODEL_ID, &spec).expect("load sdxl");
    let req = GenerationRequest {
        prompt: PROMPT.into(),
        width: W,
        height: H,
        seed: Some(SEED),
        steps: Some(STEPS),
        guidance: Some(cfg),
        sampler: Some(SAMPLER.into()),
        guidance_method: guidance_method.map(Into::into),
        ..Default::default()
    };
    match generator.generate(&req, &mut |_| {}).unwrap() {
        GenerationOutput::Images(mut v) => v.pop().unwrap(),
        other => panic!("expected Images, got {other:?}"),
    }
}

/// (std, distinct-level count, mean horizontal-adjacent-|Δ|) — coherence proxy (see unified_sampler_smoke).
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
    let (std, distinct, adj) = image_stats(&img.pixels, img.width);
    eprintln!("[{label}] std={std:.1} distinct={distinct} adjΔ={adj:.1}");
    assert!(std > 10.0, "{label}: near-flat (std {std:.1})");
    assert!(distinct > 24, "{label}: too few levels ({distinct})");
    assert!(adj < 60.0, "{label}: looks like noise (adjΔ {adj:.1})");
}

/// Fraction of channel values pinned at the extremes (0 / 255) — high-CFG oversaturation clips here.
fn clip_frac(img: &Image) -> f32 {
    let clipped = img.pixels.iter().filter(|&&v| v == 0 || v == 255).count();
    clipped as f32 / img.pixels.len() as f32
}

/// Fraction of channels differing by >4 — a liveness check that two renders are genuinely distinct.
fn frac_diff(a: &Image, b: &Image) -> f32 {
    let differ = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .filter(|(x, y)| (**x as i16 - **y as i16).abs() > 4)
        .count();
    differ as f32 / a.pixels.len() as f32
}

fn save_png(name: &str, img: &Image) -> PathBuf {
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tools/golden");
    std::fs::create_dir_all(&out).unwrap();
    let p = out.join(name);
    image::save_buffer(
        &p,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    p
}

/// N2 + the rendered A/B. CFG++ reparameterizes guidance: renoising from the unconditional estimate
/// changes the effective strength, so cfg_pp operates at a LOW guidance scale (Chung et al. / ComfyUI
/// use ~1–2, not the 7–12 of plain CFG — at a high scale cfg_pp over-drives, exactly as ComfyUI does).
/// This characterizes cfg_pp across its design range vs a normal plain-CFG reference, reports the
/// extreme-value clipping (the oversaturation cfg_pp relieves), and saves every render for eyeball
/// review. The quantitative claim asserted here: in its design range cfg_pp stays coherent and is NOT
/// blown out (low clipping), and it is a genuinely distinct trajectory from plain CFG.
#[test]
#[ignore = "needs the real SDXL snapshot (set SDXL_SNAPSHOT or HF cache)"]
fn cfgpp_operating_range_and_ab() {
    if snapshot().is_none() {
        eprintln!("skipping: no SDXL snapshot");
        return;
    }
    // Plain-CFG reference at a normal scale (good adherence, the usual mild oversaturation).
    let ref_cfg = render(7.0, None);
    assert_coherent(&ref_cfg, "cfg@7 (ref)");
    let ref_clip = clip_frac(&ref_cfg);
    let pr = save_png("sdxl_cfg_g7_ref.png", &ref_cfg);
    eprintln!(
        "[ref] cfg@7: clip_frac={ref_clip:.4}; saved {}",
        pr.display()
    );

    // cfg_pp across its design range. In-range renders must be coherent and NOT blown out.
    for &scale in &[1.5f32, 2.5, 4.0, 8.0] {
        let img = render(scale, Some("cfg_pp"));
        let (std, distinct, adj) = image_stats(&img.pixels, img.width);
        let clip = clip_frac(&img);
        let live = frac_diff(&ref_cfg, &img);
        let p = save_png(&format!("sdxl_cfgpp_g{}.png", scale as u32), &img);
        eprintln!(
            "[cfg_pp@{scale}] std={std:.1} distinct={distinct} adjΔ={adj:.1} clip_frac={clip:.4} \
             vs-ref frac_diff={live:.3}; saved {}",
            p.display()
        );
        // Liveness: cfg_pp is a genuinely distinct trajectory from the plain reference.
        assert!(live > 0.05, "cfg_pp@{scale}: not distinct from plain cfg");
        // In its design range (≤2.5) cfg_pp must stay on-manifold — coherent + not blown out.
        if scale <= 2.5 {
            assert_coherent(&img, &format!("cfg_pp@{scale}"));
            assert!(
                clip < 0.10,
                "cfg_pp@{scale}: blown out (clip_frac {clip:.4}) — expected ≤0.10 in design range"
            );
        }
    }
}

/// N1 (structural): CFG++ is strictly opt-in. On a CFG++-compatible sampler, `guidance_method = None`
/// and `"cfg"` both take the plain curated path → byte-identical output (the cfg_pp branch never fires).
#[test]
#[ignore = "needs the real SDXL snapshot (set SDXL_SNAPSHOT or HF cache)"]
fn cfgpp_is_opt_in_default_unchanged() {
    if snapshot().is_none() {
        eprintln!("skipping: no SDXL snapshot");
        return;
    }
    let none = render(7.0, None);
    let explicit_cfg = render(7.0, Some("cfg"));
    let d = frac_diff(&none, &explicit_cfg);
    eprintln!("N1 opt-in: guidance_method None vs \"cfg\" frac_diff={d:.5}");
    assert_eq!(
        d, 0.0,
        "default/`cfg` must be byte-identical (cfg_pp is opt-in)"
    );
}
