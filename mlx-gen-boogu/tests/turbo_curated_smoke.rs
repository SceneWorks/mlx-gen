//! epic 7114 (sc-7491): GATE + survey — the Boogu **Turbo** DMD student under the curated unified
//! samplers. The native DMD loop (predict → flow-renoise) is the baseline; this drives the same
//! few-step denoise through `run_flow_sampler` over the curated σ schedules. GATE: the advertised
//! stochastic samplers (`lcm`/`euler_ancestral`/`dpmpp_sde`, incl. the flow-aware `lcm`/`sgm_uniform`
//! combo) must render coherently and differ from native. The deterministic ODE solvers are surveyed
//! (printed) as the evidence they degrade the few-step student and stay off `descriptor_turbo`'s menu.
//!
//! `#[ignore]`d — needs the real Boogu Turbo snapshot (`mllm/ transformer/ vae/`), env `BOOGU_TURBO_DIR`:
//!   BOOGU_TURBO_DIR=~/.cache/huggingface/hub/models--SceneWorks--boogu-image-mlx/snapshots/<rev>/turbo \
//!     cargo test -p mlx-gen-boogu --release --test turbo_curated_smoke -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::media::Image;
use mlx_gen_boogu::{BooguPipeline, TurboOptions};

const W: u32 = 256;
const H: u32 = 256;
const STEPS: usize = 4;
const SEED: u64 = 42;
const PROMPT: &str = "a red apple on a wooden table, photorealistic";

fn snapshot() -> Option<PathBuf> {
    std::env::var("BOOGU_TURBO_DIR").ok().map(PathBuf::from)
}

/// (std, distinct-level-count, mean horizontal-adjacent-|Δ|) over an RGB8 buffer — a coherent natural
/// image has a broad histogram AND spatial smoothness; pure noise has a high adjacent Δ and a flat std.
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

fn is_coherent(img: &Image) -> bool {
    let (std, distinct, adj) = image_stats(&img.pixels, img.width);
    std > 10.0 && distinct > 24 && adj < 60.0
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

fn render(pipe: &BooguPipeline, sampler: Option<&str>, scheduler: Option<&str>) -> Image {
    let opts = TurboOptions {
        height: H,
        width: W,
        steps: STEPS,
        seed: SEED,
        conditioning_sigma: 0.001,
        sampler: sampler.map(Into::into),
        scheduler: scheduler.map(Into::into),
    };
    pipe.generate_turbo(PROMPT, &opts).expect("generate_turbo")
}

fn save(img: &Image, name: &str) {
    let dir = std::path::Path::new("/tmp/boogu_turbo_survey");
    std::fs::create_dir_all(dir).unwrap();
    let path = dir.join(format!("{name}.png"));
    image::save_buffer(
        &path,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    eprintln!("  saved {}", path.display());
}

#[test]
#[ignore = "needs the real Boogu Turbo snapshot (set BOOGU_TURBO_DIR)"]
fn turbo_curated_sampler_coherence_survey() {
    let Some(root) = snapshot() else {
        eprintln!("skipping: set BOOGU_TURBO_DIR");
        return;
    };
    let pipe = BooguPipeline::from_snapshot(&root).expect("load boogu turbo");

    let native = render(&pipe, None, None);
    let (std, distinct, adj) = image_stats(&native.pixels, native.width);
    eprintln!(
        "[native DMD]  std={std:.1} distinct={distinct} adjΔ={adj:.1} coherent={}",
        is_coherent(&native)
    );
    save(&native, "native");
    assert!(is_coherent(&native), "native DMD baseline must be coherent");

    // The curated samplers `descriptor_turbo` advertises (stochastic — they match the DMD student's
    // re-noised training regime and render at native quality, incl. flow-aware `lcm` post sc-7491).
    // These are GATED: coherent + a real swap.
    const ADVERTISED: &[&str] = &["lcm", "euler_ancestral", "dpmpp_sde"];

    // Full survey over the DMD σ grid (no scheduler re-shape). The deterministic solvers
    // (euler/ddim/heun/dpmpp_2m/uni_pc) integrate the few-step student and degrade the background
    // (surveyed/printed, NOT advertised); the stochastic ones — incl. flow-aware `lcm` (sc-7491) —
    // are gated coherent + a real swap.
    for name in [
        "euler",
        "ddim",
        "heun",
        "dpmpp_2m",
        "uni_pc",
        "lcm",
        "euler_ancestral",
        "dpmpp_sde",
    ] {
        let img = render(&pipe, Some(name), None);
        let (std, distinct, adj) = image_stats(&img.pixels, img.width);
        let frac = frac_diff(&native, &img);
        let coherent = is_coherent(&img);
        eprintln!(
            "[{name:>16}] std={std:.1} distinct={distinct} adjΔ={adj:.1} coherent={coherent} diff-vs-native={:.1}%",
            frac * 100.0
        );
        save(&img, name);
        if ADVERTISED.contains(&name) {
            assert!(
                coherent,
                "advertised turbo sampler {name:?} must be coherent"
            );
            assert!(
                frac > 0.01,
                "advertised turbo sampler {name:?} must be a real swap vs native (diff {frac})"
            );
        }
    }

    // Scheduler-axis probes — the ComfyUI-reported sweet spot is lcm + sgm_uniform. With the flow-aware
    // `noise_scaling` fix (sc-7491) `lcm` re-noises in the student's training regime, so these should be
    // coherent native-quality variants (and GATED).
    for (sampler, scheduler) in [
        ("lcm", "sgm_uniform"),
        ("euler_ancestral", "sgm_uniform"),
        ("dpmpp_sde", "sgm_uniform"),
        ("euler", "karras"),
    ] {
        let img = render(&pipe, Some(sampler), Some(scheduler));
        let (std, distinct, adj) = image_stats(&img.pixels, img.width);
        let coherent = is_coherent(&img);
        eprintln!(
            "[{sampler}+{scheduler}] std={std:.1} distinct={distinct} adjΔ={adj:.1} coherent={coherent} diff-vs-native={:.1}%",
            frac_diff(&native, &img) * 100.0
        );
        save(&img, &format!("{sampler}_{scheduler}"));
        if ADVERTISED.contains(&sampler) {
            assert!(
                coherent,
                "advertised {sampler}+{scheduler} must be coherent"
            );
        }
    }
}
