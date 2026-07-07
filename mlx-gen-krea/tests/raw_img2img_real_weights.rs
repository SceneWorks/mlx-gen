//! **sc-10224 (epic 8588, slice A) — img2img latent-init on Krea 2 Raw (true-CFG), real-weight coverage.**
//! Weight-gated (`#[ignore]`). Exercises [`KreaPipeline::generate_base_img2img`] — the CFG counterpart
//! of the distilled-Turbo `generate_turbo_img2img` (sc-8589/8590). Unlike Turbo img2img (CFG-free),
//! this runs the full-CFG Raw sampler (52 steps, guidance 3.5, dynamic mu), so it honors the guidance
//! scale + negative prompt. Not a pass/fail parity test — the GO/NO-GO is a human eyeball over the
//! saved PNGs; this harness generates them + prints objective coherence/fidelity numbers.
//!
//! Same experiment as the Turbo spike: render a reference `R` (t2i, prompt A), then img2img from `R`
//! with a restyle prompt B across a `strength` sweep. Higher strength → later start → output tracks
//! `R` more (lower MAE→R); lower strength → tracks prompt B more (lower MAE→pureB) — a monotone
//! crossover is the slider working, and every rung must stay coherent.
//!
//! ```sh
//! KREA_RAW_DIR=~/.cache/huggingface/hub/models--SceneWorks--krea-2-raw-mlx/snapshots/<rev>/q8 \
//!   cargo test -p mlx-gen-krea --release --test raw_img2img_real_weights -- --ignored --nocapture
//! ```
//! (With no env, auto-resolves the newest cached `SceneWorks/krea-2-raw-mlx` q8 turnkey.) PNGs +
//! a `SUMMARY` table land in `/tmp/krea_raw_img2img`.

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::media::Image;
use mlx_gen_krea::{KreaPipeline, TurboOptions};

const PROMPT_A: &str =
    "a photograph of a mountain landscape with a still lake and pine trees, clear blue sky, midday";
const PROMPT_B: &str =
    "a mountain landscape at sunset, warm orange and violet sky, glowing autumn foliage";
const NEGATIVE: &str = "blurry, low quality, distorted";
const DIM: u32 = 1024;
const STEPS: usize = 52;
const GUIDANCE: f32 = 3.5;

/// Resolve the Q8 turnkey subdir (`KREA_RAW_DIR` override, else the newest cached snapshot's `q8/`).
fn snapshot() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("KREA_RAW_DIR") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--SceneWorks--krea-2-raw-mlx/snapshots");
    let rev = std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("q8").join("transformer").is_dir())?;
    Some(rev.join("q8"))
}

/// (std, distinct-level-count, mean horizontal-adjacent-|Δ|). Coherent = broad histogram + spatial
/// smoothness; pure noise = high adjacent Δ + narrow std. Mirrors `img2img_spike_real_weights`.
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
        if i % stride >= 3 {
            adj_sum += (v as f64 - px[i - 3] as f64).abs();
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

/// Mean absolute per-pixel difference between two same-shape RGB8 buffers (0 = identical).
fn mae(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len(), "mae: size mismatch");
    let s: u64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
        .sum();
    s as f32 / a.len() as f32
}

fn save(img: &Image, name: &str) {
    let dir = std::path::Path::new("/tmp/krea_raw_img2img");
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

fn opts(seed: u64) -> TurboOptions {
    TurboOptions {
        width: DIM,
        height: DIM,
        steps: STEPS,
        seed,
        sampler: None,
        scheduler: None,
    }
}

#[test]
#[ignore = "needs the real Krea 2 Raw snapshot (set KREA_RAW_DIR, or cache SceneWorks/krea-2-raw-mlx)"]
fn raw_img2img_strength_sweep_is_coherent_and_cfg_driven() {
    let Some(root) = snapshot() else {
        eprintln!("skipping: no KREA_RAW_DIR and no cached SceneWorks/krea-2-raw-mlx q8 turnkey");
        return;
    };
    eprintln!("loading Krea 2 Raw (q8) from {}", root.display());
    let t_load = Instant::now();
    let pipe = KreaPipeline::from_snapshot(&root).expect("load Krea Raw pipeline");
    eprintln!("  loaded in {:.1}s", t_load.elapsed().as_secs_f32());

    // Reference R = CFG t2i(prompt A). The img2img init.
    let reference = pipe
        .generate_base(PROMPT_A, NEGATIVE, GUIDANCE, &opts(0))
        .expect("t2i reference");
    save(&reference, "00_reference_promptA");

    // Pure-B anchor = CFG t2i(prompt B) at the sweep seed. The strength→0 limit (reference ignored).
    let pure_b = pipe
        .generate_base(PROMPT_B, NEGATIVE, GUIDANCE, &opts(1))
        .expect("t2i pureB");
    save(&pure_b, "01_pureB_promptB_s0");

    let strengths = [0.2f32, 0.35, 0.5, 0.65, 0.8, 0.9];
    let mut rows: Vec<String> = Vec::new();
    rows.push(format!(
        "{:>8}  {:>5}  {:>7} {:>8} {:>7}  {:>9}  {:>10} {:>11}",
        "strength", "start", "std", "distinct", "adjΔ", "coherent", "MAE→R", "MAE→pureB"
    ));
    let (rs, rd, ra) = image_stats(&reference.pixels, DIM);
    rows.push(format!(
        "{:>8}  {:>5}  {:>7.1} {:>8} {:>7.1}  {:>9}  {:>10} {:>11}",
        "R(t2iA)",
        "-",
        rs,
        rd,
        ra,
        is_coherent(&reference),
        "0.0",
        "-"
    ));

    for &s in &strengths {
        let start = ((STEPS as f32 * s) as usize).max(1);
        let t = Instant::now();
        let out = pipe
            .generate_base_img2img(PROMPT_B, NEGATIVE, GUIDANCE, &reference, s, &opts(1))
            .expect("img2img render");
        let secs = t.elapsed().as_secs_f32();
        let (std, distinct, adj) = image_stats(&out.pixels, DIM);
        let m_r = mae(&out.pixels, &reference.pixels);
        let m_b = mae(&out.pixels, &pure_b.pixels);
        rows.push(format!(
            "{s:>8.2}  {start:>5}  {std:>7.1} {distinct:>8} {adj:>7.1}  {:>9}  {m_r:>10.2} {m_b:>11.2}",
            is_coherent(&out)
        ));
        save(&out, &format!("s{:02}_strength_{s:.2}", (s * 100.0) as u32));
        eprintln!("  strength {s:.2} (start {start}/{STEPS}) rendered in {secs:.1}s");
    }

    eprintln!("\n===== SUMMARY (sc-10224 Raw CFG img2img strength sweep) =====");
    eprintln!(
        "prompt A (reference): {PROMPT_A}\nprompt B (restyle)  : {PROMPT_B}\n{DIM}x{DIM} · {STEPS} steps · guidance {GUIDANCE} · q8\n"
    );
    for r in &rows {
        eprintln!("{r}");
    }
    eprintln!(
        "\nRead: MAE→R should fall and MAE→pureB should rise as strength increases (the slider moving \
         the output from prompt-B toward the reference). 'coherent' must hold across the usable middle. \
         PNGs in /tmp/krea_raw_img2img for the visual GO/NO-GO."
    );

    // Weak invariant (the VERDICT is the eyeball): the highest-strength render — which keeps the most
    // of the reference — stays a coherent image, proving the CFG img2img path denoises cleanly.
    let hi = pipe
        .generate_base_img2img(PROMPT_B, NEGATIVE, GUIDANCE, &reference, 0.9, &opts(1))
        .expect("hi-strength render");
    assert!(
        is_coherent(&hi),
        "strength 0.9 Raw CFG img2img must remain a coherent image, not mush/noise"
    );
}
