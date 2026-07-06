//! **sc-8589 (epic 8588, slice A) — img2img latent-init on Krea 2 Turbo, real-weight coverage.**
//! Weight-gated (`#[ignore]`). Exercises [`KreaPipeline::generate_turbo_img2img`] on the distilled
//! few-step CFG-free Turbo (A0 sc-8589 validated the strength window; A1 sc-8590 productionized the
//! entrypoint) and MAPS the usable `strength` window. The UI slider is FULL-range 0–1 (A2/A3); this
//! window is guidance, NOT a clamp. Not a pass/fail parity test — the GO/NO-GO is a human eyeball over
//! the saved PNGs; this harness generates them and prints objective coherence + fidelity numbers.
//!
//! Experiment: render a reference `R` (t2i, prompt A), then img2img from `R` with a **restyle** prompt
//! B across a `strength` sweep. Higher strength → later start → output should track `R` more (lower
//! MAE-to-R), lower strength → output should track prompt B more (lower MAE-to-pureB) — a monotone
//! crossover is the slider working. Coherence must hold across the usable middle of the range.
//!
//! ```sh
//! KREA_TURBO_DIR=~/.cache/huggingface/hub/models--SceneWorks--krea-2-turbo-mlx/snapshots/<rev>/q8 \
//!   cargo test -p mlx-gen-krea --release --test img2img_spike_real_weights -- --ignored --nocapture
//! ```
//! (With no env, auto-resolves the newest cached `SceneWorks/krea-2-turbo-mlx` q8 turnkey.) PNGs +
//! a `SUMMARY` table land in `/tmp/krea_img2img_spike`.

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::media::Image;
use mlx_gen_krea::{KreaPipeline, TurboOptions};

const PROMPT_A: &str =
    "a photograph of a mountain landscape with a still lake and pine trees, clear blue sky, midday";
const PROMPT_B: &str =
    "a mountain landscape at sunset, warm orange and violet sky, glowing autumn foliage";
// People / pose probe (run #2). Same pose described in every prompt so the latent AND the text agree
// on pose — only identity (woman → bearded man) or style (photo → anime) differs.
const REF_PERSON: &str =
    "a full-body studio photograph of a young woman in a red athletic outfit, standing in a dynamic \
     pose with her right arm raised straight overhead and her left hand on her hip, plain light-grey \
     seamless background, sharp focus";
const CHANGE_ID: &str =
    "a full-body studio photograph of a muscular older man with a grey beard in a red athletic \
     outfit, standing with his right arm raised straight overhead and his left hand on his hip, \
     plain light-grey seamless background, sharp focus";
const CHANGE_STYLE: &str =
    "a cel-shaded anime illustration, full body, a person in a red athletic outfit standing with \
     right arm raised straight overhead and left hand on the hip, plain light-grey background, clean \
     lineart, flat colors";
const DIM: u32 = 1024;
const STEPS: usize = 8;

/// Resolve the Q8 turnkey subdir (`KREA_TURBO_DIR` override, else the newest cached snapshot's `q8/`).
fn snapshot() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("KREA_TURBO_DIR") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--SceneWorks--krea-2-turbo-mlx/snapshots");
    let rev = std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("q8").join("transformer").is_dir())?;
    Some(rev.join("q8"))
}

/// (std, distinct-level-count, mean horizontal-adjacent-|Δ|). Coherent = broad histogram + spatial
/// smoothness; pure noise = high adjacent Δ + narrow std. Mirrors `e2e_real_weights::image_stats`.
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

/// Mean absolute per-pixel difference between two same-shape RGB8 buffers (0 = identical). A cheap
/// similarity proxy: it conflates color + structure, but a monotone trend across the sweep is enough
/// to show the strength knob moving the output between the reference and the pure-prompt anchor.
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
    let dir = std::path::Path::new("/tmp/krea_img2img_spike");
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
#[ignore = "needs the real Krea 2 Turbo snapshot (set KREA_TURBO_DIR, or cache SceneWorks/krea-2-turbo-mlx)"]
fn img2img_strength_sweep_maps_usable_window() {
    let Some(root) = snapshot() else {
        eprintln!(
            "skipping: no KREA_TURBO_DIR and no cached SceneWorks/krea-2-turbo-mlx q8 turnkey"
        );
        return;
    };
    eprintln!("loading Krea 2 Turbo (q8) from {}", root.display());
    let t_load = Instant::now();
    let pipe = KreaPipeline::from_snapshot(&root).expect("load Krea Turbo pipeline");
    eprintln!("  loaded in {:.1}s", t_load.elapsed().as_secs_f32());

    // Reference R = t2i(prompt A). The img2img init.
    let reference = pipe
        .generate_turbo(PROMPT_A, &opts(0))
        .expect("t2i reference");
    save(&reference, "00_reference_promptA");

    // Pure-B anchor = t2i(prompt B) at the sweep seed. The strength→0 limit (reference ignored).
    let pure_b = pipe.generate_turbo(PROMPT_B, &opts(1)).expect("t2i pureB");
    save(&pure_b, "01_pureB_promptB_s0");

    // Strength sweep: img2img from R with the restyle prompt B. floor(8·s) start step in the header.
    let strengths = [0.2f32, 0.35, 0.5, 0.65, 0.8, 0.9];
    let mut rows: Vec<String> = Vec::new();
    rows.push(format!(
        "{:>8}  {:>5}  {:>7} {:>8} {:>7}  {:>9}  {:>10} {:>11}",
        "strength", "start", "std", "distinct", "adjΔ", "coherent", "MAE→R", "MAE→pureB"
    ));
    // Anchors in the same table for reference.
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
            .generate_turbo_img2img(PROMPT_B, &reference, s, &opts(1))
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

    eprintln!("\n===== SUMMARY (sc-8589 img2img strength sweep) =====");
    eprintln!(
        "prompt A (reference): {PROMPT_A}\nprompt B (restyle)  : {PROMPT_B}\n{}x{DIM} · {STEPS} steps · q8\n",
        DIM
    );
    for r in &rows {
        eprintln!("{r}");
    }
    eprintln!(
        "\nRead: MAE→R should fall and MAE→pureB should rise as strength increases (the slider moving \
         the output from prompt-B toward the reference). 'coherent' must hold across the usable middle. \
         PNGs in /tmp/krea_img2img_spike for the visual GO/NO-GO + range call."
    );

    // Weak invariants (the VERDICT is the eyeball, not these): the sweep ran end-to-end and the
    // highest-strength render — which keeps the most of the reference — stays coherent.
    let hi = pipe
        .generate_turbo_img2img(PROMPT_B, &reference, 0.9, &opts(1))
        .expect("hi-strength render");
    assert!(
        is_coherent(&hi),
        "strength 0.9 img2img must remain a coherent image, not mush/noise"
    );
}

/// **sc-8589 run #2 — people / pose probe.** Can img2img take a person reference, keep the POSE, and
/// change IDENTITY or STYLE? This maps the img2img ↔ pose-ControlNet (epic 8459) boundary: img2img
/// preserves *composition* (pose rides along as structure) but is NOT an explicit pose lock —
/// changing identity/style pulls away from the reference, which also loosens pose. Renders a person
/// reference, then sweeps strength for (a) an identity swap and (b) a style swap, printing MAE→ref
/// (structure retention) and saving PNGs for the eyeball on where — if anywhere — pose survives while
/// the person/style flips. No hard assert: the verdict is visual.
#[test]
#[ignore = "needs the real Krea 2 Turbo snapshot"]
fn people_pose_identity_style_probe() {
    let Some(root) = snapshot() else {
        eprintln!("skipping: no Krea Turbo snapshot");
        return;
    };
    eprintln!("loading Krea 2 Turbo (q8) from {}", root.display());
    let pipe = KreaPipeline::from_snapshot(&root).expect("load pipeline");

    let reference = pipe
        .generate_turbo(REF_PERSON, &opts(7))
        .expect("person reference");
    save(&reference, "p00_reference_person");

    for (tag, prompt) in [("identity", CHANGE_ID), ("style", CHANGE_STYLE)] {
        eprintln!("\n--- change {tag} (keep pose) ---");
        for &s in &[0.35f32, 0.5, 0.65] {
            let start = ((STEPS as f32 * s) as usize).max(1);
            let out = pipe
                .generate_turbo_img2img(prompt, &reference, s, &opts(7))
                .expect("img2img person render");
            let m = mae(&out.pixels, &reference.pixels);
            eprintln!(
                "  {tag} strength {s:.2} (start {start}/{STEPS}): MAE->ref {m:.1} coherent={}",
                is_coherent(&out)
            );
            save(&out, &format!("p_{tag}_s{:02}", (s * 100.0) as u32));
        }
    }
    eprintln!(
        "\nEyeball /tmp/krea_img2img_spike/p_*.png: does the POSE hold while identity/style changes? \
         A usable mid-strength band = img2img covers it; if pose only holds when identity/style does \
         NOT change → pose-locked identity swap needs epic 8459 (ControlNet), not img2img."
    );
}
