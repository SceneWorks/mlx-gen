//! SD3.5-Large **default empty-negative** end-to-end real-weight golden (sc-9311, follow-up to F-004 /
//! sc-9090).
//!
//! ## What this gate covers that the other e2e tests do NOT
//!
//! The existing `e2e_real_weights` smoke renders with an EXPLICIT negative prompt
//! (`negative_prompt: Some("blurry, low quality, distorted")`). That path never exercises the
//! **default uncond branch** — the one taken when `req.negative_prompt` is unset, which encodes the
//! *empty* string `""` through `pipeline::clip_ids("")`. That empty-CLIP path is exactly the F-004 fix
//! (an earlier `is_empty() → Vec::new()` shortcut dropped the BOS, producing 77×EOS and shifting the
//! pooled-at-argmax EOS selection — same bug family as z-image sc-8958). sc-9090 added a *synthetic*
//! equivalence unit test (`pipeline::tests::empty_prompt_clip_ids_keep_bos_and_match_tokenize_path`),
//! but the DEFAULT path was still not golden-covered end-to-end on real weights. This test closes that
//! gap: it renders with `negative_prompt = None` and A/B's the result against a diffusers reference
//! dumped with an **empty** negative (`negative_prompt=""`).
//!
//! ## Requirements + how to run (`#[ignore]`d — never runs on a fresh `cargo test`)
//!
//! Needs the licensed `stabilityai/stable-diffusion-3.5-large` snapshot (or `SD3_LARGE_SNAPSHOT`) +
//! Metal, AND the diffusers reference golden `tools/golden/sd3_5_large_empty_negative_e2e.safetensors`
//! (gitignored — it cannot be committed: it derives from licensed weights and needs a torch/diffusers
//! env this workspace does not have). Produce the golden from the frozen mflux fork's diffusers venv,
//! then run the test:
//!
//! ```sh
//! # 1. dump the diffusers reference (empty negative, f32, 256²/20-step true-CFG):
//! cd ~/repos/mflux && .venv-0312/bin/python ~/repos/mlx-gen/tools/dump_sd3_empty_negative_e2e_golden.py
//! # 2. run the gate:
//! SD3_LARGE_SNAPSHOT=/path/to/stable-diffusion-3.5-large \
//!   cargo test -p mlx-gen-sd3 --release --test e2e_empty_negative_real_weights -- --ignored --nocapture
//! ```
//!
//! When the golden is ABSENT the test prints a clear "run the dump script" message and returns Ok
//! (a clean skip) rather than panicking confusingly — so `--ignored` on a machine without the fixture
//! is a no-op, not a spurious failure.

use std::path::PathBuf;

use mlx_gen::image::decoded_to_image;
use mlx_gen::media::Image;
use mlx_gen::weights::Weights;
use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};
use mlx_rs::{Array, Dtype};

// Force the linker to keep `mlx-gen-sd3`'s `inventory::submit!` registration static (reached only via
// the `mlx_gen::load` registry — the CLAUDE.md "Linkage gotcha").
use mlx_gen_sd3 as sd3;

/// The render prompt — must match `PROMPT` in `tools/dump_sd3_empty_negative_e2e_golden.py`.
const PROMPT: &str = "a photograph of a red fox sitting in a green meadow, sharp focus, daylight";
/// Small/fast render so the f32 diffusers dump is feasible — must match the dump script.
const SIZE: u32 = 256;
const STEPS: u32 = 20;
const GUIDANCE: f32 = 3.5;
const SEED: u64 = 7;

/// Resolve the SD3.5-Large snapshot dir: `SD3_LARGE_SNAPSHOT` override, else the first snapshot in the
/// HF hub cache (mirrors `e2e_real_weights::snapshot`).
fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("SD3_LARGE_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").expect("HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--stabilityai--stable-diffusion-3.5-large/snapshots");
    std::fs::read_dir(&snaps)
        .unwrap_or_else(|_| panic!("no SD3.5-Large snapshots under {snaps:?}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("set SD3_LARGE_SNAPSHOT or populate the HF hub cache")
}

/// The diffusers empty-negative reference golden path (env override for a non-default location).
fn golden_path() -> PathBuf {
    if let Ok(p) = std::env::var("SD3_EMPTY_NEG_GOLDEN") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tools/golden/sd3_5_large_empty_negative_e2e.safetensors")
}

/// Load the golden, or `None` (with a clear message) when it is absent — a clean skip, NOT a panic.
fn try_golden() -> Option<Weights> {
    let path = golden_path();
    match Weights::from_file(&path) {
        Ok(w) => Some(w),
        Err(_) => {
            eprintln!(
                "SKIP e2e_empty_negative: golden {} not found — dump it with \
                 tools/dump_sd3_empty_negative_e2e_golden.py (see the file-level doc). \
                 Set SD3_EMPTY_NEG_GOLDEN to point at a non-default location.",
                path.display()
            );
            None
        }
    }
}

/// Percentage of RGB8 bytes differing by more than 8 levels (the crate's cross-build px>8 convention).
fn px_gt8(a: &Image, b: &Image) -> f32 {
    assert_eq!(a.pixels.len(), b.pixels.len(), "image size mismatch");
    let n = a.pixels.len();
    let c = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .filter(|(x, y)| (**x as i32 - **y as i32).abs() > 8)
        .count();
    100.0 * c as f32 / n as f32
}

fn f32a(a: &Array) -> Array {
    a.as_dtype(Dtype::Float32).unwrap()
}

/// (peak, mean) relative error of `a` vs reference `b`, both flattened (mirrors flux2's `rel`).
fn rel(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let a = f32a(a).reshape(&[n]).unwrap();
    let b = f32a(b).reshape(&[n]).unwrap();
    let (xs, ys) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = ys.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let mabs = (ys.iter().map(|y| y.abs()).sum::<f32>() / ys.len() as f32).max(1e-12);
    let max_d = xs
        .iter()
        .zip(ys)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let mean_d = xs.iter().zip(ys).map(|(x, y)| (x - y).abs()).sum::<f32>() / xs.len() as f32;
    (max_d / peak, mean_d / mabs)
}

/// PRIMARY sc-9311 GATE: render with the DEFAULT (unset) negative prompt and A/B the decoded image
/// against the diffusers empty-negative reference. This is the only e2e path that exercises
/// `clip_ids("")` for the uncond branch (F-004). A wiring/BOS regression → ~100% px>8; a correct
/// empty-negative path + cross-backend f32 drift over a 20-step true-CFG sampler is bounded.
#[test]
#[ignore = "needs the SD3.5-Large snapshot (SD3_LARGE_SNAPSHOT) + tools/golden/sd3_5_large_empty_negative_e2e.safetensors + Metal"]
fn default_empty_negative_matches_diffusers() {
    // Reference a crate symbol so the generator's `inventory::submit!` static is linked.
    assert_eq!(sd3::MODEL_ID, "sd3_5_large");

    let Some(g) = try_golden() else {
        return; // clean skip — golden absent
    };

    let gen = mlx_gen::load(
        sd3::MODEL_ID,
        &LoadSpec::new(WeightsSource::Dir(snapshot())),
    )
    .expect("load sd3_5_large");

    // The load-bearing bit: negative_prompt = None → the pipeline encodes "" for the uncond branch.
    let req = GenerationRequest {
        prompt: PROMPT.into(),
        negative_prompt: None,
        width: SIZE,
        height: SIZE,
        count: 1,
        seed: Some(SEED),
        steps: Some(STEPS),
        guidance: Some(GUIDANCE),
        ..Default::default()
    };

    let out = gen.generate(&req, &mut |_| {}).expect("generate");
    let GenerationOutput::Images(images) = out else {
        panic!("expected Images");
    };
    assert_eq!(images.len(), 1);
    assert_eq!((images[0].width, images[0].height), (SIZE, SIZE));

    let gimg = decoded_to_image(g.require("decoded").expect("golden `decoded` latent")).unwrap();
    let px = px_gt8(&images[0], &gimg);

    // If the golden also stored the empty-negative uncond conditioning, report the tighter chaos-free
    // signal too (optional — older dumps may not have it).
    if let (Ok(neg_pooled), Ok(neg_ctx)) = (
        g.require("neg_pooled").cloned(),
        g.require("neg_context").cloned(),
    ) {
        eprintln!(
            "empty-negative uncond conditioning present in golden (pooled {:?}, context {:?})",
            neg_pooled.shape(),
            neg_ctx.shape()
        );
        let _ = rel; // `rel` is available for a conditioning A/B when a matching dump exists.
    }

    eprintln!(
        "sd3 empty-negative e2e: {px:.2}% px>8 vs diffusers f32 (default unset-negative uncond path)"
    );
    // Wiring/BOS regression → ~100%; correct empty-negative path + cross-backend f32 drift over a
    // 20-step true-CFG sampler is bounded.
    assert!(
        px < 25.0,
        "default empty-negative render diverged from the diffusers empty-negative reference: \
         {px:.2}% px>8 — this points at the F-004 empty-CLIP uncond path (clip_ids(\"\") BOS)"
    );
}
