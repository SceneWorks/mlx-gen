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

/// PRIMARY sc-9311 GATE (chaos-free): F-004 is about the *conditioning* the DEFAULT (unset) negative
/// prompt produces — `clip_ids("")` must keep BOS (`[BOS, EOS, pad…]`), not 77×EOS. So the load-bearing,
/// backend-agnostic check is the empty-negative uncond conditioning (`pooled [1,2048]` + `context
/// [1,333,4096]`) vs the diffusers f32 reference: deterministic, no sampler chaos. A dropped-BOS /
/// wiring regression shifts the pooled-at-argmax EOS slot and rewrites every CLIP hidden state →
/// mean relative error ≫ 0.5; a correct path matches within fp16(MLX)-vs-f32(torch) TE drift.
///
/// A full-trajectory cross-backend *pixel* A/B is deliberately NOT a gate: per CLAUDE.md it is
/// chaos-limited, and `seed=7` draws entirely different noise under MLX vs torch, so the two renders
/// are different images of the same prompt regardless of F-004 (the old pixel assert reported ~86%
/// px>8 on the *correct* path — a false failure). The render below is kept only as an end-to-end
/// coherence smoke check + an informational px>8 print.
#[test]
#[ignore = "needs the SD3.5-Large snapshot (SD3_LARGE_SNAPSHOT) + tools/golden/sd3_5_large_empty_negative_e2e.safetensors + Metal"]
fn default_empty_negative_matches_diffusers() {
    // Reference a crate symbol so the generator's `inventory::submit!` static is linked.
    assert_eq!(sd3::MODEL_ID, "sd3_5_large");

    let Some(g) = try_golden() else {
        return; // clean skip — golden absent
    };
    let dir = snapshot();

    // ---- PRIMARY: chaos-free empty-negative conditioning A/B (directly validates F-004) ----
    let (Ok(g_pooled), Ok(g_context)) = (
        g.require("neg_pooled").cloned(),
        g.require("neg_context").cloned(),
    ) else {
        panic!(
            "golden lacks neg_pooled/neg_context — re-dump with the current \
             tools/dump_sd3_empty_negative_e2e_golden.py"
        );
    };

    let encoders = sd3::loader::load_text_encoders(&dir).expect("load sd3 text encoders");
    let clip_tok = sd3::loader::load_clip_tokenizer(&dir).expect("load clip tokenizer");
    let t5_tok = sd3::loader::load_t5_tokenizer(&dir).expect("load t5 tokenizer");
    let clip_pad = sd3::loader::load_clip_pad_ids(&dir);

    // The uncond branch of a DEFAULT (unset) negative prompt encodes the empty string (F-004).
    let uncond = sd3::pipeline::encode_prompt(&encoders, &clip_tok, clip_pad, &t5_tok, "")
        .expect("encode empty negative");

    let (p_peak, p_mean) = rel(&uncond.pooled, &g_pooled);
    let (c_peak, c_mean) = rel(&uncond.context, &g_context);
    eprintln!(
        "sd3 empty-negative conditioning vs diffusers f32: pooled peak {p_peak:.4} mean {p_mean:.4}; \
         context peak {c_peak:.4} mean {c_mean:.4}"
    );
    // pooled is PURE CLIP (L+G pooled-at-argmax) — the sharpest F-004 signal. context mixes CLIP
    // (77 rows) + T5 (256 rows). Thresholds bound cross-backend TE precision drift; a BOS regression
    // (mean ≫ 0.5) sits far outside them.
    assert!(
        p_mean < 0.1 && p_peak < 1.0,
        "empty-negative POOLED diverged (F-004 CLIP path): peak {p_peak:.4} mean {p_mean:.4}"
    );
    assert!(
        c_mean < 0.1 && c_peak < 1.0,
        "empty-negative CONTEXT diverged (F-004 CLIP path): peak {c_peak:.4} mean {c_mean:.4}"
    );

    // ---- SECONDARY: end-to-end coherence smoke (NOT a cross-backend pixel gate) ----
    let gen = mlx_gen::load(sd3::MODEL_ID, &LoadSpec::new(WeightsSource::Dir(dir)))
        .expect("load sd3_5_large");
    let req = GenerationRequest {
        prompt: PROMPT.into(),
        negative_prompt: None, // the F-004 path: uncond encodes ""
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
    // Coherence: a real render is not a degenerate flat/black frame.
    let mean_byte =
        images[0].pixels.iter().map(|&b| b as f64).sum::<f64>() / images[0].pixels.len() as f64;
    assert!(
        mean_byte > 4.0 && mean_byte < 251.0,
        "render is a degenerate flat frame (mean byte {mean_byte:.1})"
    );

    // Informational only (chaos-limited cross-backend — see the fn doc): MLX vs torch noise makes a
    // pixel A/B against the golden's stored `decoded` image meaningless as a gate.
    if let Ok(dec) = g.require("decoded") {
        if let Ok(gimg) = decoded_to_image(dec) {
            eprintln!(
                "sd3 empty-negative e2e (informational, NOT gated): {:.2}% px>8 vs diffusers \
                 (different-RNG render)",
                px_gt8(&images[0], &gimg)
            );
        }
    }
}
