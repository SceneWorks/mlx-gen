//! E5 (sc-6393) — Boogu Base T2I pipeline: tokenizer parity (sc-6390) + a real-weight end-to-end
//! smoke that renders a coherent image.
//!
//! `tokenizer_matches_golden` needs the snapshot + the golden (`tools/golden_dump.py`); the e2e
//! smoke additionally needs ~all of a 128 GB Mac. Run:
//!   BOOGU_BASE_DIR=<snapshot> CARGO_TARGET_DIR=~/Repos/mlx-gen/target \
//!     cargo test -p mlx-gen-boogu --test pipeline_e2e -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_boogu::tokenizer::BooguTokenizer;
use mlx_gen_boogu::{BooguPipeline, EditOptions, GenerateOptions, TurboOptions};

fn snapshot_dir() -> PathBuf {
    PathBuf::from(std::env::var("BOOGU_BASE_DIR").expect("set BOOGU_BASE_DIR to the snapshot root"))
}

fn turbo_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("BOOGU_TURBO_DIR").expect("set BOOGU_TURBO_DIR to the Turbo snapshot root"),
    )
}

/// Edit-capable snapshot for the faithful image-conditioned edit. The vision-tower (semantic) edit
/// path is trained into the **`Boogu-Image-0.1-Edit`** fine-tune — the Base checkpoint produces
/// incoherent output for it (verified: the reference pipeline yields the same garbage on Base), so
/// the faithful e2e must validate against Edit. Falls back to `BOOGU_BASE_DIR` if `BOOGU_EDIT_DIR`
/// is unset (e.g. to reproduce the Base-is-incoherent baseline).
fn edit_dir() -> PathBuf {
    std::env::var("BOOGU_EDIT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| snapshot_dir())
}

fn golden_path() -> PathBuf {
    std::env::var("BOOGU_GOLDEN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME"))
                .join("Repos/mlx-gen-wt-boogu/reference/goldens/boogu_golden.safetensors")
        })
}

/// sc-6390 — the rendered T2I chat template + tokenizer reproduce the reference `processor`'s
/// `tok_input_ids` token-for-token for the golden prompt.
#[test]
#[ignore = "needs the mllm tokenizer + golden (tools/golden_dump.py)"]
fn tokenizer_matches_golden() {
    let g = Weights::from_file(golden_path()).expect("golden — run tools/golden_dump.py");
    let tok = BooguTokenizer::from_snapshot(snapshot_dir()).expect("load mllm tokenizer");

    let ids = tok.t2i_ids("a red apple on a wooden table").unwrap();
    let want: Vec<i32> = g
        .require("tok_input_ids")
        .unwrap()
        .as_dtype(mlx_rs::Dtype::Int32)
        .unwrap()
        .as_slice::<i32>()
        .to_vec();

    println!("boogu tokenizer: {} ids (golden {})", ids.len(), want.len());
    assert_eq!(
        ids, want,
        "T2I tokenization must match the reference processor"
    );
}

/// Real-weight end-to-end T2I smoke: render a small image and assert it is non-degenerate, saving a
/// PNG for visual inspection. (Coherence is judged by eye on the saved file.)
#[test]
#[ignore = "needs real weights (128 GB Mac): set BOOGU_BASE_DIR"]
fn t2i_smoke() {
    let pipe = BooguPipeline::from_snapshot(snapshot_dir()).expect("load Boogu pipeline");
    let opts = GenerateOptions {
        height: 512,
        width: 512,
        steps: 28,
        text_guidance_scale: 4.0,
        seed: 0,
    };
    let img = pipe
        .generate("a red apple on a wooden table", &opts)
        .expect("generate");

    assert_eq!(img.width, 512);
    assert_eq!(img.height, 512);
    assert_eq!(img.pixels.len(), 512 * 512 * 3);

    // Non-degenerate: a real render has spread across the 0..255 range, not a flat fill.
    let (mn, mx) = img
        .pixels
        .iter()
        .fold((255u8, 0u8), |(mn, mx), &p| (mn.min(p), mx.max(p)));
    let mean = img.pixels.iter().map(|&p| p as u64).sum::<u64>() / img.pixels.len() as u64;
    println!("render stats: min={mn} max={mx} mean={mean}");
    assert!(mx - mn > 32, "render looks degenerate (min={mn} max={mx})");

    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../reference/outputs/boogu_mlx_t2i_apple_512_s28.png");
    std::fs::create_dir_all(out.parent().unwrap()).unwrap();
    image::save_buffer(
        &out,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    println!("wrote {}", out.display());
}

/// Save an RGB8 [`mlx_gen::media::Image`] to `reference/outputs/{name}` and assert it is
/// non-degenerate (real spread, not a flat fill). Returns nothing; panics on a bad render.
fn save_and_check(img: &mlx_gen::media::Image, name: &str, label: &str) {
    assert_eq!((img.width, img.height), (512, 512));
    let (mn, mx) = img
        .pixels
        .iter()
        .fold((255u8, 0u8), |(mn, mx), &p| (mn.min(p), mx.max(p)));
    let mean = img.pixels.iter().map(|&p| p as u64).sum::<u64>() / img.pixels.len() as u64;
    println!("{label} render stats: min={mn} max={mx} mean={mean}");
    assert!(
        mx - mn > 32,
        "{label} render looks degenerate (min={mn} max={mx})"
    );

    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../reference/outputs")
        .join(name);
    std::fs::create_dir_all(out.parent().unwrap()).unwrap();
    image::save_buffer(
        &out,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    println!("wrote {}", out.display());
}

/// E7b-3 real-weight Edit (single-reference TI2I) smoke: T2I-render a reference (on Base), then run
/// the **edit** path with an instruction TWICE — image-conditioned (the faithful default: reference →
/// Qwen3-VL vision tower → image-conditioned MLLM forward + spatial ref latent) and text-only
/// (`condition_on_image = false`, the E7 baseline: spatial ref latent only). All three PNGs are saved
/// for visual A/B. Coherence is judged by eye on the saved files.
///
/// Checkpoint note: the vision-tower (semantic) edit path is trained into the **Edit** fine-tune, not
/// Base — point `BOOGU_EDIT_DIR` at `Boogu-Image-0.1-Edit` for a coherent faithful edit. On Base the
/// image-conditioned path is incoherent *by design* (the reference pipeline produces the same garbage
/// on Base), while the text-only spatial path still renders a plausible edit. The reference image is
/// always T2I-rendered on Base (`BOOGU_BASE_DIR`); the edit runs on [`edit_dir`].
#[test]
#[ignore = "needs real weights (128 GB Mac): set BOOGU_BASE_DIR (+ BOOGU_EDIT_DIR for the faithful path)"]
fn edit_smoke() {
    // Reference image (T2I) on Base, in a scope so the Base pipeline (~40 GB) is dropped before the
    // Edit pipeline loads — keeps peak memory at one pipeline, not two.
    let ref_img = {
        let base = BooguPipeline::from_snapshot(snapshot_dir()).expect("load Boogu Base pipeline");
        base.generate(
            "a red apple on a wooden table",
            &GenerateOptions {
                height: 512,
                width: 512,
                steps: 24,
                text_guidance_scale: 4.0,
                seed: 0,
            },
        )
        .expect("generate reference")
    };
    save_and_check(&ref_img, "boogu_mlx_edit_ref_apple_512.png", "reference");

    // The edit itself uses the Edit-capable snapshot (Edit fine-tune for the faithful vision path).
    let pipe = BooguPipeline::from_snapshot(edit_dir()).expect("load Boogu Edit pipeline");

    let instruction = "change the apple to a green apple";

    // Faithful image-conditioned edit (E7b-3 default).
    let edited = pipe
        .generate_edit(
            &ref_img,
            instruction,
            &EditOptions {
                height: 512,
                width: 512,
                steps: 24,
                text_guidance_scale: 4.0,
                seed: 1,
                condition_on_image: true,
                use_input_images_4_neg_instruct: false,
            },
        )
        .expect("generate_edit (image-conditioned)");
    save_and_check(
        &edited,
        "boogu_mlx_edit_green_apple_512.png",
        "edit (image-conditioned)",
    );

    // Text-only baseline (E7 path) for A/B comparison — same seed/instruction, no vision tower.
    let edited_text_only = pipe
        .generate_edit(
            &ref_img,
            instruction,
            &EditOptions {
                height: 512,
                width: 512,
                steps: 24,
                text_guidance_scale: 4.0,
                seed: 1,
                condition_on_image: false,
                use_input_images_4_neg_instruct: false,
            },
        )
        .expect("generate_edit (text-only)");
    save_and_check(
        &edited_text_only,
        "boogu_mlx_edit_green_apple_512_textonly.png",
        "edit (text-only baseline)",
    );
}

/// Real-weight Turbo (DMD few-step) smoke: render with the Turbo checkpoint + DMD sampler (no CFG)
/// and assert a non-degenerate image, saving a PNG. Needs the Turbo snapshot (`BOOGU_TURBO_DIR`).
#[test]
#[ignore = "needs real Turbo weights (128 GB Mac): set BOOGU_TURBO_DIR"]
fn turbo_smoke() {
    let pipe = BooguPipeline::from_snapshot(turbo_dir()).expect("load Boogu Turbo pipeline");
    let opts = TurboOptions {
        height: 768,
        width: 768,
        steps: 4,
        seed: 0,
        conditioning_sigma: 0.001,
    };
    let img = pipe
        .generate_turbo("a red apple on a wooden table", &opts)
        .expect("generate_turbo");

    assert_eq!((img.width, img.height), (768, 768));
    let (mn, mx) = img
        .pixels
        .iter()
        .fold((255u8, 0u8), |(mn, mx), &p| (mn.min(p), mx.max(p)));
    let mean = img.pixels.iter().map(|&p| p as u64).sum::<u64>() / img.pixels.len() as u64;
    println!("turbo render stats: min={mn} max={mx} mean={mean}");
    assert!(
        mx - mn > 32,
        "turbo render looks degenerate (min={mn} max={mx})"
    );

    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../reference/outputs/boogu_mlx_turbo_apple_768_s4.png");
    std::fs::create_dir_all(out.parent().unwrap()).unwrap();
    image::save_buffer(
        &out,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    println!("wrote {}", out.display());
}
