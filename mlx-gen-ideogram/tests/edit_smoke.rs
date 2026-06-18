//! sc-6303/6330 — Ideogram 4 **edit** end-to-end smoke: img2img (Remix) and mask inpaint (Edit) on
//! the real converted weights. Proves the edit path runs end-to-end and that the inpaint mask
//! actually routes keep-vs-repaint (a caption-independent correctness check, not bit-parity).
//!
//! `#[ignore]` — needs the converted snapshot (~53 GB). Run:
//!   IDEOGRAM4_MLX=~/.cache/ideogram4-mlx-convert \
//!     cargo test -p mlx-gen-ideogram --test edit_smoke -- --ignored --nocapture

mod common;

use std::path::PathBuf;

use common::CAPTION_JSON;
use mlx_gen::array::host_i32;
use mlx_gen::media::Image;
use mlx_gen::CancelFlag;
use mlx_gen_ideogram::Ideogram4Pipeline;
use mlx_rs::{Array, Dtype};

fn snapshot_dir() -> PathBuf {
    std::env::var("IDEOGRAM4_MLX")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME")).join(".cache/ideogram4-mlx-convert")
        })
}

fn envn(k: &str, d: u32) -> u32 {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

/// A structured RGB gradient source — distinguishable from a fresh generation so the VAE round-trip
/// (the keep region) can be told apart from regenerated pixels.
fn gradient_source(w: u32, h: u32) -> Image {
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = (255 * x / w.max(1)) as u8;
            let g = (255 * y / h.max(1)) as u8;
            let b = (255 * (x + y) / (w + h).max(1)) as u8;
            pixels.extend_from_slice(&[r, g, b]);
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

/// A solid grayscale mask image (`v` in 0..=255 for every pixel).
fn solid_mask(w: u32, h: u32, v: u8) -> Image {
    Image {
        width: w,
        height: h,
        pixels: vec![v; (w * h * 3) as usize],
    }
}

fn to_u8(img: &Array) -> Vec<u8> {
    host_i32(&img.as_dtype(Dtype::Int32).unwrap())
        .unwrap()
        .into_iter()
        .map(|v| v as u8)
        .collect()
}

fn mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum::<f64>()
        / a.len() as f64
}

fn assert_valid(img: &Array, h: u32, w: u32) -> Vec<u8> {
    assert_eq!(img.shape(), &[h as i32, w as i32, 3], "image shape");
    let px = to_u8(img);
    let (min, max) = (*px.iter().min().unwrap(), *px.iter().max().unwrap());
    assert!(max > min, "degenerate (constant) image — no signal");
    px
}

/// img2img (Remix): denoise from the noised source latent at a strength-derived step.
#[test]
#[ignore = "needs converted weights (~53 GB)"]
fn img2img_smoke() {
    let pipe = Ideogram4Pipeline::load(&snapshot_dir()).expect("load pipeline");
    let ids = pipe.tokenize(CAPTION_JSON).expect("tokenize");
    let res = envn("IDEOGRAM4_SMOKE_RES", 256);
    let (h, w) = (res, res);
    let steps = envn("IDEOGRAM4_SMOKE_STEPS", 50) as usize;
    let strength = std::env::var("IDEOGRAM4_SMOKE_STRENGTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.6f32);

    let source = gradient_source(w, h);
    let edit = pipe
        .prepare_edit(&source, None, strength, h, w)
        .expect("prepare img2img");
    let img = pipe
        .generate_edit_with_progress(
            &ids,
            h,
            w,
            steps,
            7.0,
            0.0,
            0,
            &edit,
            &CancelFlag::new(),
            &mut |_| {},
        )
        .expect("img2img generate");
    let px = assert_valid(&img, h, w);
    let out = std::env::temp_dir().join("ideogram4_img2img.png");
    image::RgbImage::from_raw(w, h, px)
        .unwrap()
        .save(&out)
        .unwrap();
    println!(
        "img2img strength {strength} @ {h}x{w}/{steps} → wrote {}",
        out.display()
    );
}

/// Mask inpaint (Edit): an all-black mask keeps the whole image pinned to the source (≈ VAE
/// round-trip), an all-white mask regenerates everywhere (= plain img2img). With the same seed +
/// source, the keep-all output must track the source far more closely than the repaint-all output —
/// a caption-independent proof that the mask routes keep-vs-repaint correctly.
#[test]
#[ignore = "needs converted weights (~53 GB)"]
fn inpaint_mask_routes_keep_vs_repaint() {
    let pipe = Ideogram4Pipeline::load(&snapshot_dir()).expect("load pipeline");
    let ids = pipe.tokenize(CAPTION_JSON).expect("tokenize");
    let res = envn("IDEOGRAM4_SMOKE_RES", 256);
    let (h, w) = (res, res);
    let steps = envn("IDEOGRAM4_SMOKE_STEPS", 50) as usize;
    let strength = 0.85f32;
    let seed = 7u64;

    let source = gradient_source(w, h);
    let src_px: Vec<u8> = source.pixels.clone();

    let run = |mask_v: u8| -> Vec<u8> {
        let mask = solid_mask(w, h, mask_v);
        let edit = pipe
            .prepare_edit(&source, Some(&mask), strength, h, w)
            .expect("prepare inpaint");
        let img = pipe
            .generate_edit_with_progress(
                &ids,
                h,
                w,
                steps,
                7.0,
                0.0,
                seed,
                &edit,
                &CancelFlag::new(),
                &mut |_| {},
            )
            .expect("inpaint generate");
        assert_valid(&img, h, w)
    };

    let keep_all = run(0); // all-black mask → keep the source everywhere
    let repaint_all = run(255); // all-white mask → regenerate everywhere (= img2img)

    let keep_diff = mean_abs_diff(&keep_all, &src_px);
    let repaint_diff = mean_abs_diff(&repaint_all, &src_px);
    println!(
        "keep-all |Δsource| = {keep_diff:.2}, repaint-all |Δsource| = {repaint_diff:.2} \
         (@ {h}x{w}/{steps}, strength {strength})"
    );
    // The keep-all (mask 0) output is the VAE round-trip of the source; the repaint-all (mask 1)
    // output is a fresh generation. The mask routing is correct iff keep tracks the source much
    // more closely than repaint does.
    assert!(
        keep_diff < repaint_diff,
        "inpaint mask did not pin the keep region (keep {keep_diff:.2} !< repaint {repaint_diff:.2})"
    );

    for (tag, px) in [("keep_all", &keep_all), ("repaint_all", &repaint_all)] {
        let out = std::env::temp_dir().join(format!("ideogram4_inpaint_{tag}.png"));
        image::RgbImage::from_raw(w, h, px.clone())
            .unwrap()
            .save(&out)
            .unwrap();
        println!("wrote {}", out.display());
    }
}
