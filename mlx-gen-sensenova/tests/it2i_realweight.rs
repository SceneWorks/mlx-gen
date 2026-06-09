//! sc-3189: real-weight (35GB) it2i (edit) end-to-end parity — `#[ignore]`, run locally.
//!
//! Loads the actual checkpoint, runs [`T2iModel::it2i_generate`] in the edit flow (`cfg_scale=4`,
//! `img_cfg_scale=1`) on a deterministic source image with the reference's injected noise, and
//! checks directional/structural similarity to the reference final image
//! (`tools/dump_sensenova_it2i_realweight.py`). Also asserts [`T2iModel::preprocess_image`] matches
//! the reference `pixel_values` and the prompt query encodes to the reference's condition ids.
//!
//! Run: `cargo test -p mlx-gen-sensenova --test it2i_realweight -- --ignored --nocapture`

use std::path::PathBuf;

use mlx_gen_sensenova::{
    loader::load_raw, text::load_tokenizer, NeoChatConfig, T2iModel, T2iOptions,
};
use mlx_rs::Array;

const DEFAULT_SNAPSHOT: &str = concat!(
    env!("HOME"),
    "/.cache/huggingface/hub/models--sensenova--SenseNova-U1-8B-MoT/snapshots/\
     bfa9b436503cb8aed4f2bc60e3236710cc77468d"
);

fn snapshot_dir() -> PathBuf {
    std::env::var("SENSENOVA_SNAPSHOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_SNAPSHOT))
}

fn fixture() -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/it2i_realweight_golden.safetensors"
    ))
}

fn flat(a: &Array) -> Vec<f32> {
    let n = a.shape().iter().product::<i32>();
    a.reshape(&[n]).unwrap().as_slice::<f32>().to_vec()
}

fn peak_rel(a: &Array, b: &Array) -> f32 {
    let (a, b) = (flat(a), flat(b));
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-6);
    a.iter()
        .zip(&b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()))
        / peak
}

#[test]
#[ignore = "needs the local 35GB checkpoint + dumped golden; run with --ignored"]
fn it2i_realweight_matches_reference() {
    let snap = snapshot_dir();
    let fix = fixture();
    if !snap.exists() || !fix.exists() {
        eprintln!(
            "skipping: snapshot or golden missing — see tools/dump_sensenova_it2i_realweight.py"
        );
        return;
    }

    let golden = mlx_gen::weights::Weights::from_file(&fix).expect("load golden");
    let prompt = golden.metadata("prompt").unwrap();
    let width: i32 = golden.metadata("width").unwrap().parse().unwrap();
    let height: i32 = golden.metadata("height").unwrap().parse().unwrap();
    let num_steps: usize = golden.metadata("num_steps").unwrap().parse().unwrap();
    let cfg_scale: f32 = golden.metadata("cfg").unwrap().parse().unwrap();
    let img_cfg_scale: f32 = golden.metadata("img_cfg").unwrap().parse().unwrap();
    let src = golden.require("src").unwrap().clone(); // [3, H, W] in [0,1]
    let want_pixels = golden.require("pixel_values").unwrap().clone();
    let raw_noise = golden.require("raw_noise").unwrap().clone();
    let want_image = golden.require("final_image").unwrap().clone();

    println!("loading checkpoint {} …", snap.display());
    let cfg = NeoChatConfig::from_dir(&snap).expect("config");
    let weights = load_raw(&snap).expect("weights");
    let tokenizer = load_tokenizer(&snap).expect("tokenizer");
    let model = T2iModel::from_weights(&weights, &cfg).expect("build T2iModel");

    // preprocess_image must match the reference pixel_values.
    let (pix, (gh, gw)) = model.preprocess_image(&src).expect("preprocess");
    let pp_rel = peak_rel(&pix, &want_pixels);
    println!("preprocess pixel_values: peak-rel={pp_rel:.3e}  grid={gh}x{gw}");
    assert!(pp_rel < 1e-4, "preprocess_image mismatch {pp_rel:.3e}");

    // Localize: does the condition query encode to the reference ids?
    let want_cond_ids: Vec<i32> = golden
        .require("cond_input_ids")
        .unwrap()
        .as_slice::<i32>()
        .to_vec();
    let n = (gh / 2) * (gw / 2);
    let q = format!(
        "{}<think>\n\n</think>\n\n<img>",
        mlx_gen_sensenova::build_neo1_query(
            &format!("<image>\n{prompt}"),
            mlx_gen_sensenova::SYSTEM_MESSAGE_FOR_GEN
        )
    )
    .replacen(
        "<image>",
        &format!("<img>{}</img>", "<IMG_CONTEXT>".repeat(n as usize)),
        1,
    );
    let got_cond_ids = tokenizer.encode_ids(&q, true).unwrap();
    println!(
        "cond ids: got {} vs want {}",
        got_cond_ids.len(),
        want_cond_ids.len()
    );
    assert_eq!(got_cond_ids, want_cond_ids, "condition query ids mismatch");

    let cosine = |got: &Array, want: &Array| -> f64 {
        let (g, w) = (flat(got), flat(want));
        let dot: f64 = g.iter().zip(&w).map(|(&a, &b)| a as f64 * b as f64).sum();
        let na: f64 = g.iter().map(|&a| (a as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = w.iter().map(|&b| (b as f64).powi(2)).sum::<f64>().sqrt();
        dot / (na * nb + 1e-12)
    };

    // Understanding vision features — bit-near (the source-image conditioning input).
    let want_vit = golden.require("und_vit").unwrap().clone();
    let got_vit = model
        .und_vision_features(&pix, &[(gh, gw)])
        .expect("und vit");
    let vit_rel = peak_rel(&got_vit, &want_vit);
    println!("und vision features: peak-rel={vit_rel:.3e}");
    assert!(
        vit_rel < 5e-3,
        "und vision features {vit_rel:.3e} exceeds 5e-3"
    );

    // Understanding-path prefill hidden (image-conditioned) — directional (deep f32 floor in peak).
    let want_ph = golden.require("cond_prefill_hidden").unwrap().clone();
    let got_ph = model
        .prefill_it2i_hidden(&got_cond_ids, Some(&pix), &[(gh, gw)])
        .expect("prefill hidden");
    let ph_cos = cosine(&got_ph, &want_ph);
    println!(
        "cond prefill hidden: cosine={ph_cos:.5}  peak-rel={:.3e}",
        peak_rel(&got_ph, &want_ph)
    );
    assert!(
        ph_cos > 0.999,
        "prefill hidden cosine {ph_cos:.5} below 0.999"
    );

    // The strong e2e signal: image-conditioned generation (cond-only, no CFG amplification).
    let want_cond_only = golden.require("final_cond_only").unwrap().clone();
    let cond_opts = T2iOptions {
        cfg_scale: 1.0,
        img_cfg_scale: 1.0,
        num_steps,
        ..Default::default()
    };
    let out_co = model
        .it2i_generate(
            &tokenizer,
            prompt,
            std::slice::from_ref(&src),
            width,
            height,
            &cond_opts,
            Some(&raw_noise),
        )
        .expect("it2i_generate cond-only");
    let co_cos = cosine(&out_co.image, &want_cond_only);
    println!(
        "cond-only e2e: cosine={co_cos:.5}  peak-rel={:.3e}",
        peak_rel(&out_co.image, &want_cond_only)
    );
    assert!(co_cos > 0.99, "cond-only e2e cosine {co_cos:.5} below 0.99");

    // The dual-guidance edit (cfg + img_cfg). CFG amplifies the deep-backbone f32 floor, and the
    // reference itself is precision-unstable at high cfg, so this is a directional check.
    let opts = T2iOptions {
        cfg_scale,
        img_cfg_scale,
        num_steps,
        ..Default::default()
    };
    let out = model
        .it2i_generate(
            &tokenizer,
            prompt,
            std::slice::from_ref(&src),
            width,
            height,
            &opts,
            Some(&raw_noise),
        )
        .expect("it2i_generate");
    assert!(flat(&out.image).iter().all(|v| v.is_finite()));
    let edit_cos = cosine(&out.image, &want_image);
    println!(
        "it2i edit e2e (cfg={cfg_scale}, img_cfg={img_cfg_scale}): cosine={edit_cos:.5}  peak-rel={:.3e}",
        peak_rel(&out.image, &want_image)
    );
    assert!(edit_cos > 0.85, "edit e2e cosine {edit_cos:.5} below 0.85");
}
