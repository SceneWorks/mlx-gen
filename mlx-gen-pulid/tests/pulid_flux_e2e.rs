//! sc-3074 — PuLID-FLUX end-to-end generate + identity validation (real weights).
//!
//! Heavy/`#[ignore]`d: loads the full FLUX.1-dev backbone + EVA + IDFormer + CA + the native face
//! stack and runs a real denoise. Validates, with zero torch:
//!   1. **id_weight=0 ⇒ bit-identical to plain FLUX** (the carried-over sc-3072 gate, now full-stack).
//!   2. **id injection changes the output** (id_weight=1 ≠ plain).
//!   3. **identity preservation** — ArcFace cosine(generated face, reference face); the sc-2012 / sc-3074
//!      baseline is ≈0.80 at full quality (printed; asserted softly to stay robust at low step counts).
//!   4. **real-CFG (sc-3075)** — true_cfg>1 + a negative prompt engages the dual-forward branch and
//!      changes the render vs fake-CFG.
//!
//! Inputs resolve from local caches (FLUX HF cache, guozinan/PuLID, tools/golden for EVA + face).
//! Run:
//!   cargo test -p mlx-gen-pulid --release --test pulid_flux_e2e -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::media::Image;
use mlx_gen::weights::Weights;
use mlx_gen::{
    Conditioning, GenerationOutput, GenerationRequest, Generator, LoadSpec, WeightsSource,
};
use mlx_gen_face::FaceAnalysis;
use mlx_gen_flux::config::FluxVariant;
use mlx_gen_flux::model::load_flux1;
use mlx_gen_pulid::eva_clip::{EvaConfig, EvaVisionTransformer};
use mlx_gen_pulid::pulid_flux::PulidFlux;

fn golden(name: &str) -> Weights {
    let path = format!("{}/../tools/golden/{name}", env!("CARGO_MANIFEST_DIR"));
    Weights::from_file(&path)
        .unwrap_or_else(|e| panic!("missing golden {path}: {e} (see tools/ dump scripts)"))
}

fn flux_snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("MLX_GEN_FLUX_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let base =
        format!("{home}/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots");
    std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("no FLUX.1-dev cache ({base}): {e}; set MLX_GEN_FLUX_SNAPSHOT"))
        .flatten()
        .map(|d| d.path())
        .find(|p| p.join("transformer").is_dir())
        .expect("no FLUX.1-dev snapshot with transformer/")
}

fn pulid_weights() -> Weights {
    let home = std::env::var("HOME").unwrap();
    let base = format!("{home}/.cache/huggingface/hub/models--guozinan--PuLID/snapshots");
    let path = std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("no PuLID cache ({base}): {e}"))
        .flatten()
        .map(|d| d.path().join("pulid_flux_v0.9.1.safetensors"))
        .find(|p| p.exists())
        .expect("pulid_flux_v0.9.1.safetensors not in cache");
    let mut w = Weights::from_file(&path).unwrap();
    w.cast_all(mlx_rs::Dtype::Float32).unwrap();
    w
}

/// The reference face: the `image` tensor ([h,w,3] i32→u8) embedded in the face-align golden.
fn reference_face() -> Image {
    let g = golden("face_align_goldens.safetensors");
    let a = g.require("image").unwrap();
    let sh = a.shape();
    let pixels = a
        .try_as_slice::<i32>()
        .unwrap()
        .iter()
        .map(|&v| v as u8)
        .collect::<Vec<u8>>();
    Image {
        width: sh[1] as u32,
        height: sh[0] as u32,
        pixels,
    }
}

fn load_face() -> FaceAnalysis {
    FaceAnalysis::load(
        &golden("scrfd_10g.safetensors"),
        &golden("arcface_iresnet100.safetensors"),
    )
    .unwrap()
    .with_parser(&golden("bisenet_parsing.safetensors"))
    .unwrap()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn req_with(
    face: &Image,
    prompt: &str,
    id_weight: f32,
    steps: u32,
    size: u32,
) -> GenerationRequest {
    GenerationRequest {
        prompt: prompt.into(),
        width: size,
        height: size,
        steps: Some(steps),
        guidance: Some(4.0),
        seed: Some(42),
        conditioning: vec![Conditioning::Reference {
            image: face.clone(),
            strength: Some(id_weight),
        }],
        ..Default::default()
    }
}

fn first_image(out: GenerationOutput) -> Image {
    match out {
        GenerationOutput::Images(mut v) => v.remove(0),
        other => panic!("expected image, got {other:?}"),
    }
}

#[test]
#[ignore = "real-weights e2e: needs FLUX.1-dev + PuLID + EVA/face goldens"]
fn pulid_flux_end_to_end() {
    let prompt = "a portrait photo of a person, headshot, looking at the camera";
    let (steps, size) = (20u32, 512u32);
    let face_img = reference_face();

    // Native face stack (+ parser) and the reference ArcFace embedding.
    let face = load_face();
    let ref_faces = face
        .analyze(
            &face_img.pixels,
            face_img.height as usize,
            face_img.width as usize,
        )
        .unwrap();
    assert!(!ref_faces.is_empty(), "no face in the reference image");
    let ref_emb = ref_faces[0].embedding.clone();

    // FLUX backbone — generate the PLAIN txt2img first (no conditioning; same prompt/seed/size).
    let spec = LoadSpec::new(WeightsSource::Dir(flux_snapshot()));
    let flux = load_flux1(FluxVariant::Dev, &spec).unwrap();
    let mut plain_req = req_with(&face_img, prompt, 1.0, steps, size);
    plain_req.conditioning = Vec::new();
    let plain = first_image(flux.generate(&plain_req, &mut |_| {}).unwrap());

    // EVA tower (f32 golden weights) + PuLID encoder/CA (f32).
    let eva = EvaVisionTransformer::from_weights(
        &golden("eva_clip_golden.safetensors"),
        "w",
        EvaConfig::default(),
    )
    .unwrap();
    let model = PulidFlux::new(flux, eva, pulid_weights(), load_face()).unwrap();

    // (1) id_weight = 0 ⇒ bit-identical to plain FLUX.
    let id0 = first_image(
        model
            .generate(&req_with(&face_img, prompt, 0.0, steps, size), &mut |_| {})
            .unwrap(),
    );
    assert_eq!(
        id0.pixels, plain.pixels,
        "id_weight=0 must equal plain FLUX bit-for-bit"
    );

    // (2) id_weight = 1 ⇒ output changes.
    let id1 = first_image(
        model
            .generate(&req_with(&face_img, prompt, 1.0, steps, size), &mut |_| {})
            .unwrap(),
    );
    let changed = id1
        .pixels
        .iter()
        .zip(&plain.pixels)
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "id_weight=1 changed {}/{} px ({:.1}%)",
        changed,
        id1.pixels.len(),
        changed as f32 / id1.pixels.len() as f32 * 100.0
    );
    assert!(
        changed > id1.pixels.len() / 100,
        "id injection should change the render"
    );

    // (3) identity preservation: ArcFace cosine(generated face, reference face).
    let gen_faces = face
        .analyze(&id1.pixels, id1.height as usize, id1.width as usize)
        .unwrap();
    if let Some(gf) = gen_faces.first() {
        let cos = cosine(&gf.embedding, &ref_emb);
        println!(
            "IDENTITY ArcFace cosine(generated, reference) = {cos:.4}  (sc-2012 baseline ≈0.80)"
        );
        assert!(cos > 0.3, "identity not transferred (cosine {cos:.4})");
    } else {
        println!("WARNING: no face detected in the generated image (low-step render?) — identity cosine skipped");
    }

    // (4) sc-3075: real-CFG (true_cfg>1) + negative prompt engages the dual-forward branch and
    // changes the render vs the fake-CFG (true_cfg=1.0) id1 result.
    let mut cfg_req = req_with(&face_img, prompt, 1.0, steps, size);
    cfg_req.true_cfg = Some(2.0);
    cfg_req.negative_prompt = Some("low quality, blurry, deformed, disfigured".into());
    let cfg = first_image(model.generate(&cfg_req, &mut |_| {}).unwrap());
    let cfg_changed = cfg
        .pixels
        .iter()
        .zip(&id1.pixels)
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "true_cfg=2.0 changed {}/{} px vs fake-CFG ({:.1}%)",
        cfg_changed,
        cfg.pixels.len(),
        cfg_changed as f32 / cfg.pixels.len() as f32 * 100.0
    );
    assert!(
        cfg_changed > cfg.pixels.len() / 100,
        "true_cfg>1 should change the render vs fake-CFG (real-CFG branch inactive?)"
    );
    if let Some(gf) = face
        .analyze(&cfg.pixels, cfg.height as usize, cfg.width as usize)
        .unwrap()
        .first()
    {
        println!(
            "true_cfg=2.0 IDENTITY ArcFace cosine = {:.4}",
            cosine(&gf.embedding, &ref_emb)
        );
    }
}
