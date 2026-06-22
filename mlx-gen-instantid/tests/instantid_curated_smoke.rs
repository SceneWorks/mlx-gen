//! sc-7297 / epic 7114 — curated-sampler **identity** smoke for InstantID.
//!
//! Validates that routing InstantID's dual-conditioning denoise through `denoise_curated` (when a
//! curated sampler is selected) preserves identity adherence as well as the bespoke ancestral default.
//! Env-driven so it runs against the app's REAL layout (RealVisXL backbone + the InstantID cache),
//! needing no torch and no `tools/golden` artifacts:
//!
//!   INSTANTID_SDXL_BASE=<RealVisXL snapshot dir> \
//!   INSTANTID_CONTROLNET=<InstantX/InstantID ControlNetModel dir> \
//!   INSTANTID_IP_ADAPTER=<ip-adapter.safetensors> \
//!   INSTANTID_SCRFD=<scrfd_10g.safetensors> INSTANTID_ARCFACE=<arcface_iresnet100.safetensors> \
//!   INSTANTID_REF=<reference face image> \
//!   cargo test -p mlx-gen-instantid --release --test instantid_curated_smoke -- --ignored --nocapture
//!
//! Gate (directional, per epic 3109): every sampler — the ancestral default AND each curated solver —
//! must keep the ArcFace cosine well above 0 (identity preserved). A curated solver that destabilizes
//! InstantID's strong conditioning collapses the face toward 0 / undetectable.

use std::path::PathBuf;

use mlx_gen::media::Image;
use mlx_gen::weights::Weights;
use mlx_gen::WeightsSource;
use mlx_gen_instantid::{letterbox, InstantId, InstantIdPaths, InstantIdRequest};

fn env_path(key: &str) -> PathBuf {
    PathBuf::from(std::env::var(key).unwrap_or_else(|_| panic!("set {key}")))
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Cosine similarity of two (un-normalized) ArcFace embeddings.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
    let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    (dot / (na * nb + 1e-12)) as f32
}

fn load_rgb(path: &PathBuf) -> Image {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("open ref {path:?}: {e}"))
        .to_rgb8();
    Image {
        width: img.width(),
        height: img.height(),
        pixels: img.into_raw(),
    }
}

#[test]
#[ignore = "needs RealVisXL + InstantID ControlNet + ip-adapter/scrfd/arcface + a reference face (env-driven)"]
fn curated_samplers_preserve_identity() {
    let size = env_usize("INSTANTID_SIZE", 768) as u32;
    let steps = env_usize("INSTANTID_STEPS", 20);

    let paths = InstantIdPaths {
        sdxl_base: env_path("INSTANTID_SDXL_BASE"),
        identitynet: WeightsSource::Dir(env_path("INSTANTID_CONTROLNET")),
        ip_adapter: env_path("INSTANTID_IP_ADAPTER"),
        adapters: Vec::new(),
    };
    let scrfd = Weights::from_file(env_path("INSTANTID_SCRFD")).expect("load scrfd");
    let arcface = Weights::from_file(env_path("INSTANTID_ARCFACE")).expect("load arcface");
    let model = InstantId::load(&paths)
        .expect("load InstantID")
        .with_face(&scrfd, &arcface)
        .expect("attach face stack");

    // Reference identity (its ArcFace embedding drives the IP path).
    let ref_img = load_rgb(&env_path("INSTANTID_REF"));
    let canvas = letterbox(&ref_img, size, size);
    let ref_face = model
        .largest_face(&canvas.pixels, size as usize, size as usize)
        .expect("detect reference face");
    let kps: Vec<(f32, f32)> = ref_face.kps.iter().map(|p| (p[0], p[1])).collect();
    println!(
        "[smoke] {size}x{size} steps={steps} | ref face det_score={:.3}",
        ref_face.det_score
    );

    // The ancestral default + a spread of curated solvers over the SAME dual conditioning. The default
    // (None ⇒ euler_ancestral) is the byte-exact baseline; the curated names exercise the new
    // `denoise_curated` route in `run_identity_denoise`.
    let samplers: [Option<&str>; 4] = [None, Some("euler"), Some("heun"), Some("dpmpp_2m")];
    let mut results: Vec<(String, f32)> = Vec::new();
    for s in samplers {
        let req = InstantIdRequest {
            prompt: "film still, a portrait photo of a person, cinematic lighting, sharp focus, \
                     high detail, looking at the camera"
                .into(),
            negative: "lowres, blurry, deformed, disfigured, cartoon, painting".into(),
            width: size,
            height: size,
            steps,
            guidance: 5.0,
            seed: 0,
            sampler: s.map(str::to_owned),
            ..Default::default()
        };
        let out = model
            .generate_with(&req, &ref_face.embedding, &kps, &mut |_| {})
            .expect("generate");
        // A destabilizing sampler can yield an image with no detectable face — treat that as
        // identity-lost (cosine 0) rather than a hard panic, so the gate reports it cleanly.
        let cos = match model.largest_face(&out.pixels, size as usize, size as usize) {
            Ok(f) => cosine(&ref_face.embedding, &f.embedding),
            Err(e) => {
                println!("[smoke]   (no face detected in output: {e})");
                0.0
            }
        };
        let name = s.unwrap_or("euler_ancestral(default)");
        println!("[smoke] sampler={name:<26} ArcFace-cosine(ref,gen) = {cos:.4}");
        results.push((name.to_string(), cos));
    }

    // Identity must be preserved on every path (well above the ~0 broken-pipeline floor).
    for (name, cos) in &results {
        assert!(
            *cos > 0.5,
            "sampler {name}: identity not preserved (cosine {cos:.4}); the curated route may \
             destabilize InstantID's conditioning"
        );
    }
}
