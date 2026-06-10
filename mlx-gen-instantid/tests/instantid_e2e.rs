//! sc-3115: InstantID T2I end-to-end + ArcFace-cosine identity preservation.
//!
//! `#[ignore]`d — needs the SDXL base snapshot, the InstantID `ControlNetModel`, the converted
//! `tools/golden/instantid/ip-adapter.safetensors` (`tools/convert_instantid.py`), the face-stack
//! weights (`tools/convert_scrfd.py` + `tools/convert_glintr100.py`), and the reference image
//! (`tools/dump_instantid_e2e_ref.py`). Fully self-contained in Rust at test time — no torch.
//!
//! Run (tune size/steps via env for a quick smoke):
//!   INSTANTID_SIZE=512 INSTANTID_STEPS=4 cargo test -p mlx-gen-instantid --release \
//!     --test instantid_e2e -- --ignored --nocapture
//!   cargo test -p mlx-gen-instantid --release --test instantid_e2e -- --ignored --nocapture
//!
//! The gate is **directional** (per epic 3109: ArcFace-cosine + coherence, NOT bit-exact): a correctly
//! wired pipeline preserves identity (cosine well above 0), a broken one collapses to ~0.

use std::path::PathBuf;

use mlx_gen::media::Image;
use mlx_gen::weights::Weights;
use mlx_gen::WeightsSource;
use mlx_gen_instantid::{letterbox, InstantId, InstantIdPaths, InstantIdRequest};

fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../tools/golden")).join(name)
}

fn golden(name: &str) -> Weights {
    let p = golden_path(name);
    Weights::from_file(&p).unwrap_or_else(|e| panic!("missing golden {p:?}: {e}"))
}

fn sdxl_base() -> PathBuf {
    if let Ok(p) = std::env::var("SDXL_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots");
    std::fs::read_dir(&snaps)
        .expect("SDXL base snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn instantid_controlnet() -> WeightsSource {
    let home = std::env::var("HOME").unwrap();
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--InstantX--InstantID/snapshots");
    let snap = std::fs::read_dir(&snaps)
        .expect("InstantID snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir");
    WeightsSource::Dir(snap.join("ControlNetModel"))
}

fn openpose_controlnet() -> WeightsSource {
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--xinsir--controlnet-openpose-sdxl-1.0/snapshots");
    let snap = std::fs::read_dir(&snaps)
        .expect("xinsir OpenPose-SDXL snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir");
    WeightsSource::Dir(snap)
}

/// A real gallery pose (apps/web/public/poses/index.json :: dance_01), COCO-18 normalized — the
/// head is visible so IdentityNet anchors the face.
fn dance_01_keypoints() -> Vec<mlx_gen_instantid::BodyPoint> {
    [
        (0.5429, 0.1454),
        (0.515, 0.2608),
        (0.4469, 0.263),
        (0.3465, 0.3332),
        (0.2275, 0.4021),
        (0.5831, 0.2587),
        (0.6376, 0.3433),
        (0.6275, 0.4365),
        (0.4841, 0.4852),
        (0.553, 0.6616),
        (0.5859, 0.8867),
        (0.553, 0.4895),
        (0.4454, 0.6917),
        (0.3623, 0.8465),
        (0.5243, 0.1354),
        (0.5386, 0.134),
        (0.4784, 0.1569),
        (0.52, 0.1512),
    ]
    .into_iter()
    .map(|(x, y)| Some((x, y)))
    .collect()
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Cosine similarity of two (un-normalized) embeddings.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
    let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    (dot / (na * nb + 1e-12)) as f32
}

fn save_png(name: &str, img: &Image) {
    let path = golden_path(name);
    let buf = image::RgbImage::from_raw(img.width, img.height, img.pixels.clone()).unwrap();
    buf.save(&path).unwrap();
    println!("  wrote {path:?}");
}

/// Load (optionally quantized) → detect reference face → generate → re-detect → ArcFace-cosine.
/// `quant_bits`: None = fp16, Some(8)/Some(4) = Q8/Q4 (sc-3116). Returns the identity cosine.
fn run_identity(quant_bits: Option<i32>, size_override: Option<u32>, out_png: &str) -> f32 {
    let size = size_override.unwrap_or_else(|| env_usize("INSTANTID_SIZE", 1024) as u32);
    let steps = env_usize("INSTANTID_STEPS", 30);
    let label = quant_bits
        .map(|b| format!("Q{b}"))
        .unwrap_or_else(|| "fp16".into());

    let paths = InstantIdPaths {
        sdxl_base: sdxl_base(),
        identitynet: instantid_controlnet(),
        ip_adapter: golden_path("instantid/ip-adapter.safetensors"),
    };
    let scrfd = golden("scrfd_10g.safetensors");
    let arcface = golden("arcface_iresnet100.safetensors");
    let mut model = InstantId::load(&paths).expect("load InstantID");
    if let Some(bits) = quant_bits {
        model = model.quantize(bits).expect("quantize");
    }
    let model = model
        .with_face(&scrfd, &arcface)
        .expect("attach face stack");

    // Reference face.
    let g = golden("instantid_e2e_ref.safetensors");
    let wh = g.require("ref_wh").unwrap().as_slice::<i32>().to_vec();
    let (rw, rh) = (wh[0] as u32, wh[1] as u32);
    let ref_img = Image {
        width: rw,
        height: rh,
        pixels: g.require("ref_img").unwrap().as_slice::<u8>().to_vec(),
    };

    // Letterbox to the output size + detect the reference face (its embedding drives the IP path).
    let canvas = letterbox(&ref_img, size, size);
    let ref_face = model
        .largest_face(&canvas.pixels, size as usize, size as usize)
        .expect("detect reference face");
    let kps: Vec<(f32, f32)> = ref_face.kps.iter().map(|p| (p[0], p[1])).collect();
    println!(
        "[instantid {label}] ref face det_score={:.3} kps[0]=({:.1},{:.1})",
        ref_face.det_score, kps[0].0, kps[0].1
    );

    // Generate.
    let req = InstantIdRequest {
        prompt: "film still, a portrait photo of a man, cinematic lighting, sharp focus, \
                 high detail, looking at the camera"
            .into(),
        negative: "lowres, blurry, deformed, disfigured, cartoon, painting".into(),
        width: size,
        height: size,
        steps,
        guidance: 5.0,
        ip_adapter_scale: 0.8,
        controlnet_scale: 0.8,
        seed: 0,
        ..Default::default()
    };
    let out = model
        .generate_with(&req, &ref_face.embedding, &kps, &mut |_| {})
        .expect("generate");
    assert_eq!((out.width, out.height), (size, size), "output dims");
    // Not a degenerate (all-zero / NaN→0) image.
    let nonzero = out.pixels.iter().filter(|&&p| p != 0).count();
    assert!(
        nonzero > out.pixels.len() / 100,
        "output looks degenerate ({nonzero} nonzero bytes)"
    );
    save_png(out_png, &out);

    // Re-detect the generated face and measure identity preservation.
    let out_face = model
        .largest_face(&out.pixels, size as usize, size as usize)
        .expect("detect generated face");
    let cos = cosine(&ref_face.embedding, &out_face.embedding);
    println!(
        "[instantid {label}] {size}x{size} steps={steps} | generated face det_score={:.3} | \
         ArcFace-cosine(ref, generated) = {cos:.4}",
        out_face.det_score
    );
    cos
}

// Directional gate (epic 3109: identity + coherence, NOT bit-exact). fp16 measures **0.8214** at the
// default 1024²/30-step settings — essentially the sc-2009 torch baseline (≈0.876). A broken pipeline
// (wrong token wiring / no IP / no IdentityNet) collapses toward 0 (the 4-step smoke sits at ~0.21).

#[test]
#[ignore = "needs SDXL base + InstantID + converted ip-adapter + face goldens + reference"]
fn instantid_t2i_preserves_identity() {
    let cos = run_identity(None, None, "instantid_e2e_out.png");
    assert!(
        cos > 0.6,
        "fp16 identity not preserved: ArcFace-cosine {cos:.4} (expected ≳0.8 at 1024²/30 steps)"
    );
}

// sc-3116 quant tests run at **512²**, not 1024²: the stock SDXL **quantized UNet collapses to a flat
// image at 1024²** (a pre-existing base-SDXL-quant defect — `mlx-gen-sdxl tests/q8_1024_probe.rs`
// reproduces it on plain SDXL Q8 txt2img, independent of InstantID; tracked separately). At 512² the
// full InstantID quant stack (UNet + IP K/V + CLIP TEs + IdentityNet) is healthy and preserves
// identity. The IdentityNet + TE quant are fine at 1024² (only the base UNet quant is affected).

#[test]
#[ignore = "needs SDXL base + InstantID + converted ip-adapter + face goldens + reference"]
fn instantid_t2i_q8_preserves_identity() {
    let cos = run_identity(Some(8), Some(512), "instantid_e2e_q8_out.png");
    assert!(
        cos > 0.5,
        "Q8 identity not preserved: ArcFace-cosine {cos:.4}"
    );
}

#[test]
#[ignore = "needs SDXL base + InstantID + converted ip-adapter + face goldens + reference"]
fn instantid_t2i_q4_preserves_identity() {
    // Q4 is more aggressive; identity should still be clearly preserved (looser floor).
    let cos = run_identity(Some(4), Some(512), "instantid_e2e_q4_out.png");
    assert!(
        cos > 0.45,
        "Q4 identity not preserved: ArcFace-cosine {cos:.4}"
    );
}

#[test]
#[ignore = "needs SDXL base + InstantID + xinsir OpenPose + ip-adapter + face goldens + reference"]
fn instantid_pose_mode_preserves_identity() {
    // sc-3117 (MultiControlNet pose mode): IdentityNet (face) + OpenPose (body skeleton from the
    // pre-supplied gallery keypoints) run together. Identity holds at the (small) full-body face —
    // directionally; the face-restoration pass (sc-3380) recovers it further. A broken MultiControlNet
    // wiring (residuals not summed / wrong control order) collapses identity toward 0.
    let size = env_usize("INSTANTID_SIZE", 1024) as u32;

    let paths = InstantIdPaths {
        sdxl_base: sdxl_base(),
        identitynet: instantid_controlnet(),
        ip_adapter: golden_path("instantid/ip-adapter.safetensors"),
    };
    let model = InstantId::load(&paths)
        .expect("load InstantID")
        .with_openpose(&openpose_controlnet())
        .expect("attach OpenPose ControlNet")
        .with_face(
            &golden("scrfd_10g.safetensors"),
            &golden("arcface_iresnet100.safetensors"),
        )
        .expect("attach face stack");

    let g = golden("instantid_e2e_ref.safetensors");
    let wh = g.require("ref_wh").unwrap().as_slice::<i32>().to_vec();
    let ref_img = Image {
        width: wh[0] as u32,
        height: wh[1] as u32,
        pixels: g.require("ref_img").unwrap().as_slice::<u8>().to_vec(),
    };

    let req = InstantIdRequest {
        prompt: "full body photo of a man standing, cinematic lighting, sharp focus, high detail"
            .into(),
        negative: "lowres, blurry, deformed, disfigured, extra limbs".into(),
        width: size,
        height: size,
        ..Default::default()
    };
    let out = model
        .generate_pose(&req, &ref_img, &dance_01_keypoints(), &mut |_| {})
        .expect("generate pose");
    assert_eq!((out.width, out.height), (size, size), "square pose output");
    let nonzero = out.pixels.iter().filter(|&&p| p != 0).count();
    assert!(
        nonzero > out.pixels.len() / 100,
        "pose output looks degenerate ({nonzero} nonzero bytes)"
    );
    save_png("instantid_e2e_pose_out.png", &out);

    // Identity vs the reference embedding (the generated full-body face is small, so the cosine is
    // lower than the framed-portrait path; gate directionally for a clear positive signal).
    let canvas = letterbox(&ref_img, size, size);
    let ref_face = model
        .largest_face(&canvas.pixels, size as usize, size as usize)
        .expect("detect reference face");
    let out_face = model
        .largest_face(&out.pixels, size as usize, size as usize)
        .expect("detect generated face");
    let cos = cosine(&ref_face.embedding, &out_face.embedding);
    println!("[instantid pose] {size}x{size} | ArcFace-cosine(ref, generated) = {cos:.4}");
    assert!(
        cos > 0.3,
        "pose-mode identity not preserved: ArcFace-cosine {cos:.4} (full-body face is small; \
         expected a clear positive signal, sc-3380 face-restore recovers it further)"
    );
}

#[test]
#[ignore = "needs SDXL base + InstantID + xinsir OpenPose + ip-adapter + face goldens + reference"]
fn instantid_face_restore_improves_identity() {
    // sc-3380: the face-restoration pass recovers identity at full-body framing. Generate a pose
    // (small full-body face), then run restore_face — the re-rendered + feathered-paste crop should
    // measurably raise the ArcFace-cosine vs the un-restored base (the reference: ~0.38 → ~0.88).
    let size = env_usize("INSTANTID_SIZE", 1024) as u32;

    let paths = InstantIdPaths {
        sdxl_base: sdxl_base(),
        identitynet: instantid_controlnet(),
        ip_adapter: golden_path("instantid/ip-adapter.safetensors"),
    };
    let model = InstantId::load(&paths)
        .expect("load InstantID")
        .with_openpose(&openpose_controlnet())
        .expect("attach OpenPose ControlNet")
        .with_face(
            &golden("scrfd_10g.safetensors"),
            &golden("arcface_iresnet100.safetensors"),
        )
        .expect("attach face stack");

    let g = golden("instantid_e2e_ref.safetensors");
    let wh = g.require("ref_wh").unwrap().as_slice::<i32>().to_vec();
    let ref_img = Image {
        width: wh[0] as u32,
        height: wh[1] as u32,
        pixels: g.require("ref_img").unwrap().as_slice::<u8>().to_vec(),
    };

    // The reference embedding (identity target) from the letterboxed reference.
    let canvas = letterbox(&ref_img, size, size);
    let ref_face = model
        .largest_face(&canvas.pixels, size as usize, size as usize)
        .expect("detect reference face");

    let req = InstantIdRequest {
        prompt:
            "full body photo of a person standing, cinematic lighting, sharp focus, high detail"
                .into(),
        negative: "lowres, blurry, deformed, disfigured, extra limbs".into(),
        width: size,
        height: size,
        ..Default::default()
    };
    let base = model
        .generate_pose(&req, &ref_img, &dance_01_keypoints(), &mut |_| {})
        .expect("generate pose");
    let base_face = model
        .largest_face(&base.pixels, size as usize, size as usize)
        .expect("detect base face");
    let base_cos = cosine(&ref_face.embedding, &base_face.embedding);

    // Restore with the gender-neutral default prompt (sc-3380 bug fix).
    let restore_req = InstantIdRequest {
        prompt: mlx_gen_instantid::FACE_RESTORE_PROMPT.into(),
        ..req.clone()
    };
    let restored = model
        .restore_face(&restore_req, &base, &ref_face.embedding, &mut |_| {})
        .expect("restore face");
    assert_eq!(
        (restored.width, restored.height),
        (size, size),
        "restore keeps the base dims"
    );
    save_png("instantid_e2e_restore_out.png", &restored);

    let out_face = model
        .largest_face(&restored.pixels, size as usize, size as usize)
        .expect("detect restored face");
    let restored_cos = cosine(&ref_face.embedding, &out_face.embedding);
    println!(
        "[instantid restore] {size}x{size} | ArcFace-cosine base={base_cos:.4} restored={restored_cos:.4}"
    );
    assert!(
        restored_cos > base_cos,
        "face-restore should improve identity: base {base_cos:.4} → restored {restored_cos:.4}"
    );
    assert!(
        restored_cos > 0.55,
        "restored identity weak: ArcFace-cosine {restored_cos:.4} (expected a strong recovery)"
    );
}

#[test]
#[ignore = "needs SDXL base + InstantID + converted ip-adapter + face goldens + reference"]
fn instantid_view_angle_preserves_identity() {
    // sc-3117 (multi-view kps): rotate the reference identity to a named view from VIEW_ANGLE_KPS.
    // Identity holds across the view (ArcFace is view-tolerant); sc-2009 measured 0.81-0.89 at the
    // target angle. A moderate turn (three-quarter) is gated directionally.
    let size = env_usize("INSTANTID_SIZE", 1024) as u32;
    let view = "three_quarter_right";

    let paths = InstantIdPaths {
        sdxl_base: sdxl_base(),
        identitynet: instantid_controlnet(),
        ip_adapter: golden_path("instantid/ip-adapter.safetensors"),
    };
    let model = InstantId::load(&paths)
        .expect("load InstantID")
        .with_face(
            &golden("scrfd_10g.safetensors"),
            &golden("arcface_iresnet100.safetensors"),
        )
        .expect("attach face stack");

    let g = golden("instantid_e2e_ref.safetensors");
    let wh = g.require("ref_wh").unwrap().as_slice::<i32>().to_vec();
    let ref_img = Image {
        width: wh[0] as u32,
        height: wh[1] as u32,
        pixels: g.require("ref_img").unwrap().as_slice::<u8>().to_vec(),
    };

    let req = InstantIdRequest {
        prompt: "film still, a portrait photo of a man, cinematic lighting, sharp focus".into(),
        negative: "lowres, blurry, deformed".into(),
        width: size,
        height: size,
        ..Default::default()
    };
    let out = model
        .generate_angle(&req, &ref_img, view, &mut |_| {})
        .expect("generate view angle");
    save_png("instantid_e2e_angle_out.png", &out);

    // Identity vs the reference's (frontal) embedding.
    let canvas = letterbox(&ref_img, size, size);
    let ref_face = model
        .largest_face(&canvas.pixels, size as usize, size as usize)
        .expect("detect reference face");
    let out_face = model
        .largest_face(&out.pixels, size as usize, size as usize)
        .expect("detect generated face");
    let cos = cosine(&ref_face.embedding, &out_face.embedding);
    println!("[instantid view={view}] {size}x{size} | ArcFace-cosine(ref, generated) = {cos:.4}");
    assert!(
        cos > 0.4,
        "view-angle identity not preserved: ArcFace-cosine {cos:.4} (a turned view reduces cosine \
         vs a frontal reference; expected a clear positive signal)"
    );
}

#[test]
#[ignore = "needs SDXL base + InstantID + converted ip-adapter + face goldens + reference"]
fn instantid_honors_cancellation() {
    // sc-4380 (F-096 sibling): InstantID must honor the engine cancellation contract.
    // 1) A pre-cancelled request aborts before any tensor work.
    // 2) A flag tripped mid-denoise (from the progress callback) stops the loop at the next step.
    let paths = InstantIdPaths {
        sdxl_base: sdxl_base(),
        identitynet: instantid_controlnet(),
        ip_adapter: golden_path("instantid/ip-adapter.safetensors"),
    };
    let scrfd = golden("scrfd_10g.safetensors");
    let arcface = golden("arcface_iresnet100.safetensors");
    let model = InstantId::load(&paths)
        .expect("load InstantID")
        .with_face(&scrfd, &arcface)
        .expect("attach face stack");

    let g = golden("instantid_e2e_ref.safetensors");
    let wh = g.require("ref_wh").unwrap().as_slice::<i32>().to_vec();
    let ref_img = Image {
        width: wh[0] as u32,
        height: wh[1] as u32,
        pixels: g.require("ref_img").unwrap().as_slice::<u8>().to_vec(),
    };

    // Pre-cancelled: aborts immediately with the canonical error.
    let req = InstantIdRequest {
        prompt: "a portrait photo".into(),
        width: 512,
        height: 512,
        steps: 8,
        ..Default::default()
    };
    req.cancel.cancel();
    let err = model
        .generate(&req, &ref_img, &mut |_| {})
        .unwrap_err()
        .to_string();
    assert!(err.contains("generation cancelled"), "got: {err}");

    // Mid-denoise: trip the flag from the first Step callback; the loop must stop early.
    let req = InstantIdRequest {
        prompt: "a portrait photo".into(),
        width: 512,
        height: 512,
        steps: 8,
        ..Default::default()
    };
    let cancel = req.cancel.clone();
    let mut steps_seen = 0u32;
    let err = model
        .generate(&req, &ref_img, &mut |p| {
            if let mlx_gen::Progress::Step { current, .. } = p {
                steps_seen = steps_seen.max(current);
                cancel.cancel();
            }
        })
        .unwrap_err()
        .to_string();
    assert!(err.contains("generation cancelled"), "got: {err}");
    assert!(
        (1..=2).contains(&steps_seen),
        "denoise should stop right after the cancel trip (saw step {steps_seen})"
    );
}
