//! End-to-end `FaceAnalysis` parity vs insightface + facexlib (sc-3085).
//!
//! Drives the full native stack from a raw RGB image (`t1.jpg`), zero Python:
//!   1. **detector blob** — our cv2-faithful resize-to-fit 640 + pad + normalize matches the
//!      insightface blob.
//!   2. **analyze** — `Vec<Face>` (largest-first) reproduces insightface `app.get()`: face count,
//!      bbox IoU, kps L2, det_score; embedding cos vs canonical math (`emb_ref`) ≥ 0.998 and vs
//!      insightface ORT (`emb_onnx`, reported — the sc-3131 gap).
//!   3. **face_features_image** — the PuLID crop-path output matches torch facexlib (`bisenet_goldens`).
//!
//! Latency of `analyze()` is printed.
//!
//! Reuses the sc-3083/3084 goldens (gitignored) — hence `#[ignore]`. Run:
//!   ~/.dwpose-spike/venv/bin/python tools/dump_face_align_golden.py
//!   ~/.bisenet-spike/venv/bin/python tools/convert_bisenet.py
//!   cargo test -p mlx-gen-face --release --test face_analysis_parity -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_gen_face::{detector_blob, FaceAnalysis};
use std::time::Instant;

fn golden(name: &str) -> Weights {
    let path = format!("{}/../tools/golden/{name}", env!("CARGO_MANIFEST_DIR"));
    Weights::from_file(&path)
        .unwrap_or_else(|e| panic!("missing golden {path}: {e}\nRun the dump_*.py tools first."))
}

fn u8_image(g: &Weights) -> (Vec<u8>, usize, usize) {
    let a = g.require("image").unwrap();
    let sh = a.shape();
    let bytes = a
        .try_as_slice::<i32>()
        .unwrap()
        .iter()
        .map(|&v| v as u8)
        .collect();
    (bytes, sh[0] as usize, sh[1] as usize)
}

fn kps5(g: &Weights, key: &str) -> [[f32; 2]; 5] {
    let v = g
        .require(key)
        .unwrap()
        .try_as_slice::<f32>()
        .unwrap()
        .to_vec();
    let mut k = [[0.0f32; 2]; 5];
    for (i, p) in k.iter_mut().enumerate() {
        p[0] = v[i * 2];
        p[1] = v[i * 2 + 1];
    }
    k
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let (x1, y1) = (a[0].max(b[0]), a[1].max(b[1]));
    let (x2, y2) = (a[2].min(b[2]), a[3].min(b[3]));
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area = |r: &[f32; 4]| (r[2] - r[0]) * (r[3] - r[1]);
    inter / (area(a) + area(b) - inter)
}

fn kps_l2(a: &[[f32; 2]; 5], b: &[[f32; 2]; 5]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(p, q)| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt())
        .sum::<f32>()
        / 5.0
}

#[test]
#[ignore = "needs local goldens (dump_face_align_golden.py + convert_bisenet.py)"]
fn face_analysis_end_to_end() {
    let g = golden("face_align_goldens.safetensors");
    let app = FaceAnalysis::load(
        &golden("scrfd_10g.safetensors"),
        &golden("arcface_iresnet100.safetensors"),
    )
    .unwrap()
    .with_parser(&golden("bisenet_parsing.safetensors"))
    .unwrap();
    let (img, h, w) = u8_image(&g);
    let n = g.require("n_faces").unwrap().item::<i32>() as usize;
    println!("image {h}x{w}, insightface {n} faces");

    // (1) detector blob parity vs insightface
    let (blob, det_scale) = detector_blob(&img, h, w);
    let bg = blob.try_as_slice::<f32>().unwrap();
    let wb = g.require("blob").unwrap().try_as_slice::<f32>().unwrap();
    let blob_max = bg
        .iter()
        .zip(wb)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let blob_scale = g.require("det_scale").unwrap().item::<f32>();
    println!(
        "detector blob: max|Δ| {blob_max:.5}, det_scale {det_scale:.5} (golden {blob_scale:.5})"
    );

    // (2) analyze (largest-first), timed
    let t = Instant::now();
    let faces = app.analyze(&img, h, w).unwrap();
    let dt = t.elapsed();
    // warm timing
    let t2 = Instant::now();
    let faces2 = app.analyze(&img, h, w).unwrap();
    let dt2 = t2.elapsed();
    println!(
        "analyze: {} faces, {:.0} ms (cold) / {:.0} ms (warm)",
        faces.len(),
        dt.as_secs_f64() * 1e3,
        dt2.as_secs_f64() * 1e3
    );
    assert_eq!(faces.len(), n, "face count vs insightface");
    let _ = faces2;

    // goldens are also sorted largest-first → index-aligned
    let (mut min_iou, mut max_kl2) = (1.0f32, 0.0f32);
    let (mut min_ref, mut min_ort) = (1.0f32, 1.0f32);
    for (i, f) in faces.iter().enumerate() {
        let gb = g
            .require(&format!("bbox.{i}"))
            .unwrap()
            .try_as_slice::<f32>()
            .unwrap();
        let want_bbox = [gb[0], gb[1], gb[2], gb[3]];
        let want_kps = kps5(&g, &format!("kps.{i}"));
        let want_emb_ref = g
            .require(&format!("emb_ref.{i}"))
            .unwrap()
            .try_as_slice::<f32>()
            .unwrap();
        let want_emb_ort = g
            .require(&format!("emb_onnx.{i}"))
            .unwrap()
            .try_as_slice::<f32>()
            .unwrap();
        let bi = iou(&f.bbox, &want_bbox);
        let kl = kps_l2(&f.kps, &want_kps);
        let rc = cosine(&f.embedding, want_emb_ref);
        let oc = cosine(&f.embedding, want_emb_ort);
        min_iou = min_iou.min(bi);
        max_kl2 = max_kl2.max(kl);
        min_ref = min_ref.min(rc);
        min_ort = min_ort.min(oc);
        println!("  face {i}: IoU {bi:.4}, kps L2 {kl:.3}px, det_score {:.4}, emb cos ref {rc:.6} / ort {oc:.4}", f.det_score);
    }
    println!("ANALYZE: min IoU {min_iou:.4}, max kps L2 {max_kl2:.3}px, min emb cos(ref) {min_ref:.4}, min emb cos(ORT) {min_ort:.4}");

    // (3) face_features_image (PuLID crop path) vs torch facexlib — for the largest face
    let bg2 = golden("bisenet_goldens.safetensors");
    let ffi = app.face_features_image(&img, h, w, &faces[0]).unwrap();
    let got = ffi.try_as_slice::<f32>().unwrap();
    let want = bg2
        .require("face_features_image")
        .unwrap()
        .try_as_slice::<f32>()
        .unwrap();
    let (mut ffi_diff, mut ffi_max) = (0usize, 0.0f32);
    for (a, b) in got.iter().zip(want) {
        let d = (a - b).abs();
        if d > 1e-6 {
            ffi_diff += 1;
        }
        ffi_max = ffi_max.max(d);
    }
    let ffi_frac = ffi_diff as f32 / got.len() as f32;
    println!(
        "face_features_image (face 0): {ffi_diff} px differ ({:.4}%), max|Δ| {ffi_max:.4}",
        ffi_frac * 100.0
    );

    // gates
    assert!(
        blob_max < 0.02,
        "detector blob diverged from cv2 resize: {blob_max}"
    );
    assert!((det_scale - blob_scale).abs() < 1e-5, "det_scale mismatch");
    assert!(min_iou > 0.98, "bbox IoU vs insightface too low: {min_iou}");
    assert!(max_kl2 < 1.5, "kps L2 vs insightface too high: {max_kl2}px");
    assert!(
        min_ref >= 0.99,
        "embedding cos vs canonical math too low: {min_ref}"
    );
    assert!(
        ffi_frac < 0.02,
        "face_features_image diverged from torch: {:.4}%",
        ffi_frac * 100.0
    );
}
