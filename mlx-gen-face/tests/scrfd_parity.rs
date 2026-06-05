//! SCRFD-10g parity vs the onnx reference + insightface (sc-3082).
//!
//! Two gates: (1) the network reproduces the 9 raw onnx outputs per stride; (2) the full
//! detect (decode + NMS) matches insightface's authoritative detections on `t1.jpg` (the spike's
//! "bbox IoU + kps L2 vs insightface app.get()" acceptance). Goldens from `tools/convert_scrfd.py`
//! live under `tools/golden/` (gitignored) — hence `#[ignore]`.
//!
//! Run:
//!   ~/.dwpose-spike/venv/bin/python tools/convert_scrfd.py
//!   cargo test -p mlx-gen-face --release --test scrfd_parity -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_gen_face::{Detection, Scrfd};

fn golden(name: &str) -> Weights {
    let path = format!("{}/../tools/golden/{name}", env!("CARGO_MANIFEST_DIR"));
    Weights::from_file(&path)
        .unwrap_or_else(|e| panic!("missing golden {path}: {e}\nRun tools/convert_scrfd.py first."))
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    inter / (area_a + area_b - inter)
}

#[test]
#[ignore = "needs local goldens from tools/convert_scrfd.py"]
fn scrfd_network_and_detect_parity() {
    let w = golden("scrfd_10g.safetensors");
    let g = golden("scrfd_goldens.safetensors");
    let input = g.require("input").unwrap(); // [1,640,640,3] f32 NHWC
    let net = Scrfd::from_weights(&w).unwrap();

    // --- (1) network parity: raw per-stride outputs vs onnx
    let raw = net.raw_outputs(input).unwrap();
    let mut worst = 0.0f32;
    for (stride, scores, bbox, kps) in &raw {
        for (label, got) in [("score", scores), ("bbox", bbox), ("kps", kps)] {
            let want = g.require(&format!("{label}.{stride}")).unwrap();
            assert_eq!(got.shape(), want.shape(), "{label}.{stride} shape");
            let gv = got.try_as_slice::<f32>().unwrap();
            let wv = want.try_as_slice::<f32>().unwrap();
            let m = gv
                .iter()
                .zip(wv)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("  raw {label}.{stride}: max abs diff = {m:.6}");
            worst = worst.max(m);
        }
    }
    println!("network parity worst max abs diff = {worst:.6}");
    assert!(worst < 5e-3, "SCRFD network diverged from onnx: {worst}");

    // --- (2) detect parity vs insightface (authoritative)
    let det_scale = g.require("det_scale").unwrap().item::<f32>();
    let got: Vec<Detection> = net.detect(input, det_scale, 0.5, 0.4).unwrap();

    let gb = g.require("det_bboxes").unwrap(); // [M,5] x1,y1,x2,y2,score (orig coords)
    let gk = g.require("det_kpss").unwrap(); // [M,5,2]
    let m = gb.shape()[0] as usize;
    let gbv = gb.try_as_slice::<f32>().unwrap();
    let gkv = gk.try_as_slice::<f32>().unwrap();
    println!("insightface faces = {m}, mlx faces = {}", got.len());
    assert_eq!(got.len(), m, "face count mismatch vs insightface");

    // match each insightface box to its best mlx box by IoU, then check kps L2.
    let mut min_iou = 1.0f32;
    let mut max_kps_l2 = 0.0f32;
    for i in 0..m {
        let want_box = [gbv[i * 5], gbv[i * 5 + 1], gbv[i * 5 + 2], gbv[i * 5 + 3]];
        let (best, biou) =
            got.iter()
                .map(|d| (d, iou(&d.bbox, &want_box)))
                .fold(
                    (&got[0], 0.0f32),
                    |acc, x| if x.1 > acc.1 { x } else { acc },
                );
        let mut kl2 = 0.0f32;
        for p in 0..5 {
            let wx = gkv[(i * 5 + p) * 2];
            let wy = gkv[(i * 5 + p) * 2 + 1];
            kl2 += ((best.kps[p][0] - wx).powi(2) + (best.kps[p][1] - wy).powi(2)).sqrt();
        }
        kl2 /= 5.0;
        println!(
            "  face {i}: IoU = {biou:.5}, score Δ = {:.4}, mean kps L2 = {kl2:.4}px",
            (best.score - gbv[i * 5 + 4]).abs()
        );
        min_iou = min_iou.min(biou);
        max_kps_l2 = max_kps_l2.max(kl2);
    }
    println!("min IoU = {min_iou:.5}, max mean-kps L2 = {max_kps_l2:.4}px");
    assert!(min_iou > 0.99, "box IoU vs insightface too low: {min_iou}");
    assert!(
        max_kps_l2 < 1.0,
        "kps L2 vs insightface too high: {max_kps_l2}px"
    );
}
