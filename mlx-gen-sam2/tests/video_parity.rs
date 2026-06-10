//! sc-3714 — SAM2 **video propagation** parity vs the MLX-native reference video predictor
//! (`avbiswas/sam2-mlx` `SAM2VideoPredictor`, the impl this crate ports). This is the video-side GO
//! gate: a first-frame box must produce a temporally-consistent mask sequence matching the reference
//! across the whole clip.
//!
//! Golden: `tools/dump_sam2_video_golden.py` runs the reference predictor on a fixed synthetic clip
//! (a translating rectangle, 4 frames) + a first-frame box and bundles the converted weights, the
//! preprocessed `images` [4,3,1024,1024], the `box_xyxy`, and the reference per-frame **low-res**
//! mask logits `low_res_masks` [4,1,256,256] + `object_scores` [4,1]. Both run MLX Metal, so parity
//! is near-bit; we feed the Rust port the identical pixels so the comparison isolates the model from
//! frame decode / preprocessing.
//!
//! Run:
//!   PYTHONPATH=/tmp/sam2-mlx/src ~/mlx-flux-venv/bin/python tools/dump_sam2_video_golden.py --size large
//!   cargo test -p mlx-gen-sam2 --release --test video_parity -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_gen_sam2::{Sam2ModelSize, Sam2VideoPredictor};
use mlx_rs::{Array, Dtype};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/sam2_video_golden_large.safetensors"
);

fn golden() -> Weights {
    Weights::from_file(GOLDEN).unwrap_or_else(|e| {
        panic!("missing {GOLDEN}: {e}\nRun tools/dump_sam2_video_golden.py --size large first.")
    })
}

fn flat(a: &Array) -> Vec<f32> {
    let n: i32 = a.shape().iter().product();
    a.reshape(&[n])
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

/// Cosine similarity of two flat logit maps.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += (x as f64).powi(2);
        nb += (y as f64).powi(2);
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// IoU of the thresholded (`> 0`) binary masks — the metric that actually matters for a track.
fn iou(a: &[f32], b: &[f32]) -> f32 {
    let (mut inter, mut union) = (0u64, 0u64);
    for (&x, &y) in a.iter().zip(b) {
        let (xa, yb) = (x > 0.0, y > 0.0);
        if xa && yb {
            inter += 1;
        }
        if xa || yb {
            union += 1;
        }
    }
    if union == 0 {
        1.0
    } else {
        inter as f32 / union as f32
    }
}

#[test]
#[ignore = "needs local golden from tools/dump_sam2_video_golden.py --size large"]
fn video_propagation_matches_mlx_reference_large() {
    let g = golden();

    // Clip + prompt fixtures from the golden.
    let images = g.require("images").unwrap().clone(); // [T,3,1024,1024]
    let hw = flat(g.require("video_hw").unwrap());
    let (video_h, video_w) = (hw[0] as u32, hw[1] as u32);
    let bx = flat(g.require("box_xyxy").unwrap());
    let box_xyxy = [bx[0], bx[1], bx[2], bx[3]];
    let t = images.shape()[0];

    let predictor = Sam2VideoPredictor::from_weights_for_size(&g, Sam2ModelSize::Large).unwrap();
    let mut state = predictor.init_state_from_pixels(images, video_h, video_w);
    predictor.add_new_box(&mut state, 0, box_xyxy).unwrap();
    let results = predictor.propagate(&mut state).unwrap();
    assert_eq!(results.len(), t as usize, "propagated frame count");

    // Reference per-frame low-res logits, [T,1,256,256].
    let want = g.require("low_res_masks").unwrap().clone();

    let mut worst_cos = f32::INFINITY;
    let mut worst_iou = f32::INFINITY;
    for (frame_idx, low) in &results {
        let got = flat(low);
        let ref_frame = want
            .take_axis(Array::from_int(*frame_idx), 0)
            .unwrap()
            .reshape(&[256 * 256])
            .unwrap();
        let want_v = flat(&ref_frame);
        let c = cosine(&got, &want_v);
        let i = iou(&got, &want_v);
        println!("frame {frame_idx}: cosine {c:.6} IoU {i:.6}");
        worst_cos = worst_cos.min(c);
        worst_iou = worst_iou.min(i);
    }
    println!("worst-frame: cosine {worst_cos:.6} IoU {worst_iou:.6}");

    // Temporal-consistency GO gate: every frame's mask must closely match the reference track.
    assert!(worst_cos > 0.99, "worst-frame cosine {worst_cos:.6}");
    assert!(worst_iou > 0.97, "worst-frame IoU {worst_iou:.6}");
}

/// F-166 regression: correcting a prompted frame *before* `propagate` — the documented
/// `add_new_box(f0)` → `add_correction_points(f0, …)` sequence — must not panic. The corrected cond
/// frame has no encoded memory yet (memory is encoded lazily in the propagation preflight), so the
/// second call's `condition_with_memories` must skip it instead of `expect`-panicking.
#[test]
#[ignore = "needs local golden from tools/dump_sam2_video_golden.py --size large"]
fn correction_before_propagate_does_not_panic() {
    let g = golden();
    let images = g.require("images").unwrap().clone();
    let hw = flat(g.require("video_hw").unwrap());
    let (video_h, video_w) = (hw[0] as u32, hw[1] as u32);
    let bx = flat(g.require("box_xyxy").unwrap());
    let box_xyxy = [bx[0], bx[1], bx[2], bx[3]];

    let predictor = Sam2VideoPredictor::from_weights_for_size(&g, Sam2ModelSize::Large).unwrap();
    let mut state = predictor.init_state_from_pixels(images, video_h, video_w);
    predictor.add_new_box(&mut state, 0, box_xyxy).unwrap();

    // A positive correction click at the box center on the SAME frame, before any propagate.
    let (cx, cy) = ((bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0);
    let refined = predictor
        .add_correction_points(&mut state, 0, &[[cx, cy]], &[1])
        .expect("correcting a prompted frame before propagate must not panic (F-166)");
    assert_eq!(
        refined.shape(),
        &[1, 1, 256, 256],
        "refined low-res mask logits"
    );
}
