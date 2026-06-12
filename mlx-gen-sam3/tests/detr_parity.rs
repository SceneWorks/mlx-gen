//! SAM3-C DETR-detector parity (sc-4921): load the real `facebook/sam3` weights, run the DETR
//! encoder + decoder + presence + scoring on the reference's 72² FPN feature + text features, and
//! check pred_logits / pred_boxes / presence against the torch oracle
//! (`scripts/spikes/sam3_oracle/dump_detr_fixture.py`).
//!
//! Run:
//!   SAM3_WEIGHTS=.../model.safetensors \
//!   SAM3_DETR_FIXTURE=scripts/spikes/sam3_oracle/detr_fixture.safetensors \
//!     cargo test -p mlx-gen-sam3 --release --test detr_parity -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_gen_sam3::{Sam3Detector, Sam3DetrConfig};
use mlx_rs::ops::{abs, max, multiply, sigmoid, sqrt, subtract, sum};
use mlx_rs::Array;

fn f32_of(a: &Array) -> f32 {
    a.as_dtype(mlx_rs::Dtype::Float32).unwrap().item::<f32>()
}

fn cosine(a: &Array, b: &Array) -> f32 {
    let dot = f32_of(&sum(multiply(a, b).unwrap(), None).unwrap());
    let na = f32_of(&sqrt(sum(multiply(a, a).unwrap(), None).unwrap()).unwrap());
    let nb = f32_of(&sqrt(sum(multiply(b, b).unwrap(), None).unwrap()).unwrap());
    dot / (na * nb)
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    f32_of(&max(abs(subtract(a, b).unwrap()).unwrap(), None).unwrap())
}

#[test]
#[ignore = "needs SAM3_WEIGHTS + SAM3_DETR_FIXTURE"]
fn detr_detector_matches_oracle() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to facebook/sam3 model.safetensors");
    let fixture_path = std::env::var("SAM3_DETR_FIXTURE")
        .unwrap_or_else(|_| "scripts/spikes/sam3_oracle/detr_fixture.safetensors".to_string());

    let w = Weights::from_file(&weights_path).expect("load sam3 weights");
    let fx = Weights::from_file(&fixture_path).expect("load detr fixture");
    let det = Sam3Detector::from_weights(&w, "detector_model", &Sam3DetrConfig::sam3())
        .expect("build detector");

    // fpn_72 NCHW [1,256,72,72] → NHWC [1,72,72,256]
    let fpn_nchw = fx.require("fpn_72").unwrap().clone();
    let vision = fpn_nchw.transpose_axes(&[0, 2, 3, 1]).unwrap();
    let text = fx.require("text_features").unwrap().clone();
    let mask: Vec<i32> = fx
        .require("attention_mask")
        .unwrap()
        .as_slice::<i32>()
        .to_vec();

    let out = det
        .forward(&vision, &text, &mask)
        .expect("detector forward");

    let want_logits = fx.require("pred_logits").unwrap().clone();
    let want_boxes = fx.require("pred_boxes").unwrap().clone();
    let want_presence = fx.require("presence_logits").unwrap().clone();

    let logits_cos = cosine(&out.pred_logits, &want_logits);
    let boxes_cos = cosine(&out.pred_boxes, &want_boxes);
    let presence_diff = max_abs_diff(&out.presence_logits, &want_presence);
    let logits_maxabs = max_abs_diff(&out.pred_logits, &want_logits);

    // End-to-end instance count: score = sigmoid(logits)·sigmoid(presence) > 0.5
    let instances = |logits: &Array, presence: &Array| -> usize {
        let s = multiply(sigmoid(logits).unwrap(), sigmoid(presence).unwrap()).unwrap();
        s.as_slice::<f32>().iter().filter(|&&x| x > 0.5).count()
    };
    let got_n = instances(&out.pred_logits, &out.presence_logits);
    let want_n = instances(&want_logits, &want_presence);

    println!(
        "pred_logits: cosine={logits_cos:.6} max_abs={logits_maxabs:.4} | pred_boxes: cosine={boxes_cos:.6} | presence: |Δ|={presence_diff:.4} | instances got={got_n} want={want_n}"
    );

    assert!(
        logits_cos > 0.999,
        "pred_logits cosine {logits_cos:.6} < 0.999"
    );
    assert!(
        boxes_cos > 0.999,
        "pred_boxes cosine {boxes_cos:.6} < 0.999"
    );
    // presence is a single clamped (±10) logit through 6 decoder layers + the presence MLP; ~1% of
    // the Metal f32-matmul floor is expected (after sigmoid it's invisible: 0.9959 vs 0.9957).
    assert!(
        presence_diff < 0.15,
        "presence |Δ| {presence_diff:.4} too large"
    );
    assert_eq!(got_n, want_n, "instance count mismatch");
}
