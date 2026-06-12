//! SAM3-D / end-to-end parity (sc-4922): run the full image segmenter (PE vision → CLIP text →
//! DETR → mask head) from the reference pixel_values and check the produced instance masks against
//! the torch oracle (`scripts/spikes/sam3_oracle/dump_e2e_fixture.py`).
//!
//! Run:
//!   SAM3_WEIGHTS=.../model.safetensors \
//!   SAM3_E2E_FIXTURE=scripts/spikes/sam3_oracle/e2e_fixture.safetensors \
//!     cargo test -p mlx-gen-sam3 --release --test e2e_parity -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_gen_sam3::Sam3ImageSegmenter;
use mlx_rs::ops::{multiply, sum};
use mlx_rs::Array;

/// IoU of two binary `[h, w]` masks (uint8 0/1).
fn iou(a: &Array, b: &Array) -> f32 {
    let af = a.as_dtype(mlx_rs::Dtype::Float32).unwrap();
    let bf = b.as_dtype(mlx_rs::Dtype::Float32).unwrap();
    let inter = sum(multiply(&af, &bf).unwrap(), None)
        .unwrap()
        .item::<f32>();
    let sa = sum(&af, None).unwrap().item::<f32>();
    let sb = sum(&bf, None).unwrap().item::<f32>();
    let union = sa + sb - inter;
    if union <= 0.0 {
        1.0
    } else {
        inter / union
    }
}

#[test]
#[ignore = "needs SAM3_WEIGHTS + SAM3_E2E_FIXTURE"]
fn full_segmenter_matches_oracle() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to facebook/sam3 model.safetensors");
    let fixture_path = std::env::var("SAM3_E2E_FIXTURE")
        .unwrap_or_else(|_| "scripts/spikes/sam3_oracle/e2e_fixture.safetensors".to_string());

    let w = Weights::from_file(&weights_path).expect("load sam3 weights");
    let fx = Weights::from_file(&fixture_path).expect("load e2e fixture");
    let seg = Sam3ImageSegmenter::from_weights(&w).expect("build segmenter");

    let pixel_values = fx.require("pixel_values").unwrap().clone();
    let input_ids = fx.require("input_ids").unwrap().clone();
    let mask: Vec<i32> = fx
        .require("attention_mask")
        .unwrap()
        .as_slice::<i32>()
        .to_vec();

    // target_wh (1,1) keeps boxes in [0,1]; masks come back at native 288².
    let got = seg
        .segment(&pixel_values, &input_ids, &mask, (1.0, 1.0), 0.5, 0.5)
        .expect("segment");

    let want_masks = fx.require("instance_masks").unwrap().clone(); // [n,288,288] uint8
    let want_n = want_masks.shape()[0] as usize;
    println!("instances: got {} want {}", got.len(), want_n);
    assert_eq!(got.len(), want_n, "instance count mismatch");

    // Reference instances and ours are both in query order → compare index-aligned.
    let mut worst_iou = 1.0f32;
    for (i, inst) in got.iter().enumerate() {
        let want = want_masks
            .take_axis(Array::from_slice(&[i as i32], &[1]), 0)
            .unwrap()
            .reshape(&[288, 288])
            .unwrap();
        let m = iou(&inst.mask, &want);
        worst_iou = worst_iou.min(m);
        println!("  instance {i}: score={:.3} mask IoU={:.4}", inst.score, m);
    }
    assert!(
        worst_iou > 0.95,
        "worst instance mask IoU {worst_iou:.4} below 0.95"
    );
}
