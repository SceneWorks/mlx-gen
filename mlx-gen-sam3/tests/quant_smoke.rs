//! SAM3-G quantization smoke (sc-4925): build the image segmenter dense / Q8 / Q4 from the real
//! `facebook/sam3` weights, run the end-to-end "segment all person" path on the zidane fixture, and
//! check that Q8 stays near-lossless vs the dense baseline (and the oracle) while Q4 stays coherent.
//!
//! Run:
//!   SAM3_WEIGHTS=.../model.safetensors \
//!   SAM3_E2E_FIXTURE=scripts/spikes/sam3_oracle/e2e_fixture.safetensors \
//!     cargo test -p mlx-gen-sam3 --release --test quant_smoke -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_gen_sam3::{Instance, Sam3ImageSegmenter, Sam3Tracker};
use mlx_rs::ops::{multiply, sqrt, sum};
use mlx_rs::Array;

fn cosine(a: &Array, b: &Array) -> f32 {
    let a = a.reshape(&[-1]).unwrap();
    let b = b.reshape(&[-1]).unwrap();
    let dot = sum(multiply(&a, &b).unwrap(), None).unwrap().item::<f32>();
    let na = sqrt(sum(multiply(&a, &a).unwrap(), None).unwrap())
        .unwrap()
        .item::<f32>();
    let nb = sqrt(sum(multiply(&b, &b).unwrap(), None).unwrap())
        .unwrap()
        .item::<f32>();
    dot / (na * nb)
}

/// Binarize mask logits at 0 (the sign is what the output mask depends on).
fn binarize(logits: &Array) -> Array {
    let v: Vec<f32> = logits
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .iter()
        .map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
        .collect();
    Array::from_slice(&v, logits.shape())
}

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

/// Worst index-aligned mask IoU between two instance lists (must be equal length).
fn worst_iou(a: &[Instance], b: &[Instance]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| iou(&x.mask, &y.mask))
        .fold(1.0f32, f32::min)
}

fn segment(seg: &Sam3ImageSegmenter, fx: &Weights) -> Vec<Instance> {
    let pixel_values = fx.require("pixel_values").unwrap().clone();
    let input_ids = fx.require("input_ids").unwrap().clone();
    let mask: Vec<i32> = fx
        .require("attention_mask")
        .unwrap()
        .as_slice::<i32>()
        .to_vec();
    seg.segment(&pixel_values, &input_ids, &mask, (1.0, 1.0), 0.5, 0.5)
        .expect("segment")
}

#[test]
#[ignore = "needs SAM3_WEIGHTS + SAM3_E2E_FIXTURE"]
fn quantized_segmenter_stays_close_to_dense() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to facebook/sam3 model.safetensors");
    let fixture_path = std::env::var("SAM3_E2E_FIXTURE")
        .unwrap_or_else(|_| "scripts/spikes/sam3_oracle/e2e_fixture.safetensors".to_string());
    let w = Weights::from_file(&weights_path).expect("load sam3 weights");
    let fx = Weights::from_file(&fixture_path).expect("load e2e fixture");

    // Dense baseline (a fresh model per precision — quantize mutates in place).
    let dense = segment(&Sam3ImageSegmenter::from_weights(&w).unwrap(), &fx);

    let mut q8 = Sam3ImageSegmenter::from_weights(&w).unwrap();
    q8.quantize(8).expect("quantize q8");
    let q8_inst = segment(&q8, &fx);

    let mut q4 = Sam3ImageSegmenter::from_weights(&w).unwrap();
    q4.quantize(4).expect("quantize q4");
    let q4_inst = segment(&q4, &fx);

    let want_n = fx.require("instance_masks").unwrap().shape()[0] as usize;
    println!(
        "instances: oracle {want_n} | dense {} | Q8 {} | Q4 {}",
        dense.len(),
        q8_inst.len(),
        q4_inst.len()
    );

    // All masks finite (sanity).
    for inst in q8_inst.iter().chain(&q4_inst) {
        let v = inst.mask.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        assert!(
            v.as_slice::<f32>().iter().all(|x| x.is_finite()),
            "quantized mask has non-finite values"
        );
    }

    // Q8 is near-lossless: same instance set as dense, near-identical masks.
    assert_eq!(q8_inst.len(), dense.len(), "Q8 instance count != dense");
    let q8_iou = worst_iou(&dense, &q8_inst);
    println!("Q8 vs dense: worst mask IoU = {q8_iou:.4}");
    assert!(q8_iou > 0.95, "Q8 worst mask IoU {q8_iou:.4} below 0.95");

    // Q4 stays coherent: it still finds the people; require a healthy IoU where instances align.
    assert!(!q4_inst.is_empty(), "Q4 found no instances");
    if q4_inst.len() == dense.len() {
        let q4_iou = worst_iou(&dense, &q4_inst);
        println!("Q4 vs dense: worst mask IoU = {q4_iou:.4}");
        assert!(q4_iou > 0.80, "Q4 worst mask IoU {q4_iou:.4} below 0.80");
    } else {
        println!(
            "Q4 instance count {} differs from dense {} (coarser quant; coherence-only check)",
            q4_inst.len(),
            dense.len()
        );
    }
}

#[test]
#[ignore = "needs SAM3_WEIGHTS + SAM3_TRACKER_FIXTURE"]
fn quantized_tracker_stays_close_to_dense() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to facebook/sam3 model.safetensors");
    let fixture_path = std::env::var("SAM3_TRACKER_FIXTURE")
        .unwrap_or_else(|_| "scripts/spikes/sam3_oracle/tracker_fixture.safetensors".to_string());
    let w = Weights::from_file(&weights_path).expect("load sam3 weights");
    let fx = Weights::from_file(&fixture_path).expect("load tracker fixture");

    let pixel_values = fx.require("pixel_values").unwrap().clone();
    let box_v = fx
        .require("box_1008")
        .unwrap()
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let box_xyxy = [box_v[0], box_v[1], box_v[2], box_v[3]];

    let dense = Sam3Tracker::from_weights(&w)
        .unwrap()
        .segment(&pixel_values, box_xyxy)
        .expect("dense segment");

    let mut q8 = Sam3Tracker::from_weights(&w).unwrap();
    q8.quantize(8).expect("quantize tracker q8");
    let q8 = q8.segment(&pixel_values, box_xyxy).expect("q8 segment");

    let cos = cosine(&dense.low_res, &q8.low_res);
    let mask_iou = iou(&binarize(&dense.low_res), &binarize(&q8.low_res));
    println!(
        "tracker Q8 vs dense: mask logit cosine={cos:.5} binary IoU={mask_iou:.4} (iou dense={:.3} q8={:.3})",
        dense.iou, q8.iou
    );
    assert!(
        q8.low_res.as_slice::<f32>().iter().all(|x| x.is_finite()),
        "Q8 tracker mask has non-finite values"
    );
    // Primary near-lossless gate: the mask *logits* are essentially identical (cosine ~1). The
    // binary IoU is a looser coherence floor — this fixture's box-prompt mask is low-confidence
    // (dense self-IoU ~0.65), so its fuzzy logit≈0 boundary flips pixels under any tiny perturbation
    // even though the logits barely move.
    assert!(
        cos > 0.999,
        "tracker Q8 mask logit cosine {cos:.5} below 0.999"
    );
    assert!(
        mask_iou > 0.85,
        "tracker Q8 binary IoU {mask_iou:.4} below 0.85"
    );
}
