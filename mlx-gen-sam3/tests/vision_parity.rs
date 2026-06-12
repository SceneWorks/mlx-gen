//! SAM3-A vision-encoder parity (sc-4919): load the real `facebook/sam3` weights, run the PE
//! backbone + FPN neck, and check the four feature maps against the torch oracle fixture
//! (`scripts/spikes/sam3_oracle/dump_vision_fixture.py`).
//!
//! Run:
//!   SAM3_WEIGHTS=$HOME/.cache/huggingface/hub/models--facebook--sam3/snapshots/<rev>/model.safetensors \
//!   SAM3_VISION_FIXTURE=scripts/spikes/sam3_oracle/vision_fixture.safetensors \
//!     cargo test -p mlx-gen-sam3 --release --test vision_parity -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_gen_sam3::{Sam3VisionConfig, Sam3VisionEncoder};
use mlx_rs::ops::{abs, maximum, multiply, sqrt, subtract, sum};
use mlx_rs::Array;

fn scalar_f32(a: &Array) -> f32 {
    a.as_dtype(mlx_rs::Dtype::Float32).unwrap().item::<f32>()
}

/// Cosine similarity between two arrays (flattened).
fn cosine(a: &Array, b: &Array) -> f32 {
    let dot = scalar_f32(&sum(multiply(a, b).unwrap(), None).unwrap());
    let na = scalar_f32(&sqrt(sum(multiply(a, a).unwrap(), None).unwrap()).unwrap());
    let nb = scalar_f32(&sqrt(sum(multiply(b, b).unwrap(), None).unwrap()).unwrap());
    dot / (na * nb)
}

#[test]
#[ignore = "needs SAM3_WEIGHTS=<facebook/sam3 model.safetensors> + SAM3_VISION_FIXTURE"]
fn vision_encoder_matches_oracle() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to facebook/sam3 model.safetensors");
    let fixture_path = std::env::var("SAM3_VISION_FIXTURE")
        .unwrap_or_else(|_| "scripts/spikes/sam3_oracle/vision_fixture.safetensors".to_string());

    let w = Weights::from_file(&weights_path).expect("load sam3 weights");
    let fx = Weights::from_file(&fixture_path).expect("load vision fixture");

    let enc = Sam3VisionEncoder::from_weights(
        &w,
        "detector_model.vision_encoder",
        &Sam3VisionConfig::sam3(),
    )
    .expect("build vision encoder");

    let pixel_values = fx
        .require("pixel_values")
        .expect("fixture pixel_values")
        .clone();
    let fpn = enc.forward(&pixel_values).expect("vision forward");

    assert_eq!(fpn.len(), 4, "expected 4 FPN levels");
    let mut worst_cos = 1.0f32;
    for (i, got_nhwc) in fpn.iter().enumerate() {
        // ours NHWC [1,H,W,256] → NCHW to match the fixture
        let got = got_nhwc.transpose_axes(&[0, 3, 1, 2]).unwrap();
        let want = fx
            .require(&format!("fpn_{i}"))
            .expect("fixture fpn")
            .clone();
        assert_eq!(got.shape(), want.shape(), "fpn_{i} shape");

        let diff = abs(subtract(&got, &want).unwrap()).unwrap();
        let max_abs = scalar_f32(&mlx_rs::ops::max(&diff, None).unwrap());
        let denom = scalar_f32(
            &maximum(
                mlx_rs::ops::max(abs(&want).unwrap(), None).unwrap(),
                Array::from_f32(1e-6),
            )
            .unwrap(),
        );
        let cos = cosine(&got, &want);
        worst_cos = worst_cos.min(cos);
        println!(
            "fpn_{i} {:?}: cosine={:.6}  max_abs={:.5}  max_rel={:.5}",
            want.shape(),
            cos,
            max_abs,
            max_abs / denom
        );
    }
    assert!(
        worst_cos > 0.99,
        "worst FPN cosine {worst_cos:.6} below 0.99"
    );
}
