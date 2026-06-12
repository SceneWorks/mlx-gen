//! SAM3-F2 tracker memory-encoder parity (sc-4924): load the real `facebook/sam3` weights, run the
//! `Sam3Tracker` memory encoder (`encode_new_memory` + `prepare_mask_for_mem`), and check it against
//! the torch oracle (`scripts/spikes/sam3_oracle/dump_memory_fixture.py`).
//!
//! Run:
//!   SAM3_WEIGHTS=$HOME/.cache/huggingface/hub/models--facebook--sam3/snapshots/<rev>/model.safetensors \
//!   SAM3_MEMORY_FIXTURE=scripts/spikes/sam3_oracle/memory_fixture.safetensors \
//!     cargo test -p mlx-gen-sam3 --release --test memory_parity -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_gen_sam3::Sam3Tracker;
use mlx_rs::ops::{add, multiply, sqrt, sum};
use mlx_rs::{Array, Dtype};

fn scalar(a: &Array) -> f32 {
    a.as_dtype(Dtype::Float32).unwrap().item::<f32>()
}

fn cosine(a: &Array, b: &Array) -> f32 {
    let a = a.reshape(&[-1]).unwrap();
    let b = b.reshape(&[-1]).unwrap();
    let dot = scalar(&sum(multiply(&a, &b).unwrap(), None).unwrap());
    let na = scalar(&sqrt(sum(multiply(&a, &a).unwrap(), None).unwrap()).unwrap());
    let nb = scalar(&sqrt(sum(multiply(&b, &b).unwrap(), None).unwrap()).unwrap());
    dot / (na * nb)
}

/// NCHW `[1,C,H,W]` fixture → NHWC `[1,H,W,C]` to match our layout.
fn to_nhwc(a: &Array) -> Array {
    a.transpose_axes(&[0, 2, 3, 1]).unwrap()
}

#[test]
#[ignore = "needs SAM3_WEIGHTS=<facebook/sam3 model.safetensors> + SAM3_MEMORY_FIXTURE"]
fn memory_encoder_matches_oracle() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to facebook/sam3 model.safetensors");
    let fixture_path = std::env::var("SAM3_MEMORY_FIXTURE")
        .unwrap_or_else(|_| "scripts/spikes/sam3_oracle/memory_fixture.safetensors".to_string());

    let w = Weights::from_file(&weights_path).expect("load sam3 weights");
    let fx = Weights::from_file(&fixture_path).expect("load memory fixture");
    let tracker = Sam3Tracker::from_weights(&w).expect("build tracker");

    let pix_feat = to_nhwc(fx.require("pix_feat").unwrap()); // [1,72,72,256]
    let pred_high_res = fx.require("pred_masks_high_res").unwrap().clone(); // [1,1,1008,1008]
    let obj_score = scalar(fx.require("object_score_logits").unwrap());

    // --- Stage 1: mask prep (bilinear resize 1008→1152 + sigmoid/binarize + scale·20−10).
    for (key, is_pts) in [("mask_for_mem", false), ("mask_for_mem_bin", true)] {
        let got = tracker
            .prepare_mask_for_mem(&pred_high_res, is_pts)
            .unwrap(); // [1,1152,1152,1]
        let want = to_nhwc(fx.require(key).unwrap()); // [1,1152,1152,1]
        assert_eq!(got.shape(), want.shape(), "{key} shape");
        let c = cosine(&got, &want);
        println!("{key}: cosine={c:.7}");
        assert!(c > 0.9999, "{key} cosine {c}");
    }

    // --- Stage 2: full encoder (mask prep → downsampler/feature_projection/fuser/projection) +
    // sine position encoding. obj_score>0 here, so occlusion is inactive (final == raw).
    let out = tracker
        .encode_new_memory(&pix_feat, &pred_high_res, obj_score, false)
        .expect("encode_new_memory");
    let want_feat = to_nhwc(fx.require("maskmem_features_final").unwrap()); // [1,72,72,64]
    let want_pos = to_nhwc(fx.require("maskmem_pos_enc").unwrap());
    assert_eq!(out.features.shape(), want_feat.shape(), "features shape");
    assert_eq!(out.pos.shape(), want_pos.shape(), "pos shape");
    let cf = cosine(&out.features, &want_feat);
    let cp = cosine(&out.pos, &want_pos);
    println!("maskmem_features: cosine={cf:.7}\nmaskmem_pos_enc: cosine={cp:.7}");
    // features run stacked convs (MLX Metal reduced-precision matmul); pos enc is deterministic.
    assert!(cf > 0.999, "features cosine {cf}");
    assert!(cp > 0.99999, "pos_enc cosine {cp}");

    // --- Stage 3: occlusion add. Force object absent (score ≤ 0): features should gain the
    // occlusion spatial embedding over the whole grid. Expected = oracle raw + occ (NHWC broadcast).
    let occ = w
        .require("tracker_model.occlusion_spatial_embedding_parameter")
        .unwrap()
        .reshape(&[1, 1, 1, 64])
        .unwrap();
    let raw = to_nhwc(fx.require("maskmem_features_raw").unwrap());
    let expected_occ = add(&raw, &occ).unwrap();
    let out_occ = tracker
        .encode_new_memory(&pix_feat, &pred_high_res, -1.0, false)
        .expect("encode_new_memory occluded");
    let co = cosine(&out_occ.features, &expected_occ);
    println!("occlusion-add: cosine={co:.7}");
    assert!(co > 0.999, "occlusion-add cosine {co}");
}
