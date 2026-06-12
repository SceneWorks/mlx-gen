//! SAM3-B text-encoder parity (sc-4920): load the real `facebook/sam3` weights, run the CLIP-H
//! text tower + projection, and check the projected text features against the torch oracle
//! (`scripts/spikes/sam3_oracle/dump_text_fixture.py`). Also checks the tokenizer ids.
//!
//! Run:
//!   SAM3_WEIGHTS=.../model.safetensors \
//!   SAM3_TEXT_FIXTURE=scripts/spikes/sam3_oracle/text_fixture.safetensors \
//!   SAM3_TOKENIZER=.../tokenizer.json \
//!     cargo test -p mlx-gen-sam3 --release --test text_parity -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_gen_sam3::{Sam3TextConfig, Sam3TextEncoder, Sam3Tokenizer};
use mlx_rs::ops::{multiply, sqrt, sum};
use mlx_rs::Array;

fn scalar_f32(a: &Array) -> f32 {
    a.as_dtype(mlx_rs::Dtype::Float32).unwrap().item::<f32>()
}

fn cosine(a: &Array, b: &Array) -> f32 {
    let dot = scalar_f32(&sum(multiply(a, b).unwrap(), None).unwrap());
    let na = scalar_f32(&sqrt(sum(multiply(a, a).unwrap(), None).unwrap()).unwrap());
    let nb = scalar_f32(&sqrt(sum(multiply(b, b).unwrap(), None).unwrap()).unwrap());
    dot / (na * nb)
}

#[test]
#[ignore = "needs SAM3_WEIGHTS + SAM3_TEXT_FIXTURE (+ SAM3_TOKENIZER for the tokenizer check)"]
fn text_encoder_matches_oracle() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to facebook/sam3 model.safetensors");
    let fixture_path = std::env::var("SAM3_TEXT_FIXTURE")
        .unwrap_or_else(|_| "scripts/spikes/sam3_oracle/text_fixture.safetensors".to_string());

    let w = Weights::from_file(&weights_path).expect("load sam3 weights");
    let fx = Weights::from_file(&fixture_path).expect("load text fixture");
    let cfg = Sam3TextConfig::sam3();
    let enc = Sam3TextEncoder::from_weights(
        &w,
        "detector_model.text_encoder.text_model",
        "detector_model.text_projection",
        &cfg,
    )
    .expect("build text encoder");

    // Optional: verify the tokenizer reproduces the reference ids for "person".
    if let Ok(tok_path) = std::env::var("SAM3_TOKENIZER") {
        let tok = Sam3Tokenizer::from_file(&tok_path, &cfg).expect("load tokenizer");
        let (ids, mask) = tok.encode("person").expect("tokenize");
        let ids_host = ids.as_slice::<i32>().to_vec();
        assert_eq!(&ids_host[..3], &[49406, 2533, 49407], "person ids[:3]");
        assert_eq!(ids_host.len(), 32, "padded to 32");
        assert_eq!(&mask[..4], &[1, 1, 1, 0], "attention mask");
        println!("tokenizer: 'person' → ids[:3]={:?} ok", &ids_host[..3]);
    }

    let mut worst = 1.0f32;
    for concept in ["person", "car"] {
        let input_ids = fx.require(&format!("{concept}.input_ids")).unwrap().clone();
        let mask: Vec<i32> = fx
            .require(&format!("{concept}.attention_mask"))
            .unwrap()
            .as_slice::<i32>()
            .to_vec();
        let n_valid = mask.iter().filter(|&&m| m == 1).count();

        let got = enc.forward(&input_ids, &mask).expect("text forward"); // [1,N,256]
        let want = fx
            .require(&format!("{concept}.text_features"))
            .unwrap()
            .clone();
        assert_eq!(got.shape(), want.shape(), "{concept} text_features shape");

        // Slice the valid (non-padding) tokens — what the detector consumes.
        let valid_idx = Array::from_slice(
            &(0..n_valid as i32).collect::<Vec<i32>>(),
            &[n_valid as i32],
        );
        let got_v = got.take_axis(&valid_idx, 1).unwrap();
        let want_v = want.take_axis(&valid_idx, 1).unwrap();

        let cos_full = cosine(&got, &want);
        let cos_valid = cosine(&got_v, &want_v);
        worst = worst.min(cos_valid);
        println!(
            "{concept}: cosine_valid({n_valid} tok)={cos_valid:.6}  cosine_full={cos_full:.6}"
        );
    }
    assert!(
        worst > 0.999,
        "worst valid-token cosine {worst:.6} below 0.999"
    );
}
