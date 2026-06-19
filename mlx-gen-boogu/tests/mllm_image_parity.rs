//! E7b-2 (sc-6479) — real-weight parity for the Boogu Qwen3-VL **image-conditioned MLLM forward**.
//!
//! Feeds the reference's own vision outputs (`image_embeds` + the 3 deepstack features, from the
//! golden) into [`BooguTextEncoder::last_hidden_with_image`] and checks the merged `last_hidden_state`
//! matches the transformers `Qwen3VLModel` forward (vision splice + 3-D interleaved MRoPE + deepstack
//! injection). Using the golden's vision tensors isolates the LM forward from the vision tower (which
//! E7b-1 already validated). Goldens come from `reference/dump_boogu_vision_golden.py`.
//!
//! `#[ignore]` — needs the Base snapshot (`mllm/`) + the golden:
//!   BOOGU_BASE_DIR=<snapshot> CARGO_TARGET_DIR=~/Repos/mlx-gen/target \
//!     cargo test -p mlx-gen-boogu --test mllm_image_parity -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_boogu::load_text_encoder;
use mlx_rs::ops::{multiply, sqrt, sum};
use mlx_rs::{Array, Dtype};

/// Qwen3-VL image placeholder token (`mllm/config.json::image_token_id`).
const IMAGE_TOKEN_ID: i32 = 151655;

fn cosine(a: &Array, b: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    let dot = sum(multiply(&a, &b).unwrap(), false).unwrap();
    let na = sqrt(sum(multiply(&a, &a).unwrap(), false).unwrap()).unwrap();
    let nb = sqrt(sum(multiply(&b, &b).unwrap(), false).unwrap()).unwrap();
    (dot / (na * nb)).item::<f32>()
}

fn snapshot_dir() -> PathBuf {
    PathBuf::from(std::env::var("BOOGU_BASE_DIR").expect("set BOOGU_BASE_DIR to the snapshot root"))
}

fn golden_path() -> PathBuf {
    std::env::var("BOOGU_VISION_GOLDEN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME"))
                .join("Repos/mlx-gen-wt-boogu/reference/goldens/boogu_vision.safetensors")
        })
}

#[test]
#[ignore = "needs real weights + golden (dump_boogu_vision_golden.py)"]
fn mllm_image_matches_reference() {
    let g = Weights::from_file(golden_path()).expect("golden — run dump_boogu_vision_golden.py");
    let te = load_text_encoder(snapshot_dir()).expect("load Qwen3-VL text tower");

    let input_ids = g
        .require("input_ids")
        .unwrap()
        .as_dtype(Dtype::Int32)
        .unwrap();
    let attn = g
        .require("attention_mask")
        .unwrap()
        .as_dtype(Dtype::Int32)
        .unwrap();
    // Feed the exact vision tensors the reference mllm consumed internally (bf16 tower), isolating
    // the LM merged forward (the f32 tower port is validated separately in E7b-1).
    let image_embeds = g.require("image_embeds").unwrap().clone();
    let deepstack: Vec<Array> = (0..3)
        .map(|i| g.require(&format!("deepstack_{i}")).unwrap().clone())
        .collect();

    let grid = g
        .require("image_grid_thw")
        .unwrap()
        .as_dtype(Dtype::Int32)
        .unwrap();
    let gv = grid.as_slice::<i32>();
    let grid_thw = [gv[0], gv[1], gv[2]];

    let out = te
        .last_hidden_with_image(
            &input_ids,
            &attn,
            &image_embeds,
            &deepstack,
            grid_thw,
            IMAGE_TOKEN_ID,
        )
        .expect("image-conditioned MLLM forward");

    let want = g.require("mllm_last_hidden").unwrap();
    assert_eq!(out.shape(), want.shape(), "last_hidden_state shape");
    let c = cosine(&out, want);
    println!("MLLM image-conditioned last_hidden parity cosine = {c:.7}");
    assert!(
        c > 0.999,
        "MLLM image-conditioned parity cosine {c} too low"
    );
}
