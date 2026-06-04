//! S3a single video-block parity vs the reference `BasicAVTransformerBlock` (sc-2679 S3a).
//!
//! `#[ignore]`d: needs the real `ltx_2_3_base_q8` `transformer.safetensors` (~20 GB). The committed
//! golden (`tests/fixtures/ltx_block_golden.safetensors`, from `tools/dump_ltx_block_golden.py`)
//! holds the reference **f32** block I/O over synthetic inputs; this test loads the SAME block-0 Q8
//! weights, dequantizes them to f32, and checks the Rust `VideoBlock` reproduces the output. Honors
//! "divergence is not rounding": f32 dequant is bit-identical, so the only gap is SDPA/matmul
//! summation ordering (shared mlx ops → tight).
//!
//! Run: `LTX_BASE_DIR=… cargo test -p mlx-gen-ltx --test block_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max as max_op, subtract};
use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen_ltx::config::{LtxConfig, SplitModel};
use mlx_gen_ltx::transformer::{Precision, VideoBlock};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_block_golden.safetensors"
);

fn base_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("LTX_BASE_DIR") {
        return d.into();
    }
    let home = std::env::var("HOME").unwrap();
    std::path::PathBuf::from(home)
        .join("Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8")
}

fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(got, want).unwrap()).unwrap();
    let denom = max_op(abs(want).unwrap(), None).unwrap().item::<f32>();
    max_op(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

#[test]
#[ignore = "needs ltx_2_3_base_q8 transformer.safetensors (~20 GB)"]
fn block_matches_reference() {
    let dir = base_dir();
    let cfg = LtxConfig::from_model_dir(&dir).expect("embedded_config.json");
    let split = SplitModel::from_model_dir(&dir).expect("split_model.json");
    let w =
        Weights::from_file(dir.join("transformer.safetensors")).expect("transformer.safetensors");
    // The block-math gate: dequantize the checkpoint's Q8 weights to dense f32.
    let block = VideoBlock::load(
        &w,
        "transformer_blocks.0",
        &cfg,
        Precision::dense_f32(split.bits, split.group),
    )
    .expect("build block");

    let g = Weights::from_file(GOLDEN).expect("golden (run tools/dump_ltx_block_golden.py)");
    let x = g.require("x").unwrap();
    let context = g.require("context").unwrap();
    let timesteps = g.require("timesteps").unwrap();
    let prompt = g.require("prompt_timestep").unwrap();
    let cos = g.require("cos").unwrap();
    let sin = g.require("sin").unwrap();
    let want = g.require("out").unwrap();

    let got = block
        .forward(x, timesteps, Some(prompt), context, None, cos, sin)
        .expect("block forward");
    assert_eq!(got.shape(), want.shape(), "block output shape");
    let pr = peak_rel(&got, want);
    eprintln!("block peak_rel = {pr:.3e} shape={:?}", got.shape());
    assert!(pr < 5e-3, "block peak_rel {pr:.3e} too high");
}
