//! sc-2348 slice 3: Qwen-Image MMDiT parity vs the frozen fork.
//!
//! `#[ignore]`d — needs the local golden from `tools/dump_qwen_transformer_golden.py` (gitignored).
//! Two checks (the ~20B full transformer is validated end-to-end against the image golden later):
//!
//! - **3D RoPE**: my `QwenRope3d` vs the fork's `QwenEmbedRopeMLX` (no weights).
//! - **One dual-stream block** at small dims with the fork's synthetic weights + rope.
//!
//! Run: `cargo test -p mlx-gen-qwen-image --release --test transformer_real_weights -- --ignored`

use mlx_gen::weights::Weights;
use mlx_gen_qwen_image::transformer::{QwenRope3d, QwenTransformerBlock};
use mlx_rs::Array;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/qwen_transformer_golden.safetensors"
);

// Must match tools/dump_qwen_transformer_golden.py.
const ROPE_H: usize = 64;
const ROPE_W: usize = 48;
const ROPE_TXT: usize = 20;

fn rel_errors(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let sum_abs_b: f64 = b.iter().map(|&v| v.abs() as f64).sum();
    let sum_abs_diff: f64 = a.iter().zip(b).map(|(&x, &y)| (x - y).abs() as f64).sum();
    (max_diff / peak, (sum_abs_diff / sum_abs_b) as f32)
}

#[test]
#[ignore = "needs local transformer golden"]
fn rope_matches_fork() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let (ic, is, tc, ts) = QwenRope3d::qwen_image()
        .forward(ROPE_H, ROPE_W, ROPE_TXT)
        .unwrap();
    for (name, got, key) in [
        ("img_cos", &ic, "rope_img_cos"),
        ("img_sin", &is, "rope_img_sin"),
        ("txt_cos", &tc, "rope_txt_cos"),
        ("txt_sin", &ts, "rope_txt_sin"),
    ] {
        let want = g.require(key).unwrap();
        assert_eq!(got.shape(), want.shape(), "{name} shape");
        let (peak, mean) = rel_errors(got, want);
        println!("rope {name}: peak-rel {peak:.3e}  mean-rel {mean:.3e}");
        assert!(peak < 1e-3, "rope {name} peak-rel {peak:.3e}");
    }
}

#[test]
#[ignore = "needs local transformer golden"]
fn dual_stream_block_matches_fork() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let block = QwenTransformerBlock::from_weights(&g, "", 2, 128).unwrap();
    let (enc_out, hid_out) = block
        .forward(
            g.require("io_hidden").unwrap(),
            g.require("io_enc").unwrap(),
            g.require("io_temb").unwrap(),
            g.require("io_img_cos").unwrap(),
            g.require("io_img_sin").unwrap(),
            g.require("io_txt_cos").unwrap(),
            g.require("io_txt_sin").unwrap(),
            None,
            None,
        )
        .unwrap();
    for (name, got, key) in [
        ("hidden_out", &hid_out, "io_hidden_out"),
        ("enc_out", &enc_out, "io_enc_out"),
    ] {
        let want = g.require(key).unwrap();
        assert_eq!(got.shape(), want.shape(), "{name} shape");
        let (peak, mean) = rel_errors(got, want);
        println!("block {name}: peak-rel {peak:.3e}  mean-rel {mean:.3e}");
        assert!(mean < 2e-3, "block {name} mean-rel {mean:.3e}");
        assert!(peak < 5e-3, "block {name} peak-rel {peak:.3e}");
    }
}
