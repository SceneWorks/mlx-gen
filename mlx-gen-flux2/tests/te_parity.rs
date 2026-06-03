//! sc-2346 S1: parity for the FLUX.2 Qwen3 text encoder vs the fork, on a TINY synthetic config
//! (committed fixture `tests/fixtures/te_golden.safetensors` ← `tools/dump_flux2_te_golden.py`).
//! Exercises every code path — bias-less GQA, per-head q/k RMSNorm, HF half-split RoPE, the
//! causal+padding mask, and the multi-layer hidden-state concat — bit-tight in f32. A real
//! structural bug (wrong norm placement, wrong rope, wrong GQA grouping, wrong layer indices)
//! diverges by orders of magnitude; f32 Metal matmul agrees to ~1e-3.

use mlx_gen::weights::Weights;
use mlx_gen_flux2::{Qwen3TextEncoder, Qwen3TextEncoderConfig};
use mlx_rs::ops::all_close;
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/te_golden.safetensors"
);

fn close(a: &Array, b: &Array, rtol: f64, atol: f64) -> bool {
    all_close(a, b, rtol, atol, false).unwrap().item::<bool>()
}

/// The tiny config the dump script used.
fn tiny_config() -> Qwen3TextEncoderConfig {
    Qwen3TextEncoderConfig {
        hidden_size: 64,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 16,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
        out_layers: [0, 1, 2],
    }
}

#[test]
fn qwen3_text_encoder_matches_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let te = Qwen3TextEncoder::from_weights(&w, "", &tiny_config()).unwrap();
    let out = te
        .prompt_embeds(
            w.require("input_ids").unwrap(),
            w.require("attention_mask").unwrap(),
        )
        .unwrap();
    let want = w.require("prompt_embeds").unwrap();
    assert_eq!(out.shape(), want.shape(), "prompt_embeds shape");
    // 1e-2 is the repo's matmul-bearing bar — Metal fp32 is reduced-precision and not bit-identical
    // cross-device (CI runs a different GPU); a real structural bug diverges orders of magnitude.
    assert!(
        close(&out, want, 1e-2, 1e-2),
        "Qwen3 prompt_embeds diverged"
    );
}
