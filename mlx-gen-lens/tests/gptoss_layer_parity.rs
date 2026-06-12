//! sc-3166 — Lens gpt-oss MoE + full-decoder-layer parity vs `transformers.GptOssDecoderLayer`.
//!
//! Self-contained: the golden (`tools/golden/lens_gptoss_layer_golden.safetensors`, gitignored,
//! real-weights) embeds layer-0's weights as stored — dense attn/router/norms + the experts'
//! **MXFP4** `*_blocks`/`*_scales` (uint8) — plus the I/O. So the Rust side exercises its own MXFP4
//! dequant + MoE + residual assembly end-to-end, with no 12 GB snapshot at test time. Dump it with
//! `~/Repos/mflux/.venv/bin/python tools/dump_lens_gptoss_layer_golden.py`.
//!
//! Two gates: (1) the MoE alone (router + clamped-SwiGLU experts) on a fresh input; (2) the full
//! decoder layer (RMSNorm + attention + MoE + residuals).
//!
//! Run: `cargo test -p mlx-gen-lens --test gptoss_layer_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max, multiply, subtract, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_lens::config::GptOssConfig;
use mlx_gen_lens::text_encoder::gpt_oss::{attention_mask, GptOssDecoderLayer};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/lens_gptoss_layer_golden.safetensors"
);

fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(got, want).unwrap()).unwrap();
    let denom = max(abs(want).unwrap(), None).unwrap().item::<f32>();
    max(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

/// Cosine similarity over the flattened tensors — robust to the output's wide dynamic range.
fn cosine(got: &Array, want: &Array) -> f32 {
    let dot = sum(multiply(got, want).unwrap(), None)
        .unwrap()
        .item::<f32>();
    let na = sum(multiply(got, got).unwrap(), None)
        .unwrap()
        .item::<f32>()
        .sqrt();
    let nb = sum(multiply(want, want).unwrap(), None)
        .unwrap()
        .item::<f32>()
        .sqrt();
    dot / (na * nb).max(1e-12)
}

#[test]
#[ignore = "needs tools/golden/lens_gptoss_layer_golden.safetensors (dump_lens_gptoss_layer_golden.py)"]
fn gptoss_decoder_layer_matches_reference() {
    let g = Weights::from_file(GOLDEN).expect("gpt-oss layer golden");
    let cfg = GptOssConfig::lens();
    let l: i32 = g.metadata("L").expect("L meta").parse().unwrap();
    let (inv_freq_vec, attn_scaling) = cfg.yarn_rope();
    let inv_freq = Array::from_slice(&inv_freq_vec, &[inv_freq_vec.len() as i32]);

    let layer =
        GptOssDecoderLayer::from_weights(&g, "model.layers.0", &cfg, Dtype::Float32).unwrap();

    // --- Gate 1: MoE alone (router + clamped-SwiGLU experts + MXFP4 dequant) ---
    let moe_in = g.require("moe_in").unwrap();
    let moe_out = layer.moe().forward(moe_in).unwrap();
    let want = g.require("moe_out").unwrap();
    let moe_pr = peak_rel(&moe_out, want);
    eprintln!(
        "gpt-oss MoE: cosine {:.7} peak_rel {moe_pr:.3e}",
        cosine(&moe_out, want)
    );
    assert!(moe_pr < 2e-2, "MoE peak_rel {moe_pr:.3e} exceeds tol 2e-2");

    // --- Gate 2: full decoder layer (norm + attn + MoE + residuals) ---
    let mask = attention_mask(l, None, Dtype::Float32).unwrap();
    let x = g.require("x").unwrap();
    let out = layer.forward(x, &inv_freq, attn_scaling, &mask).unwrap();
    let want = g.require("layer_out").unwrap();
    let pr = peak_rel(&out, want);
    // peak_rel gate matches the established gemma LLM-encoder precedent (2e-2): the residual error is
    // dominated by the attention path's mlx-Metal-vs-CPU f32-matmul floor (sc-3165 standalone 2.3e-3),
    // not a structural bug — the MoE sub-block is cosine 1.0000000. cosine reported for whole-output.
    eprintln!(
        "gpt-oss decoder-layer: cosine {:.7} peak_rel {pr:.3e}",
        cosine(&out, want)
    );
    assert!(
        pr < 2e-2,
        "decoder-layer peak_rel {pr:.3e} exceeds tol 2e-2"
    );
}
