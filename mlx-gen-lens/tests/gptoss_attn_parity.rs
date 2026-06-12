//! sc-3165 — Lens gpt-oss attention-core parity vs `transformers.GptOssAttention` (eager, sink path).
//!
//! Self-contained: the golden (`tools/golden/lens_gptoss_attn_golden.safetensors`, gitignored,
//! real-weights) embeds layer-0's dense attention weights + the input/output + the reference YaRN
//! `inv_freq`, so this test needs **only** the golden file, not the 12 GB snapshot. Dump it with
//! `~/Repos/mflux/.venv/bin/python tools/dump_lens_gptoss_attn_golden.py`.
//!
//! Two gates: (1) the Rust YaRN `inv_freq`/`attention_scaling` derivation matches the reference
//! rotary embedding; (2) the f32 attention forward (GQA + sinks + RoPE) reproduces the reference
//! attention output near-bit.
//!
//! Run: `cargo test -p mlx-gen-lens --test gptoss_attn_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_lens::config::GptOssConfig;
use mlx_gen_lens::text_encoder::gpt_oss::{attention_mask, GptOssAttention};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/lens_gptoss_attn_golden.safetensors"
);

fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(got, want).unwrap()).unwrap();
    let denom = max(abs(want).unwrap(), None).unwrap().item::<f32>();
    max(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

#[test]
#[ignore = "needs tools/golden/lens_gptoss_attn_golden.safetensors (dump_lens_gptoss_attn_golden.py)"]
fn gptoss_attention_matches_reference() {
    let g = Weights::from_file(GOLDEN).expect("gpt-oss attn golden");
    let cfg = GptOssConfig::lens();
    let l: i32 = g.metadata("L").expect("L meta").parse().unwrap();

    // --- Gate 1: YaRN derivation vs the reference rotary embedding ---
    let (inv_freq_vec, attn_scaling) = cfg.yarn_rope();
    let inv_freq = Array::from_slice(&inv_freq_vec, &[inv_freq_vec.len() as i32]);
    let ref_inv = g.require("ref_inv_freq").unwrap();
    let inv_pr = peak_rel(&inv_freq, ref_inv);
    eprintln!("yarn inv_freq peak_rel vs reference: {inv_pr:.3e}");
    assert!(
        inv_pr < 1e-5,
        "YaRN inv_freq diverges from reference: {inv_pr:.3e}"
    );

    let ref_scaling: f32 = g.metadata("attention_scaling").unwrap().parse().unwrap();
    assert!(
        (attn_scaling - ref_scaling).abs() < 1e-5,
        "attention_scaling {attn_scaling} != reference {ref_scaling}"
    );

    // --- Gate 2: attention forward (f32) ---
    let attn = GptOssAttention::from_weights(&g, "model.layers.0.self_attn", &cfg, Dtype::Float32)
        .unwrap();
    // Layer 0 is sliding, but L < sliding_window so the mask equals full causal.
    let mask = attention_mask(l, None, Dtype::Float32).unwrap();
    let x = g.require("x").unwrap();

    let out = attn.forward(x, &inv_freq, attn_scaling, &mask).unwrap();
    let want = g.require("attn_out").unwrap();
    let pr = peak_rel(&out, want);
    eprintln!("gpt-oss attention-core peak_rel: {pr:.3e}");
    // f32 path: bounded by the mlx-Metal f32-matmul floor (~2-4e-3), not a structural error.
    assert!(
        pr < 5e-3,
        "attention-core peak_rel {pr:.3e} exceeds tol 5e-3"
    );
}
