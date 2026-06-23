//! sc-7569 — committed-fixture parity for the Krea 2 Qwen3-VL-4B text encoder against the
//! **transformers** `Qwen3VLTextModel` forward (an independent graph), at tiny dims.
//!
//! Exercises bias-less GQA, per-head q/k RMSNorm, HF half-split RoPE, the causal mask, and the
//! select-layer hidden-state stack + template-prefix slice — the `context` the DiT consumes. The
//! fixture is produced by `tools/dump_krea_te_golden.py` and committed, so this runs by default.
//! Tolerance 1e-2 (Metal fp32 matmul).

use mlx_gen::weights::Weights;
use mlx_gen_krea::{KreaTeConfig, KreaTextEncoder};
use mlx_rs::ops::{all_close, multiply, sqrt, sum};
use mlx_rs::{Array, Dtype};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/te_golden.safetensors"
);

fn cosine(a: &Array, b: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    let dot = sum(multiply(&a, &b).unwrap(), false).unwrap();
    let na = sqrt(sum(multiply(&a, &a).unwrap(), false).unwrap()).unwrap();
    let nb = sqrt(sum(multiply(&b, &b).unwrap(), false).unwrap()).unwrap();
    (dot / (na * nb)).item::<f32>()
}

/// Tiny config matching `tools/dump_krea_te_golden.py`.
fn tiny_te_config() -> KreaTeConfig {
    KreaTeConfig {
        hidden_size: 64,
        num_layers: 6,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 32,
        intermediate_size: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 5_000_000.0,
        select_hidden: vec![2, 4],
        prefix_tokens: 3,
    }
}

#[test]
fn te_matches_reference() {
    let w = Weights::from_file(FIXTURE)
        .unwrap_or_else(|e| panic!("load te fixture (run tools/dump_krea_te_golden.py): {e}"));
    let cfg = tiny_te_config();
    let te = KreaTextEncoder::from_weights(&w, "language_model", &cfg).unwrap();

    let hiddens = te
        .forward(
            w.require("in.input_ids").unwrap(),
            w.require("in.attention_mask").unwrap(),
        )
        .unwrap();
    let want = w.require("out.hiddens").unwrap();
    assert_eq!(hiddens.shape(), want.shape(), "stacked-context shape");

    let c = cosine(&hiddens, want);
    println!("Krea TE parity: cosine={c:.7}");
    assert!(c > 0.999, "TE cosine {c:.7} <= 0.999");
    assert!(
        all_close(&hiddens, want, 1e-2, 1e-2, false)
            .unwrap()
            .item::<bool>(),
        "TE stacked context diverged beyond 1e-2 (cosine {c:.7})"
    );
}
