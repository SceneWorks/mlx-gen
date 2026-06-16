//! sc-5915: parity for the FLUX.2-**dev** Mistral text encoder vs the PyTorch reference
//! (`transformers.MistralModel`), on a TINY synthetic config (committed fixture
//! `tests/fixtures/te_dev_golden.safetensors` ← `tools/dump_flux2_te_dev_golden.py`).
//!
//! The dev encoder reuses the klein decoder-LM graph with **qk-norm off** (Mistral's delta from
//! Qwen3). This fixture exercises exactly that path plus the dev specifics: HF half-split RoPE at
//! θ=1e9, the causal+padding mask, `hidden_size (80) != num_heads*head_dim (64)` (head_dim
//! override, mirroring dev's 5120 vs 32*128), and the multi-layer hidden-state concat. A real
//! structural bug (qk-norm wrongly applied, wrong rope/theta, wrong GQA grouping, wrong layer
//! indices) diverges by orders of magnitude; f32 Metal matmul agrees to ~1e-3.

use mlx_gen::weights::Weights;
use mlx_gen_flux2::{Qwen3TextEncoder, Qwen3TextEncoderConfig};
use mlx_rs::ops::all_close;
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/te_dev_golden.safetensors"
);

fn close(a: &Array, b: &Array, rtol: f64, atol: f64) -> bool {
    all_close(a, b, rtol, atol, false).unwrap().item::<bool>()
}

/// The tiny Mistral config the dump script used (`qk_norm: false`, θ=1e9, head_dim override).
fn tiny_dev_config() -> Qwen3TextEncoderConfig {
    Qwen3TextEncoderConfig {
        hidden_size: 80,
        n_layers: 4,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 16,
        rope_theta: 1_000_000_000.0,
        rms_norm_eps: 1e-5,
        qk_norm: false,
        out_layers: [0, 1, 2],
    }
}

#[test]
fn mistral_dev_text_encoder_matches_reference() {
    let w = Weights::from_file(FIXTURE).unwrap();
    // Tiny `MistralModel` state_dict keys (`embed_tokens.weight`, `layers.{i}.…`) load directly
    // under the empty prefix — the real dev encoder loads under `language_model.model`.
    let te = Qwen3TextEncoder::from_weights(&w, "", &tiny_dev_config()).unwrap();
    let out = te
        .prompt_embeds(
            w.require("input_ids").unwrap(),
            w.require("attention_mask").unwrap(),
        )
        .unwrap();
    let want = w.require("prompt_embeds").unwrap();
    assert_eq!(out.shape(), want.shape(), "prompt_embeds shape (1,6,240)");
    // 1e-2 is the repo's matmul-bearing bar — Metal fp32 is reduced-precision and not bit-identical
    // cross-device; a real structural bug diverges orders of magnitude.
    assert!(
        close(&out, want, 1e-2, 1e-2),
        "Mistral (dev) prompt_embeds diverged from the PyTorch reference"
    );
}
