//! sc-6266 — activation-chunking equivalence gate for the FLUX.2 MMDiT.
//!
//! Proves the [`MemoryConfig`] memory levers change only the activation *schedule*, not the result —
//! so the correctness gate (`transformer_parity.rs`, which runs with the default
//! [`MemoryConfig::OFF`]) keeps covering the math while the gated long-sequence multi-reference edit
//! path runs with the levers on (`model.rs`). Two equivalence classes, asserted separately on the
//! committed tiny fixture the parity test already carries (`tests/fixtures/transformer_golden.safetensors`):
//!   * **`eval_per_block` is exactly bit-identical** (max|Δ| == 0) — it only forces materialization
//!     of the same graph, so the multi-reference edit's pixels are unchanged. This is the dominant
//!     memory lever and the production default ([`MemoryConfig::LONG_SEQ`]), so the win is bit-exact.
//!   * **FFN sequence-chunking is numerically equivalent** (cosine ≥ 0.9999999) — the FFN is
//!     per-token so the math is identical, but MLX's Metal GEMM is tile-specialized by the row (M)
//!     dimension, so a `[chunk, k]` matmul can round slightly differently from the full `[L, k]` one
//!     (the same class as the model's own torch parity). It is off by default; on as env-tunable
//!     headroom for extreme configs.
//!
//! Self-consistent: it compares `forward(OFF)` against `forward(levered)` on the **same** model +
//! inputs, so it needs no torch reference. The deliberately tiny chunk size (down to 1 token) forces
//! the multi-chunk + ragged-remainder paths on the fixture's 4-token image sequence.

use mlx_gen::weights::Weights;
use mlx_gen_flux2::{Flux2Config, Flux2ForwardInputs, Flux2Transformer, MemoryConfig};
use mlx_rs::{Array, Dtype};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/transformer_golden.safetensors"
);

/// The tiny config the dump script used (inner = 2·8 = 16) — identical to `transformer_parity.rs`.
fn tiny_config() -> Flux2Config {
    Flux2Config {
        num_double_layers: 1,
        num_single_layers: 1,
        num_heads: 2,
        head_dim: 8,
        in_channels: 4,
        out_channels: 4,
        joint_attention_dim: 12,
        mlp_ratio: 3.0,
        timestep_channels: 16,
        axes_dim: [2, 2, 2, 2],
        rope_theta: 2000.0,
        te_hidden_size: 4,
        te_intermediate_size: 12,
        te_out_layers: [0, 1, 2],
        max_sequence_length: 512,
        num_latent_channels: 1,
        vae_scale_factor: 8,
    }
}

fn flat(a: &Array) -> Vec<f32> {
    a.reshape(&[-1])
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

/// (cosine similarity, max abs diff) between two same-shape tensors.
fn compare(a: &Array, b: &Array) -> (f32, f32) {
    let (va, vb) = (flat(a), flat(b));
    assert_eq!(va.len(), vb.len(), "shape mismatch");
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    let mut max_abs = 0f32;
    for (x, y) in va.iter().zip(&vb) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
        max_abs = max_abs.max((x - y).abs());
    }
    let cos = (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32;
    (cos, max_abs)
}

fn forward_mem(t: &Flux2Transformer, w: &Weights, mem: &MemoryConfig) -> Array {
    t.forward_with_mem(
        &Flux2ForwardInputs {
            hidden_states: w.require("hidden").unwrap(),
            encoder_hidden_states: w.require("encoder").unwrap(),
            img_ids: w.require("img_ids").unwrap(),
            txt_ids: w.require("txt_ids").unwrap(),
            timestep: 500.0,
            guidance: None,
        },
        None,
        mem,
    )
    .unwrap()
}

#[test]
fn eval_per_block_is_bit_identical() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let t = Flux2Transformer::from_weights(&w, &tiny_config()).unwrap();
    let base = forward_mem(&t, &w, &MemoryConfig::OFF);
    // LONG_SEQ = eval_per_block only (the production long-sequence default).
    let levered = forward_mem(&t, &w, &MemoryConfig::LONG_SEQ);
    assert_eq!(base.shape(), levered.shape(), "eval_per_block out shape");
    let (cos, max_abs) = compare(&base, &levered);
    assert_eq!(
        max_abs, 0.0,
        "eval_per_block must be bit-identical (max|Δ| {max_abs}, cos {cos})"
    );
}

#[test]
fn ffn_seq_chunk_is_numerically_equivalent() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let t = Flux2Transformer::from_weights(&w, &tiny_config()).unwrap();
    let base = forward_mem(&t, &w, &MemoryConfig::OFF);

    // chunk 1/2/3 over the 4-token image FFN exercise the multi-chunk + ragged-remainder paths.
    for chunk in [1usize, 2, 3] {
        let mem = MemoryConfig {
            ffn_seq_chunk: Some(chunk),
            eval_per_block: false,
        };
        let chunked = forward_mem(&t, &w, &mem);
        assert_eq!(base.shape(), chunked.shape(), "chunk {chunk} out shape");
        let (cos, max_abs) = compare(&base, &chunked);
        assert!(
            cos >= 0.999_999_9,
            "ffn chunk {chunk} diverged (cos {cos}, max|Δ| {max_abs})"
        );
    }

    // Production-style combination (eval-to-free + FFN chunk) is still equivalent.
    let combined = forward_mem(
        &t,
        &w,
        &MemoryConfig {
            ffn_seq_chunk: Some(2),
            eval_per_block: true,
        },
    );
    let (cos, max_abs) = compare(&base, &combined);
    assert!(
        cos >= 0.999_999_9,
        "eval + ffn chunk diverged (cos {cos}, max|Δ| {max_abs})"
    );
}
