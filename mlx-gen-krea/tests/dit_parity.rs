//! sc-7568 — committed-fixture parity for the Krea 2 single-stream DiT against the Krea-published
//! reference (`github.com/krea-ai/krea-2` `mmdit.py` `SingleStreamDiT`), at tiny dims.
//!
//! Block-level + full-DiT-forward parity (the story AC). The fixtures are produced by
//! `tools/dump_krea_dit_golden.py` (random seeded weights, remapped to the diffusers checkpoint keys)
//! and committed under `tests/fixtures/` — so these run by default. Tolerance 1e-2 matches the spike +
//! the rest of the repo: MLX runs fp32 matmul in reduced precision on Metal (~3–4 sig figs).

use mlx_gen::weights::Weights;
use mlx_gen_krea::transformer::block::{SingleStreamBlock, TextFusionTransformer};
use mlx_gen_krea::transformer::rope::RopeTables;
use mlx_gen_krea::{Krea2Config, Krea2Transformer};
use mlx_rs::ops::{all_close, multiply, sqrt, subtract, sum};
use mlx_rs::{Array, Dtype};

const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/");

// Tiny config shared by the dump script (`mmdit` derives axes [8,12,12] from head_dim 32).
const HEADS: i32 = 4;
const KV: i32 = 2;
const HEAD_DIM: i32 = 32;
const HIDDEN: i32 = 128;
const TXT_HEADS: i32 = 2;
const EPS: f32 = 1e-5;

fn load(name: &str) -> Weights {
    Weights::from_file(format!("{FIX}{name}")).unwrap_or_else(|e| {
        panic!("load fixture {name} (run tools/dump_krea_dit_golden.py): {e}");
    })
}

fn cosine(a: &Array, b: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    let dot = sum(multiply(&a, &b).unwrap(), false).unwrap();
    let na = sqrt(sum(multiply(&a, &a).unwrap(), false).unwrap()).unwrap();
    let nb = sqrt(sum(multiply(&b, &b).unwrap(), false).unwrap()).unwrap();
    (dot / (na * nb)).item::<f32>()
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    mlx_rs::ops::max(mlx_rs::ops::abs(subtract(&a, &b).unwrap()).unwrap(), false)
        .unwrap()
        .item::<f32>()
}

/// The #1 parity risk localized: the 3-axis interleaved RoPE table for the DiT's joint positions
/// (`cap_len` text `(0,0,0)` + an `ht×wt` grid `(0,row,col)`) must match the reference cos/sin exactly.
#[test]
fn rope_matches_reference() {
    let g = load("rope_golden.safetensors");
    // meta = [n_tok, ht, wt, ax0, ax1, ax2] (see the dump); theta fixed at 1000.
    let (cap, ht, wt) = (5usize, 4usize, 4usize);
    let (cos, sin) = RopeTables::build_t2i(cap, ht, wt, [8, 12, 12], 1000.0).joint();

    let want_cos = g.require("cos").unwrap();
    let want_sin = g.require("sin").unwrap();
    assert_eq!(cos.shape(), want_cos.shape(), "cos shape");
    assert!(
        all_close(&cos, want_cos, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>(),
        "rope cos diverged (max abs {:e})",
        max_abs_diff(&cos, want_cos)
    );
    assert!(
        all_close(&sin, want_sin, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>(),
        "rope sin diverged (max abs {:e})",
        max_abs_diff(&sin, want_sin)
    );
}

/// One `SingleStreamBlock`: DoubleSharedModulation (6-factor pre/post), the sigmoid-gated GQA
/// attention with interleaved RoPE, and the SwiGLU FFN.
#[test]
fn single_block_matches_reference() {
    let w = load("single_block_golden.safetensors");
    let blk = SingleStreamBlock::from_weights(&w, "blk", HEADS, KV, HEAD_DIM, HIDDEN, EPS).unwrap();
    let y = blk
        .forward(
            w.require("in.x").unwrap(),
            w.require("in.tvec").unwrap(),
            w.require("in.cos").unwrap(),
            w.require("in.sin").unwrap(),
        )
        .unwrap();
    let want = w.require("out.y").unwrap();
    assert_eq!(y.shape(), want.shape());
    let c = cosine(&y, want);
    println!(
        "single_block parity: cosine={c:.7} max_abs={:e}",
        max_abs_diff(&y, want)
    );
    assert!(
        all_close(&y, want, 1e-2, 1e-2, false)
            .unwrap()
            .item::<bool>(),
        "single block diverged beyond 1e-2 (cosine {c:.7})"
    );
}

/// The `TextFusionTransformer`: layer-axis aggregation (attention across the stacked layers) →
/// `projector` 12→1 collapse → token-axis refiner blocks.
#[test]
fn text_fusion_matches_reference() {
    let w = load("text_fusion_golden.safetensors");
    let tf =
        TextFusionTransformer::from_weights(&w, 2, 2, TXT_HEADS, TXT_HEADS, HEAD_DIM, EPS).unwrap();
    let y = tf.forward(w.require("in.x").unwrap()).unwrap();
    let want = w.require("out.y").unwrap();
    assert_eq!(y.shape(), want.shape());
    let c = cosine(&y, want);
    println!(
        "text_fusion parity: cosine={c:.7} max_abs={:e}",
        max_abs_diff(&y, want)
    );
    assert!(
        all_close(&y, want, 1e-2, 1e-2, false)
            .unwrap()
            .item::<bool>(),
        "text_fusion diverged beyond 1e-2 (cosine {c:.7})"
    );
}

/// Tiny config matching `tools/dump_krea_dit_golden.py::dump_dit` (the SwiGLU inner dims are read from
/// the weights, so `intermediate_size` is documentary).
fn tiny_dit_config() -> Krea2Config {
    Krea2Config {
        in_channels: 16,
        patch_size: 2,
        hidden_size: 128,
        num_attention_heads: 4,
        num_kv_heads: 2,
        attention_head_dim: 32,
        num_layers: 2,
        intermediate_size: 384,
        norm_eps: 1e-5,
        axes_dims_rope: [8, 12, 12],
        rope_theta: 1000.0,
        timestep_embed_dim: 64,
        num_text_layers: 3,
        num_layerwise_text_blocks: 2,
        num_refiner_text_blocks: 2,
        text_hidden_dim: 64,
        text_intermediate_size: 256,
        text_num_attention_heads: 2,
        text_num_kv_heads: 2,
    }
}

/// Full `SingleStreamDiT` forward: img patch-embed, the custom timestep embedding + shared modulation,
/// text fusion + `txt_in`, the joint single-stream stack under 3-axis RoPE, the final layer, and
/// unpatchify — end to end vs the reference velocity.
#[test]
fn dit_matches_reference() {
    let w = load("dit_golden.safetensors");
    let cfg = tiny_dit_config();
    cfg.validate().unwrap();
    let dit = Krea2Transformer::from_weights(&w, &cfg).unwrap();
    let velocity = dit
        .forward(
            w.require("in.latent").unwrap(),
            w.require("in.timestep").unwrap(),
            w.require("in.context").unwrap(),
            None,
        )
        .unwrap();
    let want = w.require("out.velocity").unwrap();
    assert_eq!(velocity.shape(), want.shape(), "velocity shape");
    let c = cosine(&velocity, want);
    println!(
        "full-DiT parity: cosine={c:.7} max_abs={:e}",
        max_abs_diff(&velocity, want)
    );
    assert!(c > 0.999, "full-DiT cosine {c:.7} <= 0.999");
    assert!(
        all_close(&velocity, want, 2e-2, 2e-2, false)
            .unwrap()
            .item::<bool>(),
        "full-DiT velocity diverged beyond 2e-2 (cosine {c:.7})"
    );
}
