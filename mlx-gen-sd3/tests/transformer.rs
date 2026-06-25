//! sc-7862 (SD3.5 E3): coverage for the MMDiT-Large forward.
//!
//! Two tiers, mirroring the crate convention (CLAUDE.md "Real-weight tests vs default tests"):
//!
//!   * **Default (committed, no weights):** a TINY synthetic `SD3Transformer2DModel` — same topology
//!     as SD3.5-Large (joint double-stream blocks, learned pos_embed NO RoPE, qk-RMSNorm both
//!     streams, adaLN modulation, `context_pre_only` final block, GELU FFN) but tiny widths — built
//!     from random weights for every expected diffusers tensor. Proves the forward assembles from
//!     exactly the E1-converter key set and runs end-to-end at the right latent shape, finite +
//!     statistically sane, for a square AND a non-square (centered-crop pos_embed) grid.
//!
//!   * **`#[ignore]` real-weight forward** (`SD3_TRANSFORMER=/path/to/transformer`): loads the REAL
//!     converted/quantized Large transformer and runs a 256²-grid forward (16×16 patch grid → 256
//!     image tokens + 333 text tokens), asserting the predicted-latent shape `[B,16,32,32]`, finite,
//!     and a sane statistical range. Needs the multi-GB licensed weights + Metal.
//!
//!   * **`#[ignore]` numeric A/B** (`SD3_REF_DUMP=/path/to/ref.safetensors` + `SD3_TRANSFORMER=...`):
//!     the real parity gate vs diffusers `SD3Transformer2DModel`. Consumes a reference dump
//!     (`latent`, `context`, `pooled`, `timestep`, `out`) produced by a torch/diffusers script and
//!     asserts cosine ≥ 0.99 on the predicted noise. Gated because no torch/diffusers env is present
//!     in this workspace (see the PR / FOLLOW_UPS) — run it once a dump exists.

use mlx_gen::weights::Weights;
use mlx_gen_sd3::config::Sd3Arch;
use mlx_gen_sd3::convert::expected_transformer_tensors;
use mlx_gen_sd3::transformer::Sd3Transformer;
use mlx_rs::ops::multiply;
use mlx_rs::transforms::eval;
use mlx_rs::{random, Array};

/// A small-but-complete SD3.5-like MMDiT arch: a few joint blocks, tiny widths, but every structural
/// feature SD3.5-Large has (16-ch in/out, patch 2, learned pos_embed, qk-RMSNorm, context_pre_only
/// last block). `num_heads * head_dim` is the hidden size; keep `time_proj_dim` even.
fn tiny_arch() -> Sd3Arch {
    Sd3Arch {
        num_layers: 3,
        head_dim: 8,
        num_heads: 4, // hidden = 32
        patch_size: 2,
        in_channels: 16,
        out_channels: 16,
        joint_attention_dim: 24, // the E2 context feature width
        pooled_projection_dim: 20,
        caption_projection_dim: 32, // == hidden
        pos_embed_max_size: 12,     // table spans up to a 12×12 patch grid
        time_proj_dim: 16,
        dual_attention_layers: 0, // plain MMDiT (Large topology), no MMDiT-X dual-attention blocks
    }
}

/// Build random weights for every expected diffusers transformer tensor of `arch` (NCHW conv weight,
/// as on disk). Small magnitude so the deep f32 stack stays numerically tame and finite.
fn synthetic_transformer(arch: &Sd3Arch) -> Weights {
    let key = random::key(7).unwrap();
    let mut w = Weights::empty();
    let scale = Array::from_slice(&[0.02f32], &[1]);
    for e in expected_transformer_tensors(arch) {
        let shape: Vec<i32> = e.shape.iter().map(|&d| d as i32).collect();
        let t = multiply(
            random::normal::<f32>(&shape, None, None, Some(&key)).unwrap(),
            &scale,
        )
        .unwrap();
        w.insert(e.key, t);
    }
    w
}

/// Run a tiny synthetic forward at a `(h, ww)` LATENT size (must be divisible by patch). Returns the
/// predicted-latent shape and whether it is finite.
fn run_tiny(arch: &Sd3Arch, h: i32, ww: i32, ctx_seq: i32) -> (Vec<i32>, bool, f32, f32) {
    let w = synthetic_transformer(arch);
    let model = Sd3Transformer::from_weights(&w, arch).unwrap();

    let b = 1;
    let key = random::key(11).unwrap();
    let latent =
        random::normal::<f32>(&[b, arch.in_channels as i32, h, ww], None, None, Some(&key))
            .unwrap();
    let context = random::normal::<f32>(
        &[b, ctx_seq, arch.joint_attention_dim as i32],
        None,
        None,
        Some(&key),
    )
    .unwrap();
    let pooled = random::normal::<f32>(
        &[b, arch.pooled_projection_dim as i32],
        None,
        None,
        Some(&key),
    )
    .unwrap();
    let timestep = Array::from_slice(&[500.0f32], &[b]);

    let out = model
        .forward(&latent, &context, &pooled, &timestep)
        .unwrap();
    eval([&out]).unwrap();

    let shape = out.shape().to_vec();
    let (finite, mean, std) = host_stats(&out);
    (shape, finite, mean, std)
}

/// Read an array to host and compute `(all_finite, mean, std)` in Rust — robust across the mlx-rs
/// reduction-method surface (no `mean(None)`/`std(None)` device-stream signature games).
fn host_stats(a: &Array) -> (bool, f32, f32) {
    let v: Vec<f32> = a
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let finite = v.iter().all(|x| x.is_finite());
    let n = v.len().max(1) as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    (finite, mean, var.sqrt())
}

#[test]
fn tiny_forward_square_grid_shape_and_finite() {
    let arch = tiny_arch();
    // 16×16 latent → 8×8 patch grid (64 image tokens). out = [1, 16, 16, 16].
    let (shape, finite, mean, std) = run_tiny(&arch, 16, 16, 7);
    assert_eq!(
        shape,
        vec![1, arch.out_channels as i32, 16, 16],
        "predicted latent matches the input spatial size, out_channels channels"
    );
    assert!(
        finite,
        "predicted latent is finite (mean={mean}, std={std})"
    );
    assert!(std > 0.0, "predicted latent is not degenerate (std={std})");
}

#[test]
fn tiny_forward_nonsquare_grid_exercises_centered_pos_embed_crop() {
    let arch = tiny_arch();
    // 8×16 latent → 4×8 patch grid: a non-square crop of the learned pos_embed table (top/left
    // offsets differ per axis), the path most likely to expose a pos_embed indexing bug.
    let (shape, finite, _mean, std) = run_tiny(&arch, 8, 16, 5);
    assert_eq!(shape, vec![1, arch.out_channels as i32, 8, 16]);
    assert!(finite, "non-square predicted latent is finite");
    assert!(std > 0.0);
}

/// Inference [`Sd3Transformer::forward`] must run **f32 activations regardless of the base weight
/// dtype** (sc-7883 review fix). SD3.5-large dense weights are bf16 on disk and the dense inference
/// path keeps them bf16 — but the activation dtype must NOT follow the weights, or dense inference
/// silently drops to bf16 activations (the regression this pins). The pre-PR validated path cast
/// activations to f32 unconditionally; the regression made `forward` follow `compute_dtype()` (= the
/// on-disk bf16 for a dense Large load), running bf16 activations.
///
/// We load a model with **bf16 base weights** (the on-disk-Large case) and assert:
///   1. `forward` (the inference entry) is bit-identical to the explicit-f32 seam
///      `forward_with(.., Float32)` — proving inference is f32-pinned, NOT weight-following; and
///   2. it DIFFERS from the explicit-bf16 seam `forward_with(.., Bfloat16)` by the bf16-rounding
///      magnitude — proving the body really would have changed had it followed the weight dtype (so
///      the pin is meaningful, not vacuous).
#[test]
fn inference_forward_is_f32_pinned_regardless_of_weight_dtype() {
    use mlx_rs::Dtype;

    let arch = tiny_arch();
    let w = synthetic_transformer(&arch);

    // Base weights at bf16 — mimics the SD3.5-large dense on-disk dtype, where the regression bit.
    let mut model = Sd3Transformer::from_weights(&w, &arch).unwrap();
    model.cast_weights(Dtype::Bfloat16).unwrap();

    let b = 1;
    let (h, ww, ctx_seq) = (16, 16, 7);
    let key = random::key(101).unwrap();
    let latent =
        random::normal::<f32>(&[b, arch.in_channels as i32, h, ww], None, None, Some(&key))
            .unwrap();
    let context = random::normal::<f32>(
        &[b, ctx_seq, arch.joint_attention_dim as i32],
        None,
        None,
        Some(&key),
    )
    .unwrap();
    let pooled = random::normal::<f32>(
        &[b, arch.pooled_projection_dim as i32],
        None,
        None,
        Some(&key),
    )
    .unwrap();
    let timestep = Array::from_slice(&[500.0f32], &[b]);

    let out_infer = model
        .forward(&latent, &context, &pooled, &timestep)
        .unwrap();
    let out_f32 = model
        .forward_with(&latent, &context, &pooled, &timestep, Dtype::Float32)
        .unwrap();
    let out_bf16 = model
        .forward_with(&latent, &context, &pooled, &timestep, Dtype::Bfloat16)
        .unwrap();
    eval([&out_infer, &out_f32, &out_bf16]).unwrap();

    // The velocity is always returned f32.
    assert_eq!(out_infer.dtype(), Dtype::Float32);

    let infer: Vec<f32> = out_infer.as_slice::<f32>().to_vec();
    let f32v: Vec<f32> = out_f32.as_slice::<f32>().to_vec();
    let bf16v: Vec<f32> = out_bf16.as_slice::<f32>().to_vec();
    assert_eq!(infer.len(), f32v.len());
    assert_eq!(infer.len(), bf16v.len());

    // (1) Inference is f32-pinned: bit-identical to the explicit-f32 seam, despite bf16 base weights.
    let d_f32 = infer
        .iter()
        .zip(&f32v)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        d_f32 == 0.0,
        "inference forward must be f32-pinned (== forward_with f32) regardless of base weight dtype \
         (max_abs={d_f32})"
    );

    // (2) The bf16-activation body really differs — so the pin is non-vacuous (this is what the
    //     regression would have silently run).
    let d_bf16 = infer
        .iter()
        .zip(&bf16v)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    // `> 0` (not a fixed magnitude): the bf16 seam casts activations to bf16, a genuinely distinct
    // computation, so it must differ from f32. The tiny synthetic model keeps the gap small, but any
    // non-zero gap proves the dtype parameter is live and the f32 pin is comparing real alternatives.
    assert!(
        d_bf16 > 0.0,
        "bf16-activation forward must differ from f32 (else the f32 pin is vacuous) (max_abs={d_bf16})"
    );
}

#[test]
fn pos_embed_grid_exceeding_max_size_errors() {
    let arch = tiny_arch(); // pos_embed_max_size = 12
    let w = synthetic_transformer(&arch);
    let model = Sd3Transformer::from_weights(&w, &arch).unwrap();
    // 28×28 latent → 14×14 patch grid > 12 → must error, not silently mis-index.
    let latent =
        random::normal::<f32>(&[1, arch.in_channels as i32, 28, 28], None, None, None).unwrap();
    let context =
        random::normal::<f32>(&[1, 4, arch.joint_attention_dim as i32], None, None, None).unwrap();
    let pooled =
        random::normal::<f32>(&[1, arch.pooled_projection_dim as i32], None, None, None).unwrap();
    let timestep = Array::from_slice(&[10.0f32], &[1]);
    assert!(
        model
            .forward(&latent, &context, &pooled, &timestep)
            .is_err(),
        "a patch grid larger than pos_embed_max_size must be rejected"
    );
}

#[test]
fn missing_tensor_surfaces_as_error_not_panic() {
    let arch = tiny_arch();
    let w = synthetic_transformer(&arch);
    // Drop a load-bearing tensor; construction must Err (Weights::require), never panic.
    let mut w2 = Weights::empty();
    for k in w.keys().filter(|k| *k != "proj_out.weight") {
        w2.insert(k.to_string(), w.get(k).unwrap().clone());
    }
    assert!(Sd3Transformer::from_weights(&w2, &arch).is_err());
}

// ----------------------------------------------------------------------------------------------
// Real-weight forward (shape / finite / stats). #[ignore]: needs licensed multi-GB weights + Metal.
//   SD3_TRANSFORMER=/path/to/stable-diffusion-3.5-large/transformer \
//     cargo test -p mlx-gen-sd3 --release --test transformer real_weight_forward -- --ignored --nocapture
// ----------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs SD3_TRANSFORMER=/path/to/transformer (licensed weights + Metal)"]
fn real_weight_forward_shape_finite_stats() {
    let dir = std::env::var("SD3_TRANSFORMER")
        .expect("set SD3_TRANSFORMER to the SD3.5-Large transformer/ dir");
    let arch = Sd3Arch::large();
    let model = Sd3Transformer::from_dir(std::path::Path::new(&dir), &arch)
        .expect("load real SD3.5-Large transformer");

    // 256² image → /8 VAE → 32×32 latent → /2 patch → 16×16 = 256 image tokens.
    let b = 1;
    let (hl, wl) = (32, 32);
    let ctx_seq = 333; // E2 context length
    let key = random::key(0).unwrap();
    let latent = random::normal::<f32>(
        &[b, arch.in_channels as i32, hl, wl],
        None,
        None,
        Some(&key),
    )
    .unwrap();
    let context = random::normal::<f32>(
        &[b, ctx_seq, arch.joint_attention_dim as i32],
        None,
        None,
        Some(&key),
    )
    .unwrap();
    let pooled = random::normal::<f32>(
        &[b, arch.pooled_projection_dim as i32],
        None,
        None,
        Some(&key),
    )
    .unwrap();
    let timestep = Array::from_slice(&[500.0f32], &[b]);

    let out = model
        .forward(&latent, &context, &pooled, &timestep)
        .unwrap();
    eval([&out]).unwrap();

    assert_eq!(
        out.shape(),
        &[b, arch.out_channels as i32, hl, wl],
        "real-weight predicted latent shape"
    );
    let (finite, mean, std) = host_stats(&out);
    println!(
        "[sd3 real-weight forward] shape={:?} mean={mean} std={std}",
        out.shape()
    );
    assert!(finite, "real-weight predicted latent is finite");
    // The velocity prediction over a random latent should be O(1)-ish, never collapsed or exploded.
    assert!(
        std > 1e-3 && std < 1e3,
        "real-weight predicted latent has a sane spread (std={std})"
    );
}

// ----------------------------------------------------------------------------------------------
// Numeric A/B vs diffusers SD3Transformer2DModel. #[ignore]: needs a reference dump from a
// torch/diffusers env (NOT present in this workspace). The dump is a single safetensors with
// `latent [B,16,H,W]`, `context [B,S,4096]`, `pooled [B,2048]`, `timestep [B]`, `out [B,16,H,W]`.
//   SD3_TRANSFORMER=/path/to/transformer SD3_REF_DUMP=/path/to/ref.safetensors \
//     cargo test -p mlx-gen-sd3 --release --test transformer numeric_parity -- --ignored --nocapture
// ----------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs SD3_REF_DUMP (diffusers reference dump) + SD3_TRANSFORMER — no torch env here"]
fn numeric_parity_vs_diffusers() {
    let dir =
        std::env::var("SD3_TRANSFORMER").expect("set SD3_TRANSFORMER to the transformer/ dir");
    let dump =
        std::env::var("SD3_REF_DUMP").expect("set SD3_REF_DUMP to the reference safetensors");
    let arch = Sd3Arch::large();
    let model = Sd3Transformer::from_dir(std::path::Path::new(&dir), &arch).unwrap();

    let w = Weights::from_file(&dump).expect("load reference dump");
    let latent = w.require("latent").unwrap().clone();
    let context = w.require("context").unwrap().clone();
    let pooled = w.require("pooled").unwrap().clone();
    let timestep = w.require("timestep").unwrap().clone();
    let reference = w.require("out").unwrap().clone();

    let out = model
        .forward(&latent, &context, &pooled, &timestep)
        .unwrap();
    eval([&out]).unwrap();
    assert_eq!(out.shape(), reference.shape(), "shape parity");

    let cos = cosine(&out, &reference);
    println!("[sd3 numeric parity] cosine={cos}");
    assert!(
        cos >= 0.99,
        "predicted noise cosine vs diffusers {cos} < 0.99"
    );
}

/// Flattened cosine similarity between two arrays (computed host-side).
fn cosine(a: &Array, b: &Array) -> f32 {
    let a: Vec<f32> = a
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let b: Vec<f32> = b
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-12)
}
