//! sc-7868 (SD3.5 M2): coverage for the **Medium MMDiT-X dual-attention** forward.
//!
//! Medium is structurally an SD3.5 MMDiT whose FIRST `dual_attention_layers` blocks
//! (`[0..=12]` on the real weights) carry a SECOND, image-stream-only self-attention `attn2`
//! alongside the joint attention, plus an EXTENDED 9-chunk `norm1` (`SD35AdaLayerNormZeroX`). The
//! remaining blocks are plain joint blocks; the last is `context_pre_only`. See
//! [`mlx_gen_sd3::transformer`] for the per-block forward (faithful mirror of diffusers
//! `JointTransformerBlock.forward(use_dual_attention=True)`).
//!
//! Tiers (mirroring the E3 `transformer.rs` test convention + CLAUDE.md "Real-weight vs default"):
//!
//!   * **Default (committed, no weights):** a TINY synthetic MMDiT-X — same topology as SD3.5-Medium
//!     (mix of dual-attention + plain joint blocks, learned pos_embed NO RoPE, qk-RMSNorm, 9-chunk
//!     `norm1` on dual blocks, `context_pre_only` final block) but tiny widths, built from random
//!     weights for every expected diffusers tensor. Proves the dual-attention forward assembles from
//!     exactly the M1-converter key set (incl. `attn2.*` + the 9·hidden `norm1`) and runs end-to-end
//!     at the right latent shape, finite + statistically sane, for square + non-square grids. A
//!     companion test proves a dual block is NOT a no-op vs the same block with `attn2` removed (so
//!     the `attn2` residual genuinely participates).
//!
//!   * **`#[ignore]` real-weight forward** (`SD3_MEDIUM_TRANSFORMER=/path/to/transformer`): loads the
//!     REAL converted/quantized Medium transformer and runs a 256²-grid forward (16×16 patch grid),
//!     asserting the predicted-latent shape `[B,16,32,32]`, finite, and a sane statistical range —
//!     exercising ALL 24 blocks (13 dual + 11 plain). Needs the multi-GB licensed weights + Metal.
//!
//!   * **`#[ignore]` numeric A/B** (`SD3_MEDIUM_REF_DUMP=…` + `SD3_MEDIUM_TRANSFORMER=…`): the real
//!     parity gate vs diffusers Medium `SD3Transformer2DModel`. Consumes a reference dump (`latent`,
//!     `context`, `pooled`, `timestep`, `out`) and asserts cosine ≥ 0.99. Gated because no
//!     torch/diffusers env is present in this workspace (see the PR / FOLLOW_UPS).

use mlx_gen::weights::Weights;
use mlx_gen_sd3::config::Sd3Arch;
use mlx_gen_sd3::convert::expected_transformer_tensors;
use mlx_gen_sd3::transformer::Sd3Transformer;
use mlx_rs::ops::multiply;
use mlx_rs::transforms::eval;
use mlx_rs::{random, Array};

/// A small-but-complete SD3.5-**Medium-like** MMDiT-X arch: a few blocks, the FIRST two of which are
/// dual-attention (carry `attn2` + the 9-chunk `norm1`), the rest plain. Tiny widths, but every
/// structural feature SD3.5-Medium has (16-ch in/out, patch 2, learned pos_embed, qk-RMSNorm,
/// dual-attention blocks, `context_pre_only` last block).
fn tiny_mmdit_x_arch() -> Sd3Arch {
    Sd3Arch {
        num_layers: 4,
        head_dim: 8,
        num_heads: 4, // hidden = 32
        patch_size: 2,
        in_channels: 16,
        out_channels: 16,
        joint_attention_dim: 24,
        pooled_projection_dim: 20,
        caption_projection_dim: 32, // == hidden
        pos_embed_max_size: 12,
        time_proj_dim: 16,
        dual_attention_layers: 2, // FIRST 2 blocks are MMDiT-X dual-attention; blocks 2,3 are plain
    }
}

/// Build random weights for every expected diffusers transformer tensor of `arch` (so the dual
/// blocks get their `attn2.*` + 9·hidden `norm1.linear`, and plain blocks get the 6·hidden `norm1`).
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

/// Run a tiny MMDiT-X forward at a `(h, ww)` LATENT size (must be divisible by patch).
fn run_tiny(w: &Weights, arch: &Sd3Arch, h: i32, ww: i32, ctx_seq: i32) -> Array {
    let model = Sd3Transformer::from_weights(w, arch).unwrap();
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
    out
}

#[test]
fn tiny_mmdit_x_forward_square_grid_shape_and_finite() {
    let arch = tiny_mmdit_x_arch();
    let w = synthetic_transformer(&arch);
    // 16×16 latent → 8×8 patch grid (64 image tokens). out = [1, 16, 16, 16].
    let out = run_tiny(&w, &arch, 16, 16, 7);
    assert_eq!(
        out.shape(),
        &[1, arch.out_channels as i32, 16, 16],
        "MMDiT-X predicted latent matches the input spatial size, out_channels channels"
    );
    let (finite, mean, std) = host_stats(&out);
    assert!(
        finite,
        "MMDiT-X predicted latent is finite (mean={mean}, std={std})"
    );
    assert!(
        std > 0.0,
        "MMDiT-X predicted latent is not degenerate (std={std})"
    );
}

#[test]
fn tiny_mmdit_x_forward_nonsquare_grid_exercises_centered_pos_embed_crop() {
    let arch = tiny_mmdit_x_arch();
    let w = synthetic_transformer(&arch);
    // 8×16 latent → 4×8 patch grid (asymmetric centered crop of the learned pos_embed).
    let out = run_tiny(&w, &arch, 8, 16, 5);
    assert_eq!(out.shape(), &[1, arch.out_channels as i32, 8, 16]);
    let (finite, _mean, std) = host_stats(&out);
    assert!(finite, "non-square MMDiT-X predicted latent is finite");
    assert!(std > 0.0);
}

/// The load-bearing M2 behavioral test: a dual-attention block's `attn2` residual must genuinely
/// participate. We build the SAME random weights but ZERO the `attn2.to_out.0` projection of every
/// dual block (so the `attn2` residual contributes exactly 0), and assert the output DIFFERS from the
/// full-weight forward. A regression that drops the `attn2` path entirely (treats a dual block as
/// plain) would produce the zeroed-`attn2` output for the full weights → this test would catch it.
#[test]
fn dual_attention_residual_changes_output() {
    let arch = tiny_mmdit_x_arch();
    let w_full = synthetic_transformer(&arch);

    // Build a second weight set identical except `attn2.to_out.0.{weight,bias}` zeroed on dual blocks.
    let mut w_no_attn2 = Weights::empty();
    for k in w_full.keys() {
        let t = w_full.get(k).unwrap().clone();
        let is_attn2_out = (0..arch.dual_attention_layers).any(|i| {
            k == format!("transformer_blocks.{i}.attn2.to_out.0.weight")
                || k == format!("transformer_blocks.{i}.attn2.to_out.0.bias")
        });
        if is_attn2_out {
            w_no_attn2.insert(k.to_string(), Array::zeros::<f32>(t.shape()).unwrap());
        } else {
            w_no_attn2.insert(k.to_string(), t);
        }
    }

    let out_full = run_tiny(&w_full, &arch, 16, 16, 7);
    let out_zeroed = run_tiny(&w_no_attn2, &arch, 16, 16, 7);

    let a: Vec<f32> = out_full.as_slice::<f32>().to_vec();
    let b: Vec<f32> = out_zeroed.as_slice::<f32>().to_vec();
    let max_abs_diff = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs_diff > 1e-5,
        "zeroing attn2.to_out on the dual blocks must change the output (max|Δ|={max_abs_diff}); \
         the attn2 residual is not being applied"
    );
}

/// A plain-MMDiT arch (`dual_attention_layers = 0`) must be byte-for-byte the Large topology: no
/// `attn2` consulted. Reuses the Large tiny arch shape but exercised through the M2 code path to
/// confirm the dual-attention extension did not regress the plain block.
#[test]
fn plain_blocks_still_run_when_no_dual_layers() {
    let mut arch = tiny_mmdit_x_arch();
    arch.dual_attention_layers = 0; // all plain → must not require any attn2.* / 9-chunk norm1
    let w = synthetic_transformer(&arch);
    let out = run_tiny(&w, &arch, 16, 16, 7);
    assert_eq!(out.shape(), &[1, arch.out_channels as i32, 16, 16]);
    let (finite, _m, std) = host_stats(&out);
    assert!(finite && std > 0.0);
}

#[test]
fn missing_attn2_tensor_surfaces_as_error_not_panic() {
    let arch = tiny_mmdit_x_arch();
    let w = synthetic_transformer(&arch);
    // Drop a load-bearing dual-block tensor; construction must Err (Weights::require), never panic.
    let mut w2 = Weights::empty();
    for k in w
        .keys()
        .filter(|k| *k != "transformer_blocks.0.attn2.to_q.weight")
    {
        w2.insert(k.to_string(), w.get(k).unwrap().clone());
    }
    assert!(Sd3Transformer::from_weights(&w2, &arch).is_err());
}

// ----------------------------------------------------------------------------------------------
// Real-weight Medium forward (shape / finite / stats; all 24 blocks). #[ignore]: licensed weights.
//   SD3_MEDIUM_TRANSFORMER=/path/to/stable-diffusion-3.5-medium/transformer \
//     cargo test -p mlx-gen-sd3 --release --test medium_transformer real_weight -- --ignored --nocapture
// ----------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs SD3_MEDIUM_TRANSFORMER=/path/to/transformer (licensed weights + Metal)"]
fn real_weight_medium_forward_shape_finite_stats() {
    let dir = std::env::var("SD3_MEDIUM_TRANSFORMER")
        .expect("set SD3_MEDIUM_TRANSFORMER to the SD3.5-Medium transformer/ dir");
    let arch = Sd3Arch::medium();
    assert_eq!(arch.num_layers, 24);
    assert_eq!(arch.dual_attention_layers, 13);
    let model = Sd3Transformer::from_dir(std::path::Path::new(&dir), &arch)
        .expect("load real SD3.5-Medium MMDiT-X transformer");

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
        "real-weight Medium predicted latent shape"
    );
    let (finite, mean, std) = host_stats(&out);
    println!(
        "[sd3 medium real-weight forward] shape={:?} mean={mean} std={std} (24 blocks: 13 dual + 11 plain)",
        out.shape()
    );
    assert!(finite, "real-weight Medium predicted latent is finite");
    assert!(
        std > 1e-3 && std < 1e3,
        "real-weight Medium predicted latent has a sane spread (std={std})"
    );
}

// ----------------------------------------------------------------------------------------------
// Numeric A/B vs diffusers Medium SD3Transformer2DModel. #[ignore]: needs a reference dump from a
// torch/diffusers env (NOT present in this workspace).
//   SD3_MEDIUM_TRANSFORMER=… SD3_MEDIUM_REF_DUMP=/path/to/ref.safetensors \
//     cargo test -p mlx-gen-sd3 --release --test medium_transformer numeric_parity -- --ignored --nocapture
// ----------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs SD3_MEDIUM_REF_DUMP (diffusers reference dump) + SD3_MEDIUM_TRANSFORMER — no torch env here"]
fn numeric_parity_vs_diffusers_medium() {
    let dir = std::env::var("SD3_MEDIUM_TRANSFORMER")
        .expect("set SD3_MEDIUM_TRANSFORMER to the transformer/ dir");
    let dump = std::env::var("SD3_MEDIUM_REF_DUMP")
        .expect("set SD3_MEDIUM_REF_DUMP to the reference safetensors");
    let arch = Sd3Arch::medium();
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
    println!("[sd3 medium numeric parity] cosine={cos}");
    assert!(
        cos >= 0.99,
        "predicted noise cosine vs diffusers {cos} < 0.99"
    );
}

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
