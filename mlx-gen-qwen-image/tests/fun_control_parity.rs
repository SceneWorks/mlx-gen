//! sc-8335 (epic 8236): numeric parity of the Qwen-Image **2512-Fun-Controlnet-Union** VACE control
//! branch vs the **authoritative VideoX-Fun reference** (`QwenImageControlTransformer2DModel`), and a
//! byte-confirm of the 132-ch control-context fill vs the fork's `pipeline_qwenimage_control`.
//!
//! Closes the two gaps the sc-8267 adversarial review flagged (the `#[ignore]` real-weight smoke in
//! `control_real_weights.rs` proves loader/injection/pose-effect on random inputs, but had NO numeric
//! golden vs the upstream, and the 33-ch fill was only checked vs the Z-Image sibling):
//!   * **GAP 1** — `QwenFunControlBranch::forward_control` reproduces the reference per-block hints.
//!   * **GAP 2** — `pipeline::fun_control_context_from_latents` reproduces the reference
//!     `_pack_latents([control_latents(16) | mask(1) | inpaint(16)])` (channel order/fill + 2×2 pack).
//!
//! Fixture `tests/fixtures/qwen_fun_control.safetensors` ← `tools/dump_qwen_fun_control_golden.py`
//! (a tiny synthetic control branch — 2 heads × 8, 3 control blocks, `before/after_proj` perturbed
//! off zero-init so the control path is active — run through a minimal, faithful copy of the upstream
//! torch block/attention/RoPE + `forward_control`). Committed + CI-runnable (no weights/torch/network).
//!
//! This is a *cross-framework* golden (torch-fp32 → MLX): tolerance 1e-2 covers Metal's
//! reduced-precision matmul across the block forwards (matching the Z-Image sibling
//! `z_control_transformer.rs`). The context byte-confirm is a pure gather → **exact**.

use mlx_gen::weights::Weights;
use mlx_gen_qwen_image::loader::remap_transformer_keys;
use mlx_gen_qwen_image::pipeline;
use mlx_gen_qwen_image::{QwenFunControlBranch, QwenFunControlConfig};
use mlx_rs::ops::all_close;
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/qwen_fun_control.safetensors"
);

/// The tiny fixture geometry — mirrors the constants in `tools/dump_qwen_fun_control_golden.py`.
const HEADS: i32 = 2;
const HEAD_DIM: i32 = 8;
const CONTROL_IN_DIM: i32 = 132;
/// Control-latent grid packed by GAP 2: latent spatial is `2·LH × 2·LW` (H/8 × W/8) → `width`/
/// `height` fed to the packer are `16·LW`/`16·LH`.
const LH: u32 = 2;
const LW: u32 = 3;

fn small_cfg() -> QwenFunControlConfig {
    QwenFunControlConfig {
        control_layers: vec![0, 1, 2],
        control_in_dim: CONTROL_IN_DIM,
        num_heads: HEADS,
        head_dim: HEAD_DIM,
    }
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    a.as_slice::<f32>()
        .iter()
        .zip(b.as_slice::<f32>())
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()))
}

fn max_abs(a: &Array) -> f32 {
    let n = a.shape().iter().product::<i32>();
    a.reshape(&[n])
        .unwrap()
        .as_slice::<f32>()
        .iter()
        .fold(0f32, |m, &x| m.max(x.abs()))
}

/// GAP 1: the VACE branch `forward_control` reproduces the upstream reference hints. The fixture
/// weights carry the checkpoint (diffusers) key names, so this drives them through the same
/// `remap_transformer_keys` the production `load_controlnet` uses.
#[test]
fn forward_control_hints_match_videox_reference() {
    let mut w = Weights::from_file(FIXTURE).unwrap();
    remap_transformer_keys(&mut w);
    let branch = QwenFunControlBranch::from_weights(&w, "", &small_cfg()).unwrap();
    assert_eq!(branch.num_hints(), 3, "3 control layers → 3 hints");

    let hints = branch
        .forward_control(
            w.require("in.img_embed").unwrap(),
            w.require("in.encoder_embed").unwrap(),
            w.require("in.control_context").unwrap(),
            w.require("in.temb").unwrap(),
            w.require("in.img_cos").unwrap(),
            w.require("in.img_sin").unwrap(),
            w.require("in.txt_cos").unwrap(),
            w.require("in.txt_sin").unwrap(),
            None,
            None,
        )
        .unwrap();
    assert_eq!(hints.len(), 3);

    let mut worst = 0f32;
    for (i, hint) in hints.iter().enumerate() {
        let want = w.require(&format!("out.hint_{i}")).unwrap();
        assert_eq!(hint.shape(), want.shape(), "hint {i} shape");
        // Non-degeneracy guard: a zero-init (bugged) branch would emit ~0 hints and pass a loose
        // tolerance vacuously — the reference hints are O(1), so require the MLX hint be too.
        assert!(
            max_abs(hint) > 1e-2,
            "hint {i} is ~zero (max|hint| {:.3e}); the control branch looks inert",
            max_abs(hint)
        );
        let d = max_abs_diff(hint, want);
        worst = worst.max(d);
        assert!(
            all_close(hint, want, 1e-2, 1e-2, false)
                .unwrap()
                .item::<bool>(),
            "hint {i} diverged from the VideoX-Fun reference (max|Δ| {d:.3e})"
        );
    }
    println!(
        "✓ VACE control hints match VideoX-Fun: 3 × {:?}  worst max|Δ| {worst:.3e}",
        hints[0].shape()
    );
}

/// GAP 2: the production 132-ch control-context fill + 2×2 pack byte-matches the fork's
/// `pipeline_qwenimage_control._pack_latents([control_latents | mask | inpaint])`. Pure gather (no
/// arithmetic) → exact, so any channel-order/fill/pack drift shows as a non-zero diff.
#[test]
fn control_context_fill_byte_matches_reference_prepare() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let control_latents = w.require("ctx.control_latents").unwrap(); // [1, 16, 2·LH, 2·LW]
                                                                     // width/height the packer expects: latent spatial is H/8 × W/8 = 2·LH × 2·LW.
    let (width, height) = (16 * LW, 16 * LH);
    let packed =
        pipeline::fun_control_context_from_latents(control_latents, width, height).unwrap();

    let want = w.require("ctx.pack_ref").unwrap(); // [1, LH·LW, 132]
    assert_eq!(packed.shape(), want.shape(), "packed context shape");
    let d = max_abs_diff(&packed, want);
    assert_eq!(
        d, 0.0,
        "132-ch control-context fill/pack diverged from upstream _pack_latents (max|Δ| {d:.3e})"
    );
    println!(
        "✓ 132-ch control context byte-matches upstream _pack_latents: {:?}",
        packed.shape()
    );
}
