//! sc-8978 (epic 8236): numeric parity of the FLUX.2-dev **Fun-Controlnet-Union** VACE control branch
//! vs the **authoritative VideoX-Fun reference** (`Flux2ControlTransformer2DModel`), and a byte-confirm
//! of the packed 260-ch control-context fill vs the fork's `pipeline_flux2_control`.
//!
//! The FLUX.2 sibling of the Qwen sc-8335 harness (mlx-gen #628). Closes the two gaps
//! `control_parity.rs` (sc-2292) leaves open — that suite proves only the *mechanism*
//! (scale-0 byte-parity + scale>0 injection) on random control weights, with no numeric golden vs
//! the upstream and no byte-confirm of the packed context:
//!   * **GAP 1** — `Flux2ControlBranch::forward_control` reproduces the reference per-block hints.
//!   * **GAP 2** — `pipeline::fun_control_context_from_latents` reproduces the reference packed
//!     context (`_pack_latents(control) | zero mask | zero inpaint`, the FLUX.2 pack-then-concat order).
//!
//! Fixture `tests/fixtures/flux2_fun_control.safetensors` ← `tools/dump_flux2_fun_control_golden.py`
//! (a tiny synthetic control branch — 2 heads × 8, 3 control blocks = control_layers [0,2,4],
//! `before/after_proj` perturbed off zero-init so the control path is active — run through a minimal,
//! faithful copy of the upstream torch block/attention/FeedForward/RoPE + `forward_control`). Committed
//! + CI-runnable (no weights/torch/network).
//!
//! This is a *cross-framework* golden (torch-fp32 → MLX): tolerance 1e-2 covers Metal's
//! reduced-precision matmul across the block forwards (matching the Qwen sibling `fun_control_parity.rs`
//! and the base FLUX.2 transformer parity). The context byte-confirm is a pure gather → **exact**.

use mlx_gen::weights::Weights;
use mlx_gen_flux2::{pipeline, Flux2Config, Flux2ControlBranch};
use mlx_rs::ops::all_close;
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/flux2_fun_control.safetensors"
);

/// The tiny fixture geometry — mirrors the constants in `tools/dump_flux2_fun_control_golden.py`.
/// `num_double_layers = 6` → `control_layer_places() = (0..6).step_by(2) = [0, 2, 4]` → 3 control
/// blocks. GAP 2: `IN_CH = LAT_C·4 = 16` packed control channels, mask = `IN_CH/LAT_C = 4`.
const HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const IN_CH: i32 = 16;
const LAT_C: i32 = 4;

/// The tiny config the fixture was dumped with. Only `num_heads` / `head_dim` /
/// `control_layer_places()` are read by `Flux2ControlBranch`; the rest mirror `control_parity.rs`.
fn small_cfg() -> Flux2Config {
    Flux2Config {
        num_double_layers: 6,
        num_single_layers: 1,
        num_heads: HEADS,
        head_dim: HEAD_DIM,
        in_channels: 16,
        out_channels: 16,
        joint_attention_dim: 48,
        mlp_ratio: 3.0,
        timestep_channels: 16,
        axes_dim: [2, 2, 2, 2],
        rope_theta: 2000.0,
        te_hidden_size: 16,
        te_intermediate_size: 48,
        te_out_layers: [0, 1, 2],
        max_sequence_length: 512,
        num_latent_channels: 4,
        vae_scale_factor: 8,
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

fn owned(w: &Weights, key: &str) -> Array {
    w.require(key).unwrap().clone()
}

/// Reconstruct a double-block modulation `[(shift, scale, gate); 2]` from the fixture arrays — the
/// shared FLUX.2 base double-stream modulation the control blocks reuse (passed straight into
/// `forward_control`, so no modulation weight layer is needed).
fn mod_pair(w: &Weights, stream: &str) -> [(Array, Array, Array); 2] {
    [
        (
            owned(w, &format!("in.{stream}_mod_shift_0")),
            owned(w, &format!("in.{stream}_mod_scale_0")),
            owned(w, &format!("in.{stream}_mod_gate_0")),
        ),
        (
            owned(w, &format!("in.{stream}_mod_shift_1")),
            owned(w, &format!("in.{stream}_mod_scale_1")),
            owned(w, &format!("in.{stream}_mod_gate_1")),
        ),
    ]
}

/// GAP 1: the VACE branch `forward_control` reproduces the upstream reference hints. The fixture
/// carries the control-branch weights under the checkpoint key names, including the diffusers
/// `attn.to_out.0` Sequential — so this drives them through the same `attn.to_out.0` → `attn.to_out`
/// alias the production `load_control_transformer_dev` applies before `from_weights`.
#[test]
fn forward_control_hints_match_videox_reference() {
    let cfg = small_cfg();
    let mut w = Weights::from_file(FIXTURE).unwrap();
    // Mirror the loader: the checkpoint ships `attn.to_out` as a `[Linear, Dropout]` Sequential
    // (`attn.to_out.0.weight`); the shared `DoubleBlock` loader reads `attn.to_out.weight`.
    for i in 0..cfg.control_layer_places().len() {
        w.alias(
            &format!("control_transformer_blocks.{i}.attn.to_out.0.weight"),
            &format!("control_transformer_blocks.{i}.attn.to_out.weight"),
        );
    }
    let branch = Flux2ControlBranch::from_weights(&w, "", &cfg).unwrap();
    assert_eq!(branch.num_hints(), 3, "control_layers [0,2,4] → 3 hints");

    let img_mod = mod_pair(&w, "img");
    let txt_mod = mod_pair(&w, "txt");
    let hints = branch
        .forward_control(
            w.require("in.img_embed").unwrap(),
            w.require("in.txt_embed").unwrap(),
            w.require("in.control_context").unwrap(),
            &img_mod,
            &txt_mod,
            w.require("in.cos").unwrap(),
            w.require("in.sin").unwrap(),
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
        "✓ FLUX.2 VACE control hints match VideoX-Fun: 3 × {:?}  worst max|Δ| {worst:.3e}",
        hints[0].shape()
    );
}

/// GAP 2: the production packed control context byte-matches the fork's
/// `pipeline_flux2_control` (`_pack_latents(control) | zero mask | zero inpaint`). Pure gather (no
/// arithmetic) → exact, so any channel-order/fill/pack drift shows as a non-zero diff.
#[test]
fn control_context_fill_byte_matches_reference_prepare() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let control_lat = w.require("ctx.control_lat").unwrap(); // [1, IN_CH, LH, LW]
    let packed = pipeline::fun_control_context_from_latents(control_lat, IN_CH, LAT_C).unwrap();

    let want = w.require("ctx.pack_ref").unwrap(); // [1, LH·LW, CONTROL_IN]
    assert_eq!(packed.shape(), want.shape(), "packed context shape");
    let d = max_abs_diff(&packed, want);
    assert_eq!(
        d, 0.0,
        "packed control-context fill diverged from upstream pipeline_flux2_control (max|Δ| {d:.3e})"
    );
    println!(
        "✓ FLUX.2 packed control context byte-matches upstream pipeline_flux2_control: {:?}",
        packed.shape()
    );
}
