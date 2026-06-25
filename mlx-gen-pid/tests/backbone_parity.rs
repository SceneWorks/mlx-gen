//! sc-7843: the PixDiT_T2I backbone forward matches the torch reference. Fixture =
//! `tools/dump_pid_backbone.py` (tiny non-square synthetic config, f32, committed).
//!
//! Strategy: gate the **pure host positional math** (2-D NTK image RoPE, 1-D text RoPE, pixel sin/cos
//! pos) tightly — they are sin/cos of host-deterministic angles, so a wrong (x,y) axis order / NTK
//! scaling / meshgrid convention shows up as O(0.1+), not the ~1e-6 transcendental floor. Then gate
//! the **full forward** on peak-relative error against the documented ~2.4e-3 mlx-Metal-f32 matmul
//! floor (a real structural bug — wrong reshape, fold, attention join — is orders of magnitude above
//! it). The fixture's non-square grid (H=8, W=12) + NTK ref grid 8×8 ≠ sampled 4×6 exercise the
//! axis order and per-axis theta scaling that a port most often gets wrong.

use mlx_gen::weights::Weights;
use mlx_gen_pid::backbone::{image_rope_table, sincos_2d_pos, text_rope_table};
use mlx_gen_pid::{PidConfig, PixDiT, RopeMode};
use mlx_rs::ops::{abs, max, split, subtract};
use mlx_rs::Array;

fn max_abs(a: &Array) -> f32 {
    max(abs(a).unwrap(), None).unwrap().item::<f32>()
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    max_abs(&subtract(a, b).unwrap())
}

/// The tiny config baked into `dump_pid_backbone.py`.
fn tiny_cfg() -> PidConfig {
    PidConfig {
        in_channels: 3,
        num_groups: 2,
        hidden_size: 32,
        pixel_hidden_size: 8,
        pixel_attn_hidden_size: 16,
        pixel_num_groups: 2,
        patch_depth: 2,
        pixel_depth: 2,
        patch_size: 2,
        txt_embed_dim: 12,
        txt_max_length: 5,
        use_text_rope: true,
        text_rope_theta: 10000.0,
        rope_mode: RopeMode::NtkAware,
        rope_ref_h: 16,
        rope_ref_w: 16,
        // LQ fields are irrelevant to the base backbone but required to build the struct.
        lq_in_channels: 0,
        lq_latent_channels: 16,
        lq_hidden_dim: 512,
        lq_num_res_blocks: 4,
        lq_interval: 2,
        sr_scale: 4,
        latent_spatial_down_factor: 8,
    }
}

/// `[N, half, 2]` reference table → `(cos[N,half], sin[N,half])`.
fn split_cos_sin(t: &Array) -> (Array, Array) {
    let sh = t.shape();
    let (n, half) = (sh[0], sh[1]);
    let p = split(t, 2, 2).unwrap();
    (
        p[0].reshape(&[n, half]).unwrap(),
        p[1].reshape(&[n, half]).unwrap(),
    )
}

fn fixture() -> Weights {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    Weights::from_file(format!("{dir}/pixdit_tiny.safetensors")).unwrap()
}

#[test]
fn image_rope_2d_ntk_matches() {
    let w = fixture();
    // head_dim = 16, grid 4×6 (Hs×Ws), ref grid 8×8, theta 10000, scale 16.
    let (cos, sin) = image_rope_table(16, 4, 6, 8, 8, 10000.0, 16.0);
    let (gc, gs) = split_cos_sin(w.require("__io__.rope_img").unwrap());
    assert_eq!(cos.shape(), gc.shape(), "rope_img cos shape");
    let dc = max_abs_diff(&cos, &gc);
    let ds = max_abs_diff(&sin, &gs);
    assert!(
        dc < 1e-4 && ds < 1e-4,
        "image rope max|Δ| cos={dc} sin={ds}"
    );
}

#[test]
fn text_rope_1d_matches() {
    let w = fixture();
    let (cos, sin) = text_rope_table(16, 5, 10000.0);
    let (gc, gs) = split_cos_sin(w.require("__io__.rope_txt").unwrap());
    let dc = max_abs_diff(&cos, &gc);
    let ds = max_abs_diff(&sin, &gs);
    assert!(dc < 1e-4 && ds < 1e-4, "text rope max|Δ| cos={dc} sin={ds}");
}

#[test]
fn pixel_sincos_pos_matches() {
    let w = fixture();
    let pos = sincos_2d_pos(8, 8, 12); // embed_dim=8, H=8, W=12
    let golden = w.require("__io__.pixel_pos").unwrap();
    assert_eq!(pos.shape(), golden.shape(), "pixel_pos shape");
    let d = max_abs_diff(&pos, golden);
    assert!(d < 1e-4, "pixel sin/cos pos max|Δ| = {d}");
}

#[test]
fn backbone_forward_matches() {
    let w = fixture();
    let x = w.require("__io__.x").unwrap().clone();
    let t = w.require("__io__.t").unwrap().clone();
    let y = w.require("__io__.y").unwrap().clone();
    let golden = w.require("__io__.output").unwrap().clone();

    let model = PixDiT::from_weights(&w, "", &tiny_cfg()).unwrap();
    let got = model.forward(&x, &t, &y).unwrap();
    assert_eq!(got.shape(), golden.shape(), "output shape");

    let d = max_abs_diff(&got, &golden);
    let scale = max_abs(&golden);
    let rel = d / scale;
    eprintln!("backbone forward: max|Δ|={d:.3e} scale={scale:.3e} peak-rel={rel:.3e}");
    assert!(
        rel < 2e-2,
        "backbone output peak-rel = {rel} (max|Δ|={d}, scale={scale}) — structural divergence \
         (a correct port sits near the ~2.4e-3 mlx-Metal f32 matmul floor)"
    );
}
