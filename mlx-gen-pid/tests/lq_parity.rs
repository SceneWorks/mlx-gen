//! sc-7843 component 2: the sigma-aware LQ adapter + gate-injected `PidNet` forward match the torch
//! reference. Fixture = `tools/dump_pid_lq.py` (tiny latent-only config, `lq_interval=2` over a
//! 4-block patch stream → 2 output heads + 2 gates at blocks 0 and 2; `z_to_patch_ratio=2` nearest
//! upsample). Gates: (1) the LQ projection feature sets (conv stack + heads), (2) an isolated
//! sigma-gate I/O (exp(log_alpha)·σ broadcast + sigmoid), (3) the full gate-injected forward.

use mlx_gen::weights::Weights;
use mlx_gen_pid::{PidConfig, PidNet, RopeMode};
use mlx_rs::ops::{abs, max, subtract};
use mlx_rs::Array;

fn max_abs(a: &Array) -> f32 {
    max(abs(a).unwrap(), None).unwrap().item::<f32>()
}
fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    max_abs(&subtract(a, b).unwrap())
}

/// The tiny config baked into `dump_pid_lq.py`.
fn tiny_cfg() -> PidConfig {
    PidConfig {
        in_channels: 3,
        num_groups: 2,
        hidden_size: 32,
        pixel_hidden_size: 8,
        pixel_attn_hidden_size: 16,
        pixel_num_groups: 2,
        patch_depth: 4,
        pixel_depth: 2,
        patch_size: 2,
        txt_embed_dim: 12,
        txt_max_length: 5,
        use_text_rope: true,
        text_rope_theta: 10000.0,
        rope_mode: RopeMode::NtkAware,
        rope_ref_h: 16,
        rope_ref_w: 16,
        lq_in_channels: 0,
        lq_latent_channels: 4,
        lq_hidden_dim: 8,
        lq_num_res_blocks: 2,
        lq_interval: 2,
        sr_scale: 2,
        latent_spatial_down_factor: 2,
    }
}

fn fixture() -> Weights {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    Weights::from_file(format!("{dir}/pidnet_tiny.safetensors")).unwrap()
}

#[test]
fn lq_projection_features_match() {
    let w = fixture();
    let lq_latent = w.require("__io__.lq_latent").unwrap().clone();
    let net = PidNet::from_weights(&w, "", &tiny_cfg()).unwrap();

    // patch grid pH=4, pW=6 (H8/W12 over patch 2)
    let feats = net.lq().forward(&lq_latent, 4, 6).unwrap();
    assert_eq!(feats.len(), 2, "num lq output heads");
    for (i, f) in feats.iter().enumerate() {
        let golden = w.require(&format!("__io__.lq_feat_{i}")).unwrap();
        assert_eq!(f.shape(), golden.shape(), "lq_feat_{i} shape");
        let d = max_abs_diff(f, golden);
        let rel = d / max_abs(golden);
        eprintln!("lq_feat_{i}: max|Δ|={d:.3e} peak-rel={rel:.3e}");
        assert!(rel < 2e-2, "lq_feat_{i} peak-rel={rel} (max|Δ|={d})");
    }
}

#[test]
fn sigma_gate_matches() {
    let w = fixture();
    let xg = w.require("__io__.gate_xg").unwrap().clone();
    let lqg = w.require("__io__.gate_lqg").unwrap().clone();
    let sigma = w.require("__io__.sigma").unwrap().clone();
    let golden = w.require("__io__.gate_out").unwrap();

    let net = PidNet::from_weights(&w, "", &tiny_cfg()).unwrap();
    let got = net.lq().gate(0, &xg, &lqg, &sigma).unwrap();
    assert_eq!(got.shape(), golden.shape(), "gate_out shape");
    let d = max_abs_diff(&got, golden);
    let rel = d / max_abs(golden);
    eprintln!("sigma gate: max|Δ|={d:.3e} peak-rel={rel:.3e}");
    // pure elementwise (one Linear + sigmoid) — gate tight.
    assert!(rel < 5e-3, "sigma gate peak-rel={rel} (max|Δ|={d})");
}

#[test]
fn pidnet_forward_matches() {
    let w = fixture();
    let x = w.require("__io__.x").unwrap().clone();
    let t = w.require("__io__.t").unwrap().clone();
    let y = w.require("__io__.y").unwrap().clone();
    let lq_latent = w.require("__io__.lq_latent").unwrap().clone();
    let sigma = w.require("__io__.sigma").unwrap().clone();
    let golden = w.require("__io__.output").unwrap().clone();

    let net = PidNet::from_weights(&w, "", &tiny_cfg()).unwrap();
    let got = net.forward(&x, &t, &y, &lq_latent, &sigma).unwrap();
    assert_eq!(got.shape(), golden.shape(), "output shape");
    let d = max_abs_diff(&got, &golden);
    let rel = d / max_abs(&golden);
    eprintln!(
        "pidnet forward: max|Δ|={d:.3e} scale={:.3e} peak-rel={rel:.3e}",
        max_abs(&golden)
    );
    assert!(
        rel < 2e-2,
        "PidNet output peak-rel={rel} (max|Δ|={d}) — structural divergence in the gate-injected path"
    );
}
