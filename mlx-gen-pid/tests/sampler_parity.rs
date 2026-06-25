//! sc-7843 component 3: the 4-step SDE distill sampler matches the torch `_student_sample_loop`.
//! Fixture = `tools/dump_pid_sampler.py` (tiny PidNet, faithful inline reproduction of the loop,
//! capturing the initial noise + each per-step ε + the clamped output). `Sampler::run` is
//! deterministic given (noise, ε), so this gates the step math bit-for-bit against torch (velocity→x0,
//! the SDE renoise `(1−t_next)·x0 + t_next·ε`, and the final clamp) — not the production RNG path.

use mlx_gen::weights::Weights;
use mlx_gen_pid::{PidConfig, PidNet, RopeMode, Sampler, SamplerConfig};
use mlx_rs::ops::{abs, max, subtract};
use mlx_rs::Array;

fn max_abs(a: &Array) -> f32 {
    max(abs(a).unwrap(), None).unwrap().item::<f32>()
}
fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    max_abs(&subtract(a, b).unwrap())
}

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

#[test]
fn sampler_loop_matches() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let w = Weights::from_file(format!("{dir}/sampler_tiny.safetensors")).unwrap();

    let net = PidNet::from_weights(&w, "", &tiny_cfg()).unwrap();
    let sampler = Sampler::new(&SamplerConfig::distill_4step());
    assert_eq!(sampler.steps(), 4);
    assert_eq!(sampler.num_eps(), 3);

    let noise = w.require("__io__.noise").unwrap().clone();
    let caption = w.require("__io__.caption").unwrap().clone();
    let lq_latent = w.require("__io__.lq_latent").unwrap().clone();
    let sigma = w.require("__io__.sigma").unwrap().clone();
    let eps: Vec<Array> = (0..3)
        .map(|i| w.require(&format!("__io__.eps_{i}")).unwrap().clone())
        .collect();
    let golden = w.require("__io__.output").unwrap().clone();

    let got = sampler
        .run(&net, &noise, &eps, &caption, &lq_latent, &sigma)
        .unwrap();
    assert_eq!(got.shape(), golden.shape(), "output shape");

    let d = max_abs_diff(&got, &golden);
    let scale = max_abs(&golden);
    let rel = d / scale;
    eprintln!("sampler loop: max|Δ|={d:.3e} scale={scale:.3e} peak-rel={rel:.3e}");
    assert!(
        rel < 2e-2,
        "sampler output peak-rel={rel} (max|Δ|={d}) — divergence in velocity→x0 / SDE renoise / clamp"
    );
}
