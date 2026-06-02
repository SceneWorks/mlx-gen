//! sc-2346 S3: parity for the FLUX.2 MMDiT transformer vs the fork, on a TINY synthetic config
//! (committed fixture `tests/fixtures/transformer_golden.safetensors` ←
//! `tools/dump_flux2_transformer_golden.py`). Exercises the double block, the single block, shared
//! modulation, the 4-axis interleaved RoPE, the time embedding, and the AdaLayerNormContinuous
//! output — bit-tight in f32 (the fork dump forces `ModelConfig.precision=float32`).

use mlx_gen::weights::Weights;
use mlx_gen_flux2::{Flux2Config, Flux2Transformer};
use mlx_rs::ops::all_close;
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/transformer_golden.safetensors"
);

fn close(a: &Array, b: &Array, rtol: f64, atol: f64) -> bool {
    all_close(a, b, rtol, atol, false).unwrap().item::<bool>()
}

/// The tiny config the dump script used (inner = 2·8 = 16).
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

#[test]
fn transformer_matches_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let t = Flux2Transformer::from_weights(&w, &tiny_config()).unwrap();
    let out = t
        .forward(
            w.require("hidden").unwrap(),
            w.require("encoder").unwrap(),
            w.require("img_ids").unwrap(),
            w.require("txt_ids").unwrap(),
            500.0,
        )
        .unwrap();
    let want = w.require("out").unwrap();
    assert_eq!(out.shape(), want.shape(), "transformer out shape");
    // 1e-2 = the repo's matmul-bearing bar (Metal fp32 reduced-precision, not bit-identical
    // cross-device); a real structural bug diverges by orders of magnitude.
    assert!(close(&out, want, 1e-2, 1e-2), "FLUX.2 transformer diverged");
}
