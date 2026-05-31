//! sc-2344: tiny end-to-end denoise-loop parity vs the fork.
//!
//! Fixture `tests/fixtures/denoise_loop.safetensors` ← `tools/dump_denoise_loop.py` (the tiny
//! ZImageTransformer config from the DiT parity test, run through N Euler steps). Validates the
//! loop orchestration — timestep = 1 - sigma, scheduler stepping, velocity sign — composed over
//! the independently parity-tested transformer + scheduler. 1e-2 (Metal fp32, N forwards).

use mlx_gen::weights::Weights;
use mlx_gen::FlowMatchEuler;
use mlx_gen_z_image::{denoise, ZImageTransformer, ZImageTransformerConfig};
use mlx_rs::ops::all_close;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/denoise_loop.safetensors"
);

fn small_cfg() -> ZImageTransformerConfig {
    ZImageTransformerConfig {
        patch_size: 2,
        f_patch_size: 1,
        in_channels: 4,
        dim: 96,
        n_layers: 2,
        n_refiner_layers: 1,
        n_heads: 4,
        norm_eps: 1e-5,
        cap_feat_dim: 32,
        rope_theta: 256.0,
        t_scale: 1000.0,
        axes_dims: vec![8, 8, 8],
        axes_lens: vec![64, 64, 64],
        frequency_embedding_size: 256,
    }
}

#[test]
fn denoise_loop_matches_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let transformer = ZImageTransformer::from_weights(&w, "w", small_cfg()).unwrap();

    // Reconstruct the exact schedule the fork used (sigmas dumped directly).
    let sigmas = w.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let scheduler = FlowMatchEuler { sigmas };

    let init = w.require("init").unwrap().clone();
    let cap_feats = w.require("cap_feats").unwrap();
    let final_latents = denoise(&transformer, &scheduler, init, cap_feats).unwrap();

    let want = w.require("final_latents").unwrap();
    assert_eq!(final_latents.shape(), want.shape(), "final latents shape");
    assert!(
        all_close(&final_latents, want, 1e-2, 1e-2, false)
            .unwrap()
            .item::<bool>(),
        "denoise loop diverged from the fork"
    );
}
