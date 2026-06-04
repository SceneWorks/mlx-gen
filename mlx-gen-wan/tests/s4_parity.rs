//! S4 parity gate: the dense **T2V pipeline** (`pipeline::denoise` + VAE decode) must reproduce the
//! `mlx_video` reference's CFG denoise loop + decode.
//!
//! Self-contained committed fixture (`tools/dump_s4_fixtures.py`): a tiny seeded dense `WanModel` +
//! tiny z16 `WanVAE`, with **injected** context + initial noise (RNG isn't portable across
//! mlx-python/mlx-rs), run through the reference's Euler CFG loop. Runs in CI, no real weights.
//!
//! The DiT runs **bf16** (the production regime, sc-2678 S3), so the final-latent gap is the known
//! cross-build bf16 kernel delta (MLX 0.31.1+patches vs the reference's native 0.31.2) accumulated
//! over the loop — bounded, not a code bug. The orchestration logic (scheduler stepping, CFG
//! combine, resolution/seq-len math) is gated bit-tight by the f32 unit tests in `pipeline.rs`.

use mlx_gen::weights::Weights;
use mlx_gen_wan::config::WanModelConfig;
use mlx_gen_wan::pipeline::{decode_to_frames, denoise};
use mlx_gen_wan::scheduler::SolverKind;
use mlx_gen_wan::{WanTransformer, WanVae};

fn fixture() -> Weights {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/s4_pipeline.safetensors"
    );
    Weights::from_file(path)
        .unwrap_or_else(|e| panic!("read {path}: {e} (run dump_s4_fixtures.py)"))
}

/// The tiny dense config the fixture was dumped with (mirrors `dump_s4_fixtures.py`).
fn tiny_cfg() -> WanModelConfig {
    let mut c = WanModelConfig::wan21_t2v_1_3b();
    c.dim = 128;
    c.num_heads = 1; // head_dim 128
    c.num_layers = 2;
    c.ffn_dim = 256;
    c.freq_dim = 256;
    c.text_dim = 32;
    c.text_len = 8;
    c.in_dim = 16;
    c.out_dim = 16;
    c.vae_z_dim = 16;
    c.dual_model = false;
    c
}

fn diff(got: &[f32], exp: &[f32]) -> (f32, f64) {
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut sum_ref = 0f64;
    for (g, e) in got.iter().zip(exp.iter()) {
        let d = (g - e).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
        sum_ref += e.abs() as f64;
    }
    (max_abs, sum_abs / sum_ref.max(1e-9))
}

#[test]
fn wan_t2v_pipeline_matches_reference() {
    let w = fixture();
    let cfg = tiny_cfg();
    let dit = WanTransformer::from_weights(&w, &cfg).expect("build DiT");
    let vae = WanVae::from_weights(&w).expect("build VAE");

    let ctx_cond = dit.embed_text(w.require("ctx_cond").unwrap()).unwrap();
    let ctx_uncond = dit.embed_text(w.require("ctx_uncond").unwrap()).unwrap();
    let init_noise = w.require("init_noise").unwrap();

    let mut steps_seen = 0usize;
    let latents = denoise(
        &dit,
        SolverKind::Euler,
        cfg.num_train_timesteps,
        4,   // steps
        5.0, // shift
        3.0, // guidance
        &ctx_cond,
        Some(&ctx_uncond),
        init_noise,
        &mut |_| steps_seen += 1,
    )
    .expect("denoise");
    assert_eq!(steps_seen, 4, "progress callback fired per step");

    // Final-latent parity (the denoise loop output).
    let exp_lat = w.require("final_latents").unwrap();
    assert_eq!(latents.shape(), exp_lat.shape(), "final latent shape");
    let (la_max, la_mr) = diff(latents.as_slice::<f32>(), exp_lat.as_slice::<f32>());
    println!(
        "[latents] shape={:?} max|Δ|={la_max:.3e} mean_rel={la_mr:.3e}",
        latents.shape()
    );

    // Full e2e: decode my latents, compare to the reference's decoded video (f32, [-1,1]).
    let video = vae
        .decode(&latents.reshape(&prepend1(latents.shape())).unwrap())
        .unwrap();
    let exp_vid = w.require("video").unwrap();
    assert_eq!(video.shape(), exp_vid.shape(), "video shape");
    let (vid_max, vid_mr) = diff(video.as_slice::<f32>(), exp_vid.as_slice::<f32>());
    println!(
        "[video]   shape={:?} max|Δ|={vid_max:.3e} mean_rel={vid_mr:.3e}",
        video.shape()
    );

    // Measured: latents mean_rel ~6.4e-3, video ~5.7e-3 — the bf16 DiT cross-build delta (MLX
    // 0.31.1+patches vs the reference's 0.31.2), scaled down per step by the Euler dt. Gate at 2e-2
    // (3× headroom for cross-machine bf16-kernel variance); a logic bug gives mean_rel ~O(1). The
    // f32 orchestration (scheduler/CFG/seq-len) is gated bit-tight by the pipeline.rs unit tests.
    assert!(la_mr < 2e-2, "latents diverged: mean_rel={la_mr:.3e}");
    assert!(vid_mr < 2e-2, "video diverged: mean_rel={vid_mr:.3e}");

    // decode_to_frames yields uint8 [F, H, W, 3] of the right shape.
    let frames = decode_to_frames(&vae, &latents, None).unwrap();
    let vsh = exp_vid.shape(); // [1,3,F,H,W]
    assert_eq!(
        frames.shape(),
        &[vsh[2], vsh[3], vsh[4], 3],
        "frame tensor [F,H,W,3]"
    );
    assert_eq!(frames.dtype(), mlx_rs::Dtype::Uint8);
}

fn prepend1(shape: &[i32]) -> Vec<i32> {
    let mut s = vec![1];
    s.extend_from_slice(shape);
    s
}
