//! sc-11130 hot-path rewrites — behavior-preservation guards that run by default on the committed
//! tiny fixture (no real weights / Metal snapshot).
//!
//! - **F-155**: `Sampler::sample` and `Sampler::sample_tiled` must draw the *same* seeded noise/ε
//!   stream (they now share one `draw_noise_eps`). With a tile larger than the output the tiled forward
//!   collapses to the exact whole-image `net.forward`, so the two outputs must be **element-equal** — a
//!   direct proof the shared draw did not fork the RNG sequence between the tiled and whole paths.
//! - **F-154**: `PidDecoder::decode` / `decode_tiled` under a failing `PID_CAPTURE_LATENT` must not
//!   panic — the capture is best-effort (the casts no longer `unwrap()`), and `decode_tiled` honors the
//!   env var symmetrically with `decode`.

use mlx_gen::decoder::LatentDecoder;
use mlx_gen::weights::Weights;
use mlx_gen_pid::{PidConfig, PidDecoder, PidNet, RopeMode, Sampler, SamplerConfig};
use mlx_rs::ops::{abs, max, subtract};
use mlx_rs::Array;

/// The tiny latent-only config baked into `dump_pid_sampler.py` (matches `sampler_parity.rs`).
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
    Weights::from_file(format!("{dir}/sampler_tiny.safetensors")).unwrap()
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    max(abs(subtract(a, b).unwrap()).unwrap(), None)
        .unwrap()
        .item::<f32>()
}

#[test]
fn sample_and_whole_image_tiled_are_element_equal() {
    let w = fixture();
    let net = PidNet::from_weights(&w, "", &tiny_cfg()).unwrap();
    let sampler = Sampler::new(&SamplerConfig::distill_4step());

    let caption = w.require("__io__.caption").unwrap().clone();
    let lq_latent = w.require("__io__.lq_latent").unwrap().clone();
    let sigma = w.require("__io__.sigma").unwrap().clone();
    let nsh = w.require("__io__.noise").unwrap().shape().to_vec();
    let (b, h, wd) = (nsh[0], nsh[2], nsh[3]);

    // A tile far larger than the output → the tiled forward is a single, exact `net.forward`. So the
    // ONLY thing that could differ between `sample` and `sample_tiled` is the seeded draw — which they
    // now share (F-155). Same seed ⇒ byte-for-byte identical output.
    let seed = 4242;
    let whole = sampler
        .sample(&net, &caption, &lq_latent, &sigma, b, h, wd, seed, None)
        .unwrap();
    let tiled = sampler
        .sample_tiled(
            &net, &caption, &lq_latent, &sigma, b, h, wd, seed, 100_000, 256, None,
        )
        .unwrap();
    assert_eq!(whole.shape(), tiled.shape());
    assert_eq!(
        max_abs_diff(&whole, &tiled),
        0.0,
        "sample vs whole-image sample_tiled must be element-equal (shared RNG draw)"
    );
}

#[test]
fn capture_failure_does_not_panic_mid_decode() {
    let w = fixture();
    let net = PidNet::from_weights(&w, "", &tiny_cfg()).unwrap();
    let caption = w.require("__io__.caption").unwrap().clone();
    let lq_latent = w.require("__io__.lq_latent").unwrap().clone();

    let decoder = PidDecoder::new(
        net,
        Sampler::new(&SamplerConfig::distill_4step()),
        caption,
        0.0,
        2, // scale
        2, // vae_compression
        7, // seed
    );

    // Point capture at a path whose parent does not exist → the safetensors save fails. A best-effort
    // capture (F-154) must log and continue, never `unwrap()`-panic the decode. RUST_TEST_THREADS=1
    // (forced) makes this process-global env write safe within the test binary.
    std::env::set_var(
        "PID_CAPTURE_LATENT",
        "/nonexistent-dir-sc11130/does/not/exist/capture.safetensors",
    );

    let whole = decoder.decode(&lq_latent);
    assert!(
        whole.is_ok(),
        "decode survived a failing capture: {whole:?}"
    );
    // decode_tiled now captures too (symmetry) — same best-effort contract. Tile >> output → 1 tile.
    let tiled = decoder.decode_tiled(&lq_latent, 100_000, 256);
    assert!(
        tiled.is_ok(),
        "decode_tiled survived a failing capture: {tiled:?}"
    );

    std::env::remove_var("PID_CAPTURE_LATENT");
}
