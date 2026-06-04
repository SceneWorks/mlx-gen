//! I2V-14B parity gate (sc-2681): the **channel-concat image→video** path must reproduce the
//! `mlx_video` reference's `is_i2v_channel_concat` setup + dual-expert denoise.
//!
//! Self-contained committed fixture (`tools/dump_i2v_fixtures.py`): two tiny seeded dual-expert
//! `WanModel`s with **in_dim 36** (16 noise + 20 conditioning), a tiny z16 `WanVAE` **with encoder**,
//! and a synthetic RGB image, run through the reference's I2V path (cover-fit/center-crop preprocess;
//! VAE-encode the first-frame video plus a 4-channel temporal mask into `y = [mask, z_video]`
//! `[20, …]`; the boundary-switched 0.9 loop with `y` channel-concatenated onto the noise latent).
//!
//! Three checks, mirroring the story's acceptance:
//!   1. [`preprocess_i2v_image`] is **bit-exact** to the reference's PIL-LANCZOS cover-fit + crop.
//!   2. [`build_i2v_y`] reproduces the reference's `y` (VAE-encode + mask + concat).
//!   3. [`denoise_moe`] with `Some(y)` (the **36-channel patch-embed** input) reproduces the golden
//!      latents + decoded video (bf16 DiT cross-build envelope, gated at 2e-2 like S4/S5).

use mlx_gen::weights::Weights;
use mlx_gen::Image;
use mlx_gen_wan::config::WanModelConfig;
use mlx_gen_wan::pipeline::{build_i2v_y, denoise_moe, preprocess_i2v_image};
use mlx_gen_wan::scheduler::SolverKind;
use mlx_gen_wan::{Expert, WanTransformer, WanVae};

// Must match tools/dump_i2v_fixtures.py.
const FRAMES: usize = 5;
const HEIGHT: u32 = 16;
const WIDTH: u32 = 16;
const IMG_H: u32 = 40;
const IMG_W: u32 = 48;
const VAE_STRIDE: (usize, usize, usize) = (4, 8, 8);

fn load(name: &str) -> Weights {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    Weights::from_file(&path)
        .unwrap_or_else(|e| panic!("read {path}: {e} (run dump_i2v_fixtures.py)"))
}

/// The tiny dual I2V config the fixture was dumped with (`dump_i2v_fixtures.py`).
fn tiny_cfg() -> WanModelConfig {
    let mut c = WanModelConfig::wan22_i2v_14b();
    c.dim = 128;
    c.num_heads = 1;
    c.num_layers = 2;
    c.ffn_dim = 256;
    c.freq_dim = 256;
    c.text_dim = 32;
    c.text_len = 8;
    c.in_dim = 36;
    c.out_dim = 16;
    c.vae_z_dim = 16;
    c.boundary = 0.9;
    c.num_train_timesteps = 1000;
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

fn prepend1(shape: &[i32]) -> Vec<i32> {
    let mut s = vec![1];
    s.extend_from_slice(shape);
    s
}

#[test]
fn wan_i2v_channel_concat_matches_reference() {
    let low_w = load("i2v_low.safetensors");
    let high_w = load("i2v_high.safetensors");
    let cfg = tiny_cfg();

    // Rebuild the raw input image from the dumped uint8 [IMG_H, IMG_W, 3] buffer.
    let img_bytes = low_w.require("img_uint8").unwrap();
    let image = Image {
        width: IMG_W,
        height: IMG_H,
        pixels: img_bytes.as_slice::<u8>().to_vec(),
    };

    // (1) preprocess: cover-fit LANCZOS + center-crop + normalize — must be bit-exact to PIL.
    let img_chw = preprocess_i2v_image(&image, WIDTH, HEIGHT).expect("preprocess");
    let exp_chw = low_w.require("img_chw").unwrap();
    assert_eq!(img_chw.shape(), exp_chw.shape(), "img_chw shape");
    let (px_max, px_mr) = diff(img_chw.as_slice::<f32>(), exp_chw.as_slice::<f32>());
    println!("[preprocess] max|Δ|={px_max:.3e} mean_rel={px_mr:.3e}");
    assert!(
        px_max < 1e-5,
        "preprocess not bit-exact to PIL LANCZOS: max|Δ|={px_max:.3e}"
    );

    // (2) build_i2v_y: VAE-encode first-frame video + temporal mask + concat → [20, T_lat, h, w].
    let vae = WanVae::from_weights(&low_w).expect("VAE (with encoder)");
    let y = build_i2v_y(&vae, &image, FRAMES, HEIGHT, WIDTH, VAE_STRIDE).expect("build_i2v_y");
    let exp_y = low_w.require("y").unwrap();
    assert_eq!(y.shape(), exp_y.shape(), "y shape");
    let (y_max, y_mr) = diff(y.as_slice::<f32>(), exp_y.as_slice::<f32>());
    println!(
        "[build_i2v_y] shape={:?} max|Δ|={y_max:.3e} mean_rel={y_mr:.3e}",
        y.shape()
    );
    // VAE encode is f32 (bit-exact in S2); the only non-mask values come through it.
    assert!(
        y_mr < 2e-3,
        "y diverged from reference: mean_rel={y_mr:.3e}"
    );

    // (3) dual-expert MoE denoise with the 36-channel (noise ⊕ y) patch-embed input.
    let low_dit = WanTransformer::from_weights(&low_w, &cfg).expect("low DiT");
    let high_dit = WanTransformer::from_weights(&high_w, &cfg).expect("high DiT");

    let ctx_cond = low_w.require("ctx_cond").unwrap();
    let ctx_uncond = low_w.require("ctx_uncond").unwrap();
    let init_noise = low_w.require("init_noise").unwrap();

    let low = Expert {
        transformer: &low_dit,
        ctx_cond: low_dit.embed_text(ctx_cond).unwrap(),
        ctx_uncond: Some(low_dit.embed_text(ctx_uncond).unwrap()),
        guidance: 3.5,
    };
    let high = Expert {
        transformer: &high_dit,
        ctx_cond: high_dit.embed_text(ctx_cond).unwrap(),
        ctx_uncond: Some(high_dit.embed_text(ctx_uncond).unwrap()),
        guidance: 3.5,
    };
    let boundary_timestep = cfg.boundary * cfg.num_train_timesteps as f32; // 900

    let latents = denoise_moe(
        &low,
        &high,
        boundary_timestep,
        SolverKind::Euler,
        cfg.num_train_timesteps,
        4,
        5.0,
        init_noise,
        Some(&y),
        &mut |_| {},
    )
    .expect("denoise_moe");

    let exp_lat = low_w.require("final_latents").unwrap();
    assert_eq!(latents.shape(), exp_lat.shape(), "final latent shape");
    let (la_max, la_mr) = diff(latents.as_slice::<f32>(), exp_lat.as_slice::<f32>());
    println!(
        "[i2v latents] shape={:?} max|Δ|={la_max:.3e} mean_rel={la_mr:.3e}",
        latents.shape()
    );

    let video = vae
        .decode(&latents.reshape(&prepend1(latents.shape())).unwrap())
        .unwrap();
    let exp_vid = low_w.require("video").unwrap();
    assert_eq!(video.shape(), exp_vid.shape(), "video shape");
    let (vid_max, vid_mr) = diff(video.as_slice::<f32>(), exp_vid.as_slice::<f32>());
    println!(
        "[i2v video]   shape={:?} max|Δ|={vid_max:.3e} mean_rel={vid_mr:.3e}",
        video.shape()
    );

    // bf16 DiT cross-build envelope (see S4/S5); a y-concat / 36-ch patch-embed / routing bug gives
    // mean_rel ~O(1), not a few e-2.
    assert!(la_mr < 2e-2, "i2v latents diverged: mean_rel={la_mr:.3e}");
    assert!(vid_mr < 2e-2, "i2v video diverged: mean_rel={vid_mr:.3e}");
}
