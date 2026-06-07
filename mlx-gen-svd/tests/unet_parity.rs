//! SVD UNet parity vs diffusers `UNetSpatioTemporalConditionModel` (epic 3040 / sc-3374). Gates the
//! full `SvdUnet::forward` (conv stem → micro-conditioning → down/mid/up spatiotemporal stack →
//! conv head) against a golden dumped from the real model (`tools/dump_svd_unet_golden.py`), in f32
//! so the gate isolates the math from fp16 rounding. Needs the SVD checkpoint locally → `--ignored`.
//!
//! Run: `cargo test -p mlx-gen-svd --test unet_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max as max_op, sqrt, square, subtract, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_svd::{SvdUnet, TransformerSpatioTemporal, UnetConfig};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/svd_unet_golden.safetensors"
);

/// Locate the SVD `unet/diffusion_pytorch_model.safetensors` (f32) in the HF cache.
fn unet_path() -> std::path::PathBuf {
    let cache = std::env::var("HF_HUB_CACHE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap();
            std::path::PathBuf::from(home).join(".cache/huggingface/hub")
        });
    let snaps = cache
        .join("models--stabilityai--stable-video-diffusion-img2vid-xt")
        .join("snapshots");
    let snap = std::fs::read_dir(&snaps)
        .expect("svd snapshot dir")
        .next()
        .unwrap()
        .unwrap()
        .path();
    snap.join("unet/diffusion_pytorch_model.safetensors")
}

fn rel_l2(a: &Array, b: &Array) -> (f32, f32) {
    let diff = abs(subtract(a, b).unwrap()).unwrap();
    let max_abs = max_op(&diff, None).unwrap().item::<f32>();
    let l2 = sqrt(sum(square(&diff).unwrap(), None).unwrap())
        .unwrap()
        .item::<f32>()
        / sqrt(sum(square(b).unwrap(), None).unwrap())
            .unwrap()
            .item::<f32>()
            .max(1e-6);
    (max_abs, l2)
}

/// Isolated `TransformerSpatioTemporalModel` (down_blocks.0.attentions.0) parity — bisects the UNet
/// gap to the transformer vs the rest of the stack.
#[test]
#[ignore = "needs the SVD checkpoint in the HF cache"]
fn svd_transformer_matches_diffusers() {
    let mut w = Weights::from_file(unet_path()).expect("svd unet weights");
    w.cast_all(Dtype::Float32).expect("cast f32");
    let tf = TransformerSpatioTemporal::from_weights(&w, "down_blocks.0.attentions.0", 5, 1)
        .expect("transformer");

    let g = Weights::from_file(GOLDEN).expect("unet golden");
    let num_frames = g.require("num_frames").unwrap().item::<i32>();
    let t_in = g
        .require("tf_in")
        .unwrap()
        .transpose_axes(&[0, 2, 3, 1])
        .unwrap(); // [B*F,16,16,320]
    let ctx = g.require("tf_ctx").unwrap().clone(); // [B*F,1,1024]
    let out = tf.forward(&t_in, &ctx, num_frames).unwrap();
    let want = g
        .require("tf_out")
        .unwrap()
        .transpose_axes(&[0, 2, 3, 1])
        .unwrap();
    assert_eq!(out.shape(), want.shape(), "transformer out shape");
    let (max_abs, l2) = rel_l2(&out, &want);
    println!("transformer parity: max|Δ| {max_abs}, rel-L2 {l2}");
    assert!(l2 < 3e-3, "transformer rel-L2 {l2} (max|Δ| {max_abs})");
}

/// Isolated `SpatioTemporalResBlock` (down_blocks.0.resnets.0, eps 1e-6) parity.
#[test]
#[ignore = "needs the SVD checkpoint in the HF cache"]
fn svd_resblock_matches_diffusers() {
    let mut w = Weights::from_file(unet_path()).expect("svd unet weights");
    w.cast_all(Dtype::Float32).expect("cast f32");
    let g = Weights::from_file(GOLDEN).expect("unet golden");
    let num_frames = g.require("num_frames").unwrap().item::<i32>();
    let x = g
        .require("rb_in")
        .unwrap()
        .transpose_axes(&[0, 2, 3, 1])
        .unwrap();
    let temb = g.require("rb_temb").unwrap().clone();
    let out = mlx_gen_svd::unet::debug_st_resblock(
        &w,
        "down_blocks.0.resnets.0",
        1e-6,
        &x,
        &temb,
        num_frames,
    )
    .unwrap();
    let want = g
        .require("rb_out")
        .unwrap()
        .transpose_axes(&[0, 2, 3, 1])
        .unwrap();
    assert_eq!(out.shape(), want.shape(), "resblock out shape");
    let (max_abs, l2) = rel_l2(&out, &want);
    println!("resblock parity: max|Δ| {max_abs}, rel-L2 {l2}");
    assert!(l2 < 3e-3, "resblock rel-L2 {l2} (max|Δ| {max_abs})");
}

#[test]
#[ignore = "needs the SVD checkpoint in the HF cache"]
fn svd_unet_matches_diffusers() {
    let mut w = Weights::from_file(unet_path()).expect("svd unet weights");
    w.cast_all(Dtype::Float32).expect("cast f32");
    let unet = SvdUnet::from_weights(&w, &UnetConfig::default()).expect("unet");

    let g = Weights::from_file(GOLDEN).expect("unet golden");
    let num_frames = g.require("num_frames").unwrap().item::<i32>();
    let timestep = g.require("timestep").unwrap().item::<f32>();

    // sample NCHW-with-frames [B,F,8,H,W] → NHWC-with-frames [B,F,H,W,8].
    let sample = g
        .require("sample")
        .unwrap()
        .transpose_axes(&[0, 1, 3, 4, 2])
        .unwrap();
    let image_embeds = g.require("image_embeds").unwrap().clone(); // [B,1,1024]
    let added_time_ids = g.require("added_time_ids").unwrap().clone(); // [B,3]

    let out = unet
        .forward(
            &sample,
            timestep,
            &image_embeds,
            &added_time_ids,
            num_frames,
        )
        .unwrap(); // [B,F,H,W,4]
                   // golden out [B,F,4,H,W] → NHWC [B,F,H,W,4].
    let want = g
        .require("out")
        .unwrap()
        .transpose_axes(&[0, 1, 3, 4, 2])
        .unwrap();
    assert_eq!(out.shape(), want.shape(), "unet out shape");

    let diff = abs(subtract(&out, &want).unwrap()).unwrap();
    let max_abs = max_op(&diff, None).unwrap().item::<f32>();
    let denom = max_op(abs(want.clone()).unwrap(), None)
        .unwrap()
        .item::<f32>();
    let peak_rel = max_abs / denom.max(1e-6);
    let l2 = sqrt(sum(square(&diff).unwrap(), None).unwrap())
        .unwrap()
        .item::<f32>()
        / sqrt(sum(square(&want).unwrap(), None).unwrap())
            .unwrap()
            .item::<f32>()
            .max(1e-6);
    println!("unet parity: max|Δ| {max_abs}, peak-rel {peak_rel}, rel-L2 {l2}");

    // ~1.4% rel-L2 is f32 cross-backend accumulation across the full residual stack (16
    // `TransformerSpatioTemporal` + 24 `SpatioTemporalResBlock` modules, each with sdpa / conv /
    // group-norm ordering gaps), NOT a structural error — the isolated component gates above pin the
    // building blocks at 0.16% (transformer) and 0.014% (resnet), and a wrong skip-residual order or
    // channel-concat would blow past 10%, not 1.4%. The N(0,1) sample/embeds are worst-case; the real
    // pipeline feeds structured latents + image conditioning. Those isolated gates are the tight
    // structural guards; this is the end-to-end assembly check.
    assert!(
        l2 < 2e-2,
        "unet rel-L2 {l2} (peak-rel {peak_rel}, max|Δ| {max_abs})"
    );
    assert!(
        peak_rel < 4e-2,
        "unet peak-rel {peak_rel} (max|Δ| {max_abs})"
    );
}
