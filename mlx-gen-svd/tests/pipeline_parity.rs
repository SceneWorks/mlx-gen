//! SVD pipeline e2e parity vs diffusers `StableVideoDiffusionPipeline` (epic 3040 / sc-3375). Gates
//! the deterministic core — `SvdPipeline::denoise` (frame-wise CFG v-prediction Euler loop with
//! image-latent channel-concat) + `decode` (chunked temporal VAE decode) — against a golden dumped
//! from the real components fed identical conditioning + init noise (`tools/dump_svd_pipeline_golden.py`),
//! in f32. Needs the SVD checkpoint locally → `--ignored`.
//!
//! Run: `cargo test -p mlx-gen-svd --test pipeline_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max as max_op, sqrt, square, subtract, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_svd::{
    ImageEncoderConfig, SchedulerConfig, SvdImageEncoder, SvdPipeline, SvdUnet, SvdVae, UnetConfig,
    VaeConfig,
};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/svd_pipeline_golden.safetensors"
);

fn snapshot() -> std::path::PathBuf {
    let cache = std::env::var("HF_HUB_CACHE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/huggingface/hub")
        });
    let snaps = cache
        .join("models--stabilityai--stable-video-diffusion-img2vid-xt")
        .join("snapshots");
    std::fs::read_dir(&snaps)
        .expect("svd snapshot dir")
        .next()
        .unwrap()
        .unwrap()
        .path()
}

fn load_f32(path: std::path::PathBuf) -> Weights {
    let mut w = Weights::from_file(path).expect("weights");
    w.cast_all(Dtype::Float32).expect("cast f32");
    w
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

#[test]
#[ignore = "needs the SVD checkpoint in the HF cache"]
fn svd_pipeline_matches_diffusers() {
    let snap = snapshot();
    let vae = SvdVae::from_weights(
        &load_f32(snap.join("vae/diffusion_pytorch_model.safetensors")),
        &VaeConfig::default(),
    )
    .unwrap();
    let unet = SvdUnet::from_weights(
        &load_f32(snap.join("unet/diffusion_pytorch_model.safetensors")),
        &UnetConfig::default(),
    )
    .unwrap();
    let enc = SvdImageEncoder::from_weights(
        &load_f32(snap.join("image_encoder/model.safetensors")),
        &ImageEncoderConfig::default(),
    )
    .unwrap();
    let pipe = SvdPipeline::new(enc, vae, unet, SchedulerConfig::default());

    let g = Weights::from_file(GOLDEN).expect("pipeline golden");
    let meta = g.require("meta").unwrap();
    let (num_frames, steps) = (
        meta.as_slice::<i32>()[0],
        meta.as_slice::<i32>()[1] as usize,
    );
    let guidance = g.require("guidance").unwrap();
    let (min_g, max_g) = (guidance.as_slice::<f32>()[0], guidance.as_slice::<f32>()[1]);

    // NCHW-with-frames [1,F,4,8,8] → NHWC [1,F,8,8,4]; embeds [1,1,1024] unchanged.
    let to_nhwc5 = |name: &str| {
        g.require(name)
            .unwrap()
            .transpose_axes(&[0, 1, 3, 4, 2])
            .unwrap()
    };
    let init_latents = to_nhwc5("init_latents");
    let image_latents = to_nhwc5("image_latents");
    let image_embeds = g.require("image_embeds").unwrap().clone();
    let added_time_ids = g.require("added_time_ids").unwrap().clone();

    // --- denoise ---
    let latents = pipe
        .denoise(
            &init_latents,
            &image_embeds,
            &image_latents,
            &added_time_ids,
            num_frames,
            steps,
            min_g,
            max_g,
        )
        .unwrap();
    let want_latents = to_nhwc5("final_latents");
    assert_eq!(latents.shape(), want_latents.shape(), "final latents shape");
    let (l_abs, l_l2) = rel_l2(&latents, &want_latents);
    println!("denoise parity: max|Δ| {l_abs}, rel-L2 {l_l2}");

    // --- decode --- isolated: decode the *golden* final latents so this measures the temporal VAE
    // decode alone (not the propagated denoise gap). golden frames [1,3,F,64,64] → NHWC [1,F,64,64,3].
    let frames = pipe.decode(&want_latents, num_frames, num_frames).unwrap();
    let want_frames = g
        .require("frames")
        .unwrap()
        .transpose_axes(&[0, 2, 3, 4, 1])
        .unwrap();
    assert_eq!(frames.shape(), want_frames.shape(), "frames shape");
    let (f_abs, f_l2) = rel_l2(&frames, &want_frames);
    println!("decode parity (golden latents): max|Δ| {f_abs}, rel-L2 {f_l2}");

    // Deterministic e2e (identical conditioning + init noise) → f32 cross-backend accumulation only.
    // denoise carries the per-step UNet gap (sc-3374, ~1.4% rel-L2/step) through the CFG combine
    // (guidance up to 3× amplifies cond−uncond) and 2 Euler steps → ~4%; the isolated UNet gate
    // (sc-3374) is the structural guard. decode (on identical golden latents) is the S1 temporal VAE
    // and stays tight — confirming the chunked-decode + scaling wiring is exact.
    assert!(l_l2 < 6e-2, "denoise rel-L2 {l_l2} (max|Δ| {l_abs})");
    assert!(f_l2 < 5e-3, "decode rel-L2 {f_l2} (max|Δ| {f_abs})");
}
