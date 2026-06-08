//! sc-3623: real-weight structural check for the XLabs FLUX IP-Adapter modules.
//!
//! `#[ignore]`d — loads the cached `openai/clip-vit-large-patch14` image tower (sc-3622) + the
//! `XLabs-AI/flux-ip-adapter` `ip_adapter.safetensors`. Run:
//!
//! ```text
//! cargo test -p mlx-gen-flux --release --test ip_adapter_real_weights -- --ignored --nocapture
//! ```
//!
//! This validates the IP-Adapter primitive in isolation from the 24 GB FLUX transformer (the full
//! denoise-loop A/B is sc-3624). It proves, on **real** weights:
//!   1. `FluxIpAdapter` loads all 19 double-block K/V projections + the `ImageProjModel`.
//!   2. CLIP `image_embeds` `[1,768]` → image-prompt tokens `[1,4,4096]` (the ImageProjModel runs).
//!   3. `double_block_ip` produces a finite, deterministic per-block residual `[1,img_seq,3072]`
//!      from a post-RoPE image query, and the head reshape (24×128) lines up.
//!   4. `scale = 0` and out-of-range block indices short-circuit to `None` (the no-op / CFG-uncond
//!      path), so the plain txt2img render is untouched.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::Image;
use mlx_gen_flux::transformer::DitImageInjector;
use mlx_gen_flux::{FluxIpAdapter, FluxIpImageEncoder, FluxIpInjector};
use mlx_rs::ops::{abs, subtract};
use mlx_rs::{Array, Dtype};

fn hf_snapshot(repo: &str, file: &str) -> PathBuf {
    let home = std::env::var("HOME").unwrap();
    let snaps =
        PathBuf::from(home).join(format!(".cache/huggingface/hub/models--{repo}/snapshots"));
    let dir = std::fs::read_dir(&snaps)
        .unwrap_or_else(|_| panic!("HF cache snapshots dir for {repo}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir");
    dir.join(file)
}

fn clip_vit_l_weights() -> Weights {
    if let Ok(p) = std::env::var("CLIP_VIT_L_SNAPSHOT") {
        let p = PathBuf::from(p);
        let file = if p.is_dir() {
            p.join("model.safetensors")
        } else {
            p
        };
        return Weights::from_file(file).unwrap();
    }
    Weights::from_file(hf_snapshot(
        "openai--clip-vit-large-patch14",
        "model.safetensors",
    ))
    .unwrap()
}

fn xlabs_ip_weights() -> Weights {
    let file = std::env::var("FLUX_IP_ADAPTER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| hf_snapshot("XLabs-AI--flux-ip-adapter", "ip_adapter.safetensors"));
    Weights::from_file(file).unwrap()
}

fn gradient(w: u32, h: u32) -> Image {
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.push((x % 256) as u8);
            pixels.push((y % 256) as u8);
            pixels.push(((x + y) % 256) as u8);
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

fn all_finite(a: &Array) -> bool {
    mlx_rs::ops::max(abs(a).unwrap(), None)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .item::<f32>()
        .is_finite()
}

#[test]
#[ignore = "loads openai/clip-vit-large-patch14 + XLabs-AI/flux-ip-adapter weights"]
fn flux_ip_adapter_modules_on_real_weights() {
    // CLIP image embeds (sc-3622) → IP-Adapter (sc-3623).
    let embeds = FluxIpImageEncoder::from_weights(&clip_vit_l_weights())
        .unwrap()
        .encode(&gradient(512, 512))
        .unwrap();
    assert_eq!(embeds.shape(), &[1, 768]);

    let adapter = FluxIpAdapter::from_weights(&xlabs_ip_weights()).unwrap();
    assert_eq!(adapter.num_blocks(), 19, "FLUX has 19 double blocks");

    // ImageProjModel: [1,768] → [1,4,4096].
    let tokens = adapter.tokens(&embeds).unwrap();
    assert_eq!(
        tokens.shape(),
        &[1, 4, 4096],
        "image-prompt tokens [1,4,4096]"
    );
    assert!(all_finite(&tokens), "image-prompt tokens finite");

    // Synthetic post-RoPE image query [1, HEADS=24, img_seq, HEAD_DIM=128].
    let img_seq = 256;
    mlx_rs::random::seed(7).unwrap();
    let img_q = mlx_rs::random::normal::<f32>(&[1, 24, img_seq, 128], None, None, None).unwrap();

    let inj = FluxIpInjector::new(&adapter, &embeds, 0.7).unwrap();

    // Residual shape + finiteness, for the first and last double block (head reshape must line up).
    for b in [0usize, 18] {
        let r = inj
            .double_block_ip(b, &img_q)
            .unwrap()
            .unwrap_or_else(|| panic!("block {b} must inject at scale>0"));
        assert_eq!(
            r.shape(),
            &[1, img_seq, 3072],
            "block {b} residual [1,{img_seq},3072]"
        );
        assert!(all_finite(&r), "block {b} residual finite");
    }

    // Determinism (same weights + same query → byte-identical).
    let r0a = inj.double_block_ip(0, &img_q).unwrap().unwrap();
    let r0b = inj.double_block_ip(0, &img_q).unwrap().unwrap();
    let dmax = mlx_rs::ops::max(abs(subtract(&r0a, &r0b).unwrap()).unwrap(), None)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .item::<f32>();
    assert_eq!(dmax, 0.0, "double_block_ip must be deterministic");

    // No-op paths: out-of-range block index, and scale=0 (CFG uncond / disabled).
    assert!(
        inj.double_block_ip(19, &img_q).unwrap().is_none(),
        "block 19 is out of range → None"
    );
    let disabled = FluxIpInjector::disabled(&adapter, &embeds).unwrap();
    assert!(
        disabled.double_block_ip(0, &img_q).unwrap().is_none(),
        "scale=0 → None (no-op)"
    );

    let rms = {
        let sq = mlx_rs::ops::multiply(&r0a, &r0a).unwrap();
        mlx_rs::ops::mean(&sq, None)
            .unwrap()
            .sqrt()
            .unwrap()
            .item::<f32>()
    };
    println!("[flux-ip] tokens[1,4,4096] + 19 block residuals OK; block0 residual rms={rms:.4}");
}
