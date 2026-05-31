//! sc-2352: end-to-end validation of the Z-Image port against a real-weights golden run.
//!
//! `#[ignore]`d — needs the real `Tongyi-MAI/Z-Image-Turbo` weights in the HF cache and the
//! golden produced by `tools/dump_z_image_golden.py` (gitignored, local). Run with:
//!   cargo test -p mlx-gen-z-image --release --test e2e_real_weights -- --ignored --nocapture
//!
//! Validates each stage of the pipeline on real bf16 weights against the fork's intermediates.

use std::path::PathBuf;

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::weights::Weights;
use mlx_gen::FlowMatchEuler;
use mlx_gen_z_image::text_encoder::{TextEncoder, ZTextEncoderConfig};
use mlx_gen_z_image::vae::{Vae, VaeDecoderConfig};
use mlx_gen_z_image::{
    create_noise, decoded_to_image, denoise, unpack_latents, ZImageTransformer,
    ZImageTransformerConfig,
};
use mlx_rs::{Array, Dtype};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/z_image_golden.safetensors"
);

/// Locate the Z-Image-Turbo snapshot dir (env override, else the HF cache).
fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("ZIMAGE_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

/// Peak-relative error `max|a-b| / max|b|` — the meaningful metric for high-dynamic-range
/// tensors compared against a bf16 golden.
fn peak_rel(a: &Array, b: &Array) -> f32 {
    // reshape to 1-D forces C-order materialization (decode/transpose views would otherwise
    // expose physical, not logical, order through as_slice).
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    max_diff / peak
}

fn bf16(a: &Array) -> Array {
    a.as_dtype(Dtype::Bfloat16).unwrap()
}

/// cap_feats = encoder_out[0, :num_valid, :] via a range gather (mlx-rs has no slice op).
fn slice_valid(encoder_out: &Array, num_valid: i32) -> Array {
    let sh = encoder_out.shape();
    let (s, h) = (sh[1], sh[2]);
    let flat = encoder_out.reshape(&[s, h]).unwrap();
    let idx = Array::from_slice(&(0..num_valid).collect::<Vec<i32>>(), &[num_valid]);
    flat.take_axis(&idx, 0).unwrap()
}

#[test]
#[ignore = "needs real Z-Image weights + local golden"]
fn e2e_text_encoder_matches_golden() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let num_valid: i32 = g.metadata("num_valid").unwrap().parse().unwrap();

    let w = Weights::from_dir(snapshot().join("text_encoder")).unwrap();
    let enc = TextEncoder::from_weights(&w, "model", &ZTextEncoderConfig::z_image()).unwrap();

    let out = enc
        .forward(
            g.require("input_ids").unwrap(),
            g.require("attention_mask").unwrap(),
        )
        .unwrap();
    let cap = slice_valid(&out, num_valid);

    let golden = g.require("cap_feats").unwrap();
    assert_eq!(cap.shape(), golden.shape(), "cap_feats shape");

    let a = cap.as_slice::<f32>();
    let b = golden.as_slice::<f32>();
    let max_abs_g = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let mean_diff: f32 =
        a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).sum::<f32>() / a.len() as f32;
    // Peak-relative error: the meaningful metric for a high-dynamic-range tensor (values reach
    // ~1.4e4) compared against a bf16 golden after a 35-layer f32 forward.
    let peak_rel = max_diff / max_abs_g;
    println!(
        "cap_feats: max|golden|={max_abs_g:.1} max|diff|={max_diff:.3} peak_rel={peak_rel:.2e} mean|diff|={mean_diff:.5}"
    );
    println!("mine[0..6]  = {:?}", &a[..6]);
    println!("golden[0..6]= {:?}", &b[..6]);
    assert!(
        peak_rel < 2e-3,
        "cap_feats diverged from the fork: peak-relative error {peak_rel:.2e} >= 2e-3"
    );
    println!(
        "✓ text encoder: cap_feats {:?} matches the fork golden (peak-rel {peak_rel:.2e})",
        cap.shape()
    );
}

fn load_real_transformer() -> ZImageTransformer {
    let mut w = Weights::from_dir(snapshot().join("transformer")).unwrap();
    for (from, to) in [
        ("t_embedder.mlp.0.weight", "t_embedder.linear1.weight"),
        ("t_embedder.mlp.0.bias", "t_embedder.linear1.bias"),
        ("t_embedder.mlp.2.weight", "t_embedder.linear2.weight"),
        ("t_embedder.mlp.2.bias", "t_embedder.linear2.bias"),
        (
            "all_final_layer.2-1.adaLN_modulation.1.weight",
            "all_final_layer.2-1.adaLN_modulation.0.weight",
        ),
        (
            "all_final_layer.2-1.adaLN_modulation.1.bias",
            "all_final_layer.2-1.adaLN_modulation.0.bias",
        ),
    ] {
        w.alias(from, to);
    }
    ZImageTransformer::from_weights(&w, "", ZImageTransformerConfig::turbo()).unwrap()
}

#[test]
#[ignore = "needs real Z-Image weights + local golden"]
fn e2e_transformer_single_forward_matches_golden() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let transformer = load_real_transformer();

    // First step in f32 (rules out bf16): v0 = transformer(init, 1 - sigma[0], cap_feats).
    let timestep0 = 1.0 - sigmas[0];
    let v = transformer
        .forward(
            g.require("init").unwrap(),
            timestep0,
            g.require("cap_feats").unwrap(),
        )
        .unwrap();
    let golden = g.require("v0").unwrap();
    assert_eq!(v.shape(), golden.shape(), "v0 shape");
    let pr = peak_rel(&v, golden);
    println!(
        "transformer single forward: v0 peak_rel={pr:.2e} shape={:?}",
        v.shape()
    );
    assert!(
        pr < 5e-2,
        "single transformer forward diverged at real resolution: peak_rel {pr:.2e}"
    );
    println!("✓ transformer single forward matches golden");
}

#[test]
#[ignore = "needs real Z-Image weights + local golden"]
fn e2e_denoise_loop_matches_golden() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let scheduler = FlowMatchEuler { sigmas };

    let mut w = Weights::from_dir(snapshot().join("transformer")).unwrap();
    // Checkpoint→internal rename: the DiT timestep embedder is `mlp.{0,2}` on disk, `linear{1,2}`
    // internally (the fork's weight mapping). Everything else matches directly.
    for (from, to) in [
        ("t_embedder.mlp.0.weight", "t_embedder.linear1.weight"),
        ("t_embedder.mlp.0.bias", "t_embedder.linear1.bias"),
        ("t_embedder.mlp.2.weight", "t_embedder.linear2.weight"),
        ("t_embedder.mlp.2.bias", "t_embedder.linear2.bias"),
        // final layer's adaLN is Sequential(SiLU, Linear) -> Linear at index 1 on disk, 0 internally.
        (
            "all_final_layer.2-1.adaLN_modulation.1.weight",
            "all_final_layer.2-1.adaLN_modulation.0.weight",
        ),
        (
            "all_final_layer.2-1.adaLN_modulation.1.bias",
            "all_final_layer.2-1.adaLN_modulation.0.bias",
        ),
    ] {
        w.alias(from, to);
    }
    let transformer =
        ZImageTransformer::from_weights(&w, "", ZImageTransformerConfig::turbo()).unwrap();

    // Match the fork's bf16 path: init noise + cap_feats fed to the DiT as bf16.
    let init = bf16(g.require("init").unwrap());
    let cap = bf16(g.require("cap_feats").unwrap());
    let out = denoise(&transformer, &scheduler, init, &cap).unwrap();
    let out = out.as_dtype(Dtype::Float32).unwrap();

    let golden = g.require("final_latents").unwrap();
    assert_eq!(out.shape(), golden.shape(), "final latents shape");
    let pr = peak_rel(&out, golden);
    println!(
        "denoise: final_latents peak_rel={pr:.2e} shape={:?}",
        out.shape()
    );
    // bf16 accumulation over 4 iterative steps (each feeding the next) compounds; the decoded
    // image is near-pixel-perfect, so this peak-relative latent drift is benign.
    assert!(pr < 1e-1, "final latents diverged: peak_rel {pr:.2e}");
    println!("✓ denoise loop matches golden (peak-rel {pr:.2e})");
}

/// Remap the diffusers VAE checkpoint (`decoder.*`, flat conv names, NCHW conv weights) to the
/// crate's internal decoder naming (`conv.`/`norm.` wrappers) with conv weights transposed to
/// NHWC `[out,kH,kW,in]`. Inserts the remapped (un-prefixed) keys alongside the originals.
fn remap_vae_decoder(w: &mut Weights) {
    let keys: Vec<String> = w
        .keys()
        .filter(|k| k.starts_with("decoder."))
        .map(String::from)
        .collect();
    for k in keys {
        let rest = k.strip_prefix("decoder.").unwrap();
        let (target, transpose): (String, bool) = match rest {
            "conv_in.weight" => ("conv_in.conv.weight".into(), true),
            "conv_in.bias" => ("conv_in.conv.bias".into(), false),
            "conv_out.weight" => ("conv_out.conv.weight".into(), true),
            "conv_out.bias" => ("conv_out.conv.bias".into(), false),
            "conv_norm_out.weight" => ("conv_norm_out.norm.weight".into(), false),
            "conv_norm_out.bias" => ("conv_norm_out.norm.bias".into(), false),
            _ => {
                let is_conv_w = rest.ends_with(".weight")
                    && (rest.contains(".conv1.")
                        || rest.contains(".conv2.")
                        || rest.contains(".conv_shortcut.")
                        || rest.contains(".upsamplers.0.conv."));
                (rest.to_string(), is_conv_w)
            }
        };
        let t = w.require(&k).unwrap().clone();
        let t = if transpose {
            t.transpose_axes(&[0, 2, 3, 1]).unwrap() // NCHW -> NHWC conv weight
        } else {
            t
        };
        w.insert(target, t);
    }
}

#[test]
#[ignore = "needs real Z-Image weights + local golden"]
fn e2e_vae_and_image_matches_golden() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let mut w = Weights::from_dir(snapshot().join("vae")).unwrap();
    remap_vae_decoder(&mut w);
    let vae = Vae::from_weights(&w, "", &VaeDecoderConfig::default_z_image()).unwrap();

    // golden final_latents [16,1,H,W] -> unpack [1,16,H,W] -> [1,16,1,H,W] for decode.
    let latents = g.require("final_latents").unwrap();
    let unpacked = unpack_latents(latents).unwrap();
    let sh = unpacked.shape();
    let latent5 = unpacked.reshape(&[sh[0], sh[1], 1, sh[2], sh[3]]).unwrap();
    let decoded = vae.decode(&latent5).unwrap(); // f32 (latents f32, weights bf16 -> promote)
    let decoded = decoded.as_dtype(Dtype::Float32).unwrap();

    let golden = g.require("decoded").unwrap();
    assert_eq!(decoded.shape(), golden.shape(), "decoded shape");
    let pr = peak_rel(&decoded, golden);
    let m = decoded.as_slice::<f32>();
    let gg = golden.as_slice::<f32>();
    let rng = |s: &[f32]| {
        (
            s.iter().cloned().fold(f32::MAX, f32::min),
            s.iter().cloned().fold(f32::MIN, f32::max),
        )
    };
    println!("vae: decoded peak_rel={pr:.2e} shape={:?}", decoded.shape());
    println!("  mine range  = {:?}", rng(m));
    println!("  golden range= {:?}", rng(gg));
    println!("  mine[0..4]={:?} golden[0..4]={:?}", &m[..4], &gg[..4]);
    assert!(pr < 2e-2, "VAE decode diverged: peak_rel {pr:.2e}");

    // RGB8 image: my decoded vs the golden decoded, both through decoded_to_image.
    let img = decoded_to_image(&decoded).unwrap();
    let gimg = decoded_to_image(golden).unwrap();
    let differ = img
        .pixels
        .iter()
        .zip(&gimg.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 2)
        .count();
    println!(
        "✓ vae+image: {}x{}, {} / {} pixels differ by >2",
        img.width,
        img.height,
        differ,
        img.pixels.len()
    );
    assert!(
        differ < img.pixels.len() / 50,
        "too many pixel diffs: {differ}"
    );
}

#[test]
#[ignore = "needs real Z-Image weights + local golden"]
fn e2e_full_pipeline_generates_fox() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let snap = snapshot();
    let (w, h, seed, steps) = (256u32, 256u32, 42u64, 4usize);

    // 1. Tokenize "a fox" with the Qwen chat template, pad to 512 — and validate the ids match
    //    the fork's golden (proves the tokenizer + chat template).
    let tok = TextTokenizer::from_file(
        snap.join("tokenizer/tokenizer.json"),
        TokenizerConfig {
            max_length: 512,
            pad_token_id: 151643,
            chat_template: ChatTemplate::QwenInstruct,
            pad_to_max_length: true,
        },
    )
    .unwrap();
    let t = tok.tokenize("a fox").unwrap();
    let num_valid: i32 = g.metadata("num_valid").unwrap().parse().unwrap();
    let take_n =
        |a: &Array| a.reshape(&[-1]).unwrap().as_slice::<i32>()[..num_valid as usize].to_vec();
    assert_eq!(
        take_n(&t.input_ids),
        take_n(g.require("input_ids").unwrap()),
        "tokenizer input_ids diverge from the fork"
    );

    // 2. Text encoder -> cap_feats.
    let te = TextEncoder::from_weights(
        &Weights::from_dir(snap.join("text_encoder")).unwrap(),
        "model",
        &ZTextEncoderConfig::z_image(),
    )
    .unwrap();
    let enc = te.forward(&t.input_ids, &t.attention_mask).unwrap();
    let cap = slice_valid(&enc, num_valid);

    // 3. Seeded noise -> denoise (everything bf16, matching the fork's path).
    let transformer = load_real_transformer();
    let scheduler = FlowMatchEuler::for_image(steps, w, h);
    let noise = bf16(&create_noise(seed, w, h).unwrap());
    let latents = denoise(&transformer, &scheduler, noise, &bf16(&cap)).unwrap();

    // 4. VAE decode -> RGB8 image.
    let mut vw = Weights::from_dir(snap.join("vae")).unwrap();
    remap_vae_decoder(&mut vw);
    let vae = Vae::from_weights(&vw, "", &VaeDecoderConfig::default_z_image()).unwrap();
    let unpacked = unpack_latents(&latents).unwrap();
    let sh = unpacked.shape();
    let latent5 = unpacked.reshape(&[sh[0], sh[1], 1, sh[2], sh[3]]).unwrap();
    let decoded = vae
        .decode(&latent5)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let img = decoded_to_image(&decoded).unwrap();

    // Save the Rust render for visual inspection.
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tools/golden/rust_fox.png");
    image::save_buffer(
        &out,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();

    // Compare to the fork's golden image (my latents differ from the golden by bf16-loop drift,
    // so allow a small fraction of pixels to differ).
    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let differ = img
        .pixels
        .iter()
        .zip(&gimg.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count();
    println!(
        "✓ full pipeline: prompt->image {}x{}; {} / {} pixels differ by >8 from the fork; saved {}",
        img.width,
        img.height,
        differ,
        img.pixels.len(),
        out.display()
    );
    assert!(
        differ < img.pixels.len() / 20,
        "full-pipeline image diverges: {differ} pixels"
    );
}
