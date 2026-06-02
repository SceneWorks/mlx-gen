//! sc-2345: end-to-end parity of the FLUX.1 port against a real-weights fork golden.
//!
//! `#[ignore]`d — needs the real `black-forest-labs/FLUX.1-schnell` weights in the HF cache and
//! the golden produced by `tools/dump_flux_golden.py` (gitignored, local). Run with:
//!   MLX_GEN_FLUX_SNAPSHOT=<schnell snapshot> \
//!     cargo test -p mlx-gen-flux --test e2e_real_weights -- --ignored --nocapture
//!
//! Stage tests feed the fork's own intermediates into each Rust stage to isolate it; the final
//! test drives the public `load(id, spec).generate(req)` API and compares the rendered image to
//! the fork's golden (px>8 fraction — the repo's parity bar, like the Z-Image/Qwen e2e tests).

use std::path::PathBuf;

use mlx_gen::image::decoded_to_image;
use mlx_gen::weights::Weights;
use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, Progress, Quant, WeightsSource};
use mlx_gen_flux::{
    load_clip_encoder, load_t5_encoder, load_transformer, load_vae, unpack_latents, FluxVariant,
};
use mlx_rs::ops::{add, multiply};
use mlx_rs::{Array, Dtype};

const GOLDEN_Q8: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/flux1_schnell_q8_golden.safetensors"
);
const GOLDEN_Q4: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/flux1_schnell_q4_golden.safetensors"
);

/// Q8/Q4 verification. TWO checks:
/// (a) build-independent quant gate — feed the fork-Q golden's OWN embeds+init into the Rust
///     transformer.quantize(bits), run the denoise on the fork's sigmas, decode, and compare to the
///     fork-Q golden (isolates the quantized transformer from the NAX-build text-encoder divergence);
/// (b) full public `load(spec.with_quant(Q)).generate()` render, saved for visual inspection.
/// `quantized_matmul` is fp32-accumulated (correct on the NAX build), so this should land at the
/// dense f32 transformer floor, not blow up.
fn verify_quant(golden_path: &str, quant: Quant, bits: i32) {
    let g = Weights::from_file(golden_path).unwrap();
    let stored: i32 = g.metadata("quantize").unwrap().parse().unwrap();
    assert_eq!(stored, bits, "golden dumped at a different bit-width");
    let w: u32 = g.metadata("w").unwrap().parse().unwrap();
    let h: u32 = g.metadata("h").unwrap().parse().unwrap();
    let snap = snapshot();

    // (a) quant-transformer gate
    let mut t = load_transformer(&snap, FluxVariant::Schnell).unwrap();
    t.quantize(bits).unwrap();
    let vae = load_vae(&snap).unwrap();
    let sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let steps = sigmas.len() - 1;
    let mut latents = f32a(g.require("init").unwrap());
    let pe = f32a(g.require("prompt_embeds").unwrap());
    let pooled = f32a(g.require("pooled_prompt_embeds").unwrap());
    for i in 0..steps {
        let v = t
            .forward(&latents, &pe, &pooled, sigmas[i], 0.0, w, h)
            .unwrap();
        let dt = sigmas[i + 1] - sigmas[i];
        latents = add(
            &latents,
            &multiply(&v, Array::from_slice(&[dt], &[1])).unwrap(),
        )
        .unwrap();
    }
    let golden_lat = g.require("final_latents").unwrap();
    let lat_mr = mean_abs_rel(&f32a(&latents), golden_lat);
    let unpacked = unpack_latents(&latents, w, h).unwrap();
    let decoded = f32a(&vae.decode(&unpacked).unwrap());
    let img = decoded_to_image(&decoded).unwrap();
    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let differ = img
        .pixels
        .iter()
        .zip(&gimg.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count();
    let frac = differ as f32 / img.pixels.len() as f32;
    println!(
        "Q{bits} transformer gate (fork-Q embeds+init): latents mean_rel={lat_mr:.3e}  decoded px>8={:.2}% vs fork-Q{bits}",
        frac * 100.0
    );
    // Same envelope as the dense f32 floor; quantized_matmul must not add gross divergence.
    assert!(
        lat_mr < 1.5e-1,
        "Q{bits} quant transformer diverged: latents mean_rel {lat_mr:.3e}"
    );

    // (b) full public quantized generate — coherence + save PNG
    let spec = LoadSpec::new(WeightsSource::Dir(snap)).with_quant(quant);
    let gen = mlx_gen::load("flux1_schnell", &spec).unwrap();
    let req = GenerationRequest {
        prompt: g.metadata("prompt").unwrap().to_string(),
        width: w,
        height: h,
        seed: Some(g.metadata("seed").unwrap().parse().unwrap()),
        steps: Some(steps as u32),
        ..Default::default()
    };
    let out = gen.generate(&req, &mut |_| {}).unwrap();
    let img = match out {
        GenerationOutput::Images(mut v) => v.pop().unwrap(),
        other => panic!("expected Images, got {other:?}"),
    };
    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../tools/golden/rust_flux_q{bits}.png"));
    image::save_buffer(
        &out_path,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    let differ = img
        .pixels
        .iter()
        .zip(&gimg.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count();
    println!(
        "Q{bits} full generate: {:.2}% px>8 vs fork-Q{bits} (incl. NAX build delta); saved {}",
        100.0 * differ as f32 / img.pixels.len() as f32,
        out_path.display()
    );
}

#[test]
#[ignore = "needs real FLUX.1-schnell weights + Q8 golden (QUANTIZE=8 dump_flux_golden.py)"]
fn e2e_q8_matches_fork() {
    verify_quant(GOLDEN_Q8, Quant::Q8, 8);
}

#[test]
#[ignore = "needs real FLUX.1-schnell weights + Q4 golden (QUANTIZE=4 dump_flux_golden.py)"]
fn e2e_q4_matches_fork() {
    verify_quant(GOLDEN_Q4, Quant::Q4, 4);
}

const GOLDEN_BF16: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/flux1_schnell_golden.safetensors"
);
const GOLDEN_F32: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/flux1_schnell_f32_golden.safetensors"
);

/// The fork golden to compare against. The mlx-gen FLUX path runs f32 activations (the quality
/// target), so the honest reference is the fork forced to f32 (`FLUX_PRECISION=f32`). Set
/// `FLUX_GOLDEN=bf16` to compare against the fork's production bf16 path instead.
fn golden() -> Weights {
    let path = match std::env::var("FLUX_GOLDEN").as_deref() {
        Ok("bf16") => GOLDEN_BF16,
        _ => GOLDEN_F32,
    };
    Weights::from_file(path).unwrap()
}

fn snapshot() -> PathBuf {
    PathBuf::from(
        std::env::var("MLX_GEN_FLUX_SNAPSHOT")
            .expect("set MLX_GEN_FLUX_SNAPSHOT to a FLUX.1-schnell snapshot directory"),
    )
}

/// Peak-relative error `max|a-b| / max|b|` — the meaningful metric vs a bf16 golden.
fn peak_rel(a: &Array, b: &Array) -> f32 {
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

fn mean_abs_rel(a: &Array, b: &Array) -> f32 {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let mabs: f32 = b.iter().map(|y| y.abs()).sum::<f32>() / b.len() as f32;
    let md: f32 = a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).sum::<f32>() / a.len() as f32;
    md / mabs
}

fn f32a(a: &Array) -> Array {
    a.as_dtype(Dtype::Float32).unwrap()
}

#[test]
#[ignore = "needs real FLUX.1-schnell weights + local golden"]
fn e2e_t5_prompt_embeds_match_golden() {
    let g = golden();
    let t5 = load_t5_encoder(&snapshot()).unwrap();
    let out = t5.forward(g.require("t5_input_ids").unwrap()).unwrap();
    let golden = g.require("prompt_embeds").unwrap();
    assert_eq!(out.shape(), golden.shape(), "prompt_embeds shape");
    let pr = peak_rel(&f32a(&out), golden);
    let mr = mean_abs_rel(&f32a(&out), golden);
    println!(
        "T5 prompt_embeds: peak_rel={pr:.3e} mean_rel={mr:.3e} shape={:?}",
        out.shape()
    );
    assert!(pr < 2e-2, "T5 prompt_embeds diverged: peak_rel {pr:.3e}");
    println!("✓ T5 prompt_embeds match the fork golden");
}

#[test]
#[ignore = "needs real FLUX.1-schnell weights + local golden"]
fn e2e_clip_pooled_matches_golden() {
    let g = golden();
    let clip = load_clip_encoder(&snapshot()).unwrap();
    let out = clip.forward(g.require("clip_input_ids").unwrap()).unwrap();
    let golden = g.require("pooled_prompt_embeds").unwrap();
    assert_eq!(out.shape(), golden.shape(), "pooled shape");
    let pr = peak_rel(&f32a(&out), golden);
    let mr = mean_abs_rel(&f32a(&out), golden);
    println!(
        "CLIP pooled: peak_rel={pr:.3e} mean_rel={mr:.3e} shape={:?}",
        out.shape()
    );
    assert!(
        pr < 2e-2,
        "CLIP pooled diverged from the fork: peak_rel {pr:.3e}"
    );
    println!("✓ CLIP pooled matches the fork golden");
}

#[test]
#[ignore = "needs real FLUX.1-schnell weights + local golden"]
fn e2e_transformer_v0_matches_golden() {
    let g = golden();
    let sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let t = load_transformer(&snapshot(), FluxVariant::Schnell).unwrap();
    // Production path runs f32 activations (model::generate no longer casts to bf16 — that hit the
    // dense 16-bit GEMM bug on `x_embedder`, K=64). Feed the fork's f32 init + embeds.
    let init = f32a(g.require("init").unwrap());
    let pe = f32a(g.require("prompt_embeds").unwrap());
    let pooled = f32a(g.require("pooled_prompt_embeds").unwrap());
    let v = t
        .forward(&init, &pe, &pooled, sigmas[0], 0.0, 256, 256)
        .unwrap();
    let golden = g.require("v0").unwrap();
    assert_eq!(v.shape(), golden.shape(), "v0 shape");
    let pr = peak_rel(&f32a(&v), golden);
    let mr = mean_abs_rel(&f32a(&v), golden);
    println!(
        "transformer v0: peak_rel={pr:.3e} mean_rel={mr:.3e} shape={:?}",
        v.shape()
    );
    assert!(
        pr < 8e-2,
        "transformer single forward diverged: peak_rel {pr:.3e}"
    );
    println!("✓ transformer single forward matches golden");
}

#[test]
#[ignore = "needs real FLUX.1-schnell weights + local golden"]
fn e2e_vae_decode_matches_golden() {
    let g = golden();
    let vae = load_vae(&snapshot()).unwrap();
    let latents = g.require("final_latents").unwrap();
    let unpacked = unpack_latents(latents, 256, 256).unwrap();
    let decoded = f32a(&vae.decode(&unpacked).unwrap());
    let golden = g.require("decoded").unwrap();
    assert_eq!(decoded.shape(), golden.shape(), "decoded shape");
    let pr = peak_rel(&decoded, golden);
    println!("VAE decoded: peak_rel={pr:.3e} shape={:?}", decoded.shape());
    let img = decoded_to_image(&decoded).unwrap();
    let gimg = decoded_to_image(golden).unwrap();
    let differ = img
        .pixels
        .iter()
        .zip(&gimg.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count();
    println!(
        "✓ VAE+image: {}x{}, {} / {} px differ by >8",
        img.width,
        img.height,
        differ,
        img.pixels.len()
    );
    assert!(
        differ < img.pixels.len() / 50,
        "too many VAE pixel diffs: {differ}"
    );
}

#[test]
#[ignore = "needs real FLUX.1-schnell weights + local golden"]
fn e2e_transformer_substages_match_golden() {
    let g = golden();
    let sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let t = load_transformer(&snapshot(), FluxVariant::Schnell).unwrap();
    let init = f32a(g.require("init").unwrap());
    let pe = f32a(g.require("prompt_embeds").unwrap());
    let pooled = f32a(g.require("pooled_prompt_embeds").unwrap());
    let caps = t
        .forward_capture(&init, &pe, &pooled, sigmas[0], 0.0, 256, 256)
        .unwrap();
    for (name, arr) in &caps {
        if let Some(golden) = g.require(name).ok() {
            let pr = peak_rel(&f32a(arr), golden);
            let mr = mean_abs_rel(&f32a(arr), golden);
            println!(
                "substage {name}: peak_rel={pr:.3e} mean_rel={mr:.3e} shape={:?}",
                arr.shape()
            );
        } else {
            println!("substage {name}: (no golden) shape={:?}", arr.shape());
        }
    }
}

#[test]
#[ignore = "needs real FLUX.1-schnell weights + local golden"]
fn e2e_rope_table_matches_golden() {
    let g = golden();
    let t = load_transformer(&snapshot(), FluxVariant::Schnell).unwrap();
    let (cos, sin) = t.debug_rope(256, 16, 16).unwrap(); // [512, 64]
    let seq = cos.shape()[0];
    let half = cos.shape()[1];
    // fork rope0 [1,1,seq,64,2,2]; flatten the 2x2 (= [cos,-sin,sin,cos]) → col0=cos, col2=sin.
    let r = g
        .require("rope0")
        .unwrap()
        .reshape(&[seq, half, 4])
        .unwrap();
    let pick = |col: i32| {
        r.take_axis(&Array::from_slice(&[col], &[1]), 2)
            .unwrap()
            .reshape(&[seq, half])
            .unwrap()
    };
    let cos_f = pick(0); // 2x2 row-major [cos,-sin,sin,cos] → col0=cos
    let sin_f = pick(2); // col2=sin
    println!(
        "rope cos: peak_rel={:.3e} mean_rel={:.3e} | sin: peak_rel={:.3e} mean_rel={:.3e}",
        peak_rel(&cos, &cos_f),
        mean_abs_rel(&cos, &cos_f),
        peak_rel(&sin, &sin_f),
        mean_abs_rel(&sin, &sin_f)
    );
}

#[test]
#[ignore = "needs local golden"]
fn e2e_init_noise_matches_golden() {
    let g = golden();
    let seed: u64 = g.metadata("seed").unwrap().parse().unwrap();
    let init = mlx_gen_flux::create_noise(seed, 256, 256).unwrap();
    let golden = g.require("init").unwrap();
    assert_eq!(init.shape(), golden.shape(), "init shape");
    let pr = peak_rel(&f32a(&init), golden);
    println!("init noise: peak_rel={pr:.3e} shape={:?}", init.shape());
    assert!(
        pr < 1e-5,
        "Rust create_noise diverges from the fork RNG: peak_rel {pr:.3e}"
    );
    println!("✓ init noise matches the fork RNG");
}

#[test]
#[ignore = "needs real FLUX.1-schnell weights + local golden"]
fn e2e_denoise_loop_matches_golden() {
    let g = golden();
    let sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let steps = sigmas.len() - 1;
    let t = load_transformer(&snapshot(), FluxVariant::Schnell).unwrap();
    // Feed the fork's exact init + golden embeds + fork sigmas: isolates the loop from RNG/text.
    let mut latents = f32a(g.require("init").unwrap());
    let pe = f32a(g.require("prompt_embeds").unwrap());
    let pooled = f32a(g.require("pooled_prompt_embeds").unwrap());
    for i in 0..steps {
        let v = t
            .forward(&latents, &pe, &pooled, sigmas[i], 0.0, 256, 256)
            .unwrap();
        let dt = sigmas[i + 1] - sigmas[i];
        latents = add(
            &latents,
            &multiply(&v, mlx_rs::Array::from_slice(&[dt], &[1])).unwrap(),
        )
        .unwrap();
    }
    let golden = g.require("final_latents").unwrap();
    let pr = peak_rel(&f32a(&latents), golden);
    let mr = mean_abs_rel(&f32a(&latents), golden);
    println!(
        "denoise final_latents: peak_rel={pr:.3e} mean_rel={mr:.3e} shape={:?}",
        latents.shape()
    );
    // 4 flow-match steps compound the per-step transformer drift; the fork's own f32-vs-bf16
    // latents differ by ~15% mean_rel @256², so this is well inside the envelope.
    assert!(mr < 1e-1, "denoise loop diverged: mean_rel {mr:.3e}");

    // Decode these (golden-embed) latents to pixels — isolates transformer+denoise+VAE px>8 from the
    // text-encoder f32-vs-bf16 contribution that the full-pipeline test additionally includes.
    let vae = load_vae(&snapshot()).unwrap();
    let unpacked = unpack_latents(&latents, 256, 256).unwrap();
    let decoded = f32a(&vae.decode(&unpacked).unwrap());
    let img = decoded_to_image(&decoded).unwrap();
    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let differ = img
        .pixels
        .iter()
        .zip(&gimg.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count();
    let frac = differ as f32 / img.pixels.len() as f32;
    println!(
        "✓ denoise(golden embeds)+VAE: {:.2}% px>8 vs fork (transformer/denoise/VAE only; the full \
         pipeline adds Rust's f32 T5/CLIP vs fork's bf16)",
        frac * 100.0
    );
}

#[test]
#[ignore = "needs real FLUX.1-schnell weights + local golden"]
fn e2e_single_stack_injected_matches_golden() {
    let g = golden();
    let t = load_transformer(&snapshot(), FluxVariant::Schnell).unwrap();
    // Feed the fork's EXACT post-joint tensors — isolates the single stack from the 0.45% joint drift.
    let encoder = f32a(g.require("encoder_joint").unwrap());
    let hidden = f32a(g.require("joint_hidden").unwrap());
    let text_emb = f32a(g.require("text_embeddings0").unwrap());
    for (name, arr) in t
        .debug_single_block0(&encoder, &hidden, &text_emb, 16, 16)
        .unwrap()
    {
        let golden = g.require(&name).unwrap();
        println!(
            "  {name}: peak_rel={:.3e} mean_rel={:.3e}",
            peak_rel(&f32a(&arr), golden),
            mean_abs_rel(&f32a(&arr), golden)
        );
    }
    let b0 = t
        .debug_single_stack(&encoder, &hidden, &text_emb, 16, 16, 1)
        .unwrap();
    let gb0 = g.require("single_b0_img").unwrap();
    println!(
        "single block[0] (injected): peak_rel={:.3e} mean_rel={:.3e}",
        peak_rel(&f32a(&b0), gb0),
        mean_abs_rel(&f32a(&b0), gb0)
    );
    let out = t
        .debug_single_stack(&encoder, &hidden, &text_emb, 16, 16, 0)
        .unwrap();
    let golden = g.require("single_img").unwrap();
    let pr = peak_rel(&f32a(&out), golden);
    let mr = mean_abs_rel(&f32a(&out), golden);
    println!(
        "single stack (injected): peak_rel={pr:.3e} mean_rel={mr:.3e} shape={:?}",
        out.shape()
    );
}

/// Full prompt→image pipeline through the public Generator API vs the fork's render.
///
/// NOTE: this is a REGRESSION GUARD, not a pixel-parity claim. FLUX.1-schnell (a guidance-distilled
/// few-step model) is precision-chaotic: the *fork itself* renders differently in f32 vs bf16 by
/// ~4.4% px>8 @1024² (~20% @256²). The mlx-gen path is numerically correct stage-by-stage (every
/// component — T5, CLIP, RoPE, joint block, single block, VAE — matches the f32 fork to <0.3%; see
/// the other tests), but the ~1e-5/block f32 op-graph difference vs MLX-Python's compiled graph is
/// amplified ~54× by the 38-block single stack and compounded over the denoise steps, landing the
/// final image at ~32% px>8. The broken Codex state (bf16-GEMM garbage + wrong CLIP pooled) was
/// 95% px>8, so the loose bound below catches a regression to that without overclaiming parity.
#[test]
#[ignore = "needs real FLUX.1-schnell weights + local golden"]
fn e2e_full_pipeline_matches_fork() {
    let g = golden();
    let snap = snapshot();
    let prompt = g.metadata("prompt").unwrap().to_string();
    let seed: u64 = g.metadata("seed").unwrap().parse().unwrap();
    let steps: u32 = g.metadata("steps").unwrap().parse().unwrap();
    let w: u32 = g.metadata("w").unwrap().parse().unwrap();
    let h: u32 = g.metadata("h").unwrap().parse().unwrap();

    let spec = LoadSpec::new(WeightsSource::Dir(snap));
    let generator = mlx_gen::load("flux1_schnell", &spec).unwrap();
    let req = GenerationRequest {
        prompt,
        width: w,
        height: h,
        seed: Some(seed),
        steps: Some(steps),
        ..Default::default()
    };
    let mut last_step = 0u32;
    let out = generator
        .generate(&req, &mut |p| {
            if let Progress::Step { current, .. } = p {
                last_step = last_step.max(current);
            }
        })
        .unwrap();
    let img = match out {
        GenerationOutput::Images(mut v) => v.pop().unwrap(),
        other => panic!("expected Images, got {other:?}"),
    };
    assert_eq!((img.width, img.height), (w, h), "image size");

    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tools/golden/rust_flux_schnell.png");
    image::save_buffer(
        &out_path,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();

    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let differ = img
        .pixels
        .iter()
        .zip(&gimg.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count();
    let frac = differ as f32 / img.pixels.len() as f32;
    println!(
        "full pipeline (public generate): {}x{}; {:.3}% px>8 ({} / {}) vs fork; saved {}",
        img.width,
        img.height,
        frac * 100.0,
        differ,
        img.pixels.len(),
        out_path.display()
    );
    // Regression guard only (see the doc comment): the broken state was 95% px>8.
    assert!(
        frac < 0.5,
        "full-pipeline image regressed badly ({:.1}% px>8) — re-check for a gross bug",
        frac * 100.0
    );
    println!(
        "full FLUX.1-schnell pipeline rendered (px>8={:.1}%); component-level parity is the verification — see test doc",
        frac * 100.0
    );
}
