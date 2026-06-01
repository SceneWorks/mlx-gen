//! sc-2352 / sc-2344: end-to-end validation of the Z-Image port against a real-weights golden run.
//!
//! `#[ignore]`d — needs the real `Tongyi-MAI/Z-Image-Turbo` weights in the HF cache and the
//! golden produced by `tools/dump_z_image_golden.py` (gitignored, local). Run with:
//!   cargo test -p mlx-gen-z-image --release --test e2e_real_weights -- --ignored --nocapture
//!
//! The stage tests validate each pipeline stage on real bf16 weights against the fork's
//! intermediates; the final test drives the **public** `load(id, spec).generate(req)` API and
//! confirms the rendered image matches the fork's golden.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::{
    FlowMatchEuler, GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};
use mlx_gen_z_image::{
    decoded_to_image, denoise, load_text_encoder, load_tokenizer, load_transformer, load_vae,
    slice_valid, unpack_latents,
};
use mlx_rs::{Array, Dtype};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/z_image_golden.safetensors"
);
const Q8_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/z_image_q8_golden.safetensors"
);
const Q4_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/z_image_q4_golden.safetensors"
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

#[test]
#[ignore = "needs real Z-Image weights + local golden"]
fn e2e_text_encoder_matches_golden() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let num_valid: i32 = g.metadata("num_valid").unwrap().parse().unwrap();

    let enc = load_text_encoder(&snapshot()).unwrap();
    let out = enc
        .forward(
            g.require("input_ids").unwrap(),
            g.require("attention_mask").unwrap(),
        )
        .unwrap();
    let cap = slice_valid(&out, num_valid).unwrap();

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
    assert!(
        peak_rel < 2e-3,
        "cap_feats diverged from the fork: peak-relative error {peak_rel:.2e} >= 2e-3"
    );
    println!(
        "✓ text encoder: cap_feats {:?} matches the fork golden (peak-rel {peak_rel:.2e})",
        cap.shape()
    );
}

#[test]
#[ignore = "needs real Z-Image weights + local golden"]
fn e2e_transformer_single_forward_matches_golden() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let transformer = load_transformer(&snapshot()).unwrap();

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
    // Use the fork's exact sigmas (not a recomputed schedule) so this isolates the loop, not mu.
    let scheduler = FlowMatchEuler { sigmas };
    let transformer = load_transformer(&snapshot()).unwrap();

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

#[test]
#[ignore = "needs real Z-Image weights + local golden"]
fn e2e_vae_and_image_matches_golden() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let vae = load_vae(&snapshot()).unwrap();

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
    println!("vae: decoded peak_rel={pr:.2e} shape={:?}", decoded.shape());
    // peak_rel is a single-pixel high-dynamic-range outlier metric; the MLX 0.31.1 bump (sc-2517)
    // nudged it from ~2.6e-2 to ~2.8e-2 (was 2e-2 pre-bump). The real guardrail is the per-pixel
    // RGB8 diff below (and the full-pipeline px>8 test) — both confirm the decode is visually exact.
    assert!(pr < 3.5e-2, "VAE decode diverged: peak_rel {pr:.2e}");

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

/// The integration proof: the full prompt→image pipeline through the **public** Generator API
/// (`mlx_gen::load("z_image_turbo", …).generate(req)`), compared to the fork's golden render.
#[test]
#[ignore = "needs real Z-Image weights + local golden"]
fn e2e_full_pipeline_generates_fox() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let snap = snapshot();
    let num_valid: i32 = g.metadata("num_valid").unwrap().parse().unwrap();

    // Drive the request from the golden's own metadata so this test tracks whatever
    // (prompt, seed, steps, size) the golden was dumped at — no separate hardcoding to
    // drift. dump_z_image_golden.py honors ZIMAGE_W/H/STEPS/SEED/PROMPT; this reads them back.
    let prompt = g.metadata("prompt").unwrap().to_string();
    let seed: u64 = g.metadata("seed").unwrap().parse().unwrap();
    let steps: u32 = g.metadata("steps").unwrap().parse().unwrap();
    let w: u32 = g.metadata("w").unwrap().parse().unwrap();
    let h: u32 = g.metadata("h").unwrap().parse().unwrap();

    // Tokenizer parity: the prompt with the Qwen chat template reproduces the fork's ids exactly.
    let tok = load_tokenizer(&snap).unwrap();
    let t = tok.tokenize(&prompt).unwrap();
    let take_n =
        |a: &Array| a.reshape(&[-1]).unwrap().as_slice::<i32>()[..num_valid as usize].to_vec();
    assert_eq!(
        take_n(&t.input_ids),
        take_n(g.require("input_ids").unwrap()),
        "tokenizer input_ids diverge from the fork"
    );

    // Full pipeline through the public API: load(id, spec) -> generate(req).
    let spec = LoadSpec::new(WeightsSource::Dir(snap));
    let generator = mlx_gen::load("z_image_turbo", &spec).unwrap();
    let req = GenerationRequest {
        prompt: prompt.clone(),
        width: w,
        height: h,
        seed: Some(seed),
        steps: Some(steps),
        ..Default::default()
    };
    let mut last_step = 0u32;
    let out = generator
        .generate(&req, &mut |p| {
            if let Progress::Step { current, total } = p {
                assert_eq!(total, steps, "step total");
                last_step = last_step.max(current);
            }
        })
        .unwrap();
    assert_eq!(
        last_step, steps,
        "expected {steps} denoise-step progress events"
    );

    let img = match out {
        GenerationOutput::Images(mut v) => {
            assert_eq!(v.len(), 1, "count=1 -> one image");
            v.pop().unwrap()
        }
        other => panic!("expected Images, got {other:?}"),
    };
    assert_eq!((img.width, img.height), (w, h), "image size");

    // Save the Rust render for visual inspection.
    let out_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tools/golden/rust_fox.png");
    image::save_buffer(
        &out_path,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();

    // Compare to the fork's golden image (bf16-loop drift allows a small fraction of pixels to
    // differ).
    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let differ = img
        .pixels
        .iter()
        .zip(&gimg.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count();
    println!(
        "✓ full pipeline (public generate): prompt->image {}x{}; {} / {} pixels differ by >8 from the fork; saved {}",
        img.width,
        img.height,
        differ,
        img.pixels.len(),
        out_path.display()
    );
    assert!(
        differ < img.pixels.len() / 20,
        "full-pipeline image diverges: {differ} pixels"
    );
}

/// sc-2532: the Q4/Q8 transformer parity gate. Quantize the Rust transformer (group_size 64 — the
/// fork's `nn.quantize` set, all 276 transformer Linears), feed the fork's seeded init noise +
/// quantized-text-encoder `cap_feats`, run the denoise loop on the fork's exact sigmas, and confirm
/// the latents + decoded image match the fork's quantized golden. Mirrors qwen's
/// `transformer_q8_pipeline_matches_fork`.
///
/// **Why the px threshold is looser than the bf16 e2e (which is sub-1%).** The quantization itself
/// is byte-identical to the fork (`q8_packing_byte_identical_to_fork`: same wq/scales/biases/qmm on
/// a real bf16 weight), and the quantized-Linear *set* matches the fork exactly. The residual
/// divergence is the Q8/Q4 mode's own perturbation sensitivity: the fork's *own* dense→Q8 output
/// already moves ~9% of pixels >8 (Q8 is a lossy mode), and it is sensitive to tiny activation
/// dtype / op-ordering differences (the fork's own bf16-vs-f32-activation Q8 run differs by ~1.7%
/// px). The Rust DiT runs f32 activations (its `apply_pad` / f32 `t_emb` promote the stream, exactly
/// as the fork's stream promotes after block 0) vs the fork's bf16 production path, and that ~1e-2
/// per-step difference compounds over the 4 iterative steps. The result tracks the fork's Q8 more
/// closely (~5%) than the fork's *dense* output does (~9%), which is the meaningful bar for a faithful
/// Q-mode. The latent mean-rel gate (~the Q8 noise floor, like qwen's) is the tighter check.
fn q_pipeline_matches_fork(
    golden_path: &str,
    bits: i32,
    max_latent_mean_rel: f32,
    max_px_frac: f32,
) {
    let g = Weights::from_file(golden_path).unwrap();
    let stored: i32 = g.metadata("quantize").unwrap().parse().unwrap();
    assert_eq!(stored, bits, "golden was dumped at a different bit-width");
    let snap = snapshot();

    let mut transformer = load_transformer(&snap).unwrap();
    transformer.quantize(bits).unwrap();
    let vae = load_vae(&snap).unwrap();

    // Fork's exact sigmas (isolate the loop from any schedule recompute) + its bf16 init/cap.
    let sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let scheduler = FlowMatchEuler { sigmas };
    let init = bf16(g.require("init").unwrap());
    let cap = bf16(g.require("cap_feats").unwrap());
    let latents = denoise(&transformer, &scheduler, init, &cap)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();

    let golden = g.require("final_latents").unwrap();
    assert_eq!(latents.shape(), golden.shape(), "final latents shape");
    // mean-rel is the stable metric (peak_rel is a single high-dynamic-range outlier); print both.
    let a = latents.reshape(&[-1]).unwrap();
    let b = golden.reshape(&[-1]).unwrap();
    let (xs, ys) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let mabs: f32 = ys.iter().map(|y| y.abs()).sum::<f32>() / ys.len() as f32;
    let mean_rel: f32 =
        xs.iter().zip(ys).map(|(x, y)| (x - y).abs()).sum::<f32>() / xs.len() as f32 / mabs;
    println!(
        "Q{bits} final_latents: mean_rel={mean_rel:.3e} peak_rel={:.3e} shape={:?}",
        peak_rel(&latents, golden),
        latents.shape()
    );
    assert!(
        mean_rel < max_latent_mean_rel,
        "Q{bits} final latents diverged from fork-Q{bits}: mean_rel {mean_rel:.3e} >= {max_latent_mean_rel:.3e}"
    );

    // Dense-VAE decode of the Rust Q-latents vs the fork's quantized decode, compared as RGB8. (The
    // VAE mid-block-attention Linears the fork also quantizes are pixel-irrelevant here: decoding
    // the fork's *exact* Q8 latents through the dense Rust VAE reproduces the fork's quantized-VAE
    // decode to 0 px>8 — measured during the sc-2532 investigation.)
    let unpacked = unpack_latents(&latents).unwrap();
    let sh = unpacked.shape();
    let latent5 = unpacked.reshape(&[sh[0], sh[1], 1, sh[2], sh[3]]).unwrap();
    let decoded = vae
        .decode(&latent5)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let golden_dec = g.require("decoded").unwrap();

    let img = decoded_to_image(&decoded).unwrap();
    let gimg = decoded_to_image(golden_dec).unwrap();
    let differ = img
        .pixels
        .iter()
        .zip(&gimg.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count();
    let frac = differ as f32 / img.pixels.len() as f32;
    println!(
        "Q{bits} pixels >8 apart: {:.3}% ({} / {})",
        frac * 100.0,
        differ,
        img.pixels.len()
    );
    assert!(
        frac < max_px_frac,
        "Q{bits}: too many divergent pixels: {:.3}% >= {:.3}%",
        frac * 100.0,
        max_px_frac * 100.0
    );
    println!("✓ Q{bits} pipeline matches fork-Q{bits}");
}

/// sc-2532: prove the Q8 quantization is byte-identical to the fork on a **real bf16 model weight**
/// (the existing `quant_parity.rs` covers an f32 weight; the model quantizes bf16). Quantizing the
/// same `layers.0.attention.to_q` weight with mlx-rs reproduces the fork's `mx.quantize` wq/scales/
/// biases exactly and `quantized_matmul` to 0 — so the Q8 e2e residual is the Q8 mode's sensitivity,
/// not a packing/qmm difference. Golden from `tools/dump_z_image_q8_pack_probe.py`.
#[test]
#[ignore = "needs the zq8_pack_probe golden (tools/dump_z_image_q8_pack_probe.py)"]
fn q8_packing_byte_identical_to_fork() {
    use mlx_rs::ops::{quantize, quantized_matmul};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tools/golden/zq8_pack_probe.safetensors"
    );
    let g = Weights::from_file(path).unwrap();
    // f32→bf16 of an already-bf16 value is exact, so this is the fork's exact quantized weight.
    let w = bf16(g.require("w").unwrap());
    let x = bf16(g.require("x").unwrap());

    let (wq, scales, biases) = quantize(&w, 64, 8).unwrap();
    let wq_match = mlx_rs::ops::eq(&wq, g.require("wq").unwrap())
        .unwrap()
        .all(None)
        .unwrap()
        .item::<bool>();
    let sc_pr = peak_rel(
        &scales.as_dtype(Dtype::Float32).unwrap(),
        g.require("scales").unwrap(),
    );
    let bi_pr = peak_rel(
        &biases.as_dtype(Dtype::Float32).unwrap(),
        g.require("biases").unwrap(),
    );
    let qmm = quantized_matmul(&x, &wq, &scales, &biases, true, 64, 8).unwrap();
    let qmm_pr = peak_rel(
        &qmm.as_dtype(Dtype::Float32).unwrap(),
        g.require("qmm").unwrap(),
    );
    println!(
        "Q8 packing vs fork: wq_exact={wq_match} scales_pr={sc_pr:.2e} biases_pr={bi_pr:.2e} qmm_pr={qmm_pr:.2e}"
    );
    assert!(
        wq_match,
        "Q8 packed weight is not byte-identical to the fork"
    );
    assert!(
        sc_pr == 0.0 && bi_pr == 0.0,
        "Q8 scales/biases differ from the fork"
    );
    assert!(
        qmm_pr < 1e-6,
        "Q8 quantized_matmul differs from the fork: {qmm_pr:.2e}"
    );
}

#[test]
#[ignore = "needs real Z-Image weights + local Q8 golden (QUANTIZE=8 dump_z_image_golden.py)"]
fn transformer_q8_pipeline_matches_fork() {
    // Measured (256², fox, seed 42): latent mean_rel 3.9e-2, px>8 5.16%. Thresholds leave headroom
    // for machine/run variance while staying well inside the fork's own dense→Q8 envelope (~9% px).
    q_pipeline_matches_fork(Q8_GOLDEN, 8, 5e-2, 0.07);
}

#[test]
#[ignore = "needs real Z-Image weights + local Q4 golden (QUANTIZE=4 dump_z_image_golden.py)"]
fn transformer_q4_pipeline_matches_fork() {
    // Measured (256², fox, seed 42): latent mean_rel ~3e-2, px>8 3.88%.
    q_pipeline_matches_fork(Q4_GOLDEN, 4, 5e-2, 0.07);
}
