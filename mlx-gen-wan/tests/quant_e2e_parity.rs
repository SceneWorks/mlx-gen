//! sc-2682 **end-to-end** Q4/Q8 MoE parity gate (`#[ignore]` — needs the 54 GB converted A14B
//! checkpoint). The quantized twin of `s6_real_parity.rs`: it runs the genuine
//! `Wan14b::generate` chain with **both experts quantized independently** (`WanTransformer::quantize`,
//! the per-expert requirement) and compares the final latents + decoded video against a reference-Q
//! golden dumped from `mlx_video` on the same converted weights + injected noise
//! (`tools/dump_quant_e2e_fixtures.py`). Verifies Q4 **and** Q8 (per [[false-green-gates-mask-descope]]).
//!
//! The transformer-isolated forward is already **bit-exact** (`quant_parity.rs`, max|Δ|=0.0 at Q4 AND
//! Q8) — that is the decisive proof the quantize is byte-faithful. This gate proves the wired
//! per-expert MoE path + the f32 scheduler/VAE hold up end-to-end at both bit-widths, comparing
//! Rust↔reference at *matched precision* against the measured dense cross-build floor (mean_rel, the
//! A14B e2e convention) plus a coarse px>8 outlier guard on the decoded uint8 frames.
//!
//! Run (after `dump_quant_e2e_fixtures.py` wrote the per-bits fixtures):
//! ```text
//! WAN_A14B_MODEL_DIR=~/.cache/mlx-gen-models/wan2_2_t2v_a14b_mlx_bf16 \
//! WAN_A14B_QUANT_FIXTURE=/tmp/wan_a14b_quant \
//!   cargo test -p mlx-gen-wan --test quant_e2e_parity -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_wan::config::WanModelConfig;
use mlx_gen_wan::pipeline::denoise_moe;
use mlx_gen_wan::scheduler::SolverKind;
use mlx_gen_wan::{decode_to_frames, load_tokenizer, Expert, Umt5Encoder, WanTransformer, WanVae};

fn env_path(var: &str) -> Option<PathBuf> {
    std::env::var_os(var).map(|s| PathBuf::from(shellexpand_home(&s.to_string_lossy())))
}

fn shellexpand_home(s: &str) -> String {
    if let Some(rest) = s.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return format!("{}/{rest}", home.to_string_lossy());
        }
    }
    s.to_string()
}

fn diff(got: &[f32], exp: &[f32]) -> (f32, f64) {
    let (mut max_abs, mut sum_abs, mut sum_ref) = (0f32, 0f64, 0f64);
    for (g, e) in got.iter().zip(exp.iter()) {
        let d = (g - e).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
        sum_ref += e.abs() as f64;
    }
    (max_abs, sum_abs / sum_ref.max(1e-9))
}

/// px>8 on the decoded video: map the raw [-1, 1] decode to uint8 (`round((x+1)/2·255)`, the
/// production frame conversion) and count pixels differing by more than 8 levels.
fn px_over_8(got: &[f32], exp: &[f32]) -> f64 {
    let to_u8 = |x: f32| -> i32 { (((x + 1.0) * 0.5 * 255.0).round().clamp(0.0, 255.0)) as i32 };
    let over = got
        .iter()
        .zip(exp.iter())
        .filter(|(g, e)| (to_u8(**g) - to_u8(**e)).abs() > 8)
        .count();
    100.0 * over as f64 / got.len() as f64
}

fn prepend1(shape: &[i32]) -> Vec<i32> {
    let mut s = vec![1];
    s.extend_from_slice(shape);
    s
}

/// `prequantized = false`: quantize the bf16 experts in-memory after load (the `spec.quantize`
/// path). `prequantized = true`: load a pre-quantized snapshot whose experts ship packed on disk
/// (`from_weights` builds them quantized via the `config.json` manifest — the `loading.py` path), at
/// reduced peak memory; no in-memory re-quantize. Both compare to the SAME reference-Q golden.
fn run(bits: i32, prequantized: bool) {
    let dir_env = if prequantized {
        format!("WAN_A14B_Q{bits}_DIR")
    } else {
        "WAN_A14B_MODEL_DIR".to_string()
    };
    let default_prequant = env_path("HOME")
        .map(|h| h.join(format!(".cache/mlx-gen-models/wan2_2_t2v_a14b_mlx_q{bits}")));
    let model_dir = match env_path(&dir_env).or(if prequantized { default_prequant } else { None })
    {
        Some(p) if p.join("config.json").exists() => p,
        _ => {
            eprintln!("skip: set {dir_env} to the converted A14B model dir");
            return;
        }
    };
    let base = match env_path("WAN_A14B_QUANT_FIXTURE") {
        Some(p) => p,
        None => {
            eprintln!("skip: set WAN_A14B_QUANT_FIXTURE (run dump_quant_e2e_fixtures.py first)");
            return;
        }
    };
    let fixture = {
        let name = format!(
            "{}_q{bits}.safetensors",
            base.file_name().unwrap().to_string_lossy()
        );
        base.with_file_name(name)
    };
    if !fixture.exists() {
        eprintln!(
            "skip: {} not found (run dump_quant_e2e_fixtures.py)",
            fixture.display()
        );
        return;
    }

    let cfg = WanModelConfig::from_model_dir(&model_dir).expect("read config.json");
    assert!(cfg.dual_model, "expected the dual-expert A14B config");
    let (low_gs, high_gs) = match cfg.sample_guide_scale {
        mlx_gen_wan::GuideScale::Dual { low, high } => (low, high),
        other => panic!("expected dual guide scale, got {other:?}"),
    };

    let fx = Weights::from_file(&fixture).expect("read fixture");
    let noise = fx.require("noise").unwrap();
    let exp_ctx = fx.require("context").unwrap();
    let exp_ctx_null = fx.require("context_null").unwrap();
    let exp_lat = fx.require("final_latents").unwrap();
    let exp_vid = fx.require("video").unwrap();

    let prompt = "a red fox trotting across a snowy meadow at sunrise, cinematic";
    let steps = 6usize;
    let shift = cfg.sample_shift;

    // --- Real-weight UMT5 encode (re-checks real T5 parity) ---
    let tokenizer = load_tokenizer(model_dir.join("tokenizer.json"), cfg.text_len).unwrap();
    let t5_w = Weights::from_file(model_dir.join("t5_encoder.safetensors")).expect("t5 weights");
    let enc = Umt5Encoder::from_weights(&t5_w, &cfg).expect("umt5");
    let context = enc.encode(&tokenizer, prompt).unwrap();
    let context_null = enc.encode(&tokenizer, &cfg.sample_neg_prompt).unwrap();
    let (cx_max, cx_mr) = diff(context.as_slice::<f32>(), exp_ctx.as_slice::<f32>());
    let (cn_max, cn_mr) = diff(
        context_null.as_slice::<f32>(),
        exp_ctx_null.as_slice::<f32>(),
    );
    println!("[Q{bits} t5 context]      max|Δ|={cx_max:.3e} mean_rel={cx_mr:.3e}");
    println!("[Q{bits} t5 context_null] max|Δ|={cn_max:.3e} mean_rel={cn_mr:.3e}");
    drop(enc);
    drop(t5_w);

    // --- Both experts, quantized independently, dual-expert MoE denoise ---
    // Pre-quantized snapshots ship packed → `from_weights` builds them quantized from the manifest
    // (`cfg.quantization`); the bf16 path quantizes in-memory after load. Either way both experts end
    // up Q4/Q8, independently.
    let low_w = Weights::from_file(model_dir.join("low_noise_model.safetensors")).expect("low");
    let high_w = Weights::from_file(model_dir.join("high_noise_model.safetensors")).expect("high");
    let mut low_dit = WanTransformer::from_weights(&low_w, &cfg).expect("low DiT");
    let mut high_dit = WanTransformer::from_weights(&high_w, &cfg).expect("high DiT");
    if prequantized {
        assert_eq!(
            cfg.quantization.map(|q| q.bits),
            Some(bits),
            "manifest bits"
        );
    } else {
        low_dit.quantize(bits, None).expect("quantize low");
        high_dit.quantize(bits, None).expect("quantize high");
    }

    let low = Expert {
        transformer: &low_dit,
        ctx_cond: low_dit.embed_text(&context).unwrap(),
        ctx_uncond: Some(low_dit.embed_text(&context_null).unwrap()),
        guidance: low_gs,
    };
    let high = Expert {
        transformer: &high_dit,
        ctx_cond: high_dit.embed_text(&context).unwrap(),
        ctx_uncond: Some(high_dit.embed_text(&context_null).unwrap()),
        guidance: high_gs,
    };
    let boundary_timestep = cfg.boundary * cfg.num_train_timesteps as f32;

    let latents = denoise_moe(
        &low,
        &high,
        boundary_timestep,
        SolverKind::UniPC,
        cfg.num_train_timesteps,
        steps,
        shift,
        noise,
        None,
        &mut |i| println!("  Q{bits} step {i}/{steps}"),
    )
    .expect("denoise_moe");

    assert_eq!(latents.shape(), exp_lat.shape(), "final latent shape");
    let (la_max, la_mr) = diff(latents.as_slice::<f32>(), exp_lat.as_slice::<f32>());
    println!(
        "[Q{bits} latents] shape={:?} max|Δ|={la_max:.3e} mean_rel={la_mr:.3e}",
        latents.shape()
    );
    drop(low_dit);
    drop(high_dit);
    drop(low_w);
    drop(high_w);

    // --- Real z16 VAE decode → px>8 on the decoded frames ---
    let vae_w = Weights::from_file(model_dir.join("vae.safetensors")).expect("vae");
    let vae = WanVae::from_weights(&vae_w).expect("vae");
    let video = vae
        .decode(&latents.reshape(&prepend1(latents.shape())).unwrap())
        .unwrap();
    assert_eq!(video.shape(), exp_vid.shape(), "video shape");
    let (vid_max, vid_mr) = diff(video.as_slice::<f32>(), exp_vid.as_slice::<f32>());
    let over8 = px_over_8(video.as_slice::<f32>(), exp_vid.as_slice::<f32>());
    println!(
        "[Q{bits} video]   shape={:?} max|Δ|={vid_max:.3e} mean_rel={vid_mr:.3e} px>8={over8:.4}%",
        video.shape()
    );

    // Exercise the product frame-assembly path too.
    let frames_u8 = decode_to_frames(&vae, &latents, None).unwrap();
    let images = mlx_gen_wan::frames_to_images(&frames_u8).unwrap();
    assert_eq!(images.len(), exp_vid.shape()[2] as usize, "frame count");

    // Gate against the *matched-precision* reference golden, floor-relative (the A14B e2e convention,
    // s6_real's 8e-2 mean_rel envelope) — NOT a flat px>8 bar ([[adapter-residual-bf16-sc2718]]).
    //
    // The decisive proof that the quantize is byte-faithful is the **bit-exact** 5B per-forward
    // (`quant_parity.rs`, max|Δ|=0.0 at Q4 AND Q8). So the e2e Rust-vs-ref residual here is just the
    // pre-existing **dense cross-build floor**, measured at this exact geometry:
    //   dense Rust↔ref: latents 1.82e-2, video 2.10e-2 mean_rel  (tools/dump_s6_real_fixtures.py).
    // Observed matched-precision residuals sit at that floor: Q8 latents 1.80e-2 / video 1.83e-2 /
    // px>8 0.56%; Q4 latents 3.86e-2 / video 3.18e-2 / px>8 3.26% (Q4 adds a small trajectory-
    // sensitivity increment — 4-bit is coarser). For contrast, the *quant effect itself* (how much
    // quantization changes the reference video vs dense) is an order larger and identical on both
    // sides: Q8 12% px>8 / 4.5e-2, Q4 49% px>8 / 2.35e-1 — that is the inherent quality cost, matched
    // byte-for-byte (the 5B forward proves it), not a port residual. A scale/predicate/wiring bug
    // would push Rust↔ref to O(1) (≫ the quant effect), caught decisively by every bound below.
    assert!(
        cx_mr < 1e-2,
        "Q{bits} t5 context diverged: mean_rel={cx_mr:.3e}"
    );
    assert!(
        cn_mr < 1e-2,
        "Q{bits} t5 context_null diverged: mean_rel={cn_mr:.3e}"
    );
    assert!(
        la_mr < 8e-2,
        "Q{bits} latents diverged: mean_rel={la_mr:.3e}"
    );
    assert!(
        vid_mr < 8e-2,
        "Q{bits} video diverged: mean_rel={vid_mr:.3e}"
    );
    // Coarse outlier guard: ~2× the Q4 floor (3.26%), well clear of an O(1) bug (≥12% px>8).
    assert!(over8 < 6.0, "Q{bits} video px>8 too high: {over8:.4}%");
}

// --- Load-time quant (bf16 snapshot + spec.quantize) ---
#[test]
#[ignore = "needs the 54 GB converted Wan2.2-T2V-A14B checkpoint + dump_quant_e2e_fixtures.py"]
fn wan_a14b_q4_e2e_matches_reference() {
    run(4, false);
}

#[test]
#[ignore = "needs the 54 GB converted Wan2.2-T2V-A14B checkpoint + dump_quant_e2e_fixtures.py"]
fn wan_a14b_q8_e2e_matches_reference() {
    run(8, false);
}

// --- Consume a pre-quantized snapshot (the low-peak production path: loads ~16 GB Q4 / 29 GB Q8
// instead of 54 GB bf16). `from_weights` builds the experts quantized from disk; same golden. ---
#[test]
#[ignore = "needs the pre-quantized A14B snapshot (dump_quant_snapshot.py) + dump_quant_e2e_fixtures.py"]
fn wan_a14b_q4_prequantized_e2e_matches_reference() {
    run(4, true);
}

#[test]
#[ignore = "needs the pre-quantized A14B snapshot (dump_quant_snapshot.py) + dump_quant_e2e_fixtures.py"]
fn wan_a14b_q8_prequantized_e2e_matches_reference() {
    run(8, true);
}
