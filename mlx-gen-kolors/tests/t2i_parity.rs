//! Kolors T2I end-to-end parity vs diffusers `KolorsPipeline` (sc-3094).
//!
//! `#[ignore]`d: needs the Kolors snapshot (TE+UNet+VAE) + the materialized `tokenizer.json` + the
//! golden from `tools/dump_kolors_t2i_golden.py`. Both gates feed diffusers' exact initial noise (the
//! Euler step is non-ancestral → deterministic, so the only RNG is the init noise).
//!
//! **Why the gate is structured this way.** The single U-Net forward matches diffusers to ~5e-4
//! (sc-3093) and the scheduler is bit-identical, but a *full* 8-step CFG-5 trajectory cannot be
//! bit-compared to a **torch** reference: the torch-CPU-vs-MLX-Metal per-step f32 floor (~5e-4)
//! compounds through the (chaotic) sampler — measured ~1.6×/step at CFG-1 (final mean-rel ~2.8e-2)
//! and ~10× worse under CFG-5. This is an inherent cross-backend property (it is exactly why the
//! repo's SDXL e2e gate uses a same-backend *MLX* reference, not torch), NOT a port defect. So the
//! correctness gate is the **deterministic early-step latent integration** (gate A) — which exercises
//! the whole loop: init scaling, `scale_model_input`, the U-Net (with `encoder_hid_proj` + 5632
//! add-embedding), the CFG combine, and the Euler step — and the full render (gate B) is a coherence
//! + cross-backend-delta report, not a bit gate.
//!
//! Run: `cargo test -p mlx-gen-kolors --release --test t2i_parity -- --ignored --nocapture`

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::DiffusionSampler;
use mlx_gen_kolors::sampler::KolorsEulerSampler;
use mlx_gen_kolors::unet::load_unet_kolors_dtype;
use mlx_gen_kolors::Kolors;
use mlx_rs::ops::{add, concatenate_axis, multiply, subtract};
use mlx_rs::{Array, Dtype};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/kolors_t2i_golden.safetensors"
);

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("KOLORS_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Kwai-Kolors--Kolors-diffusers/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn rel(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let (a, b) = (a.reshape(&[n]).unwrap(), b.reshape(&[n]).unwrap());
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-9);
    let mabs = (b.iter().map(|v| v.abs()).sum::<f32>() / b.len() as f32).max(1e-9);
    let max_d = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let mean_d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32;
    (max_d / peak, mean_d / mabs)
}

#[test]
#[ignore = "needs the Kolors snapshot + tokenizer.json + tools/golden/kolors_t2i_golden.safetensors"]
fn kolors_t2i_matches_diffusers() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let steps: usize = g.metadata("steps").unwrap().parse().unwrap();
    let cfg: f32 = g.metadata("cfg").unwrap().parse().unwrap();
    let h: i32 = g.metadata("h").unwrap().parse().unwrap();
    let w: i32 = g.metadata("w").unwrap().parse().unwrap();
    let init_noise = g.require("init_noise").unwrap();

    // ---- Gate A: deterministic early-step latent integration (the correctness gate). ----
    // Replays the denoise loop op-for-op against the dumped conditioning + init noise and checks the
    // step-0 / step-1 latents — the whole loop (init scaling, scale_model_input, U-Net, CFG, step),
    // before deep cross-backend accumulation sets in.
    let sampler = KolorsEulerSampler::kolors(steps, Dtype::Float32).unwrap();
    let unet = load_unet_kolors_dtype(&snapshot(), Dtype::Float32).unwrap();
    let cond = concatenate_axis(
        &[
            g.require("pos_context").unwrap(),
            g.require("neg_context").unwrap(),
        ],
        0,
    )
    .unwrap();
    let pooled = concatenate_axis(
        &[
            g.require("pos_pooled").unwrap(),
            g.require("neg_pooled").unwrap(),
        ],
        0,
    )
    .unwrap();
    let mut tid = Vec::new();
    for _ in 0..2 {
        tid.extend_from_slice(&[w as f32, h as f32, 0.0, 0.0, w as f32, h as f32]);
    }
    // _get_add_time_ids = (orig_h, orig_w, 0, 0, tgt_h, tgt_w); square here so order is moot.
    let time_ids = Array::from_slice(&tid, &[2, 6]);

    let mut x = sampler.scale_initial_noise(init_noise).unwrap();
    for i in 0..2usize {
        let x_in = sampler.scale_model_input(&x, i).unwrap();
        let x_unet = concatenate_axis(&[&x_in, &x_in], 0).unwrap();
        let eps = unet
            .forward(&x_unet, sampler.timestep(i), &cond, &pooled, &time_ids)
            .unwrap();
        let row = |k: i32| eps.take_axis(Array::from_slice(&[k], &[1]), 0).unwrap();
        let (text, ng) = (row(0), row(1));
        let cfg_eps = add(
            &ng,
            multiply(
                subtract(&text, &ng).unwrap(),
                Array::from_slice(&[cfg], &[1]),
            )
            .unwrap(),
        )
        .unwrap();
        x = sampler.step(&cfg_eps, &x, i).unwrap();
        x.eval().unwrap();
        let key = if i == 0 {
            "step0_latents"
        } else {
            "step1_latents"
        };
        let (p, m) = rel(&x, g.require(key).unwrap());
        println!("gate A step{i}: peak_rel={p:.3e} mean_rel={m:.3e}");
        // step0 ~4e-3, step1 ~1e-2 (single-forward 5e-4 floor × CFG-5, minimal accumulation).
        let (pt, mt) = if i == 0 {
            (1.5e-2, 6e-3)
        } else {
            (3.5e-2, 1.5e-2)
        };
        assert!(
            p < pt,
            "gate A step{i} peak_rel {p:.3e} exceeds {pt:.1e} (loop wiring bug)"
        );
        assert!(
            m < mt,
            "gate A step{i} mean_rel {m:.3e} exceeds {mt:.1e} (loop wiring bug)"
        );
    }
    println!(
        "✓ gate A: early-step latent integration matches diffusers (loop+scheduler+CFG correct)"
    );

    // ---- Gate B: full Rust pipeline render (coherence + cross-backend delta report). ----
    let kolors = Kolors::load(&snapshot(), Dtype::Float32).expect("load Kolors");
    let prompt = g.metadata("prompt").unwrap();
    let negative = g.metadata("negative").unwrap();
    let r_pos = kolors.encode(prompt).expect("encode pos");
    let r_neg = kolors.encode(negative).expect("encode neg");
    let r_latents = kolors
        .denoise_latents(
            init_noise,
            &r_pos,
            &r_neg,
            steps,
            cfg,
            h,
            w,
            &mlx_gen::CancelFlag::new(),
            &mut |_p| {},
        )
        .expect("denoise");
    let image = kolors.decode(&r_latents).expect("decode");

    let gi = g.require("image").unwrap();
    let n = gi.shape().iter().product::<i32>();
    let want: Vec<u8> = gi
        .reshape(&[n])
        .unwrap()
        .as_slice::<f32>()
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    assert_eq!(image.pixels.len(), want.len(), "image size");
    let mean_abs: f64 = image
        .pixels
        .iter()
        .zip(&want)
        .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs() as f64)
        .sum::<f64>()
        / want.len() as f64;
    let over8 = image
        .pixels
        .iter()
        .zip(&want)
        .filter(|(&a, &b)| (a as i32 - b as i32).abs() > 8)
        .count();
    let mean_g: f64 = want.iter().map(|&v| v as f64).sum::<f64>() / want.len() as f64;
    let mean_r: f64 =
        image.pixels.iter().map(|&v| v as f64).sum::<f64>() / image.pixels.len() as f64;
    println!(
        "gate B (full render): mean|Δ|={mean_abs:.2}/255, px>8={:.1}%, mean(rust)={mean_r:.1} vs mean(diffusers)={mean_g:.1}",
        over8 as f64 / want.len() as f64 * 100.0
    );
    // Coherence: the render is a real, non-degenerate image whose global brightness tracks the
    // reference (the cross-backend trajectory drift is pixel-level, not a structurally different
    // scene). A gross wiring bug (NaN / blank / wrong scene) blows past these.
    assert!(
        image.pixels.iter().any(|&p| p > 16) && image.pixels.iter().any(|&p| p < 239),
        "degenerate render"
    );
    assert!(
        (mean_r - mean_g).abs() < 20.0,
        "render brightness off by {:.1} (gross divergence)",
        (mean_r - mean_g).abs()
    );
    println!("✓ Kolors T2I full pipeline renders coherently (cross-backend px delta is chaos-limited, see note)");
}
