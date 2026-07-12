//! Real-weights LoRA/LoKr **training** tests for mlx-gen-anima (sc-10522). `#[ignore]`d and
//! real-weights-gated — they need the licensed `circlestone-labs/Anima` base snapshot in the HF cache
//! and Metal. Run single-threaded (heavy Metal train + gen in one process):
//!
//!   cargo test -p mlx-gen-anima --release --test training -- --ignored --nocapture --test-threads=1
//!
//! Each test overfits a tiny synthetic dataset (a handful of vivid-magenta images + a trigger
//! caption), trains an adapter, then proves it worked two ways: (1) a **fixed-batch velocity-MSE**
//! reduction on a training image (base model vs trained adapter, same σ/noise — a σ-noise-free overfit
//! signal, unlike the per-step training loss which is dominated by the random per-step σ), and (2) a
//! **visible output change** when the adapter is reloaded through the sc-10521 inference path. PNGs go
//! to `$ANIMA_TRAIN_OUT` (default `/tmp/anima_sc10522`). Knobs: `ANIMA_TRAIN_STEPS`, `ANIMA_TRAIN_EDGE`,
//! `ANIMA_GEN_STEPS`, `ANIMA_TRAIN_LR`.

use std::path::PathBuf;

use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::{random, Array, Dtype};

use mlx_gen::runtime::{AdapterKind, AdapterSpec, CancelFlag};
use mlx_gen::weights::Weights;
use mlx_gen::{
    LoadSpec, NetworkType, Progress, TrainingConfig, TrainingItem, TrainingProgress,
    TrainingRequest, WeightsSource,
};

use mlx_gen_anima::config::Variant;
use mlx_gen_anima::pipeline::{AnimaPipeline, GenOptions};
use mlx_gen_anima::training::load_trainer_base;

// -------------------------------------------------------------------------------------------------
// Fixtures
// -------------------------------------------------------------------------------------------------

/// Glob the Anima base snapshot's `split_files/` dir (DiT + VAE + TE); `None` if absent (test skips).
fn split_files() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--circlestone-labs--Anima/snapshots");
    std::fs::read_dir(&base)
        .ok()?
        .filter_map(|e| e.ok())
        .find_map(|e| {
            let p = e.path().join("split_files");
            p.join("diffusion_models").is_dir().then_some(p)
        })
}

fn out_dir() -> PathBuf {
    let d = PathBuf::from(
        std::env::var("ANIMA_TRAIN_OUT").unwrap_or_else(|_| "/tmp/anima_sc10522".into()),
    );
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}
fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Write `n` vivid-magenta training images (a distinctive style easy to overfit) + a trigger caption
/// each, into a fresh dataset dir; return the `TrainingItem`s. Each image is a saturated magenta field
/// with a small green marker at a per-image offset (so they aren't byte-identical).
fn tiny_magenta_dataset(n: usize) -> Vec<TrainingItem> {
    let dir = out_dir().join("dataset");
    std::fs::create_dir_all(&dir).unwrap();
    let mut items = Vec::new();
    for i in 0..n {
        let mut img = image::RgbImage::from_pixel(256, 256, image::Rgb([236u8, 26, 224])); // magenta
        let ox = 40 + (i as u32 % 3) * 50;
        let oy = 40 + (i as u32 / 3) * 50;
        for y in oy..(oy + 32).min(256) {
            for x in ox..(ox + 32).min(256) {
                img.put_pixel(x, y, image::Rgb([40, 220, 60]));
            }
        }
        let path = dir.join(format!("magenta_{i}.png"));
        img.save(&path).unwrap();
        items.push(TrainingItem {
            image_path: path,
            caption: "sksanima style, vivid magenta background, 1girl".into(),
            control_image_path: None,
        });
    }
    items
}

/// Base training config for a hard overfit (full 508-target surface: empty `lora_target_modules`).
fn overfit_config(network_type: NetworkType) -> TrainingConfig {
    TrainingConfig {
        rank: 16,
        alpha: 16.0,
        learning_rate: env_f32("ANIMA_TRAIN_LR", 1e-4),
        steps: env_u32("ANIMA_TRAIN_STEPS", 250),
        resolution: env_u32("ANIMA_TRAIN_EDGE", 512),
        seed: 0,
        network_type,
        decompose_factor: -1,
        save_every: 0,
        optimizer: "adamw".into(),
        ..Default::default()
    }
}

fn train_request(
    items: Vec<TrainingItem>,
    config: TrainingConfig,
    file_name: &str,
) -> TrainingRequest {
    TrainingRequest {
        items,
        config,
        output_dir: out_dir(),
        file_name: file_name.into(),
        trigger_words: vec!["sksanima".into()],
        cancel: CancelFlag::default(),
    }
}

/// Run training, returning `(adapter_path, first_loss, last_loss, min_loss)` (per-step losses are
/// σ-noisy; the rigorous convergence signal is [`velocity_mse`]).
fn run_training(req: &TrainingRequest) -> (PathBuf, f32, f32, f32) {
    let spec = LoadSpec::new(WeightsSource::Dir(
        split_files().expect("Anima base snapshot"),
    ));
    let mut trainer = load_trainer_base(&spec).expect("load anima trainer");
    let mut losses: Vec<f32> = Vec::new();
    let mut on_progress = |p: TrainingProgress| {
        if let TrainingProgress::Training { step, loss, .. } = p {
            losses.push(loss);
            if step == 1 || step % 50 == 0 {
                eprintln!("  [train] step {step:>4}  loss {loss:.5}");
            }
        }
    };
    let out = trainer.train(req, &mut on_progress).expect("train");
    let first = losses.first().copied().unwrap_or(f32::NAN);
    let last = losses.last().copied().unwrap_or(f32::NAN);
    let min = losses.iter().copied().fold(f32::INFINITY, f32::min);
    eprintln!(
        "  [train] {} steps: per-step loss {first:.5} → {last:.5} (min {min:.5})",
        out.steps
    );
    (out.adapter_path, first, last, min)
}

// -------------------------------------------------------------------------------------------------
// Fixed-batch velocity MSE — the σ-noise-free overfit signal
// -------------------------------------------------------------------------------------------------

/// Velocity-MSE of `pipeline` on a fixed (image, caption, σ, noise): encode the image → clean latent
/// x0, encode the caption → conditioner output `enc`, form `x_t = (1−σ)x0 + σ·noise`, run the DiT, and
/// return `mean((v − (noise − x0))²)` in f32. Deterministic — the σ and noise are fixed, so the ONLY
/// thing that varies between a base pipeline and an adapter-applied pipeline is the trained delta. A
/// lower value on the training image after training = the adapter learned to reconstruct it (overfit).
fn velocity_mse(pipeline: &AnimaPipeline, caption: &str, edge: u32, sigma: f32) -> f32 {
    let img = image::open(out_dir().join("dataset/magenta_0.png"))
        .unwrap()
        .to_rgb8();
    let core = mlx_gen::media::Image {
        width: img.width(),
        height: img.height(),
        pixels: img.into_raw(),
    };
    let nchw = mlx_gen_qwen_image::preprocess_init_image(&core, edge, edge).unwrap();
    let x0 = pipeline.components().vae.encode(&nchw).unwrap(); // [1,16,1,edge/8,edge/8]
    let enc = pipeline.encode_prompt(caption).unwrap(); // [1,512,1024] bf16
    let noise =
        random::normal::<f32>(x0.shape(), None, None, Some(&random::key(4242).unwrap())).unwrap();
    let one_minus = Array::from_slice(&[1.0 - sigma], &[1]);
    let s = Array::from_slice(&[sigma], &[1]);
    let x_t = add(
        multiply(&x0, &one_minus).unwrap(),
        multiply(&noise, &s).unwrap(),
    )
    .unwrap();
    let v = pipeline
        .components()
        .dit
        .forward(
            &x_t.as_dtype(Dtype::Bfloat16).unwrap(),
            &s,
            &enc,
            Dtype::Bfloat16,
        )
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let target = subtract(&noise, &x0).unwrap();
    let diff = subtract(&v, &target).unwrap();
    let mse = diff.square().unwrap().mean(None).unwrap();
    mlx_rs::transforms::eval([&mse]).unwrap();
    mse.item::<f32>()
}

// -------------------------------------------------------------------------------------------------
// Generation + image helpers
// -------------------------------------------------------------------------------------------------

fn generate(
    pipeline: &AnimaPipeline,
    prompt: &str,
    seed: u64,
    steps: usize,
) -> mlx_gen::media::Image {
    let opts = GenOptions {
        width: 512,
        height: 512,
        steps,
        guidance: 4.5,
        seed,
        sampler: None,
        scheduler: None,
    };
    let cancel = CancelFlag::default();
    let mut noop = |_p: Progress| {};
    pipeline
        .generate(prompt, "", Variant::Base, &opts, &cancel, &mut noop)
        .expect("generate")
}

fn save_png(img: &mlx_gen::media::Image, path: &PathBuf) {
    let buf =
        image::RgbImage::from_raw(img.width, img.height, img.pixels.clone()).expect("rgb buffer");
    buf.save(path).expect("save png");
}

/// Mean absolute per-channel pixel difference (0..255) between two equal-size RGB images.
fn mean_abs_diff(a: &mlx_gen::media::Image, b: &mlx_gen::media::Image) -> f32 {
    assert_eq!(a.pixels.len(), b.pixels.len());
    let sum: u64 = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
        .sum();
    sum as f32 / a.pixels.len() as f32
}

/// Mean magenta-ness = mean(R) + mean(B) − 2·mean(G); rises as an image tilts magenta.
fn magenta_score(img: &mlx_gen::media::Image) -> f32 {
    let (mut r, mut g, mut b) = (0u64, 0u64, 0u64);
    for px in img.pixels.chunks_exact(3) {
        r += px[0] as u64;
        g += px[1] as u64;
        b += px[2] as u64;
    }
    let n = (img.pixels.len() / 3) as f32;
    (r as f32 + b as f32 - 2.0 * g as f32) / n
}

/// `(n_lora_A_keys, has_nonzero_conditioner_B, has_nonzero_dit_B, has_alpha_key)`.
fn inspect_lora_file(path: &PathBuf) -> (usize, bool, bool, bool) {
    let w = Weights::from_file(path).expect("load saved lora");
    let keys: Vec<String> = w.keys().map(str::to_string).collect();
    let n_a = keys
        .iter()
        .filter(|k| k.ends_with(".lora_A.weight"))
        .count();
    let has_alpha = keys.iter().any(|k| k.ends_with(".alpha"));
    let nonzero = |k: &str| -> bool {
        let a = w.require(k).unwrap();
        let s = a
            .as_dtype(Dtype::Float32)
            .unwrap()
            .abs()
            .unwrap()
            .sum(None)
            .unwrap();
        mlx_rs::transforms::eval([&s]).unwrap();
        s.item::<f32>() > 0.0
    };
    let cond_nonzero = keys
        .iter()
        .filter(|k| k.contains("llm_adapter") && k.ends_with(".lora_B.weight"))
        .any(|k| nonzero(k));
    let dit_nonzero = keys
        .iter()
        .filter(|k| !k.contains("llm_adapter") && k.ends_with(".lora_B.weight"))
        .any(|k| nonzero(k));
    (n_a, cond_nonzero, dit_nonzero, has_alpha)
}

/// `(max |Δ|, key_mismatches)` between two saved LoRA adapter files over their tensors (compared f32).
/// A resumed run that reproduces the uninterrupted one has `max |Δ| ≈ 0` and no key mismatches.
fn adapter_max_abs_diff(a: &PathBuf, b: &PathBuf) -> (f32, usize) {
    let wa = Weights::from_file(a).expect("load adapter a");
    let wb = Weights::from_file(b).expect("load adapter b");
    let ka: std::collections::HashSet<String> = wa.keys().map(str::to_string).collect();
    let kb: std::collections::HashSet<String> = wb.keys().map(str::to_string).collect();
    let mut max = 0f32;
    for k in ka.intersection(&kb) {
        let ta = wa.require(k).unwrap().as_dtype(Dtype::Float32).unwrap();
        let tb = wb.require(k).unwrap().as_dtype(Dtype::Float32).unwrap();
        let d = subtract(&ta, &tb)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap();
        mlx_rs::transforms::eval([&d]).unwrap();
        max = max.max(d.item::<f32>());
    }
    (max, ka.symmetric_difference(&kb).count())
}

// -------------------------------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------------------------------

/// Full LoRA end-to-end: overfit → save (508 targets, conditioner moved, no alpha) → the trained
/// adapter reduces the fixed-batch velocity-MSE on the training image → reload through the inference
/// path (508 applied) → generate → the output visibly changes and tilts toward the overfit magenta.
#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot; SLOW (2B DiT train + gen)"]
fn train_lora_end_to_end() {
    if split_files().is_none() {
        eprintln!("skip: no Anima snapshot");
        return;
    }
    let edge = env_u32("ANIMA_TRAIN_EDGE", 512);
    let gen_steps = env_u32("ANIMA_GEN_STEPS", 12) as usize;
    let caption = "sksanima style, vivid magenta background, 1girl";
    let prompt = "sksanima style, a portrait of a girl";
    let src = WeightsSource::Dir(split_files().unwrap());

    let items = tiny_magenta_dataset(4);
    let req = train_request(
        items,
        overfit_config(NetworkType::Lora),
        "anima_sc10522_lora.safetensors",
    );
    let (adapter_path, first, last, min) = run_training(&req);
    assert!(first.is_finite() && last.is_finite(), "non-finite loss");

    // File convention: 508 targets, conditioner AND DiT both moved (non-zero B), no alpha key.
    let (n_a, cond_moved, dit_moved, has_alpha) = inspect_lora_file(&adapter_path);
    assert_eq!(
        n_a, 508,
        "expected 508 LoRA targets (448 DiT + 60 conditioner), got {n_a}"
    );
    assert!(
        cond_moved,
        "conditioner (llm_adapter) lora_B is all-zero — it did NOT train"
    );
    assert!(dit_moved, "DiT lora_B is all-zero — it did NOT train");
    assert!(!has_alpha, "shipped Anima convention carries NO alpha key");

    // Rigorous convergence: fixed-batch velocity-MSE on the training image, base vs trained.
    let base = AnimaPipeline::from_source(&src, Variant::Base).expect("base pipeline");
    let mse_base = velocity_mse(&base, caption, edge, 0.5);
    let img_base = generate(&base, prompt, 1234, gen_steps);
    drop(base);
    mlx_rs::memory::clear_cache();

    let mut lora = AnimaPipeline::from_source(&src, Variant::Base).expect("lora pipeline");
    let report = lora
        .apply_adapters(&[AdapterSpec::new(
            adapter_path.clone(),
            1.0,
            AdapterKind::Lora,
        )])
        .expect("apply lora");
    assert_eq!(
        report.applied, 508,
        "LoRA must reload with 508 targets, got {}",
        report.applied
    );
    let mse_lora = velocity_mse(&lora, caption, edge, 0.5);
    let img_lora = generate(&lora, prompt, 1234, gen_steps);

    let base_png = out_dir().join("lora_baseline.png");
    let lora_png = out_dir().join("lora_applied.png");
    save_png(&img_base, &base_png);
    save_png(&img_lora, &lora_png);

    let diff = mean_abs_diff(&img_base, &img_lora);
    let (mb, ml) = (magenta_score(&img_base), magenta_score(&img_lora));
    eprintln!("  [lora] per-step loss {first:.5}→{last:.5} (min {min:.5})");
    eprintln!("  [lora] fixed-batch velocity-MSE: base {mse_base:.5} → trained {mse_lora:.5}");
    eprintln!("  [lora] round-trip mean|Δpixel|={diff:.2}; magenta base {mb:.1} → lora {ml:.1}");
    eprintln!(
        "  [lora] PNGs: {} | {}",
        base_png.display(),
        lora_png.display()
    );

    assert!(mse_lora < mse_base * 0.9, "adapter did not reduce the training-image velocity-MSE meaningfully: base {mse_base:.5} → trained {mse_lora:.5}");
    assert!(
        diff > 3.0,
        "LoRA had no visible effect (mean|Δpixel| {diff:.2} ≤ 3)"
    );
    assert!(
        ml > mb,
        "LoRA did not push output toward the overfit magenta style ({mb:.1} → {ml:.1})"
    );
}

/// sc-10642 mid-run RESUME validation. Train `steps` with `save_every = K` (K < steps) so the run drops
/// a resume snapshot (optimizer state + 508 factors + `{step, update_idx}`) at step K; then a FRESH
/// trainer with `cfg.resume = true` restores that snapshot and runs K+1..steps. Because the snapshot
/// lands on an optimizer-update boundary (`gradient_accumulation = 1`), the resumed adapter must
/// reproduce the uninterrupted run's adapter to fp tolerance, and the resumed run must start at step K+1
/// (proving it restored the step count, not restarted at 0) with the full 508-target surface asserted.
#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot; SLOW (2B DiT train ×2)"]
fn train_resume_matches_uninterrupted() {
    if split_files().is_none() {
        eprintln!("skip: no Anima snapshot");
        return;
    }
    let steps = env_u32("ANIMA_RESUME_STEPS", 8);
    let save_every = env_u32("ANIMA_RESUME_SAVE_EVERY", 4);
    assert!(
        save_every > 0 && save_every < steps,
        "need a snapshot strictly before the final step (save_every {save_every} < steps {steps})"
    );
    let edge = env_u32("ANIMA_TRAIN_EDGE", 384);
    let make_cfg = |resume: bool| TrainingConfig {
        rank: 16,
        alpha: 16.0,
        learning_rate: 1e-4,
        steps,
        resolution: edge,
        seed: 7,
        network_type: NetworkType::Lora,
        decompose_factor: -1,
        save_every,
        resume,
        optimizer: "adamw".into(),
        ..Default::default()
    };
    let file_name = "anima_sc10642_resume.safetensors";
    let spec = LoadSpec::new(WeightsSource::Dir(split_files().unwrap()));

    // (1) Uninterrupted run — drops a step-K resume snapshot in output_dir + the final adapter.
    let req_u = train_request(tiny_magenta_dataset(4), make_cfg(false), file_name);
    let mut trainer_u = load_trainer_base(&spec).expect("load trainer");
    let mut u_min = u32::MAX;
    let out_u = {
        let mut on_p = |p: TrainingProgress| {
            if let TrainingProgress::Training { step, .. } = p {
                u_min = u_min.min(step);
            }
        };
        trainer_u
            .train(&req_u, &mut on_p)
            .expect("uninterrupted train")
    };
    assert_eq!(u_min, 1, "uninterrupted run starts at step 1");
    assert_eq!(out_u.steps, steps, "uninterrupted run completes all steps");
    drop(trainer_u);
    mlx_rs::memory::clear_cache();

    // Copy the uninterrupted adapter aside — the resume run overwrites the same `file_name` path.
    let uninterrupted_path = out_dir().join("anima_sc10642_uninterrupted.safetensors");
    std::fs::copy(&out_u.adapter_path, &uninterrupted_path).expect("copy uninterrupted adapter");

    // (2) Resume run — a fresh trainer restores the step-K snapshot and continues K+1..steps.
    let req_r = train_request(tiny_magenta_dataset(4), make_cfg(true), file_name);
    let mut trainer_r = load_trainer_base(&spec).expect("reload trainer");
    let (mut r_min, mut r_max) = (u32::MAX, 0u32);
    let out_r = {
        let mut on_p = |p: TrainingProgress| {
            if let TrainingProgress::Training { step, .. } = p {
                r_min = r_min.min(step);
                r_max = r_max.max(step);
            }
        };
        trainer_r.train(&req_r, &mut on_p).expect("resume train")
    };

    // (3) Assertions: resumed at K+1 (step count restored, NOT from 0), ran to the end, reproduced U.
    let (max_d, mismatched) = adapter_max_abs_diff(&uninterrupted_path, &out_r.adapter_path);
    eprintln!(
        "[sc-10642] resumed steps {r_min}..={r_max}; resume vs uninterrupted: max |Δ| = {max_d:e} \
         ({mismatched} key mismatches)"
    );
    assert_eq!(
        r_min,
        save_every + 1,
        "resumed run must start at step K+1 (=snapshot+1), not 0/1 — the step count was restored"
    );
    assert_eq!(r_max, steps, "resumed run reaches the final step");
    assert_eq!(out_r.steps, steps, "resume reports the absolute final step");
    assert_eq!(
        mismatched, 0,
        "resumed adapter has a different key set than the uninterrupted one"
    );
    assert!(
        max_d <= 1e-6,
        "resumed adapter must reproduce the uninterrupted run to tolerance (max |Δ| {max_d:e})"
    );
}

/// Full LoKr end-to-end: same as the LoRA test but `NetworkType::Lokr` with a `decompose_factor`; the
/// bare-key `lokr_*` file reloads through the sc-10521 LoKr path (508 applied), reduces the fixed-batch
/// MSE, and changes output.
#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot; SLOW (2B DiT train + gen)"]
fn train_lokr_end_to_end() {
    if split_files().is_none() {
        eprintln!("skip: no Anima snapshot");
        return;
    }
    let edge = env_u32("ANIMA_TRAIN_EDGE", 512);
    let gen_steps = env_u32("ANIMA_GEN_STEPS", 12) as usize;
    let caption = "sksanima style, vivid magenta background, 1girl";
    let prompt = "sksanima style, a portrait of a girl";
    let src = WeightsSource::Dir(split_files().unwrap());

    let items = tiny_magenta_dataset(4);
    let mut cfg = overfit_config(NetworkType::Lokr);
    cfg.decompose_factor = -1; // auto/balanced Kronecker split
    let req = train_request(items, cfg, "anima_sc10522_lokr.safetensors");
    let (adapter_path, first, last, min) = run_training(&req);
    assert!(first.is_finite() && last.is_finite(), "non-finite loss");

    // LoKr file carries the Kronecker factors for all 508 targets (lokr_w1 present per target).
    let w = Weights::from_file(&adapter_path).expect("load saved lokr");
    let n_w1 = w.keys().filter(|k| k.ends_with(".lokr_w1")).count();
    assert_eq!(n_w1, 508, "expected 508 LoKr targets, got {n_w1}");
    assert!(
        w.keys()
            .any(|k| k.contains("llm_adapter") && k.ends_with(".lokr_w1")),
        "conditioner (llm_adapter) LoKr factors missing"
    );

    let base = AnimaPipeline::from_source(&src, Variant::Base).expect("base pipeline");
    let mse_base = velocity_mse(&base, caption, edge, 0.5);
    let img_base = generate(&base, prompt, 1234, gen_steps);
    drop(base);
    mlx_rs::memory::clear_cache();

    let mut lokr = AnimaPipeline::from_source(&src, Variant::Base).expect("lokr pipeline");
    let report = lokr
        .apply_adapters(&[AdapterSpec::new(
            adapter_path.clone(),
            1.0,
            AdapterKind::Lokr,
        )])
        .expect("apply lokr");
    assert_eq!(
        report.applied, 508,
        "LoKr must reload with 508 targets, got {}",
        report.applied
    );
    let mse_lokr = velocity_mse(&lokr, caption, edge, 0.5);
    let img_lokr = generate(&lokr, prompt, 1234, gen_steps);

    let base_png = out_dir().join("lokr_baseline.png");
    let lokr_png = out_dir().join("lokr_applied.png");
    save_png(&img_base, &base_png);
    save_png(&img_lokr, &lokr_png);

    let diff = mean_abs_diff(&img_base, &img_lokr);
    let (mb, ml) = (magenta_score(&img_base), magenta_score(&img_lokr));
    eprintln!("  [lokr] per-step loss {first:.5}→{last:.5} (min {min:.5})");
    eprintln!("  [lokr] fixed-batch velocity-MSE: base {mse_base:.5} → trained {mse_lokr:.5}");
    eprintln!("  [lokr] round-trip mean|Δpixel|={diff:.2}; magenta base {mb:.1} → lokr {ml:.1}");
    eprintln!(
        "  [lokr] PNGs: {} | {}",
        base_png.display(),
        lokr_png.display()
    );

    assert!(mse_lokr < mse_base * 0.9, "LoKr did not reduce the training-image velocity-MSE: base {mse_base:.5} → trained {mse_lokr:.5}");
    assert!(
        diff > 3.0,
        "LoKr had no visible effect (mean|Δpixel| {diff:.2} ≤ 3)"
    );
}

/// Target-enumeration anchor (the mutation-check made permanent): the full trainable surface is
/// exactly 508 = 448 DiT + 60 conditioner. Loads only the base DiT+conditioner, no train.
#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot"]
fn trainable_surface_is_508_dit_plus_conditioner() {
    let Some(split) = split_files() else {
        eprintln!("skip: no Anima snapshot");
        return;
    };
    use mlx_gen::adapters::{prefixed_paths, AdaptableHost};
    use mlx_gen_anima::loader::AnimaComponents;
    let comps = AnimaComponents::load(&WeightsSource::Dir(split), Variant::Base).expect("load");
    let dit = comps.dit.adaptable_paths();
    let cond = prefixed_paths("llm_adapter", &comps.conditioner);
    eprintln!(
        "  DiT targets {}, conditioner targets {}",
        dit.len(),
        cond.len()
    );
    assert_eq!(
        dit.len(),
        448,
        "DiT target surface must be 448 (28 blocks × 16)"
    );
    assert_eq!(
        cond.len(),
        60,
        "conditioner target surface must be 60 (6 blocks × 10)"
    );
    assert_eq!(dit.len() + cond.len(), 508);
    assert!(cond.iter().all(|p| p.starts_with("llm_adapter.blocks.")));
}

/// sc-10641 — in-training preview sampling, real weights end-to-end. Runs a short magenta overfit with
/// previews enabled at a fixed cadence and asserts:
///   (1) INTERVAL — `TrainingProgress::Sample` events fire at EXACTLY the cadence steps (multiples of
///       `sample_every` within `[1, steps]`), `sample_prompts.len()` per cadence, 1-based `index`.
///   (2) NON-DEGENERATE — every preview is a correctly-sized RGB bitmap with real pixel variance (a
///       genuine denoise+decode, not an empty stub).
///   (3) REFLECTS ADAPTER STATE — prompt[0]'s final-cadence preview is more magenta than its first, and
///       more magenta than a base-model generation of the same prompt. That can only happen if the
///       preview samples the LIVE in-training adapter (the DiT AND the conditioner adapters, sc-10522);
///       a preview that reused a cached/base conditioner output could not track the overfit.
#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot; SLOW (2B DiT train + periodic preview gens)"]
fn train_emits_in_training_previews() {
    if split_files().is_none() {
        eprintln!("skip: no Anima snapshot");
        return;
    }
    let steps = env_u32("ANIMA_TRAIN_STEPS", 250);
    let sample_every = (steps / 5).max(1);
    let mut cfg = overfit_config(NetworkType::Lora);
    cfg.steps = steps;
    cfg.sample_every = sample_every;
    cfg.sample_steps = env_u32("ANIMA_PREVIEW_STEPS", 6);
    cfg.sample_guidance_scale = 4.5; // base is a CFG variant
    cfg.sample_prompts = vec![
        "sksanima style, a portrait of a girl".into(),
        "sksanima style, a landscape".into(),
    ];
    let edge = cfg.resolution; // 512 → bucketed to itself
    let n_prompts = cfg.sample_prompts.len() as u32;
    let prompt0 = cfg.sample_prompts[0].clone();
    let req = train_request(
        tiny_magenta_dataset(4),
        cfg,
        "anima_sc10641_preview.safetensors",
    );

    let spec = LoadSpec::new(WeightsSource::Dir(split_files().unwrap()));
    let mut trainer = load_trainer_base(&spec).expect("load anima trainer");
    type Sample = (u32, u32, u32, mlx_gen::media::Image);
    let mut samples: Vec<Sample> = Vec::new();
    let mut on_progress = |p: TrainingProgress| {
        if let TrainingProgress::Sample {
            step,
            index,
            total,
            image,
            ..
        } = p
        {
            samples.push((step, index, total, image));
        }
    };
    trainer.train(&req, &mut on_progress).expect("train");
    drop(trainer);
    mlx_rs::memory::clear_cache();

    // (1) INTERVAL — exactly the cadence multiples, each with `n_prompts` 1-based-indexed previews.
    let expected_steps: Vec<u32> = (1..=steps)
        .filter(|&s| s.is_multiple_of(sample_every))
        .collect();
    let got_steps: Vec<u32> = {
        let mut v: Vec<u32> = samples.iter().map(|s| s.0).collect();
        v.dedup();
        v
    };
    assert_eq!(
        got_steps, expected_steps,
        "previews must fire at exactly the cadence steps (every {sample_every})"
    );
    for step in &expected_steps {
        let at: Vec<&Sample> = samples.iter().filter(|s| s.0 == *step).collect();
        assert_eq!(
            at.len() as u32,
            n_prompts,
            "cadence {step}: expected {n_prompts} previews, got {}",
            at.len()
        );
        for (i, s) in at.iter().enumerate() {
            assert_eq!(s.1, i as u32 + 1, "cadence {step}: 1-based index");
            assert_eq!(s.2, n_prompts, "cadence {step}: total == prompt count");
        }
    }

    // (2) NON-DEGENERATE — right size + real pixel variance.
    for (step, index, _t, img) in &samples {
        assert_eq!(
            (img.width, img.height),
            (edge, edge),
            "preview step {step}/{index}: wrong size"
        );
        assert_eq!(img.pixels.len(), (edge * edge * 3) as usize);
        let (lo, hi) = img
            .pixels
            .iter()
            .fold((255u8, 0u8), |(lo, hi), &p| (lo.min(p), hi.max(p)));
        assert!(
            hi > lo,
            "preview step {step}/{index} is a flat constant image (min==max) — not a real render"
        );
    }

    // (3) REFLECTS ADAPTER STATE — prompt[0]'s magenta tilt grows first→last cadence, and the final
    // preview is more magenta than a base-model generation (the overfit target the LoRA is learning).
    let first_step = *expected_steps.first().unwrap();
    let last_step = *expected_steps.last().unwrap();
    let p0 = |step: u32| -> &mlx_gen::media::Image {
        &samples
            .iter()
            .find(|s| s.0 == step && s.1 == 1)
            .expect("prompt0 preview at cadence")
            .3
    };
    let (first, last) = (p0(first_step), p0(last_step));
    let (mf, ml) = (magenta_score(first), magenta_score(last));
    save_png(first, &out_dir().join("preview_first.png"));
    save_png(last, &out_dir().join("preview_last.png"));

    let base =
        AnimaPipeline::from_source(&WeightsSource::Dir(split_files().unwrap()), Variant::Base)
            .expect("base pipeline");
    let img_base = generate(&base, &prompt0, 1234, cfg_preview_steps());
    let mb = magenta_score(&img_base);
    save_png(&img_base, &out_dir().join("preview_base.png"));

    eprintln!(
        "[sc-10641] {} previews across {} cadences; prompt0 magenta: base {mb:.1}, first(step {first_step}) {mf:.1} → last(step {last_step}) {ml:.1}",
        samples.len(),
        expected_steps.len()
    );
    assert!(
        ml > mf,
        "previews must track the in-training adapter: prompt0 should tilt toward the overfit magenta \
         from the first cadence ({mf:.1}) to the last ({ml:.1})"
    );
    assert!(
        ml > mb,
        "the final preview must reflect the trained adapter (more magenta than the base model: \
         base {mb:.1} vs final preview {ml:.1})"
    );
}

/// Preview denoise-step count used by the base-model comparison generation in the preview test.
fn cfg_preview_steps() -> usize {
    env_u32("ANIMA_PREVIEW_STEPS", 6) as usize
}
