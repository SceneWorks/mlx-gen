//! sc-3839/sc-3840: real-weight e2e parity vs torch `diffusers` ChromaPipeline, f32 both sides.
//! Goldens = `tools/dump_chroma_e2e_golden.py {hd,base,flash}`. `#[ignore]` — each needs the
//! corresponding ~18GB snapshot; run with
//! `cargo test -p mlx-gen-chroma --test e2e_real_weights -- --ignored --nocapture`.
//!
//! HD is the comprehensive gate (masked T5 encode + single real-weight DiT forward + full image).
//! base/flash reuse the identical model path (validated on HD) and differ only in the sigma schedule
//! (beta / static-shift-1.0), so their gates are the full-generate image + final latents.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, LoadSpec, Progress, WeightsSource};
use mlx_gen_chroma::{encode_prompt, load_chroma, ChromaVariant};
use mlx_rs::ops::{abs, concatenate_axis, max, multiply, subtract, sum};
use mlx_rs::{Array, Dtype};

const PROMPT: &str = "a photograph of an astronaut riding a horse";
const NEG: &str = "";
const W: u32 = 256;
const H: u32 = 256;
const STEPS: u32 = 4;

fn hf_snapshot(repo: &str) -> PathBuf {
    let cache = std::env::var("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/huggingface/hub")
        });
    let snaps = cache.join(format!("models--lodestones--{repo}/snapshots"));
    std::fs::read_dir(&snaps)
        .unwrap_or_else(|_| panic!("snapshot not found under {}", snaps.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn peak_rel(got: &Array, golden: &Array) -> f32 {
    let d = max(abs(subtract(got, golden).unwrap()).unwrap(), None)
        .unwrap()
        .item::<f32>();
    let s = max(abs(golden).unwrap(), None).unwrap().item::<f32>();
    d / s
}

/// Relative L2 `‖got−golden‖₂ / ‖golden‖₂` — robust to single-element outliers (unlike peak-rel).
fn rel_l2(got: &Array, golden: &Array) -> f32 {
    let l2 = |a: &Array| -> f32 {
        sum(multiply(a, a).unwrap(), None)
            .unwrap()
            .item::<f32>()
            .sqrt()
    };
    l2(&subtract(got, golden).unwrap()) / l2(golden)
}

/// Fraction of decoded pixels differing from the golden image (`[1,3,H,W]` in `[-1,1]`) by > `thr`/255.
fn image_px_fraction(img: &mlx_gen::Image, golden: &Array, thr: f32) -> f32 {
    let gi: Vec<f32> = golden
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let n = (H * W) as usize;
    let mut bad = 0usize;
    for c in 0..3 {
        for p in 0..n {
            let gv = ((gi[c * n + p] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0);
            let mv = img.pixels[p * 3 + c] as f32; // Image is HWC RGB u8
            if (gv - mv).abs() > thr {
                bad += 1;
            }
        }
    }
    bad as f32 / (3 * n) as f32
}

/// Full-generate image parity for a variant: denoise from the golden's init latent, compare final
/// latents (rel-L2) + decoded image (px>8). Shared by base/flash; HD adds the tighter gates below.
fn run_image_parity(variant: ChromaVariant, repo: &str, fixture: &str, guidance: f32) {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let g = Weights::from_file(format!("{dir}/{fixture}.safetensors")).unwrap();
    let model = load_chroma(
        variant,
        &LoadSpec::new(WeightsSource::Dir(hf_snapshot(repo))),
    )
    .unwrap_or_else(|e| panic!("load {repo}: {e}"));

    let init = g.require("init_latents").unwrap();
    let mut nop = |_p: Progress| {};
    let cancel = CancelFlag::default();
    let final_latents = model
        .denoise(
            PROMPT,
            NEG,
            W,
            H,
            STEPS,
            guidance,
            init.clone(),
            &cancel,
            &mut nop,
        )
        .unwrap();
    let fl_l2 = rel_l2(&final_latents, g.require("final_latents").unwrap());
    let img = model.decode(&final_latents, W, H, None).unwrap();
    let f8 = image_px_fraction(&img, g.require("image").unwrap(), 8.0);
    eprintln!("[{repo}] final rel-L2 = {fl_l2:.4}  image px>8 = {f8:.4}");
    assert!(
        fl_l2 < 0.08,
        "{repo}: final latents diverged (rel-L2 {fl_l2})"
    );
    assert!(f8 < 0.08, "{repo}: decoded image diverged ({f8} px>8)");
}

#[test]
#[ignore = "needs the ~18GB Chroma1-HD snapshot"]
fn chroma_hd_e2e_matches_diffusers() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let g = Weights::from_file(format!("{dir}/chroma_e2e_hd.safetensors")).unwrap();
    let model = load_chroma(
        ChromaVariant::Hd,
        &LoadSpec::new(WeightsSource::Dir(hf_snapshot("Chroma1-HD"))),
    )
    .expect("load Chroma1-HD");

    // 1. masked T5 encode parity (sc-3838 numeric).
    let (pe, pm) = encode_prompt(model.tokenizer_ref(), model.t5_ref(), PROMPT).unwrap();
    let pe_rel = peak_rel(&pe, g.require("prompt_embeds").unwrap());
    eprintln!("prompt_embeds peak-rel = {pe_rel:.4}");
    assert!(pe_rel < 5e-2, "masked T5 prompt_embeds diverged: {pe_rel}");
    let (nege, _) = encode_prompt(model.tokenizer_ref(), model.t5_ref(), NEG).unwrap();
    eprintln!(
        "neg_embeds peak-rel = {:.4}",
        peak_rel(&nege, g.require("neg_embeds").unwrap())
    );
    let pm_diff = max(
        abs(subtract(&pm, g.require("prompt_mask").unwrap()).unwrap()).unwrap(),
        None,
    )
    .unwrap()
    .item::<f32>();
    assert_eq!(pm_diff, 0.0, "transformer text mask diverged");

    // 2. single real-weight DiT forward (tight) — feed the *golden* embeds to isolate the DiT.
    let golden_embeds = g.require("prompt_embeds").unwrap();
    let l = golden_embeds.shape()[1];
    let txt_ids = Array::from_slice(&vec![0f32; (l * 3) as usize], &[l, 3]);
    let si = ((H / 16) * (W / 16)) as i32;
    let ones = Array::ones::<f32>(&[1, si]).unwrap();
    let full_mask = concatenate_axis(&[g.require("prompt_mask").unwrap(), &ones], 1).unwrap();
    let noise_pred = model
        .transformer_ref()
        .forward(
            g.require("init_latents").unwrap(),
            golden_embeds,
            g.require("timestep").unwrap(),
            g.require("img_ids").unwrap(),
            &txt_ids,
            Some(&full_mask),
        )
        .unwrap();
    let np_rel = peak_rel(&noise_pred, g.require("noise_pred").unwrap());
    eprintln!("noise_pred peak-rel = {np_rel:.4}");
    assert!(np_rel < 5e-2, "single DiT forward diverged: {np_rel}");

    // 3. full true-CFG denoise + decode.
    run_image_parity(ChromaVariant::Hd, "Chroma1-HD", "chroma_e2e_hd", 4.0);
}

#[test]
#[ignore = "needs the ~18GB Chroma1-HD snapshot"]
fn chroma_hd_quant_bounded() {
    // sc-3841: Q8/Q4 over the DiT block linears. Measure the quant perturbation on a single forward
    // vs the dense golden noise_pred (the quant *effect*; there's no MLX quant reference for Chroma).
    use mlx_gen::Quant;
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let g = Weights::from_file(format!("{dir}/chroma_e2e_hd.safetensors")).unwrap();
    let golden_np = g.require("noise_pred").unwrap();
    let golden_embeds = g.require("prompt_embeds").unwrap();
    let l = golden_embeds.shape()[1];
    let txt_ids = Array::from_slice(&vec![0f32; (l * 3) as usize], &[l, 3]);
    let si = ((H / 16) * (W / 16)) as i32;
    let ones = Array::ones::<f32>(&[1, si]).unwrap();
    let full_mask = concatenate_axis(&[g.require("prompt_mask").unwrap(), &ones], 1).unwrap();

    // Chroma's DiT quantizes cleanly: measured Q8 ≈0.3% / Q4 ≈1.7% rel-L2 on a single forward.
    for (q, gate) in [(Quant::Q8, 0.015_f32), (Quant::Q4, 0.04_f32)] {
        let spec = LoadSpec::new(WeightsSource::Dir(hf_snapshot("Chroma1-HD"))).with_quant(q);
        let model = load_chroma(ChromaVariant::Hd, &spec).expect("load quantized Chroma1-HD");
        let np = model
            .transformer_ref()
            .forward(
                g.require("init_latents").unwrap(),
                golden_embeds,
                g.require("timestep").unwrap(),
                g.require("img_ids").unwrap(),
                &txt_ids,
                Some(&full_mask),
            )
            .unwrap();
        let rl = rel_l2(&np, golden_np);
        eprintln!("{q:?} noise_pred rel-L2 vs dense = {rl:.4}");
        assert!(rl < gate, "{q:?} quant perturbation too large: {rl}");
    }
}

#[test]
#[ignore = "needs the ~18GB Chroma1-Base snapshot"]
fn chroma_base_e2e_matches_diffusers() {
    // Base uses the beta sigma schedule (use_beta_sigmas).
    run_image_parity(ChromaVariant::Base, "Chroma1-Base", "chroma_e2e_base", 4.0);
}

/// sc-5514 / sc-5399: the per-step `eval` in the denoise loop makes the per-step `CancelFlag`
/// check actually interrupt a render. Trip the flag from a timer thread shortly after the render
/// starts (an external user cancel mid-denoise, exactly as the SceneWorks worker's cancel poll
/// does) and assert the run aborts early with `Error::Canceled`, having executed far fewer than the
/// requested steps. Before the fix the denoise built ONE lazy graph and returned `Ok` (the real
/// compute is deferred to VAE decode) before the async cancel was ever observed by the per-step
/// check — so it ran every step and never aborted, which this test catches via the `Err(Canceled)`
/// assertion. 256² so a step is cheap; 28 steps so an ~80ms cancel leaves most of them unrun.
#[test]
#[ignore = "needs the ~18GB Chroma1-Flash snapshot"]
fn chroma_flash_async_cancel_interrupts_denoise() {
    use std::time::{Duration, Instant};

    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let g = Weights::from_file(format!("{dir}/chroma_e2e_flash.safetensors")).unwrap();
    let init = g.require("init_latents").unwrap().clone();
    let model = load_chroma(
        ChromaVariant::Flash,
        &LoadSpec::new(WeightsSource::Dir(hf_snapshot("Chroma1-Flash"))),
    )
    .expect("load Chroma1-Flash");

    const CANCEL_STEPS: u32 = 28;
    let cancel = CancelFlag::default();
    let cancel_bg = cancel.clone();
    let bg = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(80));
        cancel_bg.cancel();
    });

    let mut steps_seen = 0u32;
    let mut on_p = |p: Progress| {
        if matches!(p, Progress::Step { .. }) {
            steps_seen += 1;
        }
    };
    let started = Instant::now();
    let result = model.denoise(
        PROMPT,
        NEG,
        W,
        H,
        CANCEL_STEPS,
        1.0,
        init,
        &cancel,
        &mut on_p,
    );
    let elapsed = started.elapsed();
    bg.join().unwrap();

    eprintln!(
        "[cancel] steps_seen={steps_seen}/{CANCEL_STEPS} elapsed={elapsed:?} canceled={}",
        result.is_err()
    );
    assert!(
        matches!(result, Err(Error::Canceled)),
        "an async mid-denoise cancel must abort with Error::Canceled (steps_seen={steps_seen}/{CANCEL_STEPS})"
    );
    assert!(
        steps_seen < CANCEL_STEPS,
        "cancel must stop the denoise early; ran all {steps_seen}/{CANCEL_STEPS} steps"
    );
}

#[test]
#[ignore = "needs the ~18GB Chroma1-Flash snapshot"]
fn chroma_flash_e2e_matches_diffusers() {
    // Flash is the few-step distilled model (static shift 1.0, CFG≈1). `guidance == 1.0` exercises the
    // single-forward path (F-095): `denoise` skips the negative T5 encode + negative DiT forward and
    // returns `pos` exactly, so matching the diffusers golden here guards that the skip is correct.
    // NB: this (and every golden) steps with flow-match **Euler**; the production Flash **Heun**
    // default (sc-5392) is gated same-backend by `chroma_flash_heun_path_gated` below.
    run_image_parity(
        ChromaVariant::Flash,
        "Chroma1-Flash",
        "chroma_e2e_flash",
        1.0,
    );
}

/// sc-6903 (F-005): production Chroma1-Flash defaults to the **Heun** sampler (sc-5392), but the
/// diffusers reference — and therefore every committed golden — steps with flow-match **Euler**
/// (`ChromaPipeline` has no Heun scheduler), so the Euler-forced `denoise` goldens never exercise the
/// Heun second-forward+average arm that production actually runs. No torch Heun reference can be
/// produced, so gate it **same-backend**: from the identical golden init, compare the production Heun
/// path against the (golden-validated) Euler path. The shared encode/sigma/RoPE/mask/CFG body is one
/// method post-dedup, so the Euler golden already covers it; this isolates the Heun-only step logic.
#[test]
#[ignore = "needs the ~18GB Chroma1-Flash snapshot"]
fn chroma_flash_heun_path_gated() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let g = Weights::from_file(format!("{dir}/chroma_e2e_flash.safetensors")).unwrap();
    let init = g.require("init_latents").unwrap().clone();
    let model = load_chroma(
        ChromaVariant::Flash,
        &LoadSpec::new(WeightsSource::Dir(hf_snapshot("Chroma1-Flash"))),
    )
    .expect("load Chroma1-Flash");

    let mut nop = |_p: Progress| {};
    let cancel = CancelFlag::default();
    // Euler == the validated golden trajectory (gated by chroma_flash_e2e_matches_diffusers); compute
    // it same-backend here so the Heun comparison cancels any port-vs-torch baseline offset.
    let euler = model
        .denoise(
            PROMPT,
            NEG,
            W,
            H,
            STEPS,
            1.0,
            init.clone(),
            &cancel,
            &mut nop,
        )
        .unwrap();
    // The production default for Flash: Some("heun") routes through ChromaSamplerKind::from_request.
    let heun = model
        .denoise_with_sampler_name(
            PROMPT,
            NEG,
            W,
            H,
            STEPS,
            1.0,
            init.clone(),
            Some("heun"),
            &cancel,
            &mut nop,
        )
        .unwrap();

    let d = rel_l2(&heun, &euler);
    eprintln!("[Chroma1-Flash] Heun-vs-Euler rel-L2 = {d:.5}");
    // Distinctness (load-bearing regression guard): Heun runs an extra forward + averages, so it MUST
    // differ from Euler. A future edit that silently collapses the Heun arm back to Euler trips this.
    assert!(
        d.is_finite() && d > 1e-4,
        "Heun must differ from Euler (second-forward arm not exercised); rel-L2 {d}"
    );
    // Coherence envelope: Heun is a higher-order refinement of the *same* velocity field, so it stays
    // in Euler's neighborhood — garbage/divergence (NaN, broken shared body) escapes this bound. 0.5
    // is a generous envelope; tighten to the observed magnitude on the first real-weight run.
    assert!(
        d < 0.5,
        "Heun diverged from Euler — not a coherent refinement; rel-L2 {d}"
    );
    // And it must still decode to a real image (catches a non-finite Heun latent at the VAE boundary).
    let img = model.decode(&heun, W, H, None).unwrap();
    assert_eq!(img.pixels.len(), (W * H * 3) as usize);
    assert!(
        img.pixels.iter().any(|&p| p != img.pixels[0]),
        "Heun decode produced a degenerate flat image"
    );
}
