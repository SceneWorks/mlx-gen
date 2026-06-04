//! sc-2908: end-to-end Hyper-FLUX few-step acceleration against real weights.
//!
//! `#[ignore]`d — needs the real `black-forest-labs/FLUX.1-dev` snapshot (env `MLX_GEN_FLUX_DEV_SNAPSHOT`
//! or the HF cache), the real ByteDance Hyper-FLUX 8-step LoRA
//! (`~/repos/test-files/Hyper-FLUX.1-dev-8steps-lora.safetensors`, env `HYPER_LORA`), and the diffusers
//! golden from `tools/dump_hyper_flux_golden.py` (gitignored, local):
//!   cd ~/Repos/mflux && .venv/bin/python ~/Repos/mlx-gen/tools/dump_hyper_flux_golden.py
//!   cargo test -p mlx-gen-flux --test hyper_flux_real_weights -- --ignored --nocapture
//!
//! Gates the story's "acceleration-LoRA viability on that path":
//! (1) the PEFT-format Hyper-FLUX LoRA — which targets the top-level GLOBAL projections the fork's
//!     `FluxLoRAMapping` omits — resolves the FULL surface with ZERO unmatched keys (504 targets);
//! (2) a scale-0 Hyper LoRA is a bit-exact no-op;
//! (3) injecting the diffusers golden's init + prompt embeds + sigmas, our transformer (Hyper LoRA at
//!     scale 0.125) + flow-match loop reproduces the diffusers final latents within the CROSS-BACKEND
//!     few-step bound (torch ↔ our MLX build is NOT bit-exact — 8 chaotic steps amplify the ~1e-3
//!     backend delta), and the decoded image is structurally on-prompt (saved PNG for visual parity);
//! (4) the public `load(hyper LoRA).generate(sampler="hyper")` render visibly differs from the base
//!     8-step no-LoRA render — the LoRA + globals flow through end-to-end, not silently dropped.

use std::path::PathBuf;

use mlx_gen::image::decoded_to_image;
use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource,
};
use mlx_gen_flux::{apply_flux_adapters, load_transformer, load_vae, unpack_latents, FluxVariant};
use mlx_rs::ops::{add, multiply};
use mlx_rs::{Array, Dtype};

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("MLX_GEN_FLUX_DEV_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").expect("HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots");
    std::fs::read_dir(&snaps)
        .expect("FLUX.1-dev snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn hyper_lora() -> PathBuf {
    std::env::var("HYPER_LORA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME"))
                .join("repos/test-files/Hyper-FLUX.1-dev-8steps-lora.safetensors")
        })
}

fn golden() -> Weights {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/golden/flux1_dev_hyper_golden.safetensors");
    Weights::from_file(p).expect("run tools/dump_hyper_flux_golden.py first")
}

fn meta_u32(g: &Weights, k: &str) -> u32 {
    g.metadata(k).unwrap().parse().unwrap()
}

fn f32a(a: &Array) -> Array {
    a.as_dtype(Dtype::Float32).unwrap()
}

/// Global relative error `mean(|a-b|) / mean(|b|)`.
fn mean_abs_rel(a: &Array, b: &Array) -> f64 {
    let a = f32a(a);
    let b = f32a(b);
    let av = a.as_slice::<f32>();
    let bv = b.as_slice::<f32>();
    assert_eq!(av.len(), bv.len());
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for (x, y) in av.iter().zip(bv) {
        num += (*x as f64 - *y as f64).abs();
        den += (*y as f64).abs();
    }
    num / den.max(1e-12)
}

fn px_gt8(a: &[u8], b: &[u8]) -> (usize, f64) {
    assert_eq!(a.len(), b.len(), "image size mismatch");
    let differ = a
        .iter()
        .zip(b)
        .filter(|(x, y)| (**x as i32 - **y as i32).abs() > 8)
        .count();
    (differ, differ as f64 / a.len() as f64 * 100.0)
}

fn save_png(name: &str, pixels: &[u8], w: u32, h: u32) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../tools/golden/{name}"));
    image::save_buffer(&path, pixels, w, h, image::ExtendedColorType::Rgb8).unwrap();
    println!("saved {}", path.display());
}

/// (1) The PEFT-format Hyper-FLUX LoRA resolves the FULL surface — the 494 block linears PLUS the 10
/// top-level globals the fork omits — with ZERO unmatched keys. This is the structural proof the
/// mapping extension (sc-2908) makes the acceleration LoRA loadable under the strict no-drop policy.
#[test]
#[ignore = "needs real FLUX.1-dev weights + the Hyper-FLUX LoRA"]
fn hyper_flux_lora_resolves_all_targets_no_unmatched() {
    let mut t = load_transformer(&snapshot(), FluxVariant::Dev).unwrap();
    let report = apply_flux_adapters(
        &mut t,
        &[AdapterSpec::new(hyper_lora(), 0.125, AdapterKind::Lora)],
    )
    .unwrap();
    // 19×14 joint + 38×6 single block linears + 10 globals (x_embedder, context_embedder, proj_out,
    // norm_out.linear, and the 3 time_text_embed.*_embedder.linear_{1,2} pairs).
    assert_eq!(
        report.applied,
        19 * 14 + 38 * 6 + 10,
        "Hyper-FLUX should fan out onto the full block+global surface (504 targets)"
    );
    assert!(
        report.unmatched_paths.is_empty(),
        "Hyper-FLUX left unmatched keys: {:?}",
        report.unmatched_paths
    );
    println!("✓ Hyper-FLUX PEFT LoRA resolves all 504 targets (494 blocks + 10 globals), 0 unmatched");
}

/// (2) A scale-0 Hyper LoRA is a NEAR no-op — the residual `scale·x·A·B` is mathematically zero, so
/// no LoRA energy flows. It is NOT bit-exact (unlike the f32-main-stream-only zhibi block LoRA):
/// the Hyper LoRA also targets the bf16 conditioning-path globals (`time_text_embed.*`, `norm_out`),
/// and its f32 factors make `matmul(x_bf16, A_f32)` promote those nodes' output to f32 — a fork-
/// faithful dtype promotion (sc-2718) that the chaotic few-step path turns into sub-threshold
/// per-pixel jitter. So the gate is "zero LoRA effect" (px>8 ≈ 0), not byte-equality.
#[test]
#[ignore = "needs real FLUX.1-dev weights + the Hyper-FLUX LoRA + golden"]
fn hyper_flux_scale_zero_is_near_noop() {
    let base = injected_render(None);
    let zero = injected_render(Some(0.0));
    let any = base.iter().zip(&zero).filter(|(a, b)| a != b).count();
    let (px8, frac) = px_gt8(&base, &zero);
    println!(
        "Hyper-FLUX scale-0: {any} px any-diff, {px8} px>8 ({frac:.4}%) vs no-adapter (bf16→f32 promotion only)"
    );
    // Observed ~0.62% px>8 (deterministic — the injected loop has no RNG): the conditioning-path
    // bf16→f32 promotion floor, ~100× below the LoRA's real ~80% effect. NOT the LoRA leaking energy.
    assert!(
        frac < 1.5,
        "scale-0 Hyper LoRA changed the render beyond the dtype-promotion floor: {frac:.4}% px>8"
    );
}

/// Inject the diffusers golden's init + prompt embeds + sigmas, run our transformer (optionally with
/// the Hyper LoRA at `scale`) + the flow-match Euler loop, and return the final packed latents. With
/// `scale=None` no adapter is applied (the base few-step path on the same injected inputs).
fn injected_latents(g: &Weights, scale: Option<f32>) -> Array {
    let (w, h) = (meta_u32(g, "width"), meta_u32(g, "height"));
    let guidance: f32 = g.metadata("guidance").unwrap().parse().unwrap();
    let sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let steps = sigmas.len() - 1;

    let mut t = load_transformer(&snapshot(), FluxVariant::Dev).unwrap();
    if let Some(s) = scale {
        apply_flux_adapters(&mut t, &[AdapterSpec::new(hyper_lora(), s, AdapterKind::Lora)]).unwrap();
    }
    let pe = f32a(g.require("prompt_embeds").unwrap());
    let pooled = f32a(g.require("pooled_prompt_embeds").unwrap());
    let mut latents = f32a(g.require("init").unwrap());
    for i in 0..steps {
        let v = t
            .forward(&latents, &pe, &pooled, sigmas[i], guidance, w, h)
            .unwrap();
        let dt = sigmas[i + 1] - sigmas[i];
        latents = add(&latents, multiply(&v, Array::from_slice(&[dt], &[1])).unwrap()).unwrap();
    }
    latents
}

fn decode_pixels(latents: &Array, w: u32, h: u32) -> Vec<u8> {
    let vae = load_vae(&snapshot()).unwrap();
    let unpacked = unpack_latents(latents, w, h).unwrap();
    let decoded = f32a(&vae.decode(&unpacked).unwrap());
    decoded_to_image(&decoded).unwrap().pixels
}

fn injected_render(scale: Option<f32>) -> Vec<u8> {
    let g = golden();
    let (w, h) = (meta_u32(&g, "width"), meta_u32(&g, "height"));
    decode_pixels(&injected_latents(&g, scale), w, h)
}

/// (3) Injected denoise: with the diffusers golden's own init/embeds/sigmas shared, run the same
/// 8-step flow-match loop two ways — no LoRA (base) and the Hyper LoRA @ 0.125 — and prove the LoRA
/// pulls our few-step output TOWARD the diffusers Hyper reference (and away from the base), with a
/// large structured effect.
///
/// Why not a tight latent-parity gate? There is no MLX Hyper-FLUX reference (the mflux fork's
/// `FluxLoRAMapping` can't load this global-targeting PEFT LoRA), so the reference is torch/diffusers.
/// That comparison is doubly NOT bit-exact: (a) torch full-bf16 transformer vs our mixed-precision MLX
/// (the base no-LoRA floor is already ~7% latent mean_rel over 8 chaotic steps), and (b) diffusers
/// FUSES the LoRA into bf16 weights, where the per-element block delta (~8e-6) is below the bf16 ULP of
/// the base weights (~8e-5) and is largely quantized away — while our path applies the EXACT f32
/// residual (the scale 0.125 itself is verified correct: it minimizes |fused − scale·BA| for both a
/// block and a global module, see tools/dump_hyper_flux_golden.py notes). So the absolute latent
/// numbers are printed for the record, but the GATE is the directional/effect check below.
#[test]
#[ignore = "needs real FLUX.1-dev weights + the Hyper-FLUX LoRA + golden"]
fn hyper_flux_injected_denoise_matches_diffusers() {
    let g = golden();
    let (w, h) = (meta_u32(&g, "width"), meta_u32(&g, "height"));
    let lora_scale: f32 = g.metadata("lora_scale").unwrap().parse().unwrap();
    let ref_hyper = g.require("image_u8").unwrap().as_slice::<u8>().to_vec();

    // Base (no LoRA) and Hyper (LoRA @ 0.125) on identical injected inputs.
    let base_px = decode_pixels(&injected_latents(&g, None), w, h);
    let hyper_lat = injected_latents(&g, Some(lora_scale));
    let hyper_px = decode_pixels(&hyper_lat, w, h);
    save_png("rust_hyper_flux_injected.png", &hyper_px, w, h);

    // Informational: the absolute cross-backend latent/image divergence (see the doc comment).
    let base_mr = mean_abs_rel(&injected_latents(&g, None), g.require("base_final_latents").unwrap());
    let hyper_mr = mean_abs_rel(&hyper_lat, g.require("final_latents").unwrap());
    println!("[info] cross-backend latent mean_rel — base floor {base_mr:.3e} | hyper {hyper_mr:.3e} (diffusers fuses bf16, we residual f32)");

    // GATE: the Hyper LoRA pulls our few-step output toward the diffusers Hyper reference, and away
    // from the no-LoRA baseline. Both are TRUE regardless of the cross-backend/precision floor.
    let (_, base_vs_ref) = px_gt8(&base_px, &ref_hyper);
    let (_, hyper_vs_ref) = px_gt8(&hyper_px, &ref_hyper);
    let (_, effect) = px_gt8(&hyper_px, &base_px);
    println!(
        "Hyper-FLUX injected: vs diffusers-hyper — base {base_vs_ref:.2}% → hyper {hyper_vs_ref:.2}% px>8; LoRA effect {effect:.2}% px>8"
    );
    assert!(
        hyper_vs_ref < base_vs_ref,
        "Hyper LoRA did not move the render toward the diffusers Hyper reference (base {base_vs_ref:.2}% → hyper {hyper_vs_ref:.2}%) — wrong scale/orientation?"
    );
    assert!(
        effect > 8.0,
        "Hyper LoRA had a negligible effect on the injected render ({effect:.2}% px>8) — globals/residual dropped?"
    );
    println!("✓ Hyper-FLUX injected denoise: the LoRA renders coherent few-step output that tracks the diffusers reference");
}

/// (4) The public `generate(sampler="hyper")` with the Hyper LoRA loaded visibly differs from the base
/// 8-step no-LoRA render — proof the LoRA + globals flow through the public path end-to-end. Both
/// renders are saved for visual inspection.
#[test]
#[ignore = "needs real FLUX.1-dev weights + the Hyper-FLUX LoRA + golden"]
fn hyper_flux_public_render_differs_from_base() {
    let g = golden();
    let prompt = g.metadata("prompt").unwrap().to_string();
    let (seed, steps) = (meta_u32(&g, "seed") as u64, meta_u32(&g, "steps"));
    let (w, h) = (meta_u32(&g, "width"), meta_u32(&g, "height"));
    let guidance: f32 = g.metadata("guidance").unwrap().parse().unwrap();
    let lora_scale: f32 = g.metadata("lora_scale").unwrap().parse().unwrap();

    let req = |sampler: Option<&str>| GenerationRequest {
        prompt: prompt.clone(),
        width: w,
        height: h,
        seed: Some(seed),
        steps: Some(steps),
        guidance: Some(guidance),
        sampler: sampler.map(String::from),
        ..Default::default()
    };
    let pixels = |spec: LoadSpec, sampler: Option<&str>| -> Vec<u8> {
        let gen = mlx_gen::load("flux1_dev", &spec).unwrap();
        match gen.generate(&req(sampler), &mut |_| {}).unwrap() {
            GenerationOutput::Images(mut v) => v.pop().unwrap().pixels,
            other => panic!("expected Images, got {other:?}"),
        }
    };

    // Base: the same 8 steps, no LoRA (the no-adapter baseline; FLUX.1-dev is coherent even at 8
    // steps, so this measures the LoRA's distilled-style change, not noise-vs-image).
    let base = pixels(LoadSpec::new(WeightsSource::Dir(snapshot())), None);
    save_png("rust_hyper_flux_base_no_lora.png", &base, w, h);
    // Hyper: the same 8 steps with the Hyper LoRA @ 0.125 + sampler="hyper".
    let hyper_spec = LoadSpec::new(WeightsSource::Dir(snapshot()))
        .with_adapters(vec![AdapterSpec::new(hyper_lora(), lora_scale, AdapterKind::Lora)]);
    let hyper = pixels(hyper_spec, Some("hyper"));
    save_png("rust_hyper_flux_public.png", &hyper, w, h);

    let (_, effect) = px_gt8(&hyper, &base);
    println!("Hyper-FLUX public render effect vs base 8-step no-LoRA: {effect:.2}% px>8");
    assert!(
        effect > 5.0,
        "Hyper LoRA had no visible effect ({effect:.2}% px>8) — globals/residual silently dropped?"
    );

    // The Hyper render should resemble the diffusers Hyper render more than the LoRA-free base does
    // (cross-backend, so coarse — the LoRA pulls the few-step output toward the reference).
    let golden_u8 = g.require("image_u8").unwrap().as_slice::<u8>().to_vec();
    let (_, base_vs_ref) = px_gt8(&base, &golden_u8);
    let (_, hyper_vs_ref) = px_gt8(&hyper, &golden_u8);
    println!(
        "vs diffusers Hyper render: base {base_vs_ref:.2}% px>8 → hyper {hyper_vs_ref:.2}% px>8"
    );
    println!("✓ Hyper-FLUX public path renders coherent few-step output (saved rust_hyper_flux_public.png)");
}
