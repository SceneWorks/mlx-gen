//! LoRA/LoKr injection tests for mlx-gen-anima (sc-10521). All `#[ignore]`d and real-weights-gated —
//! they need BOTH the `circlestone-labs/Anima` base snapshot (DiT checkpoints) and the
//! `circlestone-labs/Anima-Official-LoRAs` snapshot in the HF cache, plus Metal. Run with:
//!   cargo test -p mlx-gen-anima --release --test lora_injection -- --ignored --nocapture
//! The α = r ⇒ scale-1.0 weight-level checks live in a sibling process-isolated binary,
//! `tests/scale_convention.rs` (see its module doc for the Metal-stream reason). Both binaries — and
//! `tests/real_weights.rs` / `tests/velocity_convention.rs` — run under the full documented invocation
//!   cargo test -p mlx-gen-anima --release -- --ignored
//!
//! CI runs NONE of these (no weights); the injection *math* (PEFT LoRA / LoKr install, alpha/rank
//! fold, stacking) is covered in CI by the shared core `src/adapters/loader.rs` unit tests, and the
//! anima capability advertisement (`supports_lora`/`supports_lokr`) by `model.rs::descriptors_surface`.
//!
//! ## Why these tests, and not the story's "turbo reproduction"
//! The story proposed proving injection + scale by reproducing the merged `anima-turbo-v1.0` checkpoint
//! from `anima-base-v1.0` + `anima-turbo-lora-v0.2`. **That premise is false against the shipped
//! weights** (measured, [`turbo_checkpoint_is_not_base_plus_lora`]): on the DiT the LoRA weight delta is
//! ORTHOGONAL to `anima-turbo − anima-base` — cos ≈ +0.001 on the representative `blocks.0.self_attn
//! .q_proj` target this test prints, and across the DiT the best single-scale fit (s ≈ +0.069) still
//! leaves a relative residual ≈ 1.0000 (~0% of the checkpoint delta explained); `|anima-turbo −
//! anima-base|` is ~9× the LoRA delta's magnitude. `anima-turbo-v1.0` is an independent
//! fine-tune, NOT this LoRA merged onto base — so no scale reproduces it. (The conditioner cannot help
//! decide this: the turbo LoRA's 60 `llm_adapter.*` `lora_B` factors are all zero-initialized, so
//! `B·A ≡ 0` there and applying the LoRA leaves the conditioner unchanged — *consistent with*, but not
//! evidence for, the base↔turbo conditioner identity.) These tests therefore validate the two
//! silent-failure classes the reproduction test was meant to catch, DIRECTLY and more tightly:
//!   1. `llm_adapter` injection actually happens (count 508/448 + the DiT-only-host mutation), and
//!   2. the α = r ⇒ scale 1.0 convention (weight-level: injected forward == base + B·A, + the
//!      halve-scale mutation) — in `tests/scale_convention.rs`.

mod common;

use mlx_rs::ops::{matmul, multiply, subtract};
use mlx_rs::Dtype;

use mlx_gen::adapters::loader::{apply_adapter_specs_autoprefix, apply_adapters_strict};
use mlx_gen::adapters::AdaptableHost;
use mlx_gen::runtime::{AdapterKind, AdapterSpec};
use mlx_gen::weights::Weights;
use mlx_gen::WeightsSource;

use mlx_gen_anima::apply_anima_adapters;
use mlx_gen_anima::config::Variant;

use common::{l2, load_base, lora_spec, randn, split_files, style_lora, synth_lokr, turbo_lora};

// -------------------------------------------------------------------------------------------------
// 1. Injected-target COUNT (correctness bar #1) — the sc-10274 partial-injection guard.
// -------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs the circlestone-labs/Anima + Anima-Official-LoRAs snapshots"]
fn turbo_lora_injects_508_style_injects_448() {
    // Turbo LoRA = 448 DiT (28×16) + 60 conditioner (6×10) = 508 targets. (The 60 conditioner `lora_B`
    // are zero-initialized, but the injection MECHANISM still installs a residual on each — the count
    // proves every target routed, independent of whether its trained delta is non-zero.)
    let mut c = load_base();
    let report = apply_anima_adapters(
        &mut c.dit,
        &mut c.conditioner,
        &[lora_spec(turbo_lora(), 1.0)],
    )
    .expect("apply turbo LoRA");
    assert_eq!(
        report.applied, 508,
        "turbo LoRA must inject all 448 DiT + 60 adapter targets"
    );
    assert!(
        report.unmatched_paths.is_empty(),
        "no target may be unmatched: {:?}",
        report.unmatched_paths
    );

    // Style LoRA = 448 DiT-only, zero conditioner targets.
    let mut c2 = load_base();
    let report2 = apply_anima_adapters(
        &mut c2.dit,
        &mut c2.conditioner,
        &[lora_spec(style_lora(), 1.0)],
    )
    .expect("apply style LoRA");
    assert_eq!(
        report2.applied, 448,
        "DiT-only style LoRA must inject exactly 448 targets"
    );
    assert!(report2.unmatched_paths.is_empty());
}

#[test]
#[ignore = "needs the circlestone-labs/Anima + Anima-Official-LoRAs snapshots"]
fn adaln_modulation_pairs_are_injected() {
    // The three adaLN-modulation down/up pairs (`.1`/`.2`) are the ones most likely to be skipped.
    // Assert each is actually adapted after injecting the turbo LoRA.
    let mut c = load_base();
    apply_anima_adapters(
        &mut c.dit,
        &mut c.conditioner,
        &[lora_spec(turbo_lora(), 1.0)],
    )
    .unwrap();
    for adaln in [
        "adaln_modulation_self_attn",
        "adaln_modulation_cross_attn",
        "adaln_modulation_mlp",
    ] {
        for updown in ["1", "2"] {
            let lin = c
                .dit
                .adaptable_mut(&["blocks", "3", adaln, updown])
                .unwrap_or_else(|| panic!("no adaptable linear at blocks.3.{adaln}.{updown}"));
            assert_eq!(
                lin.adapters().len(),
                1,
                "blocks.3.{adaln}.{updown} not adapted"
            );
        }
    }
}

// -------------------------------------------------------------------------------------------------
// 2. MUTATION: a DiT-only injection walk (the sc-10274 bug) must FAIL LOUDLY, not partial-load.
// -------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs the circlestone-labs/Anima + Anima-Official-LoRAs snapshots"]
fn mutation_dit_only_walk_drops_60_adapter_targets() {
    // Route the turbo LoRA through the DiT host ALONE (skipping the `llm_adapter` conditioner) — the
    // exact partial-injection regression sc-10274 was. The strict applier MUST error on the 60
    // unmatched conditioner targets; a lenient walk reports only 448 applied + 60 unmatched.
    //
    // NB: for THIS file the dropped 60 conditioner deltas are all zero (`B ≡ 0`), so dropping them has
    // no numerical effect on the turbo LoRA specifically. This guard protects the injection MECHANISM
    // for future non-zero conditioner LoRAs — and for the shipped `anima-rl-v0.1`, which also carries
    // 60 `llm_adapter.*` targets — so the sc-10274 "loads at partial strength, looks fine" failure
    // class cannot recur silently. The count/routing is enforced regardless of the trained magnitudes.
    let mut dit_only = load_base();
    let lenient =
        apply_adapter_specs_autoprefix(&mut dit_only.dit, &[lora_spec(turbo_lora(), 1.0)])
            .expect("lenient apply");
    assert_eq!(
        lenient.applied, 448,
        "DiT-only walk injects only the 448 DiT targets"
    );
    assert_eq!(
        lenient.unmatched_paths.len(),
        60,
        "the 60 llm_adapter targets are dropped"
    );

    let mut dit_only2 = load_base();
    let err = apply_adapters_strict(
        &mut dit_only2.dit,
        &[lora_spec(turbo_lora(), 1.0)],
        "anima-dit-only",
    )
    .expect_err("strict apply onto a DiT-only host must ERROR on the dropped adapter targets");
    let msg = err.to_string();
    assert!(
        msg.contains("matched no module"),
        "expected an unmatched-target error, got: {msg}"
    );
}

// -------------------------------------------------------------------------------------------------
// 3. The turbo-reproduction reality check — encodes the finding that turbo ≠ base + LoRA.
// -------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot"]
fn turbo_checkpoint_is_not_base_plus_lora() {
    // Cosine between the turbo LoRA's weight delta (B·A) and (W_turbo − W_base) for a representative
    // DiT target. If anima-turbo-v1.0 were base + this LoRA, cos ≈ 1; measured cos ≈ 0 ⇒ independent
    // fine-tune. This documents WHY the story's reproduction test is invalid (not an injection/scale
    // bug). Also confirms the conditioner is byte-identical base↔turbo AND that the turbo LoRA's 60
    // conditioner `lora_B` are all zero.
    let split = split_files().expect("Anima base snapshot");
    let base_w =
        Weights::from_file(split.join("diffusion_models/anima-base-v1.0.safetensors")).unwrap();
    let turbo_w =
        Weights::from_file(split.join("diffusion_models/anima-turbo-v1.0.safetensors")).unwrap();
    let lw = Weights::from_file(turbo_lora()).unwrap();

    let wb = base_w
        .require("net.blocks.0.self_attn.q_proj.weight")
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let wt = turbo_w
        .require("model.diffusion_model.blocks.0.self_attn.q_proj.weight")
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let a = lw
        .require("diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight")
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let b = lw
        .require("diffusion_model.blocks.0.self_attn.q_proj.lora_B.weight")
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();

    let lora_delta = matmul(&b, &a).unwrap(); // [out,in]
    let turbo_delta = subtract(&wt, &wb).unwrap();
    let dot = multiply(&lora_delta, &turbo_delta)
        .unwrap()
        .sum(None)
        .unwrap()
        .item::<f32>();
    let nl = l2(&lora_delta);
    let nt = l2(&turbo_delta);
    let cos = dot / (nl * nt).max(1e-12);
    println!("cos(B·A, W_turbo−W_base) = {cos:+.4}  |B·A|={nl:.5}  |turbo−base|={nt:.5}");
    assert!(cos.abs() < 0.2, "turbo checkpoint UNEXPECTEDLY reproduces base+LoRA (cos {cos:.3}) — re-examine the premise");

    // Ground truth for the "60 zero-initialized conditioner slots" claim used throughout this crate:
    // EVERY `llm_adapter.*` `lora_B` in the turbo LoRA is exactly zero (`B ≡ 0` ⇒ `B·A ≡ 0`). Asserted
    // here rather than only stated in prose, so a future re-export that trained the conditioner would
    // trip this and force the surrounding reasoning to be revisited.
    let (mut cond_b_total, mut cond_b_nonzero) = (0usize, 0usize);
    for k in lw.keys() {
        if k.contains("llm_adapter") && k.ends_with(".lora_B.weight") {
            cond_b_total += 1;
            if l2(lw.require(k).unwrap()) > 0.0 {
                cond_b_nonzero += 1;
            }
        }
    }
    println!("conditioner lora_B: {cond_b_total} total, {cond_b_nonzero} non-zero");
    assert_eq!(
        cond_b_total, 60,
        "turbo LoRA must carry 60 conditioner lora_B slots"
    );
    assert_eq!(
        cond_b_nonzero, 0,
        "all 60 conditioner lora_B are zero-initialized (B·A ≡ 0); applying the LoRA never moves the \
         conditioner"
    );

    // Conditioner is byte-identical base↔turbo (measured |turbo−base| ≈ 9.3e-6, bf16 re-export
    // rounding). NOTE this is NOT independent proof that turbo ≠ base + LoRA: because the LoRA's
    // conditioner `B ≡ 0` (asserted above), applying it never moves the conditioner, so this identity
    // is consistent with base+LoRA rather than contradictory. The non-reproduction conclusion rests on
    // the DiT orthogonality above.
    let cb = base_w
        .require("net.llm_adapter.blocks.0.self_attn.q_proj.weight")
        .unwrap();
    let ct = turbo_w
        .require("model.diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.weight")
        .unwrap();
    let cond_diff = l2(&subtract(
        ct.as_dtype(Dtype::Float32).unwrap(),
        cb.as_dtype(Dtype::Float32).unwrap(),
    )
    .unwrap());
    println!("||conditioner turbo−base|| = {cond_diff:.6e}");
    assert!(
        cond_diff < 1e-2,
        "conditioner differs base↔turbo (expected identical): {cond_diff:.3e}"
    );
}

// -------------------------------------------------------------------------------------------------
// 4. LoKr load + apply, and stacked LoRA + LoKr (mixed).
// -------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot"]
fn lokr_loads_and_applies_to_dit_and_conditioner() {
    let mut c = load_base();
    let spec = AdapterSpec::new(synth_lokr(), 1.0, AdapterKind::Lokr);
    let report = apply_anima_adapters(&mut c.dit, &mut c.conditioner, &[spec]).expect("apply LoKr");
    assert_eq!(
        report.applied, 2,
        "LoKr must apply to both its DiT and conditioner targets"
    );
    assert!(
        report.unmatched_paths.is_empty(),
        "unmatched: {:?}",
        report.unmatched_paths
    );
    // Both targets carry exactly one (LoKr) adapter, and the residual is non-trivial.
    assert_eq!(
        c.dit
            .adaptable_mut(&["blocks", "1", "self_attn", "k_proj"])
            .unwrap()
            .adapters()
            .len(),
        1
    );
    assert_eq!(
        c.conditioner
            .adaptable_mut(&["blocks", "1", "self_attn", "k_proj"])
            .unwrap()
            .adapters()
            .len(),
        1
    );
}

#[test]
#[ignore = "needs the circlestone-labs/Anima + Anima-Official-LoRAs snapshots"]
fn stacked_lora_plus_lokr_mixed() {
    // Apply the turbo LoRA (508 targets) AND the synthetic LoKr (2 targets) in one strict call.
    let mut c = load_base();
    let specs = vec![
        lora_spec(turbo_lora(), 1.0),
        AdapterSpec::new(synth_lokr(), 1.0, AdapterKind::Lokr),
    ];
    let report =
        apply_anima_adapters(&mut c.dit, &mut c.conditioner, &specs).expect("apply stacked");
    assert_eq!(report.applied, 510, "508 LoRA + 2 LoKr targets");
    // blocks.1.self_attn.k_proj is hit by BOTH the turbo LoRA and the LoKr → two stacked adapters.
    let lin = c
        .dit
        .adaptable_mut(&["blocks", "1", "self_attn", "k_proj"])
        .unwrap();
    assert_eq!(
        lin.adapters().len(),
        2,
        "mixed LoRA + LoKr must stack on the same module"
    );
    // A forward runs (both residuals compose over the shared base).
    let x = randn(&[1, 8, 2048]);
    assert_eq!(lin.forward(&x).unwrap().shape(), &[1, 8, 2048]);
}

// -------------------------------------------------------------------------------------------------
// 5. End-to-end: anima_base + turbo LoRA generates a coherent image (proves the injected LoRA runs
//    through the full pipeline). NOTE: not compared to anima-turbo-v1.0 — see
//    `turbo_checkpoint_is_not_base_plus_lora`; they are different models.
// -------------------------------------------------------------------------------------------------

fn grayscale_std(pixels: &[u8]) -> f32 {
    let gray: Vec<f32> = pixels
        .chunks(3)
        .map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32)
        .collect();
    let mean = gray.iter().sum::<f32>() / gray.len() as f32;
    (gray.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / gray.len() as f32).sqrt()
}

#[test]
#[ignore = "needs the circlestone-labs/Anima + Anima-Official-LoRAs snapshots; SLOW (2B DiT denoise)"]
fn base_plus_turbo_lora_generates_coherent_image() {
    use mlx_gen::runtime::CancelFlag;
    use mlx_gen::Progress;
    use mlx_gen_anima::pipeline::{AnimaPipeline, GenOptions};

    let split = split_files().expect("Anima base snapshot");
    let mut pipeline =
        AnimaPipeline::from_source(&WeightsSource::Dir(split), Variant::Base).expect("pipeline");
    let report = pipeline
        .apply_adapters(&[lora_spec(turbo_lora(), 1.0)])
        .expect("apply turbo LoRA");
    assert_eq!(report.applied, 508);

    // Turbo regime (few-step, CFG-free) — the LoRA is a distillation adapter.
    let opts = GenOptions {
        width: 1024,
        height: 1024,
        steps: 10,
        guidance: 1.0,
        seed: 42,
        sampler: None,
        scheduler: None,
    };
    let cancel = CancelFlag::default();
    let mut prog = |_p: Progress| {};
    let img = pipeline
        .generate(
            "an anime girl with silver hair, detailed illustration, masterpiece",
            "",
            Variant::Base,
            &opts,
            &cancel,
            &mut prog,
        )
        .expect("generate");
    let std = grayscale_std(&img.pixels);
    println!("[base+turbo-lora] grayscale std = {std:.2}");
    assert_eq!((img.width, img.height), (1024, 1024));
    assert!(
        std > 8.0,
        "base+turbo-LoRA output is near-blank (std {std:.2}) — injection likely broken"
    );
}
