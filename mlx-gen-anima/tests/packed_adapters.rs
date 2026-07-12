//! Real-weights proof that a **packed (Q4/Q8) Anima DiT accepts LoRA and LoKr** (sc-10578).
//!
//! All `#[ignore]`d and real-weights-gated: they need the `circlestone-labs/Anima` base snapshot
//! (DiT + TE + VAE) and the `Anima-Official-LoRAs` snapshot in the HF cache, plus Metal. Run with:
//!   cargo test -p mlx-gen-anima --release --test packed_adapters -- --ignored --nocapture
//!
//! CI runs none of these (no weights). The *math* — that a packed base installs the structured
//! Kronecker form and that the residual matches the materialized one — is covered in CI by the shared
//! core unit tests in `mlx-gen/src/adapters/loader.rs`
//! (`lokr_on_packed_base_installs_structured_and_matches_dense` and its mutation twin).
//!
//! ## What sc-10578 actually changed, and what these tests guard
//! `mlx-gen-anima/src/model.rs` used to reject `spec.quantize.is_some() && !spec.adapters.is_empty()`
//! outright. Nothing downstream needed to change to make the pair work: the DiT's linears are
//! `AdaptableLinear`s that packed-detect off the on-disk `{base}.scales`, and `AdaptableLinear`
//! evaluates `base(x) + Σ adapter.residual(x)` — the epic-10043 additive branch
//! `y = xW_packed + scale·(xA)B`, with the packed codes never dequantized or mutated.
//!
//! Two silent-failure classes are guarded here, on real 2B weights:
//!   1. **The residual must ride on the packed base, which must never be mutated.**
//!      `packed_dit_lora_is_additive_over_packed_codes` captures the packed forward and the packed
//!      code/scale/bias triple BEFORE injection, then asserts the post-injection forward equals
//!      `base + scale·(x·Aᵀ)·Bᵀ` AND that the packed triple is bit-identical afterwards.
//!   2. **A LoKr on a packed base must not materialize `[out,in]`.** The converter keeps the
//!      conditioner dense while packing the DiT, so ONE LoKr file lands on both kinds of base at once:
//!      the DiT target must install `Adapter::LokrStructured`, the conditioner target the materialized
//!      `Adapter::Lokr`. That mixed expectation is far tighter than either alone.
//!
//! ## Read the numbers honestly
//! The `rel_err` in test 1 comes out at exactly `0.000e0`, and that is EXPECTED rather than
//! impressive: `apply_lora_peft` stores the factors already transposed, so `Adapter::Lora::residual`
//! computes `matmul(matmul(x, Aᵀ), Bᵀ)` — the same op sequence, on the same arrays, that the
//! "independent" reference here rebuilds from the raw file. That assertion therefore proves the
//! residual is applied *at the right scale*, and nothing more. What proves the residual is **additive
//! over packed codes** rather than folded into a dequantized weight is structural, and is asserted
//! separately: `is_quantized()` still true after injection, `dense_weight()` still `None`, the adapter
//! stack is a `Lora` residual rather than a merged weight, the packed triple is unchanged bit-for-bit,
//! and a scale-0 adapter is an exact no-op. A fold would break every one of those while leaving
//! `rel_err` at ~0.

mod common;

use std::path::{Path, PathBuf};

use mlx_rs::ops::{add, array_eq, matmul, multiply};
use mlx_rs::Array;

use mlx_gen::adapters::{AdaptableHost, Adapter};
use mlx_gen::weights::Weights;
use mlx_gen::WeightsSource;

use mlx_gen_anima::config::Variant;
use mlx_gen_anima::convert::quantize_anima_dit;
use mlx_gen_anima::loader::AnimaComponents;

use common::{lora_spec, randn, rel_err, split_files, style_lora, synth_lokr};

/// Anima's DiT self-attention `q_proj` on block 0 — `[2048, 2048]`, group-aligned, so the converter
/// packs it. Present in the official style LoRA (a DiT-only, 448-target file).
const DIT_Q_PATH: &[&str] = &["blocks", "0", "self_attn", "q_proj"];
const DIT_Q_KEY: &str = "diffusion_model.blocks.0.self_attn.q_proj";

/// Build (and cache) a packed tier: a `split_files/` tree whose `diffusion_models/` holds the
/// Q`bits`-packed base DiT, with `text_encoders/` and `vae/` symlinked to the real snapshot. This is
/// exactly the layout the SceneWorks worker's convert-at-install step produces (sc-10517).
///
/// Cached across runs — quantizing the 3.9 GB bf16 DiT takes real time, and the output is a pure
/// function of (source checkpoint, bits, group_size).
fn packed_split_files(bits: i32) -> PathBuf {
    let real = split_files().expect("Anima base snapshot");
    let root = std::env::temp_dir().join(format!("anima_sc10578_q{bits}/split_files"));
    let dit_dst = root
        .join("diffusion_models")
        .join(Variant::Base.dit_filename());

    if !dit_dst.is_file() {
        std::fs::create_dir_all(dit_dst.parent().unwrap()).unwrap();
        let dit_src = real
            .join("diffusion_models")
            .join(Variant::Base.dit_filename());
        eprintln!("[sc-10578] packing {} → q{bits} …", dit_src.display());
        quantize_anima_dit(&dit_src, &dit_dst, bits, 64).expect("quantize DiT");
    }
    // Symlink the components the converter leaves dense (idempotent).
    for sub in ["text_encoders", "vae"] {
        let dst = root.join(sub);
        if !dst.exists() {
            std::os::unix::fs::symlink(real.join(sub), &dst).unwrap();
        }
    }
    root
}

fn load_packed(bits: i32) -> AnimaComponents {
    let root = packed_split_files(bits);
    AnimaComponents::load(&WeightsSource::Dir(root), Variant::Base).expect("load packed components")
}

/// `scale · (x·Aᵀ)·Bᵀ` from the RAW LoRA file factors — deliberately NOT `Adapter::residual`, so the
/// assertion cannot pass by re-transcribing the implementation it is checking.
fn raw_lora_residual(lw: &Weights, key: &str, x: &Array, scale: f32) -> Array {
    let a = lw.require(&format!("{key}.lora_A.weight")).unwrap(); // [r, in]
    let b = lw.require(&format!("{key}.lora_B.weight")).unwrap(); // [out, r]
    let r = matmul(matmul(x, a.t()).unwrap(), b.t()).unwrap();
    multiply(&r, Array::from_f32(scale).as_dtype(r.dtype()).unwrap()).unwrap()
}

/// The packed DiT's linears really are quantized — otherwise every assertion below is vacuous
/// (a dense checkpoint would just take the ordinary dense path and pass).
fn assert_dit_is_packed(c: &mut AnimaComponents, bits: i32) {
    let lin = c.dit.adaptable_mut(DIT_Q_PATH).expect("q_proj");
    assert!(
        lin.is_quantized(),
        "q{bits}: the DiT must load PACKED — otherwise this test proves nothing"
    );
    assert!(
        lin.dense_weight().is_none(),
        "q{bits}: a packed base exposes no dense weight"
    );
    assert_eq!(lin.base_shape(), vec![2048, 2048]);
}

// -------------------------------------------------------------------------------------------------
// 1. LoRA rides additively on the packed codes.
// -------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs the circlestone-labs/Anima + Anima-Official-LoRAs snapshots; SLOW (packs a 3.9 GB DiT)"]
fn packed_dit_lora_is_additive_over_packed_codes() {
    let lw = Weights::from_file(style_lora()).expect("style LoRA");

    for bits in [4, 8] {
        let mut c = load_packed(bits);
        assert_dit_is_packed(&mut c, bits);

        // Capture the PACKED base forward AND the packed triple before injection. Force-eval: the
        // 448-target install below recreates the default Metal stream, which would strand a lazily-held
        // value.
        let x = randn(&[1, 8, 2048]);
        let (y_base, codes_before, scales_before) = {
            let lin = c.dit.adaptable_mut(DIT_Q_PATH).unwrap();
            let y = lin.forward(&x).unwrap();
            let (codes, scales, ..) = lin.quantized_params().expect("packed triple");
            (y, codes.clone(), scales.clone())
        };
        mlx_rs::transforms::eval([&x, &y_base, &codes_before, &scales_before]).unwrap();

        let report = mlx_gen_anima::apply_anima_adapters(
            &mut c.dit,
            &mut c.conditioner,
            &[lora_spec(style_lora(), 1.0)],
        )
        .expect("apply style LoRA onto a packed DiT");
        assert_eq!(
            report.applied, 448,
            "q{bits}: the DiT-only style LoRA must inject all 448 targets"
        );

        let lin = c.dit.adaptable_mut(DIT_Q_PATH).unwrap();
        assert!(
            lin.is_quantized(),
            "q{bits}: injection must NOT dequantize the base"
        );
        assert!(
            lin.dense_weight().is_none(),
            "q{bits}: a fold would have produced a dense weight"
        );
        assert!(
            matches!(lin.adapters(), [Adapter::Lora { .. }]),
            "q{bits}: a LoRA installs as a forward-time residual, not a weight fold"
        );

        // THE discriminating assertion. A fold-into-dequantized-weight would leave `rel_err` at ~0 and
        // still look right; it cannot leave the packed codes untouched. `y = xW_packed + scale·(xA)B`
        // means the base is read, never written.
        {
            let (codes_after, scales_after, ..) = lin.quantized_params().expect("still packed");
            assert!(
                array_eq(&codes_before, codes_after, None)
                    .unwrap()
                    .item::<bool>(),
                "q{bits}: the packed u32 codes were MUTATED by adapter injection"
            );
            assert!(
                array_eq(&scales_before, scales_after, None)
                    .unwrap()
                    .item::<bool>(),
                "q{bits}: the packed scales were MUTATED by adapter injection"
            );
        }

        // The claim: packed_forward_after == packed_forward_before + scale·(xAᵀ)Bᵀ.
        // (~0 by construction — see the module doc. Its job is to pin the SCALE, not the additivity.)
        let y_inj = lin.forward(&x).unwrap();
        let want = add(&y_base, raw_lora_residual(&lw, DIT_Q_KEY, &x, 1.0)).unwrap();
        let err = rel_err(&y_inj, &want);
        println!("[sc-10578 q{bits}] additive-branch rel_err = {err:.3e}");
        assert!(
            err < 5e-3,
            "q{bits}: injected packed forward must equal packed base + scale·B·A (rel_err {err:.3e})"
        );

        // MUTATION: at scale 0 the residual vanishes, so the forward must return to the packed base
        // exactly. If this does not hold, `y_base` was not really the pre-injection packed forward and
        // the assertion above is measuring nothing.
        let mut c0 = load_packed(bits);
        let y0_base = c0
            .dit
            .adaptable_mut(DIT_Q_PATH)
            .unwrap()
            .forward(&x)
            .unwrap();
        mlx_rs::transforms::eval([&y0_base]).unwrap();
        mlx_gen_anima::apply_anima_adapters(
            &mut c0.dit,
            &mut c0.conditioner,
            &[lora_spec(style_lora(), 0.0)],
        )
        .unwrap();
        let y0_inj = c0
            .dit
            .adaptable_mut(DIT_Q_PATH)
            .unwrap()
            .forward(&x)
            .unwrap();
        let err0 = rel_err(&y0_inj, &y0_base);
        assert!(
            err0 < 1e-6,
            "q{bits}: a scale-0 LoRA must be a no-op over the packed base (rel_err {err0:.3e})"
        );
    }
}

// -------------------------------------------------------------------------------------------------
// 2. LoKr: structured on the packed DiT, materialized on the dense conditioner — in ONE install.
// -------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot; SLOW (packs a 3.9 GB DiT)"]
fn packed_dit_lokr_is_structured_while_dense_conditioner_stays_materialized() {
    let lokr = synth_lokr();
    let bits = 4;
    let mut c = load_packed(bits);
    assert_dit_is_packed(&mut c, bits);

    // The converter keeps the bundled conditioner dense (sc-10517 policy) — verify, don't assume.
    let cond_path: &[&str] = &["blocks", "1", "self_attn", "k_proj"];
    assert!(
        !c.conditioner
            .adaptable_mut(cond_path)
            .expect("conditioner k_proj")
            .is_quantized(),
        "the conditioner must stay dense bf16 in every tier — the mixed expectation below depends on it"
    );

    let report = mlx_gen_anima::apply_anima_adapters(
        &mut c.dit,
        &mut c.conditioner,
        &[mlx_gen::runtime::AdapterSpec::new(
            lokr,
            1.0,
            mlx_gen::runtime::AdapterKind::Lokr,
        )],
    )
    .expect("apply LoKr across a packed DiT and a dense conditioner");
    assert_eq!(report.applied, 2, "one DiT target + one conditioner target");

    // Packed DiT target → the deferred Kronecker form. `[2048,2048]` is never allocated.
    let dit_lin = c
        .dit
        .adaptable_mut(&["blocks", "1", "self_attn", "k_proj"])
        .unwrap();
    let dit_elems = match dit_lin.adapters() {
        [Adapter::LokrStructured { factors }] => {
            assert_eq!(factors.w1.shape(), &[64, 64]);
            assert_eq!(factors.w2.shape(), &[32, 32]);
            factors.w1.size() + factors.w2.size()
        }
        [Adapter::Lokr { .. }] => panic!(
            "the packed DiT target installed a MATERIALIZED [2048,2048] delta — this is exactly the \
             memory regression sc-10578 exists to prevent"
        ),
        other => panic!("unexpected adapter stack of len {}", other.len()),
    };

    // Dense conditioner target → unchanged materialized delta (fork-parity path preserved).
    let cond_lin = c.conditioner.adaptable_mut(cond_path).unwrap();
    match cond_lin.adapters() {
        [Adapter::Lokr { delta, .. }] => assert_eq!(delta.shape(), &[1024, 1024]),
        [Adapter::LokrStructured { .. }] => {
            panic!("a DENSE base must keep the materialized delta — other families' goldens rely on it")
        }
        other => panic!("unexpected adapter stack of len {}", other.len()),
    }

    // The memory claim, measured in elements held rather than inferred: the structured form holds
    // 64² + 32² = 5120 elements where the materialized delta would hold 2048² = 4_194_304 — a ~819×
    // reduction, per target, across all 448 packed DiT targets.
    let materialized = 2048usize * 2048;
    println!(
        "[sc-10578] structured LoKr holds {dit_elems} elems vs {materialized} materialized \
         ({:.0}× smaller)",
        materialized as f32 / dit_elems as f32
    );
    assert!(
        dit_elems * 100 < materialized,
        "structured factors must be orders of magnitude smaller than the [out,in] delta"
    );
}

// -------------------------------------------------------------------------------------------------
// 3. End-to-end: q4 + the official style LoRA generates a coherent, visibly-restyled image.
// -------------------------------------------------------------------------------------------------

fn grayscale_std(pixels: &[u8]) -> f32 {
    let gray: Vec<f32> = pixels
        .chunks(3)
        .map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32)
        .collect();
    let mean = gray.iter().sum::<f32>() / gray.len() as f32;
    (gray.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / gray.len() as f32).sqrt()
}

/// Mean absolute per-pixel difference, 0..255.
fn mean_abs_diff(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as f32 - *y as f32).abs())
        .sum::<f32>()
        / a.len() as f32
}

fn write_ppm(path: &Path, pixels: &[u8], w: u32, h: u32) {
    let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
    out.extend_from_slice(pixels);
    std::fs::write(path, out).unwrap();
}

#[test]
#[ignore = "needs both Anima snapshots; VERY SLOW (packs a 3.9 GB DiT, then two 30-step 1024² denoises)"]
fn packed_q4_plus_style_lora_generates_a_visibly_restyled_image() {
    use mlx_gen::runtime::CancelFlag;
    use mlx_gen::Progress;
    use mlx_gen_anima::pipeline::{AnimaPipeline, GenOptions};

    let root = packed_split_files(4);
    let opts = GenOptions {
        width: 1024,
        height: 1024,
        steps: 30,
        guidance: 4.5,
        seed: 42,
        sampler: None,
        scheduler: None,
    };
    let prompt =
        "masterpiece, best quality, score_7, safe, 1girl, silver hair, castle, dramatic lighting";
    let negative = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts";

    let gen = |adapters: &[mlx_gen::runtime::AdapterSpec]| {
        let mut p = AnimaPipeline::from_source(&WeightsSource::Dir(root.clone()), Variant::Base)
            .expect("packed q4 pipeline");
        if !adapters.is_empty() {
            let r = p.apply_adapters(adapters).expect("apply LoRA on q4");
            assert_eq!(r.applied, 448);
        }
        let cancel = CancelFlag::default();
        let mut prog = |_p: Progress| {};
        p.generate(prompt, negative, Variant::Base, &opts, &cancel, &mut prog)
            .expect("generate")
    };

    let plain = gen(&[]);
    let styled = gen(&[lora_spec(style_lora(), 1.0)]);

    let dir = std::env::temp_dir().join("anima_sc10578_images");
    std::fs::create_dir_all(&dir).unwrap();
    write_ppm(&dir.join("q4_plain.ppm"), &plain.pixels, 1024, 1024);
    write_ppm(&dir.join("q4_style_lora.ppm"), &styled.pixels, 1024, 1024);
    println!("[sc-10578] wrote images to {}", dir.display());

    let (s_plain, s_styled) = (grayscale_std(&plain.pixels), grayscale_std(&styled.pixels));
    println!("[sc-10578] grayscale std: q4 plain = {s_plain:.2}, q4 + style LoRA = {s_styled:.2}");

    // Both must be real images, not noise or a blank field.
    assert!(
        s_plain > 8.0,
        "q4 plain output is near-blank (std {s_plain:.2})"
    );
    assert!(
        s_styled > 8.0,
        "q4 + style LoRA output is near-blank (std {s_styled:.2}) — injection likely broken"
    );

    // The LoRA must actually change the image. Same prompt, same seed, same schedule: any difference
    // is the adapter. (A silently-inert adapter — the sc-10274 class — would give ~0 here.)
    let diff = mean_abs_diff(&plain.pixels, &styled.pixels);
    println!("[sc-10578] mean |Δpixel| plain vs styled = {diff:.2} / 255");
    assert!(
        diff > 2.0,
        "the style LoRA left the q4 output essentially unchanged (mean |Δ| {diff:.2}) — it is inert"
    );
}
