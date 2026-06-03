//! sc-2643: FLUX.2-klein-9b Q4/Q8 quantization parity. The fork quantizes the WHOLE model
//! (transformer + Qwen3 TE + VAE) via `nn.quantize(predicate=hasattr to_quantized, bits)` at the
//! default `ModelConfig.precision=bf16` — every `nn.Linear` plus the TE token `nn.Embedding`,
//! group_size 64. The Rust port loads f32 but casts to bf16 before packing (sc-2604), so the
//! packing must be byte-identical to the fork's.
//!
//! `#[ignore]`d — needs the real FLUX.2-klein-9b snapshot + the goldens from
//! `tools/dump_flux2_quant_golden.py` (run once per bit-width: `BITS=8` and `BITS=4`).
//!   cd ~/repos/mflux && BITS=8 .venv/bin/python ~/repos/mlx-gen/tools/dump_flux2_quant_golden.py
//!   cargo test -p mlx-gen-flux2 --release --test quant_real_weights -- --ignored --nocapture
//!
//! Gates (honoring the sc-2532 false-green lesson — full path, not fed intermediates):
//!  1. **packing byte-parity** — the loaded Q8/Q4 `wq`/`scales`/`biases` of three representative
//!     modules (one per packing scenario: transformer Linear, TE Embedding, f32-loaded VAE Linear
//!     with bias) are bit-exact to the fork's `nn.quantize(bf16)`.
//!  2. **e2e render** — the public `load(Q).generate()` render vs the fork's `--quantize` decoded
//!     image (px>8). Rust runs f32 activations vs the fork's bf16, so this is a bounded coherence
//!     floor (like the dense e2e), not bit-parity; a wiring/scope bug would blow it past the floor.

use std::path::PathBuf;

use mlx_gen::image::decoded_to_image;
use mlx_gen::media::Image;
use mlx_gen::weights::Weights;
use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};
use mlx_gen_flux2::{load_text_encoder, load_transformer, load_vae};
use mlx_rs::ops::eq;
use mlx_rs::{Array, Dtype};

const PROMPT: &str = "a red fox resting in fresh snow under soft winter light";

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("MLX_GEN_FLUX2_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").expect("HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-9b/snapshots");
    std::fs::read_dir(&snaps)
        .expect("snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn golden(bits: i32) -> Weights {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../tools/golden/flux2_quant_q{bits}.safetensors"));
    Weights::from_file(&path).unwrap_or_else(|_| {
        panic!(
            "missing {} — run `BITS={bits} python tools/dump_flux2_quant_golden.py`",
            path.display()
        )
    })
}

fn all_eq(a: &Array, b: &Array) -> bool {
    a.shape() == b.shape()
        && a.dtype() == b.dtype()
        && eq(a, b).unwrap().all(None).unwrap().item::<bool>()
}

/// Assert the loaded `(wq, scales, biases)` of one probe module byte-match the fork golden.
fn check_probe(label: &str, got: (&Array, &Array, &Array, i32, i32), g: &Weights, bits: i32) {
    let (wq, scales, biases, gs, b) = got;
    assert_eq!(gs, 64, "{label}: group_size");
    assert_eq!(b, bits, "{label}: bits");
    assert!(
        all_eq(wq, g.require(&format!("{label}_weight")).unwrap()),
        "{label}: Q{bits} packed wq not byte-identical to fork nn.quantize(bf16)"
    );
    assert!(
        all_eq(scales, g.require(&format!("{label}_scales")).unwrap()),
        "{label}: Q{bits} scales not byte-identical (sc-2604 bf16-cast chokepoint)"
    );
    assert!(
        all_eq(biases, g.require(&format!("{label}_biases")).unwrap()),
        "{label}: Q{bits} biases not byte-identical"
    );
    println!("  ✓ {label}: Q{bits} wq/scales/biases byte-identical to fork");
}

#[test]
#[ignore = "needs real FLUX.2-klein-9b snapshot + tools/golden/flux2_quant_q{8,4}.safetensors"]
fn q8_q4_packing_byte_match() {
    let snap = snapshot();
    for bits in [8, 4] {
        let g = golden(bits);
        println!("FLUX.2 quant byte-parity Q{bits}:");

        let mut t = load_transformer(&snap).unwrap();
        t.quantize(bits).unwrap();
        check_probe("t_to_q", t.probe_quant_to_q().unwrap(), &g, bits);

        let mut te = load_text_encoder(&snap).unwrap();
        te.quantize(bits).unwrap();
        check_probe("te_embed", te.probe_quant_embed().unwrap(), &g, bits);

        let mut vae = load_vae(&snap).unwrap();
        vae.quantize(bits).unwrap();
        check_probe("vae_enc_q", vae.probe_quant_enc_q().unwrap(), &g, bits);
    }
}

fn px_gt8(a: &Image, b: &Image) -> f32 {
    assert_eq!(a.pixels.len(), b.pixels.len(), "image size mismatch");
    let n = a.pixels.len();
    let c = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .filter(|(x, y)| (**x as i32 - **y as i32).abs() > 8)
        .count();
    100.0 * c as f32 / n as f32
}

fn f32a(a: &Array) -> Array {
    a.as_dtype(Dtype::Float32).unwrap()
}

/// (peak_rel, mean_rel) of `a` vs reference `b`.
fn rel(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let a = f32a(a).reshape(&[n]).unwrap();
    let b = f32a(b).reshape(&[n]).unwrap();
    let (xs, ys) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = ys.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let mabs = (ys.iter().map(|y| y.abs()).sum::<f32>() / ys.len() as f32).max(1e-12);
    let max_d = xs
        .iter()
        .zip(ys)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let mean_d = xs.iter().zip(ys).map(|(x, y)| (x - y).abs()).sum::<f32>() / xs.len() as f32;
    (max_d / peak, mean_d / mabs)
}

/// Chaos-free correctness gate (the tight one): feed the Rust quantized transformer the fork's own
/// noise / prompt_embeds / ids and compare the step-0 velocity to the fork's Q-render `v0`. One
/// forward, no sampler → isolates the quantized transformer forward from the sampler's chaos. The
/// only difference left is Rust's f32 activations vs the fork's bf16 (the weights are byte-identical,
/// proven by `q8_q4_packing_byte_match`), so this is bounded and small; a quant scope/wiring bug
/// would blow it up. Honors the sc-2532 lesson by also keeping the full-path gates below.
fn v0_matches_fork(bits: i32) {
    let g = golden(bits);
    let mut t = load_transformer(&snapshot()).unwrap();
    t.quantize(bits).unwrap();
    let v = t
        .forward(
            g.require("noise0").unwrap(),
            g.require("prompt_embeds").unwrap(),
            g.require("latent_ids").unwrap(),
            g.require("text_ids").unwrap(),
            1000.0,
        )
        .unwrap();
    let (peak, mean) = rel(&v, g.require("v0").unwrap());
    println!("flux2 Q{bits} v0 (chaos-free quantized transformer): peak_rel={peak:.4} mean_rel={mean:.4} (f32-act vs fork bf16-act, byte-identical weights)");
    assert!(
        mean < 5e-2,
        "Q{bits} step-0 velocity diverged from the fork: mean_rel={mean} (quant forward bug?)"
    );
}

#[test]
#[ignore = "needs real FLUX.2-klein-9b snapshot + tools/golden/flux2_quant_q8.safetensors"]
fn q8_v0_matches_fork() {
    v0_matches_fork(8);
}

#[test]
#[ignore = "needs real FLUX.2-klein-9b snapshot + tools/golden/flux2_quant_q4.safetensors"]
fn q4_v0_matches_fork() {
    v0_matches_fork(4);
}

fn render(quant: Option<Quant>, size: u32) -> Image {
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    spec.quantize = quant;
    let gen = mlx_gen::load("flux2_klein_9b", &spec).unwrap();
    let req = GenerationRequest {
        prompt: PROMPT.into(),
        width: size,
        height: size,
        count: 1,
        seed: Some(0),
        steps: Some(4),
        ..Default::default()
    };
    let GenerationOutput::Images(mut images) = gen.generate(&req, &mut |_| {}).unwrap() else {
        panic!("expected images");
    };
    images.pop().unwrap()
}

/// Full-path render gate. Correctness is already carried by the byte-parity (exact weights) + v0
/// (faithful forward) gates above; this test characterizes the *render*, which goes through the
/// chaos-sensitive 4-step flow-match sampler. Two clean, build-internal facts are asserted:
///   * in-build Q-vs-dense (both f32 activations): quantizing the weights is a bounded perturbation
///     of the dense f32 render, and Q4 (4-bit) perturbs at least as much as Q8 (8-bit) — the
///     monotonicity a scope/wiring bug breaks.
///   * vs the fork's `--quantize` decoded image: REPORTED, with only a garbage floor asserted. Rust
///     runs f32 activations while the fork runs bf16 (identical weights, proven by byte-parity), so
///     this number is the f32-vs-bf16 composition difference amplified by the chaos sampler — large
///     at 256² (the FLUX low-res precision-chaos lesson), and it shrinks at 512² as the sampler
///     becomes less chaos-sensitive. It is a coherence check, not a parity claim (the dense e2e
///     forces the fork to f32 to dodge this; the quant path can't, since f32 changes the scales).
#[test]
#[ignore = "needs real FLUX.2-klein-9b snapshot + tools/golden/flux2_quant_q{8,4}.safetensors"]
fn quant_render_full_path() {
    let g8 = golden(8);
    let g4 = golden(4);

    // 256²: in-build perturbation of dense + the (chaos-amplified) vs-fork numbers.
    let dense = render(None, 256);
    let q8 = render(Some(Quant::Q8), 256);
    let q4 = render(Some(Quant::Q4), 256);
    let q8_vs_dense = px_gt8(&q8, &dense);
    let q4_vs_dense = px_gt8(&q4, &dense);
    let q8_fork_256 = px_gt8(
        &q8,
        &decoded_to_image(g8.require("decoded").unwrap()).unwrap(),
    );
    let q4_fork_256 = px_gt8(
        &q4,
        &decoded_to_image(g4.require("decoded").unwrap()).unwrap(),
    );
    println!("flux2 256² in-build (both f32-act): Q8-vs-dense {q8_vs_dense:.2}%  Q4-vs-dense {q4_vs_dense:.2}% px>8");
    println!("flux2 256² vs fork (f32-act vs bf16-act + cross-build, chaos-amplified): Q8 {q8_fork_256:.2}%  Q4 {q4_fork_256:.2}% px>8");

    // 512²: the story's higher-res check — less chaos, so the vs-fork gap shrinks toward parity.
    let q8_512 = render(Some(Quant::Q8), 512);
    let q4_512 = render(Some(Quant::Q4), 512);
    let q8_fork_512 = px_gt8(
        &q8_512,
        &decoded_to_image(g8.require("decoded_512").unwrap()).unwrap(),
    );
    let q4_fork_512 = px_gt8(
        &q4_512,
        &decoded_to_image(g4.require("decoded_512").unwrap()).unwrap(),
    );
    println!("flux2 512² vs fork (f32-act vs bf16-act + cross-build): Q8 {q8_fork_512:.2}%  Q4 {q4_fork_512:.2}% px>8");

    // Quantization is a bounded perturbation of the dense f32 render, monotone in bit-width.
    assert!(
        q4_vs_dense >= q8_vs_dense,
        "non-monotonic: Q4 {q4_vs_dense}% < Q8 {q8_vs_dense}% vs dense (wiring bug?)"
    );
    // Garbage floor: a scope/wiring bug (wrong module set quantized) renders incoherent (~100%).
    for (name, px) in [
        ("Q8-vs-dense-256", q8_vs_dense),
        ("Q4-vs-dense-256", q4_vs_dense),
        ("Q8-vs-fork-256", q8_fork_256),
        ("Q4-vs-fork-256", q4_fork_256),
        ("Q8-vs-fork-512", q8_fork_512),
        ("Q4-vs-fork-512", q4_fork_512),
    ] {
        assert!(
            px < 70.0,
            "{name} not coherent: {px}% px>8 (scope/wiring bug?)"
        );
    }
    // The chaos genuinely subsides with resolution: the vs-fork gap is smaller at 512² than 256².
    assert!(
        q8_fork_512 < q8_fork_256,
        "Q8 vs-fork did not shrink at 512² ({q8_fork_512}%) vs 256² ({q8_fork_256}%)"
    );
}
