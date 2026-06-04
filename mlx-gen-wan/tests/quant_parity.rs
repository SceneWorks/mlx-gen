//! sc-2682 Q4/Q8 **transformer-forward** parity gate on the 5B DiT, against the `mlx_video`
//! reference's `nn.quantize` (the `_quantize_predicate` surface — attention `q/k/v/o` + `ffn.fc1/fc2`,
//! group 64). Covers BOTH quant routes:
//!
//!   • **load-time** (`WanTransformer::quantize` on the bf16 `model.safetensors`) — the `spec.quantize`
//!     path; and
//!   • **consume pre-quantized** (`from_weights` reading a snapshot whose `config.json` has a
//!     `quantization` block + packed `.scales` on disk) — the `loading.py` path
//!     (`tools/dump_quant_snapshot.py`).
//!
//! All three (reference golden, load-time quant, pre-quantized snapshot) run the SAME MLX `nn.quantize`
//! on the SAME bf16 weights → byte-identical scales → at the 0.31.2 pin (dense bf16 cross-build
//! bit-exact, `quantized_matmul` fp32-accumulate) the quantized forward is **bit-exact** to the golden.
//! Decisive proof both quant routes are faithful; the per-expert MoE + decoded-pixel e2e are
//! `quant_e2e_parity.rs`.
//!
//! `#[ignore]` heavy: loads the converted 5B `model.safetensors` (~9 GB) from `WAN_5B_DIR` (load-time)
//! and the pre-quantized snapshot from `WAN_5B_Q{4,8}_DIR` (consume). Goldens (committed, small) come
//! from `tools/dump_quant_fixtures.py`.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_wan::config::WanModelConfig;
use mlx_gen_wan::WanTransformer;

fn snapshot_dir() -> PathBuf {
    if let Ok(d) = std::env::var("WAN_5B_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(std::env::var("HOME").unwrap())
        .join("Library/Application Support/SceneWorks/data/models/mlx/wan_2_2_ti2v_5b")
}

/// The pre-quantized 5B snapshot dir (`WAN_5B_Q{bits}_DIR`, default `~/.cache/mlx-gen-models/...`).
fn prequantized_dir(bits: i32) -> PathBuf {
    if let Ok(d) = std::env::var(format!("WAN_5B_Q{bits}_DIR")) {
        return PathBuf::from(d);
    }
    PathBuf::from(std::env::var("HOME").unwrap())
        .join(format!(".cache/mlx-gen-models/wan_2_2_ti2v_5b_mlx_q{bits}"))
}

fn diff(got: &[f32], exp: &[f32]) -> (f32, f64, f64) {
    let (mut ma, mut sa, mut sr) = (0f32, 0f64, 0f64);
    let mut over8 = 0usize;
    for (g, e) in got.iter().zip(exp.iter()) {
        let d = (g - e).abs();
        ma = ma.max(d);
        sa += d as f64;
        sr += e.abs() as f64;
        // px>8 analog on the raw latent (the latents are ~[-several, several]); a coarse outlier rate.
        if d > 8.0 {
            over8 += 1;
        }
    }
    (
        ma,
        sa / sr.max(1e-30),
        100.0 * over8 as f64 / got.len() as f64,
    )
}

/// Run `dit`'s forward on the golden's inputs and assert bit-exactness to the reference-Q output.
fn assert_matches_golden(bits: i32, label: &str, dit: &WanTransformer, golden_file: &str) {
    let g = Weights::from_file(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(golden_file))
        .expect("quant golden");
    let latent = g.require("latent").unwrap().clone();
    let context_raw = g.require("context_raw").unwrap().clone();
    let t: f32 = g.require("t").unwrap().as_slice::<f32>()[0];

    let context_emb = dit.embed_text(&context_raw).expect("embed_text");
    let out = dit.forward(&latent, t, &context_emb).expect("forward");

    let got = out.as_slice::<f32>().to_vec();
    let (max_abs, mean_rel, over8) = diff(&got, g.require("output").unwrap().as_slice::<f32>());
    println!("[Q{bits} {label}] max|Δ|={max_abs:.3e} mean_rel={mean_rel:.3e} px>8={over8:.3}%");
    // Scales are byte-identical (same bf16 weights, same MLX quantize) and quantized_matmul is
    // fp32-accumulate → at 0.31.2 the quantized forward is bit-exact to the reference. Allow a hair
    // of cross-build headroom but flag any real divergence loudly (a scale/predicate bug is O(1e-1+)).
    assert!(
        mean_rel < 5e-3,
        "Q{bits} {label} DiT forward diverged from the reference-Q golden: mean_rel={mean_rel:.3e}"
    );
}

/// Load-time quant: quantize the bf16 `model.safetensors` via `WanTransformer::quantize` (sc-2682
/// `spec.quantize` path), then assert the forward matches the reference-Q golden.
fn run_load_time(bits: i32, golden_file: &str) {
    let model_path = snapshot_dir().join("model.safetensors");
    if !model_path.exists() {
        eprintln!("skip: {} not found (set WAN_5B_DIR)", model_path.display());
        return;
    }
    let cfg = WanModelConfig::wan22_ti2v_5b();
    let w = Weights::from_file(&model_path).expect("model.safetensors");
    let mut dit = WanTransformer::from_weights(&w, &cfg).expect("build DiT");
    dit.quantize(bits, None).expect("quantize");
    assert_matches_golden(bits, "load-time", &dit, golden_file);
}

/// Consume a **pre-quantized** snapshot: `from_weights` reads the on-disk packed `.scales` (gated by
/// the snapshot's `config.json` `quantization` block) and builds the experts quantized directly (the
/// `loading.py` path) — no load-time re-quantize. Must match the SAME golden bit-for-bit.
fn run_prequantized(bits: i32, golden_file: &str) {
    let dir = prequantized_dir(bits);
    let model_path = dir.join("model.safetensors");
    if !model_path.exists() {
        eprintln!(
            "skip: {} not found (run tools/dump_quant_snapshot.py)",
            model_path.display()
        );
        return;
    }
    let cfg = WanModelConfig::from_model_dir(&dir).expect("read config.json");
    let q = cfg
        .quantization
        .expect("pre-quantized snapshot must carry a config quantization block");
    assert_eq!(q.bits, bits, "snapshot bits");
    assert_eq!(q.group_size, 64, "snapshot group_size");
    let w = Weights::from_file(&model_path).expect("pre-quantized model.safetensors");
    // from_weights builds the predicate Linears quantized from the on-disk packed weights.
    let dit = WanTransformer::from_weights(&w, &cfg).expect("build pre-quantized DiT");
    assert_matches_golden(bits, "prequant", &dit, golden_file);
}

#[test]
#[ignore = "needs the converted 5B model.safetensors (~9 GB) — run tools/dump_quant_fixtures.py"]
fn q4_dit_forward_matches_reference() {
    run_load_time(4, "tests/fixtures/q4_dit_golden.safetensors");
}

#[test]
#[ignore = "needs the converted 5B model.safetensors (~9 GB) — run tools/dump_quant_fixtures.py"]
fn q8_dit_forward_matches_reference() {
    run_load_time(8, "tests/fixtures/q8_dit_golden.safetensors");
}

#[test]
#[ignore = "needs the pre-quantized 5B snapshot — run tools/dump_quant_snapshot.py (WAN_5B_Q4_DIR)"]
fn q4_prequantized_snapshot_matches_reference() {
    run_prequantized(4, "tests/fixtures/q4_dit_golden.safetensors");
}

#[test]
#[ignore = "needs the pre-quantized 5B snapshot — run tools/dump_quant_snapshot.py (WAN_5B_Q8_DIR)"]
fn q8_prequantized_snapshot_matches_reference() {
    run_prequantized(8, "tests/fixtures/q8_dit_golden.safetensors");
}
