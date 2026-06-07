//! sc-3071 — IDFormer perceiver-resampler parity vs the torch f32 reference.
//!
//! Golden: `tools/dump_idformer_golden.py` (loads `pulid_encoder.*` into the vendored torch
//! `IDFormer`, runs f32 on deterministic `id_cond` + 5 EVA hidden states). Gate is cosine-primary:
//! torch-CPU-f32 vs MLX-Metal-f32 leaves a small reduced-precision floor over the 10-layer stack;
//! cos≈1.0 is the structural-correctness signal.
//!
//! Run:
//!   cargo test -p mlx-gen-pulid --release --test idformer_parity -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_gen_pulid::idformer::{IdFormer, IdFormerConfig};
use mlx_rs::Array;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/idformer_golden.safetensors"
);

fn golden() -> Weights {
    Weights::from_file(GOLDEN).unwrap_or_else(|e| {
        panic!("missing {GOLDEN}: {e}\nRun tools/dump_idformer_golden.py first (see file header).")
    })
}

fn slice(a: &Array) -> Vec<f32> {
    let n: i32 = a.shape().iter().product();
    a.reshape(&[n])
        .unwrap()
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

fn rel_errors(got: &Array, want: &Array) -> (f32, f32) {
    let a = slice(got);
    let b = slice(want);
    assert_eq!(a.len(), b.len());
    let peak_ref = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let max_diff = a
        .iter()
        .zip(&b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let sum_ref: f64 = b.iter().map(|&v| v.abs() as f64).sum();
    let sum_diff: f64 = a.iter().zip(&b).map(|(&x, &y)| (x - y).abs() as f64).sum();
    (max_diff / peak_ref, (sum_diff / sum_ref) as f32)
}

fn cosine(got: &Array, want: &Array) -> f32 {
    let a = slice(got);
    let b = slice(want);
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(&b) {
        dot += x as f64 * y as f64;
        na += (x as f64).powi(2);
        nb += (y as f64).powi(2);
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

#[test]
#[ignore = "needs local golden from tools/dump_idformer_golden.py"]
fn idformer_matches_torch_f32() {
    let g = golden();
    let model = IdFormer::from_weights(&g, "pulid_encoder", IdFormerConfig::default()).unwrap();
    let id_cond = g.require("id_cond").unwrap();
    let hidden: Vec<Array> = (0..5)
        .map(|i| g.require(&format!("hidden_{i}")).unwrap().clone())
        .collect();
    let got = model.forward(id_cond, &hidden).unwrap();
    assert_eq!(got.shape(), &[1, 32, 2048]);

    let want = g.require("id_embedding").unwrap();
    let (peak, mean) = rel_errors(&got, want);
    let cos = cosine(&got, want);
    println!("id_embedding: peak-rel {peak:.3e} mean-rel {mean:.3e} cos {cos:.6}");
    assert!(cos > 0.9995, "id_embedding cosine {cos:.6}");
    assert!(mean < 1.0e-2, "id_embedding mean-rel {mean:.3e}");
}

#[test]
#[ignore = "needs local golden from tools/dump_idformer_golden.py"]
fn idformer_bf16_floor() {
    let g = golden();
    let mut gw = golden();
    gw.cast_all(mlx_rs::Dtype::Bfloat16).unwrap();
    let model = IdFormer::from_weights(&gw, "pulid_encoder", IdFormerConfig::default()).unwrap();
    let hidden: Vec<Array> = (0..5)
        .map(|i| gw.require(&format!("hidden_{i}")).unwrap().clone())
        .collect();
    let got = model
        .forward(gw.require("id_cond").unwrap(), &hidden)
        .unwrap();
    let cos = cosine(&got, g.require("id_embedding").unwrap());
    println!("bf16 id_embedding: cos {cos:.6}");
    assert!(cos > 0.99, "bf16 id_embedding cosine {cos:.6}");
}
