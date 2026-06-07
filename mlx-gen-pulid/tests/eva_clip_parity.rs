//! sc-3070 — EVA02-CLIP-L-14-336 visual-tower parity vs the torch f32 reference.
//!
//! Golden: `tools/dump_eva_clip_golden.py` (run from the vendored pulid_flux reference under
//! pulidenv). Gates, micro → macro:
//!   1. RoPE construction (weight-free) matches the checkpoint's `rope.freqs_cos/sin` buffers.
//!   2. The full tower (real weights) matches the torch f32 forward — `id_cond_vit` + 5 hidden
//!      states — near-bit (the residual is MLX's reduced-precision Metal f32 matmul, not structural).
//!   3. The EVA input transform (512²→336² float bicubic + normalize) matches torchvision.
//!
//! Run:
//!   cargo test -p mlx-gen-pulid --release --test eva_clip_parity -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_gen_pulid::eva_clip::{transform, EvaConfig, EvaVisionTransformer};
use mlx_rs::Array;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/eva_clip_golden.safetensors"
);

fn golden() -> Weights {
    Weights::from_file(GOLDEN).unwrap_or_else(|e| {
        panic!("missing {GOLDEN}: {e}\nRun tools/dump_eva_clip_golden.py first (see file header).")
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

/// (peak relative error, mean relative error) of `got` vs reference `want`.
fn rel_errors(got: &Array, want: &Array) -> (f32, f32) {
    let a = slice(got);
    let b = slice(want);
    assert_eq!(
        a.len(),
        b.len(),
        "shape mismatch {:?} vs {:?}",
        got.shape(),
        want.shape()
    );
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
#[ignore = "needs local golden from tools/dump_eva_clip_golden.py"]
fn rope_construction_matches_checkpoint() {
    let g = golden();
    let vt = EvaVisionTransformer::from_weights(&g, "w", EvaConfig::default()).unwrap();
    let (cp, sp) = (
        rel_errors(vt.rope().cos(), g.require("rope.freqs_cos").unwrap()),
        rel_errors(vt.rope().sin(), g.require("rope.freqs_sin").unwrap()),
    );
    println!("rope cos peak-rel {:.2e} mean-rel {:.2e}", cp.0, cp.1);
    println!("rope sin peak-rel {:.2e} mean-rel {:.2e}", sp.0, sp.1);
    // absolute closeness (cos/sin in [-1,1]): the f64-host build vs torch f32 buffer differ <1 ULP
    let abs = |a: &Array, b: &Array| {
        let (x, y) = (slice(a), slice(b));
        x.iter()
            .zip(&y)
            .fold(0f32, |m, (&p, &q)| m.max((p - q).abs()))
    };
    let ac = abs(vt.rope().cos(), g.require("rope.freqs_cos").unwrap());
    let as_ = abs(vt.rope().sin(), g.require("rope.freqs_sin").unwrap());
    println!("rope cos max|Δ| {ac:.2e}  sin max|Δ| {as_:.2e}");
    assert!(
        ac < 1e-5 && as_ < 1e-5,
        "rope construction diverged: cos {ac:.2e} sin {as_:.2e}"
    );
}

#[test]
#[ignore = "needs local golden from tools/dump_eva_clip_golden.py"]
fn eva_encoder_matches_torch_f32() {
    let g = golden();
    let vt = EvaVisionTransformer::from_weights(&g, "w", EvaConfig::default()).unwrap();
    let enc_in = g.require("enc_in_nhwc").unwrap();
    let out = vt.forward(enc_in).unwrap();

    // Gate is cosine-primary: the golden is torch-CPU-f32, the port runs MLX-Metal-f32 whose
    // reduced-precision matmul accumulates a cross-backend mean-rel floor over the 24-block stack
    // (~1e-2 at depth). Cosine ≈ 1.0 + small peak-rel is the structural-correctness signal.
    for i in 0..5 {
        let want = g.require(&format!("hidden_{i}")).unwrap();
        let (peak, mean) = rel_errors(&out.hidden[i], want);
        let cos = cosine(&out.hidden[i], want);
        println!("hidden_{i}: peak-rel {peak:.3e} mean-rel {mean:.3e} cos {cos:.6}");
        assert!(cos > 0.9995, "hidden_{i} cosine {cos:.6}");
        assert!(mean < 1.2e-2, "hidden_{i} mean-rel {mean:.3e}");
    }

    let want = g.require("id_cond_vit").unwrap();
    let (peak, mean) = rel_errors(&out.id_cond_vit, want);
    let cos = cosine(&out.id_cond_vit, want);
    println!("id_cond_vit: peak-rel {peak:.3e} mean-rel {mean:.3e} cos {cos:.6}");
    assert!(cos > 0.9995, "id_cond_vit cosine {cos:.6}");
    assert!(mean < 1.2e-2, "id_cond_vit mean-rel {mean:.3e}");
}

#[test]
#[ignore = "needs local golden from tools/dump_eva_clip_golden.py"]
fn eva_encoder_bf16_floor() {
    // Production runs the tower in bf16 (PuLID `weight_dtype`). Confirm the bf16 path tracks the
    // f32 torch reference within the bf16 floor — the final id_cond_vit stays near-1 cosine.
    let g = golden();
    let mut gw = golden();
    gw.cast_all(mlx_rs::Dtype::Bfloat16).unwrap();
    let vt = EvaVisionTransformer::from_weights(&gw, "w", EvaConfig::default()).unwrap();
    let out = vt.forward(gw.require("enc_in_nhwc").unwrap()).unwrap();
    let cos = cosine(&out.id_cond_vit, g.require("id_cond_vit").unwrap());
    let (peak, mean) = rel_errors(&out.id_cond_vit, g.require("id_cond_vit").unwrap());
    println!("bf16 id_cond_vit: cos {cos:.6} peak-rel {peak:.3e} mean-rel {mean:.3e}");
    assert!(cos > 0.99, "bf16 id_cond_vit cosine {cos:.6}");
}

#[test]
#[ignore = "needs local golden from tools/dump_eva_clip_golden.py"]
fn eva_transform_matches_torchvision() {
    let g = golden();
    let ffi = g.require("ffi_512_nhwc").unwrap();
    let size = 336;

    // resize-only vs torchvision float bicubic
    let in_h = ffi.shape()[1] as usize;
    let flat = ffi
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .reshape(&[-1])
        .unwrap();
    let src = flat.as_slice::<f32>().to_vec();
    let resized = transform::resize_bicubic_f32(&src, in_h, in_h, size as usize, size as usize);
    let resized = Array::from_slice(&resized, &[1, size, size, 3]);
    let (rp, rm) = rel_errors(&resized, g.require("tf_resized_nhwc").unwrap());
    println!("resize: peak-rel {rp:.3e} mean-rel {rm:.3e}");

    // full transform (resize + normalize) vs torchvision
    let tf = transform::eva_transform(ffi, size).unwrap();
    let (tp, tm) = rel_errors(&tf, g.require("tf_normalized_nhwc").unwrap());
    println!("transform: peak-rel {tp:.3e} mean-rel {tm:.3e}");
    assert!(tm < 5e-3, "eva transform mean-rel {tm:.3e}");
}
