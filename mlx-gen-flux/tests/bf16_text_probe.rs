//! sc-2345: is the bf16 text-path garbage the pinned-build GEMM bug, or my code?
//! Uses the SAME mean-rel metric as `bf16_matmul_sweep.rs` and cross-checks the harness against
//! known tripwire cells. Run: cargo test -p mlx-gen-flux --test bf16_text_probe -- --ignored --nocapture

use mlx_rs::ops::matmul;
use mlx_rs::random;
use mlx_rs::{Array, Dtype};

/// Mean-absolute relative error sum|a-b|/sum|b| (the tripwire's metric).
fn mean_rel(a: &Array, b: &Array) -> f64 {
    let n = b.shape().iter().product::<i32>();
    let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let num: f64 = a.iter().zip(b).map(|(x, y)| (*x - *y).abs() as f64).sum();
    let den: f64 = b.iter().map(|y| y.abs() as f64).sum();
    num / den
}

fn probe(label: &str, a_shape: &[i32], b_shape: &[i32]) {
    let a = random::normal::<f32>(a_shape, None, None, Some(&random::key(0).unwrap())).unwrap();
    let b = random::normal::<f32>(b_shape, None, None, Some(&random::key(1).unwrap())).unwrap();
    let reff = matmul(&a, &b).unwrap();
    let bf16 = matmul(
        a.as_dtype(Dtype::Bfloat16).unwrap(),
        b.as_dtype(Dtype::Bfloat16).unwrap(),
    )
    .unwrap();
    println!(
        "{label:22} a={a_shape:?} b={b_shape:?}  mean_rel={:.4}",
        mean_rel(&bf16, &reff)
    );
}

#[test]
#[ignore = "diagnostic probe"]
fn bf16_matmul_text_shapes() {
    println!("-- cross-check vs tripwire (M>=2 & K<=512 = garbage; else safe) --");
    probe("xcheck_safe_K1024", &[256, 1024], &[1024, 1024]); // tripwire: SAFE
    probe("xcheck_garbage_K256", &[256, 256], &[256, 1024]); // tripwire: GARBAGE
    probe("xcheck_safe_K3072", &[256, 3072], &[3072, 1024]); // tripwire: SAFE
    println!("-- text-encoder shapes (N varied — tripwire fixed N=1024) --");
    // T5 attention scores / *v (K<=512 -> expected garbage zone)
    probe("T5_scores_K64", &[1, 64, 256, 64], &[1, 64, 64, 256]);
    probe("T5_attn_v_K256", &[1, 64, 256, 256], &[1, 64, 256, 64]);
    // T5 projections (K=4096, N=4096) and FF (K=4096->10240, K=10240->4096)
    probe("T5_proj_K4096_N4096", &[256, 4096], &[4096, 4096]);
    probe("T5_ff_wi_K4096_N10240", &[256, 4096], &[4096, 10240]);
    probe("T5_ff_wo_K10240_N4096", &[256, 10240], &[10240, 4096]);
    // CLIP (M=77): proj K=768 N=768, fc1 K=768 N=3072, fc2 K=3072 N=768
    probe("CLIP_proj_K768_N768", &[77, 768], &[768, 768]);
    probe("CLIP_fc1_K768_N3072", &[77, 768], &[768, 3072]);
    probe("CLIP_fc2_K3072_N768", &[77, 3072], &[3072, 768]);
    // Same K/M as CLIP_proj but vary N, to isolate whether N matters
    probe("CLIP_K768_N768", &[77, 768], &[768, 768]);
    probe("CLIP_K768_N1536", &[77, 768], &[768, 1536]);
    probe("CLIP_K768_N3072", &[77, 768], &[768, 3072]);
}
