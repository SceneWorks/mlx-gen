//! sc-2714: tripwire verifying the NAX 16-bit DENSE Metal GEMM is correct on the pinned build.
//!
//! BACKGROUND. On the pmetal NAX builds (macOS-26 / Metal-400), upstream MLX's
//! `matmul(16bit, 16bit)` — both operands bf16 *or* both f16 — returned GARBAGE (mean-relative
//! error vs an f32 reference ~0.3–2.3: right scale, uncorrelated, no NaN) for `M >= 2` with
//! `K <= 512`, plus very large `M`, because `matmul.cpp` routed those shapes into the broken
//! `steel_gemm_fused_nax_*` matrix-unit kernels (lmstudio#1356, mlx#3196/#3337). `M = 1` (gemv),
//! f32, and `quantized_matmul` (fp32 accumulation, mlx#963) were always correct. This was the root
//! of the FLUX Rust↔fork divergence and forced f32 detours in every adapter/embedder bf16 path.
//!
//! FIX (sc-2714). Our `michaeltrefry/mlx-rs` fork patches `mlx/backend/metal/matmul.cpp` to gate
//! every NAX GEMM dispatch site to f32/TF32 only (`enable_tf32() && dtype == float32`), so 16-bit
//! operands fall through to the correct non-NAX `steel_gemm_fused`. The f32/TF32 NAX path (the 2.7×
//! DiT speedup) is untouched. TEMPORARY carry until upstream fixes the NAX 16-bit kernel; see memory
//! `pmetal-mlx-bf16-matmul-bug`.
//!
//! THIS TEST is the per-build guarantor that the patch actually applied: it sweeps the former
//! garbage zone and asserts 16-bit dense is now correct (≈16-bit rounding, not garbage). It FAILS on
//! a NAX build that is MISSING the patch (e.g. the FetchContent `git apply` silently no-op'd) or if a
//! future MLX bump reintroduces the broken dispatch. On non-NAX builds 16-bit dense uses correct
//! fallback kernels, so it passes there too. Needs no weights, only MLX. When upstream fixes the NAX
//! kernel and we drop the fork patch, this keeps passing (just stops being load-bearing).

use mlx_rs::{ops::matmul, random, Array, Dtype};

/// Mean-absolute relative error: sum|a-b| / sum|b|, computed in f32.
fn rel(a: &Array, b: &Array) -> f64 {
    let n = b.shape().iter().product::<i32>();
    let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let num: f64 = a.iter().zip(b).map(|(x, y)| (*x - *y).abs() as f64).sum();
    let den: f64 = b.iter().map(|y| y.abs() as f64).sum();
    num / den
}

// Always-on guard: with the sc-2714 fork patch, 16-bit dense is correct across the whole grid, so
// this asserts correctness on every build (NAX-patched or non-NAX). On a NAX build whose MLX is
// missing the patch it (rightly) FAILS. Run: `cargo test -p mlx-gen-qwen-image --release --test
// bf16_matmul_sweep -- --nocapture`.
#[test]
fn nax_16bit_dense_gemm_is_patched() {
    // Distinct keys for the two operands so no (M,K)==(K,N) cell degenerates to A == B.
    let ka = random::key(0).unwrap();
    let kb = random::key(1).unwrap();
    let n = 1024i32; // out_features.
    let ms = [1, 2, 4, 16, 256, 1024];
    let ks = [64, 128, 256, 512, 1024, 3072];

    // Former garbage zone = M>=2 AND (K<=512 OR M==1024). Post-patch every cell must be correct.
    let mut worst_former_garbage = 0.0f64; // max over former-garbage cells — must be LOW (patched)
    let mut worst_safe = 0.0f64; // max over always-safe cells — must be LOW (kernel correct)
    println!("  *=former-garbage-zone (M>=2 & (K<=512 or M==1024))   N={n}");
    println!("    M     K      bf16      f16");
    for &m in &ms {
        for &k in &ks {
            let a = random::normal::<f32>(&[m, k], None, None, Some(&ka)).unwrap();
            let b = random::normal::<f32>(&[k, n], None, None, Some(&kb)).unwrap();
            let reff = matmul(&a, &b).unwrap();
            let bf16 = matmul(
                a.as_dtype(Dtype::Bfloat16).unwrap(),
                b.as_dtype(Dtype::Bfloat16).unwrap(),
            )
            .unwrap();
            let f16 = matmul(
                a.as_dtype(Dtype::Float16).unwrap(),
                b.as_dtype(Dtype::Float16).unwrap(),
            )
            .unwrap();
            let r_bf16 = rel(&bf16, &reff);
            let r_f16 = rel(&f16, &reff);
            let former_garbage = m >= 2 && (k <= 512 || m == 1024);
            let mark = if former_garbage { '*' } else { ' ' };
            println!("{mark} {m:5} {k:5}  {r_bf16:8.4}  {r_f16:8.4}");
            // bf16 and f16 share the dispatch; take the WORSE of the two as the signal.
            let cell = r_bf16.max(r_f16);
            if former_garbage {
                worst_former_garbage = worst_former_garbage.max(cell);
            } else {
                worst_safe = worst_safe.max(cell);
            }
        }
    }
    println!(
        "max former-garbage rel: {worst_former_garbage:.4}   max safe-zone rel: {worst_safe:.4}"
    );

    // Always-safe cells (M=1 gemv, large-K split-K): the kernel was always correct here.
    assert!(
        worst_safe < 0.05,
        "a 16-bit GEMM safe cell diverged ({worst_safe:.4} ≥ 0.05) — unexpected; re-characterize."
    );
    // GUARANTOR: the former garbage zone is now correct. If this fails on a NAX build, the sc-2714
    // patch did NOT take effect (check the fork's combined.patch / FetchContent `git apply`) or a
    // future MLX reintroduced the broken NAX 16-bit dispatch.
    assert!(
        worst_former_garbage < 0.05,
        "NAX 16-bit dense GEMM is GARBAGE again ({worst_former_garbage:.4} ≥ 0.05): the sc-2714 \
         matmul.cpp gate is not in effect. Verify the mlx-rs fork patch applied to the MLX build."
    );
}
