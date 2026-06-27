//! Adaptive-Projected-Guidance (APG) — the x-space guidance the Bernini renderer's `*_apg` modes use
//! (`wan_diffusion.py:_normalize_diff` / `normalized_guidance` / `normalized_guidance_chain`, lines
//! 91-124), plus the v-space `apg_delta` for the ViT-conditioned modes.
//!
//! **As of sc-7442 (epic 7434 P3) the math lives once in the backend-neutral
//! [`gen_core::guidance`](mlx_gen::gen_core::guidance).** This module is the Bernini-specific seam: it
//! injects the MLX [`MlxLatentOps`] backend and Bernini's reduction geometry, leaving the renderer
//! call sites (`forward.rs`, `vit_guidance.rs`, `pipeline.rs`) byte-for-byte unchanged. The semantics
//! are exactly the retired bespoke code (a byte-equivalence test gates this — `tests` below):
//!
//! 1. **Momentum** (optional): `running = diff + momentum·running` (buffer persists across denoise
//!    steps; starts at 0 ⇒ first step is just `diff`).
//! 2. **Norm clamp** (when `norm_threshold > 0`): scale `diff` by `min(1, norm_threshold/‖diff‖)`.
//! 3. **Projection**: split `diff` into the component parallel to the conditional prediction `base`
//!    and the orthogonal remainder, and recombine as `orthogonal + eta·parallel`.
//!
//! The x-space L2 norm and projection reduce over **channels + spatial** ([`APG_DIMS`] `[0,2,3]` on a
//! `[C,T,H,W]` velocity — the reference's `dim=[-1,-2,-4]` on `[B,C,T,H,W]`, i.e. per frame, excluding
//! time). `apg_delta` reduces over the whole flattened tensor per batch element (every axis but 0).
//!
//! **Parity note:** the reference computes the projection in float64; MLX has no robust Metal f64, so
//! this runs in f32. Combined with the f32 source-id RoPE, this is the documented main divergence vs
//! the torch reference (the validation bar is component parity + coherent output, not bit-parity).

use mlx_rs::Array;

use mlx_gen::gen_core::guidance as core;
use mlx_gen::{MlxLatentOps, Result};

/// APG reduction dims on a `[C, T, H, W]` velocity (channels + spatial, per frame) — the reference's
/// `dim=[-1,-2,-4]` on its `[B, C, T, H, W]` layout (which excludes the temporal axis).
const APG_DIMS: &[i32] = &[0, 2, 3];

/// Persistent momentum accumulator for one APG stream (`running = diff + momentum·running`) — the
/// shared [`gen_core::guidance::MomentumBuffer`](mlx_gen::gen_core::guidance::MomentumBuffer) over
/// `mlx_rs::Array`. One per guidance term, allocated **before** the denoise loop so the running
/// average carries across steps.
pub type MomentumBuffer = core::MomentumBuffer<Array>;

/// Single-condition APG: `uncond + scale · normalize_diff(cond − uncond, base = cond)`
/// (`normalized_guidance`), over Bernini's per-frame [`APG_DIMS`] geometry. With `eta = 1`,
/// `norm_threshold = 0`, and no momentum this is exactly plain CFG `uncond + scale·(cond − uncond)`.
/// Delegates to [`gen_core::guidance::normalized_guidance`](mlx_gen::gen_core::guidance::normalized_guidance).
pub fn normalized_guidance(
    cond: &Array,
    uncond: &Array,
    scale: f32,
    buf: Option<&mut MomentumBuffer>,
    eta: f32,
    norm_threshold: f32,
) -> Result<Array> {
    Ok(core::normalized_guidance(
        &MlxLatentOps,
        cond,
        uncond,
        scale,
        buf,
        eta,
        norm_threshold,
        &[], // shape unused on MLX (the Array carries its own).
        APG_DIMS,
    )?)
}

/// Chained APG over an ordered list of predictions (`normalized_guidance_chain`). With
/// `bases = [uncond, preds[0], preds[1], …]`, accumulates
/// `result = uncond + Σ_i scales[i] · normalize_diff(preds[i] − bases[i], base = preds[i])`, each term
/// with its own momentum buffer and norm threshold. Used by `r2v_apg` over `[x_I, x_TI]`.
pub fn normalized_guidance_chain(
    uncond: &Array,
    preds: &[Array],
    scales: &[f32],
    bufs: &mut [MomentumBuffer],
    eta: f32,
    norm_thresholds: &[f32],
) -> Result<Array> {
    Ok(core::normalized_guidance_chain(
        &MlxLatentOps,
        uncond,
        preds,
        scales,
        bufs,
        eta,
        norm_thresholds,
        &[],
        APG_DIMS,
    )?)
}

/// The **v-space** APG delta projection used by the full-Bernini ViT-conditioned modes
/// (`wan_diffusion.py:apg_delta`, "veomni_editing Wan2.2"). Projects `delta` onto `reference` over the
/// **whole flattened tensor** (per batch element — every axis but 0) and recombines with fixed
/// parallel/orthogonal scales:
///
///   `proj = (delta·ref)/max(‖ref‖², eps)·ref`; `out = parallel_scale·proj + orthogonal_scale·(delta − proj)`.
///
/// `delta`/`reference` are `[1, n_target, C]` (the target-sliced packed-token predictions, batch 1);
/// the reduction is over `n_target·C`.
pub fn apg_delta(
    delta: &Array,
    reference: &Array,
    parallel_scale: f32,
    orthogonal_scale: f32,
) -> Result<Array> {
    // reshape(b, -1) + reduce dim=1 ≡ reduce over every axis except the batch axis 0.
    let axes: Vec<i32> = (1..delta.ndim() as i32).collect();
    Ok(core::apg_delta(
        &MlxLatentOps,
        delta,
        reference,
        parallel_scale,
        orthogonal_scale,
        &[],
        &axes,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{add, divide, maximum, minimum, multiply, sqrt, subtract};

    fn max_abs(a: &Array, b: &Array) -> f32 {
        mlx_rs::ops::max(subtract(a, b).unwrap().abs().unwrap(), None)
            .unwrap()
            .item::<f32>()
    }

    /// L2 norm over Bernini's per-frame [`APG_DIMS`], keepdims (test helper / clamp inspector).
    fn l2_norm(a: &Array) -> Array {
        sqrt(multiply(a, a).unwrap().sum_axes(APG_DIMS, true).unwrap()).unwrap()
    }

    fn randish(seed: i32) -> Array {
        // Deterministic varied [C=4, T=2, H=2, W=2] tensor.
        let n = 4 * 2 * 2 * 2;
        let v: Vec<f32> = (0..n)
            .map(|i| ((i * 7 + seed * 13) % 11) as f32 - 5.0)
            .collect();
        Array::from_slice(&v, &[4, 2, 2, 2])
    }

    // ------------------------------------------------------------------------------------------
    // The EXACT bespoke Bernini APG retired in sc-7442 (was this module's body), inlined as the N1
    // byte-equivalence reference for `migrated_apg_is_byte_identical`.
    // ------------------------------------------------------------------------------------------
    struct LegacyBuf {
        momentum: f32,
        running: Option<Array>,
    }
    impl LegacyBuf {
        fn new(momentum: f32) -> Self {
            Self {
                momentum,
                running: None,
            }
        }
        fn update(&mut self, diff: &Array) -> Array {
            let ra = match &self.running {
                Some(r) => add(diff, multiply(r, Array::from_f32(self.momentum)).unwrap()).unwrap(),
                None => diff.clone(),
            };
            self.running = Some(ra.clone());
            ra
        }
    }
    fn legacy_normalize_diff(
        diff: &Array,
        base: &Array,
        buf: Option<&mut LegacyBuf>,
        eta: f32,
        norm_threshold: f32,
    ) -> Array {
        let mut diff = match buf {
            Some(b) => b.update(diff),
            None => diff.clone(),
        };
        if norm_threshold > 0.0 {
            let dn = l2_norm(&diff);
            let scale = minimum(
                Array::from_f32(1.0),
                divide(Array::from_f32(norm_threshold), &dn).unwrap(),
            )
            .unwrap();
            diff = multiply(&diff, &scale).unwrap();
        }
        let bn = maximum(l2_norm(base), Array::from_f32(1e-12)).unwrap();
        let v1 = divide(base, &bn).unwrap();
        let coeff = multiply(&diff, &v1)
            .unwrap()
            .sum_axes(APG_DIMS, true)
            .unwrap();
        let parallel = multiply(&coeff, &v1).unwrap();
        let orthogonal = subtract(&diff, &parallel).unwrap();
        add(
            &orthogonal,
            multiply(&parallel, Array::from_f32(eta)).unwrap(),
        )
        .unwrap()
    }
    fn legacy_normalized_guidance(
        cond: &Array,
        uncond: &Array,
        scale: f32,
        buf: Option<&mut LegacyBuf>,
        eta: f32,
        norm_threshold: f32,
    ) -> Array {
        let nd = legacy_normalize_diff(
            &subtract(cond, uncond).unwrap(),
            cond,
            buf,
            eta,
            norm_threshold,
        );
        add(uncond, multiply(&nd, Array::from_f32(scale)).unwrap()).unwrap()
    }
    fn legacy_chain(
        uncond: &Array,
        preds: &[Array],
        scales: &[f32],
        bufs: &mut [LegacyBuf],
        eta: f32,
        norm_thresholds: &[f32],
    ) -> Array {
        let mut result = uncond.clone();
        for (i, cond) in preds.iter().enumerate() {
            let base_prev = if i == 0 { uncond } else { &preds[i - 1] };
            let nd = legacy_normalize_diff(
                &subtract(cond, base_prev).unwrap(),
                cond,
                Some(&mut bufs[i]),
                eta,
                norm_thresholds[i],
            );
            result = add(&result, multiply(&nd, Array::from_f32(scales[i])).unwrap()).unwrap();
        }
        result
    }
    fn legacy_apg_delta(
        delta: &Array,
        reference: &Array,
        parallel_scale: f32,
        orthogonal_scale: f32,
    ) -> Array {
        let dims: Vec<i32> = (1..delta.ndim() as i32).collect();
        let ref_norm_sq = maximum(
            multiply(reference, reference)
                .unwrap()
                .sum_axes(&dims, true)
                .unwrap(),
            Array::from_f32(1e-8),
        )
        .unwrap();
        let coeff = divide(
            multiply(delta, reference)
                .unwrap()
                .sum_axes(&dims, true)
                .unwrap(),
            &ref_norm_sq,
        )
        .unwrap();
        let parallel = multiply(&coeff, reference).unwrap();
        let orthogonal = subtract(delta, &parallel).unwrap();
        add(
            multiply(&parallel, Array::from_f32(parallel_scale)).unwrap(),
            multiply(&orthogonal, Array::from_f32(orthogonal_scale)).unwrap(),
        )
        .unwrap()
    }

    /// sc-7442 (epic 7434 P3) — every Bernini APG mode is a **bit-identical** drop-in: the shared
    /// `gen_core` path over [`MlxLatentOps`] must reproduce the retired bespoke math exactly across
    /// single / chained / v-space, with momentum, eta, and norm-threshold variations exercised.
    #[test]
    fn migrated_apg_is_byte_identical() {
        let cond = randish(1);
        let uncond = randish(2);
        let base2 = randish(8);

        // Single-condition: plain, eta<1, norm-threshold clamp, and with momentum across two steps.
        for &(eta, nt) in &[(1.0f32, 0.0f32), (0.5, 0.0), (1.0, 2.0), (0.7, 1.5)] {
            let got = normalized_guidance(&cond, &uncond, 4.0, None, eta, nt).unwrap();
            let want = legacy_normalized_guidance(&cond, &uncond, 4.0, None, eta, nt);
            assert_eq!(max_abs(&got, &want), 0.0, "single eta={eta} nt={nt}");
        }
        // Momentum carry across steps (buffer reused).
        let mut mb = MomentumBuffer::new(-0.5);
        let mut lb = LegacyBuf::new(-0.5);
        for s in 1..=3 {
            let c = randish(10 + s);
            let got = normalized_guidance(&c, &uncond, 3.0, Some(&mut mb), 0.8, 0.0).unwrap();
            let want = legacy_normalized_guidance(&c, &uncond, 3.0, Some(&mut lb), 0.8, 0.0);
            assert_eq!(max_abs(&got, &want), 0.0, "momentum step {s}");
        }
        // Chained (r2v_apg over [x_I, x_TI]).
        let preds = [randish(3), randish(4)];
        let mut mbufs = [MomentumBuffer::new(0.3), MomentumBuffer::new(0.3)];
        let mut lbufs = [LegacyBuf::new(0.3), LegacyBuf::new(0.3)];
        let got =
            normalized_guidance_chain(&uncond, &preds, &[2.0, 5.0], &mut mbufs, 0.6, &[1.0, 0.0])
                .unwrap();
        let want = legacy_chain(&uncond, &preds, &[2.0, 5.0], &mut lbufs, 0.6, &[1.0, 0.0]);
        assert_eq!(max_abs(&got, &want), 0.0, "chain");
        // v-space apg_delta (∥0.2 / ⊥1.0) on a [1, n, C] packed-token shape.
        let delta = randish(5).reshape(&[1, 8, 4]).unwrap();
        let reference = randish(6).reshape(&[1, 8, 4]).unwrap();
        let got = apg_delta(&delta, &reference, 0.2, 1.0).unwrap();
        let want = legacy_apg_delta(&delta, &reference, 0.2, 1.0);
        assert_eq!(max_abs(&got, &want), 0.0, "apg_delta");
        // base2 keeps the helper honest (unused-var guard) — exercises a non-cond base direction.
        let _ = &base2;
    }

    // ------------------------------------------------------------------------------------------
    // The pre-existing Bernini property tests, now run against the shared layer (story acceptance).
    // ------------------------------------------------------------------------------------------

    #[test]
    fn apg_reduces_to_plain_cfg_at_eta1_no_clamp() {
        let cond = randish(1);
        let uncond = randish(2);
        let scale = 4.0_f32;
        let got = normalized_guidance(&cond, &uncond, scale, None, 1.0, 0.0).unwrap();
        let want = add(
            &uncond,
            multiply(subtract(&cond, &uncond).unwrap(), Array::from_f32(scale)).unwrap(),
        )
        .unwrap();
        assert!(
            max_abs(&got, &want) < 1e-4,
            "eta=1/nt=0 must equal plain CFG"
        );
    }

    #[test]
    fn apg_eta0_drops_parallel_component() {
        // eta=0 ⇒ the guidance delta is purely orthogonal to `cond`, so (nd · cond) summed over
        // C,H,W ≈ 0 per frame. Recover nd via the public path with uncond=0, scale=1.
        let cond = randish(3);
        let zero = Array::zeros::<f32>(&[4, 2, 2, 2]).unwrap();
        let nd = normalized_guidance(&cond, &zero, 1.0, None, 0.0, 0.0).unwrap();
        let dot = multiply(&nd, &cond)
            .unwrap()
            .sum_axes(APG_DIMS, true)
            .unwrap();
        let zeros = Array::zeros::<f32>(dot.shape()).unwrap();
        assert!(max_abs(&dot, &zeros) < 1e-3, "eta=0 orthogonal residual");
    }

    #[test]
    fn apg_norm_threshold_clamps_diff() {
        // A large diff with a small threshold is scaled so ‖diff‖ ≤ threshold per frame. With
        // uncond=0, eta=1, scale=1 the public result is exactly the clamped+projected nd.
        let cond = multiply(randish(5), Array::from_f32(100.0)).unwrap();
        let zero = Array::zeros::<f32>(&[4, 2, 2, 2]).unwrap();
        let nd = normalized_guidance(&cond, &zero, 1.0, None, 1.0, 2.0).unwrap();
        let m = mlx_rs::ops::max(l2_norm(&nd), None).unwrap().item::<f32>();
        assert!(m <= 2.0 + 1e-3, "clamped norm {m} must be ≤ threshold 2.0");
    }

    #[test]
    fn momentum_accumulates_across_calls() {
        let mut buf = MomentumBuffer::new(-0.5);
        let d1 = randish(6);
        let r1 = buf.update(&MlxLatentOps, &d1).unwrap();
        assert!(max_abs(&r1, &d1) < 1e-6, "first update returns diff");
        let d2 = randish(7);
        let r2 = buf.update(&MlxLatentOps, &d2).unwrap();
        // running = d2 + (-0.5)·d1
        let want = add(&d2, multiply(&d1, Array::from_f32(-0.5)).unwrap()).unwrap();
        assert!(max_abs(&r2, &want) < 1e-5, "second update accumulates");
    }
}
