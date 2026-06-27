//! The guidance axis (epic 7434): the fourth orthogonal sampling layer — given the conditional and
//! unconditional model predictions, HOW they are combined. Backend-neutral, written once here and
//! delegated to by every engine (Option A), the same way the unified sampler/scheduler framework
//! (epic 7114) added the integration method and sigma schedule.
//!
//! Three pieces:
//! 1. [`GuidanceMethod`] — the curated, extensible policy vocabulary (`cfg` / `cfg_rescale` / `apg` /
//!    `cfg_pp`) the contract validates and the engines match on (sc-7436).
//! 2. [`GuidanceOps`] — the minimal **axis-parameterized** backend op extension (sc-7437). [`LatentOps`]
//!    is deliberately reduction-free; cfg_rescale and APG need L2 norms and projections whose reduction
//!    geometry differs per model (per-token `[-1]`, per-frame `[C,H,W]`, whole-flattened). So the
//!    reductions take an explicit `(shape, axes)` and return a FULL-shape broadcast result, keeping the
//!    guidance library itself purely elementwise. gen-core keeps its zero-tensor-dep invariant: the
//!    reference impl is over `Vec<f32>`; real backends supply MLX/candle impls (sc-7439/7440).
//! 3. The guidance library — [`cfg`], [`cfg_rescale`], and the full APG surface ([`MomentumBuffer`],
//!    [`normalized_guidance`], [`normalized_guidance_chain`], [`apg_delta`]) lifted backend-neutrally
//!    from Lens (`mlx-gen-lens/src/schedule.rs`) and Bernini (`mlx-gen-bernini/src/guidance.rs`)
//!    (sc-7438). CFG++ (`cfg_pp`) is realized at the sampler layer ([`crate::sampling::cfgpp`]), not
//!    here — its "combine" is the plain guided output plus the unconditional estimate forwarded to the
//!    solver, so it needs none of these reductions.

use crate::sampling::LatentOps;
use crate::Result;

/// `F.normalize` default eps — the denominator floor for the APG unit base direction.
const NORMALIZE_EPS: f32 = 1e-12;
/// `clamp_min` floor for the `apg_delta` reference norm² (the reference's `eps = 1e-8`).
const APG_DELTA_EPS: f32 = 1e-8;
/// `cfg_rescale` denominator floor (Lin et al. / Lens `max(‖comb‖, 1e-12)`).
const RESCALE_EPS: f32 = 1e-12;

// =================================================================================================
// GuidanceMethod — the curated policy vocabulary (sc-7436).
// =================================================================================================

/// How conditional/unconditional predictions are combined. The guidance analog of
/// [`crate::sampling::Solver`]: the engine matches on it; the contract validates the string form
/// against the per-model-per-backend [`crate::generator::Capabilities::supported_guidance_methods`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuidanceMethod {
    /// Plain classifier-free guidance: `uncond + scale·(cond − uncond)`.
    Cfg,
    /// Norm-rescaled CFG (Lin et al.): carry the conditional prediction's per-axis norm.
    CfgRescale,
    /// Adaptive Projected Guidance: momentum + norm-clamp + orthogonal/parallel projection.
    Apg,
    /// CFG++ (Chung et al.): plain guided combine, but the SAMPLER renoises from the unconditional
    /// branch. Realized in [`crate::sampling::cfgpp`]; gated to ddim/euler/dpmpp_2m.
    CfgPp,
}

impl GuidanceMethod {
    /// Parse the canonical lowercase name (UI / recipe / contract vocabulary). Unknown ⇒ `None`
    /// (callers fall back to the engine default + emit an event, N3).
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "cfg" => Self::Cfg,
            "cfg_rescale" => Self::CfgRescale,
            "apg" => Self::Apg,
            "cfg_pp" => Self::CfgPp,
            _ => return None,
        })
    }

    /// The canonical lowercase name (round-trips with [`Self::from_name`]).
    pub fn name(self) -> &'static str {
        match self {
            Self::Cfg => "cfg",
            Self::CfgRescale => "cfg_rescale",
            Self::Apg => "apg",
            Self::CfgPp => "cfg_pp",
        }
    }

    /// Every guidance method, in menu order.
    pub const ALL: [GuidanceMethod; 4] = [Self::Cfg, Self::CfgRescale, Self::Apg, Self::CfgPp];
}

// =================================================================================================
// GuidanceOps — the axis-parameterized backend op extension (sc-7437).
// =================================================================================================

/// The reduction/elementwise ops the guidance library needs beyond [`LatentOps`] (scale/add/sub/axpy).
/// Minimal by design — sized exactly against cfg_rescale + APG. The two reductions take an explicit
/// `(shape, axes)` (MLX/candle tensors carry their own shape and may ignore `shape`; the `Vec<f32>`
/// reference needs it) and return a result broadcast back to the FULL shape, so the library above
/// stays elementwise and never juggles keepdims/broadcasting.
pub trait GuidanceOps: LatentOps {
    /// Elementwise `a · b` (same shape).
    fn mul(&self, a: &Self::Latent, b: &Self::Latent) -> Result<Self::Latent>;
    /// Elementwise `a / b` (same shape; callers floor the denominator).
    fn div(&self, a: &Self::Latent, b: &Self::Latent) -> Result<Self::Latent>;
    /// Elementwise `max(x, s)`.
    fn clamp_min(&self, x: &Self::Latent, s: f32) -> Result<Self::Latent>;
    /// Elementwise `min(x, s)`.
    fn clamp_max(&self, x: &Self::Latent, s: f32) -> Result<Self::Latent>;
    /// Elementwise `sel > 0 ? a : b` (the cfg_rescale `torch.where(‖comb‖>0, …, 1)` guard).
    fn select_positive(
        &self,
        sel: &Self::Latent,
        a: &Self::Latent,
        b: &Self::Latent,
    ) -> Result<Self::Latent>;
    /// `sqrt(Σ x² over axes)`, broadcast back to full shape — the per-geometry L2 norm.
    fn norm_over(&self, x: &Self::Latent, shape: &[usize], axes: &[i32]) -> Result<Self::Latent>;
    /// `Σ (a · b) over axes`, broadcast back to full shape — the projection inner product.
    fn dot_over(
        &self,
        a: &Self::Latent,
        b: &Self::Latent,
        shape: &[usize],
        axes: &[i32],
    ) -> Result<Self::Latent>;
}

/// `c · ones_like(x)` for `c ≥ 0` — a full-shape constant tensor built without a dedicated op
/// (`max(0·x, c)`). Used for the scalar numerators (`norm_threshold`, `1.0`) the reductions divide.
fn const_like<G: GuidanceOps>(g: &G, x: &G::Latent, c: f32) -> Result<G::Latent> {
    debug_assert!(c >= 0.0, "const_like only builds non-negative constants");
    g.clamp_min(&g.scale(x, 0.0)?, c)
}

// =================================================================================================
// cfg — the plain combine.
// =================================================================================================

/// Plain classifier-free guidance: `uncond + scale·(cond − uncond)`. Axis-agnostic.
pub fn cfg<G: GuidanceOps>(
    g: &G,
    cond: &G::Latent,
    uncond: &G::Latent,
    scale: f32,
) -> Result<G::Latent> {
    let diff = g.sub(cond, uncond)?;
    g.axpy(1.0, uncond, scale, &diff)
}

// =================================================================================================
// cfg_rescale — Lin et al. norm-rescaled CFG (port of mlx-gen-lens/src/schedule.rs:58-74).
// =================================================================================================

/// Norm-rescaled CFG: `comb = cfg(...)`, then rescale `comb` to carry `cond`'s L2 norm over `axes`:
/// `comb · (‖cond‖ / max(‖comb‖, 1e-12))`, with the `where(‖comb‖>0, …, 1)` guard. Lens reduces over
/// the channel axis (`[-1]`) for per-token `[B, seq, C]`; the geometry is the caller's `axes`.
pub fn cfg_rescale<G: GuidanceOps>(
    g: &G,
    cond: &G::Latent,
    uncond: &G::Latent,
    scale: f32,
    shape: &[usize],
    axes: &[i32],
) -> Result<G::Latent> {
    let comb = cfg(g, cond, uncond, scale)?;
    let cond_norm = g.norm_over(cond, shape, axes)?;
    let comb_norm = g.norm_over(&comb, shape, axes)?;
    let denom = g.clamp_min(&comb_norm, RESCALE_EPS)?;
    let ratio = g.div(&cond_norm, &denom)?;
    let ones = const_like(g, &comb_norm, 1.0)?;
    // scale = where(‖comb‖ > 0, ‖cond‖/max(‖comb‖, eps), 1).
    let rescale = g.select_positive(&comb_norm, &ratio, &ones)?;
    g.mul(&comb, &rescale)
}

// =================================================================================================
// APG — the full Bernini surface (port of mlx-gen-bernini/src/guidance.rs).
// =================================================================================================

/// Persistent momentum accumulator for one APG stream (mirrors upstream `MomentumBuffer`). Allocate
/// one per guidance term BEFORE the denoise loop so the running average carries across steps.
pub struct MomentumBuffer<L> {
    momentum: f32,
    running: Option<L>,
}

impl<L: Clone> MomentumBuffer<L> {
    pub fn new(momentum: f32) -> Self {
        Self {
            momentum,
            running: None,
        }
    }

    /// `running = diff + momentum·running`; the first call (running = 0) returns `diff` unchanged.
    pub fn update<G: GuidanceOps<Latent = L>>(&mut self, g: &G, diff: &L) -> Result<L> {
        let ra = match &self.running {
            Some(r) => g.axpy(1.0, diff, self.momentum, r)?,
            None => diff.clone(),
        };
        self.running = Some(ra.clone());
        Ok(ra)
    }
}

/// The APG core: momentum → norm-clamp → orthogonal/parallel projection against `base`, returning
/// `orthogonal + eta·parallel`. `base` is the conditional prediction (the projection reference).
#[allow(clippy::too_many_arguments)]
fn normalize_diff<G: GuidanceOps>(
    g: &G,
    diff: &G::Latent,
    base: &G::Latent,
    buf: Option<&mut MomentumBuffer<G::Latent>>,
    eta: f32,
    norm_threshold: f32,
    shape: &[usize],
    axes: &[i32],
) -> Result<G::Latent> {
    let mut diff = match buf {
        Some(b) => b.update(g, diff)?,
        None => diff.clone(),
    };
    if norm_threshold > 0.0 {
        // scale = min(1, norm_threshold / ‖diff‖).
        let dn = g.norm_over(&diff, shape, axes)?;
        let num = const_like(g, &dn, norm_threshold)?;
        let scale = g.clamp_max(&g.div(&num, &dn)?, 1.0)?;
        diff = g.mul(&diff, &scale)?;
    }
    // Unit base direction (F.normalize): base / max(‖base‖, eps).
    let bn = g.clamp_min(&g.norm_over(base, shape, axes)?, NORMALIZE_EPS)?;
    let v1 = g.div(base, &bn)?;
    // parallel = (diff·v1)·v1; orthogonal = diff − parallel; out = orthogonal + eta·parallel.
    let coeff = g.dot_over(&diff, &v1, shape, axes)?;
    let parallel = g.mul(&coeff, &v1)?;
    let orthogonal = g.sub(&diff, &parallel)?;
    g.axpy(1.0, &orthogonal, eta, &parallel)
}

/// Single-condition APG: `uncond + scale · normalize_diff(cond − uncond, base = cond)`. With
/// `eta = 1`, `norm_threshold = 0`, and no momentum this is exactly plain [`cfg`].
#[allow(clippy::too_many_arguments)]
pub fn normalized_guidance<G: GuidanceOps>(
    g: &G,
    cond: &G::Latent,
    uncond: &G::Latent,
    scale: f32,
    buf: Option<&mut MomentumBuffer<G::Latent>>,
    eta: f32,
    norm_threshold: f32,
    shape: &[usize],
    axes: &[i32],
) -> Result<G::Latent> {
    let diff = g.sub(cond, uncond)?;
    let nd = normalize_diff(g, &diff, cond, buf, eta, norm_threshold, shape, axes)?;
    g.axpy(1.0, uncond, scale, &nd)
}

/// Chained APG over an ordered list of predictions (`normalized_guidance_chain`). Accumulates
/// `result = uncond + Σ_i scales[i] · normalize_diff(preds[i] − bases[i], base = preds[i])`, where
/// `bases = [uncond, preds[0], preds[1], …]`, each term with its own momentum buffer + norm threshold.
#[allow(clippy::too_many_arguments)]
pub fn normalized_guidance_chain<G: GuidanceOps>(
    g: &G,
    uncond: &G::Latent,
    preds: &[G::Latent],
    scales: &[f32],
    bufs: &mut [MomentumBuffer<G::Latent>],
    eta: f32,
    norm_thresholds: &[f32],
    shape: &[usize],
    axes: &[i32],
) -> Result<G::Latent> {
    let mut result = uncond.clone();
    for (i, cond) in preds.iter().enumerate() {
        let base_prev = if i == 0 { uncond } else { &preds[i - 1] };
        let diff = g.sub(cond, base_prev)?;
        let nd = normalize_diff(
            g,
            &diff,
            cond,
            Some(&mut bufs[i]),
            eta,
            norm_thresholds[i],
            shape,
            axes,
        )?;
        result = g.axpy(1.0, &result, scales[i], &nd)?;
    }
    Ok(result)
}

/// The v-space APG delta projection (Bernini `apg_delta`): project `delta` onto `reference` over the
/// caller's `axes` (the reference uses the whole flattened tensor per batch element) and recombine
/// with fixed parallel/orthogonal scales:
/// `proj = (delta·ref)/max(‖ref‖², eps)·ref`; `out = parallel_scale·proj + orthogonal_scale·(delta − proj)`.
pub fn apg_delta<G: GuidanceOps>(
    g: &G,
    delta: &G::Latent,
    reference: &G::Latent,
    parallel_scale: f32,
    orthogonal_scale: f32,
    shape: &[usize],
    axes: &[i32],
) -> Result<G::Latent> {
    // ‖ref‖² = dot(ref, ref); coeff = (delta·ref)/max(‖ref‖², eps).
    let ref_dot = g.dot_over(reference, reference, shape, axes)?;
    let ref_norm_sq = g.clamp_min(&ref_dot, APG_DELTA_EPS)?;
    let coeff = g.div(&g.dot_over(delta, reference, shape, axes)?, &ref_norm_sq)?;
    let parallel = g.mul(&coeff, reference)?;
    let orthogonal = g.sub(delta, &parallel)?;
    g.axpy(parallel_scale, &parallel, orthogonal_scale, &orthogonal)
}

// =================================================================================================
// CpuLatentOps reference impl of GuidanceOps (host-only proof; the arithmetic real backends match).
// =================================================================================================

use crate::sampling::CpuLatentOps;

/// Row-major strides for a shape.
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

/// Σ `data` over `axes`, broadcast back to the full shape: every element receives the sum of its
/// reduction group (the elements sharing its coordinates on the non-reduced axes). O(n·ndim) — a
/// host reference, not a hot path.
fn sum_over_broadcast(data: &[f32], shape: &[usize], axes: &[i32]) -> Vec<f32> {
    let ndim = shape.len();
    let total: usize = shape.iter().product();
    debug_assert_eq!(data.len(), total, "sum_over_broadcast: data/shape mismatch");
    let st = strides(shape);
    // Normalize axes to a reduced-axis mask.
    let mut reduced = vec![false; ndim];
    for &ax in axes {
        let a = if ax < 0 { ax + ndim as i32 } else { ax };
        reduced[a as usize] = true;
    }
    // Map each flat index to its group-representative flat index (reduced coords zeroed).
    let group_rep = |flat: usize| -> usize {
        let mut rep = 0usize;
        for d in 0..ndim {
            if !reduced[d] {
                let coord = (flat / st[d]) % shape[d];
                rep += coord * st[d];
            }
        }
        rep
    };
    let mut group_sum = vec![0.0f64; total];
    for (flat, &v) in data.iter().enumerate() {
        group_sum[group_rep(flat)] += v as f64;
    }
    (0..total)
        .map(|flat| group_sum[group_rep(flat)] as f32)
        .collect()
}

impl GuidanceOps for CpuLatentOps {
    fn mul(&self, a: &Vec<f32>, b: &Vec<f32>) -> Result<Vec<f32>> {
        debug_assert_eq!(a.len(), b.len(), "GuidanceOps::mul shape mismatch");
        Ok(a.iter().zip(b).map(|(&x, &y)| x * y).collect())
    }

    fn div(&self, a: &Vec<f32>, b: &Vec<f32>) -> Result<Vec<f32>> {
        debug_assert_eq!(a.len(), b.len(), "GuidanceOps::div shape mismatch");
        Ok(a.iter().zip(b).map(|(&x, &y)| x / y).collect())
    }

    fn clamp_min(&self, x: &Vec<f32>, s: f32) -> Result<Vec<f32>> {
        Ok(x.iter().map(|&v| v.max(s)).collect())
    }

    fn clamp_max(&self, x: &Vec<f32>, s: f32) -> Result<Vec<f32>> {
        Ok(x.iter().map(|&v| v.min(s)).collect())
    }

    fn select_positive(&self, sel: &Vec<f32>, a: &Vec<f32>, b: &Vec<f32>) -> Result<Vec<f32>> {
        debug_assert!(
            sel.len() == a.len() && a.len() == b.len(),
            "select shape mismatch"
        );
        Ok((0..sel.len())
            .map(|i| if sel[i] > 0.0 { a[i] } else { b[i] })
            .collect())
    }

    fn norm_over(&self, x: &Vec<f32>, shape: &[usize], axes: &[i32]) -> Result<Vec<f32>> {
        let sq: Vec<f32> = x.iter().map(|&v| v * v).collect();
        Ok(sum_over_broadcast(&sq, shape, axes)
            .into_iter()
            .map(|s| s.sqrt())
            .collect())
    }

    fn dot_over(
        &self,
        a: &Vec<f32>,
        b: &Vec<f32>,
        shape: &[usize],
        axes: &[i32],
    ) -> Result<Vec<f32>> {
        debug_assert_eq!(a.len(), b.len(), "GuidanceOps::dot_over shape mismatch");
        let prod: Vec<f32> = a.iter().zip(b).map(|(&x, &y)| x * y).collect();
        Ok(sum_over_broadcast(&prod, shape, axes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPS: CpuLatentOps = CpuLatentOps;

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    fn ramp(n: usize, seed: i32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as i32 * 7 + seed * 13) % 11) as f32 - 5.0)
            .collect()
    }

    #[test]
    fn method_names_round_trip() {
        for m in GuidanceMethod::ALL {
            assert_eq!(GuidanceMethod::from_name(m.name()), Some(m));
        }
        assert_eq!(GuidanceMethod::from_name("nope"), None);
    }

    #[test]
    fn sum_over_broadcast_geometries() {
        // [C=2, H=2] tensor, values 1..=4 row-major: [[1,2],[3,4]].
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let shape = [2usize, 2];
        // Reduce channel axis 0 (per-column): col0=1+3=4, col1=2+4=6 → broadcast [[4,6],[4,6]].
        assert_eq!(
            sum_over_broadcast(&data, &shape, &[0]),
            vec![4.0, 6.0, 4.0, 6.0]
        );
        // Reduce last axis -1 (per-row): row0=3, row1=7 → [[3,3],[7,7]].
        assert_eq!(
            sum_over_broadcast(&data, &shape, &[-1]),
            vec![3.0, 3.0, 7.0, 7.0]
        );
        // Whole-flattened: 10 everywhere.
        assert_eq!(sum_over_broadcast(&data, &shape, &[0, 1]), vec![10.0; 4]);
    }

    #[test]
    fn cfg_is_plain_combine() {
        let cond = ramp(8, 1);
        let uncond = ramp(8, 2);
        let got = cfg(&OPS, &cond, &uncond, 4.0).unwrap();
        let want: Vec<f32> = cond
            .iter()
            .zip(&uncond)
            .map(|(&c, &u)| u + 4.0 * (c - u))
            .collect();
        assert!(max_abs(&got, &want) < 1e-6);
    }

    #[test]
    fn cfg_rescale_matches_hand_reference_per_token() {
        // [B=1, seq=2, C=2]; rescale over channel axis -1 (Lens geometry).
        let shape = [1usize, 2, 2];
        let cond = vec![3.0_f32, 4.0, 1.0, 0.0]; // token0 norm 5, token1 norm 1
        let uncond = vec![0.0_f32, 0.0, 0.0, 0.0];
        let scale = 2.0_f32;
        let got = cfg_rescale(&OPS, &cond, &uncond, scale, &shape, &[-1]).unwrap();
        // comb = 2·cond; ‖comb‖ per token = 2·‖cond‖; rescale = ‖cond‖/‖comb‖ = 0.5 → comb·0.5 = cond.
        assert!(
            max_abs(&got, &cond) < 1e-5,
            "rescale should restore cond's norm"
        );
    }

    #[test]
    fn cfg_rescale_zero_comb_guards_to_one() {
        // cond non-zero but comb == 0 (scale 0) → where(‖comb‖>0,…,1) keeps comb (== 0) unscaled.
        let shape = [1usize, 1, 3];
        let cond = vec![1.0_f32, 2.0, 2.0];
        let uncond = vec![0.0_f32, 0.0, 0.0];
        let got = cfg_rescale(&OPS, &cond, &uncond, 0.0, &shape, &[-1]).unwrap();
        assert!(
            max_abs(&got, &uncond) < 1e-6,
            "zero comb must stay zero, not divide"
        );
    }

    /// Bernini's own invariant, backend-neutral: apg @ eta=1, nt=0, no momentum == plain CFG.
    #[test]
    fn apg_reduces_to_plain_cfg() {
        for axes in [&[0i32][..], &[-1i32][..], &[0i32, 1][..]] {
            let shape = [2usize, 3];
            let cond = ramp(6, 3);
            let uncond = ramp(6, 4);
            let got = normalized_guidance(&OPS, &cond, &uncond, 4.0, None, 1.0, 0.0, &shape, axes)
                .unwrap();
            let want = cfg(&OPS, &cond, &uncond, 4.0).unwrap();
            assert!(
                max_abs(&got, &want) < 1e-4,
                "apg(eta=1,nt=0) != cfg for axes {axes:?}"
            );
        }
    }

    #[test]
    fn apg_eta0_is_orthogonal_to_base() {
        // eta=0 ⇒ nd ⟂ cond: (nd · cond) over the reduction axis ≈ 0.
        let shape = [4usize, 2];
        let cond = ramp(8, 3);
        let uncond = ramp(8, 5);
        let diff: Vec<f32> = cond.iter().zip(&uncond).map(|(&c, &u)| c - u).collect();
        let nd = normalize_diff(&OPS, &diff, &cond, None, 0.0, 0.0, &shape, &[-1]).unwrap();
        let dot = OPS.dot_over(&nd, &cond, &shape, &[-1]).unwrap();
        assert!(
            dot.iter().all(|&v| v.abs() < 1e-3),
            "eta=0 residual {dot:?}"
        );
    }

    #[test]
    fn apg_norm_threshold_clamps() {
        // Large diff, small threshold, eta=1, base=diff ⇒ nd's per-group norm ≤ threshold.
        let shape = [4usize, 2];
        let cond = ramp(8, 5).iter().map(|&v| v * 100.0).collect::<Vec<_>>();
        let diff: Vec<f32> = cond.clone(); // uncond = 0 ⇒ diff = cond
        let nd = normalize_diff(&OPS, &diff, &cond, None, 1.0, 2.0, &shape, &[-1]).unwrap();
        let norms = OPS.norm_over(&nd, &shape, &[-1]).unwrap();
        assert!(
            norms.iter().all(|&n| n <= 2.0 + 1e-3),
            "clamped norm {norms:?}"
        );
    }

    #[test]
    fn momentum_accumulates() {
        let mut buf = MomentumBuffer::new(-0.5);
        let d1 = ramp(6, 6);
        let r1 = buf.update(&OPS, &d1).unwrap();
        assert!(max_abs(&r1, &d1) < 1e-6, "first update returns diff");
        let d2 = ramp(6, 7);
        let r2 = buf.update(&OPS, &d2).unwrap();
        let want: Vec<f32> = d2.iter().zip(&d1).map(|(&a, &b)| a - 0.5 * b).collect();
        assert!(max_abs(&r2, &want) < 1e-5, "running = d2 - 0.5·d1");
    }

    #[test]
    fn apg_delta_orthogonal_plus_parallel_scales() {
        // delta ∥ reference ⇒ pure parallel: out = parallel_scale·delta.
        let shape = [1usize, 4];
        let reference = vec![1.0_f32, 2.0, 3.0, 4.0];
        let delta: Vec<f32> = reference.iter().map(|&v| 3.0 * v).collect(); // delta = 3·ref
        let got = apg_delta(&OPS, &delta, &reference, 0.2, 1.0, &shape, &[-1]).unwrap();
        let want: Vec<f32> = delta.iter().map(|&v| 0.2 * v).collect();
        assert!(
            max_abs(&got, &want) < 1e-4,
            "parallel delta → parallel_scale·delta"
        );
    }

    #[test]
    fn chain_equals_sequential_single_terms() {
        // Two-term chain with eta=1, nt=0, no momentum effect on first call: result =
        // uncond + s0·(p0−uncond) + s1·(p1−p0) (each normalize_diff reduces to its diff).
        let shape = [2usize, 3];
        let uncond = ramp(6, 1);
        let p0 = ramp(6, 2);
        let p1 = ramp(6, 3);
        let mut bufs = vec![MomentumBuffer::new(0.0), MomentumBuffer::new(0.0)];
        let got = normalized_guidance_chain(
            &OPS,
            &uncond,
            &[p0.clone(), p1.clone()],
            &[2.0, 1.5],
            &mut bufs,
            1.0,
            &[0.0, 0.0],
            &shape,
            &[-1],
        )
        .unwrap();
        let want: Vec<f32> = (0..6)
            .map(|i| uncond[i] + 2.0 * (p0[i] - uncond[i]) + 1.5 * (p1[i] - p0[i]))
            .collect();
        assert!(max_abs(&got, &want) < 1e-4, "chain != sequential terms");
    }
}
