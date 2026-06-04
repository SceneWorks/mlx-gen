//! Adapter framework — LoRA + LoKr applied as forward-time residuals over a shared
//! base. Quantized-safe: the base is never fused/mutated. Ported from the sc-2338
//! spike; mirrors the Python mflux fork's `LoKrLinear` / `FusedLoRALinear` (sc-2216).
//!
//! The base is a real `nn::Linear` *or* `nn::QuantizedLinear` (sc-2342), so quantization
//! and adapters compose: `base(x) + Σ adapter.residual(x)`. Forward is taken by `&self`
//! (we call the underlying ops directly rather than the `&mut self` `Module` trait), so a
//! whole model tree can be evaluated through shared references.
//!
//! Adapters are installed by dotted path via [`AdaptableHost`] / [`install_adapter`] — the
//! Rust stand-in for Python's dynamic `getattr`-swap, since mlx-rs flattens module params to
//! `Array` leaves and cannot replace a submodule in place.

use mlx_rs::{
    module::Param,
    nn::{Linear, QuantizedLinear},
    ops::{add, addmm, kron, matmul, multiply, quantized_matmul},
    Array, Dtype,
};

use crate::array::scalar;
use crate::Result;

pub mod loader;

/// Reconstruct a LoKr weight delta `ΔW = (alpha/rank) · kron(w1, w2)`, reshaped to the
/// base weight's logical `[out, in]` and cast to `out_dtype`. Each Kronecker factor is either
/// full (`w1` / `w2`) or a low-rank product (`w1_a @ w1_b` / `w2_a @ w2_b`). Mirrors
/// PEFT/LyCORIS `LoKrLayer.get_delta_weight` (pending the sc-2324 cross-impl parity check).
///
/// `out_dtype` is `Bfloat16` for the fork-parity residual path (Z-Image/Qwen — PARITY-BF16,
/// sc-2609) and `Float32` for the SDXL merge path (f32-everywhere, no fork to match — sc-2640).
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_lokr_delta(
    alpha: f32,
    rank: f32,
    base_shape: &[i32],
    w1: Option<&Array>,
    w1_a: Option<&Array>,
    w1_b: Option<&Array>,
    w2: Option<&Array>,
    w2_a: Option<&Array>,
    w2_b: Option<&Array>,
    out_dtype: Dtype,
) -> Result<Array> {
    let factor1 = match (w1, w1_a, w1_b) {
        (Some(w), _, _) => w.clone(),
        (_, Some(a), Some(b)) => matmul(a, b)?,
        _ => return Err("LoKr: w1 missing (need full w1 or w1_a@w1_b)".into()),
    };
    let factor2 = match (w2, w2_a, w2_b) {
        (Some(w), _, _) => w.clone(),
        (_, Some(a), Some(b)) => matmul(a, b)?,
        _ => return Err("LoKr: w2 missing (need full w2 or w2_a@w2_b)".into()),
    };
    let delta = multiply(&kron(&factor1, &factor2)?, scalar(alpha / rank))?;
    Ok(delta.reshape(base_shape)?.as_dtype(out_dtype)?)
}

/// One adapter's contribution WITHOUT the base, so a host can sum stacked adapters over
/// a single base application.
pub enum Adapter {
    /// LoRA: `residual = scale · x·A·B`.
    Lora { a: Array, b: Array, scale: f32 },
    /// LoKr: `residual = scale · x·ΔWᵀ`; `delta` stored bf16 (see [`reconstruct_lokr_delta`]).
    Lokr { delta: Array, scale: f32 },
}

impl Adapter {
    pub fn residual(&self, x: &Array) -> Result<Array> {
        // Adapter math runs in f32. LoRA's low-rank second matmul is `[seq,r]·[r,out]` with
        // `K = rank ≤ 512` — exactly the dense 16-bit×16-bit Metal GEMM the NAX build mis-runs
        // (M≥2 & K≤512; see `mlx-gen-qwen-image/tests/bf16_matmul_sweep.rs`). On the bf16 txt2img
        // path that GEMM would get bf16 activations and return garbage. f32 sidesteps the bug, is
        // strictly more accurate, and still matches the fork's (bug-free wheel) bf16 residual
        // within tolerance once cast back. The result is returned in the activation dtype so
        // `base(x) + residual` stays in the base dtype (PARITY-BF16 on the bf16 path; f32 base → f32).
        let xf = x.as_dtype(Dtype::Float32)?;
        let r = match self {
            Adapter::Lora { a, b, scale } => {
                let a = a.as_dtype(Dtype::Float32)?;
                let b = b.as_dtype(Dtype::Float32)?;
                multiply(&matmul(&matmul(&xf, &a)?, &b)?, scalar(*scale))?
            }
            Adapter::Lokr { delta, scale } => {
                let d = delta.as_dtype(Dtype::Float32)?;
                multiply(&matmul(&xf, d.t())?, scalar(*scale))?
            }
        };
        Ok(r.as_dtype(x.dtype())?)
    }
}

/// A linear base — dense or quantized — evaluated through a shared reference. Mirrors the
/// `forward` of mlx-rs's `nn::Linear` / `nn::QuantizedLinear` but without requiring `&mut`.
pub enum LinearBase {
    Dense(Linear),
    Quantized(QuantizedLinear),
}

impl LinearBase {
    fn forward(&self, x: &Array) -> Result<Array> {
        Ok(match self {
            LinearBase::Dense(l) => {
                // Mirror MLX `nn.Linear` exactly: the biased case is a FUSED `addmm(bias, x, Wᵀ)`
                // — accumulate `x·Wᵀ`, add bias, round to the output dtype ONCE. A separate
                // `matmul` then `add` rounds the matmul, *then* rounds the bias add again — a
                // ~1.4e-3 double-rounding error per biased Linear in bf16 that compounds over a
                // deep net (sc-2779; localized in the Wan DiT, q_proj 1.4e-3 → ~4e-7 with addmm).
                // f32-INVISIBLE and therefore safe for every crate today: with f32 activations
                // (the current Z-Image/Qwen/FLUX path, even with bf16 weights) `addmm == matmul+add`
                // bit-for-bit, because nothing rounds to bf16 mid-op (verified, sc-2779). It bites
                // only once a path runs bf16 activations (the sc-2718–2721 reverts). The unbiased
                // case stays a plain `matmul`, as mlx-rs's own `Linear::forward` does.
                match l.bias.value.as_ref() {
                    Some(b) => addmm(b, x, l.weight.value.t(), 1.0, 1.0)?,
                    None => matmul(x, l.weight.value.t())?,
                }
            }
            LinearBase::Quantized(q) => {
                // Activations are fed to `quantized_matmul` AS-IS — no dtype upcast. `quantized_matmul`
                // accumulates in fp32 (mlx#963) and is correct at every activation shape/dtype, so it
                // was never the buggy op: the NAX 16-bit-GEMM bug lived in the *dense* 16-bit×16-bit
                // Metal GEMM, and that is now fixed at the toolchain level (sc-2772 — metal target ≥26.2).
                // The former bf16→f32 upcast here (sc-2719) guarded a proven non-bug and is removed:
                // feeding bf16 activations straight in matches the fork's own quantized compute dtype
                // (bf16 latents → `quantized_matmul` → bf16), so it is strictly *more* faithful, not less.
                // Weights stay Q4/Q8 throughout. (`q8_smoke.rs` exercises the bf16-activation path.)
                let mut y = quantized_matmul(
                    x,
                    &q.inner.weight.value,
                    &q.scales.value,
                    &q.biases.value,
                    true,
                    q.group_size,
                    q.bits,
                )?;
                if let Some(b) = q.inner.bias.value.as_ref() {
                    y = add(&y, b)?;
                }
                y
            }
        })
    }
}

/// A linear base plus a stack of adapters, applied as `base(x) + Σ adapter.residual(x)`.
/// Quantized-safe: the base weight is never mutated.
pub struct AdaptableLinear {
    base: LinearBase,
    adapters: Vec<Adapter>,
}

impl AdaptableLinear {
    /// Build from a raw `[out, in]` weight (and optional bias) — the common path when
    /// loading dense (bf16/fp16/fp32) checkpoints via the `weights` module.
    pub fn dense(weight: Array, bias: Option<Array>) -> Self {
        Self::from_linear(Linear {
            weight: Param::new(weight),
            bias: Param::new(bias),
        })
    }

    /// Wrap an existing dense `nn::Linear`.
    pub fn from_linear(linear: Linear) -> Self {
        Self {
            base: LinearBase::Dense(linear),
            adapters: Vec::new(),
        }
    }

    /// Wrap an existing `nn::QuantizedLinear` (sc-2342 quantized weights).
    pub fn from_quantized(q: QuantizedLinear) -> Self {
        Self {
            base: LinearBase::Quantized(q),
            adapters: Vec::new(),
        }
    }

    /// Stack a new adapter (LoRA or LoKr) on top of any already installed.
    pub fn push(&mut self, adapter: Adapter) {
        self.adapters.push(adapter);
    }

    pub fn adapters(&self) -> &[Adapter] {
        &self.adapters
    }

    /// Merge a precomputed `[out, in]` delta into the dense base weight (`W += δ`) — the in-place
    /// LoRA/LoKr *merge*, distinct from the forward-time [`Adapter::residual`] stack. The merge
    /// reproduces a reference's merged-weight forward (`(W+δ)·x`) bit-for-bit, where a residual
    /// (`W·x + δ·x`) differs by ~1 ULP; on a chaos-sensitive sampler (SDXL's ancestral) that 1-ULP
    /// cascades to a visible whole-image divergence, so the SDXL provider merges (matching the
    /// vendored `lora.py` `module.weight += delta`) rather than stacking residuals. `delta` is cast
    /// to the base weight's dtype before the add. Errors on a quantized base — a LoRA must be merged
    /// into the dense (e.g. f32) weight BEFORE quantization (the fork merges pre-quantize too).
    pub fn merge_dense_delta(&mut self, delta: &Array) -> Result<()> {
        match &mut self.base {
            LinearBase::Dense(l) => {
                let merged = add(&l.weight.value, &delta.as_dtype(l.weight.value.dtype())?)?;
                l.weight = Param::new(merged);
                Ok(())
            }
            LinearBase::Quantized(_) => Err(
                "merge_dense_delta: base is quantized; a LoRA must be merged before quantization"
                    .into(),
            ),
        }
    }

    /// `true` once the base has been quantized (Q4/Q8).
    pub fn is_quantized(&self) -> bool {
        matches!(self.base, LinearBase::Quantized(_))
    }

    /// Diagnostic accessor: the quantized base's `(packed_weight, scales, biases, bias, group_size,
    /// bits)`, or `None` if the base is still dense. Used by the sc-2604 Q8 root-cause diagnostic to
    /// byte-compare the *loaded* model's quantization against the fork's `mx.quantize` (the
    /// `qmm_smallk` probe only exercised the free `quantize` op, not `try_from_linear`).
    /// Diagnostic accessor: the dense base's `(weight, bias)`, or `None` if already quantized.
    /// Used by the sc-2604 diagnostic to inspect the loaded weight dtype before quantization.
    pub fn dense_weight(&self) -> Option<(&Array, Option<&Array>)> {
        match &self.base {
            LinearBase::Dense(l) => Some((&l.weight.value, l.bias.value.as_ref())),
            LinearBase::Quantized(_) => None,
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn quantized_params(&self) -> Option<(&Array, &Array, &Array, Option<&Array>, i32, i32)> {
        match &self.base {
            LinearBase::Quantized(q) => Some((
                &q.inner.weight.value,
                &q.scales.value,
                &q.biases.value,
                q.inner.bias.value.as_ref(),
                q.group_size,
                q.bits,
            )),
            LinearBase::Dense(_) => None,
        }
    }

    /// The base weight's logical `[out, in]` shape — what a LoKr delta must reshape to.
    /// For a quantized base the packed weight is opaque, so recover it from the scales grid
    /// (`[out, in/group_size]`) times the group size.
    pub fn base_shape(&self) -> Vec<i32> {
        match &self.base {
            LinearBase::Dense(l) => l.weight.value.shape().to_vec(),
            LinearBase::Quantized(q) => {
                let s = q.scales.value.shape();
                vec![s[0], s[1] * q.group_size]
            }
        }
    }

    /// Quantize the dense base in place to Q4/Q8 (`group_size` defaults to 64), the mlx-rs
    /// equivalent of `nn.quantize` over this Linear. No-op if already quantized. Adapters are
    /// forward-time residuals over the (now quantized) base, so they are unaffected — this is
    /// why the base is never fused: fusing would force re-quantization on every adapter swap.
    pub fn quantize(&mut self, bits: i32, group_size: Option<i32>) -> Result<()> {
        if let LinearBase::Dense(l) = &self.base {
            // PARITY-BF16 (sc-2609): downcast for fork parity. f32 quantization (f32 group scales)
            // is *more* accurate; we cast to bf16 only to byte-match the fork's golden. Flip to f32
            // for quality once parity is no longer the goal — f32 is safe (the qmm path never hits
            // the bf16-GEMM bug). Rationale below.
            //
            // The fork (mflux) loads every weight at bf16 — its compute dtype — and quantizes THAT.
            // Some checkpoints (e.g. Z-Image-Turbo's transformer) ship f32 on disk; quantizing the
            // as-loaded f32 weight yields group `scales` that differ from the fork's bf16 scales by
            // ~0.13% (the integer `wq` codes and `biases` survive the perturbation, the scales do
            // not), which compounds into the base-model Q8/Q4 e2e residual (sc-2604). Cast weight +
            // bias to bf16 first so the packing is byte-identical to the fork. No-op when already
            // bf16 (e.g. Qwen, whose checkpoint is bf16-native — which is why its Q8 already matched).
            let weight = l.weight.value.as_dtype(Dtype::Bfloat16)?;
            let bias = l
                .bias
                .value
                .as_ref()
                .map(|b| b.as_dtype(Dtype::Bfloat16))
                .transpose()?;
            let linear = Linear {
                weight: Param::new(weight),
                bias: Param::new(bias),
            };
            let q = QuantizedLinear::try_from_linear(
                linear,
                group_size.unwrap_or(crate::quant::DEFAULT_GROUP_SIZE),
                bits,
            )?;
            self.base = LinearBase::Quantized(q);
        }
        Ok(())
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut out = self.base.forward(x)?;
        for adapter in &self.adapters {
            out = add(&out, &adapter.residual(x)?)?;
        }
        Ok(out)
    }
}

/// A module tree that can resolve a dotted parameter path (split into segments) to the
/// [`AdaptableLinear`] living there, so an adapter can be installed onto it. This is the
/// hand-written form of the macro the full adapter framework (sc-2343) will generate.
pub trait AdaptableHost {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear>;

    /// Enumerate every adapter target reachable through the kohya `lora_unet_` convention, as
    /// dotted paths in the trained-file (diffusers) naming that [`adaptable_mut`](Self::adaptable_mut)
    /// accepts. Used to build the `flattened → dotted` lookup that disambiguates kohya keys (whose
    /// `.`→`_` flattening cannot be re-split blindly — module names like `to_out.0` / `feed_forward.w1`
    /// already contain underscores). Mirrors the fork's explicit per-target `lora_unet_…` patterns
    /// (sc-2618): block-indexed layer targets only — the families' fork mappings carry no `lora_unet_`
    /// pattern for global targets, which stay reachable via the diffusers/peft dotted form.
    ///
    /// Every returned path MUST resolve via [`adaptable_mut`](Self::adaptable_mut) and the set MUST be
    /// collision-free once flattened (both guarded by tests). The default is empty — a host that does
    /// not override it has no kohya support and a kohya file applied to it surfaces every key as
    /// unmatched (loud), never silently dropped.
    fn adaptable_paths(&self) -> Vec<String> {
        Vec::new()
    }
}

/// Prefix each of `host`'s [`AdaptableHost::adaptable_paths`] with `‹prefix›.` — the enumeration
/// analog of a parent's `["‹prefix›", rest @ ..] => sub.adaptable_mut(rest)` delegation, so a
/// composite host can build its full path list from its children's relative ones (sc-2618 kohya).
pub fn prefixed_paths(prefix: &str, host: &impl AdaptableHost) -> Vec<String> {
    host.adaptable_paths()
        .iter()
        .map(|p| format!("{prefix}.{p}"))
        .collect()
}

/// Install an adapter onto the [`AdaptableLinear`] addressed by `dotted` (e.g.
/// `"attention.to_q"`). Errors if the path resolves to no adaptable linear.
pub fn install_adapter(
    host: &mut impl AdaptableHost,
    dotted: &str,
    adapter: Adapter,
) -> Result<()> {
    let parts: Vec<&str> = dotted.split('.').collect();
    host.adaptable_mut(&parts)
        .ok_or_else(|| format!("no adaptable linear at path: {dotted}"))?
        .push(adapter);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{all_close, array_eq};

    fn lokr_2x2() -> Array {
        reconstruct_lokr_delta(
            8.0,
            4.0,
            &[2, 2],
            Some(&Array::from_slice(&[0.5f32, 0.6], &[2, 1])),
            None,
            None,
            Some(&Array::from_slice(&[0.7f32, 0.8], &[1, 2])),
            None,
            None,
            Dtype::Bfloat16,
        )
        .unwrap()
    }

    #[test]
    fn lokr_delta_stored_bf16() {
        assert_eq!(lokr_2x2().dtype(), Dtype::Bfloat16);
    }

    #[test]
    fn scale_zero_lokr_is_bit_exact_noop() {
        let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 2]);
        let mut lin = AdaptableLinear::dense(w, None);
        let base = lin.forward(&x).unwrap();
        lin.push(Adapter::Lokr {
            delta: lokr_2x2(),
            scale: 0.0,
        });
        let out = lin.forward(&x).unwrap();
        assert!(array_eq(&out, &base, false).unwrap().item::<bool>());
    }

    #[test]
    fn residual_in_bf16_runs_f32_and_returns_activation_dtype() {
        // The LoRA second matmul `[seq,r]·[r,out]` (K=rank=4≤512, M=seq=4≥2) is the dense 16-bit
        // GEMM the NAX build mis-runs; `residual` must compute it in f32 and return the activation
        // dtype. So a bf16-input residual must (a) be bf16 and (b) match the f32 reference within
        // bf16 rounding — NOT diverge (which is what the buggy bf16 GEMM would produce).
        let a32 = Array::from_slice(
            &(0..8).map(|i| i as f32 * 0.1 - 0.4).collect::<Vec<_>>(),
            &[2, 4],
        );
        let b32 = Array::from_slice(
            &(0..8).map(|i| i as f32 * 0.05).collect::<Vec<_>>(),
            &[4, 2],
        );
        let x32 = Array::from_slice(&[1.0f32, -2.0, 0.5, 0.25, -1.0, 2.0], &[3, 2]);
        let lora = Adapter::Lora {
            a: a32.as_dtype(Dtype::Bfloat16).unwrap(),
            b: b32.as_dtype(Dtype::Bfloat16).unwrap(),
            scale: 0.5,
        };
        let got = lora
            .residual(&x32.as_dtype(Dtype::Bfloat16).unwrap())
            .unwrap();
        assert_eq!(
            got.dtype(),
            Dtype::Bfloat16,
            "residual returns the activation dtype"
        );

        // f32 reference, rounded to bf16 the way `residual` casts its result back.
        let want = multiply(
            matmul(matmul(&x32, &a32).unwrap(), &b32).unwrap(),
            scalar(0.5),
        )
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        assert!(
            all_close(&got, &want, 5e-2, 5e-2, false)
                .unwrap()
                .item::<bool>(),
            "bf16 residual diverged from the f32 reference (bf16 GEMM bug?)"
        );
    }

    #[test]
    fn biased_dense_forward_is_fused_addmm() {
        // sc-2779: the biased dense base must be a FUSED `addmm(bias, x, Wᵀ)`, not `matmul`+`add`.
        // In bf16 the two differ (double-rounding), so feed bf16 activations and assert the forward
        // is bit-exact to `addmm` and bit-distinct from `matmul`+`add` — i.e. the fusion is real.
        let n = 4 * 64;
        let w = Array::from_slice(
            &(0..64 * 64)
                .map(|i| (i as f32 * 0.013).sin() * 0.05)
                .collect::<Vec<_>>(),
            &[64, 64],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let bias = Array::from_slice(
            &(0..64)
                .map(|i| (i as f32 * 0.7).cos() * 0.1)
                .collect::<Vec<_>>(),
            &[64],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let x = Array::from_slice(
            &(0..n)
                .map(|i| (i as f32 * 0.031).sin() * 0.5)
                .collect::<Vec<_>>(),
            &[4, 64],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();

        let lin = AdaptableLinear::dense(w.clone(), Some(bias.clone()));
        let got = lin.forward(&x).unwrap();

        let want_addmm = addmm(&bias, &x, w.t(), 1.0, 1.0).unwrap();
        assert!(
            array_eq(&got, &want_addmm, false).unwrap().item::<bool>(),
            "biased dense forward must be bit-exact to addmm(bias, x, Wᵀ)"
        );

        // And it must NOT be the double-rounding matmul+add (which is what the bug looked like).
        let matmul_add = add(matmul(&x, w.t()).unwrap(), &bias).unwrap();
        assert!(
            !array_eq(&got, &matmul_add, false).unwrap().item::<bool>(),
            "bf16 addmm should differ from matmul+add (double-rounding) — fusion not applied?"
        );
    }

    #[test]
    fn biased_dense_forward_f32_acts_match_matmul_add_bit_exact() {
        // sc-2779 golden-safety invariant: with f32 activations (the current Z-Image/Qwen/FLUX path,
        // even over bf16 weights), `addmm == matmul+add` bit-for-bit — nothing rounds to bf16
        // mid-op. This is why lifting the core to addmm cannot regress any current f32-act golden.
        let w = Array::from_slice(
            &(0..64 * 64)
                .map(|i| (i as f32 * 0.013).sin() * 0.05)
                .collect::<Vec<_>>(),
            &[64, 64],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap(); // bf16 weights
        let bias = Array::from_slice(
            &(0..64)
                .map(|i| (i as f32 * 0.7).cos() * 0.1)
                .collect::<Vec<_>>(),
            &[64],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let x = Array::from_slice(
            &(0..4 * 64)
                .map(|i| (i as f32 * 0.031).sin() * 0.5)
                .collect::<Vec<_>>(),
            &[4, 64],
        ); // f32 activations

        let got = AdaptableLinear::dense(w.clone(), Some(bias.clone()))
            .forward(&x)
            .unwrap();
        let matmul_add = add(matmul(&x, w.t()).unwrap(), &bias).unwrap();
        assert!(
            array_eq(&got, &matmul_add, false).unwrap().item::<bool>(),
            "f32-activation addmm must be bit-exact to matmul+add (no golden regression)"
        );
    }

    #[test]
    fn merge_dense_delta_adds_to_weight_and_zero_is_noop() {
        let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 2]);

        // A zero delta is a bit-exact no-op (`W + 0 == W`) — the scale-0 LoRA invariant.
        let mut lin = AdaptableLinear::dense(w.clone(), None);
        let base = lin.forward(&x).unwrap();
        lin.merge_dense_delta(&Array::from_slice(&[0.0f32; 4], &[2, 2]))
            .unwrap();
        assert!(array_eq(lin.forward(&x).unwrap(), &base, false)
            .unwrap()
            .item::<bool>());

        // A nonzero delta is exactly `(W + δ)·x`.
        let delta = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75], &[2, 2]);
        let mut lin2 = AdaptableLinear::dense(w.clone(), None);
        lin2.merge_dense_delta(&delta).unwrap();
        let want = AdaptableLinear::dense(add(&w, &delta).unwrap(), None)
            .forward(&x)
            .unwrap();
        assert!(array_eq(lin2.forward(&x).unwrap(), &want, false)
            .unwrap()
            .item::<bool>());

        // Merging into a quantized base is rejected (must merge before quantization).
        let mut lin3 = AdaptableLinear::dense(
            Array::from_slice(
                &(0..4096).map(|i| i as f32 * 1e-3).collect::<Vec<_>>(),
                &[64, 64],
            ),
            None,
        );
        lin3.quantize(8, None).unwrap();
        assert!(lin3
            .merge_dense_delta(&Array::from_slice(&[0.0f32; 4096], &[64, 64]))
            .is_err());
    }

    #[test]
    fn stacks_mixed_lora_and_lokr_summing_residuals() {
        let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 2]);
        let mut lin = AdaptableLinear::dense(w, None);
        let base = lin.forward(&x).unwrap();
        let lora = Adapter::Lora {
            a: Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]),
            b: Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75], &[2, 2]),
            scale: 0.5,
        };
        let lokr = Adapter::Lokr {
            delta: lokr_2x2(),
            scale: 0.7,
        };
        let lora_r = lora.residual(&x).unwrap();
        let lokr_r = lokr.residual(&x).unwrap();
        lin.push(lora);
        lin.push(lokr);
        assert_eq!(lin.adapters().len(), 2);
        let expected = add(add(&base, &lora_r).unwrap(), &lokr_r).unwrap();
        assert!(
            all_close(lin.forward(&x).unwrap(), &expected, 1e-4, 1e-2, false)
                .unwrap()
                .item::<bool>()
        );
    }
}
