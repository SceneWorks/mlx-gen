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

/// Fuse a **conv-layer** LoRA pair into a single conv-weight delta, returned in the trained-file
/// NCHW `[out, in, kH, kW]` layout (sc-2919). Conv LoRAs decompose a conv into a spatial `down`
/// (`lora_down`, `[rank, in, kH, kW]`) followed by a 1×1 `up` (`lora_up`, `[out, rank, 1, 1]`); the
/// fused weight is the composition of those two convs:
///   `δ[o, i, y, x] = Σ_r up[o, r] · down[r, i, y, x]`,
/// which is exactly `up[out, rank] @ down[rank, in·kH·kW]` reshaped back to `[out, in, kH, kW]` —
/// bit-identical to PEFT/diffusers' `Conv2d` LoRA fusion (`F.conv2d(down.permute(1,0,2,3), up)`),
/// and uniform across 1×1 and k×k kernels. Then scaled by `(alpha/rank)·scale`.
///
/// Precision mirrors the SDXL Linear `lora_delta`: the `up @ down` matmul runs in **f32** (correct,
/// and avoids the former NAX 16-bit-GEMM bug — now fixed at the toolchain level, sc-2772) and is
/// rounded back through the factors' source dtype (f16 for community/accel LoRAs), so the result is
/// the same value an f16 reference fusion would produce, returned as f32 for the caller to cast to
/// the conv weight's dtype on merge. The merge itself is the chaos-safe `W += δ` (the SDXL ancestral
/// sampler needs a merged weight, not a forward-time residual — cf. [`AdaptableLinear::merge_dense_delta`]).
pub fn conv_lora_delta(
    down: &Array,
    up: &Array,
    alpha: f32,
    rank: f32,
    scale: f32,
) -> Result<Array> {
    let src = up.dtype(); // f16 for kohya/community LoRAs; f32 makes the round-trip a no-op.
    let ds = down.shape(); // [rank, in, kH, kW]
    let us = up.shape(); // [out, rank, 1, 1]
    let (r, cin, kh, kw) = (ds[0], ds[1], ds[2], ds[3]);
    let out = us[0];
    let down2 = down.reshape(&[r, cin * kh * kw])?; // [rank, in·kH·kW]
    let up2 = up.reshape(&[out, r])?; // [out, rank]
    let ba = matmul(
        &up2.as_dtype(Dtype::Float32)?,
        &down2.as_dtype(Dtype::Float32)?,
    )?;
    let ba = ba.as_dtype(src)?.as_dtype(Dtype::Float32)?;
    // effective_scale in f64 then f32, matching a reference's Python-float arithmetic.
    let eff = ((alpha as f64 / rank as f64) * scale as f64) as f32;
    Ok(multiply(&ba, scalar(eff))?.reshape(&[out, cin, kh, kw])?)
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
    /// One adapter's forward-time contribution `scale · …`, replicating the fork's `LoRALinear`
    /// / `LoKrLinear` `.residual` **byte-for-byte** (sc-2718). No dtype is forced: the earlier f32
    /// upcast (sc-2602/2719) was a workaround for the NAX 16-bit dense GEMM returning garbage on the
    /// low-rank `[seq,r]·[r,out]` matmul (`K = rank ≤ 512`, `M ≥ 2`); that GEMM is now correct at the
    /// toolchain level (sc-2772 — Metal target ≥ 26.2), so the math runs in the natural promoted
    /// dtype exactly as the fork does — restoring parity (the f32 forcing was the DEVIATION):
    ///   * LoRA — `scale · (x·A)·B` with `A`/`B` kept at their loaded (file) dtype. The fork never
    ///     casts the factors, so a bf16 `x` against f32 factors (the goldens ship f32) promotes to
    ///     f32; a bf16-factor file runs bf16 (the formerly-buggy shape, now safe).
    ///   * LoKr — `scale · x·ΔWᵀ` with `ΔW` (stored bf16) cast to the **activation dtype** — bf16 on
    ///     the bf16 path — mirroring the fork's `delta.astype(x.dtype)`.
    ///
    /// The result is NOT cast back: `base(x) + residual` promotes just as the fork's `out + residual`
    /// does. An f32-activation target is unchanged (FLUX.2; Qwen's f32 image stream; SDXL merges
    /// instead) — the residual was f32 before and stays f32. A bf16-activation target now runs the
    /// residual in bf16 like the fork (Z-Image's latents; Qwen's bf16 text stream); validated against
    /// the fork goldens (Z-Image / Qwen LoRA+LoKr) — px>8 byte-identical to the old forced-f32 path,
    /// i.e. the dtype change is sub-threshold while restoring fork-faithfulness (sc-2718). `scale` is
    /// applied through a dtype-matched scalar so the multiply preserves the residual's dtype, matching
    /// the fork's weak Python-float `scale * …` (a strong f32 scalar would wrongly promote a bf16
    /// residual to f32; verified against MLX).
    pub fn residual(&self, x: &Array) -> Result<Array> {
        let (r, scale) = match self {
            Adapter::Lora { a, b, scale } => (matmul(&matmul(x, a)?, b)?, *scale),
            Adapter::Lokr { delta, scale } => {
                let d = delta.as_dtype(x.dtype())?;
                (matmul(x, d.t())?, *scale)
            }
        };
        // Dtype-matched scalar → preserves the residual's dtype (the fork's weak-float `scale * …`).
        Ok(multiply(&r, &scalar(scale).as_dtype(r.dtype())?)?)
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

    /// Build a quantized base from **already-packed** parts read off disk — a *pre-quantized*
    /// checkpoint (group-wise affine `weight` u32 codes + `scales` + `biases`, optional dense
    /// `bias`). The consume-side counterpart to [`quantize`](Self::quantize): no re-quantization
    /// happens, the on-disk scales are used as-is. Mirrors the fork's `loading.py` — `nn.quantize`
    /// stubs then `load_weights` of the packed tensors — but as a direct construction. `group_size`
    /// and `bits` come from the checkpoint's manifest (e.g. Wan's `config.json` `quantization` block).
    pub fn from_quantized_parts(
        weight: Array,
        scales: Array,
        biases: Array,
        bias: Option<Array>,
        group_size: i32,
        bits: i32,
    ) -> Self {
        Self::from_quantized(QuantizedLinear {
            group_size,
            bits,
            scales: Param::new(scales),
            biases: Param::new(biases),
            inner: Linear {
                weight: Param::new(weight),
                bias: Param::new(bias),
            },
        })
    }

    /// Stack a new adapter (LoRA or LoKr) on top of any already installed.
    pub fn push(&mut self, adapter: Adapter) {
        self.adapters.push(adapter);
    }

    /// Replace the entire adapter stack. The forward-time counterpart to [`push`](Self::push) used
    /// by **training** (sc-3042/3039): each optimizer step produces new trainable LoRA factor arrays,
    /// so the trainer re-injects a single fresh `Adapter::Lora` per target every step rather than
    /// accumulating residuals. Setting the SAME `(transpose, alpha/rank fold, scale)` an inference
    /// reload applies (`adapters::loader::install_lora_groups`) makes the trained adapter round-trip
    /// bit-for-bit. An empty `Vec` clears the stack (back to the bare frozen base).
    pub fn set_adapters(&mut self, adapters: Vec<Adapter>) {
        self.adapters = adapters;
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

/// A dense Conv2d weight (mlx NHWC `[out, kH, kW, in]`) plus its optional bias, that can have a
/// conv-layer LoRA delta merged into it (sc-2919). Convs in this codebase are **merge-only**: they
/// are never quantized and never carry a forward-time residual, so unlike [`AdaptableLinear`] there
/// is no adapter stack or quantized variant — just the mergeable weight and the accessors a forward
/// pass needs. The merge takes a delta in the trained-file NCHW layout and folds it in chaos-safely
/// (`W += δ`), the conv analog of [`AdaptableLinear::merge_dense_delta`].
pub struct AdaptableConv2d {
    /// NHWC `[out, kH, kW, in]` — the layout `mlx_gen::nn::conv2d` expects.
    weight: Array,
    bias: Option<Array>,
}

impl AdaptableConv2d {
    /// Wrap an already-NHWC conv weight (`[out, kH, kW, in]`) and optional bias.
    pub fn new(weight_nhwc: Array, bias: Option<Array>) -> Self {
        Self {
            weight: weight_nhwc,
            bias,
        }
    }

    /// The NHWC `[out, kH, kW, in]` weight, to feed `mlx_gen::nn::conv2d`.
    pub fn weight(&self) -> &Array {
        &self.weight
    }

    /// The optional conv bias.
    pub fn bias(&self) -> Option<&Array> {
        self.bias.as_ref()
    }

    /// Merge a conv LoRA `delta` — given in the **trained-file NCHW** `[out, in, kH, kW]` layout (what
    /// [`conv_lora_delta`] returns) — into the stored NHWC weight: transpose NCHW→NHWC, cast to the
    /// weight's dtype, and add (`W += δ`). Reproduces a reference's merged-weight conv forward
    /// bit-for-bit (a residual would differ by ~1 ULP and cascade on the chaos sampler — see
    /// [`AdaptableLinear::merge_dense_delta`]). A zero delta is a bit-exact no-op (`W + 0 == W`).
    pub fn merge_conv_delta(&mut self, delta_nchw: &Array) -> Result<()> {
        // [out, in, kH, kW] → [out, kH, kW, in] to match the stored NHWC weight.
        let delta_nhwc = delta_nchw.transpose_axes(&[0, 2, 3, 1])?;
        self.weight = add(&self.weight, &delta_nhwc.as_dtype(self.weight.dtype())?)?;
        Ok(())
    }
}

/// A module tree that can resolve a dotted parameter path (split into segments) to the
/// [`AdaptableLinear`] living there, so an adapter can be installed onto it. This is the
/// hand-written form of the macro the full adapter framework (sc-2343) will generate.
pub trait AdaptableHost {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear>;

    /// Resolve a dotted path to the [`AdaptableConv2d`] living there, for conv-layer LoRA merging
    /// (sc-2919) — the conv analog of [`adaptable_mut`](Self::adaptable_mut). The default is empty:
    /// only the SDXL U-Net (the one conv-bearing adapter host) overrides it; DiT/MMDiT families
    /// (Z-Image, Qwen, FLUX) have no conv adapter targets (their only convs live in the un-adapted
    /// VAE / patch-embed), so a conv-shaped key applied to them surfaces as skipped, never merged.
    fn adaptable_conv_mut(&mut self, _path: &[&str]) -> Option<&mut AdaptableConv2d> {
        None
    }

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

    /// Enumerate the host's **BFL / ComfyUI** fused→split adapter targets (sc-2743), the orthogonal
    /// axis to the kohya `lora_unet_` flattening of [`adaptable_paths`](Self::adaptable_paths). A
    /// [`BflTarget`](loader::BflTarget) maps one source key spelling (in any of the BFL prefix
    /// conventions — `lora_unet_…`, `diffusion_model.…`, `base_model.model.…`) to a diffusers module
    /// path, optionally row-slicing the up/down factor so a *fused* source linear (BFL `…img_attn.qkv`,
    /// `…linear1`) fans out into the model's *split* targets (`attn.to_q/to_k/to_v`, …). Mirrors the
    /// fork's `Flux2LoRAMapping._get_bfl_*` + the `base_model.model.` global renames.
    ///
    /// The default is empty — only FLUX.2/FLUX.1 expose a BFL surface (Z-Image/Qwen/SDXL have none),
    /// so a BFL file applied to a host without one surfaces every key as unmatched (loud), never
    /// silently dropped. The per-target slices MUST be byte-faithful to `LoraTransforms` (guarded by
    /// tests).
    fn bfl_targets(&self) -> Vec<loader::BflTarget> {
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
    fn lokr_residual_runs_in_activation_dtype() {
        // sc-2718: the f32 bug-workaround is gone (NAX 16-bit dense GEMM fixed at the toolchain
        // level, sc-2772). A LoKr residual now runs in the ACTIVATION dtype — bf16 on the bf16 path
        // — mirroring the fork's `scale · matmul(x, delta.astype(x.dtype).T)`. So a bf16-input LoKr
        // residual must (a) return bf16 and (b) match the f32 reference within bf16 rounding — NOT
        // diverge (which is what the old buggy bf16 GEMM produced and the f32 detour avoided).
        let delta = lokr_2x2(); // bf16
        let x32 = Array::from_slice(&[1.0f32, -2.0, 0.5, 0.25, -1.0, 2.0], &[3, 2]);
        let lokr = Adapter::Lokr {
            delta: delta.clone(),
            scale: 0.5,
        };

        let got = lokr
            .residual(&x32.as_dtype(Dtype::Bfloat16).unwrap())
            .unwrap();
        assert_eq!(
            got.dtype(),
            Dtype::Bfloat16,
            "bf16-input LoKr residual runs in the activation dtype"
        );

        let want = multiply(
            matmul(&x32, delta.as_dtype(Dtype::Float32).unwrap().t()).unwrap(),
            scalar(0.5),
        )
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        assert!(
            all_close(&got, &want, 5e-2, 5e-2, false)
                .unwrap()
                .item::<bool>(),
            "bf16 LoKr residual diverged from the f32 reference (bf16 GEMM bug?)"
        );
    }

    #[test]
    fn lora_residual_is_fork_faithful_no_forced_dtype() {
        // sc-2718: LoRA factors keep their loaded dtype and the result is NOT cast back, replicating
        // the fork's `scale · matmul(matmul(x, lora_A), lora_B)` byte-for-byte.
        let a32 = Array::from_slice(
            &(0..8).map(|i| i as f32 * 0.1 - 0.4).collect::<Vec<_>>(),
            &[2, 4],
        );
        let b32 = Array::from_slice(
            &(0..8).map(|i| i as f32 * 0.05).collect::<Vec<_>>(),
            &[4, 2],
        );
        let x_bf16 = Array::from_slice(&[1.0f32, -2.0, 0.5, 0.25, -1.0, 2.0], &[3, 2])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();

        // f32 factors (the goldens' dtype): a bf16 `x` promotes the residual to f32 — and it is
        // byte-exact to the fork's `scale · (x·A)·B` (no forced dtype, no cast-back).
        let lora_f32 = Adapter::Lora {
            a: a32.clone(),
            b: b32.clone(),
            scale: 0.5,
        };
        let got_f32 = lora_f32.residual(&x_bf16).unwrap();
        assert_eq!(
            got_f32.dtype(),
            Dtype::Float32,
            "f32 factors promote the residual to f32 (fork-faithful, not forced)"
        );
        let want_f32 = multiply(
            matmul(matmul(&x_bf16, &a32).unwrap(), &b32).unwrap(),
            scalar(0.5),
        )
        .unwrap();
        assert!(
            array_eq(&got_f32, &want_f32, false).unwrap().item::<bool>(),
            "LoRA residual must be byte-exact to the fork's scale·(x·A)·B"
        );

        // bf16 factors: the residual runs bf16 — the `[seq,r]·[r,out]` (K=rank=4≤512, M=seq=3≥2)
        // shape the NAX build mis-ran before sc-2772 — and matches the f32 reference within bf16
        // rounding (NOT garbage), proving the GEMM bug is gone so the f32 detour is unneeded.
        let lora_bf16 = Adapter::Lora {
            a: a32.as_dtype(Dtype::Bfloat16).unwrap(),
            b: b32.as_dtype(Dtype::Bfloat16).unwrap(),
            scale: 0.5,
        };
        let got_bf16 = lora_bf16.residual(&x_bf16).unwrap();
        assert_eq!(
            got_bf16.dtype(),
            Dtype::Bfloat16,
            "bf16 factors keep the residual in the activation dtype"
        );
        let want_bf16 = multiply(
            matmul(
                matmul(x_bf16.as_dtype(Dtype::Float32).unwrap(), &a32).unwrap(),
                &b32,
            )
            .unwrap(),
            scalar(0.5),
        )
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        assert!(
            all_close(&got_bf16, &want_bf16, 5e-2, 5e-2, false)
                .unwrap()
                .item::<bool>(),
            "bf16 LoRA residual diverged from the f32 reference (bf16 GEMM bug?)"
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
    fn conv_lora_delta_one_by_one_matches_hand_fold() {
        // sc-2919: a 1×1 conv LoRA (rank 2, in 2, out 2). down/up are `[*, *, 1, 1]`; the fused
        // delta is `Σ_r up[o,r]·down[r,i]`, scaled by alpha/rank. Hand-computed independently:
        //   down2 = [[1,2],[3,4]] (rank,in); up2 = [[5,6],[7,8]] (out,rank)
        //   δ[0,0]=5·1+6·3=23  δ[0,1]=5·2+6·4=34  δ[1,0]=7·1+8·3=31  δ[1,1]=7·2+8·4=46
        //   eff = alpha/rank = 4/2 = 2 → [[46,68],[62,92]]
        let down = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2, 1, 1]);
        let up = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2, 1, 1]);
        let delta = conv_lora_delta(&down, &up, 4.0, 2.0, 1.0).unwrap();
        assert_eq!(delta.shape(), &[2, 2, 1, 1]);
        let want = Array::from_slice(&[46.0f32, 68.0, 62.0, 92.0], &[2, 2, 1, 1]);
        assert!(all_close(&delta, &want, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn conv_lora_delta_kxk_rank1_broadcasts_spatial_kernel() {
        // sc-2919: a 3×3-shaped (here 2×2) conv LoRA with rank 1 reduces to `δ[o,i,y,x] =
        // up[o]·down[0,i,y,x]` — proving the spatial kernel is preserved (not collapsed). in=1, out=2.
        //   down[0,0,:,:] = [[1,2],[3,4]]; up = [10, 20]
        //   δ[0] = 10·[1,2,3,4] = [10,20,30,40];  δ[1] = 20·[...] = [20,40,60,80]
        let down = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let up = Array::from_slice(&[10.0f32, 20.0], &[2, 1, 1, 1]);
        let delta = conv_lora_delta(&down, &up, 1.0, 1.0, 1.0).unwrap();
        assert_eq!(delta.shape(), &[2, 1, 2, 2]);
        let want = Array::from_slice(
            &[10.0f32, 20.0, 30.0, 40.0, 20.0, 40.0, 60.0, 80.0],
            &[2, 1, 2, 2],
        );
        assert!(all_close(&delta, &want, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>());
        // The user scale composes multiplicatively (scale 0 ⇒ a zero delta ⇒ no-op merge).
        let zero = conv_lora_delta(&down, &up, 1.0, 1.0, 0.0).unwrap();
        assert!(
            array_eq(&zero, Array::zeros::<f32>(&[2, 1, 2, 2]).unwrap(), false)
                .unwrap()
                .item::<bool>()
        );
    }

    #[test]
    fn merge_conv_delta_transposes_nchw_and_adds() {
        // sc-2919: NHWC weight `[out=1, kH=1, kW=1, in=2]` = [1, 2]; a 1×1 NCHW delta
        // `[out=1, in=2, kH=1, kW=1]` = [0.5, 0.25] transposes to NHWC [0.5, 0.25] and adds → [1.5, 2.25].
        let w = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 1, 2]);
        let mut conv = AdaptableConv2d::new(w.clone(), None);
        let delta = Array::from_slice(&[0.5f32, 0.25], &[1, 2, 1, 1]);
        conv.merge_conv_delta(&delta).unwrap();
        let want = Array::from_slice(&[1.5f32, 2.25], &[1, 1, 1, 2]);
        assert!(array_eq(conv.weight(), &want, false)
            .unwrap()
            .item::<bool>());

        // A zero delta is a bit-exact no-op.
        let mut conv2 = AdaptableConv2d::new(w.clone(), None);
        conv2
            .merge_conv_delta(&Array::zeros::<f32>(&[1, 2, 1, 1]).unwrap())
            .unwrap();
        assert!(array_eq(conv2.weight(), &w, false).unwrap().item::<bool>());
    }

    #[test]
    fn merge_conv_delta_kxk_zero_is_noop_and_nonzero_lands_in_nhwc() {
        // sc-2919 regression: with a k×k (here 3×3, out≠in) kernel the NCHW→NHWC transpose is a real
        // permutation (not the trivial 1×1 case), so a zero delta must STILL be a bit-exact no-op — i.e.
        // the merge must not permute/scramble the weight. NHWC weight [out=2, kH=3, kW=3, in=4].
        let n = 2 * 3 * 3 * 4;
        let wv: Vec<f32> = (0..n).map(|i| i as f32 * 0.01 - 0.3).collect();
        let w = Array::from_slice(&wv, &[2, 3, 3, 4]);

        let mut conv = AdaptableConv2d::new(w.clone(), None);
        conv.merge_conv_delta(&Array::zeros::<f32>(&[2, 4, 3, 3]).unwrap())
            .unwrap();
        assert_eq!(
            conv.weight().as_slice::<f32>(),
            w.as_slice::<f32>(),
            "a zero k×k conv delta must be a bit-exact no-op (no permutation/scramble)"
        );

        // A nonzero NCHW delta must land at the matching NHWC position: δ_nchw[o,i,y,x] adds to
        // weight_nhwc[o,y,x,i]. Put a single spike at nchw (o=1,i=2,y=0,x=2) → nhwc (1,0,2,2).
        let mut spike = vec![0f32; n];
        // nchw flat index for [2,4,3,3] at (1,2,0,2) = ((1*4+2)*3+0)*3+2 = 56.
        spike[56] = 5.0;
        let mut conv2 = AdaptableConv2d::new(w.clone(), None);
        conv2
            .merge_conv_delta(&Array::from_slice(&spike, &[2, 4, 3, 3]))
            .unwrap();
        // nhwc flat index for [2,3,3,4] at (1,0,2,2) = ((1*3+0)*3+2)*4+2 = 46.
        let nhwc_idx = 46usize;
        let got = conv2.weight().as_slice::<f32>();
        for (j, (&g, &b)) in got.iter().zip(&wv).enumerate() {
            let want = if j == nhwc_idx { b + 5.0 } else { b };
            assert_eq!(
                g, want,
                "conv delta landed at wrong NHWC index (got change at {j})"
            );
        }
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
