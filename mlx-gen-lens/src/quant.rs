//! Packed (pre-quantized) weight loading — the consume side of [`crate::convert`].
//!
//! A pre-quantized Q4/Q8 snapshot stores each quantized Linear as the packed triple
//! `{base}.weight` (u32 codes) + `{base}.scales` + `{base}.biases`. The shared
//! [`mlx_gen::quant::lin`] **auto-detects** it by the presence of `{base}.scales` and builds the
//! quantized module directly — so a published Q4 snapshot loads packed with no dense bf16/f32
//! transient. A dense snapshot (no `.scales`) loads dense exactly as before, so the same loader
//! serves both.
//!
//! Lens quantizes **two** components (wired in [`crate::pipeline::LensPipeline::load_quant`] +
//! [`crate::pipeline::LensPipeline::quantize_dit`]):
//!
//! * **DiT** ([`crate::dit::LensTransformer::quantize`]) — the compute-heavy linears `img_in`,
//!   `txt_in`, `proj_out` and every block's fused-QKV attention projections (`img_qkv`, `txt_qkv`,
//!   `to_out.0`, `to_add_out`) + bias-less SwiGLU MLPs (`img_mlp`/`txt_mlp` `w1`/`w2`/`w3`). The
//!   timestep embedder, the AdaLN modulations (`img_mod`/`txt_mod`/`norm_out.linear`), and every
//!   RMSNorm/QK-norm stay full precision. These are ordinary diffusers `[out, in]` `.weight` (+ bias)
//!   Linears, so the shared [`lin`] loader below serves them verbatim.
//! * **gpt-oss encoder MoE experts** ([`crate::text_encoder::encoder::LensTextEncoder`] via
//!   [`crate::text_encoder::gpt_oss::GptOssMoe`]) — the 20 B-param bulk. In the DENSE source these are
//!   **MXFP4** (`experts.{gate_up,down}_proj_{blocks,scales}`), dequantized then re-quantized to MLX
//!   group-wise affine Q4/Q8 at load. In a **packed** snapshot they are stored as the stacked triple
//!   `experts.{gate_up,down}_proj.{weight,scales,biases}` `[E, out, …]` that
//!   [`load_packed_experts`] slices per-expert — byte-identical to the load-time
//!   `Proj::into_quantized` (per-row affine quant commutes with the axis-0 expert split, the same
//!   argument as SDXL's GEGLU row-slice). The router / attention / embedding / norms stay dense.
//!
//! The VAE is the shared Flux.2 decoder and is **never quantized** (runs f32) — every tier ships it
//! dense. The optional [`crate::reasoner`] is a *separate* gpt-oss copy loaded on demand (off by
//! default) and is **not** on the pipeline quant path, so it is not packed here (it dequantizes its
//! own MXFP4 experts if ever attached — mirroring the dense load).
//!
//! Group-B per-crate template (sc-8669 / sc-8763), a thin wrapper over the shared
//! `mlx_gen::quant::{lin, DEFAULT_GROUP_SIZE}`.

#[cfg(test)]
use mlx_rs::ops::split;
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

/// Group size the converter writes — the codebase-wide `mlx_gen::quant::DEFAULT_GROUP_SIZE` (64),
/// matching the load-time `.quantize` seams (`AdaptableLinear::quantize` /
/// `Proj::into_quantized`, which both default to 64).
pub(crate) const GROUP_SIZE: i32 = mlx_gen::quant::DEFAULT_GROUP_SIZE;

/// Load `{base}` as an [`AdaptableLinear`] at Lens's [`GROUP_SIZE`] — packed when `{base}.scales`
/// is present (a pre-quantized snapshot), else dense. The shared [`mlx_gen::quant::lin`]. Used by the
/// DiT loader for every quantizable projection.
pub(crate) fn lin(w: &Weights, base: &str, bias: bool) -> Result<AdaptableLinear> {
    mlx_gen::quant::lin(w, base, bias, GROUP_SIZE)
}

/// Bits inferred from the **stacked** packed shapes at [`GROUP_SIZE`]: the expert triple is
/// `[E, out, …]`, so the last axis carries the packed columns — `scales` `[E, out, in/gs]` ⇒
/// `in = scales.cols·gs`; the u32-packed `weight` `[E, out, in·bits/32]` ⇒ `bits = wq.cols·32/in`.
/// The 3-D MoE analogue of the shared `mlx_gen::quant::packed_bits` (which handles the 2-D
/// per-linear triple), letting the expert slice build its `Proj::Quant` parts without threading
/// bits through a side manifest.
///
/// F-011 (fix-travel): a corrupt/mis-converted packed snapshot is untrusted input — a non-3-D
/// stack would panic on the shape index, zero scales cols would integer-divide-by-zero, and a
/// mis-packed weight yields a bit-width outside Q4/Q8. All three are typed errors instead.
fn packed_bits_stacked(wq: &Array, scales: &Array) -> Result<i32> {
    let wshape = wq.shape();
    let sshape = scales.shape();
    if wshape.len() != 3 || sshape.len() != 3 {
        return Err(Error::Msg(format!(
            "packed experts: weight and scales must be 3-D [E, out, …], got weight {:?} / \
             scales {:?}",
            wshape, sshape
        )));
    }
    let in_dim = sshape[2] * GROUP_SIZE;
    if in_dim == 0 {
        return Err(Error::Msg(format!(
            "packed experts: zero input dim (scales cols {} × group_size {GROUP_SIZE})",
            sshape[2]
        )));
    }
    let bits = wshape[2] * 32 / in_dim;
    if !matches!(bits, 4 | 8) {
        return Err(Error::Msg(format!(
            "packed experts: inferred bit-width {bits} ∉ {{4, 8}} (weight cols {}, in_dim \
             {in_dim}); snapshot is corrupt or mis-converted",
            wshape[2]
        )));
    }
    Ok(bits)
}

/// A stacked packed MoE expert projection `[E, out, …]` for grouped GEMM (F-021, sc-9500) — the
/// forward-time consumer of a packed turnkey. `wq`/`scales`/`biases` are `[E, out, …]`, `bias` is the
/// dense `[E, out]` expert bias; `group_size`/`bits` describe the pack. Fed straight to
/// [`crate::text_encoder::gpt_oss::GptOssMoe`]'s stacked `gather_qmm` bank (no per-expert split).
pub(crate) struct StackedPack {
    pub wq: Array,
    pub scales: Array,
    pub biases: Array,
    pub bias: Array,
    pub group_size: i32,
    pub bits: i32,
}

/// Load the stacked packed `{name}_proj` for all `E` experts from
/// `{prefix}.experts.{name}_proj.{weight,scales,biases}` `[E, out, …]` + the dense bias
/// `{prefix}.experts.{name}_proj_bias` `[E, out]`, inferring `bits` from the shapes. Used as-is
/// (no split) by the grouped-GEMM MoE forward — the on-disk stacked layout is exactly what
/// `gather_qmm` indexes per token via `rhs_indices`.
pub(crate) fn load_packed_stack(w: &Weights, prefix: &str, name: &str) -> Result<StackedPack> {
    let base = format!("{prefix}.experts.{name}_proj");
    let wq = w.require(&format!("{base}.weight"))?; // [E, out, in*bits/32]
    let scales = w.require(&format!("{base}.scales"))?; // [E, out, in/gs]
    let biases = w.require(&format!("{base}.biases"))?; // [E, out, in/gs]
    let bias = w.require(&format!("{prefix}.experts.{name}_proj_bias"))?; // [E, out]
    let bits = packed_bits_stacked(wq, scales)?;
    Ok(StackedPack {
        wq: wq.clone(),
        scales: scales.clone(),
        biases: biases.clone(),
        bias: bias.clone(),
        group_size: GROUP_SIZE,
        bits,
    })
}

/// Whether `{prefix}.experts.{name}_proj.scales` is present — i.e. this is a **packed** encoder
/// snapshot for that expert projection (`name` = `"gate_up"` / `"down"`). Distinguishes a
/// pre-quantized turnkey from the dense MXFP4 source without reading a manifest.
pub(crate) fn has_packed_experts(w: &Weights, prefix: &str, name: &str) -> bool {
    w.get(&format!("{prefix}.experts.{name}_proj.scales"))
        .is_some()
}

/// One packed MoE expert projection's parts, sliced from a stacked `[E, …]` pre-quantized triple —
/// **test-only** (the byte-identity round-trip oracle below). The forward path consumes the stack
/// whole via [`load_packed_stack`]; only [`load_packed_experts_from_stack`] still slices per expert,
/// to prove that slicing commutes with the load-time per-expert `Proj::into_quantized`.
#[cfg(test)]
pub(crate) struct PackedExpertProj {
    pub wq: Array,
    pub scales: Array,
    pub biases: Array,
    pub bias: Array,
}

/// Split a stacked packed expert triple + dense bias (`[E, out, …]` / `[E, out]`) into one
/// [`PackedExpertProj`] per expert (axis-0 split + squeeze) — **test-only**. Byte-identical to the
/// per-expert load-time pack because affine quantization is per-row (axis 0 = expert), so the
/// stack/split commutes with the per-expert quantize; the round-trip test asserts exactly this.
#[cfg(test)]
pub(crate) fn load_packed_experts_from_stack(
    wq: &Array,
    scales: &Array,
    biases: &Array,
    bias: &Array,
) -> Result<Vec<PackedExpertProj>> {
    // Shape-validating (F-011) — must come before the `shape()[0]` read below.
    packed_bits_stacked(wq, scales)?;
    let e = wq.shape()[0];

    let wq_e = split(wq, e, 0)?;
    let sc_e = split(scales, e, 0)?;
    let bi_e = split(biases, e, 0)?;
    let bs_e = split(bias, e, 0)?;

    let mut out = Vec::with_capacity(e as usize);
    for i in 0..e as usize {
        out.push(PackedExpertProj {
            wq: wq_e[i].squeeze_axes(&[0])?,
            scales: sc_e[i].squeeze_axes(&[0])?,
            biases: bi_e[i].squeeze_axes(&[0])?,
            bias: bs_e[i].squeeze_axes(&[0])?,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A well-formed stacked Q4 triple derives bits = 4: scales `[E=2, out=8, in/gs=2]` ⇒ in 128;
    /// wq `[2, 8, 128·4/32=16]`.
    #[test]
    fn packed_bits_stacked_derives_q4() {
        let scales = Array::zeros::<f32>(&[2, 8, 2]).unwrap();
        let wq = Array::zeros::<u32>(&[2, 8, 16]).unwrap();
        assert_eq!(packed_bits_stacked(&wq, &scales).unwrap(), 4);
    }

    /// wq `[2, 8, 128·8/32=32]` on the same scales ⇒ bits 8.
    #[test]
    fn packed_bits_stacked_derives_q8() {
        let scales = Array::zeros::<f32>(&[2, 8, 2]).unwrap();
        let wq = Array::zeros::<u32>(&[2, 8, 32]).unwrap();
        assert_eq!(packed_bits_stacked(&wq, &scales).unwrap(), 8);
    }

    /// F-011: a 2-D (non-stacked) triple must be a typed error, not a shape-index panic.
    #[test]
    fn packed_bits_stacked_rejects_non_3d() {
        let scales = Array::zeros::<f32>(&[8, 2]).unwrap();
        let wq = Array::zeros::<u32>(&[8, 16]).unwrap();
        let err = packed_bits_stacked(&wq, &scales).unwrap_err().to_string();
        assert!(err.contains("must be 3-D"), "{err}");
    }

    /// F-011: `[E, out, 0]` scales ⇒ in_dim 0 — the pre-fix integer divide-by-zero.
    #[test]
    fn packed_bits_stacked_rejects_zero_cols() {
        let scales = Array::zeros::<f32>(&[2, 8, 0]).unwrap();
        let wq = Array::zeros::<u32>(&[2, 8, 16]).unwrap();
        let err = packed_bits_stacked(&wq, &scales).unwrap_err().to_string();
        assert!(err.contains("zero input dim"), "{err}");
    }

    /// F-011: a mis-packed weight (bits 3 here) is outside Q4/Q8 ⇒ typed error.
    #[test]
    fn packed_bits_stacked_rejects_non_q4_q8_width() {
        // scales cols 4 ⇒ in_dim 256; wq cols 24 ⇒ bits 24·32/256 = 3.
        let scales = Array::zeros::<f32>(&[2, 8, 4]).unwrap();
        let wq = Array::zeros::<u32>(&[2, 8, 24]).unwrap();
        let err = packed_bits_stacked(&wq, &scales).unwrap_err().to_string();
        assert!(err.contains("∉ {4, 8}"), "{err}");
    }
}
