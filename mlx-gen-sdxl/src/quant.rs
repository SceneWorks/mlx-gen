//! Packed (pre-quantized) weight loading — the consume side of [`crate::convert`].
//!
//! A pre-quantized Q4/Q8 snapshot stores each quantized Linear as the packed triple
//! `{base}.weight` (u32 codes) + `{base}.scales` + `{base}.biases`. The [`lin`] loader
//! **auto-detects** it by the presence of `{base}.scales` and builds the quantized module directly —
//! so a published Q4 snapshot loads packed with no dense fp16/f32 transient. A dense snapshot (no
//! `.scales`) loads dense exactly as before, so the same loader serves both.
//!
//! SDXL quantizes **three** components (the fork's `nn.quantize`, wired in [`crate::model::load`]):
//! the U-Net's true Linears (time/add-embedding MLPs, every cross-attention transformer's
//! attention + GEGLU FFN + `proj_in`/`proj_out`, each ResNet's `time_emb_proj`, plus the Kolors
//! `encoder_hid_proj` when present) and BOTH CLIP text encoders (q/k/v/out + mlp fc1/fc2 + the TE2
//! `text_projection`). The **convs stay dense** (`conv_in`/`conv_out`, resnet `conv1`/`conv2`, the
//! up/down samplers, and the 1×1-`conv_shortcut`-as-Linear — sc-3329), the norms stay dense, and the
//! CLIP token/position embeddings stay dense (gather lookups, not matmuls) — the converter's
//! per-component predicates match this scope exactly. The **VAE is never quantized** (it runs f32),
//! so all three tiers ship a dense VAE.
//!
//! This is the Group-B per-crate template (sc-8669), a thin wrapper over `mlx_gen::quant::lin` — plus
//! [`lin_geglu_half`], the one SDXL-specific twist: the GEGLU `ff.net.0.proj` is stored on disk as a
//! single `[2·hidden, D]` tensor and row-split into the value/gate halves at load. Because
//! group-wise affine quantization is per-row independent, packing the whole `[2·hidden, D]` and
//! row-slicing the packed triple `[lo:hi]` on load is byte-identical to the load-time split-then-
//! quantize the dense path runs.

use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// Group size the converter writes — the codebase-wide `mlx_gen::quant::DEFAULT_GROUP_SIZE` (64).
pub(crate) const GROUP_SIZE: i32 = 64;

/// Load `{base}` as an [`AdaptableLinear`] at SDXL's [`GROUP_SIZE`] — packed when `{base}.scales`
/// is present (a pre-quantized snapshot), else dense. The shared [`mlx_gen::quant::lin`].
pub(crate) fn lin(w: &Weights, base: &str, bias: bool) -> Result<AdaptableLinear> {
    mlx_gen::quant::lin(w, base, bias, GROUP_SIZE)
}

/// Bits inferred from the packed shapes at [`GROUP_SIZE`]: `scales` is `[out, in/gs]` ⇒
/// `in = scales.cols·gs`; the u32-packed `weight` is `[out, in·bits/32]` ⇒ `bits = wq.cols·32/in`.
/// (Duplicates the private `mlx_gen::quant::packed_bits` so [`lin_geglu_half`] can build the
/// `from_quantized_parts` for a row-slice without threading bits through a manifest.)
fn packed_bits(wq: &Array, scales: &Array) -> i32 {
    let in_dim = scales.shape()[1] * GROUP_SIZE;
    wq.shape()[1] * 32 / in_dim
}

/// Load the value/gate **half** of a GEGLU `ff.net.0.proj` — rows `[lo, hi)` of the stored
/// `[2·hidden, D]` Linear (bias included). Packed when `{base}.scales` is present (row-slice the u32
/// codes + scales + biases + dense bias along axis 0 — valid because group-wise affine quantization
/// is per-row independent, so this is byte-identical to the dense path's split-then-quantize), else
/// row-slice the dense weight + bias. `base` is `…transformer_blocks.{k}.ff.net.0.proj`.
pub(crate) fn lin_geglu_half(w: &Weights, base: &str, lo: i32, hi: i32) -> Result<AdaptableLinear> {
    let idx = Array::from_slice(&(lo..hi).collect::<Vec<i32>>(), &[hi - lo]);
    let row_slice = |a: &Array| -> Result<Array> { Ok(a.take_axis(&idx, 0)?) };
    let bias_full = w.require(&format!("{base}.bias"))?;
    let bias = Some(row_slice(bias_full)?);
    if let Some(scales) = w.get(&format!("{base}.scales")) {
        let wq = w.require(&format!("{base}.weight"))?;
        let biases = w.require(&format!("{base}.biases"))?;
        let bits = packed_bits(wq, scales);
        return Ok(AdaptableLinear::from_quantized_parts(
            row_slice(wq)?,
            row_slice(scales)?,
            row_slice(biases)?,
            bias,
            GROUP_SIZE,
            bits,
        ));
    }
    Ok(AdaptableLinear::dense(
        row_slice(w.require(&format!("{base}.weight"))?)?,
        bias,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{eq, quantize};
    use mlx_rs::Dtype;

    /// The invariant that makes [`lin_geglu_half`]'s packed row-slice byte-identical to the dense
    /// split-then-quantize: group-wise affine quantization is per-row, so row-slicing axis 0 of the
    /// packed triple commutes with the split. Pack a `[4, 128]` weight whole, row-slice `[0:2]` of
    /// the packed codes, and compare to packing just rows `[0:2]` of the dense weight — byte-equal.
    #[test]
    fn packed_row_slice_commutes_with_split() {
        let full = Array::from_slice(
            &(0..4 * 128).map(|i| (i as f32).cos()).collect::<Vec<_>>(),
            &[4, 128],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let (wq, _sc, _bi) = quantize(&full, GROUP_SIZE, 4).unwrap();
        let idx = Array::from_slice(&[0i32, 1], &[2]);
        let sliced_wq = wq.take_axis(&idx, 0).unwrap();
        let half = full.take_axis(&idx, 0).unwrap();
        let (ewq, _, _) = quantize(&half, GROUP_SIZE, 4).unwrap();
        assert!(
            eq(&sliced_wq, &ewq)
                .unwrap()
                .all(None)
                .unwrap()
                .item::<bool>(),
            "packed row-slice must equal packing the sliced rows"
        );
    }
}
