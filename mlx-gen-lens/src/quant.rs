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

use mlx_rs::ops::split;
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

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

/// Bits inferred from the packed shapes at [`GROUP_SIZE`]: `scales` last-axis is `in/gs` ⇒
/// `in = scales.cols·gs`; the u32-packed `weight` last-axis is `in·bits/32` ⇒
/// `bits = wq.cols·32/in`. (Duplicates the private `mlx_gen::quant::packed_bits` so the MoE expert
/// slice can build its `Proj::Quant` parts without threading bits through a side manifest.)
fn packed_bits(wq_cols: i32, scales_cols: i32) -> i32 {
    let in_dim = scales_cols * GROUP_SIZE;
    wq_cols * 32 / in_dim
}

/// One packed MoE expert projection's parts, sliced from a stacked `[E, …]` pre-quantized triple.
/// `wq`/`scales`/`biases` are `[out, …]` (expert `e`'s slice, leading E axis dropped), `bias` is the
/// dense `[out]` expert bias; `group_size`/`bits` describe the pack. Fed straight to
/// [`crate::text_encoder::gpt_oss::Proj::from_packed_parts`].
pub(crate) struct PackedExpertProj {
    pub wq: Array,
    pub scales: Array,
    pub biases: Array,
    pub bias: Array,
    pub group_size: i32,
    pub bits: i32,
}

/// Whether `{prefix}.experts.{name}_proj.scales` is present — i.e. this is a **packed** encoder
/// snapshot for that expert projection (`name` = `"gate_up"` / `"down"`). Distinguishes a
/// pre-quantized turnkey from the dense MXFP4 source without reading a manifest.
pub(crate) fn has_packed_experts(w: &Weights, prefix: &str, name: &str) -> bool {
    w.get(&format!("{prefix}.experts.{name}_proj.scales"))
        .is_some()
}

/// Load all `E` experts' packed `{name}_proj` from the stacked triple
/// `{prefix}.experts.{name}_proj.{weight,scales,biases}` `[E, out, …]` + the dense bias
/// `{prefix}.experts.{name}_proj_bias` `[E, out]`, returning one [`PackedExpertProj`] per expert.
///
/// The stack is split along axis 0 (E) into `[out, …]` per-expert parts — byte-identical to the
/// dense path's per-expert `Proj::into_quantized`, because group-wise affine quantization is per-row
/// (axis 0 = expert here), so slicing the packed stack commutes with quantizing each expert
/// separately (the SDXL GEGLU row-slice argument, one axis up). `e` is read from the weight's shape.
pub(crate) fn load_packed_experts(
    w: &Weights,
    prefix: &str,
    name: &str,
) -> Result<Vec<PackedExpertProj>> {
    let base = format!("{prefix}.experts.{name}_proj");
    load_packed_experts_from_stack(
        w.require(&format!("{base}.weight"))?, // [E, out, in*bits/32]
        w.require(&format!("{base}.scales"))?, // [E, out, in/gs]
        w.require(&format!("{base}.biases"))?, // [E, out, in/gs]
        w.require(&format!("{prefix}.experts.{name}_proj_bias"))?, // [E, out]
    )
}

/// Split a stacked packed expert triple + dense bias (`[E, out, …]` / `[E, out]`) into one
/// [`PackedExpertProj`] per expert (axis-0 split + squeeze). Byte-identical to the per-expert
/// load-time pack because affine quantization is per-row (axis 0 = expert), so the stack/split
/// commutes with the per-expert quantize. Shared by [`load_packed_experts`] and the round-trip test.
pub(crate) fn load_packed_experts_from_stack(
    wq: &Array,
    scales: &Array,
    biases: &Array,
    bias: &Array,
) -> Result<Vec<PackedExpertProj>> {
    let e = wq.shape()[0];
    let bits = packed_bits(wq.shape()[2], scales.shape()[2]);

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
            group_size: GROUP_SIZE,
            bits,
        });
    }
    Ok(out)
}
