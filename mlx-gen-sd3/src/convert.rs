//! Native (Rust/MLX) Stable Diffusion 3.5 **diffusers → MLX** transformer converter +
//! architecture validation (sc-7860, the SD3.5 E1 slice).
//!
//! Stable Diffusion 3.5 ships its `transformer/` as a standard diffusers `SD3Transformer2DModel`
//! safetensors tree (bf16). Unlike the FLUX.2-klein single-file converter (which row-splits a fused
//! BFL `qkv` and swaps an adaLN half), the SD3.5 diffusers layout is **already** the unfused,
//! per-projection convention this crate's MMDiT (E3) consumes — `to_q`/`to_k`/`to_v`/`to_out.0` on
//! the image stream and `add_q_proj`/`add_k_proj`/`add_v_proj`/`to_add_out` on the text stream — so
//! the key map is a pure 1:1 rename (here, the identity). The load-bearing work in E1 is therefore
//! the **architecture validation**: an exhaustive, shape-checked expected-tensor table derived from
//! the real-weight-confirmed arch (sc-7850), against which a converted/known tensor set is asserted.
//! This catches a wrong-repo / wrong-shape / truncated checkpoint at convert time rather than as
//! garbage at generate time.
//!
//! The arch facts (38 layers, hidden 2432, 38 heads, head_dim 64, patch 2, 16-ch in/out, joint
//! 4096, pooled 2048, caption 2432, qk_norm rms_norm, learned pos_embed `[1, 36864, 2432]` +
//! `pos_embed.proj` `[2432, 16, 2, 2]`, NO RoPE) are documented in [`crate::config`].
//!
//! Diffusers `SD3Transformer2DModel` top-level keys (real-weight confirmed):
//!   * `pos_embed.pos_embed`                                    `[1, 36864, 2432]` (learned, NO RoPE)
//!   * `pos_embed.proj.{weight,bias}`                           `[2432, 16, 2, 2]` / `[2432]`
//!   * `time_text_embed.timestep_embedder.linear_1.{weight,bias}`   `[2432, 256]` / `[2432]`
//!   * `time_text_embed.timestep_embedder.linear_2.{weight,bias}`   `[2432, 2432]`
//!   * `time_text_embed.text_embedder.linear_1.{weight,bias}`       `[2432, 2048]` / `[2432]`
//!   * `time_text_embed.text_embedder.linear_2.{weight,bias}`       `[2432, 2432]`
//!   * `context_embedder.{weight,bias}`                             `[2432, 4096]` / `[2432]`
//!   * `norm_out.linear.{weight,bias}`                             `[4864, 2432]` (AdaLN-continuous, 2·hidden)
//!   * `proj_out.{weight,bias}`                                     `[64, 2432]` / `[64]`
//!
//! Per `transformer_blocks.{i}` (all 38 are double-stream joint blocks; the LAST, `i == 37`, is
//! `context_pre_only` — it drops `attn.to_add_out`, `ff_context.*`, and its `norm1_context.linear`
//! is AdaLN-continuous `[2·hidden]` not AdaLN-zero `[6·hidden]`):
//!   * `norm1.linear.{weight,bias}`            `[6·hidden, hidden]`  (AdaLayerNormZero)
//!   * `norm1_context.linear.{weight,bias}`    `[6·hidden, hidden]`  (AdaLayerNormZero; LAST: `[2·hidden]`)
//!   * `attn.to_q/to_k/to_v.{weight,bias}`     `[hidden, hidden]`
//!   * `attn.to_out.0.{weight,bias}`           `[hidden, hidden]`
//!   * `attn.add_q_proj/add_k_proj/add_v_proj.{weight,bias}`  `[hidden, hidden]`
//!   * `attn.to_add_out.{weight,bias}`         `[hidden, hidden]`    (absent on the LAST block)
//!   * `attn.norm_q/norm_k/norm_added_q/norm_added_k.weight`  `[head_dim]`  (RMSNorm, no bias)
//!   * `ff.net.0.proj.{weight,bias}`           `[4·hidden, hidden]`  (GELU-approx gate)
//!   * `ff.net.2.{weight,bias}`                `[hidden, 4·hidden]`
//!   * `ff_context.net.0.proj.{weight,bias}`   `[4·hidden, hidden]`  (absent on the LAST block)
//!   * `ff_context.net.2.{weight,bias}`        `[hidden, 4·hidden]`  (absent on the LAST block)

use std::collections::HashMap;
use std::path::Path;

use mlx_gen::quant::{load_dir_map, quantize_map, save_map};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::transforms::eval;
use mlx_rs::Array;

use crate::config::Sd3Arch;

/// AdaLayerNormZero packs 6 chunks (shift/scale/gate × msa + mlp). The non-final blocks' `norm1`
/// and `norm1_context` both use it.
const ADALN_ZERO_CHUNKS: usize = 6;
/// AdaLayerNormContinuous packs 2 chunks (shift, scale). The final block's `norm1_context` and the
/// model-level `norm_out` both use it.
const ADALN_CONT_CHUNKS: usize = 2;
/// diffusers `FeedForward(activation_fn="gelu-approximate")` hidden expansion factor.
const FF_MULT: usize = 4;

/// An expected tensor: its diffusers key and its shape (with `-1` meaning "any" for dims that vary
/// with the input image — none here, the transformer's tensors are all fixed-shape).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpectedTensor {
    pub key: String,
    pub shape: Vec<i64>,
}

impl ExpectedTensor {
    fn new(key: impl Into<String>, shape: impl Into<Vec<i64>>) -> Self {
        Self {
            key: key.into(),
            shape: shape.into(),
        }
    }
}

fn hidden(a: &Sd3Arch) -> i64 {
    a.hidden() as i64
}

/// The exhaustive expected diffusers tensor set (key + shape) for an `SD3Transformer2DModel` of the
/// given arch. This is the **architecture-validation ground truth** — every tensor the converter is
/// expected to produce and the E3 MMDiT loader is expected to consume, with no extras and none
/// missing. Pure host-side derivation from [`Sd3Arch`]; no weights, no I/O.
pub fn expected_transformer_tensors(arch: &Sd3Arch) -> Vec<ExpectedTensor> {
    let h = hidden(arch);
    // ---- top-level (non-block) tensors --------------------------------------------------------
    // Learned positional table (NO RoPE). [1, pos_embed_max_size^2, hidden].
    let mut out: Vec<ExpectedTensor> = vec![ExpectedTensor::new(
        "pos_embed.pos_embed",
        vec![1, arch.pos_embed_len() as i64, h],
    )];
    // Patchify Conv2d: [hidden, in_channels, patch, patch].
    out.push(ExpectedTensor::new(
        "pos_embed.proj.weight",
        vec![
            h,
            arch.in_channels as i64,
            arch.patch_size as i64,
            arch.patch_size as i64,
        ],
    ));
    out.push(ExpectedTensor::new("pos_embed.proj.bias", vec![h]));

    // Combined timestep + pooled-text embedder.
    out.push(ExpectedTensor::new(
        "time_text_embed.timestep_embedder.linear_1.weight",
        vec![h, arch.time_proj_dim as i64],
    ));
    out.push(ExpectedTensor::new(
        "time_text_embed.timestep_embedder.linear_1.bias",
        vec![h],
    ));
    out.push(ExpectedTensor::new(
        "time_text_embed.timestep_embedder.linear_2.weight",
        vec![h, h],
    ));
    out.push(ExpectedTensor::new(
        "time_text_embed.timestep_embedder.linear_2.bias",
        vec![h],
    ));
    out.push(ExpectedTensor::new(
        "time_text_embed.text_embedder.linear_1.weight",
        vec![h, arch.pooled_projection_dim as i64],
    ));
    out.push(ExpectedTensor::new(
        "time_text_embed.text_embedder.linear_1.bias",
        vec![h],
    ));
    out.push(ExpectedTensor::new(
        "time_text_embed.text_embedder.linear_2.weight",
        vec![h, h],
    ));
    out.push(ExpectedTensor::new(
        "time_text_embed.text_embedder.linear_2.bias",
        vec![h],
    ));

    // Context (T5/CLIP joint) embedder: [caption_projection_dim, joint_attention_dim].
    out.push(ExpectedTensor::new(
        "context_embedder.weight",
        vec![
            arch.caption_projection_dim as i64,
            arch.joint_attention_dim as i64,
        ],
    ));
    out.push(ExpectedTensor::new(
        "context_embedder.bias",
        vec![arch.caption_projection_dim as i64],
    ));

    // ---- per-block tensors --------------------------------------------------------------------
    for i in 0..arch.num_layers {
        let is_last = i + 1 == arch.num_layers;
        out.extend(expected_block_tensors(arch, i, is_last));
    }

    // ---- output head --------------------------------------------------------------------------
    // AdaLayerNormContinuous modulation (shift, scale) over the final hidden.
    out.push(ExpectedTensor::new(
        "norm_out.linear.weight",
        vec![ADALN_CONT_CHUNKS as i64 * h, h],
    ));
    out.push(ExpectedTensor::new(
        "norm_out.linear.bias",
        vec![ADALN_CONT_CHUNKS as i64 * h],
    ));
    // Unpatchify projection: [patch*patch*out_channels, hidden].
    out.push(ExpectedTensor::new(
        "proj_out.weight",
        vec![arch.patch_out_dim() as i64, h],
    ));
    out.push(ExpectedTensor::new(
        "proj_out.bias",
        vec![arch.patch_out_dim() as i64],
    ));

    out
}

/// Expected tensors for one `transformer_blocks.{i}` joint block. `is_last` ⇒ `context_pre_only`:
/// the text stream is read-only after attention, so `attn.to_add_out`, `ff_context.*`, and the
/// AdaLN-zero `norm1_context` are replaced/dropped (its `norm1_context` becomes AdaLN-continuous).
fn expected_block_tensors(arch: &Sd3Arch, i: usize, is_last: bool) -> Vec<ExpectedTensor> {
    let h = hidden(arch);
    let head = arch.head_dim as i64;
    let p = format!("transformer_blocks.{i}");
    let mut t: Vec<ExpectedTensor> = Vec::new();

    let lin = |t: &mut Vec<ExpectedTensor>, name: &str, out_dim: i64, in_dim: i64| {
        t.push(ExpectedTensor::new(
            format!("{p}.{name}.weight"),
            vec![out_dim, in_dim],
        ));
        t.push(ExpectedTensor::new(
            format!("{p}.{name}.bias"),
            vec![out_dim],
        ));
    };

    // AdaLN modulation for the image stream (always AdaLN-zero, 6 chunks).
    lin(&mut t, "norm1.linear", ADALN_ZERO_CHUNKS as i64 * h, h);
    // AdaLN modulation for the text stream: zero (6) normally, continuous (2) on the final block.
    let ctx_chunks = if is_last {
        ADALN_CONT_CHUNKS
    } else {
        ADALN_ZERO_CHUNKS
    } as i64;
    lin(&mut t, "norm1_context.linear", ctx_chunks * h, h);

    // Joint attention — image stream projections.
    lin(&mut t, "attn.to_q", h, h);
    lin(&mut t, "attn.to_k", h, h);
    lin(&mut t, "attn.to_v", h, h);
    lin(&mut t, "attn.to_out.0", h, h);
    // Joint attention — text stream projections.
    lin(&mut t, "attn.add_q_proj", h, h);
    lin(&mut t, "attn.add_k_proj", h, h);
    lin(&mut t, "attn.add_v_proj", h, h);
    if !is_last {
        // The text-stream output projection — absent when the text stream is pre-only.
        lin(&mut t, "attn.to_add_out", h, h);
    }
    // qk-RMSNorm (rms_norm): per-head weight, NO bias.
    t.push(ExpectedTensor::new(
        format!("{p}.attn.norm_q.weight"),
        vec![head],
    ));
    t.push(ExpectedTensor::new(
        format!("{p}.attn.norm_k.weight"),
        vec![head],
    ));
    t.push(ExpectedTensor::new(
        format!("{p}.attn.norm_added_q.weight"),
        vec![head],
    ));
    t.push(ExpectedTensor::new(
        format!("{p}.attn.norm_added_k.weight"),
        vec![head],
    ));

    // Image-stream feed-forward (GELU-approx; net.0.proj gate then net.2 down).
    lin(&mut t, "ff.net.0.proj", FF_MULT as i64 * h, h);
    lin(&mut t, "ff.net.2", h, FF_MULT as i64 * h);
    if !is_last {
        // Text-stream feed-forward — absent when the text stream is pre-only.
        lin(&mut t, "ff_context.net.0.proj", FF_MULT as i64 * h, h);
        lin(&mut t, "ff_context.net.2", h, FF_MULT as i64 * h);
    }

    t
}

/// Map a diffusers `SD3Transformer2DModel` tensor set (`src`) onto the MLX-side key set this crate
/// consumes. The SD3.5 diffusers layout is already the unfused, per-projection convention the E3
/// MMDiT uses, so this is a pure 1:1 rename — currently the identity over the validated key set.
/// Implemented as an explicit pass (rather than `clone()`) so it (a) drops any non-arch tensor a
/// checkpoint might carry and (b) is the single seam to add a remap should the E3 struct layout
/// diverge from a diffusers name (the mlx-gen "keys ≠ struct layout" rule).
///
/// Casting to bf16 mirrors the fork's `mx.eval` on a bf16 state dict; pass the dtype unchanged here
/// and let the caller decide (the dense path keeps bf16; the quant path packs).
pub fn build_target_state_dict(src: &Weights, arch: &Sd3Arch) -> Result<HashMap<String, Array>> {
    let mut out: HashMap<String, Array> = HashMap::new();
    for ExpectedTensor { key, .. } in expected_transformer_tensors(arch) {
        let t = src.require(&key)?;
        out.insert(key, t.clone());
    }
    Ok(out)
}

/// Validate a known tensor set (key → shape) against the expected SD3.5 arch. Reports missing,
/// extra (non-arch), and shape-mismatched keys. `provided` is `(key, shape)` pairs — works equally
/// for a converted in-memory map (shapes from `Array::shape`) and a checkpoint read via the
/// safetensors header alone (no weight body).
pub fn validate_arch<'a, I>(arch: &Sd3Arch, provided: I) -> Result<()>
where
    I: IntoIterator<Item = (&'a str, &'a [i64])>,
{
    let expected: HashMap<String, Vec<i64>> = expected_transformer_tensors(arch)
        .into_iter()
        .map(|e| (e.key, e.shape))
        .collect();
    let provided: HashMap<&str, &[i64]> = provided.into_iter().collect();

    let mut missing: Vec<&String> = expected
        .keys()
        .filter(|k| !provided.contains_key(k.as_str()))
        .collect();
    let mut extra: Vec<&&str> = provided
        .keys()
        .filter(|k| !expected.contains_key(**k))
        .collect();
    let mut bad_shape: Vec<String> = provided
        .iter()
        .filter_map(|(k, shape)| {
            expected.get(*k).and_then(|exp| {
                if shape_matches(exp, shape) {
                    None
                } else {
                    Some(format!("{k} (expected {exp:?}, got {shape:?})"))
                }
            })
        })
        .collect();

    if missing.is_empty() && extra.is_empty() && bad_shape.is_empty() {
        return Ok(());
    }
    missing.sort();
    extra.sort();
    bad_shape.sort();
    Err(Error::Msg(format!(
        "SD3.5 architecture validation FAILED: {} missing, {} extra, {} shape mismatch. \
         expected {} tensors. missing={:?} extra={:?} shape={:?}",
        missing.len(),
        extra.len(),
        bad_shape.len(),
        expected.len(),
        &missing[..missing.len().min(5)],
        &extra[..extra.len().min(5)],
        &bad_shape[..bad_shape.len().min(5)],
    )))
}

/// A dim matches if equal or the expected dim is `-1` (wildcard). Rank must match.
fn shape_matches(expected: &[i64], got: &[i64]) -> bool {
    expected.len() == got.len() && expected.iter().zip(got).all(|(&e, &g)| e == -1 || e == g)
}

/// The total number of transformer tensors the validator expects for a given arch — handy for a
/// quick "did we load the whole checkpoint" count check (the real Large transformer is 1227 tensors
/// per the spike; the VAE / text encoders add the rest of a full snapshot).
pub fn expected_tensor_count(arch: &Sd3Arch) -> usize {
    expected_transformer_tensors(arch).len()
}

/// Read a safetensors file's tensor names + shapes from the JSON header alone (no weights). The
/// format is an 8-byte little-endian header length followed by that many UTF-8 JSON bytes mapping
/// `name → { "shape": [...], … }`. Mirrors `mlx_gen_flux2::convert`'s header reader — never reads
/// the (multi-GB) weight body, so validating a real checkpoint costs a few KB of I/O.
pub fn safetensors_header_shapes(path: &Path) -> Result<HashMap<String, Vec<i64>>> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let file_len = file.metadata()?.len();
    if file_len < 8 {
        return Err(Error::Msg(format!(
            "{}: too small to be a safetensors file",
            path.display()
        )));
    }
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let n = u64::from_le_bytes(len_buf);
    const MAX_HEADER: u64 = 256 << 20; // 256 MiB — far above any real header
    if n > MAX_HEADER || 8 + n > file_len {
        return Err(Error::Msg(format!(
            "{}: safetensors header length out of range",
            path.display()
        )));
    }
    let mut header_bytes = vec![0u8; n as usize];
    file.read_exact(&mut header_bytes)?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes).map_err(|e| {
        Error::Msg(format!(
            "{}: bad safetensors header JSON: {e}",
            path.display()
        ))
    })?;
    let obj = header.as_object().ok_or_else(|| {
        Error::Msg(format!(
            "{}: safetensors header is not an object",
            path.display()
        ))
    })?;
    let mut shapes = HashMap::new();
    for (k, v) in obj {
        if k == "__metadata__" {
            continue;
        }
        let shape = v
            .get("shape")
            .and_then(|s| s.as_array())
            .ok_or_else(|| Error::Msg(format!("{}: tensor {k} has no shape", path.display())))?
            .iter()
            .map(|d| d.as_i64().unwrap_or(-1))
            .collect();
        shapes.insert(k.clone(), shape);
    }
    Ok(shapes)
}

/// Validate a real on-disk SD3.5 `transformer/` directory's tensor set against [`Sd3Arch`] using
/// only the safetensors headers (no weight load). Catches a wrong-repo / wrong-shape / truncated
/// transformer before any multi-GB load.
pub fn validate_transformer_dir(arch: &Sd3Arch, transformer_dir: &Path) -> Result<()> {
    let mut shapes: HashMap<String, Vec<i64>> = HashMap::new();
    let mut shards: Vec<std::path::PathBuf> = std::fs::read_dir(transformer_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    shards.sort();
    if shards.is_empty() {
        return Err(Error::Msg(format!(
            "no transformer safetensors in {}",
            transformer_dir.display()
        )));
    }
    for shard in &shards {
        shapes.extend(safetensors_header_shapes(shard)?);
    }
    let provided: Vec<(&str, &[i64])> = shapes
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_slice()))
        .collect();
    validate_arch(arch, provided.iter().copied())
}

// ----------------------------------------------------------------------------------------------
// Quantization (sc-7860): offline Q4/Q8 packing of the dense bf16 SD3.5 transformer.
// ----------------------------------------------------------------------------------------------

/// The qk-RMSNorm suffixes that are NOT Linears and so stay dense (the four per-block norms). Every
/// other `…​.weight` in the transformer is a bias-carrying Linear the fork's `nn.quantize` packs.
const DENSE_NORM_SUFFIXES: &[&str] = &[".norm_q", ".norm_k", ".norm_added_q", ".norm_added_k"];

/// `true` iff the transformer base key (an `…​.weight` name minus `.weight`) names a quantizable
/// Linear — i.e. it is not one of the qk-RMSNorms and not the learned `pos_embed` table (a
/// non-Linear parameter). The `quantize_map` shape guard is the backstop; this is faithfulness.
fn is_dit_quant_target(base: &str) -> bool {
    if base == "pos_embed.pos_embed" {
        return false;
    }
    !DENSE_NORM_SUFFIXES.iter().any(|s| base.ends_with(s))
}

/// Selectively Q4/Q8-quantize an SD3.5 transformer weight map in place: each matched,
/// group-quantizable `{base}.weight` becomes the packed triple (`weight` u32 codes + `scales` +
/// `biases`) via MLX `quantize` (bf16, group `group_size`), byte-identical to the load-time
/// `AdaptableLinear::quantize`. Norms / `pos_embed` / non-divisible weights pass through dense.
pub fn quantize_sd3_transformer(
    map: HashMap<String, Array>,
    bits: i32,
    group_size: i32,
) -> Result<HashMap<String, Array>> {
    quantize_map(map, bits, group_size, is_dit_quant_target)
}

/// Offline one-shot: read the dense bf16 `src` SD3.5 `transformer/` dir (sharded `*.safetensors` +
/// `config.json`), validate it against [`Sd3Arch`], pre-quantize, and write a `dst` transformer dir
/// — a single packed Q4/Q8 `diffusion_pytorch_model.safetensors` + a `config.json` carrying the
/// `quantization` manifest. `group_size` is the mflux/reference default of 64.
pub fn quantize_sd3_dir(
    arch: &Sd3Arch,
    src: &Path,
    dst: &Path,
    bits: i32,
    group_size: i32,
) -> Result<()> {
    validate_transformer_dir(arch, src)?;
    std::fs::create_dir_all(dst)?;
    let map = load_dir_map(src)?;
    let quantized = quantize_sd3_transformer(map, bits, group_size)?;
    // Materialize before saving (mirrors the fork's explicit `mx.eval`).
    let arrays: Vec<&Array> = quantized.values().collect();
    eval(arrays)?;
    save_map(&dst.join("diffusion_pytorch_model.safetensors"), &quantized)?;
    write_quantized_config(src, dst, bits, group_size)?;
    Ok(())
}

/// Copy `src/config.json` to `dst/config.json` with a `"quantization": {"bits", "group_size"}`
/// block added (the manifest a packed loader reads). A missing source config starts from `{}`.
fn write_quantized_config(src: &Path, dst: &Path, bits: i32, group_size: i32) -> Result<()> {
    let src_cfg = src.join("config.json");
    let mut v: serde_json::Value = if src_cfg.exists() {
        serde_json::from_str(&std::fs::read_to_string(&src_cfg)?)
            .map_err(|e| Error::Msg(format!("sd3: parse {}: {e}", src_cfg.display())))?
    } else {
        serde_json::json!({})
    };
    v["quantization"] = serde_json::json!({ "bits": bits, "group_size": group_size });
    let text = serde_json::to_string_pretty(&v)
        .map_err(|e| Error::Msg(format!("sd3: serialize config.json: {e}")))?;
    std::fs::create_dir_all(dst)?;
    std::fs::write(dst.join("config.json"), text)?;
    Ok(())
}
