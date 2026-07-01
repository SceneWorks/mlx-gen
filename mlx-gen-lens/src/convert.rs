//! Offline pre-quantization (sc-8763): read a dense Lens snapshot and write a packed Q4/Q8 snapshot
//! that [`crate::quant`] (via [`crate::pipeline::LensPipeline::load_quant`]) loads with no dense
//! transient. Mirrors `mlx_gen_sdxl::convert` / `mlx_gen_z_image::convert` (same `mlx_rs::ops::quantize`
//! seam, byte-equal to the load-time `.quantize`), differing in the Lens two-component quant scope and
//! the **encoder's MXFP4→MLX-affine** re-quant.
//!
//! Lens quantizes **two** components at load (the fork's `nn.quantize`, wired in
//! [`crate::pipeline::LensPipeline::load_quant`] + [`quantize_dit`](crate::pipeline::LensPipeline::quantize_dit)):
//!
//! * **DiT** ([`quantize_lens_transformer_dir`]) — the compute-heavy diffusers `[out, in]` Linears
//!   `img_in`, `txt_in`, `proj_out` and every block's fused-QKV attention (`img_qkv`/`txt_qkv`/
//!   `to_out.0`/`to_add_out`) + bias-less SwiGLU MLPs (`img_mlp`/`txt_mlp` `w1`/`w2`/`w3`). The
//!   timestep embedder (`time_text_embed.*`), the AdaLN modulations (`img_mod`/`txt_mod`/
//!   `norm_out.linear`), and every RMSNorm/QK-norm stay full precision — [`is_transformer_target`]
//!   matches that scope exactly (a missed site = codes loaded as dense floats = a garbage render, the
//!   completeness gate being the real-weight render in `tests/prequantize_real_weights.rs`).
//! * **gpt-oss encoder MoE experts** ([`quantize_lens_text_encoder_dir`]) — the 20 B-param bulk. In the
//!   DENSE source they are **MXFP4** (`experts.{gate_up,down}_proj_{blocks,scales}`); here they are
//!   dequantized then re-quantized to MLX group-64 affine Q4/Q8 (reusing the *exact* load path via
//!   [`crate::text_encoder::gpt_oss::prequantize_expert_proj`], so the pack is byte-identical to the
//!   load-time dequant-then-quantize) and stored **stacked** as `experts.{gate_up,down}_proj.{weight,
//!   scales,biases}`. The router / attention / embedding / norms / `lm_head` pass through **dense**
//!   verbatim.
//!
//! The VAE is the shared Flux.2 decoder and is **never quantized** — every tier ships it dense (mirror
//! the source `vae/`). Tokenizer / scheduler / configs / `model_index.json` copy verbatim.
//!
//! Group-B per-crate converter template (sc-8669 / sc-8763). The `bf16` tier is the dense source itself
//! (no conversion — mirror it, deref symlinks).

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;

use mlx_gen::quant::{load_dir_map, quantize_map, save_map};
use mlx_gen::{Error, Result};

use crate::config::GptOssConfig;
use crate::quant::GROUP_SIZE;
use crate::text_encoder::gpt_oss::prequantize_expert_proj;

// ============================================================================================
// DiT (transformer) pack predicate — operates on the **base** (on-disk key minus `.weight`).
// ============================================================================================

/// The DiT dense-Linear leaves that stay **full precision** (matching `LensTransformer::quantize`,
/// which quantizes only `img_in`/`txt_in`/`proj_out` + the block attention/SwiGLU, never these):
/// the timestep embedder, the AdaLN modulations, and `norm_out.linear`. These are 2-D linears the
/// shape guard would otherwise pack, so exclude them explicitly.
fn is_transformer_target(base: &str) -> bool {
    !base.starts_with("time_text_embed.")
        && !base.ends_with(".img_mod.1")
        && !base.ends_with(".txt_mod.1")
        && base != "norm_out.linear"
}

/// Copy `src/config.json` to `dst/config.json` with a `"quantization": {"bits", "group_size"}` block
/// added (HF/diffusers-compat; the Rust loaders auto-detect via `{base}.scales` and ignore it). A
/// missing source config starts from an empty object.
fn write_quantized_config(src: &Path, dst: &Path, bits: i32, group_size: i32) -> Result<()> {
    let src_cfg = src.join("config.json");
    let mut v: serde_json::Value = if src_cfg.exists() {
        serde_json::from_str(&std::fs::read_to_string(&src_cfg)?)
            .map_err(|e| Error::Msg(format!("lens: parse {}: {e}", src_cfg.display())))?
    } else {
        serde_json::json!({})
    };
    v["quantization"] = serde_json::json!({ "bits": bits, "group_size": group_size });
    let text = serde_json::to_string_pretty(&v)
        .map_err(|e| Error::Msg(format!("lens: serialize config.json: {e}")))?;
    std::fs::create_dir_all(dst)?;
    std::fs::write(dst.join("config.json"), text)?;
    Ok(())
}

/// Pre-quantize the DiT `transformer/` dir → a packed `diffusion_pytorch_model.safetensors` +
/// annotated `config.json` in `dst`. Uses the shared [`quantize_map`] (2-D + `in % gs == 0` shape
/// guard skips the 1-D norms) with [`is_transformer_target`] excluding the full-precision linears.
/// `bits` = 4 (Q4) or 8 (Q8) at group size [`GROUP_SIZE`].
pub fn quantize_lens_transformer_dir(src: &Path, dst: &Path, bits: i32) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    let map = quantize_map(load_dir_map(src)?, bits, GROUP_SIZE, is_transformer_target)?;
    save_map(&dst.join("diffusion_pytorch_model.safetensors"), &map)?;
    write_quantized_config(src, dst, bits, GROUP_SIZE)
}

// ============================================================================================
// gpt-oss encoder — MXFP4 experts → stacked MLX-affine packs; everything else passes through.
// ============================================================================================

/// Pre-quantize the gpt-oss `text_encoder/` dir → a packed `model.safetensors` + annotated
/// `config.json` in `dst`. For each layer's MoE, the MXFP4 experts
/// (`experts.{gate_up,down}_proj_{blocks,scales}` + `_bias`) are dequantized then re-quantized to MLX
/// group-64 affine Q4/Q8 via the *exact* load path ([`prequantize_expert_proj`]) and written **stacked**
/// as `experts.{gate_up,down}_proj.{weight,scales,biases}` (dropping the MXFP4 source tensors). Every
/// other tensor — `embed_tokens`, `self_attn.*`, `router.*`, `*_layernorm`, `model.norm`, `lm_head` —
/// passes through **dense** unchanged. `bits` = 4 / 8.
///
/// One layer is packed at a time and the dense bf16 dequant transient is `eval`'d + freed inside
/// [`prequantize_expert_proj`] before the next, so the full 20 B bf16 stack never co-resides (the
/// converter runs in the same memory envelope as the load-time quant path).
pub fn quantize_lens_text_encoder_dir(src: &Path, dst: &Path, bits: i32) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    let cfg = GptOssConfig::lens();
    let src_map = load_dir_map(src)?;

    let mut out: HashMap<String, Array> = HashMap::with_capacity(src_map.len());
    // Pass through every non-MXFP4 tensor verbatim; the MXFP4 expert tensors are consumed per layer
    // below and replaced by the stacked packs (so drop them from the pass-through).
    for (k, v) in &src_map {
        let is_mxfp4 = k.contains(".experts.gate_up_proj_blocks")
            || k.contains(".experts.gate_up_proj_scales")
            || k.contains(".experts.down_proj_blocks")
            || k.contains(".experts.down_proj_scales")
            // The biases are re-emitted (bf16) alongside the packs, so drop the source copies too.
            || k.contains(".experts.gate_up_proj_bias")
            || k.contains(".experts.down_proj_bias");
        if !is_mxfp4 {
            out.insert(k.clone(), v.clone());
        }
    }

    // Pack each layer's two expert projections. `num_layers` is the FULL 24 — the converter packs the
    // whole encoder regardless of the runtime capture-layer prefix (a hosted tier serves any selection).
    let require = |k: &str| -> Result<&Array> {
        src_map.get(k).ok_or_else(|| {
            Error::Msg(format!(
                "lens encoder convert: missing `{k}` in {}",
                src.display()
            ))
        })
    };
    for i in 0..cfg.num_layers {
        let prefix = format!("model.layers.{i}.mlp");
        for name in ["gate_up", "down"] {
            let blocks = require(&format!("{prefix}.experts.{name}_proj_blocks"))?;
            let scales = require(&format!("{prefix}.experts.{name}_proj_scales"))?;
            let bias = require(&format!("{prefix}.experts.{name}_proj_bias"))?;
            let pack = prequantize_expert_proj(blocks, scales, bias, bits, GROUP_SIZE)?;
            let base = format!("{prefix}.experts.{name}_proj");
            out.insert(format!("{base}.weight"), pack.weight);
            out.insert(format!("{base}.scales"), pack.scales);
            out.insert(format!("{base}.biases"), pack.biases);
            out.insert(format!("{base}_bias"), pack.bias);
        }
    }

    save_map(&dst.join("model.safetensors"), &out)?;
    write_quantized_config(src, dst, bits, GROUP_SIZE)
}

// ============================================================================================
// Turnkey assembly.
// ============================================================================================

/// Recursively copy a directory's files, resolving symlinks (HF snapshots symlink into
/// `../../blobs/…`) to real bytes so the assembled tier is self-contained and HF-uploadable.
fn copy_dir(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        let target = dst.join(entry.file_name());
        if path.is_dir() {
            copy_dir(&path, &target)?;
        } else {
            let real = std::fs::canonicalize(&path)?;
            std::fs::copy(&real, &target)?;
        }
    }
    Ok(())
}

/// Assemble a full pre-quantized turnkey Lens snapshot in `dst_root`: pack the DiT + the gpt-oss
/// encoder MoE experts, mirror the dense VAE, and copy the tokenizer / scheduler / `model_index.json`
/// (+ assets / LICENSE / README) verbatim. The result loads via
/// [`crate::pipeline::LensPipeline::load_quant`] (packed weights auto-detect) with no dense transient.
/// `bits` = 4 (Q4 tier) or 8 (Q8 tier). The bf16 tier is the dense source itself (no conversion —
/// mirror it; see the tier builder in `tests/prequantize_real_weights.rs`).
pub fn prequantize_turnkey(src_root: &Path, dst_root: &Path, bits: i32) -> Result<()> {
    std::fs::create_dir_all(dst_root)?;
    quantize_lens_transformer_dir(
        &src_root.join("transformer"),
        &dst_root.join("transformer"),
        bits,
    )?;
    quantize_lens_text_encoder_dir(
        &src_root.join("text_encoder"),
        &dst_root.join("text_encoder"),
        bits,
    )?;
    // VAE stays dense (never quantized) — mirror the source `vae/` verbatim (deref symlinks).
    let vae_src = src_root.join("vae");
    if vae_src.exists() {
        copy_dir(&vae_src, &dst_root.join("vae"))?;
    }
    // Tokenizer + scheduler + (optional) assets are non-weight — copy verbatim.
    for rel in ["tokenizer", "scheduler", "assets"] {
        let s = src_root.join(rel);
        if s.exists() {
            copy_dir(&s, &dst_root.join(rel))?;
        }
    }
    for f in [
        "model_index.json",
        "LICENSE",
        "LICENSE.md",
        "LICENSE.txt",
        "README.md",
    ] {
        let s = src_root.join(f);
        if s.exists() {
            let real = std::fs::canonicalize(&s)?;
            std::fs::copy(&real, dst_root.join(f))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{eq, quantize};
    use mlx_rs::{Array, Dtype};

    #[test]
    fn transformer_predicate_matches_quantize_scope() {
        // Quantized leaves (block attention/SwiGLU + the three top-level linears).
        for base in [
            "img_in",
            "txt_in",
            "proj_out",
            "transformer_blocks.0.attn.img_qkv",
            "transformer_blocks.0.attn.txt_qkv",
            "transformer_blocks.0.attn.to_out.0",
            "transformer_blocks.0.attn.to_add_out",
            "transformer_blocks.0.img_mlp.w1",
            "transformer_blocks.0.img_mlp.w2",
            "transformer_blocks.0.img_mlp.w3",
            "transformer_blocks.0.txt_mlp.w1",
        ] {
            assert!(is_transformer_target(base), "{base} should be packed");
        }
        // Full-precision 2-D linears (must stay dense).
        for base in [
            "time_text_embed.timestep_embedder.linear_1",
            "time_text_embed.timestep_embedder.linear_2",
            "transformer_blocks.0.img_mod.1",
            "transformer_blocks.0.txt_mod.1",
            "norm_out.linear",
        ] {
            assert!(!is_transformer_target(base), "{base} should stay dense");
        }
    }

    fn byte_equal(a: &Array, b: &Array) -> bool {
        a.shape() == b.shape()
            && a.dtype() == b.dtype()
            && eq(a, b).unwrap().all(None).unwrap().item::<bool>()
    }

    /// The DiT pack is byte-identical to the load-time `.quantize` (bf16 cast, group 64) — the
    /// sc-8669 round-trip guarantee. A modulation Linear (predicate-excluded) and a 1-D norm
    /// (shape-guarded) stay dense.
    #[test]
    fn transformer_quantize_map_byte_identical_and_skips_dense() {
        let w = Array::from_slice(
            &(0..64 * 128).map(|i| (i as f32).sin()).collect::<Vec<_>>(),
            &[64, 128],
        );
        let mut map: HashMap<String, Array> = HashMap::new();
        map.insert("transformer_blocks.0.attn.img_qkv.weight".into(), w.clone());
        // A modulation Linear (2-D but predicate-excluded → stays dense f32).
        map.insert(
            "transformer_blocks.0.img_mod.1.weight".into(),
            Array::ones::<f32>(&[96, 128]).unwrap(),
        );
        // A 1-D RMSNorm (shape-guarded → stays dense).
        map.insert(
            "transformer_blocks.0.img_norm1.weight".into(),
            Array::ones::<f32>(&[128]).unwrap(),
        );

        let out = quantize_map(map, 4, GROUP_SIZE, is_transformer_target).unwrap();

        let wq = out
            .get("transformer_blocks.0.attn.img_qkv.weight")
            .expect("packed");
        assert_eq!(wq.dtype(), Dtype::Uint32, "Q4 codes are u32-packed");
        let scales = out.get("transformer_blocks.0.attn.img_qkv.scales").unwrap();
        let biases = out.get("transformer_blocks.0.attn.img_qkv.biases").unwrap();
        let (ewq, esc, ebi) =
            quantize(w.as_dtype(Dtype::Bfloat16).unwrap(), GROUP_SIZE, 4).unwrap();
        assert!(byte_equal(wq, &ewq), "packed weight != load-time quantize");
        assert!(byte_equal(scales, &esc), "scales != load-time quantize");
        assert!(byte_equal(biases, &ebi), "biases != load-time quantize");

        // The modulation Linear stays dense (predicate-excluded).
        let m = out.get("transformer_blocks.0.img_mod.1.weight").unwrap();
        assert_eq!(m.dtype(), Dtype::Float32, "img_mod stays dense");
        assert!(!out.contains_key("transformer_blocks.0.img_mod.1.scales"));
        // The norm stays dense.
        let n = out.get("transformer_blocks.0.img_norm1.weight").unwrap();
        assert_eq!(n.dtype(), Dtype::Float32, "norm stays dense");
    }
}
