//! Offline pre-quantization: read the dense converted snapshot and write a packed Q4/Q8 snapshot
//! that [`crate::quant`] loads with no dense transient. Mirrors `mlx_gen_flux2::convert` /
//! `mlx_gen_ideogram::convert` (same `mlx_rs::ops::quantize`, byte-equal to the load-time
//! `AdaptableLinear::quantize` / `nn.quantize(bf16)`), differing only in the Z-Image key layout and
//! the three target predicates below.
//!
//! Z-Image quantizes **all three** components (the fork's whole-model `nn.quantize`, wired in
//! [`crate::model::load`]): the DiT transformer, the Qwen3 text encoder, and the VAE's mid-block
//! attention. Each component dir is packed independently; the tokenizer / scheduler / configs pass
//! through dense. The result loads via the standard [`crate::model::load`] path — the packed weights
//! auto-detect ([`crate::quant`]) with no dense bf16 transient (sc-8670). This is the Group-B
//! per-crate converter template (sc-8669).

use std::path::Path;

use mlx_gen::quant::{load_dir_map, quantize_map, save_map};
use mlx_gen::{Error, Result};

use crate::quant::GROUP_SIZE;

// ============================================================================================
// Per-component pack predicates (operate on the **base** = the on-disk key minus its `.weight`).
// The shared [`quantize_map`] shape guard (2-D, `in % group_size == 0`, `in >= group_size`) is the
// backstop, so these are faithfulness + documentation, not the only safety net.
// ============================================================================================

/// DiT dense-passthrough suffixes — the RMSNorm / LayerNorm scales (all 1-D). Everything else
/// `…​.weight` in the transformer is a `nn.Linear` the fork's `nn.quantize` packs: the image/caption
/// embedders, the timestep + final layers, and every block / context-block attention QKV/out, SwiGLU
/// FFN, and adaLN projection. The pad tokens are not `.weight` keys and never reach the predicate.
const DIT_DENSE_NORM_SUFFIXES: &[&str] = &[
    ".attention_norm1",
    ".attention_norm2",
    ".ffn_norm1",
    ".ffn_norm2",
    ".norm_q",
    ".norm_k",
];

/// `true` iff a DiT base names a quantizable Linear — i.e. it is neither a [`DIT_DENSE_NORM_SUFFIXES`]
/// norm nor the caption RMSNorm `cap_embedder.0` (1-D; also shape-guarded).
fn is_dit_target(base: &str) -> bool {
    base != "cap_embedder.0" && !DIT_DENSE_NORM_SUFFIXES.iter().any(|s| base.ends_with(s))
}

/// `true` iff a text-encoder base names a quantizable tensor: a GQA / SwiGLU `*_proj` Linear or the
/// `embed_tokens` table. The `q_norm`/`k_norm` per-head RMSNorms, the `input_layernorm` /
/// `post_attention_layernorm` block norms, and the unused final `model.norm` all pass through dense.
fn is_te_target(base: &str) -> bool {
    base.ends_with(".embed_tokens") || base.ends_with("_proj")
}

/// `true` iff a VAE base names one of the mid-block attention projections — the only quantizable
/// leaves in the otherwise-conv VAE. The convs (4-D) and GroupNorm scales (1-D) pass through dense
/// (shape-guarded). Matches both the `decoder.` and `encoder.` mid blocks (raw diffusers keys).
fn is_vae_target(base: &str) -> bool {
    base.ends_with(".to_q")
        || base.ends_with(".to_k")
        || base.ends_with(".to_v")
        || base.ends_with(".to_out.0")
}

// ============================================================================================
// Per-component dir converters.
// ============================================================================================

/// Copy `src/config.json` to `dst/config.json` with a `"quantization": {"bits", "group_size"}`
/// block added (HF/diffusers-compat; the Rust loaders auto-detect via `{base}.scales` and ignore
/// it). A missing source config starts from an empty object.
fn write_quantized_config(src: &Path, dst: &Path, bits: i32, group_size: i32) -> Result<()> {
    let src_cfg = src.join("config.json");
    let mut v: serde_json::Value = if src_cfg.exists() {
        serde_json::from_str(&std::fs::read_to_string(&src_cfg)?)
            .map_err(|e| Error::Msg(format!("z-image: parse {}: {e}", src_cfg.display())))?
    } else {
        serde_json::json!({})
    };
    v["quantization"] = serde_json::json!({ "bits": bits, "group_size": group_size });
    let text = serde_json::to_string_pretty(&v)
        .map_err(|e| Error::Msg(format!("z-image: serialize config.json: {e}")))?;
    std::fs::create_dir_all(dst)?;
    std::fs::write(dst.join("config.json"), text)?;
    Ok(())
}

/// Pre-quantize the DiT `transformer` dir (sharded `*.safetensors` + `config.json`) → a packed
/// `model.safetensors` + annotated `config.json` in `dst`. `bits` = 4 (Q4) or 8 (Q8); group size is
/// the codebase default 64. Packs every Linear, leaves the RMSNorms / pad tokens dense.
pub fn quantize_z_image_transformer(src: &Path, dst: &Path, bits: i32) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    let map = quantize_map(load_dir_map(src)?, bits, GROUP_SIZE, is_dit_target)?;
    save_map(&dst.join("model.safetensors"), &map)?;
    write_quantized_config(src, dst, bits, GROUP_SIZE)
}

/// Pre-quantize the `text_encoder` dir → a packed `model.safetensors` + annotated `config.json` in
/// `dst`. Packs the GQA/SwiGLU `*_proj` Linears and the token embedding; the norms pass through
/// dense. The `embed_tokens` table is bf16-native, so the converter's bf16-cast pack is byte-equal
/// to the load-time `TokenEmbedding::quantize(bits, cast_to_bf16=false)`.
pub fn quantize_z_image_text_encoder(src: &Path, dst: &Path, bits: i32) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    let map = quantize_map(load_dir_map(src)?, bits, GROUP_SIZE, is_te_target)?;
    save_map(&dst.join("model.safetensors"), &map)?;
    write_quantized_config(src, dst, bits, GROUP_SIZE)
}

/// Pre-quantize the `vae` dir → a packed `model.safetensors` + annotated `config.json` in `dst`.
/// Packs only the mid-block attention projections (the convs are 4-D, the GroupNorms 1-D — both
/// shape-guarded dense). Operates on the **raw diffusers** keys (`decoder.*` / `encoder.*`); the
/// loader's NCHW→NHWC conv transpose + name remap leaves the packed `{base}.{weight,scales,biases}`
/// untouched (they are not conv weights), so the packed attention loads correctly.
pub fn quantize_z_image_vae(src: &Path, dst: &Path, bits: i32) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    let map = quantize_map(load_dir_map(src)?, bits, GROUP_SIZE, is_vae_target)?;
    save_map(&dst.join("model.safetensors"), &map)?;
    write_quantized_config(src, dst, bits, GROUP_SIZE)
}

/// Recursively copy a directory's files (one level of nesting is enough for the tokenizer/scheduler
/// dirs).
fn copy_dir(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        let target = dst.join(entry.file_name());
        if path.is_dir() {
            copy_dir(&path, &target)?;
        } else {
            // Resolve symlinks (HF snapshots symlink into `../../blobs/…`) to real bytes so the
            // assembled tier is self-contained.
            let real = std::fs::canonicalize(&path)?;
            std::fs::copy(&real, &target)?;
        }
    }
    Ok(())
}

/// Assemble a full pre-quantized turnkey snapshot in `dst_root`: pack the transformer, text encoder,
/// and VAE, and copy the dense tokenizer / scheduler / `model_index.json` verbatim. The result loads
/// via [`crate::model::load`] (the packed weights auto-detect) with no dense transient. Pass a `bits`
/// of 4 for the Q4 tier or 8 for the Q8 tier (sc-8670 / epic 8506).
pub fn prequantize_turnkey(src_root: &Path, dst_root: &Path, bits: i32) -> Result<()> {
    std::fs::create_dir_all(dst_root)?;
    quantize_z_image_transformer(
        &src_root.join("transformer"),
        &dst_root.join("transformer"),
        bits,
    )?;
    quantize_z_image_text_encoder(
        &src_root.join("text_encoder"),
        &dst_root.join("text_encoder"),
        bits,
    )?;
    quantize_z_image_vae(&src_root.join("vae"), &dst_root.join("vae"), bits)?;
    // Dense passthrough — small relative to the three packed components; the loaders read them as-is.
    for rel in ["tokenizer", "scheduler"] {
        let s = src_root.join(rel);
        if s.exists() {
            copy_dir(&s, &dst_root.join(rel))?;
        }
    }
    for f in ["model_index.json", "LICENSE.md", "LICENSE", "README.md"] {
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
    use std::collections::HashMap;

    #[test]
    fn dit_predicate_packs_every_linear_not_the_norms() {
        for base in [
            "all_x_embedder.2-1",
            "cap_embedder.1",
            "t_embedder.mlp.0",
            "t_embedder.mlp.2",
            "all_final_layer.2-1.linear",
            "all_final_layer.2-1.adaLN_modulation.1",
            "layers.0.attention.to_q",
            "layers.29.attention.to_out.0",
            "noise_refiner.0.feed_forward.w2",
            "context_refiner.1.attention.to_v",
            "layers.0.adaLN_modulation.0",
            "control_layers.0.after_proj",
            "control_layers.0.before_proj",
        ] {
            assert!(is_dit_target(base), "{base} should be packed");
        }
        for base in [
            "cap_embedder.0",
            "layers.0.attention.norm_q",
            "layers.0.attention.norm_k",
            "layers.0.attention_norm1",
            "layers.0.attention_norm2",
            "layers.0.ffn_norm1",
            "layers.0.ffn_norm2",
        ] {
            assert!(!is_dit_target(base), "{base} should stay dense");
        }
    }

    #[test]
    fn te_predicate_packs_projections_and_embedding_only() {
        for base in [
            "model.embed_tokens",
            "model.layers.0.self_attn.q_proj",
            "model.layers.35.self_attn.o_proj",
            "model.layers.7.mlp.gate_proj",
            "model.layers.7.mlp.down_proj",
        ] {
            assert!(is_te_target(base), "{base} should be packed");
        }
        for base in [
            "model.layers.0.self_attn.q_norm",
            "model.layers.0.self_attn.k_norm",
            "model.layers.0.input_layernorm",
            "model.layers.0.post_attention_layernorm",
            "model.norm",
        ] {
            assert!(!is_te_target(base), "{base} should stay dense");
        }
    }

    #[test]
    fn vae_predicate_packs_attention_projections_only() {
        for base in [
            "decoder.mid_block.attentions.0.to_q",
            "decoder.mid_block.attentions.0.to_k",
            "decoder.mid_block.attentions.0.to_v",
            "decoder.mid_block.attentions.0.to_out.0",
            "encoder.mid_block.attentions.0.to_q",
        ] {
            assert!(is_vae_target(base), "{base} should be packed");
        }
        for base in [
            "decoder.mid_block.attentions.0.group_norm",
            "decoder.mid_block.resnets.0.conv1",
            "decoder.conv_in.conv",
            "decoder.up_blocks.0.upsamplers.0.conv",
        ] {
            assert!(!is_vae_target(base), "{base} should stay dense");
        }
    }

    fn byte_equal(a: &Array, b: &Array) -> bool {
        a.shape() == b.shape()
            && a.dtype() == b.dtype()
            && eq(a, b).unwrap().all(None).unwrap().item::<bool>()
    }

    /// The packed triple a DiT Linear becomes is byte-identical to the op the load-time
    /// `AdaptableLinear::quantize` runs (bf16 cast, group 64) — the sc-8670 round-trip guarantee:
    /// pre-quantize-on-disk == quantize-at-load. The 1-D qk-norm stays dense (predicate-excluded
    /// *and* shape-guarded).
    #[test]
    fn quantize_map_packs_targets_byte_identical_to_load_time_quantize() {
        let w = Array::from_slice(
            &(0..64 * 128).map(|i| (i as f32).sin()).collect::<Vec<_>>(),
            &[64, 128],
        );
        let mut map: HashMap<String, Array> = HashMap::new();
        map.insert("layers.0.attention.to_q.weight".into(), w.clone());
        map.insert(
            "layers.0.attention.norm_q.weight".into(),
            Array::ones::<f32>(&[128]).unwrap(),
        );

        let out = quantize_map(map, 4, GROUP_SIZE, is_dit_target).unwrap();

        let wq = out.get("layers.0.attention.to_q.weight").expect("packed");
        assert_eq!(wq.dtype(), Dtype::Uint32, "Q4 codes are u32-packed");
        let scales = out.get("layers.0.attention.to_q.scales").unwrap();
        let biases = out.get("layers.0.attention.to_q.biases").unwrap();
        let (ewq, esc, ebi) =
            quantize(w.as_dtype(Dtype::Bfloat16).unwrap(), GROUP_SIZE, 4).unwrap();
        assert!(byte_equal(wq, &ewq), "packed weight != load-time quantize");
        assert!(byte_equal(scales, &esc), "scales != load-time quantize");
        assert!(byte_equal(biases, &ebi), "biases != load-time quantize");

        let n = out
            .get("layers.0.attention.norm_q.weight")
            .expect("dense norm");
        assert_eq!(n.dtype(), Dtype::Float32, "norm unchanged");
        assert!(!out.contains_key("layers.0.attention.norm_q.scales"));
    }
}
