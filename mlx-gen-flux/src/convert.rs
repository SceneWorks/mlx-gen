//! Offline pre-quantization: read the dense FLUX.1 snapshot and write a packed Q4/Q8 snapshot that
//! [`crate::quant`] (+ the crate-local `TokenEmbedding` packed-detect + the shared Z-Image VAE) loads
//! with no dense transient. Mirrors `mlx_gen_z_image::convert` / `mlx_gen_qwen_image::convert` (same
//! `mlx_rs::ops::quantize`, byte-equal to the load-time `.quantize` seams), differing only in the
//! FLUX key layout and that it packs **four** components.
//!
//! FLUX.1 quantizes all three model parts (the fork's whole-model `nn.quantize`, wired in
//! [`crate::model::load`]): the DiT transformer, the CLIP + T5 text encoders, and the (shared
//! Z-Image) VAE's mid-block attention. The transformer / CLIP / T5 fork predicates quantize **every**
//! quantizable Linear + embedding, so the shared [`quantize_map`] shape guard (2-D, `in % gs == 0`,
//! `in >= gs`) exactly selects them — the 1-D LayerNorm / RMSNorm scales pass through dense. The VAE
//! packs only its attention projections (convs are 4-D, GroupNorms 1-D — both shape-guarded dense),
//! matching the Z-Image VAE quant scope; the loader's prefix-drop + conv-transpose remap leaves the
//! packed attn `{weight,scales,biases}` untouched. The result loads via [`crate::model::load`] with
//! no dense bf16 transient (sc-8670). Group-B per-crate converter template (sc-8669).

use std::path::Path;

use mlx_gen::quant::{load_dir_map, quantize_map, save_map};
use mlx_gen::{Error, Result};

use crate::quant::GROUP_SIZE;

// ============================================================================================
// Pack predicates (operate on the **base** = the on-disk key minus its `.weight`).
// ============================================================================================

/// Pack every quantizable leaf, leaning on the shared [`quantize_map`] shape guard to skip the 1-D
/// norms and 4-D convs. The FLUX transformer / CLIP / T5 fork predicates quantize **every** Linear +
/// token/position/relative-bias embedding (there is no quantizable-but-skipped 2-D leaf in these
/// components), so "pack all shape-eligible tensors" is byte-identical to the load-time `.quantize`.
fn pack_all(_base: &str) -> bool {
    true
}

/// VAE: pack only the mid-block attention projections — the sole quantizable leaves in the otherwise
/// all-conv (shared Z-Image) VAE. Matches `mlx_gen_z_image::convert`'s VAE predicate; operates on the
/// raw diffusers keys (`decoder.*` / `encoder.*`), which the loader's remap leaves packed.
fn is_vae_target(base: &str) -> bool {
    base.ends_with(".to_q")
        || base.ends_with(".to_k")
        || base.ends_with(".to_v")
        || base.ends_with(".to_out.0")
}

/// Copy `src/config.json` to `dst/config.json` with a `"quantization": {"bits", "group_size"}` block
/// added (HF/diffusers-compat; the Rust loaders auto-detect via `{base}.scales` and ignore it). A
/// missing source config starts from an empty object.
fn write_quantized_config(src: &Path, dst: &Path, bits: i32, group_size: i32) -> Result<()> {
    let src_cfg = src.join("config.json");
    let mut v: serde_json::Value = if src_cfg.exists() {
        serde_json::from_str(&std::fs::read_to_string(&src_cfg)?)
            .map_err(|e| Error::Msg(format!("flux: parse {}: {e}", src_cfg.display())))?
    } else {
        serde_json::json!({})
    };
    v["quantization"] = serde_json::json!({ "bits": bits, "group_size": group_size });
    let text = serde_json::to_string_pretty(&v)
        .map_err(|e| Error::Msg(format!("flux: serialize config.json: {e}")))?;
    std::fs::create_dir_all(dst)?;
    std::fs::write(dst.join("config.json"), text)?;
    Ok(())
}

/// Pre-quantize one component dir (`src` sharded/single `*.safetensors` + `config.json`) → a packed
/// `model.safetensors` + annotated `config.json` in `dst`. `is_target` is the component's pack
/// predicate; `bits` = 4 (Q4) or 8 (Q8) at group size [`GROUP_SIZE`].
fn quantize_component(
    src: &Path,
    dst: &Path,
    bits: i32,
    is_target: fn(&str) -> bool,
) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    let map = quantize_map(load_dir_map(src)?, bits, GROUP_SIZE, is_target)?;
    save_map(&dst.join("model.safetensors"), &map)?;
    write_quantized_config(src, dst, bits, GROUP_SIZE)
}

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

/// Assemble a full pre-quantized turnkey snapshot in `dst_root`: pack the transformer, CLIP text
/// encoder (`text_encoder/`), T5 text encoder (`text_encoder_2/`), and VAE, and copy the dense
/// tokenizers / scheduler / `model_index.json` verbatim. The result loads via [`crate::model::load`]
/// (packed weights auto-detect) with no dense transient. `bits` = 4 (Q4 tier) or 8 (Q8 tier). The
/// bf16 tier is the dense source itself (no conversion — mirror it).
pub fn prequantize_turnkey(src_root: &Path, dst_root: &Path, bits: i32) -> Result<()> {
    std::fs::create_dir_all(dst_root)?;
    quantize_component(
        &src_root.join("transformer"),
        &dst_root.join("transformer"),
        bits,
        pack_all,
    )?;
    quantize_component(
        &src_root.join("text_encoder"),
        &dst_root.join("text_encoder"),
        bits,
        pack_all,
    )?;
    quantize_component(
        &src_root.join("text_encoder_2"),
        &dst_root.join("text_encoder_2"),
        bits,
        pack_all,
    )?;
    quantize_component(
        &src_root.join("vae"),
        &dst_root.join("vae"),
        bits,
        is_vae_target,
    )?;
    for rel in ["tokenizer", "tokenizer_2", "scheduler"] {
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
    use std::collections::HashMap;

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
            "decoder.conv_in",
        ] {
            assert!(!is_vae_target(base), "{base} should stay dense");
        }
    }

    fn byte_equal(a: &Array, b: &Array) -> bool {
        a.shape() == b.shape()
            && a.dtype() == b.dtype()
            && eq(a, b).unwrap().all(None).unwrap().item::<bool>()
    }

    /// The packed triple a Linear becomes is byte-identical to the op the load-time `.quantize` runs
    /// (bf16 cast, group 64) — the sc-8670 round-trip guarantee: pre-quantize-on-disk ==
    /// quantize-at-load. A 1-D norm stays dense (shape-guarded out).
    #[test]
    fn quantize_map_packs_targets_byte_identical_to_load_time_quantize() {
        let w = Array::from_slice(
            &(0..64 * 128).map(|i| (i as f32).sin()).collect::<Vec<_>>(),
            &[64, 128],
        );
        let mut map: HashMap<String, Array> = HashMap::new();
        map.insert("transformer_blocks.0.attn.to_q.weight".into(), w.clone());
        map.insert(
            "transformer_blocks.0.norm1.norm.weight".into(),
            Array::ones::<f32>(&[128]).unwrap(),
        );

        let out = quantize_map(map, 4, GROUP_SIZE, pack_all).unwrap();

        let wq = out
            .get("transformer_blocks.0.attn.to_q.weight")
            .expect("packed");
        assert_eq!(wq.dtype(), Dtype::Uint32, "Q4 codes are u32-packed");
        let scales = out.get("transformer_blocks.0.attn.to_q.scales").unwrap();
        let biases = out.get("transformer_blocks.0.attn.to_q.biases").unwrap();
        let (ewq, esc, ebi) =
            quantize(w.as_dtype(Dtype::Bfloat16).unwrap(), GROUP_SIZE, 4).unwrap();
        assert!(byte_equal(wq, &ewq), "packed weight != load-time quantize");
        assert!(byte_equal(scales, &esc), "scales != load-time quantize");
        assert!(byte_equal(biases, &ebi), "biases != load-time quantize");

        // The 1-D norm is shape-guarded out (stays dense f32, no packed triple).
        let n = out.get("transformer_blocks.0.norm1.norm.weight").unwrap();
        assert_eq!(n.dtype(), Dtype::Float32, "norm unchanged");
        assert!(!out.contains_key("transformer_blocks.0.norm1.norm.scales"));
    }
}
