//! Offline pre-quantization: read the dense Qwen-Image snapshot and write a packed Q4/Q8 snapshot
//! that [`crate::quant`] loads with no dense transient. Mirrors `mlx_gen_z_image::convert` /
//! `mlx_gen_flux2::convert` (same `mlx_rs::ops::quantize`, byte-equal to the load-time
//! `AdaptableLinear::quantize`), differing only in the Qwen-Image key layout and that **only the
//! transformer is packed**.
//!
//! Qwen-Image quantizes the **transformer only** (the fork's `nn.quantize(transformer, bits)`,
//! wired in [`crate::model::load`]): the Qwen2.5-VL text encoder is `skip_quantization`
//! ("Quantization causes significant semantic degradation") and the VAE is all-conv (no quantizable
//! leaves). So the converter packs `transformer/` and copies `text_encoder/`, `vae/`, `tokenizer/`,
//! `processor/`, `scheduler/`, and the configs through **dense** (bf16). The result loads via the
//! standard [`crate::model::load`] path — the packed transformer auto-detects ([`crate::quant`])
//! with no dense bf16 transient (sc-8670). This is the Group-B per-crate converter template
//! (sc-8669); the dense-TE shape matches FLUX.2-klein (sc-8711).

use std::collections::HashMap;
use std::path::Path;

use mlx_gen::quant::{
    copy_dir, copy_turnkey_assets, load_dir_map, quantize_map, save_map, write_quantized_config,
};
use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_rs::Array;

use crate::quant::GROUP_SIZE;

// ============================================================================================
// Transformer pack predicate (operates on the **raw on-disk diffusers base** = key minus `.weight`,
// before the loader's [`crate::loader::remap_transformer_keys`] rename — the converter reads raw).
// The shared [`quantize_map`] shape guard (2-D, `in % group_size == 0`, `in >= group_size`) is the
// backstop, so this is faithfulness + documentation, not the only safety net.
// ============================================================================================

/// Transformer dense-passthrough suffixes — the per-head q/k RMSNorm scales (all 1-D). Everything
/// else `…​.weight` in the transformer is a `nn.Linear` the fork's `nn.quantize(transformer)` packs:
/// the image/text embedders (`img_in`/`txt_in`), the timestep MLP, every block's adaLN modulation
/// (`{img,txt}_mod`), joint-attention QKV/out projections, gated FFN, and the final `proj_out` /
/// `norm_out.linear`.
const DENSE_NORM_SUFFIXES: &[&str] = &[".norm_q", ".norm_k", ".norm_added_q", ".norm_added_k"];

/// `true` iff a transformer base names a quantizable Linear — i.e. it is neither a
/// [`DENSE_NORM_SUFFIXES`] attention norm nor the top-level text RMSNorm `txt_norm` (1-D; also
/// shape-guarded).
fn is_transformer_target(base: &str) -> bool {
    base != "txt_norm" && !DENSE_NORM_SUFFIXES.iter().any(|s| base.ends_with(s))
}

// ============================================================================================
// Converter.
// ============================================================================================

/// Pre-quantize the MMDiT `transformer` dir (sharded `*.safetensors` + index + `config.json`) → a
/// packed `model.safetensors` + annotated `config.json` in `dst`. `bits` = 4 (Q4) or 8 (Q8); group
/// size is the codebase default 64. Packs every Linear, leaves the q/k RMSNorms + `txt_norm` dense.
pub fn quantize_qwen_image_transformer(src: &Path, dst: &Path, bits: i32) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    let map = quantize_map(load_dir_map(src)?, bits, GROUP_SIZE, is_transformer_target)?;
    save_map(&dst.join("model.safetensors"), &map)?;
    write_quantized_config(src, dst, bits, GROUP_SIZE)
}

/// Assemble a full pre-quantized turnkey snapshot in `dst_root`: pack the transformer and copy the
/// **dense** text encoder, VAE, tokenizer, processor, scheduler, and `model_index.json` verbatim.
/// The result loads via [`crate::model::load`] (the packed transformer auto-detects, the dense TE +
/// VAE load as-is) with no dense transformer transient. Pass `bits` of 4 for the Q4 tier or 8 for
/// the Q8 tier (sc-8670 / epic 8506). The bf16 tier is the dense source itself (no conversion — just
/// mirror it).
pub fn prequantize_turnkey(src_root: &Path, dst_root: &Path, bits: i32) -> Result<()> {
    std::fs::create_dir_all(dst_root)?;
    quantize_qwen_image_transformer(
        &src_root.join("transformer"),
        &dst_root.join("transformer"),
        bits,
    )?;
    // Dense passthrough — the TE (skip_quantization) + VAE (all-conv) + the tokenizer/processor/
    // scheduler trees load as-is. Sizeable (the Qwen2.5-VL TE dominates), but unquantized by design.
    for rel in ["text_encoder", "vae", "tokenizer", "processor", "scheduler"] {
        let s = src_root.join(rel);
        if s.exists() {
            copy_dir(&s, &dst_root.join(rel))?;
        }
    }
    copy_turnkey_assets(src_root, dst_root)
}

// ============================================================================================
// 2512-Fun-Controlnet-Union control-branch converter (sc-9517 / epic 9083).
// ============================================================================================

/// Pre-quantize the alibaba-pai **2512-Fun-Controlnet-Union** control checkpoint — a single
/// `Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors` overlay (the canonical MLX overlay), or a
/// dir of shards — into a packed `model.safetensors` + a provenance `config.json` in `dst`. Mirrors
/// [`QwenFunControlBranch::quantize`](crate::control_transformer::QwenFunControlBranch::quantize)'s
/// scope exactly: every control-block joint-attention / gated-FFN / adaLN Linear plus each block's
/// `after_proj` and block-0's `before_proj` pack group-64; the `control_img_in` patch embedder (132
/// in-features → `% 64 != 0`) and the 1-D per-head attn RMSNorms stay **dense** — both shape-guarded
/// by [`quantize_map`], so reusing the base [`is_transformer_target`] predicate (the control blocks
/// share the base block key layout 1:1) is faithfulness + documentation, not the only net.
///
/// The packed keys keep their raw diffusers spelling; [`crate::loader::load_controlnet`] applies the
/// same `remap_transformer_keys` to the packed tier as to the dense checkpoint and its projections
/// packed-detect via `linear_from`, so the tier loads with no dense transient (and
/// `QwenImageControl`'s load-time `.quantize()` no-ops on it — see [`crate::model_control`]). The
/// candle lane consumes the SAME hosted tier via `candle_gen::quant` (epic 9083). `bits` = 4 or 8.
pub fn quantize_qwen_control_branch(src: &Path, dst: &Path, bits: i32) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    // The overlay ships as a single `.safetensors` (canonical) or a dir of shards; read either into a
    // key→Array map (the shared `load_dir_map` only handles dirs, so a file goes via `from_file`).
    let w = if src.is_dir() {
        Weights::from_dir(src)?
    } else {
        Weights::from_file(src)?
    };
    let map: HashMap<String, Array> = w
        .keys()
        .map(|k| (k.to_string(), w.get(k).expect("listed key").clone()))
        .collect();
    let packed = quantize_map(map, bits, GROUP_SIZE, is_transformer_target)?;
    save_map(&dst.join("model.safetensors"), &packed)?;
    // Provenance: a `{"quantization": {...}}` config.json (the loaders auto-detect packed weights via
    // `{base}.scales` and ignore this block) + the source repo's README/LICENSE, so the hosted tier
    // is self-describing + licensed (F-045). The overlay ships no config.json → the block starts from
    // an empty object.
    let src_root = if src.is_dir() {
        src
    } else {
        src.parent().unwrap_or(src)
    };
    write_quantized_config(src_root, dst, bits, GROUP_SIZE)?;
    copy_turnkey_assets(src_root, dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{eq, quantize};
    use mlx_rs::Dtype;

    #[test]
    fn transformer_predicate_packs_every_linear_not_the_norms() {
        for base in [
            "img_in",
            "txt_in",
            "proj_out",
            "norm_out.linear",
            "time_text_embed.timestep_embedder.linear_1",
            "time_text_embed.timestep_embedder.linear_2",
            "transformer_blocks.0.attn.to_q",
            "transformer_blocks.59.attn.to_out.0",
            "transformer_blocks.0.attn.add_k_proj",
            "transformer_blocks.0.attn.to_add_out",
            "transformer_blocks.0.img_mod.1",
            "transformer_blocks.0.txt_mod.1",
            "transformer_blocks.7.img_mlp.net.0.proj",
            "transformer_blocks.7.img_mlp.net.2",
            "transformer_blocks.7.txt_mlp.net.0.proj",
            "transformer_blocks.7.txt_mlp.net.2",
        ] {
            assert!(is_transformer_target(base), "{base} should be packed");
        }
        for base in [
            "txt_norm",
            "transformer_blocks.0.attn.norm_q",
            "transformer_blocks.0.attn.norm_k",
            "transformer_blocks.0.attn.norm_added_q",
            "transformer_blocks.0.attn.norm_added_k",
        ] {
            assert!(!is_transformer_target(base), "{base} should stay dense");
        }
    }

    fn byte_equal(a: &Array, b: &Array) -> bool {
        a.shape() == b.shape()
            && a.dtype() == b.dtype()
            && eq(a, b).unwrap().all(None).unwrap().item::<bool>()
    }

    /// The packed triple a transformer Linear becomes is byte-identical to the op the load-time
    /// `AdaptableLinear::quantize` runs (bf16 cast, group 64) — the sc-8670 round-trip guarantee:
    /// pre-quantize-on-disk == quantize-at-load. The 1-D q-norm stays dense (predicate-excluded
    /// *and* shape-guarded).
    #[test]
    fn quantize_map_packs_targets_byte_identical_to_load_time_quantize() {
        let w = Array::from_slice(
            &(0..64 * 128).map(|i| (i as f32).sin()).collect::<Vec<_>>(),
            &[64, 128],
        );
        let mut map: HashMap<String, Array> = HashMap::new();
        map.insert("transformer_blocks.0.attn.to_q.weight".into(), w.clone());
        map.insert(
            "transformer_blocks.0.attn.norm_q.weight".into(),
            Array::ones::<f32>(&[128]).unwrap(),
        );

        let out = quantize_map(map, 4, GROUP_SIZE, is_transformer_target).unwrap();

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

        let n = out
            .get("transformer_blocks.0.attn.norm_q.weight")
            .expect("dense norm");
        assert_eq!(n.dtype(), Dtype::Float32, "norm unchanged");
        assert!(!out.contains_key("transformer_blocks.0.attn.norm_q.scales"));
    }
}
