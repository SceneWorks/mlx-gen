//! Offline pre-quantization: read a dense SenseNova-U1 snapshot and write a packed Q4/Q8 turnkey that
//! [`crate::quant`] (via [`crate::model::load`]) loads with no dense bf16/f32 transient. Mirrors
//! `mlx_gen_sdxl::convert` / `mlx_gen_lens::convert` (same `mlx_gen::quant::quantize_map`, byte-equal
//! to the load-time `.quantize` seam), differing in SenseNova's **flat** checkpoint layout and its
//! backbone-only quant scope.
//!
//! SenseNova-U1 is a **unified** MoT model — no separate VAE or text encoder, and one **flat** sharded
//! safetensors set (`model-0000N-of-00008.safetensors`), not diffusers component sub-dirs. What
//! [`crate::model::load`] quantizes ([`crate::t2i::T2iModel::quantize`] →
//! [`crate::qwen3::Qwen3Backbone::quantize`]) is exactly the **decoder-stack Linears**: every layer's
//! four attention projections (`{q,k,v,o}_proj`) + three SwiGLU Linears (`gate/up/down_proj`), on
//! **both** the understanding (`""`) and generation (`_mot_gen`) paths. Everything else stays dense —
//! the token embedding + `lm_head` (2-D but gather/output matmuls the backbone keeps dense), all
//! RMSNorms + QK-norms (1-D, shape-guarded out anyway), the two Conv vision embedders (4-D, shape-
//! guarded out), and the flow-matching `fm_head` + timestep/noise-scale embedder Linears (2-D, kept
//! dense — precision-sensitive flow head). [`is_backbone_linear`] names the decoder-stack projections
//! exactly, so a 2-D dense target (`embed_tokens` / `lm_head` / `fm_head`) is never packed.
//!
//! "MoT" is Mixture of **Transformers** (two *dense* parallel stacks with distinct `_mot_gen`-suffixed
//! keys), **not** Mixture of Experts — there are no stacked `[E, …]` expert tensors and no fused proj
//! tensors, so each target packs as the plain per-Linear triple with no slicing.
//!
//! Group-B per-crate converter template (sc-8669 / sc-8771). The completeness gate is the real-weight
//! render in `tests/prequantize_real_weights.rs`: a missed pack site (or a wrongly-packed dense
//! tensor) loads u32 codes as dense floats → a garbage render.

use std::path::Path;

use mlx_gen::quant::{load_dir_map, quantize_map, save_map};
use mlx_gen::{Error, Result};

use crate::quant::GROUP_SIZE;

/// The single packed weight file the turnkey ships (replaces the source's 8 dense shards). The
/// loader globs `*.safetensors`, so one flat file suffices.
const PACKED_WEIGHTS_FILE: &str = "model.safetensors";

/// Non-weight assets copied verbatim into the turnkey so it is a self-contained, HF-uploadable load
/// root: the config the loader parses, the materialized fast tokenizer + its source vocab, the chat
/// template, and the license/readme. (`model.safetensors.index.json` is intentionally NOT copied —
/// it indexes the source's 8 shards, which the single packed file replaces; the loader globs
/// `*.safetensors` and never reads the index.)
const ASSET_FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "README.md",
    "README_CN.md",
];

/// Whether `base` (a checkpoint key minus its `.weight`) is a **decoder-stack Linear** the backbone
/// quantizes — a `language_model.model.layers.{i}` attention projection (`{q,k,v,o}_proj`, optionally
/// `_mot_gen`) or a SwiGLU Linear (`{gate,up,down}_proj` under `mlp` / `mlp_mot_gen`). Excludes every
/// other 2-D `.weight` (`embed_tokens`, `lm_head`, the `fm_head`/timestep-embedder Linears) so they
/// stay dense, matching [`crate::qwen3::Qwen3Backbone::quantize`] exactly. The suffix quirk (the
/// `_mot_gen` marker attaches to the *proj* segment for attention but the *mlp* segment for the MLP)
/// is handled by matching the un-suffixed proj name at the end.
fn is_backbone_linear(base: &str) -> bool {
    // Only the language-model decoder stack; skip vision / fm_modules / top-level heads.
    let Some(rest) = base.strip_prefix("language_model.model.layers.") else {
        return false;
    };
    // `rest` = `{i}.<...>` — require an attention or MLP projection tail.
    let Some((_layer, tail)) = rest.split_once('.') else {
        return false;
    };
    // Attention: `self_attn.{q,k,v,o}_proj` (+ optional `_mot_gen`).
    if let Some(proj) = tail.strip_prefix("self_attn.") {
        let proj = proj.strip_suffix("_mot_gen").unwrap_or(proj);
        return matches!(proj, "q_proj" | "k_proj" | "v_proj" | "o_proj");
    }
    // SwiGLU MLP: `mlp.{g}` or `mlp_mot_gen.{g}`.
    for mlp in ["mlp.", "mlp_mot_gen."] {
        if let Some(proj) = tail.strip_prefix(mlp) {
            return matches!(proj, "gate_proj" | "up_proj" | "down_proj");
        }
    }
    false
}

/// Copy a source file (dereferencing HF-cache symlinks to real bytes) into `dst_root` under its
/// original name. Missing optional assets are skipped silently (a snapshot may ship `LICENSE` but not
/// `LICENSE.md`, etc.). Returns whether the file existed and was copied.
fn copy_asset(src_root: &Path, dst_root: &Path, name: &str) -> Result<bool> {
    let src = src_root.join(name);
    if !src.exists() {
        return Ok(false);
    }
    let real = std::fs::canonicalize(&src)?;
    std::fs::copy(&real, dst_root.join(name))?;
    Ok(true)
}

/// Assemble a full pre-quantized turnkey SenseNova-U1 snapshot in `dst_root`: pack the backbone
/// decoder-stack Linears (both paths) into one `model.safetensors`, and copy the config / tokenizer /
/// chat-template / license assets verbatim (deref symlinks). The result loads via
/// [`crate::model::load`] (packed weights auto-detect) with no dense transient. `bits` = 4 (Q4 tier)
/// or 8 (Q8 tier). The **bf16 tier** is the dense source itself (no conversion — mirror its shards +
/// these same assets; see the tier builder in `tests/prequantize_real_weights.rs`).
pub fn prequantize_turnkey(src_root: &Path, dst_root: &Path, bits: i32) -> Result<()> {
    std::fs::create_dir_all(dst_root)?;

    // Flat checkpoint: load every shard into one map, pack the backbone Linears, write one file. The
    // shared `quantize_map` shape guard additionally skips the 1-D norms and 4-D vision convs; the
    // `is_backbone_linear` predicate keeps the 2-D dense heads (embed/lm_head/fm_head) dense.
    let map = load_dir_map(src_root)?;
    let packed = quantize_map(map, bits, GROUP_SIZE, is_backbone_linear)?;
    save_map(&dst_root.join(PACKED_WEIGHTS_FILE), &packed)?;

    // Non-weight assets — the loader parses config.json + tokenizer.json; the rest travel for HF
    // completeness. At minimum config.json + tokenizer.json must be present in the source.
    for name in ASSET_FILES {
        copy_asset(src_root, dst_root, name)?;
    }
    for required in ["config.json", "tokenizer.json"] {
        if !dst_root.join(required).exists() {
            return Err(Error::Msg(format!(
                "sensenova convert: source snapshot {} is missing {required} — the turnkey cannot \
                 load without it",
                src_root.display()
            )));
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
    fn predicate_matches_backbone_linears_only() {
        // Decoder-stack projections (both paths) → packed.
        for base in [
            "language_model.model.layers.0.self_attn.q_proj",
            "language_model.model.layers.0.self_attn.k_proj_mot_gen",
            "language_model.model.layers.41.self_attn.o_proj_mot_gen",
            "language_model.model.layers.7.mlp.gate_proj",
            "language_model.model.layers.7.mlp.down_proj",
            "language_model.model.layers.7.mlp_mot_gen.up_proj",
        ] {
            assert!(is_backbone_linear(base), "{base} should pack");
        }
        // Everything else stays dense: heads, norms, QK-norms, vision, fm head/embedders.
        for base in [
            "language_model.model.embed_tokens",
            "language_model.lm_head",
            "language_model.model.norm",
            "language_model.model.norm_mot_gen",
            "language_model.model.layers.0.input_layernorm",
            "language_model.model.layers.0.self_attn.q_norm",
            "language_model.model.layers.0.self_attn.q_norm_hw_mot_gen",
            "fm_modules.fm_head.0",
            "fm_modules.fm_head.2",
            "fm_modules.timestep_embedder.mlp.0",
            "fm_modules.noise_scale_embedder.mlp.2",
            "vision_model.embeddings.patch_embedding",
            "fm_modules.vision_model_mot_gen.embeddings.dense_embedding",
        ] {
            assert!(!is_backbone_linear(base), "{base} should stay dense");
        }
    }

    fn byte_equal(a: &Array, b: &Array) -> bool {
        a.shape() == b.shape()
            && a.dtype() == b.dtype()
            && eq(a, b).unwrap().all(None).unwrap().item::<bool>()
    }

    /// The packed triple a backbone Linear becomes is byte-identical to the op the load-time
    /// `.quantize` runs (bf16 cast, group 64) — the sc-8669 round-trip guarantee: pre-quantize-on-disk
    /// == quantize-at-load. A dense head (`lm_head`) and a 1-D norm stay dense (predicate / shape
    /// guard).
    #[test]
    fn quantize_map_packs_backbone_byte_identical_to_load_time_quantize() {
        let w = Array::from_slice(
            &(0..64 * 128).map(|i| (i as f32).sin()).collect::<Vec<_>>(),
            &[64, 128],
        );
        let mut map: HashMap<String, Array> = HashMap::new();
        // A gen-path attention projection (packs) + a dense head (`lm_head`, stays dense) + a 1-D
        // norm (shape-guarded dense).
        map.insert(
            "language_model.model.layers.0.self_attn.q_proj_mot_gen.weight".into(),
            w.clone(),
        );
        map.insert(
            "language_model.lm_head.weight".into(),
            Array::from_slice(
                &(0..64 * 128).map(|i| (i as f32).cos()).collect::<Vec<_>>(),
                &[64, 128],
            ),
        );
        map.insert(
            "language_model.model.layers.0.input_layernorm.weight".into(),
            Array::ones::<f32>(&[128]).unwrap(),
        );

        let out = quantize_map(map, 4, GROUP_SIZE, is_backbone_linear).unwrap();

        let base = "language_model.model.layers.0.self_attn.q_proj_mot_gen";
        let wq = out.get(&format!("{base}.weight")).expect("packed");
        assert_eq!(wq.dtype(), Dtype::Uint32, "Q4 codes are u32-packed");
        let scales = out.get(&format!("{base}.scales")).unwrap();
        let biases = out.get(&format!("{base}.biases")).unwrap();
        let (ewq, esc, ebi) =
            quantize(w.as_dtype(Dtype::Bfloat16).unwrap(), GROUP_SIZE, 4).unwrap();
        assert!(byte_equal(wq, &ewq), "packed weight != load-time quantize");
        assert!(byte_equal(scales, &esc), "scales != load-time quantize");
        assert!(byte_equal(biases, &ebi), "biases != load-time quantize");

        // `lm_head` stays dense (predicate) — no packed triple.
        let lm = out.get("language_model.lm_head.weight").unwrap();
        assert_eq!(lm.dtype(), Dtype::Float32, "lm_head unchanged (dense)");
        assert!(!out.contains_key("language_model.lm_head.scales"));
        // The 1-D norm stays dense (shape guard).
        let n = out
            .get("language_model.model.layers.0.input_layernorm.weight")
            .unwrap();
        assert_eq!(n.dtype(), Dtype::Float32, "norm unchanged");
    }
}
