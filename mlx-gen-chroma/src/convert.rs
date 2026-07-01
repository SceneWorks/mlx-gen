//! Offline pre-quantization: read a dense Chroma diffusers snapshot and write a packed Q4/Q8 turnkey
//! that [`crate::quant`] (via [`crate::model::load_chroma`]) loads with no dense bf16/f32 transient.
//! Mirrors `mlx_gen_sdxl::convert` / `mlx_gen_sensenova::convert` (same `mlx_gen::quant::quantize_map`,
//! byte-equal to the load-time `.quantize` seam), differing in the Chroma key layout and quant scope.
//!
//! Chroma quantizes **one** component — the DiT `transformer/` (the fork's `nn.quantize`, wired in
//! [`crate::transformer::ChromaTransformer::quantize`]): the double blocks' attention + FFN Linears
//! and the single blocks' attention + `proj_mlp`/`proj_out`. Everything else stays **dense in every
//! tier**:
//!
//! * The transformer's own `x_embedder` / `context_embedder` / top-level `proj_out` and the entire
//!   distilled-guidance **Approximator** (`distilled_guidance_layer.*`, which drives all per-block
//!   modulation) — small / precision-sensitive, kept dense to match [`is_transformer_target`].
//! * The T5-XXL **text encoder** (`text_encoder/`) and the FLUX.1 16-ch **VAE** (`vae/`) — never
//!   quantized (a measurably-0% memory-only win, and not wired in the loader), so both are mirrored
//!   verbatim (deref symlinks) into every tier.
//!
//! The per-component pack predicate matches the loader's `.quantize` scope exactly — a missed site (or
//! a wrongly-packed dense tensor) loads u32 codes as dense floats → a garbage render. The completeness
//! gate is the real-weight render in `tests/prequantize_real_weights.rs`.
//!
//! Group-B per-crate converter template (sc-8669 / sc-8777).

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;

use mlx_gen::quant::{quantize_map, save_map};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::quant::GROUP_SIZE;

/// The single packed weight file the turnkey ships for the transformer (replaces the source's sharded
/// `diffusion_pytorch_model-0000N-of-0000M.safetensors`). The loader globs `*.safetensors` under
/// `transformer/`, so one flat file suffices; its stem matches the dense master so nothing downstream
/// changes.
const TRANSFORMER_FILE: &str = "diffusion_pytorch_model.safetensors";

// ============================================================================================
// Pack predicate (operates on the **base** = the on-disk key minus its `.weight`).
// ============================================================================================

/// Whether a `transformer/` key's `base` is a **block Linear** the DiT quantizes — matching
/// [`crate::transformer::ChromaTransformer::quantize`] exactly:
///
/// * a **double** block (`transformer_blocks.{i}`): attention `to_q`/`to_k`/`to_v`/`to_out.0`,
///   `add_q_proj`/`add_k_proj`/`add_v_proj`/`to_add_out`, and the FFN `ff.net.0.proj`/`ff.net.2` +
///   `ff_context.net.0.proj`/`ff_context.net.2`;
/// * a **single** block (`single_transformer_blocks.{i}`): attention `to_q`/`to_k`/`to_v`, plus
///   `proj_mlp` and `proj_out`.
///
/// Everything else stays dense: the per-block QK-norms / added-norms (1-D `norm_*.weight`, also
/// shape-guarded out by [`quantize_map`]), and every top-level module (`x_embedder`,
/// `context_embedder`, the top-level `proj_out`, and the whole `distilled_guidance_layer.*`
/// Approximator). The `single_*` prefix is tested before the `transformer_blocks.` prefix so a single
/// block (which also starts with `…transformer_blocks.` textually) is classified by its own rule.
fn is_transformer_target(base: &str) -> bool {
    if let Some(rest) = base.strip_prefix("single_transformer_blocks.") {
        // rest = `{i}.<tail>`
        let Some((_i, tail)) = rest.split_once('.') else {
            return false;
        };
        return matches!(
            tail,
            "attn.to_q" | "attn.to_k" | "attn.to_v" | "proj_mlp" | "proj_out"
        );
    }
    if let Some(rest) = base.strip_prefix("transformer_blocks.") {
        let Some((_i, tail)) = rest.split_once('.') else {
            return false;
        };
        return matches!(
            tail,
            "attn.to_q"
                | "attn.to_k"
                | "attn.to_v"
                | "attn.to_out.0"
                | "attn.add_q_proj"
                | "attn.add_k_proj"
                | "attn.add_v_proj"
                | "attn.to_add_out"
                | "ff.net.0.proj"
                | "ff.net.2"
                | "ff_context.net.0.proj"
                | "ff_context.net.2"
        );
    }
    false
}

/// Load a component dir's safetensors (single or sharded) into one key→`Array` map. Chroma ships the
/// transformer as sharded `diffusion_pytorch_model-0000N-of-0000M.safetensors`; the shard keys are
/// disjoint, so we merge them (a duplicate key across shards is a corrupt snapshot → error).
fn load_component_map(dir: &Path) -> Result<HashMap<String, Array>> {
    let w = Weights::from_dir(dir)?;
    let mut map: HashMap<String, Array> = HashMap::new();
    for k in w.keys().map(str::to_string).collect::<Vec<_>>() {
        let v = w.get(&k).expect("listed key").clone();
        if map.insert(k.clone(), v).is_some() {
            return Err(Error::Msg(format!(
                "chroma convert: duplicate key `{k}` across shards in {}",
                dir.display()
            )));
        }
    }
    Ok(map)
}

/// Copy `src/config.json` to `dst/config.json` with a `"quantization": {"bits", "group_size"}` block
/// added (HF/diffusers-compat; the Rust loader auto-detects via `{base}.scales` and ignores it). A
/// missing source config starts from an empty object.
fn write_quantized_config(src: &Path, dst: &Path, bits: i32, group_size: i32) -> Result<()> {
    let src_cfg = src.join("config.json");
    let mut v: serde_json::Value = if src_cfg.exists() {
        serde_json::from_str(&std::fs::read_to_string(&src_cfg)?)
            .map_err(|e| Error::Msg(format!("chroma: parse {}: {e}", src_cfg.display())))?
    } else {
        serde_json::json!({})
    };
    v["quantization"] = serde_json::json!({ "bits": bits, "group_size": group_size });
    let text = serde_json::to_string_pretty(&v)
        .map_err(|e| Error::Msg(format!("chroma: serialize config.json: {e}")))?;
    std::fs::create_dir_all(dst)?;
    std::fs::write(dst.join("config.json"), text)?;
    Ok(())
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

/// Copy one top-level file (deref symlink) if it exists. Returns whether it was copied.
fn copy_file(src_root: &Path, dst_root: &Path, name: &str) -> Result<bool> {
    let src = src_root.join(name);
    if !src.exists() {
        return Ok(false);
    }
    let real = std::fs::canonicalize(&src)?;
    std::fs::copy(&real, dst_root.join(name))?;
    Ok(true)
}

/// Assemble a full pre-quantized turnkey Chroma snapshot in `dst_root`: pack the DiT `transformer/`
/// block Linears into one `transformer/diffusion_pytorch_model.safetensors` (+ annotated
/// `config.json`), mirror the dense T5 `text_encoder/` and FLUX.1 `vae/`, and copy the tokenizer /
/// scheduler / `model_index.json` / license verbatim (deref symlinks). The result loads via
/// [`crate::model::load_chroma`] (packed weights auto-detect) with no dense transient. `bits` = 4 (Q4
/// tier) or 8 (Q8 tier). The **bf16 tier** is the dense source itself (no conversion — mirror it; see
/// the tier builder in `tests/prequantize_real_weights.rs`).
pub fn prequantize_turnkey(src_root: &Path, dst_root: &Path, bits: i32) -> Result<()> {
    std::fs::create_dir_all(dst_root)?;

    // Transformer: pack the block Linears into one flat file + annotate config.
    let tr_src = src_root.join("transformer");
    if !tr_src.is_dir() {
        return Err(Error::Msg(format!(
            "chroma convert: source snapshot {} has no transformer/ dir",
            src_root.display()
        )));
    }
    let tr_dst = dst_root.join("transformer");
    std::fs::create_dir_all(&tr_dst)?;
    let map = quantize_map(
        load_component_map(&tr_src)?,
        bits,
        GROUP_SIZE,
        is_transformer_target,
    )?;
    save_map(&tr_dst.join(TRANSFORMER_FILE), &map)?;
    write_quantized_config(&tr_src, &tr_dst, bits, GROUP_SIZE)?;

    // T5 text encoder + FLUX.1 VAE stay dense (never quantized) — mirror both verbatim.
    for rel in ["text_encoder", "vae", "tokenizer", "scheduler"] {
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
        copy_file(src_root, dst_root, f)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{eq, quantize};
    use mlx_rs::{Array, Dtype};

    #[test]
    fn predicate_matches_block_linears_only() {
        // Double-block Linears (attention + FFN) → packed.
        for base in [
            "transformer_blocks.0.attn.to_q",
            "transformer_blocks.0.attn.to_k",
            "transformer_blocks.0.attn.to_v",
            "transformer_blocks.0.attn.to_out.0",
            "transformer_blocks.0.attn.add_q_proj",
            "transformer_blocks.0.attn.add_k_proj",
            "transformer_blocks.0.attn.add_v_proj",
            "transformer_blocks.0.attn.to_add_out",
            "transformer_blocks.18.ff.net.0.proj",
            "transformer_blocks.18.ff.net.2",
            "transformer_blocks.18.ff_context.net.0.proj",
            "transformer_blocks.18.ff_context.net.2",
            // Single-block Linears (attention + proj_mlp/proj_out) → packed.
            "single_transformer_blocks.0.attn.to_q",
            "single_transformer_blocks.0.attn.to_k",
            "single_transformer_blocks.0.attn.to_v",
            "single_transformer_blocks.37.proj_mlp",
            "single_transformer_blocks.37.proj_out",
        ] {
            assert!(is_transformer_target(base), "{base} should pack");
        }
        // Everything else stays dense: per-block norms, top-level embedders/proj_out, Approximator.
        for base in [
            "transformer_blocks.0.attn.norm_q",
            "transformer_blocks.0.attn.norm_k",
            "transformer_blocks.0.attn.norm_added_q",
            "transformer_blocks.0.attn.norm_added_k",
            "single_transformer_blocks.0.attn.norm_q",
            "x_embedder",
            "context_embedder",
            "proj_out",
            "distilled_guidance_layer.in_proj",
            "distilled_guidance_layer.out_proj",
            "distilled_guidance_layer.layers.0.linear_1",
            "distilled_guidance_layer.layers.4.linear_2",
        ] {
            assert!(!is_transformer_target(base), "{base} should stay dense");
        }
    }

    fn byte_equal(a: &Array, b: &Array) -> bool {
        a.shape() == b.shape()
            && a.dtype() == b.dtype()
            && eq(a, b).unwrap().all(None).unwrap().item::<bool>()
    }

    /// The packed triple a block Linear becomes is byte-identical to the op the load-time `.quantize`
    /// runs (bf16 cast, group 64) — the sc-8669 round-trip guarantee: pre-quantize-on-disk ==
    /// quantize-at-load. A top-level embedder stays dense (predicate); a 1-D norm stays dense (shape
    /// guard).
    #[test]
    fn quantize_map_packs_block_linear_byte_identical_to_load_time_quantize() {
        let w = Array::from_slice(
            &(0..64 * 128).map(|i| (i as f32).sin()).collect::<Vec<_>>(),
            &[64, 128],
        );
        let mut map: HashMap<String, Array> = HashMap::new();
        // A double-block attention proj (packs) + a top-level embedder (dense, predicate) + a 1-D
        // QK-norm (shape-guarded dense).
        map.insert("transformer_blocks.0.attn.to_q.weight".into(), w.clone());
        map.insert(
            "x_embedder.weight".into(),
            Array::from_slice(
                &(0..64 * 128).map(|i| (i as f32).cos()).collect::<Vec<_>>(),
                &[64, 128],
            ),
        );
        map.insert(
            "transformer_blocks.0.attn.norm_q.weight".into(),
            Array::ones::<f32>(&[128]).unwrap(),
        );

        let out = quantize_map(map, 4, GROUP_SIZE, is_transformer_target).unwrap();

        let base = "transformer_blocks.0.attn.to_q";
        let wq = out.get(&format!("{base}.weight")).expect("packed");
        assert_eq!(wq.dtype(), Dtype::Uint32, "Q4 codes are u32-packed");
        let scales = out.get(&format!("{base}.scales")).unwrap();
        let biases = out.get(&format!("{base}.biases")).unwrap();
        let (ewq, esc, ebi) =
            quantize(w.as_dtype(Dtype::Bfloat16).unwrap(), GROUP_SIZE, 4).unwrap();
        assert!(byte_equal(wq, &ewq), "packed weight != load-time quantize");
        assert!(byte_equal(scales, &esc), "scales != load-time quantize");
        assert!(byte_equal(biases, &ebi), "biases != load-time quantize");

        // The top-level embedder stays dense (predicate) — no packed triple.
        let xe = out.get("x_embedder.weight").unwrap();
        assert_eq!(xe.dtype(), Dtype::Float32, "x_embedder unchanged (dense)");
        assert!(!out.contains_key("x_embedder.scales"));
        // The 1-D norm stays dense (shape guard).
        let n = out.get("transformer_blocks.0.attn.norm_q.weight").unwrap();
        assert_eq!(n.dtype(), Dtype::Float32, "norm unchanged");
    }
}
