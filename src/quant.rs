//! Quantization — group-wise affine Q4/Q8, the mlx-rs equivalent of the Python mflux fork's
//! `nn.quantize(model, bits=bits)` path. mflux never passes `group_size`, so it uses MLX's
//! default of **64**. The actual quantization seam is
//! [`AdaptableLinear::quantize`](crate::adapters::AdaptableLinear::quantize), which quantizes each
//! `Linear` base in place **with the bf16-parity cast** the fork goldens require — providers route
//! through it (or their per-family loaders), so this module owns only the shared default below.
//!
//! Byte-level packing parity vs the fork (mlx 0.31) is checked in `tests/quant_parity.rs` —
//! the version-drift risk, since the crate links an older bundled MLX (mlx-rs 0.25).
//!
//! This module also owns the **group-size-parametric** packed-weight loaders ([`lin`] / [`embedding`])
//! and the **offline pre-quantization** converter primitives ([`load_dir_map`] / [`save_map`] /
//! [`quantize_map`]) that the provider crates used to copy verbatim (only their `GROUP_SIZE` differed:
//! 64 codebase-default, 32 for Boogu's `3360 = 32·105` DiT). Each provider keeps its own `GROUP_SIZE`
//! constant (and its target predicate) and passes them in (6935).

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::ops::quantize;
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use crate::adapters::AdaptableLinear;
use crate::nn::TokenEmbedding;
use crate::weights::Weights;
use crate::Result;

/// MLX's default quantization group size; mflux relies on it (never overrides). Used by the real
/// quantization seams ([`AdaptableLinear::quantize`](crate::adapters::AdaptableLinear::quantize) and
/// the Kolors ChatGLM3 quantizer).
pub const DEFAULT_GROUP_SIZE: i32 = 64;

/// Derive the quant bit-width from the packed shapes at group size `group_size`: `scales` is
/// `[out, in/gs]` ⇒ `in = scales.cols·gs`; the u32-packed `weight` is `[out, in·bits/32]` ⇒
/// `bits = wq.cols·32/in`. Exact for any group-aligned Q4/Q8 pack, so the bit-width need not be
/// carried in a side manifest.
///
/// F-011: returns `Result` and validates the shapes a corrupt/mis-converted pre-quantized snapshot
/// would otherwise mishandle: a 1-D `scales` (or `wq`) panics on the shape index; a `[out, 0]` scales
/// tensor makes `in_dim == 0` → integer divide-by-zero; a mis-packed `wq` yields bits ∉ {4,8}. The
/// shared load seam for every Group-B packed snapshot feeds straight off external `.safetensors`, so
/// these shapes are untrusted.
pub fn packed_bits(wq: &Array, scales: &Array, group_size: i32) -> Result<i32> {
    let sshape = scales.shape();
    let wshape = wq.shape();
    if sshape.len() != 2 || wshape.len() != 2 {
        return Err(crate::Error::Msg(format!(
            "packed quant: scales and weight must be 2-D, got scales {:?} / weight {:?}",
            sshape, wshape
        )));
    }
    let in_dim = sshape[1] * group_size;
    if in_dim == 0 {
        return Err(crate::Error::Msg(format!(
            "packed quant: zero input dim (scales cols {} × group_size {})",
            sshape[1], group_size
        )));
    }
    let bits = wshape[1] * 32 / in_dim;
    if !matches!(bits, 4 | 8) {
        return Err(crate::Error::Msg(format!(
            "packed quant: inferred bit-width {bits} ∉ {{4, 8}} \
             (weight cols {}, in_dim {in_dim}); snapshot is corrupt or mis-converted",
            wshape[1]
        )));
    }
    Ok(bits)
}

/// Load `{base}` as an [`AdaptableLinear`] — **packed** (Q4/Q8) when `{base}.scales` is present (a
/// pre-quantized snapshot, bit-width inferred from the shapes at `group_size`), else **dense**. `bias`
/// additionally loads the dense `{base}.bias` (the quantization's own `{base}.biases` is distinct and
/// always loaded on the packed path). Auto-detection means one loader serves both a dense bf16 and a
/// pre-quantized snapshot with no `quantization` manifest to read. Shared by the provider crates'
/// packed loaders — each passes its own `group_size`.
pub fn lin(w: &Weights, base: &str, bias: bool, group_size: i32) -> Result<AdaptableLinear> {
    let bias = if bias {
        Some(w.require(&format!("{base}.bias"))?.clone())
    } else {
        None
    };
    if let Some(scales) = w.get(&format!("{base}.scales")) {
        let wq = w.require(&format!("{base}.weight"))?;
        let bits = packed_bits(wq, scales, group_size)?;
        return Ok(AdaptableLinear::from_quantized_parts(
            wq.clone(),
            scales.clone(),
            w.require(&format!("{base}.biases"))?.clone(),
            bias,
            group_size,
            bits,
        ));
    }
    Ok(AdaptableLinear::dense(
        w.require(&format!("{base}.weight"))?.clone(),
        bias,
    ))
}

/// Load `{base}` as a [`TokenEmbedding`] — packed when `{base}.scales` is present, else dense (the
/// embedding analogue of [`lin`]).
pub fn embedding(w: &Weights, base: &str, group_size: i32) -> Result<TokenEmbedding> {
    if let Some(scales) = w.get(&format!("{base}.scales")) {
        let wq = w.require(&format!("{base}.weight"))?;
        let bits = packed_bits(wq, scales, group_size)?;
        return Ok(TokenEmbedding::from_quantized_parts(
            wq.clone(),
            scales.clone(),
            w.require(&format!("{base}.biases"))?.clone(),
            group_size,
            bits,
        ));
    }
    Ok(TokenEmbedding::Dense(
        w.require(&format!("{base}.weight"))?.clone(),
    ))
}

/// Read every tensor of `dir` (possibly sharded safetensors) into an owned key→`Array` map (MLX
/// arrays are ref-counted, so the clone is a handle copy, not a buffer copy). The read side of the
/// offline pre-quantization converters.
pub fn load_dir_map(dir: &Path) -> Result<HashMap<String, Array>> {
    let w = Weights::from_dir(dir)?;
    Ok(w.keys()
        .map(|k| (k.to_string(), w.get(k).expect("listed key").clone()))
        .collect())
}

/// Materialize (`eval`) + write a key→`Array` map to a single `path` safetensors (one file — a packed
/// component is small enough not to need sharding; the loaders glob `*.safetensors`, so one file
/// replaces the source's shards). The write side of the converters.
pub fn save_map(path: &Path, map: &HashMap<String, Array>) -> Result<()> {
    eval(map.values().collect::<Vec<_>>())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Array::save_safetensors(
        map.iter().map(|(k, v)| (k.as_str(), v)),
        None::<&HashMap<String, String>>,
        path,
    )?;
    Ok(())
}

/// Pack every `is_target` `{base}.weight` (group-quantizable 2-D, `in % group_size == 0`,
/// `in >= group_size`) into the triple `{base}.weight` (u32 codes) + `.scales` + `.biases` — the
/// weight cast to **bf16 first** so the pack is byte-identical to the load-time
/// [`AdaptableLinear::quantize`](crate::adapters::AdaptableLinear::quantize) (and the fork's
/// `nn.quantize(bf16)`). Every other tensor (norms, 1-D, non-divisible, non-target) passes through
/// unchanged. `is_target` receives the **base** (the key minus its `.weight` suffix). The shape guard
/// keeps an odd-shaped or 1-D target dense rather than crashing `quantize`.
pub fn quantize_map(
    map: HashMap<String, Array>,
    bits: i32,
    group_size: i32,
    is_target: impl Fn(&str) -> bool,
) -> Result<HashMap<String, Array>> {
    let mut out = HashMap::with_capacity(map.len());
    for (k, v) in map {
        let base = k.strip_suffix(".weight").filter(|b| is_target(b));
        let packable = base.is_some()
            && v.shape().len() == 2
            && v.shape()[1] % group_size == 0
            && v.shape()[1] >= group_size;
        if let (Some(base), true) = (base, packable) {
            let wbf16 = v.as_dtype(Dtype::Bfloat16)?;
            let (wq, scales, biases) = quantize(&wbf16, group_size, bits)?;
            out.insert(format!("{base}.weight"), wq);
            out.insert(format!("{base}.scales"), scales);
            out.insert(format!("{base}.biases"), biases);
        } else {
            out.insert(k, v);
        }
    }
    Ok(out)
}

// ============================================================================================
// Turnkey-assembly glue — the Group-B converter tail the provider `convert.rs` modules used to
// clone verbatim (sc-9108 / F-045). Each provider still owns its per-component pack predicates and
// its `prequantize_turnkey` wiring, but routes the config annotation, the symlink-resolving
// directory copy, and the non-weight asset tail through these shared helpers so the six converters
// stay byte-identical (they had drifted: z-image dropped `LICENSE.txt`; krea's copy skipped the
// symlink deref; sensenova never annotated its config).
// ============================================================================================

/// The canonical set of top-level non-weight assets a turnkey snapshot copies verbatim from the
/// dense source (in addition to the component dirs it packs / passes through): the diffusers pipeline
/// manifest and every license/readme variant an upstream repo might ship. A converter iterates this,
/// copying each that exists (deref'ing HF-cache symlinks). Owning it here fixes the F-045 drift where
/// z-image omitted `LICENSE.txt` and shipped license-less rehosted tiers.
pub const TURNKEY_ASSET_FILES: &[&str] = &[
    "model_index.json",
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "README.md",
];

/// Copy `src/config.json` to `dst/config.json` with a `"quantization": {"bits", "group_size"}` block
/// added (HF/diffusers-compat; the Rust loaders auto-detect packed weights via `{base}.scales` and
/// ignore this block — it is provenance/informational). A missing source config starts from an empty
/// object. The written bytes are `serde_json::to_string_pretty` of the merged value, byte-identical
/// across every provider that packs per-component dirs.
pub fn write_quantized_config(src: &Path, dst: &Path, bits: i32, group_size: i32) -> Result<()> {
    let src_cfg = src.join("config.json");
    let mut v: serde_json::Value = if src_cfg.exists() {
        serde_json::from_str(&std::fs::read_to_string(&src_cfg)?).map_err(|e| {
            crate::Error::Msg(format!("quant convert: parse {}: {e}", src_cfg.display()))
        })?
    } else {
        serde_json::json!({})
    };
    v["quantization"] = serde_json::json!({ "bits": bits, "group_size": group_size });
    let text = serde_json::to_string_pretty(&v)
        .map_err(|e| crate::Error::Msg(format!("quant convert: serialize config.json: {e}")))?;
    std::fs::create_dir_all(dst)?;
    std::fs::write(dst.join("config.json"), text)?;
    Ok(())
}

/// Recursively copy a directory's files, resolving symlinks (HF snapshots symlink into
/// `../../blobs/…`) to real bytes so the assembled tier is self-contained and HF-uploadable. This is
/// the symlink-resolving copy the six sibling converters share (the F-045 unification point where
/// krea's non-deref'ing copy is brought into line).
pub fn copy_dir(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        // Never carry a hidden entry into an assembled tier: an AppleDouble sidecar copied in here
        // would be uploaded with the turnkey and then break `Weights::from_dir` for everyone who
        // downloads it (SceneWorks#1333).
        if gen_core::weightsmeta::is_hidden_file(&path) {
            continue;
        }
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

/// Copy a single top-level asset `name` from `src_root` to `dst_root` if it exists, deref'ing an
/// HF-cache symlink to real bytes. Returns whether the file was present and copied. The per-file
/// building block behind [`copy_turnkey_assets`] (some converters — e.g. sensenova — need the copied?
/// signal per file).
pub fn copy_asset(src_root: &Path, dst_root: &Path, name: &str) -> Result<bool> {
    let src = src_root.join(name);
    if !src.exists() {
        return Ok(false);
    }
    let real = std::fs::canonicalize(&src)?;
    std::fs::create_dir_all(dst_root)?;
    std::fs::copy(&real, dst_root.join(name))?;
    Ok(true)
}

/// Copy every [`TURNKEY_ASSET_FILES`] entry that exists from `src_root` to `dst_root` (deref'ing
/// symlinks). The shared non-weight tail of `prequantize_turnkey`.
pub fn copy_turnkey_assets(src_root: &Path, dst_root: &Path) -> Result<()> {
    for name in TURNKEY_ASSET_FILES {
        copy_asset(src_root, dst_root, name)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// F-011: a Q4 pack at group_size 64 derives bits == 4 from the standard shapes.
    #[test]
    fn packed_bits_derives_q4() {
        // scales [out, in/gs] = [128, 4] ⇒ in_dim 256; wq [out, in·4/32] = [128, 32] ⇒ bits 4.
        let scales = Array::zeros::<f32>(&[128, 4]).unwrap();
        let wq = Array::zeros::<u32>(&[128, 32]).unwrap();
        assert_eq!(packed_bits(&wq, &scales, 64).unwrap(), 4);
    }

    #[test]
    fn packed_bits_derives_q8() {
        // scales [out, in/gs] = [64, 2] ⇒ in_dim 128; wq [out, in·8/32] = [64, 32] ⇒ bits 8.
        let scales = Array::zeros::<f32>(&[64, 2]).unwrap();
        let wq = Array::zeros::<u32>(&[64, 32]).unwrap();
        assert_eq!(packed_bits(&wq, &scales, 64).unwrap(), 8);
    }

    #[test]
    fn packed_bits_rejects_1d_shapes() {
        let scales = Array::zeros::<f32>(&[64]).unwrap(); // 1-D
        let wq = Array::zeros::<u32>(&[64, 32]).unwrap();
        let err = packed_bits(&wq, &scales, 64).unwrap_err().to_string();
        assert!(err.contains("must be 2-D"), "{err}");
    }

    #[test]
    fn packed_bits_rejects_zero_in_dim() {
        // [out, 0] scales ⇒ in_dim 0 ⇒ integer divide-by-zero today; now a typed error.
        let scales = Array::zeros::<f32>(&[64, 0]).unwrap();
        let wq = Array::zeros::<u32>(&[64, 32]).unwrap();
        let err = packed_bits(&wq, &scales, 64).unwrap_err().to_string();
        assert!(err.contains("zero input dim"), "{err}");
    }

    #[test]
    fn packed_bits_rejects_non_q4_q8_width() {
        // in_dim 256, wq cols 24 ⇒ bits 24·32/256 = 3 ∉ {4,8}.
        let scales = Array::zeros::<f32>(&[64, 4]).unwrap(); // in_dim 256
        let wq = Array::zeros::<u32>(&[64, 24]).unwrap(); // bits 3
        let err = packed_bits(&wq, &scales, 64).unwrap_err().to_string();
        assert!(err.contains("∉ {4, 8}"), "{err}");
    }
}
