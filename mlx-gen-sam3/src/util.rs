//! Shared SAM3 leaf helpers (F-071): weight-key joining and torch→MLX conv-weight permutes, lifted
//! out of the per-module copies (mirrors `mlx-gen-sam2`'s `util`). The `join` here is the
//! empty-prefix-aware variant — a superset of the plain `format!("{prefix}.{leaf}")` copies that were
//! scattered across the modules: identical for every non-empty prefix, and it returns the bare leaf
//! (rather than a malformed `.leaf`) when the prefix is empty.

use mlx_rs::Array;

use mlx_gen::Result;

/// `"{prefix}.{leaf}"` (or just `leaf` when `prefix` is empty).
pub(crate) fn join(prefix: &str, leaf: &str) -> String {
    if prefix.is_empty() {
        leaf.to_string()
    } else {
        format!("{prefix}.{leaf}")
    }
}

/// Permute a torch conv weight `[out, in, kH, kW]` (OIHW) → MLX `[out, kH, kW, in]` (OHWI).
pub(crate) fn conv_w_ohwi(w: &Array) -> Result<Array> {
    Ok(w.transpose_axes(&[0, 2, 3, 1])?)
}

/// Additive key-padding mask `[1, 1, 1, L]` (0 valid, −1e9 padded), broadcast over heads/queries.
/// The single copy behind the DETR text attention and the mask head's prompt cross-attention
/// (previously duplicated as `detr::text_key_mask` / `model::prompt_key_mask`, F-108).
pub(crate) fn text_key_mask(text_mask: &[i32]) -> Array {
    let row: Vec<f32> = text_mask
        .iter()
        .map(|&m| if m == 1 { 0.0 } else { -1e9 })
        .collect();
    Array::from_slice(&row, &[1, 1, 1, row.len() as i32])
}

/// Permute a torch transposed-conv weight `[in, out, kH, kW]` (IOHW) → MLX `[out, kH, kW, in]` (OHWI).
pub(crate) fn conv_transpose_w(w: &Array) -> Result<Array> {
    Ok(w.transpose_axes(&[1, 2, 3, 0])?)
}
