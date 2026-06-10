//! Native (Rust/MLX) FLUX.2-klein **single-file → diffusers** converter (sc-3136).
//!
//! Community FLUX.2-klein fine-tunes such as `wikeeyang/Flux2-Klein-9B-True-V2` ship the
//! transformer ONLY, as a single flat `.safetensors` in the original (ComfyUI/BFL) key
//! convention — no diffusers subfolders, no text-encoder/VAE. The [`flux2_klein_9b`] loader
//! ([`crate::loader::load_transformer`]) consumes the **diffusers** tree (its remap is a pure
//! 1:1 rename), so the on-disk tensors must already be in diffusers convention.
//!
//! [`convert_and_assemble`] reproduces, in Rust/MLX, the exact transforms the (now-retired,
//! sc-3032) Python `apps/worker/scene_worker/mlx_flux_convert.py` (sc-2235) applied — itself a
//! mirror of diffusers' `convert_flux2_transformer_checkpoint_to_diffusers`:
//!
//!   * key renames (`img_in` → `x_embedder`, `*.lin` → `*.linear`, …),
//!   * double-block fused `qkv` `[3·d, d]` row-split into `to_q`/`to_k`/`to_v` (img stream) and
//!     `add_q_proj`/`add_k_proj`/`add_v_proj` (txt stream),
//!   * single-block `linear1`/`linear2` → `to_qkv_mlp_proj`/`to_out` (1:1 — diffusers keeps the
//!     single block fused),
//!   * `final_layer.adaLN_modulation.1` → `norm_out.linear` WITH a **scale/shift swap**: BFL
//!     packs `(shift, scale)`; diffusers/this crate expect `(scale, shift)`. This one swap is
//!     load-bearing — that tensor modulates every output patch, so getting it wrong corrupts the
//!     whole image with a periodic weave (sc-2220).
//!
//! then assembles a complete local diffusers model dir by borrowing the untouched VAE / text
//! encoder / tokenizer / scheduler from an already-installed base FLUX.2-klein-9B snapshot.
//!
//! [`flux2_klein_9b`]: crate::model::load_klein_9b

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::ops::{concatenate_axis, split};
use mlx_rs::transforms::eval;
use mlx_rs::Array;

/// Borrowed-from-base subdirs: a transformer-only fine-tune does not touch these, so taking them
/// from the installed base klein-9B is correct. Symlinked (absolute) to avoid duplicating the
/// multi-GB encoder/VAE weights on disk.
const BORROWED_SUBDIRS: &[&str] = &["vae", "text_encoder", "tokenizer", "scheduler"];
/// Borrowed-from-base top-level files (copied, not symlinked — small, and must survive the
/// worker's temp→final atomic rename as real files).
const BORROWED_FILES: &[&str] = &["model_index.json"];

/// Top-level (non-block) direct renames: original → diffusers.
const TOP_RENAMES: &[(&str, &str)] = &[
    ("img_in.weight", "x_embedder.weight"),
    ("txt_in.weight", "context_embedder.weight"),
    (
        "time_in.in_layer.weight",
        "time_guidance_embed.timestep_embedder.linear_1.weight",
    ),
    (
        "time_in.out_layer.weight",
        "time_guidance_embed.timestep_embedder.linear_2.weight",
    ),
    (
        "double_stream_modulation_img.lin.weight",
        "double_stream_modulation_img.linear.weight",
    ),
    (
        "double_stream_modulation_txt.lin.weight",
        "double_stream_modulation_txt.linear.weight",
    ),
    (
        "single_stream_modulation.lin.weight",
        "single_stream_modulation.linear.weight",
    ),
    ("final_layer.linear.weight", "proj_out.weight"),
];

/// Handled separately (scale/shift swap): `final_layer.adaLN_modulation.1` → `norm_out.linear`.
const ADALN_SOURCE: &str = "final_layer.adaLN_modulation.1.weight";
const ADALN_TARGET: &str = "norm_out.linear.weight";

/// Per-double-block renames (original suffix → diffusers suffix), excluding the fused qkv tensors
/// which are row-split below.
const DOUBLE_RENAMES: &[(&str, &str)] = &[
    ("img_attn.norm.query_norm.weight", "attn.norm_q.weight"),
    ("img_attn.norm.key_norm.weight", "attn.norm_k.weight"),
    ("img_attn.proj.weight", "attn.to_out.0.weight"),
    ("img_mlp.0.weight", "ff.linear_in.weight"),
    ("img_mlp.2.weight", "ff.linear_out.weight"),
    (
        "txt_attn.norm.query_norm.weight",
        "attn.norm_added_q.weight",
    ),
    ("txt_attn.norm.key_norm.weight", "attn.norm_added_k.weight"),
    ("txt_attn.proj.weight", "attn.to_add_out.weight"),
    ("txt_mlp.0.weight", "ff_context.linear_in.weight"),
    ("txt_mlp.2.weight", "ff_context.linear_out.weight"),
];

/// Fused qkv suffix → `(q, k, v)` target suffixes, per stream.
const DOUBLE_QKV: &[(&str, [&str; 3])] = &[
    (
        "img_attn.qkv.weight",
        ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"],
    ),
    (
        "txt_attn.qkv.weight",
        [
            "attn.add_q_proj.weight",
            "attn.add_k_proj.weight",
            "attn.add_v_proj.weight",
        ],
    ),
];

/// Per-single-block renames (1:1; diffusers keeps the fused single block).
const SINGLE_RENAMES: &[(&str, &str)] = &[
    ("linear1.weight", "attn.to_qkv_mlp_proj.weight"),
    ("linear2.weight", "attn.to_out.weight"),
    ("norm.query_norm.weight", "attn.norm_q.weight"),
    ("norm.key_norm.weight", "attn.norm_k.weight"),
];

/// Count the blocks under `prefix` (`max(i)+1` over keys matching `^{prefix}.{i}.…`), the fork's
/// `_count_blocks` — derives the layer count from the checkpoint itself rather than the config.
fn count_blocks<'a>(keys: impl Iterator<Item = &'a str>, prefix: &str) -> usize {
    let pat = format!("{prefix}.");
    let mut max_idx: Option<usize> = None;
    for k in keys {
        let Some(rest) = k.strip_prefix(&pat) else {
            continue;
        };
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        // Require a trailing '.' after the index so `double_blocks.10.` parses as 10, not a prefix
        // collision; a key like `double_blocksX` (no match) is already filtered by `strip_prefix`.
        if digits.is_empty() || !rest[digits.len()..].starts_with('.') {
            continue;
        }
        if let Ok(i) = digits.parse::<usize>() {
            max_idx = Some(max_idx.map_or(i, |m| m.max(i)));
        }
    }
    max_idx.map_or(0, |m| m + 1)
}

/// Row-split a `[3·d, …]` tensor into three equal `[d, …]` chunks along axis 0 (`mx.split(t, 3)`).
fn chunk3(t: &Array) -> Result<[Array; 3]> {
    let mut parts = split(t, 3, 0)?;
    if parts.len() != 3 {
        return Err(Error::Msg(format!(
            "fused qkv split expected 3 parts, got {} (shape {:?})",
            parts.len(),
            t.shape()
        )));
    }
    let v = parts.pop().unwrap();
    let k = parts.pop().unwrap();
    let q = parts.pop().unwrap();
    Ok([q, k, v])
}

/// Split a `[2·d, …]` tensor at the midpoint and swap the halves: BFL `(shift, scale)` →
/// diffusers `(scale, shift)`. Load-bearing (sc-2220).
fn swap_halves(t: &Array) -> Result<Array> {
    let parts = split(t, 2, 0)?;
    if parts.len() != 2 {
        return Err(Error::Msg(format!(
            "adaLN half-swap expected 2 parts, got {} (shape {:?})",
            parts.len(),
            t.shape()
        )));
    }
    Ok(concatenate_axis(&[&parts[1], &parts[0]], 0)?)
}

/// Map an original-format FLUX.2-klein transformer tensor set (`src`) onto the diffusers key set
/// (the fork's `build_target_state_dict`). Pure remapping — renames + qkv row-split + the adaLN
/// half-swap; no I/O. The produced keys are exactly the base diffusers transformer's keys.
pub fn build_target_state_dict(src: &Weights) -> Result<HashMap<String, Array>> {
    let mut out: HashMap<String, Array> = HashMap::new();

    for (s, d) in TOP_RENAMES {
        out.insert((*d).to_string(), src.require(s)?.clone());
    }
    out.insert(
        ADALN_TARGET.to_string(),
        swap_halves(src.require(ADALN_SOURCE)?)?,
    );

    let n_double = count_blocks(src.keys(), "double_blocks");
    for i in 0..n_double {
        let (s, d) = (
            format!("double_blocks.{i}"),
            format!("transformer_blocks.{i}"),
        );
        for (src_suffix, [q, k, v]) in DOUBLE_QKV {
            let [tq, tk, tv] = chunk3(src.require(&format!("{s}.{src_suffix}"))?)?;
            out.insert(format!("{d}.{q}"), tq);
            out.insert(format!("{d}.{k}"), tk);
            out.insert(format!("{d}.{v}"), tv);
        }
        for (src_suffix, dst_suffix) in DOUBLE_RENAMES {
            out.insert(
                format!("{d}.{dst_suffix}"),
                src.require(&format!("{s}.{src_suffix}"))?.clone(),
            );
        }
    }

    let n_single = count_blocks(src.keys(), "single_blocks");
    for i in 0..n_single {
        let (s, d) = (
            format!("single_blocks.{i}"),
            format!("single_transformer_blocks.{i}"),
        );
        for (src_suffix, dst_suffix) in SINGLE_RENAMES {
            out.insert(
                format!("{d}.{dst_suffix}"),
                src.require(&format!("{s}.{src_suffix}"))?.clone(),
            );
        }
    }

    Ok(out)
}

/// Read a safetensors file's tensor names + shapes from the JSON header alone (no weights), the
/// fork's `_safetensors_header_keys`. The format is an 8-byte little-endian header length followed
/// by that many UTF-8 JSON bytes mapping `name → { "shape": [...], "dtype": ..., "data_offsets": … }`.
fn safetensors_header_shapes(path: &Path) -> Result<HashMap<String, Vec<i64>>> {
    use std::io::Read;

    // Read ONLY the 8-byte length prefix + the `n` header bytes — never the (multi-GB) weight body
    // (F-097: `std::fs::read` of the whole shard transiently doubled converter peak RSS just to parse
    // a few-KB header, at the worst moment — while the converted map is also resident).
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
    // A safetensors header is JSON metadata (KB–MB), never gigabytes; it must also fit in the file
    // after the 8-byte prefix. Reject an out-of-range / absurd length rather than allocating it.
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

/// Hard guard: the produced key set + shapes must exactly match the base klein diffusers
/// transformer (the ground-truth layout the loader consumes). Catches a botched remap (missing /
/// extra / wrong-shape keys) at convert time rather than as garbage at generate time.
fn validate_against_base(
    produced: &HashMap<String, Array>,
    base_transformer_dir: &Path,
) -> Result<()> {
    let mut base: HashMap<String, Vec<i64>> = HashMap::new();
    let mut shards: Vec<PathBuf> = std::fs::read_dir(base_transformer_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    shards.sort();
    for shard in &shards {
        base.extend(safetensors_header_shapes(shard)?);
    }
    if base.is_empty() {
        return Err(Error::Msg(format!(
            "no base transformer safetensors in {}",
            base_transformer_dir.display()
        )));
    }

    let mut missing: Vec<&String> = base.keys().filter(|k| !produced.contains_key(*k)).collect();
    let mut extra: Vec<&String> = produced.keys().filter(|k| !base.contains_key(*k)).collect();
    let mut bad_shape: Vec<&String> = produced
        .iter()
        .filter(|(k, v)| {
            base.get(*k)
                .is_some_and(|b| b.iter().map(|&d| d as i32).ne(v.shape().iter().copied()))
        })
        .map(|(k, _)| k)
        .collect();
    if missing.is_empty() && extra.is_empty() && bad_shape.is_empty() {
        return Ok(());
    }
    missing.sort();
    extra.sort();
    bad_shape.sort();
    Err(Error::Msg(format!(
        "conversion validation FAILED vs base transformer: {} missing, {} extra, {} shape mismatch. \
         missing={:?} extra={:?} shape={:?}",
        missing.len(),
        extra.len(),
        bad_shape.len(),
        &missing[..missing.len().min(5)],
        &extra[..extra.len().min(5)],
        &bad_shape[..bad_shape.len().min(5)],
    )))
}

/// Remove an existing path (file, symlink, or directory) if present, so a re-convert is idempotent.
fn remove_if_exists(path: &Path) -> Result<()> {
    // `symlink_metadata` does not follow the link, so a dangling symlink is still detected.
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.is_dir() => std::fs::remove_dir_all(path)?,
        Ok(_) => std::fs::remove_file(path)?,
        Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    Ok(())
}

/// Convert `source_file` (an original single-file FLUX.2-klein transformer in BFL convention) into
/// `out_dir` as a complete diffusers model dir, borrowing the VAE / text encoder / tokenizer /
/// scheduler from `base_dir` (an installed base FLUX.2-klein-9B diffusers snapshot). Returns
/// `out_dir`. The result loads directly through [`crate::model::load_klein_9b`] (engine id
/// `flux2_klein_9b`) via the worker's `modelPath` seam.
///
/// Faithful Rust/MLX port of the retired Python `mlx_flux_convert.py::convert_and_assemble`
/// (sc-2235 / sc-3136). The borrowed subdirs are absolute symlinks (so they survive the worker's
/// temp→final atomic rename without duplicating multi-GB weights); `model_index.json` and the
/// transformer `config.json` are copied as real files.
pub fn convert_and_assemble(
    source_file: impl AsRef<Path>,
    base_dir: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
) -> Result<PathBuf> {
    let source = source_file.as_ref();
    let base = base_dir.as_ref();
    let out = out_dir.as_ref();
    let base_transformer = base.join("transformer");
    if !source.is_file() {
        return Err(Error::Msg(format!(
            "source transformer file not found: {}",
            source.display()
        )));
    }
    if !base_transformer.is_dir() {
        return Err(Error::Msg(format!(
            "base transformer dir not found: {}",
            base_transformer.display()
        )));
    }

    let src = Weights::from_file(source)?;
    let produced = build_target_state_dict(&src)?;
    validate_against_base(&produced, &base_transformer)?;

    // Materialize before saving (mirrors the fork's explicit `mx.eval`).
    let arrays: Vec<&Array> = produced.values().collect();
    eval(arrays)?;

    let out_transformer = out.join("transformer");
    std::fs::create_dir_all(&out_transformer)?;
    Array::save_safetensors(
        produced.iter().map(|(k, v)| (k.as_str(), v)),
        None::<&HashMap<String, String>>,
        out_transformer.join("diffusion_pytorch_model.safetensors"),
    )?;
    std::fs::copy(
        base_transformer.join("config.json"),
        out_transformer.join("config.json"),
    )?;

    // Borrow the untouched components from the base klein snapshot.
    for name in BORROWED_FILES {
        let src_path = std::fs::canonicalize(base.join(name))?;
        let dst = out.join(name);
        remove_if_exists(&dst)?;
        std::fs::copy(&src_path, &dst)?;
    }
    for name in BORROWED_SUBDIRS {
        let src_path = base.join(name);
        if !src_path.exists() {
            return Err(Error::Msg(format!(
                "base component missing: {}",
                src_path.display()
            )));
        }
        // Absolute target so the symlink survives a temp→final rename of `out`.
        let src_path = std::fs::canonicalize(&src_path)?;
        let dst = out.join(name);
        remove_if_exists(&dst)?;
        std::os::unix::fs::symlink(&src_path, &dst)?;
    }

    Ok(out.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::all_close;

    /// Exact (bit-equal) array comparison via `all_close` with zero tolerance.
    fn exact_eq(a: &Array, b: &Array) -> bool {
        a.shape() == b.shape() && all_close(a, b, 0.0, 0.0, false).unwrap().item::<bool>()
    }

    /// F-097: the header reader parses shapes from the prefix + JSON header alone, ignoring the
    /// (here, deliberately present) weight body, and rejects an out-of-range header length.
    #[test]
    fn header_shapes_reads_header_without_body() {
        use std::io::Write;
        let header =
            br#"{"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"__metadata__":{"a":"b"}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header);
        buf.extend_from_slice(&[7u8; 24]); // the "weights" body — must not be needed.
        let path = std::env::temp_dir().join("mlx_gen_flux2_hdr_ok.safetensors");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&buf)
            .unwrap();

        let shapes = safetensors_header_shapes(&path).unwrap();
        assert_eq!(shapes.len(), 1, "__metadata__ is skipped");
        assert_eq!(shapes.get("w"), Some(&vec![2i64, 3]));
        std::fs::remove_file(&path).ok();

        // A header length larger than the file is rejected (not allocated).
        let mut bad = Vec::new();
        bad.extend_from_slice(&(1u64 << 40).to_le_bytes()); // claims a 1 TiB header
        bad.extend_from_slice(b"{}");
        let bad_path = std::env::temp_dir().join("mlx_gen_flux2_hdr_bad.safetensors");
        std::fs::File::create(&bad_path)
            .unwrap()
            .write_all(&bad)
            .unwrap();
        let err = safetensors_header_shapes(&bad_path)
            .unwrap_err()
            .to_string();
        assert!(err.contains("header length out of range"), "{err}");
        std::fs::remove_file(&bad_path).ok();
    }

    /// `chunk3` row-splits `[3·d, c]` into three `[d, c]` chunks, in order.
    #[test]
    fn chunk3_splits_rows_in_order() {
        // rows 0,1 = q ; 2,3 = k ; 4,5 = v ; each row tagged by its value.
        let t = Array::from_slice(
            &[
                0.0f32, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0,
            ],
            &[6, 2],
        );
        let [q, k, v] = chunk3(&t).unwrap();
        assert!(exact_eq(
            &q,
            &Array::from_slice(&[0.0f32, 0.0, 1.0, 1.0], &[2, 2])
        ));
        assert!(exact_eq(
            &k,
            &Array::from_slice(&[2.0f32, 2.0, 3.0, 3.0], &[2, 2])
        ));
        assert!(exact_eq(
            &v,
            &Array::from_slice(&[4.0f32, 4.0, 5.0, 5.0], &[2, 2])
        ));
    }

    /// `swap_halves` swaps the top and bottom halves: `(shift, scale)` → `(scale, shift)`.
    #[test]
    fn swap_halves_swaps_top_and_bottom() {
        // top half (rows 0,1) = "shift" = 1 ; bottom half (rows 2,3) = "scale" = 9.
        let t = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 9.0], &[4, 2]);
        let out = swap_halves(&t).unwrap();
        // After the swap, the top half must be the old bottom (scale=9), bottom the old top (shift=1).
        let expected = Array::from_slice(&[9.0f32, 9.0, 9.0, 9.0, 1.0, 1.0, 1.0, 1.0], &[4, 2]);
        assert!(exact_eq(&out, &expected));
    }

    #[test]
    fn count_blocks_handles_double_digits_and_collisions() {
        let keys = [
            "double_blocks.0.img_attn.qkv.weight",
            "double_blocks.7.img_attn.qkv.weight",
            "double_blocks.10.img_attn.qkv.weight",
            "single_blocks.0.linear1.weight",
            "single_blocks.23.linear2.weight",
            "img_in.weight",           // no block prefix
            "double_blocksX.0.weight", // prefix collision — must not match
        ];
        assert_eq!(count_blocks(keys.iter().copied(), "double_blocks"), 11);
        assert_eq!(count_blocks(keys.iter().copied(), "single_blocks"), 24);
        assert_eq!(count_blocks(keys.iter().copied(), "missing_blocks"), 0);
    }

    /// End-to-end on a synthetic single-file checkpoint shaped like the real klein transformer
    /// (tiny dims): the produced key set must match the expected diffusers layout, with the qkv
    /// split and adaLN swap applied. No real weights — exercises the pure remap.
    #[test]
    fn build_target_state_dict_synthetic_layout() {
        use std::collections::HashSet;

        // Minimal synthetic dims: d=4 (so qkv = 3·4 = 12 rows, adaLN = 2·4 = 8 rows), 2 double +
        // 1 single block. Shapes need only be split-compatible on axis 0.
        let d = 4i32;
        let ones = |rows: i32, cols: i32| Array::ones::<f32>(&[rows, cols]).unwrap();
        let mut src = Weights::from_file(write_tmp_weights(&[
            ("img_in.weight", ones(d, 8)),
            ("txt_in.weight", ones(d, 8)),
            ("time_in.in_layer.weight", ones(d, 8)),
            ("time_in.out_layer.weight", ones(d, d)),
            ("double_stream_modulation_img.lin.weight", ones(d, d)),
            ("double_stream_modulation_txt.lin.weight", ones(d, d)),
            ("single_stream_modulation.lin.weight", ones(d, d)),
            ("final_layer.linear.weight", ones(d, d)),
            ("final_layer.adaLN_modulation.1.weight", ones(2 * d, d)),
        ]))
        .unwrap();
        for i in 0..2 {
            for (suf, rows, cols) in [
                ("img_attn.qkv.weight", 3 * d, d),
                ("txt_attn.qkv.weight", 3 * d, d),
                ("img_attn.norm.query_norm.weight", d, 1),
                ("img_attn.norm.key_norm.weight", d, 1),
                ("img_attn.proj.weight", d, d),
                ("img_mlp.0.weight", d, d),
                ("img_mlp.2.weight", d, d),
                ("txt_attn.norm.query_norm.weight", d, 1),
                ("txt_attn.norm.key_norm.weight", d, 1),
                ("txt_attn.proj.weight", d, d),
                ("txt_mlp.0.weight", d, d),
                ("txt_mlp.2.weight", d, d),
            ] {
                src.insert(format!("double_blocks.{i}.{suf}"), ones(rows, cols));
            }
        }
        for (suf, rows, cols) in [
            ("linear1.weight", d, d),
            ("linear2.weight", d, d),
            ("norm.query_norm.weight", d, 1),
            ("norm.key_norm.weight", d, 1),
        ] {
            src.insert(format!("single_blocks.0.{suf}"), ones(rows, cols));
        }

        let out = build_target_state_dict(&src).unwrap();
        let got: HashSet<&str> = out.keys().map(String::as_str).collect();

        // Spot-check the load-bearing transforms produced the right keys.
        for expected in [
            "x_embedder.weight",
            "context_embedder.weight",
            "time_guidance_embed.timestep_embedder.linear_1.weight",
            "proj_out.weight",
            "norm_out.linear.weight",
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.to_k.weight",
            "transformer_blocks.0.attn.to_v.weight",
            "transformer_blocks.1.attn.add_q_proj.weight",
            "transformer_blocks.1.attn.add_v_proj.weight",
            "transformer_blocks.0.attn.to_out.0.weight",
            "transformer_blocks.0.ff.linear_in.weight",
            "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",
            "single_transformer_blocks.0.attn.to_out.weight",
        ] {
            assert!(got.contains(expected), "missing produced key: {expected}");
        }
        // No original-convention keys leaked through.
        assert!(!got.iter().any(|k| k.contains("img_attn")
            || k.contains(".lin.")
            || k.starts_with("double_blocks.")
            || k.starts_with("single_blocks.")));
        // qkv split: 6 q/k/v keys per double block; the fused source key is gone.
        assert!(!got.contains("transformer_blocks.0.attn.qkv.weight"));
        // 8 top + 1 adaLN + 2·16 double + 1·4 single = 45 keys for this synthetic layout.
        assert_eq!(out.len(), 8 + 1 + 2 * 16 + 4);

        // adaLN target keeps the [2·d, d] shape (the swap is a within-tensor reorder).
        assert_eq!(out["norm_out.linear.weight"].shape(), &[2 * d, d]);
        // qkv split each [d, d].
        assert_eq!(
            out["transformer_blocks.0.attn.to_q.weight"].shape(),
            &[d, d]
        );
    }

    /// Write tensors to a unique temp safetensors file and return its path (the test loads it back
    /// through `Weights::from_file`, the same entry the real converter uses).
    fn write_tmp_weights(entries: &[(&str, Array)]) -> PathBuf {
        // A content-derived suffix keeps parallel test cases from colliding without `Date`/`rand`
        // (both unavailable in this crate's MLX build).
        let tag: usize = entries.iter().map(|(k, _)| k.len()).sum();
        let path =
            std::env::temp_dir().join(format!("mlx_gen_flux2_convert_test_{tag}.safetensors"));
        Array::save_safetensors(
            entries.iter().map(|(k, v)| (*k, v)),
            None::<&HashMap<String, String>>,
            &path,
        )
        .unwrap();
        path
    }
}
