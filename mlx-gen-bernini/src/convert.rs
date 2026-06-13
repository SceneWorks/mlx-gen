//! sc-5144: Bernini **semantic-planner** weights converter.
//!
//! Extracts the planner components from a full `ByteDance/Bernini-Diffusers` package into a native
//! MLX planner snapshot. The full package packs *everything* — the two renderer DiTs, the planner's
//! Qwen2.5-VL backbone, the connector, the ViT decoder, the mask token, **and** a T5 copy — into one
//! combined `bernini/` safetensors index (38 F32 shards, 3315 tensors) keyed by component prefix:
//!
//! | prefix              | count | component                                    |
//! |---------------------|-------|----------------------------------------------|
//! | `diff_dec`          | 1095  | renderer high-noise DiT  (→ sc-4705)         |
//! | `diff_dec_low`      | 1095  | renderer low-noise DiT   (→ sc-4705)         |
//! | `mllm`              | 729   | Qwen2.5-VL-7B planner  (drop `lm_head` → 728) |
//! | `t5_text_encoder`   | 243   | UMT5 (redundant with the standalone dir)     |
//! | `vit_decoder`       | 140   | `DiffLoss_FM` / `SimpleMLPAdaLN` clip-diff head |
//! | `connector`         | 12    | `MLPConnector` (`proj_gen` + `pred_vit`)     |
//! | `mask_tokens`       | 1     | MAR mask token `[1, 4096, 3584]`             |
//!
//! This converter pulls the **four planner groups** (`mllm`, `connector`, `vit_decoder`,
//! `mask_tokens`) out by prefix in a single shard sweep, strips the prefix, drops `mllm.lm_head`
//! (the planner is a stateless feature extractor — no token generation), casts F32→bf16 (Qwen2.5-VL's
//! native dtype), and writes one safetensors per component. It then captures the planner knobs into a
//! `bernini_planner.json` sidecar and links/copies the shared diffusers encoders verbatim
//! (`t5_text_encoder/`, `t5_tokenizer/`, `vae/`, `scheduler/`, and the Qwen2.5-VL `mllm/`
//! tokenizer/processor JSONs).
//!
//! The renderer DiTs (`diff_dec`/`diff_dec_low`) are handled by
//! [`mlx_gen_wan::convert::assemble_bernini_renderer_snapshot`] (sc-4705), which already reads from
//! this same `bernini/` index; this converter intentionally leaves them out. Q4/Q8 is applied at
//! load time (sc-5146), like the sensenova / lens planners — not baked into the on-disk snapshot —
//! so the output here is bf16.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

/// Knob sidecar file the planner loader / module stories (sc-5132 / sc-5139 / sc-5140) read.
pub const BERNINI_PLANNER_SIDECAR: &str = "bernini_planner.json";

/// One planner component group extracted from the combined `bernini/` index.
struct Component {
    /// Source key prefix in the combined index (the bare `mask_tokens` parameter has no trailing key).
    prefix: &'static str,
    /// Output safetensors file (doubles as the group id).
    out: &'static str,
    /// Expected tensor count after the `lm_head` drop — a hard conversion guard.
    expect: usize,
}

/// The four planner groups. Renderer DiTs (`diff_dec*`) and the redundant `t5_text_encoder.*` copy
/// are deliberately absent so their tensors are skipped (the standalone `t5_text_encoder/` dir is the
/// faithful UMT5 the reference loads). No prefix here is a prefix of another, so first-match routing
/// is unambiguous.
const COMPONENTS: [Component; 4] = [
    // Qwen2.5-VL-7B: `visual.*` (390) + `model.*` (338) = 728 after dropping `mllm.lm_head.weight`.
    Component {
        prefix: "mllm.",
        out: "qwen2_5_vl.safetensors",
        expect: 728,
    },
    // MLPConnector: proj_gen {0,2,3} (5) + pred_vit {0,2,3,4} (7).
    Component {
        prefix: "connector.",
        out: "connector.safetensors",
        expect: 12,
    },
    // DiffLoss_FM net: time_embed(4)+cond_embed(2)+input_proj(2)+16·res_blocks(8)+final_layer(4).
    Component {
        prefix: "vit_decoder.",
        out: "vit_decoder.safetensors",
        expect: 140,
    },
    // The single MAR mask-token parameter `[1, num_mask_token, hidden]`.
    Component {
        prefix: "mask_tokens",
        out: "mask_tokens.safetensors",
        expect: 1,
    },
];

/// Route a combined-index key to `(output file, stripped key)`, or `None` if it is not a planner
/// tensor (a renderer DiT, the redundant T5 copy, or the dropped `mllm.lm_head`). The `mask_tokens`
/// parameter has no trailing segment, so its stripped key stays `mask_tokens`.
fn route_key(k: &str) -> Option<(&'static str, String)> {
    if k == "mllm.lm_head.weight" {
        return None;
    }
    for c in &COMPONENTS {
        if let Some(rest) = k.strip_prefix(c.prefix) {
            let key = if rest.is_empty() {
                k.to_string()
            } else {
                rest.to_string()
            };
            return Some((c.out, key));
        }
    }
    None
}

/// One sweep over the `bernini/` shards, routing each tensor to its planner group (lazy mmap-backed
/// clones, prefix stripped). The ~168 GB index is traversed once; non-planner tensors (renderer DiTs,
/// T5, the dropped `lm_head`) are never retained.
fn extract_planner_groups(
    bernini_dir: &Path,
) -> Result<HashMap<&'static str, HashMap<String, Array>>> {
    let mut shards: Vec<PathBuf> = std::fs::read_dir(bernini_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    shards.sort();
    if shards.is_empty() {
        return Err(Error::Msg(format!(
            "assemble_bernini_planner_snapshot: no .safetensors shards under {} (point at the \
             `bernini/` dir of a ByteDance/Bernini-Diffusers snapshot)",
            bernini_dir.display()
        )));
    }
    let mut groups: HashMap<&'static str, HashMap<String, Array>> =
        COMPONENTS.iter().map(|c| (c.out, HashMap::new())).collect();
    for shard in &shards {
        let w = Weights::from_file(shard)?;
        for k in w.keys() {
            if let Some((out, key)) = route_key(k) {
                groups
                    .get_mut(out)
                    .expect("component group present")
                    .insert(key, w.require(k)?.clone());
            }
        }
    }
    Ok(groups)
}

/// Cast every tensor in `map` to `dtype` in place.
fn cast_map(map: &mut HashMap<String, Array>, dtype: Dtype) -> Result<()> {
    for v in map.values_mut() {
        if v.dtype() != dtype {
            *v = v.as_dtype(dtype)?;
        }
    }
    Ok(())
}

/// Materialize + write a key→Array map to `path`.
fn save_map(path: PathBuf, map: &HashMap<String, Array>) -> Result<()> {
    let arrays: Vec<&Array> = map.values().collect();
    eval(arrays)?;
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

fn write_json(path: PathBuf, v: &serde_json::Value) -> Result<()> {
    let text = serde_json::to_string_pretty(v)
        .map_err(|e| Error::Msg(format!("serialize {}: {e}", path.display())))?;
    std::fs::write(&path, text)?;
    Ok(())
}

/// Symlink (zero-copy) or recursively copy `src` → `dst`, replacing any existing target. Symlinks the
/// whole path (file or dir) when `link`; otherwise copies, following symlinks so the snapshot is
/// self-contained (the HF cache stores the package files as `../../blobs/*` symlinks).
fn place(src: &Path, dst: &Path, link: bool) -> Result<()> {
    if !src.exists() {
        return Err(Error::Msg(format!(
            "assemble_bernini_planner_snapshot: missing package component {}",
            src.display()
        )));
    }
    // `symlink_metadata` so we detect (and clear) an existing broken symlink too.
    if std::fs::symlink_metadata(dst).is_ok() {
        if dst.is_dir() && !dst.is_symlink() {
            std::fs::remove_dir_all(dst)?;
        } else {
            std::fs::remove_file(dst)?;
        }
    }
    if link {
        std::os::unix::fs::symlink(src, dst)?;
    } else if src.is_dir() {
        copy_dir_all(src, dst)?;
    } else {
        std::fs::copy(src, dst)?;
    }
    Ok(())
}

/// Recursively copy a directory, following symlinks (HF cache entries point into `../../blobs`).
fn copy_dir_all(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        // `metadata` follows symlinks, so a symlinked file/dir is classified by its target.
        if std::fs::metadata(&from)?.is_dir() {
            copy_dir_all(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// Read the package `config.json` and distill the planner knobs the downstream module/loader stories
/// consume, falling back to the upstream `BerniniConfig` defaults where a field is absent.
fn planner_knobs(pkg: &Path) -> serde_json::Value {
    use serde_json::json;
    let cfg: serde_json::Value = std::fs::read(pkg.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_else(|| json!({}));
    let f = |k: &str, d: f64| cfg.get(k).and_then(serde_json::Value::as_f64).unwrap_or(d);
    let b = |k: &str, d: bool| cfg.get(k).and_then(serde_json::Value::as_bool).unwrap_or(d);
    let i = |k: &str, d: i64| cfg.get(k).and_then(serde_json::Value::as_i64).unwrap_or(d);
    let obj = |k: &str| cfg.get(k).cloned().unwrap_or_else(|| json!({}));
    json!({
        // planner→renderer handoff + MAR loop
        "feature_type_from_stage_one":
            cfg.get("feature_type_from_stage_one")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("masked_tgt_embed_with_qwen_txt_vit_tokens"),
        "num_mask_token": i("num_mask_token", 4096),
        "max_sequence_length": i("max_sequence_length", 512),
        "use_src_id_rotary_emb": b("use_src_id_rotary_emb", true),
        "interpolate_src_id": b("interpolate_src_id", true),
        "max_trained_src_id": i("max_trained_src_id", 5),
        // dual-expert + flow schedule (full pipeline uses boundary_ratio for the planner stage,
        // switch_dit_boundary for the renderer expert switch — keep both)
        "boundary_ratio": f("boundary_ratio", 0.417),
        "switch_dit_boundary": f("switch_dit_boundary", 0.875),
        "shift": f("shift", 3.0),
        "target_fps": i("target_fps", 16),
        // T5 combine + lengths
        "t5_combine_type":
            cfg.get("t5_combine_type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("concat_with_zero_init"),
        "t5_max_sequence_length": i("t5_max_sequence_length", 512),
        // the clip-diff (vit_decoder) + connector module configs, passed through verbatim
        "clip_diff_cfg": obj("clip_diff_cfg"),
        "connector_cfg": obj("connector_cfg"),
    })
}

/// Assemble a native MLX **Bernini planner** snapshot from a full `ByteDance/Bernini-Diffusers`
/// package (sc-5144).
///
/// Emits:
///   - `qwen2_5_vl.safetensors`  ← `mllm.*` (minus `lm_head`) → bf16  (728 tensors)
///   - `connector.safetensors`   ← `connector.*` → bf16              (12)
///   - `vit_decoder.safetensors` ← `vit_decoder.*` → bf16            (140)
///   - `mask_tokens.safetensors` ← `mask_tokens` → bf16              (1)
///   - `qwen2_5_vl_config.json`  ← copy of `mllm/config.json`
///   - `bernini_planner.json`    ← distilled planner knobs ([`planner_knobs`])
///   - `transformer_config.json` / `transformer_2_config.json` ← copied (the renderer DiT configs)
///   - `t5_text_encoder/`, `t5_tokenizer/`, `vae/`, `scheduler/`, `mllm/` ← linked or copied verbatim
///
/// `link == true` symlinks the shared diffusers dirs/configs (zero-copy; the engine resolves
/// symlinks); `false` copies them (a portable, self-contained snapshot). The four extracted component
/// safetensors are always written fresh (F32 in the package → re-saved bf16). Idempotent: existing
/// targets are replaced. The exact per-component tensor counts are asserted — a count mismatch
/// (a re-layout in a future package revision) is a hard error, not a silent partial conversion.
pub fn assemble_bernini_planner_snapshot(
    out_dir: impl AsRef<Path>,
    bernini_diffusers_dir: impl AsRef<Path>,
    link: bool,
) -> Result<PathBuf> {
    let out_dir = out_dir.as_ref();
    let pkg = bernini_diffusers_dir.as_ref();

    let bernini_dir = pkg.join("bernini");
    if !bernini_dir.is_dir() {
        return Err(Error::Msg(format!(
            "assemble_bernini_planner_snapshot: no `bernini/` dir under {} (point at a full \
             ByteDance/Bernini-Diffusers snapshot root — the renderer-only -R package lacks the planner)",
            pkg.display()
        )));
    }

    std::fs::create_dir_all(out_dir)?;

    // 1. Extract all four planner groups in a single sweep over the shards.
    let mut groups = extract_planner_groups(&bernini_dir)?;

    // 2. Hard count guard, then cast bf16 + write each component (one group materialized at a time).
    for c in &COMPONENTS {
        let mut g = groups.remove(c.out).ok_or_else(|| {
            Error::Msg(format!(
                "assemble_bernini_planner_snapshot: component group {} not collected",
                c.out
            ))
        })?;
        if g.len() != c.expect {
            return Err(Error::Msg(format!(
                "assemble_bernini_planner_snapshot: {} expected {} tensors (prefix '{}'), got {} — \
                 the Bernini-Diffusers planner layout may have changed",
                c.out,
                c.expect,
                c.prefix,
                g.len()
            )));
        }
        cast_map(&mut g, Dtype::Bfloat16)?;
        save_map(out_dir.join(c.out), &g)?;
    }

    // 3. Configs: the Qwen2.5-VL config + the distilled planner knobs + the renderer DiT configs.
    place(
        &pkg.join("mllm").join("config.json"),
        &out_dir.join("qwen2_5_vl_config.json"),
        false, // small JSON — always copy so the snapshot is readable standalone
    )?;
    write_json(out_dir.join(BERNINI_PLANNER_SIDECAR), &planner_knobs(pkg))?;
    for name in ["transformer_config.json", "transformer_2_config.json"] {
        place(&pkg.join(name), &out_dir.join(name), false)?;
    }

    // 4. Shared diffusers components, linked (zero-copy) or copied (portable).
    for name in [
        "t5_text_encoder",
        "t5_tokenizer",
        "vae",
        "scheduler",
        "mllm",
    ] {
        place(&pkg.join(name), &out_dir.join(name), link)?;
    }

    Ok(out_dir.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// sc-5144: combined-index keys route to the correct planner component + stripped key, the
    /// renderer DiTs / redundant T5 are skipped, and `mllm.lm_head` is dropped.
    #[test]
    fn planner_key_routing() {
        let cases: &[(&str, Option<(&str, &str)>)] = &[
            // Qwen2.5-VL LLM + vision, prefix stripped.
            (
                "mllm.model.layers.0.self_attn.q_proj.weight",
                Some((
                    "qwen2_5_vl.safetensors",
                    "model.layers.0.self_attn.q_proj.weight",
                )),
            ),
            (
                "mllm.model.embed_tokens.weight",
                Some(("qwen2_5_vl.safetensors", "model.embed_tokens.weight")),
            ),
            (
                "mllm.model.norm.weight",
                Some(("qwen2_5_vl.safetensors", "model.norm.weight")),
            ),
            (
                "mllm.visual.blocks.0.attn.qkv.weight",
                Some(("qwen2_5_vl.safetensors", "visual.blocks.0.attn.qkv.weight")),
            ),
            (
                "mllm.visual.merger.mlp.0.weight",
                Some(("qwen2_5_vl.safetensors", "visual.merger.mlp.0.weight")),
            ),
            (
                "mllm.visual.patch_embed.proj.weight",
                Some(("qwen2_5_vl.safetensors", "visual.patch_embed.proj.weight")),
            ),
            // Connector branches.
            (
                "connector.proj_gen.0.weight",
                Some(("connector.safetensors", "proj_gen.0.weight")),
            ),
            (
                "connector.pred_vit.3.weight",
                Some(("connector.safetensors", "pred_vit.3.weight")),
            ),
            // ViT decoder (DiffLoss_FM) keeps the `net.` substructure.
            (
                "vit_decoder.net.cond_embed.weight",
                Some(("vit_decoder.safetensors", "net.cond_embed.weight")),
            ),
            (
                "vit_decoder.net.res_blocks.7.adaLN_modulation.1.weight",
                Some((
                    "vit_decoder.safetensors",
                    "net.res_blocks.7.adaLN_modulation.1.weight",
                )),
            ),
            (
                "vit_decoder.net.final_layer.linear.weight",
                Some(("vit_decoder.safetensors", "net.final_layer.linear.weight")),
            ),
            // The bare MAR mask-token parameter keeps its name.
            (
                "mask_tokens",
                Some(("mask_tokens.safetensors", "mask_tokens")),
            ),
            // Dropped / skipped.
            ("mllm.lm_head.weight", None),
            ("diff_dec.transformer.blocks.0.attn1.to_q.weight", None),
            ("diff_dec_low.transformer_2.patch_embedding.weight", None),
            ("t5_text_encoder.shared.weight", None),
        ];
        for (k, want) in cases {
            let got = route_key(k);
            let got_ref = got.as_ref().map(|(o, s)| (*o, s.as_str()));
            assert_eq!(got_ref, *want, "routing {k}");
        }
    }

    /// The four expected counts sum to the planner tensor total (728+12+140+1 = 881), and each prefix
    /// is distinct (no prefix shadows another in first-match routing).
    #[test]
    fn component_table_consistent() {
        let total: usize = COMPONENTS.iter().map(|c| c.expect).sum();
        assert_eq!(total, 881);
        for (i, a) in COMPONENTS.iter().enumerate() {
            for (j, b) in COMPONENTS.iter().enumerate() {
                if i != j {
                    assert!(
                        !a.prefix.starts_with(b.prefix),
                        "prefix '{}' shadows '{}'",
                        b.prefix,
                        a.prefix
                    );
                }
            }
        }
    }
}
