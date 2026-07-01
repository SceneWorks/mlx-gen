//! sc-3192: the 8-step **distill LoRA** merge for the `sensenova_u1_8b_fast` variant.
//!
//! The reference ships an 8-NFE preview as a LoRA over the base checkpoint
//! (`sensenova/SenseNova-U1-8B-MoT-LoRAs` → `SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors`) that
//! is merged at load and then run at `cfg_scale=1.0` / `timestep_shift=3.0` / `num_steps=8`
//! (`docs/base_vs_distill.md`). The reference merge is `examples/t2i/inference.py` →
//! `utils/lora.py::load_and_merge_lora_weight`: for every base parameter `…W.weight` that has a
//! matching `…W.lora_down.weight` / `…W.lora_up.weight` / `…W.alpha`, add
//! `Δ = (alpha/rank)·(up @ down)` into the weight (`value += Δ`, accumulated in f32).
//!
//! The distill LoRA touches **only** the generation path — every layer's `*_mot_gen` attention
//! projections (`{q,k,v,o}_proj_mot_gen`) and SwiGLU (`mlp_mot_gen.{gate,up,down}_proj`), plus the
//! two FM-head Linears (`fm_modules.fm_head.{0,2}`) — 7·layers + 2 targets. The understanding path
//! is untouched (so VQA / it2i conditioning is unchanged by the fast variant).
//!
//! [`lora_delta`] computes one target's `[out,in]` delta; the merge is applied through the core
//! [`mlx_gen::adapters::AdaptableLinear::merge_dense_delta`] seam (gen-path projections) and a plain
//! weight add (the dense FM-head Linears), and must run **before** any Q4/Q8 quantization (the merge
//! seam errors on a quantized base, matching the reference which merges into the dense weight).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use mlx_rs::ops::{add, matmul, multiply};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

/// Marker file dropped into a **pre-merged** `sensenova_u1_8b_fast` tier (sc-8775): its presence tells
/// [`crate::model::load_fast`] the 8-step distill LoRA is already baked into the on-disk weights, so
/// the loader must NOT merge again (and cannot — a packed tier's base is quantized). Absent ⇒ the
/// legacy path (a dense base snapshot + the distill LoRA merged at load). The JSON body is provenance
/// only; the loader keys off existence.
pub const DISTILL_MERGED_MARKER: &str = "distill_merged.json";

/// The distill LoRA file name (the `--include` argument the reference docs download).
pub const DISTILL_LORA_FILE: &str = "SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors";
/// The HF Hub repo the distill LoRA ships in (for the not-found error hint).
pub const DISTILL_LORA_REPO: &str = "sensenova/SenseNova-U1-8B-MoT-LoRAs";

/// Read a scalar adapter value (the per-module `alpha`) as `f32` regardless of its on-disk dtype.
/// The distill LoRA stores `alpha` as an `I32` scalar; cast to f32 first (`as_slice::<f32>` never
/// casts and would panic on a dtype mismatch). Mirrors the core loader's `scalar_alpha`.
fn scalar_f32(a: &Array) -> Result<f32> {
    a.as_dtype(Dtype::Float32)?
        .as_slice::<f32>()
        .first()
        .copied()
        .ok_or_else(|| Error::Msg("distill LoRA: empty alpha scalar".into()))
}

/// Compute the `[out, in]` merge delta for `target` (the base weight key **without** its `.weight`
/// suffix, e.g. `…self_attn.q_proj_mot_gen`), or `None` if the LoRA does not carry that target.
///
/// `Δ = (alpha/rank)·(up @ down)` in f32 (the reference asserts the factors are f32 and does the
/// matmul + scale in f32), where `down` is `[rank, in]`, `up` is `[out, rank]`, and `rank` is
/// `down.shape[0]`. The caller casts the delta to the base weight's dtype at the merge site.
pub fn lora_delta(lora: &Weights, target: &str) -> Result<Option<Array>> {
    let down = match lora.get(&format!("{target}.lora_down.weight")) {
        Some(a) => a,
        None => return Ok(None),
    };
    let up = lora.require(&format!("{target}.lora_up.weight"))?;
    let alpha = scalar_f32(lora.require(&format!("{target}.alpha"))?)?;
    let rank = down.shape()[0] as f32;
    if rank == 0.0 {
        // Zero rank (empty/malformed down factor) → non-finite scaling → NaN-poisoned GEN-path
        // merge that silently corrupts every generation. Reject instead (sc-5252/F-002).
        return Err(Error::Msg(format!(
            "distill LoRA: zero-rank factor at '{target}'"
        )));
    }
    let scaling = alpha / rank;
    // f32 matmul + scale (reference `scaling_factor * torch.matmul(lora_up, lora_down)`).
    let down = down.as_dtype(Dtype::Float32)?;
    let up = up.as_dtype(Dtype::Float32)?;
    let delta = multiply(&matmul(&up, &down)?, Array::from_f32(scaling))?;
    Ok(Some(delta))
}

/// Merge every target the distill LoRA carries **directly into a flat checkpoint map** (the offline
/// pre-merge for `sensenova_u1_8b_fast`'s packed tiers, sc-8775) — the map-level analogue of the
/// load-time [`crate::t2i::T2iModel::merge_distill_lora`]. For each LoRA target (a key with a
/// `…{target}.lora_down.weight` sibling — the gen-path `{q,k,v,o}_proj_mot_gen` /
/// `mlp_mot_gen.{gate,up,down}_proj` projections and the two `fm_modules.fm_head.{0,2}` Linears), add
/// `Δ = (alpha/rank)·(up @ down)` into `map["{target}.weight"]` as `W = W + Δ.as_dtype(W.dtype)` —
/// **byte-identical** to both [`mlx_gen::adapters::AdaptableLinear::merge_dense_delta`] (the backbone
/// projections) and the FM-head's hand-rolled equivalent, so a pre-merged-then-packed tier equals
/// load-time-merge-then-quantize. Runs on the **dense** map *before* [`mlx_gen::quant::quantize_map`]
/// (the same ordering the loader enforces: merge the dense weight, then quantize).
///
/// Errors if the LoRA names a target the checkpoint lacks (a stale/mismatched LoRA), so a wrong file
/// fails loudly rather than silently merging a subset. Returns the number of targets merged; the
/// caller asserts full coverage (`7 · num_hidden_layers + 2`) against the config.
pub fn merge_distill_into_map(map: &mut HashMap<String, Array>, lora: &Weights) -> Result<usize> {
    // Target bases = every LoRA key stripped of its `.lora_down.weight` tail. Collect first so we are
    // not iterating `lora` while borrowing it through `lora_delta` (and the order is irrelevant — the
    // adds are independent, one per distinct target weight).
    let targets: Vec<String> = lora
        .keys()
        .filter_map(|k| k.strip_suffix(".lora_down.weight").map(str::to_string))
        .collect();
    let mut applied = 0usize;
    for target in targets {
        let Some(delta) = lora_delta(lora, &target)? else {
            continue;
        };
        let key = format!("{target}.weight");
        let base = map.get(&key).ok_or_else(|| {
            Error::Msg(format!(
                "distill merge: LoRA target '{target}' has no base weight '{key}' in the checkpoint \
                 (stale/mismatched distill LoRA?)"
            ))
        })?;
        let merged = add(base, &delta.as_dtype(base.dtype())?)?;
        map.insert(key, merged);
        applied += 1;
    }
    Ok(applied)
}

/// Resolve the distill LoRA `.safetensors` for the `fast` variant. Resolution order:
/// 1. `$SENSENOVA_DISTILL_LORA` (explicit override / CI),
/// 2. co-located in the base snapshot `root`,
/// 3. the standard HF Hub cache (`$HF_HUB_CACHE`, `$HF_HOME/hub`, or `~/.cache/huggingface/hub`).
///
/// Errors with a download hint if none resolve — the fast variant never silently falls back to the
/// un-merged base.
pub fn resolve_distill_lora(root: &Path) -> Result<PathBuf> {
    if let Ok(p) = std::env::var("SENSENOVA_DISTILL_LORA") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Ok(p);
        }
        return Err(Error::Msg(format!(
            "SENSENOVA_DISTILL_LORA={} does not exist",
            p.display()
        )));
    }
    let co_located = root.join(DISTILL_LORA_FILE);
    if co_located.exists() {
        return Ok(co_located);
    }
    if let Some(p) = hf_cache_distill_lora() {
        return Ok(p);
    }
    Err(Error::Msg(format!(
        "sensenova_u1_8b_fast: distill LoRA `{DISTILL_LORA_FILE}` not found. Download it \
         (`huggingface-cli download {DISTILL_LORA_REPO} --include {DISTILL_LORA_FILE}`) or set \
         SENSENOVA_DISTILL_LORA to its path."
    )))
}

/// Locate `DISTILL_LORA_FILE` under the HF Hub cache for [`DISTILL_LORA_REPO`], scanning each
/// `snapshots/<rev>/` directory. Honours `$HF_HUB_CACHE` and `$HF_HOME` before the `~/.cache`
/// default (the layout `huggingface_hub` itself uses).
fn hf_cache_distill_lora() -> Option<PathBuf> {
    let repo_dir = format!("models--{}", DISTILL_LORA_REPO.replace('/', "--"));
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(c) = std::env::var("HF_HUB_CACHE") {
        roots.push(PathBuf::from(c));
    }
    if let Ok(h) = std::env::var("HF_HOME") {
        roots.push(PathBuf::from(h).join("hub"));
    }
    if let Ok(home) = std::env::var("HOME") {
        roots.push(PathBuf::from(home).join(".cache/huggingface/hub"));
    }
    for snapshots in roots
        .into_iter()
        .map(|r| r.join(&repo_dir).join("snapshots"))
    {
        let Ok(revs) = std::fs::read_dir(&snapshots) else {
            continue;
        };
        for rev in revs.filter_map(|e| e.ok()) {
            let cand = rev.path().join(DISTILL_LORA_FILE);
            if cand.exists() {
                return Some(cand);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::eq;

    fn byte_equal(a: &Array, b: &Array) -> bool {
        a.shape() == b.shape()
            && a.dtype() == b.dtype()
            && eq(a, b).unwrap().all(None).unwrap().item::<bool>()
    }

    /// A minimal LoRA carrying one target: `down` `[rank,in]`, `up` `[out,rank]`, an `I32` `alpha`
    /// scalar (the on-disk dtype the real distill LoRA uses).
    fn lora_with_target(target: &str, down: Array, up: Array, alpha: i32) -> Weights {
        let mut w = Weights::empty();
        w.insert(format!("{target}.lora_down.weight"), down);
        w.insert(format!("{target}.lora_up.weight"), up);
        w.insert(format!("{target}.alpha"), Array::from_int(alpha));
        w
    }

    /// The flat-map merge reproduces `Δ = (alpha/rank)·(up@down)` added into the base weight, and is
    /// byte-identical to the load-time `merge_dense_delta` (`W + Δ.as_dtype(W.dtype)`, bf16 base) —
    /// the sc-8775 pre-merge guarantee: pre-merge-on-disk == merge-at-load. Coverage count is returned.
    #[test]
    fn merge_into_map_matches_load_time_delta_byte_identical() {
        let (out, rank, in_dim) = (8, 2, 4);
        let down = Array::from_slice(
            &(0..rank * in_dim)
                .map(|i| (i as f32).sin())
                .collect::<Vec<_>>(),
            &[rank, in_dim],
        );
        let up = Array::from_slice(
            &(0..out * rank)
                .map(|i| (i as f32).cos())
                .collect::<Vec<_>>(),
            &[out, rank],
        );
        let base = Array::from_slice(
            &(0..out * in_dim)
                .map(|i| (i as f32 * 0.1).tanh())
                .collect::<Vec<_>>(),
            &[out, in_dim],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();

        let target = "language_model.model.layers.0.self_attn.q_proj_mot_gen";
        let alpha = 8;
        let lora = lora_with_target(target, down.clone(), up.clone(), alpha);

        let key = format!("{target}.weight");
        let mut map: HashMap<String, Array> = HashMap::new();
        map.insert(key.clone(), base.clone());

        let applied = merge_distill_into_map(&mut map, &lora).unwrap();
        assert_eq!(applied, 1, "one target merged");

        // Expected: the exact op `lora_delta` + `merge_dense_delta` run at load.
        let delta = lora_delta(&lora, target).unwrap().unwrap();
        let expected = add(&base, delta.as_dtype(base.dtype()).unwrap()).unwrap();
        assert!(
            byte_equal(map.get(&key).unwrap(), &expected),
            "flat-map merge != load-time merge_dense_delta"
        );
        // Manual cross-check of the delta itself: (alpha/rank)·(up@down).
        let manual = multiply(
            matmul(&up, &down).unwrap(),
            Array::from_f32(alpha as f32 / rank as f32),
        )
        .unwrap();
        assert!(
            byte_equal(&delta, &manual),
            "delta != (alpha/rank)·(up@down)"
        );
    }

    /// A LoRA target with no matching base weight in the checkpoint is a hard error (stale/mismatched
    /// LoRA), not a silent skip.
    #[test]
    fn merge_into_map_errors_on_missing_base() {
        let lora = lora_with_target(
            "fm_modules.fm_head.0",
            Array::ones::<f32>(&[2, 4]).unwrap(),
            Array::ones::<f32>(&[8, 2]).unwrap(),
            8,
        );
        let mut map: HashMap<String, Array> = HashMap::new(); // empty — base absent
        let err = merge_distill_into_map(&mut map, &lora)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no base weight"), "got: {err}");
    }
}
