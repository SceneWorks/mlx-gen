//! SCAIL-2 **diff-patch** + cross-architecture LoRA install (sc-5684).
//!
//! sc-5451 wired the family-agnostic *residual* LoRA path: a standard `lora_down`/`lora_up` (+`alpha`)
//! file installs onto the DiT as forward-time residuals over the (possibly Q4/Q8) base. That covers
//! every SCAIL-2-native LoRA (the Bias-Aware DPO refinement LoRA, any adapter trained on SCAIL-2). It
//! does **not** cover the **lightx2v cross-architecture step-distill ("lightning") LoRAs**, which add
//! two things the residual loader can't consume:
//!
//!   1. **Diff-patch tensors.** Alongside the low-rank factors the file carries full-rank `.diff`
//!      (weight delta) and `.diff_b` (bias delta) tensors — including on layers the residual host never
//!      exposes as adapter targets: the qk-RMSNorms (`norm_q`/`norm_k`/`norm_k_img.diff`), the affine
//!      `norm3` / `img_emb.proj.{0,4}` LayerNorms, the output `head.head` (a full `.diff` rather than
//!      low-rank factors), and a `.diff_b` on **every** biased projection. This is the ComfyUI "diff
//!      patch" mechanism.
//!   2. **Cross-architecture shape mismatch.** The LoRA targets vanilla **Wan2.1-I2V-14B**, whose
//!      `patch_embedding` has in_dim 36; SCAIL-2's is in_dim **20** (16 z + 4 i2v mask) plus the extra
//!      `patch_embedding_{pose,mask}` stems. So `patch_embedding.diff` (`[5120, 36, 1, 2, 2]`) is
//!      shape-incompatible with SCAIL-2's `[5120, 20, 1, 2, 2]` and must be **deliberately skipped**.
//!      The transformer blocks (dim 5120 q/k/v/o/ffn/k_img/v_img), the dim-5120 globals, and the
//!      `img_emb` stack ARE compatible and DO transfer — only the input patch-embed differs.
//!
//! **Mechanism — in-place dense merge (option (a) from the story).** Rather than expose every norm /
//! bias as a residual adapter target, this merges the deltas directly into the raw [`Weights`] map
//! *before* the DiT is built and *before* load-time quantization: a `.diff` adds onto `{stem}.weight`,
//! a `.diff_b` adds onto `{stem}.bias`, and the low-rank factors fold `(alpha/rank)·(up·down)` onto
//! `{stem}.weight` — all uniformly. This composes with *load-time* Q4/Q8 (merge the dense weights,
//! then quantize) but **not** a pre-quantized-on-disk snapshot (packed u32 weights can't take a dense
//! delta) — the caller gates it to the dense bf16 snapshot. The lightning recipe is a *speed* lever
//! (8 steps, shift 1, CFG off) and 480p memory is activation-bound (the Q4 *weights* help little
//! there, per sc-5445), so requiring the dense base is the right trade.
//!
//! **Shape-aware skipping is loud, never silent.** A target whose weight-delta shape doesn't match
//! the SCAIL-2 base (the in_dim-36 `patch_embedding`) is skipped *as a whole module* — its coupled
//! bias delta `.diff_b` is dropped too, since it was trained jointly with the incompatible weight
//! delta — and surfaced in the report (and a file that matches nothing is a hard error). This is the
//! same "never half-apply a LoRA" contract the strict residual installer enforces.

use std::collections::BTreeMap;
use std::path::Path;

use mlx_gen::array::scalar;
use mlx_gen::gen_core::weightsmeta::{LoraAdapterMeta, LORA_ADAPTER_METADATA_KEY};
use mlx_gen::weights::Weights;
use mlx_gen::{AdapterSpec, Error, Result};
use mlx_rs::ops::{add, matmul, multiply};
use mlx_rs::{Array, Dtype};

/// LoRA key namespace prefixes stripped (longest-first). The lightx2v files use `diffusion_model.`;
/// SCAIL-2's converted DiT keys are the bare `SCAIL2Model` parameter names (no `ffn.0 → ffn.fc1`
/// rename — unlike the *converted* Wan checkpoint — because SCAIL-2 ships raw I2V module names).
const PREFIXES: [&str; 4] = [
    "model.diffusion_model.",
    "diffusion_model.",
    "base_model.model.",
    "model.",
];

#[derive(Clone, Copy)]
enum Role {
    Down,
    Up,
    Alpha,
    /// Full-rank weight delta (`.diff`).
    Diff,
    /// Full-rank bias delta (`.diff_b`).
    DiffB,
}

/// Factor / diff suffixes (exact match). `.diff_b` precedes `.diff` so a bias-delta key never mis-binds
/// as a weight delta; the kohya `lora_down`/`lora_up` and PEFT `lora_A`/`lora_B` conventions are both
/// accepted. A key matching none of these (a bundled base weight, say) is ignored.
const SUFFIXES: [(&str, Role); 7] = [
    (".lora_down.weight", Role::Down),
    (".lora_up.weight", Role::Up),
    (".lora_A.weight", Role::Down),
    (".lora_B.weight", Role::Up),
    (".alpha", Role::Alpha),
    (".diff_b", Role::DiffB),
    (".diff", Role::Diff),
];

/// The deltas a diff-patch file carries for one module.
#[derive(Default)]
struct Parts {
    down: Option<Array>,   // lora_A / lora_down → [rank, in]
    up: Option<Array>,     // lora_B / lora_up   → [out, rank]
    alpha: Option<f32>,    // per-target `.alpha` (rare in diff-patch files)
    diff: Option<Array>,   // full-rank weight delta, shape == base weight
    diff_b: Option<Array>, // full-rank bias delta, shape == base bias
}

/// What a diff-patch merge did: counts of merged weights/biases and the targets deliberately skipped
/// (cross-architecture shape mismatch) or absent from the SCAIL-2 checkpoint.
#[derive(Debug, Default)]
pub struct DiffPatchReport {
    pub merged_weights: usize,
    pub merged_biases: usize,
    /// Targets skipped because their weight-delta shape is incompatible with SCAIL-2 (the in_dim-36
    /// vanilla-Wan `patch_embedding`). Surfaced loudly — never silently dropped.
    pub skipped_cross_arch: Vec<String>,
    /// Targets that resolved to no weight in the SCAIL-2 checkpoint (orphan factor / unknown module).
    pub skipped_unmatched: Vec<String>,
}

/// `true` if `path` is a diff-patch ("lightning") LoRA — a file carrying any full-rank `.diff` (weight
/// delta) or `.diff_b` (bias delta), which the forward-time residual loader cannot consume.
pub fn has_diff_patch_keys(path: &Path) -> Result<bool> {
    let w = Weights::from_file(path)?;
    let found = w
        .keys()
        .any(|k| k.ends_with(".diff") || k.ends_with(".diff_b"));
    Ok(found)
}

/// Cast to f32 for the merge math (the deltas + base are folded in f32, then cast back to the base
/// dtype on write so the snapshot keeps its bf16 footprint).
fn f32a(x: &Array) -> Result<Array> {
    Ok(x.as_dtype(Dtype::Float32)?)
}

/// Read a scalar `.alpha` as f32 regardless of on-disk dtype.
fn read_alpha(a: &Array) -> Result<f32> {
    Ok(a.as_dtype(Dtype::Float32)?.as_slice::<f32>()[0])
}

fn strip_namespace(key: &str) -> &str {
    PREFIXES
        .iter()
        .find_map(|p| key.strip_prefix(p))
        .unwrap_or(key)
}

/// Merge every diff-patch adapter in `specs` into the **dense** SCAIL-2 weight map `w`, in place
/// (sc-5684). Call this on the freshly-loaded `dit.safetensors` weights *before* [`crate::Scail2Dit`]
/// is built and *before* any load-time quantization. Multiple files accumulate (each reads the
/// already-merged weight back from `w`). The caller must have verified `w` is the dense bf16 snapshot
/// (a pre-quantized-on-disk DiT can't take a dense delta).
pub fn merge_diff_patch_adapters(
    w: &mut Weights,
    specs: &[&AdapterSpec],
) -> Result<DiffPatchReport> {
    let mut report = DiffPatchReport::default();
    for spec in specs {
        merge_one(w, spec, &mut report)?;
    }
    Ok(report)
}

fn merge_one(w: &mut Weights, spec: &AdapterSpec, report: &mut DiffPatchReport) -> Result<()> {
    let lw = Weights::from_file(&spec.path)?;
    // Group every factor / diff key by its SCAIL-2 module path (namespace prefix stripped).
    let mut groups: BTreeMap<String, Parts> = BTreeMap::new();
    for key in lw.keys().map(str::to_string).collect::<Vec<_>>() {
        let Some((stem, role)) = SUFFIXES
            .iter()
            .find_map(|(suf, role)| key.strip_suffix(suf).map(|s| (s, *role)))
        else {
            continue; // not a factor / diff key — ignore.
        };
        let parts = groups.entry(strip_namespace(stem).to_string()).or_default();
        match role {
            Role::Down => parts.down = Some(lw.require(&key)?.clone()),
            Role::Up => parts.up = Some(lw.require(&key)?.clone()),
            Role::Alpha => parts.alpha = Some(read_alpha(lw.require(&key)?)?),
            Role::Diff => parts.diff = Some(lw.require(&key)?.clone()),
            Role::DiffB => parts.diff_b = Some(lw.require(&key)?.clone()),
        }
    }
    // PEFT/diffusers `save_lora_adapter` scaling lives in the `lora_adapter_metadata` blob (sc-5513);
    // `None` for a diff-patch file (the lightx2v files carry no blob and no `.alpha` → alpha = rank →
    // scale 1.0, the canonical lightning composition).
    let meta = LoraAdapterMeta::from_metadata(lw.metadata(LORA_ADAPTER_METADATA_KEY));
    for (stem, parts) in groups {
        merge_module(w, &stem, &parts, meta.as_ref(), spec.scale, report)?;
    }
    Ok(())
}

/// Fold one module's deltas into `w`. Weight delta = `strength·diff + (alpha/rank)·strength·(up·down)`
/// (whichever the file carries), bias delta = `strength·diff_b` — accumulated in f32, written back at
/// the base dtype. A weight-delta shape that doesn't match the SCAIL-2 base means a cross-architecture
/// target: skip the whole module (weight AND its coupled bias) and record it.
fn merge_module(
    w: &mut Weights,
    stem: &str,
    parts: &Parts,
    meta: Option<&LoraAdapterMeta>,
    strength: f32,
    report: &mut DiffPatchReport,
) -> Result<()> {
    let wkey = format!("{stem}.weight");
    let Some(base_w) = w.get(&wkey).cloned() else {
        report.skipped_unmatched.push(stem.to_string());
        return Ok(());
    };
    let base_shape = base_w.shape().to_vec();

    // --- weight delta (f32): full-rank `.diff` (@ strength) + low-rank `up·down` (@ alpha/rank·strength)
    let mut wdelta: Option<Array> = None;
    if let Some(diff) = &parts.diff {
        if diff.shape() != base_shape.as_slice() {
            report.skipped_cross_arch.push(stem.to_string());
            return Ok(()); // cross-arch (e.g. patch_embedding in_dim 36 vs 20) — skip whole module.
        }
        wdelta = Some(multiply(&f32a(diff)?, scalar(strength))?);
    }
    match (&parts.down, &parts.up) {
        (Some(down), Some(up)) => {
            // alpha precedence: per-target `.alpha` → blob `alpha_pattern`/`lora_alpha` → factor rank.
            let (cfg_alpha, cfg_rank) = meta.map_or((None, None), |m| m.effective(stem));
            let rank = cfg_rank.map(|r| r as f64).unwrap_or(down.shape()[0] as f64);
            let alpha = parts.alpha.or(cfg_alpha).map(|a| a as f64).unwrap_or(rank);
            let eff = (alpha / rank * strength as f64) as f32;
            let delta = matmul(&f32a(up)?, &f32a(down)?)?; // [out, in]
            if delta.shape() != base_shape.as_slice() {
                report.skipped_cross_arch.push(stem.to_string());
                return Ok(());
            }
            let delta = multiply(&delta, scalar(eff))?;
            wdelta = Some(match wdelta {
                Some(d) => add(&d, &delta)?,
                None => delta,
            });
        }
        (None, None) => {}
        _ => {
            // An orphan low-rank factor (its partner targeted a non-LoRA key) — surface, don't fold.
            report.skipped_unmatched.push(stem.to_string());
            return Ok(());
        }
    }
    if let Some(d) = wdelta {
        let merged = add(&f32a(&base_w)?, &d)?.as_dtype(base_w.dtype())?;
        w.insert(wkey, merged);
        report.merged_weights += 1;
    }

    // --- bias delta (`.diff_b`, @ strength) ---
    if let Some(diff_b) = &parts.diff_b {
        let bkey = format!("{stem}.bias");
        let Some(base_b) = w.get(&bkey).cloned() else {
            report.skipped_unmatched.push(bkey);
            return Ok(());
        };
        if diff_b.shape() != base_b.shape() {
            report.skipped_cross_arch.push(bkey);
            return Ok(());
        }
        let bd = multiply(&f32a(diff_b)?, scalar(strength))?;
        let merged = add(&f32a(&base_b)?, &bd)?.as_dtype(base_b.dtype())?;
        w.insert(bkey, merged);
        report.merged_biases += 1;
    }
    Ok(())
}

/// Surface a diff-patch merge's skips loudly (the only channel at load time — there is no `Progress`
/// callback yet), and error if the file(s) matched *nothing* (a format/prefix misconfiguration that
/// would otherwise silently no-op). Mirrors the wan loader's `warn_skipped_adapters` + "matched
/// nothing" contract.
pub fn report_outcome(report: &DiffPatchReport, model_id: &str) -> Result<()> {
    if !report.skipped_cross_arch.is_empty() {
        eprintln!(
            "{model_id}: lightx2v diff-patch LoRA — {} cross-architecture target(s) deliberately \
             skipped (shape-incompatible with SCAIL-2, e.g. the in_dim-36 vanilla-Wan2.1-I2V \
             patch_embedding vs SCAIL-2's in_dim 20): {:?}",
            report.skipped_cross_arch.len(),
            report.skipped_cross_arch
        );
    }
    if !report.skipped_unmatched.is_empty() {
        eprintln!(
            "{model_id}: lightx2v diff-patch LoRA — {} target(s) not present in the SCAIL-2 \
             checkpoint, skipped: {:?}",
            report.skipped_unmatched.len(),
            report.skipped_unmatched
        );
    }
    if report.merged_weights + report.merged_biases == 0 {
        return Err(Error::Msg(format!(
            "{model_id}: the diff-patch LoRA matched no SCAIL-2 module (every target skipped) — \
             likely a format / prefix mismatch, or the wrong base model"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{all_close, array_eq};
    use std::path::PathBuf;

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("mlx_gen_scail2_lora_test");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn f32(values: Vec<f32>, shape: &[i32]) -> Array {
        Array::from_slice(&values, shape)
            .as_dtype(Dtype::Bfloat16)
            .unwrap()
    }

    /// A small SCAIL-2-shaped dense weight map: one block projection (Linear + bias), one qk-RMSNorm
    /// (weight only), and a `patch_embedding` Conv stem (5-D weight + bias) standing in for the
    /// cross-architecture target — all the distinct diff-patch module shapes.
    fn synthetic_dit() -> Weights {
        let path = tmp("dit.safetensors");
        let q_w = f32(
            (0..16 * 8).map(|i| i as f32 * 0.01 - 0.3).collect(),
            &[16, 8],
        );
        let q_b = f32((0..16).map(|i| i as f32 * 0.02).collect(), &[16]);
        let norm_w = f32(vec![1.0; 8], &[8]);
        // patch_embedding Conv3d weight [out, in=4, 1, 2, 2] + bias [out].
        let pe_w = f32(
            (0..16 * 4 * 4).map(|i| i as f32 * 0.001).collect(),
            &[16, 4, 1, 2, 2],
        );
        let pe_b = f32((0..16).map(|i| i as f32 * 0.03).collect(), &[16]);
        Array::save_safetensors(
            vec![
                ("blocks.0.self_attn.q.weight", &q_w),
                ("blocks.0.self_attn.q.bias", &q_b),
                ("blocks.0.self_attn.norm_q.weight", &norm_w),
                ("patch_embedding.weight", &pe_w),
                ("patch_embedding.bias", &pe_b),
            ],
            None,
            &path,
        )
        .unwrap();
        Weights::from_file(&path).unwrap()
    }

    /// A diff-patch LoRA over the synthetic DiT: the q projection gets low-rank factors + a `.diff_b`
    /// bias delta; norm_q gets a full-rank `.diff`; patch_embedding gets a shape-INCOMPATIBLE `.diff`
    /// (in_dim 6 vs 4) + a (shape-compatible) `.diff_b` — the cross-architecture case.
    fn write_diff_patch(name: &str) -> PathBuf {
        let path = tmp(name);
        let rank = 4;
        let down = f32(
            (0..rank * 8)
                .map(|i| (i as f32 * 0.01).sin() * 0.1)
                .collect(),
            &[rank, 8],
        );
        let up = f32(
            (0..16 * rank)
                .map(|i| (i as f32 * 0.02).cos() * 0.1)
                .collect(),
            &[16, rank],
        );
        let q_diff_b = f32((0..16).map(|i| i as f32 * 0.005).collect(), &[16]);
        let norm_diff = f32((0..8).map(|i| i as f32 * 0.01).collect(), &[8]);
        // in_dim 6 ≠ the base's 4 → must be skipped as cross-architecture.
        let pe_diff = f32(
            (0..16 * 6 * 4).map(|i| i as f32 * 0.001).collect(),
            &[16, 6, 1, 2, 2],
        );
        let pe_diff_b = f32((0..16).map(|i| i as f32 * 0.04).collect(), &[16]);
        Array::save_safetensors(
            vec![
                (
                    "diffusion_model.blocks.0.self_attn.q.lora_down.weight",
                    &down,
                ),
                ("diffusion_model.blocks.0.self_attn.q.lora_up.weight", &up),
                ("diffusion_model.blocks.0.self_attn.q.diff_b", &q_diff_b),
                ("diffusion_model.blocks.0.self_attn.norm_q.diff", &norm_diff),
                ("diffusion_model.patch_embedding.diff", &pe_diff),
                ("diffusion_model.patch_embedding.diff_b", &pe_diff_b),
            ],
            None,
            &path,
        )
        .unwrap();
        path
    }

    fn spec(path: PathBuf, scale: f32) -> AdapterSpec {
        AdapterSpec::new(path, scale, mlx_gen::AdapterKind::Lora)
    }

    #[test]
    fn detects_diff_patch_file() {
        let dp = write_diff_patch("detect.safetensors");
        assert!(has_diff_patch_keys(&dp).unwrap());
        // A pure low-rank file (no .diff/.diff_b) is NOT a diff-patch file.
        let plain = tmp("plain.safetensors");
        let down = f32(vec![0.1; 4 * 8], &[4, 8]);
        let up = f32(vec![0.1; 16 * 4], &[16, 4]);
        Array::save_safetensors(
            vec![
                (
                    "diffusion_model.blocks.0.self_attn.q.lora_down.weight",
                    &down,
                ),
                ("diffusion_model.blocks.0.self_attn.q.lora_up.weight", &up),
            ],
            None,
            &plain,
        )
        .unwrap();
        assert!(!has_diff_patch_keys(&plain).unwrap());
    }

    #[test]
    fn merges_lora_diff_and_diffb_skips_cross_arch() {
        let dp = write_diff_patch("merge.safetensors");
        let mut w = synthetic_dit();
        let report = merge_diff_patch_adapters(&mut w, &[&spec(dp.clone(), 1.0)]).unwrap();

        // q.weight (lora) + norm_q.weight (diff) merged; q.bias + norm... only q has diff_b → 1 bias.
        assert_eq!(report.merged_weights, 2, "q (lora) + norm_q (diff)");
        assert_eq!(report.merged_biases, 1, "q.diff_b");
        // patch_embedding is the lone cross-architecture skip (its .diff is in_dim 6 vs base 4); its
        // .diff_b is dropped with it even though [16] would have matched.
        assert_eq!(
            report.skipped_cross_arch,
            vec!["patch_embedding".to_string()]
        );
        assert!(report.skipped_unmatched.is_empty());

        // patch_embedding stays bit-identical (skipped entirely — weight AND bias).
        let base = synthetic_dit();
        for k in ["patch_embedding.weight", "patch_embedding.bias"] {
            assert!(
                array_eq(w.require(k).unwrap(), base.require(k).unwrap(), false)
                    .unwrap()
                    .item::<bool>(),
                "{k} must be untouched (cross-arch skip)"
            );
        }

        // q.weight == base + (up·down) (alpha = rank → scale 1.0), folded in f32.
        let lw = Weights::from_file(&dp).unwrap();
        let down = lw
            .require("diffusion_model.blocks.0.self_attn.q.lora_down.weight")
            .unwrap();
        let up = lw
            .require("diffusion_model.blocks.0.self_attn.q.lora_up.weight")
            .unwrap();
        let delta = matmul(f32a(up).unwrap(), f32a(down).unwrap()).unwrap();
        let q_base = base.require("blocks.0.self_attn.q.weight").unwrap();
        let want = add(f32a(q_base).unwrap(), &delta)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        assert!(
            all_close(
                w.require("blocks.0.self_attn.q.weight").unwrap(),
                &want,
                1e-4,
                1e-4,
                false
            )
            .unwrap()
            .item::<bool>(),
            "merged q.weight must equal base + up·down"
        );
        // norm_q.weight changed (a diff was applied).
        assert!(
            !array_eq(
                w.require("blocks.0.self_attn.norm_q.weight").unwrap(),
                base.require("blocks.0.self_attn.norm_q.weight").unwrap(),
                false
            )
            .unwrap()
            .item::<bool>(),
            "norm_q.weight must be patched by its .diff"
        );
    }

    #[test]
    fn scale_zero_is_noop() {
        let dp = write_diff_patch("zero.safetensors");
        let mut w = synthetic_dit();
        let base = synthetic_dit();
        let report = merge_diff_patch_adapters(&mut w, &[&spec(dp, 0.0)]).unwrap();
        // Still "merged" (folded a zero delta), but every touched weight is bit-identical to the base.
        assert_eq!(report.merged_weights, 2);
        for k in [
            "blocks.0.self_attn.q.weight",
            "blocks.0.self_attn.q.bias",
            "blocks.0.self_attn.norm_q.weight",
        ] {
            assert!(
                all_close(
                    w.require(k).unwrap(),
                    base.require(k).unwrap(),
                    1e-3,
                    1e-3,
                    false
                )
                .unwrap()
                .item::<bool>(),
                "{k} must be ~unchanged at strength 0"
            );
        }
    }

    #[test]
    fn report_errors_when_nothing_matched() {
        // A diff-patch file whose only target isn't in the checkpoint → matched-nothing error.
        let path = tmp("nomatch.safetensors");
        let diff = f32(vec![0.1; 8], &[8]);
        Array::save_safetensors(
            vec![("diffusion_model.blocks.99.unknown.diff", &diff)],
            None,
            &path,
        )
        .unwrap();
        let mut w = synthetic_dit();
        let report = merge_diff_patch_adapters(&mut w, &[&spec(path, 1.0)]).unwrap();
        assert_eq!(report.merged_weights, 0);
        assert_eq!(
            report.skipped_unmatched,
            vec!["blocks.99.unknown".to_string()]
        );
        assert!(report_outcome(&report, "scail2_14b").is_err());
    }
}
