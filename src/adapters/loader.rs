//! Adapter-file loaders — read a trained LoRA/LoKr `.safetensors` and install it onto a
//! model tree via [`AdaptableHost`]. Closes out sc-2343's loader piece.
//!
//! **LoKr** is generic and faithfully ported from the fork's `LoKrLoader.apply`: keys are
//! bare module paths (`‹path›.lokr_w1`/`lokr_w2`, full or low-rank `_a`/`_b`) and the file
//! carries `networkType=lokr` + `alpha`/`rank` in safetensors metadata, so the delta and
//! target path are fully determined by the file — no per-model mapping table.
//!
//! **LoRA** here covers the PEFT bare-path convention (`‹prefix›‹path›.lora_A/B.weight` +
//! optional `‹path›.alpha`). The fork's *other* LoRA path — remapping diffusers/kohya key
//! conventions through per-model `LoRATarget` pattern tables — is model-specific and lands
//! with each model port (per ARCHITECTURE.md: model-specific orchestration lives with the
//! model), not in this generic framework.

use std::collections::BTreeMap;

use mlx_rs::Array;

use super::{reconstruct_lokr_delta, AdaptableHost, Adapter};
use crate::runtime::{AdapterKind, AdapterSpec};
use crate::weights::Weights;
use crate::Result;

/// PEFT LoKr per-module factor suffixes; each factor is full (`lokr_w1`/`lokr_w2`) or
/// low-rank (`_a` @ `_b`). Exact-suffix matched, so order is for readability only.
const LOKR_SUFFIXES: [&str; 6] = [
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w1",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_w2",
];

/// `true` if the file's `networkType` metadata marks it a LoKr adapter.
pub fn is_lokr(w: &Weights) -> bool {
    w.metadata("networkType")
        .map(|s| s.trim().eq_ignore_ascii_case("lokr"))
        .unwrap_or(false)
}

/// Outcome of installing an adapter file: how many target modules were adapted, and any
/// adapter keys that matched no module in the host (surfaced, never silently dropped).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ApplyReport {
    pub applied: usize,
    pub unmatched_paths: Vec<String>,
}

/// Install a LoKr adapter file onto `host`. `scale` is the user-facing strength (the
/// `alpha/rank` factor is baked into the reconstructed delta, mirroring the fork).
pub fn apply_lokr(host: &mut impl AdaptableHost, w: &Weights, scale: f32) -> Result<ApplyReport> {
    let rank = w
        .metadata("rank")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.0);
    // alpha defaults to rank (scale 1.0) when absent, matching PEFT.
    let alpha = w
        .metadata("alpha")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(rank);

    // Group every lokr_* tensor by the module path preceding the suffix.
    let keys: Vec<String> = w.keys().map(str::to_string).collect();
    let mut grouped: BTreeMap<String, BTreeMap<&str, &Array>> = BTreeMap::new();
    for key in &keys {
        for suffix in LOKR_SUFFIXES {
            if let Some(path) = key.strip_suffix(suffix) {
                let factor = &suffix[1..]; // drop the leading '.'
                grouped
                    .entry(path.to_string())
                    .or_default()
                    .insert(factor, w.require(key)?);
                break;
            }
        }
    }

    let mut report = ApplyReport::default();
    for (path, factors) in grouped {
        let parts: Vec<&str> = path.split('.').collect();
        match host.adaptable_mut(&parts) {
            Some(lin) => {
                let base_shape = lin.base_shape();
                let delta = reconstruct_lokr_delta(
                    alpha,
                    rank,
                    &base_shape,
                    factors.get("lokr_w1").copied(),
                    factors.get("lokr_w1_a").copied(),
                    factors.get("lokr_w1_b").copied(),
                    factors.get("lokr_w2").copied(),
                    factors.get("lokr_w2_a").copied(),
                    factors.get("lokr_w2_b").copied(),
                )?;
                lin.push(Adapter::Lokr { delta, scale });
                report.applied += 1;
            }
            None => report.unmatched_paths.push(path),
        }
    }
    Ok(report)
}

/// Install a PEFT-format LoRA file (`‹prefix›‹path›.lora_A.weight` / `.lora_B.weight`, with
/// optional `‹prefix›‹path›.alpha`) onto `host`. PEFT stores `lora_A: [r, in]`,
/// `lora_B: [out, r]`; we transpose to the residual form `x·A·B` (`A: [in, r]`, `B: [r, out]`)
/// and fold `alpha/rank` into `B`, matching the fork. `strip_prefix` removes a leading
/// namespace such as `"base_model.model."` or `"transformer."`.
pub fn apply_lora_peft(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
    strip_prefix: Option<&str>,
) -> Result<ApplyReport> {
    let prefix = strip_prefix.unwrap_or("");
    let mut groups: BTreeMap<String, LoraParts> = BTreeMap::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        // `lora_A/B` always carry the file's namespace prefix.
        if let Some(rest) = key.strip_prefix(prefix) {
            if let Some(path) = rest.strip_suffix(".lora_A.weight") {
                groups.entry(path.to_string()).or_default().a = Some(w.require(&key)?.clone());
                continue;
            }
            if let Some(path) = rest.strip_suffix(".lora_B.weight") {
                groups.entry(path.to_string()).or_default().b = Some(w.require(&key)?.clone());
                continue;
            }
        }
        // `alpha` may be prefixed (`<prefix><path>.alpha`) OR bare (`<path>.alpha`): some trainers
        // pair prefixed `lora_A/B` with a bare `alpha` — notably the fork's `QwenLoRAMapping`, whose
        // alpha patterns are bare-only. Resolve to the same `<path>` either way (rather than
        // stripping the A/B prefix off the alpha key and dropping a bare one) so the `alpha/rank`
        // fold is kept; a prefixed and a bare alpha that *disagree* for one path is a hard error (no
        // silent pick). Without this, a prefixed-A/B + bare-alpha file applied at the wrong
        // (unscaled) strength while reporting success (sc-2528 adversarial review).
        if let Some(path) = key
            .strip_prefix(prefix)
            .and_then(|r| r.strip_suffix(".alpha"))
            .or_else(|| key.strip_suffix(".alpha"))
        {
            if let Some(new) = w.require(&key)?.as_slice::<f32>().first().copied() {
                let slot = &mut groups.entry(path.to_string()).or_default().alpha;
                match *slot {
                    Some(existing) if existing != new => {
                        return Err(format!(
                            "LoRA alpha conflict for `{path}`: {existing} vs {new} \
                             (prefixed and bare alpha keys disagree)"
                        )
                        .into());
                    }
                    _ => *slot = Some(new),
                }
            }
        }
    }

    let mut report = ApplyReport::default();
    for (path, parts) in groups {
        let (Some(a_raw), Some(b_raw)) = (parts.a, parts.b) else {
            continue;
        };
        let parents: Vec<&str> = path.split('.').collect();
        match host.adaptable_mut(&parents) {
            Some(lin) => {
                let a = a_raw.t(); // [r, in] -> [in, r]
                let mut b = b_raw.t(); // [out, r] -> [r, out]
                if let Some(alpha) = parts.alpha {
                    let rank = a.shape()[1] as f32; // r
                    b = b.multiply(Array::from_slice(&[alpha / rank], &[1]))?;
                }
                lin.push(Adapter::Lora { a, b, scale });
                report.applied += 1;
            }
            None => report.unmatched_paths.push(path),
        }
    }
    Ok(report)
}

#[derive(Default)]
struct LoraParts {
    a: Option<Array>,
    b: Option<Array>,
    alpha: Option<f32>,
}

/// Load and install every adapter in `specs` onto `host`, stacking in order. Each spec's file is
/// read, dispatched to the LoKr or PEFT-LoRA loader by its [`AdapterKind`], applied at `spec.scale`,
/// and its [`ApplyReport`] merged into the combined result — unmatched target paths are surfaced,
/// never silently dropped. `lora_strip_prefix` is the per-family namespace stripped from PEFT-LoRA
/// keys (e.g. `"transformer."`); it does not apply to LoKr (whose keys are bare module paths).
///
/// This is the load-time seam (sc-2534): a provider calls it inside `load()` with its model's
/// [`AdaptableHost`] while the model is still mutable. Empty `specs` is a no-op (empty report).
pub fn apply_adapter_specs(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
    lora_strip_prefix: Option<&str>,
) -> Result<ApplyReport> {
    let mut combined = ApplyReport::default();
    for spec in specs {
        let w = Weights::from_file(&spec.path)?;
        let report = match spec.kind {
            AdapterKind::Lokr => apply_lokr(host, &w, spec.scale)?,
            AdapterKind::Lora => {
                // The file's metadata is authoritative; a kind/metadata mismatch is a caller error
                // (the PEFT-LoRA loader would find no `lora_A/B` keys and apply nothing) — surface it.
                if is_lokr(&w) {
                    return Err(format!(
                        "adapter {} declared Lora but its metadata says networkType=lokr",
                        spec.path.display()
                    )
                    .into());
                }
                apply_lora_peft(host, &w, spec.scale, lora_strip_prefix)?
            }
        };
        combined.applied += report.applied;
        combined.unmatched_paths.extend(report.unmatched_paths);
    }
    Ok(combined)
}

/// LoRA key namespace prefixes diffusers/peft adapter files use, tried in order; the first that any
/// key begins with is stripped. LoKr files are bare (no prefix). (kohya `lora_unet_…_` underscore
/// files flatten the module dots to underscores → they need an explicit per-target pattern matcher,
/// not path-splitting; tracked as sc-2618.) SceneWorks' trained LoRAs use `transformer.` (peft
/// `save_lora_weights`) or `diffusion_model.` (sd-scripts export) — both observed on real files.
pub const COMMON_LORA_PREFIXES: [&str; 2] = ["transformer.", "diffusion_model."];

/// The LoRA namespace prefix present in `w`'s keys, if any (see [`COMMON_LORA_PREFIXES`]).
pub fn detect_lora_prefix(w: &Weights) -> Option<&'static str> {
    let keys: Vec<&str> = w.keys().collect();
    COMMON_LORA_PREFIXES
        .into_iter()
        .find(|&p| keys.iter().any(|k| k.starts_with(p)))
        .map(|v| v as _)
}

/// [`apply_adapter_specs`] with per-file LoRA-prefix **auto-detection** ([`detect_lora_prefix`])
/// instead of a fixed prefix — the common provider path, since LoRA files vary
/// (`transformer.` / `diffusion_model.` / bare) while LoKr keys are bare. The host's key→module map
/// must match the (prefix-stripped) diffusers module paths.
pub fn apply_adapter_specs_autoprefix(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
) -> Result<ApplyReport> {
    let mut combined = ApplyReport::default();
    for spec in specs {
        let w = Weights::from_file(&spec.path)?;
        let prefix = if is_lokr(&w) {
            None
        } else {
            detect_lora_prefix(&w)
        };
        let report = apply_adapter_specs(host, std::slice::from_ref(spec), prefix)?;
        combined.applied += report.applied;
        combined.unmatched_paths.extend(report.unmatched_paths);
    }
    Ok(combined)
}

/// Provider-facing load-time adapter install: [`apply_adapter_specs_autoprefix`] plus a strict
/// no-silent-drop policy — errors if a non-empty spec list matched nothing, or if any adapter
/// target resolved to no module. `model` names the model in the error (e.g. `"z_image_turbo"`).
/// Both Z-Image and Qwen providers call this; the only per-family piece is the model's
/// `AdaptableHost` key→module map.
pub fn apply_adapters_strict(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
    model: &str,
) -> Result<ApplyReport> {
    let report = apply_adapter_specs_autoprefix(host, specs)?;
    if !specs.is_empty() && report.applied == 0 {
        return Err(format!(
            "{model} adapters: no target modules matched across {} adapter file(s) — check the \
             format/prefix (expected diffusers/peft LoRA or LoKr keys; kohya `lora_unet_` files are \
             not yet supported, sc-2618)",
            specs.len()
        )
        .into());
    }
    if !report.unmatched_paths.is_empty() {
        return Err(format!(
            "{model} adapters: {} adapter target(s) matched no module (surfaced, not silently \
             dropped): {:?}",
            report.unmatched_paths.len(),
            report.unmatched_paths
        )
        .into());
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::{AdaptableLinear, Adapter};
    use crate::runtime::{AdapterKind, AdapterSpec};
    use mlx_rs::ops::all_close;
    use std::collections::HashMap;
    use std::path::PathBuf;

    /// Minimal host with a single adaptable linear at path `["lin"]`.
    struct OneLinear {
        lin: AdaptableLinear,
    }
    impl AdaptableHost for OneLinear {
        fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
            match path {
                ["lin"] => Some(&mut self.lin),
                _ => None,
            }
        }
    }

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("mlx_gen_loader_test");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn lora_peft_transposes_and_folds_alpha() {
        // base [out=4, in=3]; PEFT lora_A [r=2, in=3], lora_B [out=4, r=2], alpha=4 (rank=2).
        let weight = Array::from_slice(
            &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &[4, 3],
        );
        let a_raw = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]);
        let b_raw = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
        let alpha = Array::from_slice(&[4.0f32], &[1]);

        let path = tmp("lora.safetensors");
        Array::save_safetensors(
            vec![
                ("lin.lora_A.weight", &a_raw),
                ("lin.lora_B.weight", &b_raw),
                ("lin.alpha", &alpha),
            ],
            None,
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();

        let mut host = OneLinear {
            lin: AdaptableLinear::dense(weight.clone(), None),
        };
        let report = apply_lora_peft(&mut host, &w, 0.5, None).unwrap();
        assert_eq!(report.applied, 1);
        assert!(report.unmatched_paths.is_empty());

        // Reference: a = A^T [in,r], b = B^T * (alpha/rank=2.0) [r,out], scale 0.5.
        let mut expected = AdaptableLinear::dense(weight, None);
        let b_scaled = b_raw
            .t()
            .multiply(Array::from_slice(&[2.0f32], &[1]))
            .unwrap();
        expected.push(Adapter::Lora {
            a: a_raw.t(),
            b: b_scaled,
            scale: 0.5,
        });

        let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
        let got = host.lin.forward(&x).unwrap();
        let want = expected.forward(&x).unwrap();
        assert!(all_close(&got, &want, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn lora_peft_folds_bare_alpha_under_a_prefix() {
        // Prefixed `lora_A/B` (`transformer.lin.lora_{A,B}.weight`) + a BARE `lin.alpha` — the
        // fork's Qwen convention (bare-only alpha patterns). The bare alpha must NOT be dropped:
        // the residual folds alpha/rank into B exactly as the all-bare case does. (sc-2528 review.)
        let weight = Array::from_slice(
            &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &[4, 3],
        );
        let a_raw = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]); // [r=2, in=3]
        let b_raw = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
        let alpha = Array::from_slice(&[4.0f32], &[1]); // rank=2 -> factor 2

        let path = tmp("lora_prefixed_bare_alpha.safetensors");
        Array::save_safetensors(
            vec![
                ("transformer.lin.lora_A.weight", &a_raw),
                ("transformer.lin.lora_B.weight", &b_raw),
                ("lin.alpha", &alpha), // BARE — no `transformer.` prefix
            ],
            None,
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();

        let mut host = OneLinear {
            lin: AdaptableLinear::dense(weight.clone(), None),
        };
        let report = apply_lora_peft(&mut host, &w, 0.5, Some("transformer.")).unwrap();
        assert_eq!(report.applied, 1);
        assert!(report.unmatched_paths.is_empty());

        // Reference: B scaled by alpha/rank = 2 (the bare alpha was honored).
        let mut expected = AdaptableLinear::dense(weight, None);
        let b_scaled = b_raw
            .t()
            .multiply(Array::from_slice(&[2.0f32], &[1]))
            .unwrap();
        expected.push(Adapter::Lora {
            a: a_raw.t(),
            b: b_scaled,
            scale: 0.5,
        });
        let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
        let got = host.lin.forward(&x).unwrap();
        let want = expected.forward(&x).unwrap();
        assert!(
            all_close(&got, &want, 1e-5, 1e-5, false)
                .unwrap()
                .item::<bool>(),
            "bare alpha under a prefix was dropped or mis-folded"
        );
    }

    #[test]
    fn lora_peft_conflicting_alpha_errors() {
        // A prefixed alpha and a bare alpha that disagree for the same path -> hard error, no
        // silent pick.
        let weight = Array::from_slice(
            &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &[4, 3],
        );
        let a_raw = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]);
        let b_raw = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
        let path = tmp("lora_conflicting_alpha.safetensors");
        Array::save_safetensors(
            vec![
                ("transformer.lin.lora_A.weight", &a_raw),
                ("transformer.lin.lora_B.weight", &b_raw),
                ("transformer.lin.alpha", &Array::from_slice(&[4.0f32], &[1])),
                ("lin.alpha", &Array::from_slice(&[8.0f32], &[1])), // disagrees
            ],
            None,
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();
        let mut host = OneLinear {
            lin: AdaptableLinear::dense(weight, None),
        };
        assert!(apply_lora_peft(&mut host, &w, 1.0, Some("transformer.")).is_err());
    }

    #[test]
    fn unmatched_paths_are_reported_not_dropped() {
        // A LoKr file targeting a path the host doesn't have -> applied 0, path reported.
        let dummy = Array::from_slice(&[1.0f32], &[1, 1]);
        let mut meta = HashMap::new();
        meta.insert("networkType".to_string(), "lokr".to_string());
        meta.insert("alpha".to_string(), "1.0".to_string());
        meta.insert("rank".to_string(), "1".to_string());
        let path = tmp("lokr_miss.safetensors");
        Array::save_safetensors(
            vec![
                ("missing.path.lokr_w1", &dummy),
                ("missing.path.lokr_w2", &dummy),
            ],
            Some(&meta),
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();
        assert!(is_lokr(&w));

        let mut host = OneLinear {
            lin: AdaptableLinear::dense(Array::from_slice(&[1.0f32], &[1, 1]), None),
        };
        let report = apply_lokr(&mut host, &w, 1.0).unwrap();
        assert_eq!(report.applied, 0);
        assert_eq!(report.unmatched_paths, vec!["missing.path".to_string()]);
    }

    /// The load-time connector stacks a mixed LoRA + LoKr spec list and is equivalent to calling
    /// the underlying loaders directly, in order.
    #[test]
    fn apply_specs_stacks_mixed_lora_and_lokr() {
        // base [out=4, in=2].
        let base_vals: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let weight = Array::from_slice(&base_vals, &[4, 2]);

        // PEFT LoRA file targeting ["lin"]: lora_A [r=2, in=2], lora_B [out=4, r=2].
        let a_raw = Array::from_slice(&[0.1f32, 0.2, -0.1, -0.2], &[2, 2]);
        let b_raw = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
        let lora_path = tmp("specs_lora.safetensors");
        Array::save_safetensors(
            vec![("lin.lora_A.weight", &a_raw), ("lin.lora_B.weight", &b_raw)],
            None,
            &lora_path,
        )
        .unwrap();

        // LoKr file targeting ["lin"]: kron(w1[2,1], w2[2,2]) -> [4,2]; alpha==rank -> factor 1.
        let w1 = Array::from_slice(&[1.0f32, 0.5], &[2, 1]);
        let w2 = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
        let mut meta = HashMap::new();
        meta.insert("networkType".to_string(), "lokr".to_string());
        meta.insert("alpha".to_string(), "1.0".to_string());
        meta.insert("rank".to_string(), "1".to_string());
        let lokr_path = tmp("specs_lokr.safetensors");
        Array::save_safetensors(
            vec![("lin.lokr_w1", &w1), ("lin.lokr_w2", &w2)],
            Some(&meta),
            &lokr_path,
        )
        .unwrap();

        let specs = vec![
            AdapterSpec {
                path: lora_path.clone(),
                scale: 0.5,
                kind: AdapterKind::Lora,
            },
            AdapterSpec {
                path: lokr_path.clone(),
                scale: 1.0,
                kind: AdapterKind::Lokr,
            },
        ];

        let mut via_specs = OneLinear {
            lin: AdaptableLinear::dense(weight.clone(), None),
        };
        let report = apply_adapter_specs(&mut via_specs, &specs, None).unwrap();
        assert_eq!(report.applied, 2);
        assert!(report.unmatched_paths.is_empty());

        // Reference: the same files through the underlying loaders directly, in order.
        let mut via_loaders = OneLinear {
            lin: AdaptableLinear::dense(weight, None),
        };
        apply_lora_peft(
            &mut via_loaders,
            &Weights::from_file(&lora_path).unwrap(),
            0.5,
            None,
        )
        .unwrap();
        apply_lokr(
            &mut via_loaders,
            &Weights::from_file(&lokr_path).unwrap(),
            1.0,
        )
        .unwrap();

        let x = Array::from_slice(&[1.0f32, -2.0], &[1, 2]);
        let got = via_specs.lin.forward(&x).unwrap();
        let want = via_loaders.lin.forward(&x).unwrap();
        assert!(all_close(&got, &want, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>());

        // Both adapters actually moved the output off the bare base.
        let base = AdaptableLinear::dense(Array::from_slice(&base_vals, &[4, 2]), None)
            .forward(&x)
            .unwrap();
        assert!(!all_close(&got, &base, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn apply_specs_empty_is_noop() {
        let weight = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let mut host = OneLinear {
            lin: AdaptableLinear::dense(weight.clone(), None),
        };
        let report = apply_adapter_specs(&mut host, &[], None).unwrap();
        assert_eq!(report, ApplyReport::default());

        let x = Array::from_slice(&[1.0f32, -1.0], &[1, 2]);
        let got = host.lin.forward(&x).unwrap();
        let want = AdaptableLinear::dense(weight, None).forward(&x).unwrap();
        assert!(all_close(&got, &want, 1e-6, 1e-6, false)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn apply_specs_reports_unmatched_paths() {
        let dummy = Array::from_slice(&[1.0f32], &[1, 1]);
        let mut meta = HashMap::new();
        meta.insert("networkType".to_string(), "lokr".to_string());
        meta.insert("alpha".to_string(), "1.0".to_string());
        meta.insert("rank".to_string(), "1".to_string());
        let path = tmp("specs_miss.safetensors");
        Array::save_safetensors(
            vec![("nope.here.lokr_w1", &dummy), ("nope.here.lokr_w2", &dummy)],
            Some(&meta),
            &path,
        )
        .unwrap();

        let specs = vec![AdapterSpec {
            path,
            scale: 1.0,
            kind: AdapterKind::Lokr,
        }];
        let mut host = OneLinear {
            lin: AdaptableLinear::dense(Array::from_slice(&[1.0f32], &[1, 1]), None),
        };
        let report = apply_adapter_specs(&mut host, &specs, None).unwrap();
        assert_eq!(report.applied, 0);
        assert_eq!(report.unmatched_paths, vec!["nope.here".to_string()]);
    }

    #[test]
    fn apply_specs_kind_metadata_mismatch_errors() {
        let dummy = Array::from_slice(&[1.0f32], &[1, 1]);
        let mut meta = HashMap::new();
        meta.insert("networkType".to_string(), "lokr".to_string());
        let path = tmp("specs_mismatch.safetensors");
        Array::save_safetensors(vec![("lin.lokr_w1", &dummy)], Some(&meta), &path).unwrap();

        // Declared Lora but the file's metadata says LoKr -> a loud error, not a silent no-op.
        let specs = vec![AdapterSpec {
            path,
            scale: 1.0,
            kind: AdapterKind::Lora,
        }];
        let mut host = OneLinear {
            lin: AdaptableLinear::dense(Array::from_slice(&[1.0f32], &[1, 1]), None),
        };
        assert!(apply_adapter_specs(&mut host, &specs, None).is_err());
    }

    #[test]
    fn detect_lora_prefix_variants() {
        let a = Array::from_slice(&[0.0f32], &[1, 1]);
        let bare = tmp("detect_bare.safetensors");
        Array::save_safetensors(vec![("lin.lora_A.weight", &a)], None, &bare).unwrap();
        assert_eq!(
            detect_lora_prefix(&Weights::from_file(&bare).unwrap()),
            None
        );

        let tf = tmp("detect_tf.safetensors");
        Array::save_safetensors(vec![("transformer.lin.lora_A.weight", &a)], None, &tf).unwrap();
        assert_eq!(
            detect_lora_prefix(&Weights::from_file(&tf).unwrap()),
            Some("transformer.")
        );

        let dm = tmp("detect_dm.safetensors");
        Array::save_safetensors(vec![("diffusion_model.lin.lora_A.weight", &a)], None, &dm)
            .unwrap();
        assert_eq!(
            detect_lora_prefix(&Weights::from_file(&dm).unwrap()),
            Some("diffusion_model.")
        );
    }

    #[test]
    fn autoprefix_strips_detected_prefix_and_applies() {
        // base [out=2, in=2]; a `transformer.`-prefixed peft LoRA on path ["lin"].
        let weight = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
        let a = Array::from_slice(&[0.1f32, 0.2, -0.1, -0.2], &[2, 2]); // [r=2, in=2]
        let b = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75], &[2, 2]); // [out=2, r=2]
        let path = tmp("autoprefix_lora.safetensors");
        Array::save_safetensors(
            vec![
                ("transformer.lin.lora_A.weight", &a),
                ("transformer.lin.lora_B.weight", &b),
            ],
            None,
            &path,
        )
        .unwrap();

        let mut host = OneLinear {
            lin: AdaptableLinear::dense(weight, None),
        };
        let specs = vec![AdapterSpec {
            path,
            scale: 1.0,
            kind: AdapterKind::Lora,
        }];
        let report = apply_adapter_specs_autoprefix(&mut host, &specs).unwrap();
        assert_eq!(
            report.applied, 1,
            "transformer.-prefixed key should resolve to lin"
        );
        assert!(report.unmatched_paths.is_empty());

        // Strict wrapper: a bare-but-unmatched target errors rather than silently dropping.
        let miss = tmp("autoprefix_miss.safetensors");
        Array::save_safetensors(
            vec![
                ("transformer.nope.lora_A.weight", &a),
                ("transformer.nope.lora_B.weight", &b),
            ],
            None,
            &miss,
        )
        .unwrap();
        let mut host2 = OneLinear {
            lin: AdaptableLinear::dense(Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]), None),
        };
        let specs2 = vec![AdapterSpec {
            path: miss,
            scale: 1.0,
            kind: AdapterKind::Lora,
        }];
        assert!(apply_adapters_strict(&mut host2, &specs2, "test").is_err());
    }
}
