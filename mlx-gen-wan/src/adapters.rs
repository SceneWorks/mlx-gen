//! Wan2.2 LoRA application (sc-2683) — wires the reference `mlx_video/lora/` into the Wan
//! generate path plus the Wan key→module map.
//!
//! **Strategy: weight MERGE** (the reference Wan path), not a forward-time residual. The reference
//! `generate_wan.py` applies LoRA through `load_wan_model` → `load_and_apply_loras` →
//! `apply_loras_to_weights`: for a bf16 expert it folds `ΔW = (B·A)·(alpha/rank)·strength` (computed
//! at the factor dtype, cast to the weight dtype) **into the weight** before the forward; for a
//! quantized expert it dequantizes the targeted layers, merges, and replaces them with bf16 Linears.
//! Its runtime-residual `LoRALinear` class exists but is **not** invoked by the Wan path (that is why
//! LTX — where the reference never wired LoRA — chose a residual instead, sc-2687). Merge is faithful
//! to the production reference (the parity gate is "vs a reference-merged golden"), cheap and exact on
//! the bf16 base, has **zero** per-step / forward cost (the [`WanTransformer`](crate::transformer)
//! is untouched and the no-adapter path is trivially byte-identical), and maps directly onto the MoE
//! high/low split — each expert's weight map is merged independently.
//!
//! **MoE high/low.** The reference forms `_loras_low = (loras)+(loras_low)` and
//! `_loras_high = (loras)+(loras_high)` and merges each onto its expert. Mirrored via
//! [`AdapterSpec::moe_expert`](mlx_gen::AdapterSpec): `None` = a shared file (merged onto **both**
//! experts), `Some(High)`/`Some(Low)` = one expert only. [`merge_wan_adapters`] is called once per
//! expert and selects the shared specs first, then this expert's specific ones (the `(loras)+(loras_*)`
//! order), so a module hit by both accumulates in the reference's order.
//!
//! **Format.** PEFT `lora_A`/`lora_B` and kohya `lora_down`/`lora_up`, optional per-module `.alpha`
//! (default = rank), `diffusion_model.`-prefixed (the real SceneWorks Wan2.2 MoE LoRAs ship PEFT,
//! bf16, rank 64, no alpha). `scale = alpha/rank` (the reference `LoRAWeights.scale`). LoKr is
//! rejected loudly (sc-2393). The kohya `lora_unet_`-**flattened** external form is not part of the
//! reference Wan surface (its `_normalize_wan_lora_key` only strips prefixes + renames dotted paths);
//! such keys resolve to no module and are surfaced (never silently dropped) — adding it would be
//! net-new beyond the fork.
//!
//! **Skips, never errors-on-skip.** Mirrors the reference (`apply_loras_to_weights` counts skipped
//! modules, never raises): a LoRA target absent from this checkpoint is reported, not fatal. The
//! caller errors only if a non-empty spec list matched *nothing* across both experts (a format/prefix
//! misconfiguration).

use std::collections::BTreeMap;

use mlx_rs::ops::{add, matmul, multiply};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::loader::is_lokr;
use mlx_gen::array::scalar;
use mlx_gen::runtime::{AdapterKind, AdapterSpec, MoeExpert};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

/// LoRA key namespace prefixes stripped (longest-first), matching the reference
/// `_normalize_wan_lora_key`. SceneWorks' trained Wan LoRAs use `diffusion_model.`.
const PREFIXES: [&str; 4] = [
    "model.diffusion_model.",
    "diffusion_model.",
    "base_model.model.",
    "model.",
];

/// Outcome of merging one expert's adapters: how many module weights were folded, how many specs
/// applied to this expert, and any LoRA module paths that resolved to no weight (surfaced, never
/// silently dropped).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct WanLoraReport {
    /// Module weights actually merged (one per resolved target, across all applicable specs).
    pub applied: usize,
    /// Specs that applied to this expert (shared + this-expert-specific).
    pub applicable: usize,
    /// LoRA module paths (normalized) that matched no weight in this checkpoint.
    pub skipped: Vec<String>,
}

#[derive(Clone, Copy)]
enum Role {
    Down, // lora_A / lora_down → A [rank, in]
    Up,   // lora_B / lora_up   → B [out, rank]
    Alpha,
}

#[derive(Default)]
struct LoraParts {
    down: Option<Array>,
    up: Option<Array>,
    alpha: Option<f32>,
}

/// PEFT + kohya factor suffixes (exact match). `lora_A`/`lora_down` are the A (down) factor;
/// `lora_B`/`lora_up` the B (up). Mirrors the reference `load_lora_weights`, which accepts both
/// conventions; the `.alpha` scalar is optional (default rank).
const SUFFIXES: [(&str, Role); 5] = [
    (".lora_A.weight", Role::Down),
    (".lora_B.weight", Role::Up),
    (".lora_down.weight", Role::Down),
    (".lora_up.weight", Role::Up),
    (".alpha", Role::Alpha),
];

/// Read a scalar `.alpha` as f32 regardless of on-disk dtype (real files ship it bf16; a direct
/// `as_slice::<f32>()` would panic on a dtype mismatch). A `[]`- or `[1]`-shaped scalar both read.
fn read_alpha(a: &Array) -> Result<f32> {
    Ok(a.as_dtype(Dtype::Float32)?.as_slice::<f32>()[0])
}

/// Normalize a LoRA module path to the Wan checkpoint's naming (the reference
/// `_normalize_wan_lora_key`): strip a known prefix, then the `convert_wan` renames
/// `ffn.0/.2 → ffn.fc1/.fc2`, `text_embedding.0/.2 → text_embedding_0/_1`,
/// `time_embedding.0/.2 → time_embedding_0/_1`, `time_projection.1 → time_projection`,
/// `patch_embedding → patch_embedding_proj`. attn `q/k/v/o` pass through. Both the `.X.` infix and
/// the bare `…X` suffix forms are handled, as the reference does (a LoRA module stem ends at the
/// module, so the suffix forms fire).
pub(crate) fn normalize_wan_key(key: &str) -> String {
    let stripped = PREFIXES
        .iter()
        .find_map(|p| key.strip_prefix(p))
        .unwrap_or(key);
    let mut t = stripped.to_string();

    // ffn.0 → ffn.fc1, ffn.2 → ffn.fc2
    t = t
        .replace(".ffn.0.", ".ffn.fc1.")
        .replace(".ffn.2.", ".ffn.fc2.");
    if let Some(h) = t.strip_suffix(".ffn.0") {
        t = format!("{h}.ffn.fc1");
    }
    if let Some(h) = t.strip_suffix(".ffn.2") {
        t = format!("{h}.ffn.fc2");
    }

    // text_embedding.0/.2 → text_embedding_0/_1
    t = t
        .replace("text_embedding.0.", "text_embedding_0.")
        .replace("text_embedding.2.", "text_embedding_1.");
    if let Some(h) = t.strip_suffix("text_embedding.0") {
        t = format!("{h}text_embedding_0");
    }
    if let Some(h) = t.strip_suffix("text_embedding.2") {
        t = format!("{h}text_embedding_1");
    }

    // time_embedding.0/.2 → time_embedding_0/_1
    t = t
        .replace("time_embedding.0.", "time_embedding_0.")
        .replace("time_embedding.2.", "time_embedding_1.");
    if let Some(h) = t.strip_suffix("time_embedding.0") {
        t = format!("{h}time_embedding_0");
    }
    if let Some(h) = t.strip_suffix("time_embedding.2") {
        t = format!("{h}time_embedding_1");
    }

    // time_projection.1 → time_projection
    t = t.replace("time_projection.1.", "time_projection.");
    if let Some(h) = t.strip_suffix("time_projection.1") {
        t = format!("{h}time_projection");
    }

    // patch_embedding → patch_embedding_proj
    if t.contains("patch_embedding") && !t.contains("patch_embedding_proj") {
        t = t.replace("patch_embedding", "patch_embedding_proj");
    }
    t
}

/// Merge one LoRA file's deltas into the weight map `w` at `spec.scale`, accumulating into `report`.
/// Mirrors `apply_lora_to_linear` per module: `ΔW = (B·A)·(alpha/rank·strength)` at the factor dtype,
/// cast to the weight dtype and added — so the no-LoRA forward and the merged forward share one bf16
/// GEMM (no per-step residual). Multiple files accumulate because each reads the (already-merged)
/// weight back from `w`.
fn merge_one(w: &mut Weights, spec: &AdapterSpec, report: &mut WanLoraReport) -> Result<()> {
    let lw = Weights::from_file(&spec.path)?;
    if spec.kind == AdapterKind::Lokr || is_lokr(&lw) {
        return Err(Error::Msg(format!(
            "wan2_2 adapter {}: LoKr is not yet supported (sc-2393); the reference Wan lora/ path is \
             LoRA-only (merge)",
            spec.path.display()
        )));
    }

    // Group factors by normalized module path.
    let mut groups: BTreeMap<String, LoraParts> = BTreeMap::new();
    for key in lw.keys().map(str::to_string).collect::<Vec<_>>() {
        let Some((stem, role)) = SUFFIXES
            .iter()
            .find_map(|(suf, role)| key.strip_suffix(suf).map(|s| (s, *role)))
        else {
            continue; // not a LoRA factor key (base weight / bundled extra) — ignore.
        };
        let parts = groups.entry(normalize_wan_key(stem)).or_default();
        match role {
            Role::Down => parts.down = Some(lw.require(&key)?.clone()),
            Role::Up => parts.up = Some(lw.require(&key)?.clone()),
            Role::Alpha => parts.alpha = Some(read_alpha(lw.require(&key)?)?),
        }
    }

    for (path, parts) in groups {
        let (Some(down), Some(up)) = (parts.down, parts.up) else {
            // A down/up whose partner targeted a non-LoRA key — skip the orphan, surface the path.
            report.skipped.push(path);
            continue;
        };
        let wkey = format!("{path}.weight");
        let Some(base) = w.get(&wkey).cloned() else {
            report.skipped.push(path);
            continue;
        };
        // lora_A: [rank, in], lora_B: [out, rank]. delta = B·A → [out, in], the weight's shape.
        let rank = down.shape()[0] as f64;
        let alpha = parts.alpha.map(|a| a as f64).unwrap_or(rank);
        // (alpha/rank)·strength as a single value, matching the reference's Python-float `scale·strength`.
        let eff = (alpha / rank * spec.scale as f64) as f32;
        let delta = matmul(&up, &down)?;
        // Dtype-matched scalar preserves the factor dtype (the reference's weak `delta * (scale*strength)`).
        let delta = multiply(&delta, &scalar(eff).as_dtype(delta.dtype())?)?;
        let merged = add(&base, &delta.as_dtype(base.dtype())?)?;
        w.insert(wkey, merged);
        report.applied += 1;
    }
    Ok(())
}

/// Merge every LoRA in `specs` that targets `expert` into the expert weight map `w` (sc-2683). Shared
/// specs (`moe_expert == None`) are applied first, then this expert's specific ones (`Some(expert)`),
/// mirroring the reference `(loras)+(loras_high/low)` order so a module hit by both accumulates in
/// the same order. LoKr is rejected (sc-2393); per-key skips are reported, not fatal (the reference
/// warns on skip). Returns the merge report; the caller enforces the "matched nothing across both
/// experts" error.
pub fn merge_wan_adapters(
    w: &mut Weights,
    specs: &[AdapterSpec],
    expert: MoeExpert,
) -> Result<WanLoraReport> {
    let mut report = WanLoraReport::default();
    // Pass 1: shared files (merged onto every expert). Pass 2: this expert's specific files.
    for spec in specs.iter().filter(|s| s.moe_expert.is_none()) {
        report.applicable += 1;
        merge_one(w, spec, &mut report)?;
    }
    for spec in specs.iter().filter(|s| s.moe_expert == Some(expert)) {
        report.applicable += 1;
        merge_one(w, spec, &mut report)?;
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{all_close, array_eq};
    use std::path::PathBuf;

    #[test]
    fn normalize_strips_prefix_and_renames() {
        // attn q/k/v/o pass through (already checkpoint naming).
        assert_eq!(
            normalize_wan_key("diffusion_model.blocks.0.self_attn.q"),
            "blocks.0.self_attn.q"
        );
        assert_eq!(
            normalize_wan_key("diffusion_model.blocks.7.cross_attn.o"),
            "blocks.7.cross_attn.o"
        );
        // ffn.0/.2 → fc1/fc2.
        assert_eq!(
            normalize_wan_key("diffusion_model.blocks.0.ffn.0"),
            "blocks.0.ffn.fc1"
        );
        assert_eq!(
            normalize_wan_key("diffusion_model.blocks.3.ffn.2"),
            "blocks.3.ffn.fc2"
        );
        // global renames + other prefixes.
        assert_eq!(
            normalize_wan_key("model.diffusion_model.text_embedding.0"),
            "text_embedding_0"
        );
        assert_eq!(
            normalize_wan_key("base_model.model.text_embedding.2"),
            "text_embedding_1"
        );
        assert_eq!(normalize_wan_key("time_embedding.0"), "time_embedding_0");
        assert_eq!(
            normalize_wan_key("diffusion_model.time_projection.1"),
            "time_projection"
        );
        assert_eq!(
            normalize_wan_key("diffusion_model.patch_embedding"),
            "patch_embedding_proj"
        );
    }

    #[test]
    fn normalize_matches_reference_golden() {
        // Parity vs the reference `_normalize_wan_lora_key` over every real lauren MoE LoRA module
        // stem (400) + synthetic global / alternate-prefix spellings, resolved against the real
        // converted A14B weight-key set (tools/dump_lora_fixtures.py). This is the load-bearing
        // piece of the merge — the Wan key→module map must be byte-identical to the reference's.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/wan_lora_keys.json"
        );
        let text = std::fs::read_to_string(path).expect("read wan_lora_keys.json fixture");
        let map: BTreeMap<String, String> = serde_json::from_str(&text).expect("parse fixture");
        assert!(
            map.len() >= 400,
            "fixture should cover the full real LoRA surface (got {})",
            map.len()
        );
        for (raw, expected) in &map {
            assert_eq!(
                &normalize_wan_key(raw),
                expected,
                "normalize_wan_key({raw}) must match the reference _normalize_wan_lora_key"
            );
        }
    }

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("mlx_gen_wan_adapters_test");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    /// Write a PEFT LoRA file (`diffusion_model.‹stem›.lora_A/B.weight`) for the given stems, with
    /// A `[rank,in]`, B `[out,rank]`, no alpha (→ scale = 1). Values are deterministic per stem.
    fn write_lora(name: &str, stems: &[(&str, i32, i32)], rank: i32, seed: f32) -> PathBuf {
        let mut entries: Vec<(String, Array)> = Vec::new();
        for (stem, out, inp) in stems {
            let a = Array::from_slice(
                &(0..rank * inp)
                    .map(|i| (i as f32 * 0.001 + seed).sin() * 0.02)
                    .collect::<Vec<_>>(),
                &[rank, *inp],
            )
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
            let b = Array::from_slice(
                &(0..out * rank)
                    .map(|i| (i as f32 * 0.0007 + seed).cos() * 0.02)
                    .collect::<Vec<_>>(),
                &[*out, rank],
            )
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
            entries.push((format!("diffusion_model.{stem}.lora_A.weight"), a));
            entries.push((format!("diffusion_model.{stem}.lora_B.weight"), b));
        }
        let path = tmp(name);
        let refs: Vec<(&str, &Array)> = entries.iter().map(|(k, v)| (k.as_str(), v)).collect();
        Array::save_safetensors(refs, None, &path).unwrap();
        path
    }

    /// A synthetic expert weight map with the two module weights the test LoRA targets, bf16.
    fn synthetic_weights() -> Weights {
        let path = tmp("base.safetensors");
        let q = Array::from_slice(
            &(0..16 * 8)
                .map(|i| i as f32 * 0.01 - 0.3)
                .collect::<Vec<_>>(),
            &[16, 8],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let fc1 = Array::from_slice(
            &(0..24 * 8)
                .map(|i| i as f32 * 0.005 - 0.2)
                .collect::<Vec<_>>(),
            &[24, 8],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        Array::save_safetensors(
            vec![
                ("blocks.0.self_attn.q.weight", &q),
                ("blocks.0.ffn.fc1.weight", &fc1),
            ],
            None,
            &path,
        )
        .unwrap();
        Weights::from_file(&path).unwrap()
    }

    fn spec(path: PathBuf, scale: f32, expert: Option<MoeExpert>) -> AdapterSpec {
        AdapterSpec {
            path,
            scale,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: expert,
        }
    }

    #[test]
    fn merge_folds_delta_bit_exact() {
        // Reference merge: W += (B·A)·(alpha/rank·strength).astype(W.dtype), at the factor dtype.
        let lora = write_lora(
            "merge.safetensors",
            &[("blocks.0.self_attn.q", 16, 8), ("blocks.0.ffn.0", 24, 8)],
            4,
            0.1,
        );
        let mut w = synthetic_weights();
        let report =
            merge_wan_adapters(&mut w, &[spec(lora.clone(), 1.0, None)], MoeExpert::High).unwrap();
        assert_eq!(report.applied, 2);
        assert!(report.skipped.is_empty());

        // Hand-compute the expected merge for the q weight.
        let lw = Weights::from_file(&lora).unwrap();
        let base = synthetic_weights();
        let q_base = base.require("blocks.0.self_attn.q.weight").unwrap();
        let a = lw
            .require("diffusion_model.blocks.0.self_attn.q.lora_A.weight")
            .unwrap();
        let b = lw
            .require("diffusion_model.blocks.0.self_attn.q.lora_B.weight")
            .unwrap();
        let delta = matmul(b, a).unwrap();
        let want = add(q_base, delta.as_dtype(q_base.dtype()).unwrap()).unwrap();
        let got = w.require("blocks.0.self_attn.q.weight").unwrap();
        assert!(
            array_eq(got, &want, false).unwrap().item::<bool>(),
            "merged q weight must be bit-exact to W + (B·A).astype(W.dtype)"
        );
        // And the ffn key was the renamed target (ffn.0 → ffn.fc1).
        assert!(w.get("blocks.0.ffn.fc1.weight").is_some());
    }

    #[test]
    fn scale_zero_is_bit_exact_noop() {
        let lora = write_lora(
            "zero.safetensors",
            &[("blocks.0.self_attn.q", 16, 8)],
            4,
            0.3,
        );
        let base = synthetic_weights();
        let mut w = synthetic_weights();
        let report = merge_wan_adapters(&mut w, &[spec(lora, 0.0, None)], MoeExpert::Low).unwrap();
        assert_eq!(report.applied, 1); // still "applied" (folded a zero delta), like the reference.
        let got = w.require("blocks.0.self_attn.q.weight").unwrap();
        let unchanged = base.require("blocks.0.self_attn.q.weight").unwrap();
        assert!(
            array_eq(got, unchanged, false).unwrap().item::<bool>(),
            "strength 0 must leave the weight bit-identical"
        );
    }

    #[test]
    fn high_low_filter_selects_shared_plus_expert() {
        let shared = write_lora(
            "shared.safetensors",
            &[("blocks.0.self_attn.q", 16, 8)],
            4,
            0.2,
        );
        let high_only = write_lora("highonly.safetensors", &[("blocks.0.ffn.0", 24, 8)], 4, 0.5);

        // Building the LOW expert: the shared file applies, the high-only file does NOT.
        let mut low = synthetic_weights();
        let low_rep = merge_wan_adapters(
            &mut low,
            &[
                spec(shared.clone(), 1.0, None),
                spec(high_only.clone(), 1.0, Some(MoeExpert::High)),
            ],
            MoeExpert::Low,
        )
        .unwrap();
        assert_eq!(low_rep.applicable, 1, "only the shared spec applies to low");
        assert_eq!(low_rep.applied, 1);

        // Building the HIGH expert: both the shared and the high-only file apply.
        let mut high = synthetic_weights();
        let high_rep = merge_wan_adapters(
            &mut high,
            &[
                spec(shared, 1.0, None),
                spec(high_only, 1.0, Some(MoeExpert::High)),
            ],
            MoeExpert::High,
        )
        .unwrap();
        assert_eq!(high_rep.applicable, 2, "shared + high-only apply to high");
        assert_eq!(high_rep.applied, 2);

        // The two experts' q weights differ from the bare base (visible effect) and the high expert's
        // ffn was merged while the low expert's was not.
        let base = synthetic_weights();
        let q_base = base.require("blocks.0.self_attn.q.weight").unwrap();
        let q_low = low.require("blocks.0.self_attn.q.weight").unwrap();
        assert!(!array_eq(q_low, q_base, false).unwrap().item::<bool>());
        let fc1_base = base.require("blocks.0.ffn.fc1.weight").unwrap();
        let fc1_low = low.require("blocks.0.ffn.fc1.weight").unwrap();
        let fc1_high = high.require("blocks.0.ffn.fc1.weight").unwrap();
        assert!(
            array_eq(fc1_low, fc1_base, false).unwrap().item::<bool>(),
            "low expert's ffn must be untouched (high-only LoRA)"
        );
        assert!(!array_eq(fc1_high, fc1_base, false).unwrap().item::<bool>());
    }

    #[test]
    fn accumulates_multiple_specs_on_one_module() {
        // Two shared LoRAs on the same module accumulate (W + d1 + d2), order-preserving.
        let l1 = write_lora(
            "acc1.safetensors",
            &[("blocks.0.self_attn.q", 16, 8)],
            4,
            0.1,
        );
        let l2 = write_lora(
            "acc2.safetensors",
            &[("blocks.0.self_attn.q", 16, 8)],
            4,
            0.9,
        );
        let mut w = synthetic_weights();
        merge_wan_adapters(
            &mut w,
            &[spec(l1.clone(), 1.0, None), spec(l2.clone(), 1.0, None)],
            MoeExpert::High,
        )
        .unwrap();

        let base = synthetic_weights();
        let mut want = base.require("blocks.0.self_attn.q.weight").unwrap().clone();
        for lpath in [&l1, &l2] {
            let lw = Weights::from_file(lpath).unwrap();
            let a = lw
                .require("diffusion_model.blocks.0.self_attn.q.lora_A.weight")
                .unwrap();
            let b = lw
                .require("diffusion_model.blocks.0.self_attn.q.lora_B.weight")
                .unwrap();
            let delta = matmul(b, a).unwrap();
            want = add(&want, delta.as_dtype(want.dtype()).unwrap()).unwrap();
        }
        let got = w.require("blocks.0.self_attn.q.weight").unwrap();
        assert!(
            all_close(got, &want, 1e-6, 1e-6, false)
                .unwrap()
                .item::<bool>(),
            "two stacked LoRAs must equal W + d1 + d2 in order"
        );
    }

    #[test]
    fn lokr_is_rejected() {
        // A spec declared LoKr is rejected loudly (sc-2393) before any merge.
        let lora = write_lora("lk.safetensors", &[("blocks.0.self_attn.q", 16, 8)], 4, 0.2);
        let lokr_spec = AdapterSpec {
            kind: AdapterKind::Lokr,
            ..spec(lora, 1.0, None)
        };
        let mut w = synthetic_weights();
        assert!(merge_wan_adapters(&mut w, &[lokr_spec], MoeExpert::High).is_err());
    }
}
