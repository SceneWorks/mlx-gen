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
//! bf16, rank 64, no alpha). `scale = alpha/rank` (the reference `LoRAWeights.scale`). **LoKr**
//! (sc-2393 — net-new; the reference Wan path is LoRA-only) is parsed by the core `parse_lokr`, its
//! per-module `[out,in]` delta reconstructed via `reconstruct_lokr_delta` (`alpha/rank` folded in),
//! and folded into the weight through the **same in-place merge** as LoRA (`merge_one_lokr`). The
//! kohya `lora_unet_`-**flattened** external form is not part of the reference Wan surface (its
//! `_normalize_wan_lora_key` only strips prefixes + renames dotted paths); such keys resolve to no
//! module and are surfaced (never silently dropped) — adding it would be net-new beyond the fork.
//!
//! **Skips, never errors-on-skip.** Mirrors the reference (`apply_loras_to_weights` counts skipped
//! modules, never raises): a LoRA target absent from this checkpoint is reported, not fatal. The
//! caller errors only if a non-empty spec list matched *nothing* across both experts (a format/prefix
//! misconfiguration).

use std::collections::BTreeMap;

use mlx_rs::ops::{add, matmul, multiply};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::loader::{
    is_loha_keys, is_lokr, is_lokr_keys, parse_loha_thirdparty, parse_lokr, parse_lokr_thirdparty,
    resolve_lokr_path,
};
use mlx_gen::adapters::{build_lokr_factors, AdaptableHost, Adapter};
use mlx_gen::array::scalar;
use mlx_gen::gen_core::weightsmeta::{LoraAdapterMeta, LORA_ADAPTER_METADATA_KEY};
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

/// Normalize a LoRA module path to the **diffusers** Wan-VACE checkpoint's naming (sc-3439). Unlike
/// [`normalize_wan_key`] (which targets the *native* converted Wan layout), the VACE transformer
/// ([`crate::vace::WanVaceTransformer`]) reads the diffusers tensor names directly, so the target is
/// already diffusers. This mirrors diffusers' own LoRA loader
/// (`_convert_non_diffusers_wan_lora_to_diffusers`): strip a known prefix, rename the **native** Wan
/// module spellings (which third-party trainers like musubi-tuner / diffusion-pipe emit) to their
/// diffusers equivalents, and pass an **already-diffusers** key (incl. every `vace_blocks.*` module,
/// which the diffusers converter does not touch and which only ever ships diffusers-named) through
/// unchanged. Renames (both the `.X.` infix and the bare `…X` suffix forms, as a LoRA module stem
/// ends at the module):
/// - `self_attn.{q,k,v,o}` → `attn1.{to_q,to_k,to_v,to_out.0}`
/// - `cross_attn.{q,k,v,o}` → `attn2.{to_q,to_k,to_v,to_out.0}`
/// - `ffn.0`/`ffn.2` → `ffn.net.0.proj`/`ffn.net.2`
/// - VACE block `before_proj`/`after_proj` → `proj_in`/`proj_out`
/// - `time_projection.1` → `condition_embedder.time_proj`; `head.head` → `proj_out`
/// - `text_embedding.0/.2` → `condition_embedder.text_embedder.linear_1/2`
/// - `time_embedding.0/.2` → `condition_embedder.time_embedder.linear_1/2`
///
/// The i2v `k_img`/`v_img` cross-attn factors (diffusers `add_k_proj`/`add_v_proj`) are intentionally
/// not mapped — the VACE host has no such modules, so they resolve to no weight and are surfaced
/// (skipped), never silently mis-folded.
pub(crate) fn normalize_vace_key(key: &str) -> String {
    let stripped = PREFIXES
        .iter()
        .find_map(|p| key.strip_prefix(p))
        .unwrap_or(key);
    let mut t = stripped.to_string();

    // VACE block hint projections: native before_proj/after_proj → diffusers proj_in/proj_out.
    t = t
        .replace(".before_proj.", ".proj_in.")
        .replace(".after_proj.", ".proj_out.");
    if let Some(h) = t.strip_suffix(".before_proj") {
        t = format!("{h}.proj_in");
    }
    if let Some(h) = t.strip_suffix(".after_proj") {
        t = format!("{h}.proj_out");
    }

    // Self-/cross-attn projections → attn1/attn2.{to_q,to_k,to_v,to_out.0}.
    for (src, dst) in [("self_attn", "attn1"), ("cross_attn", "attn2")] {
        for (n, d) in [
            ("q", "to_q"),
            ("k", "to_k"),
            ("v", "to_v"),
            ("o", "to_out.0"),
        ] {
            t = t.replace(&format!(".{src}.{n}."), &format!(".{dst}.{d}."));
            if let Some(h) = t.strip_suffix(&format!(".{src}.{n}")) {
                t = format!("{h}.{dst}.{d}");
            }
        }
    }

    // ffn.0/.2 → ffn.net.0.proj / ffn.net.2.
    t = t
        .replace(".ffn.0.", ".ffn.net.0.proj.")
        .replace(".ffn.2.", ".ffn.net.2.");
    if let Some(h) = t.strip_suffix(".ffn.0") {
        t = format!("{h}.ffn.net.0.proj");
    }
    if let Some(h) = t.strip_suffix(".ffn.2") {
        t = format!("{h}.ffn.net.2");
    }

    // Global modules (the diffusers converter's "Remaining" branch).
    t = t.replace("time_projection.1.", "condition_embedder.time_proj.");
    if let Some(h) = t.strip_suffix("time_projection.1") {
        t = format!("{h}condition_embedder.time_proj");
    }
    t = t.replace("head.head.", "proj_out.");
    if let Some(h) = t.strip_suffix("head.head") {
        t = format!("{h}proj_out");
    }
    t = t
        .replace(
            "text_embedding.0.",
            "condition_embedder.text_embedder.linear_1.",
        )
        .replace(
            "text_embedding.2.",
            "condition_embedder.text_embedder.linear_2.",
        );
    if let Some(h) = t.strip_suffix("text_embedding.0") {
        t = format!("{h}condition_embedder.text_embedder.linear_1");
    }
    if let Some(h) = t.strip_suffix("text_embedding.2") {
        t = format!("{h}condition_embedder.text_embedder.linear_2");
    }
    t = t
        .replace(
            "time_embedding.0.",
            "condition_embedder.time_embedder.linear_1.",
        )
        .replace(
            "time_embedding.2.",
            "condition_embedder.time_embedder.linear_2.",
        );
    if let Some(h) = t.strip_suffix("time_embedding.0") {
        t = format!("{h}condition_embedder.time_embedder.linear_1");
    }
    if let Some(h) = t.strip_suffix("time_embedding.2") {
        t = format!("{h}condition_embedder.time_embedder.linear_2");
    }
    t
}

/// Merge one LoRA file's deltas into the weight map `w` at `spec.scale`, accumulating into `report`.
/// Mirrors `apply_lora_to_linear` per module: `ΔW = (B·A)·(alpha/rank·strength)` at the factor dtype,
/// cast to the weight dtype and added — so the no-LoRA forward and the merged forward share one bf16
/// GEMM (no per-step residual). Multiple files accumulate because each reads the (already-merged)
/// weight back from `w`.
fn merge_one(
    w: &mut Weights,
    spec: &AdapterSpec,
    normalize: fn(&str) -> String,
    report: &mut WanLoraReport,
) -> Result<()> {
    let lw = Weights::from_file(&spec.path)?;
    if spec.kind == AdapterKind::Lokr || is_lokr(&lw) {
        // LoKr (sc-2393 — net-new; the reference Wan path is LoRA-only) merges through the same
        // in-place weight fold, with the delta reconstructed from Kronecker factors instead of B·A.
        return merge_one_lokr(w, &lw, spec.scale, normalize, report);
    }
    // Third-party LyCORIS (sc-3671): `lokr_*` / `hada_*` keys WITHOUT a `networkType=lokr` stamp
    // (kohya / ai-toolkit / lycoris-lib). `is_lokr` (peft) is handled above, so reaching here means
    // third-party; reconstruct per-module and merge like the peft path.
    if is_lokr_keys(&lw) {
        return merge_one_lokr_thirdparty(w, &lw, spec.scale, normalize, report);
    }
    if is_loha_keys(&lw) {
        return merge_one_loha_thirdparty(w, &lw, spec.scale, normalize, report);
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
        let parts = groups.entry(normalize(stem)).or_default();
        match role {
            Role::Down => parts.down = Some(lw.require(&key)?.clone()),
            Role::Up => parts.up = Some(lw.require(&key)?.clone()),
            Role::Alpha => parts.alpha = Some(read_alpha(lw.require(&key)?)?),
        }
    }

    // PEFT/diffusers `save_lora_adapter` files carry no per-target `.alpha` tensor — `lora_alpha`/`r`
    // (+ per-module overrides) live in the `lora_adapter_metadata` header blob (sc-5513). `None` for a
    // file without it (kohya / trainer files ship a `.alpha` tensor), in which case the per-target
    // `.alpha` or the factor rank is used exactly as before.
    let cfg = LoraAdapterMeta::from_metadata(lw.metadata(LORA_ADAPTER_METADATA_KEY));
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
        // Effective scaling: per-target `.alpha` tensor → `alpha_pattern`/`lora_alpha` blob → factor
        // rank (today's default). The denominator honors the blob `r`/`rank_pattern` when given
        // (always `> 0`), else the stored `down` leading dim (which equals it for a well-formed file).
        let (cfg_alpha, cfg_rank) = cfg.as_ref().map_or((None, None), |c| c.effective(&path));
        let rank = cfg_rank.map(|r| r as f64).unwrap_or(down.shape()[0] as f64);
        let alpha = parts.alpha.or(cfg_alpha).map(|a| a as f64).unwrap_or(rank);
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

/// Merge one LoKr file's deltas into the weight map `w` at `scale` (sc-2393 — net-new; the reference
/// Wan path is LoRA-only). Each module's `[out,in]` delta is reconstructed (f32, `alpha/rank` folded
/// in) from its Kronecker factors via the core `reconstruct_lokr_delta`, scaled by the user strength,
/// and folded into the weight (cast to its dtype) — the same in-place merge as the LoRA path, so the
/// no-adapter forward stays byte-identical and adapters accumulate by reading the merged weight back.
/// A target absent from this checkpoint is surfaced (skipped), never fatal.
fn merge_one_lokr(
    w: &mut Weights,
    lw: &Weights,
    scale: f32,
    normalize: fn(&str) -> String,
    report: &mut WanLoraReport,
) -> Result<()> {
    let file = parse_lokr(lw)?;
    for (raw_path, factors) in &file.groups {
        let path = normalize(raw_path);
        let wkey = format!("{path}.weight");
        let Some(base) = w.get(&wkey).cloned() else {
            report.skipped.push(path);
            continue;
        };
        // Reconstruct f32 (the SDXL merge precedent, sc-2640) — the merge casts to the weight dtype.
        let delta = file.delta(factors, base.shape(), Dtype::Float32)?;
        let delta = multiply(&delta, &scalar(scale).as_dtype(delta.dtype())?)?;
        let merged = add(&base, &delta.as_dtype(base.dtype())?)?;
        w.insert(wkey, merged);
        report.applied += 1;
    }
    Ok(())
}

/// Build the `flattened-stem → checkpoint-module-path` table from the expert weight map `w` (every
/// `‹path›.weight` key → `‹path›`), so a third-party LyCORIS file's kohya-flattened key resolves to a
/// Wan module (sc-3671). Dotted third-party keys instead go through [`normalize_wan_key`].
fn wan_module_table(w: &Weights) -> BTreeMap<String, String> {
    w.keys()
        .filter_map(|k| k.strip_suffix(".weight"))
        .map(|p| (p.replace('.', "_"), p.to_string()))
        .collect()
}

/// Resolve a third-party LoKr/LoHa raw module key to a Wan checkpoint module path: prefer the
/// flattened-stem table (kohya `lora_unet_…`), else the dotted-path `normalize` (the host's rename
/// map — [`normalize_wan_key`] for native Wan, [`normalize_vace_key`] for the diffusers VACE host —
/// which a dotted diffusers third-party file shares with the peft path).
fn resolve_wan_thirdparty(
    raw: &str,
    table: &BTreeMap<String, String>,
    normalize: fn(&str) -> String,
) -> String {
    resolve_lokr_path(raw, table)
        .map(str::to_string)
        .unwrap_or_else(|| normalize(raw))
}

/// Merge one third-party-reconstructed `[out,in]` delta into `w` at the resolved module path
/// (sc-3671): `W += δ·scale` cast to the weight dtype, the same fold as LoRA/peft-LoKr. A path with no
/// weight in this expert is surfaced (skipped), never fatal.
fn merge_wan_thirdparty_delta(
    w: &mut Weights,
    path: String,
    delta_at: impl FnOnce(&[i32]) -> Result<Array>,
    scale: f32,
    report: &mut WanLoraReport,
) -> Result<()> {
    let wkey = format!("{path}.weight");
    let Some(base) = w.get(&wkey).cloned() else {
        report.skipped.push(path);
        return Ok(());
    };
    let delta = delta_at(base.shape())?;
    let delta = multiply(&delta, &scalar(scale).as_dtype(delta.dtype())?)?;
    let merged = add(&base, &delta.as_dtype(base.dtype())?)?;
    w.insert(wkey, merged);
    report.applied += 1;
    Ok(())
}

/// Merge a third-party LyCORIS **LoKr** file (kohya/lycoris keys, per-module `.alpha`, no
/// `networkType` stamp) into `w` at `scale` (sc-3671). Reconstruction reuses the core
/// `ThirdPartyLokr::delta` (f32, lycoris per-module scale baked in); install is the same in-place
/// weight fold as the peft `merge_one_lokr`.
fn merge_one_lokr_thirdparty(
    w: &mut Weights,
    lw: &Weights,
    scale: f32,
    normalize: fn(&str) -> String,
    report: &mut WanLoraReport,
) -> Result<()> {
    let table = wan_module_table(w);
    for (raw, g) in &parse_lokr_thirdparty(lw)? {
        let path = resolve_wan_thirdparty(raw, &table, normalize);
        merge_wan_thirdparty_delta(w, path, |bs| g.delta(bs, Dtype::Float32), scale, report)?;
    }
    Ok(())
}

/// Merge a third-party LyCORIS **LoHa** file into `w` at `scale` (sc-3671). As
/// [`merge_one_lokr_thirdparty`] with the Hadamard reconstruction (`ThirdPartyLoha::delta`).
fn merge_one_loha_thirdparty(
    w: &mut Weights,
    lw: &Weights,
    scale: f32,
    normalize: fn(&str) -> String,
    report: &mut WanLoraReport,
) -> Result<()> {
    let table = wan_module_table(w);
    for (raw, g) in &parse_loha_thirdparty(lw)? {
        let path = resolve_wan_thirdparty(raw, &table, normalize);
        merge_wan_thirdparty_delta(w, path, |bs| g.delta(bs, Dtype::Float32), scale, report)?;
    }
    Ok(())
}

/// Emit a single, uniform warning for adapter targets that aren't present in the loaded checkpoint —
/// a *partial* skip, distinct from the hard "matched no module" error the model entries return. The
/// three Wan `Generator` load paths (`model.rs` ×2, `model_vace.rs`) share this so the message can't
/// drift (F-026); `eprintln!` is the only channel available at load time (no `Progress` callback, no
/// workspace logging facade). A no-op when nothing was skipped.
pub(crate) fn warn_skipped_adapters(model_id: &str, skipped: &[String]) {
    if skipped.is_empty() {
        return;
    }
    eprintln!(
        "{model_id}: {} adapter target(s) not present in this checkpoint, skipped: {skipped:?}",
        skipped.len()
    );
}

/// Merge every adapter in `specs` that targets `expert` into the expert weight map `w` (sc-2683 LoRA /
/// sc-2393 LoKr). Shared
/// specs (`moe_expert == None`) are applied first, then this expert's specific ones (`Some(expert)`),
/// mirroring the reference `(loras)+(loras_high/low)` order so a module hit by both accumulates in
/// the same order. LoRA and LoKr are dispatched per file by metadata / the spec kind; per-key skips
/// are reported, not fatal (the reference warns on skip). Returns the merge report; the caller
/// enforces the "matched nothing across both experts" error.
pub fn merge_wan_adapters(
    w: &mut Weights,
    specs: &[AdapterSpec],
    expert: MoeExpert,
) -> Result<WanLoraReport> {
    merge_adapters_into(w, specs, expert, normalize_wan_key)
}

/// Merge every adapter in `specs` onto the **diffusers-layout** Wan-VACE transformer weight map `w`
/// (sc-3439). The VACE DiT ([`crate::vace::WanVaceTransformer`]) is a single dense model (no MoE), so
/// it takes only **shared** (untagged) specs — `MoeExpert::High` is passed purely so the untagged
/// pass fires; a `moe_expert`-tagged spec is a misconfiguration the caller
/// ([`crate::model_vace`]) rejects before calling here. Identical merge math + format dispatch as
/// [`merge_wan_adapters`] (PEFT/kohya LoRA, peft LoKr, third-party LyCORIS LoKr/LoHa), differing only
/// in the key→module map: [`normalize_vace_key`] targets the diffusers `attn1/attn2.{to_*}` +
/// `ffn.net.0.proj`/`net.2` + `vace_blocks.*` host instead of the native Wan layout.
pub fn merge_vace_adapters(w: &mut Weights, specs: &[AdapterSpec]) -> Result<WanLoraReport> {
    merge_adapters_into(w, specs, MoeExpert::High, normalize_vace_key)
}

/// Per-expert VACE adapter merge for the **dual-expert (Wan2.2-A14B) VACE-Fun** — the high/low sibling
/// of [`merge_vace_adapters`]. Shared (untagged) specs merge onto this `expert`'s weight map and a spec
/// tagged for the *other* expert is skipped, exactly like [`merge_wan_adapters`]'s `MoeExpert` routing,
/// but on the diffusers-named VACE key surface ([`normalize_vace_key`]). The caller
/// ([`crate::model_vace`]) merges once per expert (`MoeExpert::High` onto `transformer/`, `Low` onto
/// `transformer_2/`) and enforces the "matched nothing across both experts" error.
pub fn merge_vace_adapters_expert(
    w: &mut Weights,
    specs: &[AdapterSpec],
    expert: MoeExpert,
) -> Result<WanLoraReport> {
    merge_adapters_into(w, specs, expert, normalize_vace_key)
}

/// Shared merge core for both the native Wan ([`merge_wan_adapters`]) and the diffusers VACE
/// ([`merge_vace_adapters`]) hosts — only the `normalize` key→module map differs. Pass 1: shared
/// (untagged) files. Pass 2: this `expert`'s specific files (the reference `(loras)+(loras_*)` order).
fn merge_adapters_into(
    w: &mut Weights,
    specs: &[AdapterSpec],
    expert: MoeExpert,
    normalize: fn(&str) -> String,
) -> Result<WanLoraReport> {
    let mut report = WanLoraReport::default();
    for spec in specs.iter().filter(|s| s.moe_expert.is_none()) {
        report.applicable += 1;
        merge_one(w, spec, normalize, &mut report)?;
    }
    for spec in specs.iter().filter(|s| s.moe_expert == Some(expert)) {
        report.applicable += 1;
        merge_one(w, spec, normalize, &mut report)?;
    }
    Ok(report)
}

// ============================================================================================
// Additive (UNMERGED) LoRA/LoKr/LoHa install (sc-10044)
// ============================================================================================
//
// The [`merge_wan_adapters`] path above FOLDS every adapter delta into a dense `Weights` map
// (`W += δ`) before [`WanTransformer::from_weights`] builds the model — which is why a *pre-quantized*
// (packed Q4/Q8) snapshot has nowhere to write the delta and the loader hard-errors (`model.rs:614`).
//
// The additive path below is the runtime alternative: it installs each adapter as a **forward-time
// residual** onto an already-built [`WanTransformer`] (any [`AdaptableHost`]) via
// [`Adapter`]-stacking, so the base weight `W` is used AS-IS — dense bf16 **or** packed Q4/Q8, never
// dequantized or mutated. Each linear then computes `base(x) + Σ scale·(x·A)·B` (LoRA) /
// `base(x) + Σ scale·x·ΔWᵀ` (LoKr/LoHa) — two small matmuls in compute precision, deferred (no out×in
// delta materialized, no per-gen dense reload). This makes LoRA usable on a packed snapshot with zero
// change to the base's quant.
//
// The math is byte-parity with the fold path: for LoRA the fold computes `ΔW = (B·A)·(alpha/rank·s)`
// and forwards `x·(W+ΔW)ᵀ = x·Wᵀ + (alpha/rank·s)·x·(B·A)ᵀ`; the residual computes
// `s·(x·Aᵀ)·(Bᵀ·(alpha/rank)) = (alpha/rank·s)·x·Aᵀ·Bᵀ` — identical (`(B·A)ᵀ = Aᵀ·Bᵀ`). This mirrors
// the core `install_lora_groups` factor convention exactly (`a = downᵀ`, `b = upᵀ`, `alpha/rank`
// folded into `b`, `scale = strength`) so a fold and an additive install of the same file agree
// within Metal matmul tolerance. LoKr/LoHa reconstruct the same `[out,in]` delta the fold uses and
// stack it as an [`Adapter::Lokr`] residual (`scale = strength`, `alpha/rank` already baked into the
// delta), with the base shape recovered from the host (works for a packed base too — see
// [`mlx_gen::adapters::AdaptableLinear::base_shape`]).
//
// Format coverage mirrors [`merge_one`]: PEFT/kohya LoRA, peft LoKr, third-party LyCORIS LoKr/LoHa.
// A target absent from (or outside the adaptable surface of) the host is surfaced (skipped), never
// fatal — matching the fold path's `skipped` reporting.

/// The adapter family a file resolves to, for the packed-snapshot routing (sc-10045/sc-10050).
/// `classify` reads the file's keys/metadata exactly as [`install_one_additive`] does, so a spec
/// routes the same way it will install: plain LoRA and **LoKr** both apply additively on a packed tier
/// (LoRA via low-rank factors, LoKr via the structured vec-trick, sc-10050); **LoHa** on a packed tier
/// is still deferred (sc-10051), so the loader rejects only LoHa up front rather than dequantizing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WanAdapterFamily {
    /// PEFT / kohya LoRA (`lora_A/B` or `lora_down/up`) — additive on any tier.
    Lora,
    /// LoKr (peft `networkType=lokr`, or third-party `lokr_*` keys) — additive on any tier: the
    /// **structured deferred** Kronecker install on a packed base (sc-10050), the materialized-delta
    /// residual / fold on a dense base.
    Lokr,
    /// LoHa (third-party `hada_*` keys) — deferred on a packed tier to sc-10051; dense-only for now.
    Loha,
}

/// Classify one adapter file into a [`WanAdapterFamily`] using the SAME precedence
/// [`install_one_additive`]/[`merge_one`] dispatch on: an explicit `AdapterKind::Lokr` spec kind, a
/// peft LoKr metadata stamp ([`is_lokr`]), third-party LoKr keys ([`is_lokr_keys`]), or LoHa keys
/// ([`is_loha_keys`]) — anything else is plain LoRA. LoKr precedes LoHa (a file is never both). Reads
/// the file once (the caller reuses the load).
fn classify_family(lw: &Weights, kind: AdapterKind) -> WanAdapterFamily {
    if kind == AdapterKind::Lokr || is_lokr(lw) || is_lokr_keys(lw) {
        WanAdapterFamily::Lokr
    } else if is_loha_keys(lw) {
        WanAdapterFamily::Loha
    } else {
        WanAdapterFamily::Lora
    }
}

/// Reject a **LoHa** (LyCORIS Hadamard) adapter on a **pre-quantized (packed Q4/Q8)** Wan snapshot
/// with an explicit, actionable **typed** error (sc-10045; narrowed by sc-10050; finalized by
/// sc-10051, Option A). Plain LoRA AND LoKr now install additively onto a packed base with no dequant
/// — LoRA via its low-rank factors, LoKr via the structured deferred-Kronecker vec-trick
/// ([`install_one_lokr_additive`], sc-10050). LoHa is the **only** family that cannot: its delta is
/// `ΔW = (B₁A₁) ⊙ (B₂A₂)`, and the element-wise (Hadamard) product entangles the two low-rank
/// branches so there is **no allocation-free deferred form** — evaluating it requires materializing
/// the full `[out,in]` bf16 delta (~dense-weight-sized, ~28 GB/expert on a 14B Wan tier). On a packed
/// tier that means holding packed-W (~7 GB/expert) **plus** that dense delta, negating the whole
/// low-memory point of the quantized tier. So rather than silently OOM or silently dequantize, the
/// loader **steers the user to the bf16 tier** (where LoHa folds into the dense weight in place) via a
/// typed [`Error::Unsupported`] — which bridges 1:1 to [`gen_core::Error::Unsupported`] so the worker
/// can surface actionable guidance instead of an opaque failure (epic 3720, F-008).
///
/// `model_id` is the registry id for the message. Scans every non-empty spec (each file loaded once)
/// and errors on the first LoHa; a LoRA/LoKr set passes through untouched. No-op on a dense base (the
/// caller only calls this when the snapshot is packed).
pub fn reject_loha_on_packed(model_id: &str, specs: &[AdapterSpec]) -> Result<()> {
    for spec in specs {
        let lw = Weights::from_file(&spec.path)?;
        if classify_family(&lw, spec.kind) == WanAdapterFamily::Loha {
            return Err(Error::Unsupported(format!(
                "{model_id}: this is a LoHa (LyCORIS Hadamard) adapter, which cannot run on a \
                 quantized (packed Q4/Q8) Wan tier. LoHa's delta is a Hadamard product of two \
                 low-rank branches, ΔW = (B₁A₁) ⊙ (B₂A₂); the element-wise product entangles the \
                 factors so there is no deferred/additive form — applying it would have to \
                 materialize the full dense weight-sized delta (~28 GB/expert on a 14B tier) on top \
                 of the packed weights, defeating the quantized tier's memory savings. Use the bf16 \
                 tier for this adapter (where LoHa merges into the dense weight in place), or pick a \
                 plain LoRA / LoKr adapter (which apply additively on any tier, sc-10050). Tracked in \
                 sc-10051 (epic 10043). Offending file: {}",
                spec.path.display()
            )));
        }
    }
    Ok(())
}

/// Install every adapter in `specs` that targets `expert` onto the native-Wan [`AdaptableHost`]
/// `host` (a [`WanTransformer`](crate::transformer)) as **forward-time residuals** — the unmerged,
/// quant-agnostic sibling of [`merge_wan_adapters`] (sc-10044). The base weights are used AS-IS
/// (dense bf16 or packed Q4/Q8, never dequantized). Shared (`moe_expert == None`) specs install
/// first, then this `expert`'s specific ones, mirroring the reference `(loras)+(loras_*)` order.
/// Returns the same [`WanLoraReport`] the fold path returns; the caller enforces the
/// "matched nothing across both experts" error.
///
/// **Packed-snapshot routing (sc-10045 / sc-10050):** on a pre-quantized base the caller first calls
/// [`reject_loha_on_packed`] so only plain LoRA and LoKr reach here; LoRA installs via its low-rank
/// factors and LoKr via the **structured deferred-Kronecker** vec-trick ([`install_one_lokr_additive`],
/// sc-10050) — both with the base staying packed, no `[out,in]` delta materialized. LoHa on a packed
/// tier stays deferred (sc-10051). On a dense base the fold path ([`merge_wan_adapters`]) is used
/// instead, so LoKr/LoHa keep working there unchanged.
pub fn apply_wan_adapters_additive(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
    expert: MoeExpert,
) -> Result<WanLoraReport> {
    apply_additive_into(host, specs, expert, normalize_wan_key)
}

/// Shared additive-install core (the forward-time-residual analog of [`merge_adapters_into`]) — Pass 1
/// shared (untagged) files, Pass 2 this `expert`'s specific files. `normalize` maps a file's raw key
/// to the host's module path ([`normalize_wan_key`] for the native Wan DiT).
fn apply_additive_into(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
    expert: MoeExpert,
    normalize: fn(&str) -> String,
) -> Result<WanLoraReport> {
    let mut report = WanLoraReport::default();
    for spec in specs.iter().filter(|s| s.moe_expert.is_none()) {
        report.applicable += 1;
        install_one_additive(host, spec, normalize, &mut report)?;
    }
    for spec in specs.iter().filter(|s| s.moe_expert == Some(expert)) {
        report.applicable += 1;
        install_one_additive(host, spec, normalize, &mut report)?;
    }
    Ok(report)
}

/// Install one adapter file onto `host` as residuals at `spec.scale` (the additive analog of
/// [`merge_one`]) — same per-file format dispatch (peft/kohya LoRA, peft LoKr, third-party LyCORIS
/// LoKr/LoHa), same normalized module paths, but pushing an [`Adapter`] rather than folding `W += δ`.
fn install_one_additive(
    host: &mut impl AdaptableHost,
    spec: &AdapterSpec,
    normalize: fn(&str) -> String,
    report: &mut WanLoraReport,
) -> Result<()> {
    let lw = Weights::from_file(&spec.path)?;
    if spec.kind == AdapterKind::Lokr || is_lokr(&lw) {
        return install_one_lokr_additive(host, &lw, spec.scale, normalize, report);
    }
    if is_lokr_keys(&lw) {
        return install_one_lokr_thirdparty_additive(host, &lw, spec.scale, normalize, report);
    }
    if is_loha_keys(&lw) {
        return install_one_loha_thirdparty_additive(host, &lw, spec.scale, normalize, report);
    }
    install_one_lora_additive(host, &lw, spec.scale, normalize, report)
}

/// Push `adapter` onto the host module at the (already-normalized) native path `path`. A path outside
/// the host's adaptable surface (e.g. a non-block target, or a block index beyond this checkpoint) is
/// surfaced in `report.skipped`, never fatal — mirroring the fold path's per-target skip.
fn push_at(
    host: &mut impl AdaptableHost,
    path: String,
    adapter: Adapter,
    report: &mut WanLoraReport,
) {
    let parts: Vec<&str> = path.split('.').collect();
    match host.adaptable_mut(&parts) {
        Some(lin) => {
            lin.push(adapter);
            report.applied += 1;
        }
        None => report.skipped.push(path),
    }
}

/// Install a PEFT/kohya LoRA file as residuals. Mirrors [`merge_one`]'s grouping + the core
/// `install_lora_groups` factor convention: `a = downᵀ [in,rank]`, `b = upᵀ [rank,out]` with
/// `alpha/rank` folded into `b`, and `scale = strength` — so `scale·(x·A)·B` reproduces the fold's
/// `(alpha/rank·strength)·x·(B·A)ᵀ` within tolerance, with `W` untouched (dense OR packed).
fn install_one_lora_additive(
    host: &mut impl AdaptableHost,
    lw: &Weights,
    strength: f32,
    normalize: fn(&str) -> String,
    report: &mut WanLoraReport,
) -> Result<()> {
    let mut groups: BTreeMap<String, LoraParts> = BTreeMap::new();
    for key in lw.keys().map(str::to_string).collect::<Vec<_>>() {
        let Some((stem, role)) = SUFFIXES
            .iter()
            .find_map(|(suf, role)| key.strip_suffix(suf).map(|s| (s, *role)))
        else {
            continue;
        };
        let parts = groups.entry(normalize(stem)).or_default();
        match role {
            Role::Down => parts.down = Some(lw.require(&key)?.clone()),
            Role::Up => parts.up = Some(lw.require(&key)?.clone()),
            Role::Alpha => parts.alpha = Some(read_alpha(lw.require(&key)?)?),
        }
    }

    let cfg = LoraAdapterMeta::from_metadata(lw.metadata(LORA_ADAPTER_METADATA_KEY));
    for (path, parts) in groups {
        let (Some(down), Some(up)) = (parts.down, parts.up) else {
            report.skipped.push(path);
            continue;
        };
        // `down` is [rank, in], `up` is [out, rank]. The residual form is `(x·A)·B` with A = downᵀ
        // [in, rank] and B = upᵀ [rank, out]; the `alpha/rank` scale is folded into B (matching the
        // core `install_lora_groups`), and the user strength stays as the `Adapter::Lora` scale.
        let a = down.t();
        let mut b = up.t();
        if a.shape().len() != 2 || b.shape().len() != 2 {
            return Err(format!(
                "wan additive LoRA at '{path}' has non-2-D factors (down {:?}, up {:?})",
                down.shape(),
                up.shape()
            )
            .into());
        }
        let (cfg_alpha, cfg_rank) = cfg.as_ref().map_or((None, None), |c| c.effective(&path));
        let factor_rank = a.shape()[1] as f64; // r
        if factor_rank == 0.0 {
            return Err(format!("wan additive LoRA at '{path}' has zero rank").into());
        }
        // Fold `alpha/rank` into B when an alpha is present (per-target `.alpha` → blob), matching the
        // fold path's `eff = alpha/rank·strength` split (the strength stays the residual scale).
        if let Some(alpha) = parts.alpha.or(cfg_alpha) {
            let rank = cfg_rank.map(|r| r as f64).unwrap_or(factor_rank);
            let ratio = (alpha as f64 / rank) as f32;
            b = multiply(&b, &scalar(ratio).as_dtype(b.dtype())?)?;
        }
        push_at(
            host,
            path,
            Adapter::Lora {
                a,
                b,
                scale: strength,
            },
            report,
        );
    }
    Ok(())
}

/// Install a peft LoKr file as residuals (sc-2393/sc-10044/sc-10050). Per target, choose the install
/// by the base's quantization:
/// - **Packed (Q4/Q8)** base (sc-10050 / epic 10043): a **structured, deferred** Kronecker residual
///   ([`Adapter::LokrStructured`]) via the vec-trick — `scale·vec(w1·reshape(x)·w2ᵀ)` — so the full
///   `[out,in]` delta is NEVER materialized and the base stays packed at plain-LoRA memory cost. The
///   `alpha/rank` factor is baked into the structured `w2` (the user strength stays the residual scale).
/// - **Dense** base (unchanged, sc-2393/sc-10044): the fork-parity materialized `[out,in]` delta
///   (bf16, `alpha/rank` folded) stacked as an [`Adapter::Lokr`] — plenty of memory on dense, and the
///   established byte-parity path.
///
/// A non-deferrable LoKr variant (a **tucker/CP** `w2`, i.e. conv-only — the Wan DiT adapter surface
/// is all Linear, so this never fires) falls back to materialization on a **dense** base, or an
/// explicit typed error on a **packed** base (never a silent OOM / wrong result). A target outside the
/// adaptable surface is skipped.
fn install_one_lokr_additive(
    host: &mut impl AdaptableHost,
    lw: &Weights,
    strength: f32,
    normalize: fn(&str) -> String,
    report: &mut WanLoraReport,
) -> Result<()> {
    let file = parse_lokr(lw)?;
    for (raw_path, factors) in &file.groups {
        let path = normalize(raw_path);
        let parts: Vec<&str> = path.split('.').collect();
        let Some(lin) = host.adaptable_mut(&parts) else {
            report.skipped.push(path);
            continue;
        };
        let base_shape = lin.base_shape();
        if lin.is_quantized() {
            // Structured deferred path (packed base). The FULL scale `(alpha/rank)·strength` is baked
            // into `w2` — the structured residual carries no separate scale, so both the `alpha/rank`
            // fold (as in the dense delta) and the user strength must live in the factors here.
            match build_lokr_factors(
                (file.alpha / file.rank) * strength,
                &base_shape,
                factors.get("lokr_w1"),
                factors.get("lokr_w1_a"),
                factors.get("lokr_w1_b"),
                factors.get("lokr_w2"),
                None, // peft LoKr never carries a tucker factor (conv-only)
                factors.get("lokr_w2_a"),
                factors.get("lokr_w2_b"),
                Dtype::Bfloat16,
            )? {
                Some(structured) => {
                    lin.push(Adapter::LokrStructured {
                        factors: structured,
                    });
                    report.applied += 1;
                }
                None => {
                    return Err(non_deferrable_lokr_error(&path));
                }
            }
        } else {
            // Dense: the established materialized-delta residual (bf16, `alpha/rank` folded in).
            let delta = file.delta(factors, &base_shape, Dtype::Bfloat16)?;
            lin.push(Adapter::Lokr {
                delta,
                scale: strength,
            });
            report.applied += 1;
        }
    }
    Ok(())
}

/// The clear, actionable error for a LoKr variant that is genuinely NOT deferrable via the vec-trick
/// on a **packed** base (only the conv-only tucker/CP `w2` form — which the all-Linear Wan DiT adapter
/// surface never produces). Rather than a silent OOM or a wrong result, point the user at the bf16
/// tier (where it materializes into the dense weight). Dense bases fall back to materialization instead.
fn non_deferrable_lokr_error(path: &str) -> mlx_gen::Error {
    format!(
        "LoKr at '{path}' uses a tucker/CP (conv) factorization that cannot be applied on a \
         quantized (packed Q4/Q8) Wan tier without materializing the full delta — use the bf16 tier \
         (where it merges into the dense weight). This form does not occur for the Wan DiT's Linear \
         adapter targets (sc-10050 / epic 10043)."
    )
    .into()
}

/// Install a third-party LyCORIS **LoKr** file as residuals (sc-3671/sc-10044). As
/// [`install_one_lokr_additive`] but with per-module lycoris factors + key resolution (kohya-flattened
/// stem table, else the dotted `normalize`). The reconstructed delta bakes in the lycoris per-module
/// scale; the user strength stays the residual scale.
fn install_one_lokr_thirdparty_additive(
    host: &mut impl AdaptableHost,
    lw: &Weights,
    strength: f32,
    normalize: fn(&str) -> String,
    report: &mut WanLoraReport,
) -> Result<()> {
    let table = host_module_table(host);
    for (raw, g) in &parse_lokr_thirdparty(lw)? {
        let path = resolve_wan_thirdparty(raw, &table, normalize);
        let parts: Vec<&str> = path.split('.').collect();
        let Some(lin) = host.adaptable_mut(&parts) else {
            report.skipped.push(path);
            continue;
        };
        let base_shape = lin.base_shape();
        if lin.is_quantized() {
            // Structured deferred path on a packed base (sc-10050): the lycoris per-module scale is
            // baked into the small `w2` factor; no `[out,in]` delta is materialized. A tucker/CP `w2`
            // (`lokr_t2`) has no 2-D form (`Ok(None)`) — the all-Linear Wan surface never hits it, but
            // guard it with the same clear error rather than a silent materialize.
            match g.factors(strength, &base_shape, Dtype::Bfloat16)? {
                Some(structured) => {
                    lin.push(Adapter::LokrStructured {
                        factors: structured,
                    });
                    report.applied += 1;
                }
                None => return Err(non_deferrable_lokr_error(&path)),
            }
        } else {
            // Dense: the established materialized-delta residual (lycoris scale baked in).
            let delta = g.delta(&base_shape, Dtype::Bfloat16)?;
            lin.push(Adapter::Lokr {
                delta,
                scale: strength,
            });
            report.applied += 1;
        }
    }
    Ok(())
}

/// Install a third-party LyCORIS **LoHa** file as residuals (sc-3671/sc-10044) — as
/// [`install_one_lokr_thirdparty_additive`] with the Hadamard reconstruction.
fn install_one_loha_thirdparty_additive(
    host: &mut impl AdaptableHost,
    lw: &Weights,
    strength: f32,
    normalize: fn(&str) -> String,
    report: &mut WanLoraReport,
) -> Result<()> {
    let table = host_module_table(host);
    for (raw, g) in &parse_loha_thirdparty(lw)? {
        let path = resolve_wan_thirdparty(raw, &table, normalize);
        install_thirdparty_delta(
            host,
            path,
            |bs| g.delta(bs, Dtype::Bfloat16),
            strength,
            report,
        )?;
    }
    Ok(())
}

/// Reconstruct a third-party `[out,in]` delta at the host target's base shape and stack it as an
/// [`Adapter::Lokr`] residual at `strength` — the additive analog of [`merge_wan_thirdparty_delta`].
/// A path outside the adaptable surface is surfaced (skipped), never fatal.
fn install_thirdparty_delta(
    host: &mut impl AdaptableHost,
    path: String,
    delta_at: impl FnOnce(&[i32]) -> Result<Array>,
    strength: f32,
    report: &mut WanLoraReport,
) -> Result<()> {
    let parts: Vec<&str> = path.split('.').collect();
    let Some(base_shape) = host.adaptable_mut(&parts).map(|lin| lin.base_shape()) else {
        report.skipped.push(path);
        return Ok(());
    };
    let delta = delta_at(&base_shape)?;
    push_at(
        host,
        path,
        Adapter::Lokr {
            delta,
            scale: strength,
        },
        report,
    );
    Ok(())
}

/// Build the `flattened-stem → native-module-path` table from the host's adaptable surface (every
/// [`AdaptableHost::adaptable_paths`] entry, dots→underscores), so a third-party LyCORIS file's
/// kohya-flattened key resolves to a native Wan module (the additive analog of [`wan_module_table`],
/// which reads the weight map — here we read the host, which is already built).
fn host_module_table(host: &impl AdaptableHost) -> BTreeMap<String, String> {
    host.adaptable_paths()
        .into_iter()
        .map(|p| (p.replace('.', "_"), p))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{all_close, array_eq};
    use std::path::PathBuf;

    /// sc-4986 — retire the "does a lightx2v Lightning LoRA actually load through mlx-gen?" risk by
    /// running the **real** distill-LoRA file's keys through the genuine [`normalize_wan_key`] and
    /// asserting every module resolves to a valid Wan DiT target (`blocks.N.{self,cross}_attn.{q,k,v,o}`
    /// or `blocks.N.ffn.{fc1,fc2}`). `#[ignore]` — needs the downloaded LoRA:
    /// ```text
    /// WAN_LIGHTNING_LORA="$HOME/.cache/huggingface/hub/models--lightx2v--Wan2.2-Lightning/snapshots/\
    /// 18bccf8884ec0a078eed79785eb4ef13ea16ce1e/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/\
    /// high_noise_model.safetensors" \
    ///   cargo test -p mlx-gen-wan lightning_lora_keys_normalize -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs a downloaded lightx2v Wan2.2 Lightning LoRA (WAN_LIGHTNING_LORA)"]
    fn lightning_lora_keys_normalize_to_wan_dit_targets() {
        let path = match std::env::var_os("WAN_LIGHTNING_LORA") {
            Some(p) => PathBuf::from(p),
            None => {
                eprintln!("skip: set WAN_LIGHTNING_LORA to a downloaded lightx2v Lightning LoRA");
                return;
            }
        };
        let lw = Weights::from_file(&path).expect("read LoRA safetensors");

        // Collapse factor keys (.lora_down/.lora_up/.alpha/.lora_A/.lora_B) to distinct module paths,
        // exactly as the merge loop keys its parts, then normalize each through the real mapper.
        let mut modules = std::collections::BTreeSet::new();
        for key in lw.keys() {
            if let Some((stem, _role)) = SUFFIXES
                .iter()
                .find_map(|(suf, r)| key.strip_suffix(suf).map(|s| (s, *r)))
            {
                modules.insert(normalize_wan_key(stem));
            }
        }
        assert!(!modules.is_empty(), "no LoRA factor keys found in {path:?}");

        // Every normalized module must hit the native converted-Wan DiT namespace. Anything else
        // would fold onto nothing (silent no-op) — the exact failure we are de-risking.
        let valid = |m: &str| -> bool {
            let Some(rest) = m.strip_prefix("blocks.") else {
                // a handful of non-block targets the distill LoRA may also touch
                return matches!(m, "head.head" | "patch_embedding_proj")
                    || m.starts_with("text_embedding_")
                    || m.starts_with("time_embedding_")
                    || m == "time_projection";
            };
            let Some((_n, tail)) = rest.split_once('.') else {
                return false;
            };
            matches!(
                tail,
                "self_attn.q"
                    | "self_attn.k"
                    | "self_attn.v"
                    | "self_attn.o"
                    | "cross_attn.q"
                    | "cross_attn.k"
                    | "cross_attn.v"
                    | "cross_attn.o"
                    | "cross_attn.k_img"
                    | "cross_attn.v_img"
                    | "ffn.fc1"
                    | "ffn.fc2"
            )
        };
        let bad: Vec<&String> = modules.iter().filter(|m| !valid(m)).collect();
        println!(
            "[lightning lora] {} distinct modules; {} resolve to valid Wan DiT targets, {} unmatched",
            modules.len(),
            modules.len() - bad.len(),
            bad.len()
        );
        if !bad.is_empty() {
            println!("[lightning lora] UNMATCHED (would fold onto nothing): {bad:?}");
        }
        // Sample the resolved targets for the log.
        for m in modules.iter().take(4) {
            println!("[lightning lora]   e.g. {m}");
        }
        assert!(
            bad.is_empty(),
            "{} Lightning LoRA module(s) normalize to non-DiT targets and would silently no-op",
            bad.len()
        );
    }

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

    #[test]
    fn normalize_vace_passes_diffusers_and_renames_native() {
        // Diffusers names (the host layout) pass through after prefix strip — incl. vace_blocks.
        for (raw, want) in [
            ("diffusion_model.blocks.0.attn1.to_q", "blocks.0.attn1.to_q"),
            (
                "diffusion_model.blocks.3.attn2.to_out.0",
                "blocks.3.attn2.to_out.0",
            ),
            (
                "diffusion_model.blocks.0.ffn.net.0.proj",
                "blocks.0.ffn.net.0.proj",
            ),
            ("blocks.2.ffn.net.2", "blocks.2.ffn.net.2"),
            (
                "diffusion_model.vace_blocks.0.attn1.to_v",
                "vace_blocks.0.attn1.to_v",
            ),
            (
                "diffusion_model.vace_blocks.0.proj_in",
                "vace_blocks.0.proj_in",
            ),
            (
                "diffusion_model.vace_blocks.7.proj_out",
                "vace_blocks.7.proj_out",
            ),
        ] {
            assert_eq!(normalize_vace_key(raw), want, "passthrough {raw}");
        }
        // Native Wan spellings (musubi / diffusion-pipe) → diffusers (the diffusers loader's map).
        for (raw, want) in [
            (
                "diffusion_model.blocks.0.self_attn.q",
                "blocks.0.attn1.to_q",
            ),
            (
                "diffusion_model.blocks.5.self_attn.o",
                "blocks.5.attn1.to_out.0",
            ),
            (
                "diffusion_model.blocks.0.cross_attn.k",
                "blocks.0.attn2.to_k",
            ),
            ("diffusion_model.blocks.0.ffn.0", "blocks.0.ffn.net.0.proj"),
            ("diffusion_model.blocks.2.ffn.2", "blocks.2.ffn.net.2"),
            // VACE block native hint projections → proj_in/proj_out (diffusers leaves these alone;
            // we complete the map for native-trained VACE LoRAs).
            (
                "diffusion_model.vace_blocks.0.before_proj",
                "vace_blocks.0.proj_in",
            ),
            (
                "diffusion_model.vace_blocks.3.after_proj",
                "vace_blocks.3.proj_out",
            ),
            (
                "diffusion_model.vace_blocks.1.self_attn.v",
                "vace_blocks.1.attn1.to_v",
            ),
            // Globals + alternate prefixes.
            (
                "diffusion_model.time_projection.1",
                "condition_embedder.time_proj",
            ),
            ("model.diffusion_model.head.head", "proj_out"),
            (
                "base_model.model.text_embedding.0",
                "condition_embedder.text_embedder.linear_1",
            ),
            (
                "diffusion_model.time_embedding.2",
                "condition_embedder.time_embedder.linear_2",
            ),
        ] {
            assert_eq!(normalize_vace_key(raw), want, "rename {raw}");
        }
    }

    #[test]
    fn normalize_vace_matches_reference_golden() {
        // sc-3439 parity gate for the diffusers-named VACE key→module map. The fixture
        // (tools/dump_wanvace_lora_keys.py) takes the base-block + global native→diffusers mappings
        // **authoritatively from the diffusers loader** (`_convert_non_diffusers_wan_lora_to_diffusers`)
        // and the vace_blocks + diffusers-passthrough entries from the shared rename rule, every target
        // verified to be a real module in the cached `Wan2.1-VACE-1.3B-diffusers` checkpoint. The Rust
        // `normalize_vace_key` must reproduce each mapping — the load-bearing piece of the VACE merge.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/wanvace_lora_keys.json"
        );
        let text = std::fs::read_to_string(path).expect("read wanvace_lora_keys.json fixture");
        let map: BTreeMap<String, String> = serde_json::from_str(&text).expect("parse fixture");
        assert!(
            map.len() >= 80,
            "fixture should cover the VACE LoRA surface (got {})",
            map.len()
        );
        for (raw, expected) in &map {
            assert_eq!(
                &normalize_vace_key(raw),
                expected,
                "normalize_vace_key({raw}) must match the diffusers VACE key map"
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

    /// Write a peft LoKr file for `blocks.0.self_attn.q` ([16,8] = kron(w1[4,2], w2[4,4])) with the
    /// given `alpha`/`rank` in metadata (sc-10044 additive tests). Deterministic factor values.
    fn write_lokr(name: &str, alpha: f32, rank: f32) -> PathBuf {
        use std::collections::HashMap;
        let w1 = Array::from_slice(
            &(0..8)
                .map(|i| (i as f32 * 0.03).sin() * 0.1)
                .collect::<Vec<_>>(),
            &[4, 2],
        );
        let w2 = Array::from_slice(
            &(0..16)
                .map(|i| (i as f32 * 0.05).cos() * 0.1)
                .collect::<Vec<_>>(),
            &[4, 4],
        );
        let meta = HashMap::from([
            ("networkType".to_string(), "lokr".to_string()),
            ("alpha".to_string(), alpha.to_string()),
            ("rank".to_string(), rank.to_string()),
        ]);
        let path = tmp(name);
        Array::save_safetensors(
            vec![
                ("blocks.0.self_attn.q.lokr_w1", &w1),
                ("blocks.0.self_attn.q.lokr_w2", &w2),
            ],
            Some(&meta),
            &path,
        )
        .unwrap();
        path
    }

    /// A LoKr-kind [`AdapterSpec`] (the [`spec`] helper defaults to `AdapterKind::Lora`).
    fn lokr_spec(path: PathBuf, scale: f32) -> AdapterSpec {
        AdapterSpec {
            kind: AdapterKind::Lokr,
            ..spec(path, scale, None)
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
    fn merge_honors_lora_adapter_metadata_alpha() {
        // sc-5513: a diffusers / PEFT `save_lora_adapter` LoRA carries NO per-target `.alpha` tensor —
        // the scaling lives in the `lora_adapter_metadata` blob. With `lora_alpha = 16`, `r = 8` (the
        // factor's true rank) the Wan merge must fold `(16/8) = 2.0`, not the pre-sc-5513 `alpha = rank`
        // default (factor 1.0).
        use std::collections::HashMap;
        let rank = 8;
        // One target, factor rank 8 (= the blob `r`): A [8, 8], B [16, 8] for the [16,8] base q weight.
        let a = Array::from_slice(
            &(0..rank * 8)
                .map(|i| (i as f32 * 0.001 + 0.1).sin() * 0.02)
                .collect::<Vec<_>>(),
            &[rank, 8],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let b = Array::from_slice(
            &(0..16 * rank)
                .map(|i| (i as f32 * 0.0007 + 0.1).cos() * 0.02)
                .collect::<Vec<_>>(),
            &[16, rank],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let path = tmp("merge_meta_alpha.safetensors");
        // Deliberately NO `.alpha` tensor — the scaling must come from the blob.
        let meta = HashMap::from([(
            "lora_adapter_metadata".to_string(),
            r#"{"lora_alpha": 16, "r": 8}"#.to_string(),
        )]);
        Array::save_safetensors(
            vec![
                ("diffusion_model.blocks.0.self_attn.q.lora_A.weight", &a),
                ("diffusion_model.blocks.0.self_attn.q.lora_B.weight", &b),
            ],
            Some(&meta),
            &path,
        )
        .unwrap();

        let mut w = synthetic_weights();
        let report = merge_wan_adapters(&mut w, &[spec(path, 1.0, None)], MoeExpert::High).unwrap();
        assert_eq!(report.applied, 1);

        // Reference: W += (B·A)·(alpha/rank = 2.0), folded at the factor dtype like the merge does.
        let base = synthetic_weights();
        let q_base = base.require("blocks.0.self_attn.q.weight").unwrap();
        let delta = matmul(&b, &a).unwrap();
        let two = scalar(2.0f32).as_dtype(delta.dtype()).unwrap();
        let want = add(
            q_base,
            multiply(&delta, &two)
                .unwrap()
                .as_dtype(q_base.dtype())
                .unwrap(),
        )
        .unwrap();
        let got = w.require("blocks.0.self_attn.q.weight").unwrap();
        assert!(
            array_eq(got, &want, false).unwrap().item::<bool>(),
            "metadata-alpha merge must fold (16/8)·strength = 2.0"
        );
        // The pre-sc-5513 default (alpha = rank = 8 ⇒ factor 1.0) would diverge by a full factor of 2.
        let one_want = add(q_base, delta.as_dtype(q_base.dtype()).unwrap()).unwrap();
        assert!(
            !array_eq(got, &one_want, false).unwrap().item::<bool>(),
            "metadata alpha must differ from the alpha=rank default"
        );
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
    fn lokr_merge_matches_reconstruct_and_scale_zero_is_noop() {
        // sc-2393: LoKr merges through the same in-place fold. `blocks.0.self_attn.q` is [16,8] =
        // kron(w1[4,2], w2[4,4]); the merged weight must equal W + (reconstruct·scale).astype(W.dtype),
        // and scale 0 must be a bit-exact no-op.
        use mlx_gen::adapters::reconstruct_lokr_delta;
        use std::collections::HashMap;

        let w1 = Array::from_slice(
            &(0..8)
                .map(|i| (i as f32 * 0.03).sin() * 0.1)
                .collect::<Vec<_>>(),
            &[4, 2],
        );
        let w2 = Array::from_slice(
            &(0..16)
                .map(|i| (i as f32 * 0.05).cos() * 0.1)
                .collect::<Vec<_>>(),
            &[4, 4],
        );
        let (alpha, rank) = (4.0f32, 4.0f32); // alpha/rank = 1.0
        let mut meta = HashMap::new();
        meta.insert("networkType".to_string(), "lokr".to_string());
        meta.insert("alpha".to_string(), alpha.to_string());
        meta.insert("rank".to_string(), rank.to_string());
        let lokr_path = tmp("lokr.safetensors");
        Array::save_safetensors(
            vec![
                ("blocks.0.self_attn.q.lokr_w1", &w1),
                ("blocks.0.self_attn.q.lokr_w2", &w2),
            ],
            Some(&meta),
            &lokr_path,
        )
        .unwrap();

        let scale = 0.5f32;
        let mut w = synthetic_weights();
        let report = merge_wan_adapters(
            &mut w,
            &[AdapterSpec {
                kind: AdapterKind::Lokr,
                ..spec(lokr_path, scale, None)
            }],
            MoeExpert::High,
        )
        .unwrap();
        assert_eq!(report.applied, 1);
        assert!(report.skipped.is_empty());

        let base = synthetic_weights();
        let q_base = base.require("blocks.0.self_attn.q.weight").unwrap();
        let delta = reconstruct_lokr_delta(
            alpha,
            rank,
            q_base.shape(),
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            Dtype::Float32,
        )
        .unwrap();
        let delta = multiply(&delta, scalar(scale).as_dtype(delta.dtype()).unwrap()).unwrap();
        let want = add(q_base, delta.as_dtype(q_base.dtype()).unwrap()).unwrap();
        let got = w.require("blocks.0.self_attn.q.weight").unwrap();
        assert!(
            array_eq(got, &want, false).unwrap().item::<bool>(),
            "merged LoKr weight must be bit-exact to W + (reconstruct·scale).astype(W.dtype)"
        );

        // scale 0 → the merged weight is bit-identical to the base.
        let mut w0 = synthetic_weights();
        merge_wan_adapters(
            &mut w0,
            &[AdapterSpec {
                kind: AdapterKind::Lokr,
                ..spec(tmp("lokr.safetensors"), 0.0, None)
            }],
            MoeExpert::High,
        )
        .unwrap();
        assert!(
            array_eq(
                w0.require("blocks.0.self_attn.q.weight").unwrap(),
                q_base,
                false
            )
            .unwrap()
            .item::<bool>(),
            "scale-0 LoKr merge must be a bit-exact no-op"
        );
    }

    /// sc-3671: a third-party (non-peft / lycoris) LoKr **and** LoHa file merges into the Wan weight
    /// map via the same `merge_wan_adapters` path (detected by keys), reconstructing the lycoris
    /// reference delta. Base weight = 0 so the merged weight equals `ΔW` exactly (the fixtures from
    /// `<repo>/tests/fixtures`, generated through `~/mlx-flux-venv`). The fixture module "proj" stands
    /// in for a Wan checkpoint module; `wan_module_table` resolves the `lycoris_proj` key to it.
    #[test]
    fn thirdparty_lycoris_merges_against_reference() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        for (dir, stem) in [
            ("sc3642_lokr", "linear_w1full_w2lr"),
            ("sc3643_loha", "linear"),
        ] {
            let base = root.join("tests/fixtures").join(dir);
            let exp =
                Weights::from_file(base.join(format!("{stem}.expected.safetensors"))).unwrap();
            let want = exp.require("proj").unwrap();
            // Base weight map: a single f32 zero "proj.weight" of the delta's shape.
            let zero = Array::zeros::<f32>(want.shape()).unwrap();
            let base_path = tmp(&format!("wan_tp_base_{stem}.safetensors"));
            Array::save_safetensors(vec![("proj.weight", &zero)], None, &base_path).unwrap();
            let mut w = Weights::from_file(&base_path).unwrap();

            let report = merge_wan_adapters(
                &mut w,
                &[spec(base.join(format!("{stem}.safetensors")), 1.0, None)],
                MoeExpert::High,
            )
            .unwrap();
            assert_eq!(report.applied, 1, "{stem}: third-party file did not merge");
            assert!(
                report.skipped.is_empty(),
                "{stem}: unexpected skip {:?}",
                report.skipped
            );

            let got = w.require("proj.weight").unwrap();
            assert!(
                all_close(got, want, 1e-4, 1e-5, false)
                    .unwrap()
                    .item::<bool>(),
                "{stem}: Wan third-party merge diverged from the lycoris reference"
            );
        }
    }

    // ====================================================================================
    // sc-3439 — VACE diffusers-named merge (`merge_vace_adapters`). Same merge math + format
    // dispatch as the native Wan path, on the diffusers `attn1/attn2.to_*` / `ffn.net.*` /
    // `vace_blocks.*` host. Bit-exact vs the in-test hand-computed `W + (B·A)` / `W + reconstruct`.
    // ====================================================================================

    /// A synthetic VACE (diffusers-layout) weight map: a base-block attn projection, a base-block
    /// FFN proj, and a vace-block hint projection — bf16, the modules the VACE LoRA tests target.
    fn synthetic_vace_weights() -> Weights {
        let path = tmp("vace_base.safetensors");
        let mk = |n: i32, scale: f32, bias: f32| {
            Array::from_slice(
                &(0..n).map(|i| i as f32 * scale - bias).collect::<Vec<_>>(),
                &[n / 8, 8],
            )
            .as_dtype(Dtype::Bfloat16)
            .unwrap()
        };
        let q = mk(16 * 8, 0.01, 0.3); // attn1.to_q [16,8]
        let fc1 = mk(24 * 8, 0.005, 0.2); // ffn.net.0.proj [24,8]
        let pin = mk(16 * 8, 0.007, 0.25); // vace_blocks.0.proj_in [16,8]
        Array::save_safetensors(
            vec![
                ("blocks.0.attn1.to_q.weight", &q),
                ("blocks.0.ffn.net.0.proj.weight", &fc1),
                ("vace_blocks.0.proj_in.weight", &pin),
            ],
            None,
            &path,
        )
        .unwrap();
        Weights::from_file(&path).unwrap()
    }

    #[test]
    fn merge_vace_folds_diffusers_named_delta_bit_exact() {
        // A diffusers-named LoRA (the host layout) folds W += B·A on the matching VACE modules,
        // including a vace_blocks Linear. Bit-exact to the hand-computed merge.
        let lora = write_lora(
            "vace_diff.safetensors",
            &[
                ("blocks.0.attn1.to_q", 16, 8),
                ("blocks.0.ffn.net.0.proj", 24, 8),
                ("vace_blocks.0.proj_in", 16, 8),
            ],
            4,
            0.2,
        );
        let mut w = synthetic_vace_weights();
        let report = merge_vace_adapters(&mut w, &[spec(lora.clone(), 1.0, None)]).unwrap();
        assert_eq!(report.applied, 3);
        assert!(report.skipped.is_empty());

        let lw = Weights::from_file(&lora).unwrap();
        let base = synthetic_vace_weights();
        for stem in [
            "blocks.0.attn1.to_q",
            "blocks.0.ffn.net.0.proj",
            "vace_blocks.0.proj_in",
        ] {
            let wkey = format!("{stem}.weight");
            let a = lw
                .require(&format!("diffusion_model.{stem}.lora_A.weight"))
                .unwrap();
            let b = lw
                .require(&format!("diffusion_model.{stem}.lora_B.weight"))
                .unwrap();
            let delta = matmul(b, a).unwrap();
            let want = add(
                base.require(&wkey).unwrap(),
                delta.as_dtype(Dtype::Bfloat16).unwrap(),
            )
            .unwrap();
            let got = w.require(&wkey).unwrap();
            assert!(
                array_eq(got, &want, false).unwrap().item::<bool>(),
                "{stem}: merged weight must be bit-exact to W + (B·A).astype(W.dtype)"
            );
        }
    }

    #[test]
    fn merge_vace_renames_native_named_lora_to_diffusers_host() {
        // A native-Wan-named LoRA (self_attn.q / ffn.0 — what musubi / diffusion-pipe emit) resolves
        // onto the diffusers host modules (attn1.to_q / ffn.net.0.proj) and folds there.
        let lora = write_lora(
            "vace_native.safetensors",
            &[("blocks.0.self_attn.q", 16, 8), ("blocks.0.ffn.0", 24, 8)],
            4,
            0.4,
        );
        let mut w = synthetic_vace_weights();
        let report = merge_vace_adapters(&mut w, &[spec(lora.clone(), 1.0, None)]).unwrap();
        assert_eq!(
            report.applied, 2,
            "native names must resolve to the diffusers host"
        );
        assert!(report.skipped.is_empty());

        let base = synthetic_vace_weights();
        // The diffusers host keys moved; the native key names are absent (they were renamed).
        let q = w.require("blocks.0.attn1.to_q.weight").unwrap();
        let q_base = base.require("blocks.0.attn1.to_q.weight").unwrap();
        assert!(!array_eq(q, q_base, false).unwrap().item::<bool>());
        assert!(w.get("blocks.0.self_attn.q.weight").is_none());
        assert!(w.get("blocks.0.ffn.net.0.proj.weight").is_some());
    }

    #[test]
    fn merge_vace_lokr_matches_reconstruct_and_scale_zero_is_noop() {
        // sc-2393 LoKr on the diffusers host: `blocks.0.attn1.to_q` is [16,8] = kron(w1[4,2],w2[4,4]).
        // Merged weight must equal W + (reconstruct·scale).astype(W.dtype); scale 0 is a bit-exact no-op.
        use mlx_gen::adapters::reconstruct_lokr_delta;
        use std::collections::HashMap;

        let w1 = Array::from_slice(
            &(0..8)
                .map(|i| (i as f32 * 0.03).sin() * 0.1)
                .collect::<Vec<_>>(),
            &[4, 2],
        );
        let w2 = Array::from_slice(
            &(0..16)
                .map(|i| (i as f32 * 0.05).cos() * 0.1)
                .collect::<Vec<_>>(),
            &[4, 4],
        );
        let (alpha, rank) = (4.0f32, 4.0f32);
        let mut meta = HashMap::new();
        meta.insert("networkType".to_string(), "lokr".to_string());
        meta.insert("alpha".to_string(), alpha.to_string());
        meta.insert("rank".to_string(), rank.to_string());
        let lokr_path = tmp("vace_lokr.safetensors");
        Array::save_safetensors(
            vec![
                ("blocks.0.attn1.to_q.lokr_w1", &w1),
                ("blocks.0.attn1.to_q.lokr_w2", &w2),
            ],
            Some(&meta),
            &lokr_path,
        )
        .unwrap();

        let scale = 0.5f32;
        let mut w = synthetic_vace_weights();
        let report = merge_vace_adapters(
            &mut w,
            &[AdapterSpec {
                kind: AdapterKind::Lokr,
                ..spec(lokr_path.clone(), scale, None)
            }],
        )
        .unwrap();
        assert_eq!(report.applied, 1);
        assert!(report.skipped.is_empty());

        let base = synthetic_vace_weights();
        let q_base = base.require("blocks.0.attn1.to_q.weight").unwrap();
        let delta = reconstruct_lokr_delta(
            alpha,
            rank,
            q_base.shape(),
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            Dtype::Float32,
        )
        .unwrap();
        let delta = multiply(&delta, scalar(scale).as_dtype(delta.dtype()).unwrap()).unwrap();
        let want = add(q_base, delta.as_dtype(q_base.dtype()).unwrap()).unwrap();
        let got = w.require("blocks.0.attn1.to_q.weight").unwrap();
        assert!(
            array_eq(got, &want, false).unwrap().item::<bool>(),
            "merged VACE LoKr weight must be bit-exact to W + (reconstruct·scale).astype(W.dtype)"
        );

        // scale 0 → bit-exact no-op.
        let mut w0 = synthetic_vace_weights();
        merge_vace_adapters(
            &mut w0,
            &[AdapterSpec {
                kind: AdapterKind::Lokr,
                ..spec(lokr_path, 0.0, None)
            }],
        )
        .unwrap();
        assert!(
            array_eq(
                w0.require("blocks.0.attn1.to_q.weight").unwrap(),
                q_base,
                false
            )
            .unwrap()
            .item::<bool>(),
            "scale-0 VACE LoKr merge must be a bit-exact no-op"
        );
    }

    #[test]
    fn merge_vace_reports_skipped_target_never_fatal() {
        // A LoRA module absent from the checkpoint is surfaced (skipped), never fatal — and a module
        // that IS present still merges in the same file.
        let lora = write_lora(
            "vace_skip.safetensors",
            &[
                ("blocks.0.attn1.to_q", 16, 8),
                ("blocks.99.attn1.to_q", 16, 8),
            ],
            4,
            0.1,
        );
        let mut w = synthetic_vace_weights();
        let report = merge_vace_adapters(&mut w, &[spec(lora, 1.0, None)]).unwrap();
        assert_eq!(report.applied, 1);
        assert_eq!(report.skipped, vec!["blocks.99.attn1.to_q".to_string()]);
    }

    // ---- Additive (unmerged) install (sc-10044) --------------------------------------------------

    use mlx_gen::adapters::AdaptableLinear;

    /// A minimal [`AdaptableHost`] mirroring the native Wan DiT's adaptable surface for one block:
    /// `blocks.0.self_attn.{q,k,v,o}`, `blocks.0.cross_attn.{q,k,v,o}`, `blocks.0.ffn.{fc1,fc2}`. It
    /// carries the two linears the test LoRA files target (`self_attn.q` and `ffn.fc1`) as real
    /// [`AdaptableLinear`]s, so [`apply_wan_adapters_additive`] exercises the genuine install path
    /// (normalize + factor convention + residual stacking) without the full ~7GB `WanTransformer`.
    struct TinyWanHost {
        q: AdaptableLinear,
        fc1: AdaptableLinear,
    }

    impl AdaptableHost for TinyWanHost {
        fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
            match path {
                ["blocks", "0", "self_attn", "q"] => Some(&mut self.q),
                ["blocks", "0", "ffn", "fc1"] => Some(&mut self.fc1),
                _ => None,
            }
        }

        fn adaptable_paths(&self) -> Vec<String> {
            vec![
                "blocks.0.self_attn.q".to_string(),
                "blocks.0.ffn.fc1".to_string(),
            ]
        }
    }

    impl TinyWanHost {
        /// Build from the shared `synthetic_weights()` map (same `[16,8]` q + `[24,8]` fc1 the fold
        /// tests use), so an additive install and a folded merge start from the identical base.
        fn from_synthetic() -> Self {
            let w = synthetic_weights();
            Self {
                q: AdaptableLinear::dense(
                    w.require("blocks.0.self_attn.q.weight").unwrap().clone(),
                    None,
                ),
                fc1: AdaptableLinear::dense(
                    w.require("blocks.0.ffn.fc1.weight").unwrap().clone(),
                    None,
                ),
            }
        }
    }

    /// f32 activations `[n, 8]` for the parity forwards (f32 in makes `addmm == matmul+add` bit-exact,
    /// so the only tolerance is the low-rank residual matmul).
    fn acts(n: i32) -> Array {
        Array::from_slice(
            &(0..n * 8)
                .map(|i| (i as f32 * 0.017).sin() * 0.5)
                .collect::<Vec<_>>(),
            &[n, 8],
        )
    }

    #[test]
    fn additive_lora_matches_folded_merge_dense() {
        // ACCEPTANCE: the additive residual on a DENSE base == the folded merge, same output within
        // Metal matmul tolerance. Fold into a `Weights` map via `merge_wan_adapters`, build a dense
        // linear from the merged weight, and compare its forward to the additive-installed host's.
        let lora = write_lora(
            "additive_parity.safetensors",
            &[("blocks.0.self_attn.q", 16, 8), ("blocks.0.ffn.0", 24, 8)],
            4,
            0.13,
        );
        let x = acts(3);

        // Folded reference.
        let mut w = synthetic_weights();
        let fold_report =
            merge_wan_adapters(&mut w, &[spec(lora.clone(), 0.75, None)], MoeExpert::High).unwrap();
        assert_eq!(fold_report.applied, 2);
        let folded_q = AdaptableLinear::dense(
            w.require("blocks.0.self_attn.q.weight").unwrap().clone(),
            None,
        );
        let folded_fc1 =
            AdaptableLinear::dense(w.require("blocks.0.ffn.fc1.weight").unwrap().clone(), None);

        // Additive.
        let mut host = TinyWanHost::from_synthetic();
        let add_report =
            apply_wan_adapters_additive(&mut host, &[spec(lora, 0.75, None)], MoeExpert::High)
                .unwrap();
        assert_eq!(add_report.applied, 2, "both q and fc1 installed additively");
        assert!(add_report.skipped.is_empty());

        for (folded, additive, name) in [
            (&folded_q, &host.q, "self_attn.q"),
            (&folded_fc1, &host.fc1, "ffn.fc1"),
        ] {
            let want = folded.forward(&x).unwrap();
            let got = additive.forward(&x).unwrap();
            assert!(
                all_close(&got, &want, 2e-2, 2e-2, false)
                    .unwrap()
                    .item::<bool>(),
                "additive residual must match the folded merge on {name} (dense)"
            );
            // Non-degenerate: the adapter actually changed the output vs the bare base.
            let bare = AdaptableLinear::dense(
                synthetic_weights()
                    .require(&format!(
                        "blocks.0.{}.weight",
                        if name == "self_attn.q" {
                            "self_attn.q"
                        } else {
                            "ffn.fc1"
                        }
                    ))
                    .unwrap()
                    .clone(),
                None,
            )
            .forward(&x)
            .unwrap();
            assert!(
                !array_eq(&got, &bare, false).unwrap().item::<bool>(),
                "additive adapter must change {name}'s output (non-degenerate)"
            );
        }
    }

    /// A quant-sized host: `blocks.0.self_attn.q` is `[128, 64]` so its `in = 64` is a multiple of the
    /// group size (64) and the base can be packed to Q4/Q8. Only the `q` target is populated.
    struct QuantWanHost {
        q: AdaptableLinear,
    }
    impl AdaptableHost for QuantWanHost {
        fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
            match path {
                ["blocks", "0", "self_attn", "q"] => Some(&mut self.q),
                _ => None,
            }
        }
        fn adaptable_paths(&self) -> Vec<String> {
            vec!["blocks.0.self_attn.q".to_string()]
        }
    }
    impl QuantWanHost {
        fn new() -> Self {
            let w = Array::from_slice(
                &(0..128 * 64)
                    .map(|i| (i as f32 * 0.0007).sin() * 0.05)
                    .collect::<Vec<_>>(),
                &[128, 64],
            )
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
            Self {
                q: AdaptableLinear::dense(w, None),
            }
        }
    }

    #[test]
    fn additive_lora_on_quantized_base_no_error_and_nondegenerate() {
        // ACCEPTANCE: a LoRA applies on a Q4/Q8 PACKED base with NO error (the old model.rs:614
        // rejection) and produces finite, non-zero output that differs from the no-adapter forward.
        // The base stays packed (never dequantized). The target is [128,64] so it is quantizable.
        let lora = write_lora(
            "additive_quant.safetensors",
            &[("blocks.0.self_attn.q", 128, 64)],
            8,
            0.21,
        );
        let x = Array::from_slice(
            &(0..3 * 64)
                .map(|i| (i as f32 * 0.017).sin() * 0.5)
                .collect::<Vec<_>>(),
            &[3, 64],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();

        for bits in [8, 4] {
            let mut host = QuantWanHost::new();
            // Pack the base to Q{bits} FIRST — so the additive install writes onto a packed snapshot.
            host.q.quantize(bits, None).unwrap();
            assert!(host.q.is_quantized(), "base must be packed before install");
            let baseline = host.q.forward(&x).unwrap();

            let report = apply_wan_adapters_additive(
                &mut host,
                &[spec(lora.clone(), 0.9, None)],
                MoeExpert::High,
            )
            .unwrap();
            assert_eq!(
                report.applied, 1,
                "q installed additively on the packed base (Q{bits})"
            );
            assert!(
                host.q.is_quantized(),
                "base must STAY packed after install (Q{bits}) — no dequant of W"
            );

            let got = host.q.forward(&x).unwrap();
            // Finite + non-zero.
            let g = got.as_dtype(Dtype::Float32).unwrap();
            let vals = g.as_slice::<f32>();
            assert!(
                vals.iter().all(|v| v.is_finite()),
                "Q{bits} additive output must be finite"
            );
            assert!(
                vals.iter().any(|v| v.abs() > 1e-6),
                "Q{bits} additive output must be non-zero"
            );
            // Differs from the no-adapter forward (the residual actually fired).
            assert!(
                !array_eq(&got, &baseline, false).unwrap().item::<bool>(),
                "Q{bits} additive adapter must change the packed base's output"
            );
        }
    }

    #[test]
    fn additive_no_adapter_is_byte_identical() {
        // ACCEPTANCE: with NO adapters installed the forward is byte-identical to the bare base — the
        // additive path adds nothing when empty (no regression to the no-adapter path).
        let x = acts(3);
        let host = TinyWanHost::from_synthetic();
        let bare_q = AdaptableLinear::dense(
            synthetic_weights()
                .require("blocks.0.self_attn.q.weight")
                .unwrap()
                .clone(),
            None,
        );
        let bare_out = bare_q.forward(&x).unwrap();
        let host_out = host.q.forward(&x).unwrap();
        assert!(
            array_eq(host_out, &bare_out, false).unwrap().item::<bool>(),
            "an un-adapted additive host must be byte-identical to the bare base"
        );

        // And an EMPTY spec list is a no-op install (applied 0, output unchanged).
        let mut host2 = TinyWanHost::from_synthetic();
        let report = apply_wan_adapters_additive(&mut host2, &[], MoeExpert::High).unwrap();
        assert_eq!(report.applied, 0);
        let host2_out = host2.q.forward(&x).unwrap();
        assert!(
            array_eq(host2_out, &bare_out, false)
                .unwrap()
                .item::<bool>(),
            "empty additive install must leave the forward byte-identical"
        );
    }

    #[test]
    fn additive_scale_zero_is_bit_exact_noop() {
        // A strength-0 additive LoRA is a bit-exact no-op (the residual is `0·…`), mirroring the
        // fold path's `scale_zero_is_bit_exact_noop`.
        let lora = write_lora(
            "additive_zero.safetensors",
            &[("blocks.0.self_attn.q", 16, 8)],
            4,
            0.3,
        );
        let x = acts(3);
        let bare = TinyWanHost::from_synthetic();
        let want = bare.q.forward(&x).unwrap();

        let mut host = TinyWanHost::from_synthetic();
        let report =
            apply_wan_adapters_additive(&mut host, &[spec(lora, 0.0, None)], MoeExpert::High)
                .unwrap();
        assert_eq!(report.applied, 1);
        let got = host.q.forward(&x).unwrap();
        assert!(
            array_eq(got, &want, false).unwrap().item::<bool>(),
            "strength 0 must leave the additive forward bit-identical"
        );
    }

    #[test]
    fn additive_lokr_matches_folded_merge_dense() {
        // The LoKr additive residual on a dense base matches the folded LoKr merge (same reconstructed
        // ΔW, one as `W+=δ`, the other as `x·δᵀ`). Reuses the fold tests' `write_lokr` helper.
        let lokr = write_lokr("additive_lokr.safetensors", 8.0, 4.0);
        let x = acts(3);

        // Folded reference: merge onto the q weight, forward the merged dense linear.
        let mut w = synthetic_weights();
        let fold =
            merge_wan_adapters(&mut w, &[lokr_spec(lokr.clone(), 0.6)], MoeExpert::High).unwrap();
        assert_eq!(fold.applied, 1);
        let folded_q = AdaptableLinear::dense(
            w.require("blocks.0.self_attn.q.weight").unwrap().clone(),
            None,
        );

        // Additive.
        let mut host = TinyWanHost::from_synthetic();
        let add_report =
            apply_wan_adapters_additive(&mut host, &[lokr_spec(lokr, 0.6)], MoeExpert::High)
                .unwrap();
        assert_eq!(add_report.applied, 1);

        let want = folded_q.forward(&x).unwrap();
        let got = host.q.forward(&x).unwrap();
        assert!(
            all_close(&got, &want, 2e-2, 2e-2, false)
                .unwrap()
                .item::<bool>(),
            "additive LoKr residual must match the folded LoKr merge (dense)"
        );
    }

    // ---- Structured (deferred) LoKr on a PACKED base (sc-10050) ----------------------------------

    /// Write a peft LoKr for `blocks.0.self_attn.q` sized to a **quantizable** `[128,64]` base
    /// (`in = 64` is a multiple of the group size): `[128,64] = kron(w1[16,8], w2[8,8])`. `alpha`/`rank`
    /// in metadata. Deterministic factor values.
    fn write_lokr_quant(name: &str, alpha: f32, rank: f32) -> PathBuf {
        use std::collections::HashMap;
        let w1 = Array::from_slice(
            &(0..16 * 8)
                .map(|i| (i as f32 * 0.013).sin() * 0.08)
                .collect::<Vec<_>>(),
            &[16, 8],
        );
        let w2 = Array::from_slice(
            &(0..8 * 8)
                .map(|i| (i as f32 * 0.021).cos() * 0.08)
                .collect::<Vec<_>>(),
            &[8, 8],
        );
        let meta = HashMap::from([
            ("networkType".to_string(), "lokr".to_string()),
            ("alpha".to_string(), alpha.to_string()),
            ("rank".to_string(), rank.to_string()),
        ]);
        let path = tmp(name);
        Array::save_safetensors(
            vec![
                ("blocks.0.self_attn.q.lokr_w1", &w1),
                ("blocks.0.self_attn.q.lokr_w2", &w2),
            ],
            Some(&meta),
            &path,
        )
        .unwrap();
        path
    }

    #[test]
    fn structured_lokr_on_quantized_matches_dense_and_stays_packed() {
        // ACCEPTANCE (sc-10050): a peft LoKr applies on a Q4/Q8 PACKED base with NO error, NO full
        // out×in delta materialized, the base STAYS packed, and its LoKr CONTRIBUTION (packed-with-LoKr
        // minus packed-baseline) matches the dense LoKr residual within tolerance. Comparing the
        // residual contribution — not the full output — isolates the adapter from the (much larger)
        // base-quantization error, which is what the vec-trick vs the materialized delta must agree on.
        // `[128,64]` target so `in` is a group-size multiple.
        let lokr = write_lokr_quant("structured_quant_lokr.safetensors", 8.0, 4.0);
        let x = Array::from_slice(
            &(0..3 * 64)
                .map(|i| (i as f32 * 0.017).sin() * 0.5)
                .collect::<Vec<_>>(),
            &[3, 64],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();

        let dense_w = Array::from_slice(
            &(0..128 * 64)
                .map(|i| (i as f32 * 0.0007).sin() * 0.05)
                .collect::<Vec<_>>(),
            &[128, 64],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();

        // Dense reference LoKr residual = (dense base + materialized LoKr) − (dense base). This is the
        // established materialized-delta path (the `install_one_lokr_additive` dense branch).
        let dense_base = AdaptableLinear::dense(dense_w.clone(), None);
        let dense_baseline = dense_base.forward(&x).unwrap();
        let mut dense_host = QuantWanHost {
            q: AdaptableLinear::dense(dense_w.clone(), None),
        };
        apply_wan_adapters_additive(
            &mut dense_host,
            &[lokr_spec(lokr.clone(), 0.6)],
            MoeExpert::High,
        )
        .unwrap();
        let dense_with_lokr = dense_host.q.forward(&x).unwrap();
        let dense_residual = mlx_rs::ops::subtract(&dense_with_lokr, &dense_baseline).unwrap();

        for bits in [8, 4] {
            let mut host = QuantWanHost {
                q: AdaptableLinear::dense(dense_w.clone(), None),
            };
            host.q.quantize(bits, None).unwrap();
            assert!(host.q.is_quantized(), "base must be packed before install");
            let baseline = host.q.forward(&x).unwrap();

            let report = apply_wan_adapters_additive(
                &mut host,
                &[lokr_spec(lokr.clone(), 0.6)],
                MoeExpert::High,
            )
            .unwrap();
            assert_eq!(report.applied, 1, "LoKr installed structurally on Q{bits}");
            assert!(report.skipped.is_empty());
            assert!(
                host.q.is_quantized(),
                "base must STAY packed after structured LoKr install (Q{bits})"
            );

            // The installed adapter is the STRUCTURED variant whose factors are the SMALL Kronecker
            // matrices — never an [out=128, in=64] delta. Assert the memory/no-materialization property
            // structurally: the only stored tensors are [16,8] (w1) and [8,8] (w2).
            match host.q.adapters() {
                [Adapter::LokrStructured { factors }] => {
                    assert_eq!(
                        factors.w1.shape(),
                        &[16, 8],
                        "w1 must be the small [a,c] factor, not [out,in]"
                    );
                    assert_eq!(
                        factors.w2.shape(),
                        &[8, 8],
                        "w2 must be the small [b,d] factor, not [out,in]"
                    );
                    // Neither factor is the [out,in]=[128,64] delta.
                    assert_ne!(factors.w1.shape(), &[128, 64]);
                    assert_ne!(factors.w2.shape(), &[128, 64]);
                }
                other => panic!(
                    "expected a single structured LoKr on the packed base, got {} adapter(s)",
                    other.len()
                ),
            }

            let got = host.q.forward(&x).unwrap();
            // Finite + non-degenerate (differs from the no-adapter forward).
            let g = got.as_dtype(Dtype::Float32).unwrap();
            assert!(
                g.as_slice::<f32>().iter().all(|v| v.is_finite()),
                "Q{bits} structured LoKr output must be finite"
            );
            assert!(
                !array_eq(&got, &baseline, false).unwrap().item::<bool>(),
                "Q{bits} structured LoKr must change the packed base's output"
            );
            // The structured LoKr CONTRIBUTION matches the dense materialized-delta residual. The
            // residual runs over the packed base's bf16 activation stream (same as the dense path here),
            // so the only difference is the vec-trick vs the materialized delta — Metal matmul tolerance.
            let quant_residual = mlx_rs::ops::subtract(&got, &baseline).unwrap();
            // The vec-trick (two small bf16 matmuls) vs the materialized delta (one bf16 matmul) round
            // differently; the residual itself is bf16, so allow the bf16-residual reduced-precision band.
            assert!(
                all_close(&quant_residual, &dense_residual, 5e-2, 8e-3, false)
                    .unwrap()
                    .item::<bool>(),
                "Q{bits} structured LoKr contribution must match the dense materialized-delta residual"
            );
        }
    }

    // ---- Packed-snapshot routing guard (sc-10045 / narrowed by sc-10050) ------------------------

    #[test]
    fn reject_loha_on_packed_passes_plain_lora_and_lokr() {
        // Plain-LoRA AND LoKr spec lists are allowed on a packed tier (sc-10050): LoRA installs
        // additively, LoKr via the structured deferred-Kronecker path — the guard is a no-op for both.
        let l1 = write_lora(
            "packroute_lora1.safetensors",
            &[("blocks.0.self_attn.q", 16, 8)],
            4,
            0.1,
        );
        let l2 = write_lora(
            "packroute_lora2.safetensors",
            &[("blocks.0.ffn.0", 24, 8)],
            4,
            0.4,
        );
        reject_loha_on_packed(
            "wan2_2_t2v_14b",
            &[spec(l1, 1.0, None), spec(l2, 1.0, Some(MoeExpert::High))],
        )
        .expect("plain LoRA files must pass the packed guard");
        // A peft LoKr file (networkType=lokr stamp) now ALSO passes — it installs via the structured
        // vec-trick (sc-10050), no longer rejected.
        let lokr = write_lokr("packroute_lokr.safetensors", 8.0, 4.0);
        reject_loha_on_packed("wan2_2_t2v_14b", &[lokr_spec(lokr, 0.6)])
            .expect("LoKr must pass the packed guard now that sc-10050 applies it structurally");
        // Third-party LoKr (no networkType stamp — detected by keys) likewise passes.
        let tp_lokr = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("tests/fixtures/sc3642_lokr/linear_w1full_w2lr.safetensors");
        reject_loha_on_packed("wan2_2_i2v_14b", &[spec(tp_lokr, 1.0, None)])
            .expect("third-party LoKr must pass the packed guard (structural apply, sc-10050)");
    }

    #[test]
    fn reject_loha_on_packed_errors_on_thirdparty_loha() {
        // sc-10051 (Option A): the committed third-party LyCORIS LoHa fixture (no networkType stamp —
        // detected by keys) is rejected on a packed tier with a CLEAR, ACTIONABLE, TYPED error. It
        // must NOT panic, NOT succeed, and NOT silently materialize the dense delta. Declared as LoRA
        // kind so the guard catches it by KEYS.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("tests/fixtures/sc3643_loha/linear.safetensors");
        let err = reject_loha_on_packed("wan2_2_i2v_14b", &[spec(path, 1.0, None)])
            .expect_err("third-party LoHa must be rejected on a packed tier");
        // Distinct typed variant the worker matches on (bridges 1:1 to gen_core::Error::Unsupported),
        // NOT a generic Msg — so the worker can surface steer-to-bf16 guidance, not an opaque failure.
        assert!(
            matches!(err, Error::Unsupported(_)),
            "LoHa-on-packed must be a typed Unsupported error, got: {err:?}"
        );
        let msg = err.to_string();
        // Actionable: names the family, the mechanism (Hadamard), and the fix (bf16 tier), + story.
        assert!(
            msg.contains("bf16") && msg.contains("sc-10051"),
            "error must point at bf16 + the LoHa deferral story (sc-10051), got: {msg}"
        );
        assert!(msg.contains("LoHa"), "error must name LoHa, got: {msg}");
        assert!(
            msg.contains("Hadamard"),
            "error must explain the Hadamard mechanism, got: {msg}"
        );
        assert!(
            msg.contains("bf16 tier"),
            "error must steer the user to the bf16 tier, got: {msg}"
        );
        assert!(
            !msg.contains("not yet wired"),
            "must NOT be the old generic 'not yet wired' message, got: {msg}"
        );
    }

    #[test]
    fn reject_loha_on_packed_unsupported_bridges_to_gen_core() {
        // The typed Unsupported must survive the mlx-gen → gen-core seam 1:1 (epic 3720, F-008) so the
        // worker's Unsupported classification (not a generic backend failure) fires with the full
        // steer-to-bf16 text intact.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("tests/fixtures/sc3643_loha/linear.safetensors");
        let err = reject_loha_on_packed("wan2_2_t2v_14b", &[spec(path, 1.0, None)])
            .expect_err("LoHa on a packed tier must error");
        let bridged: mlx_gen::gen_core::Error = err.into();
        match bridged {
            mlx_gen::gen_core::Error::Unsupported(s) => {
                assert!(
                    s.contains("bf16 tier") && s.contains("LoHa"),
                    "bridged gen-core Unsupported lost the actionable text: {s}"
                );
            }
            other => panic!("LoHa-on-packed degraded across the seam to {other:?}"),
        }
    }

    #[test]
    fn additive_skips_target_absent_from_host() {
        // A LoRA target outside the host's adaptable surface (a non-block / far-block module) is
        // surfaced (skipped), never fatal — mirroring the fold path's skip reporting.
        let lora = write_lora(
            "additive_skip.safetensors",
            &[
                ("blocks.0.self_attn.q", 16, 8),
                ("blocks.99.self_attn.q", 16, 8),
            ],
            4,
            0.1,
        );
        let mut host = TinyWanHost::from_synthetic();
        let report =
            apply_wan_adapters_additive(&mut host, &[spec(lora, 1.0, None)], MoeExpert::High)
                .unwrap();
        assert_eq!(report.applied, 1, "the present target installs");
        assert_eq!(
            report.skipped,
            vec!["blocks.99.self_attn.q".to_string()],
            "the absent target is surfaced, not fatal"
        );
    }
}
