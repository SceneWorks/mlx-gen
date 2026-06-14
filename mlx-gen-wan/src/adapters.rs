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
use mlx_gen::array::scalar;
use mlx_gen::gen_core::weightsmeta::{LoraAdapterMeta, LORA_ADAPTER_METADATA_KEY};
use mlx_gen::runtime::{AdapterKind, AdapterSpec, MoeExpert};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

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
}
