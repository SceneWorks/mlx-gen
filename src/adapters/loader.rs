//! Adapter-file loaders — read a trained LoRA/LoKr `.safetensors` and install it onto a
//! model tree via [`AdaptableHost`]. Closes out sc-2343's loader piece.
//!
//! **LoKr** is generic and faithfully ported from the fork's `LoKrLoader.apply`: keys are
//! bare module paths (`‹path›.lokr_w1`/`lokr_w2`, full or low-rank `_a`/`_b`) and the file
//! carries `networkType=lokr` + `alpha`/`rank` in safetensors metadata, so the delta and
//! target path are fully determined by the file — no per-model mapping table.
//!
//! **LoRA** here covers two on-disk conventions, both family-agnostic:
//! - **PEFT/diffusers** (`‹prefix›‹path›.lora_A/B.weight` + optional `‹path›.alpha`): dotted module
//!   paths resolve directly via [`AdaptableHost::adaptable_mut`] ([`apply_lora_peft`]).
//! - **kohya / sd-scripts** (`lora_unet_‹path, .→_›.lora_down/up.weight` + optional `.alpha`,
//!   sc-2618): the flattened module path can't be re-split blindly, so it resolves through a
//!   `flattened → dotted` table built from [`AdaptableHost::adaptable_paths`]
//!   ([`apply_lora_kohya`]). kohya `lora_down`/`lora_up` == PEFT `lora_A`/`lora_B`, so both feed the
//!   shared [`install_lora_groups`] and a kohya file yields the identical adapter to its PEFT twin.
//!
//! - **BFL / ComfyUI** (`lora_unet_double_blocks_*` / `diffusion_model.…` / `base_model.model.…`,
//!   sc-2743): a *fused* source linear (`…img_attn.qkv`, `…linear1`) is row-sliced into the model's
//!   *split* targets (`attn.to_q/to_k/to_v`, …) via per-target [`LoraRowSlice`] transforms, with BFL
//!   module renames (`img_in`→`x_embedder`). This is fused→split weight surgery, orthogonal to the
//!   kohya underscore form; the host supplies its table via [`AdaptableHost::bfl_targets`]
//!   ([`apply_lora_bfl`]). Only FLUX.2/FLUX.1 expose one; for other hosts a BFL file's keys surface as
//!   unmatched (loud), never silently dropped.

use std::collections::{BTreeMap, BTreeSet};

use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::{Array, Dtype};

use super::{
    reconstruct_loha_delta, reconstruct_lokr_delta, reconstruct_lokr_delta_scaled, AdaptableHost,
    Adapter,
};
use crate::runtime::{AdapterKind, AdapterSpec};
use crate::weights::Weights;
use crate::{Error, Result};

// The format predicates, factor-suffix tables, rank/alpha parsing, and key-alias resolution are
// backend-neutral string/metadata logic and live in gen-core (sc-3722); this module supplies the
// `Weights`/`Array` half (factor grouping + delta reconstruction). The historical
// `mlx_gen::adapters::loader::{KOHYA_PREFIX, COMMON_LORA_PREFIXES, resolve_lokr_path}` paths stay
// resolvable via these re-exports.
use gen_core::weightsmeta as wmeta;
pub use gen_core::weightsmeta::{resolve_lokr_path, COMMON_LORA_PREFIXES, KOHYA_PREFIX};

/// PEFT LoKr per-module factor suffixes (gen-core's table) — each factor is full (`lokr_w1`/
/// `lokr_w2`) or low-rank (`_a`/`_b`). Exact-suffix matched.
use gen_core::weightsmeta::LOKR_SUFFIXES;

/// `true` if the file's `networkType` metadata marks it a LoKr adapter.
pub fn is_lokr(w: &Weights) -> bool {
    wmeta::is_lokr_network_type(w.metadata("networkType"))
}

/// A parsed LoKr file: the global `(alpha, rank)` from metadata plus every module's Kronecker
/// factors grouped by path. The factor map is keyed by the bare factor name (`lokr_w1`,
/// `lokr_w1_a`, `lokr_w1_b`, `lokr_w2`, `lokr_w2_a`, `lokr_w2_b`); a module is full or low-rank.
///
/// This is the format-parsing half of a LoKr install, factored out of [`apply_lokr`] so the video
/// providers (LTX/Wan) — which install onto their crate-local `Linear`s as a forward-time residual
/// or an in-place weight merge, rather than the core [`AdaptableHost`] — reuse the exact same factor
/// grouping + metadata read and differ only in the install step. Each provider then maps the bare
/// module `path` through its own key→module table and calls [`reconstruct_lokr_delta`].
#[derive(Debug)]
pub struct LokrFile {
    pub alpha: f32,
    pub rank: f32,
    /// `module path → { factor name → tensor }`.
    pub groups: BTreeMap<String, BTreeMap<String, Array>>,
}

impl LokrFile {
    /// `alpha/rank` — the scale the fork bakes into the reconstructed delta (PEFT default `alpha=rank`
    /// ⇒ 1.0). The per-adapter user `strength` multiplies this separately at the residual/merge site.
    pub fn delta_scale(&self) -> f32 {
        self.alpha / self.rank
    }

    /// Reconstruct one module's `[out,in]` delta at `out_dtype` from its grouped factors, baking in
    /// `alpha/rank` (the user `strength` is applied separately). `base_shape` is the target linear's
    /// logical weight shape. Returns the [`reconstruct_lokr_delta`] result.
    pub fn delta(
        &self,
        factors: &BTreeMap<String, Array>,
        base_shape: &[i32],
        out_dtype: Dtype,
    ) -> Result<Array> {
        reconstruct_lokr_delta(
            self.alpha,
            self.rank,
            base_shape,
            factors.get("lokr_w1"),
            factors.get("lokr_w1_a"),
            factors.get("lokr_w1_b"),
            factors.get("lokr_w2"),
            factors.get("lokr_w2_a"),
            factors.get("lokr_w2_b"),
            out_dtype,
        )
    }
}

/// Parse a LoKr `.safetensors` into [`LokrFile`]: read `rank`/`alpha` from metadata (alpha defaults
/// to rank, i.e. scale 1.0, matching PEFT) and group every `‹path›.lokr_*` tensor by module path.
/// Shared by [`apply_lokr`] (core `AdaptableHost` install) and the video providers' crate-local
/// residual/merge installers.
pub fn parse_lokr(w: &Weights) -> Result<LokrFile> {
    // rank/alpha (alpha defaults to rank ⇒ scale 1.0, matching PEFT) — parsed in gen-core.
    let (rank, alpha) = wmeta::parse_rank_alpha(w.metadata("rank"), w.metadata("alpha"));

    let keys: Vec<String> = w.keys().map(str::to_string).collect();
    let mut groups: BTreeMap<String, BTreeMap<String, Array>> = BTreeMap::new();
    for key in &keys {
        if let Some((path, factor)) = wmeta::split_factor_key(key, &LOKR_SUFFIXES) {
            groups
                .entry(path.to_string())
                .or_default()
                .insert(factor.to_string(), w.require(key)?.clone());
        }
    }
    Ok(LokrFile {
        alpha,
        rank,
        groups,
    })
}

/// Read a scalar adapter value (an `alpha`) as `f32`, regardless of its on-disk dtype. Trained
/// adapters store `alpha` in their compute dtype: real kohya/BFL FLUX LoRAs ship it **bf16** (sc-2657),
/// and `Array::as_slice::<f32>()` `unwrap`s a hard dtype-mismatch (it never casts), so reading a bf16
/// scalar that way panics. Cast to f32 first (exact for the small integer alphas these files carry, and
/// a no-op when already f32); a `[]`- or `[1]`-shaped scalar both read as a one-element slice.
///
/// A size-0 `alpha` tensor (a malformed third-party adapter file) has no data pointer to borrow, so
/// `as_slice` — `try_as_slice().unwrap()` — would panic rather than fall through to the trailing
/// `first()`. Guard it like [`crate::array::host_i32`] so one bad file fails its single job with a
/// typed error instead of aborting the worker.
fn scalar_alpha(a: &Array) -> Result<Option<f32>> {
    if a.size() == 0 {
        // The callers only reach here for an `alpha` key that *exists* (`w.require(..)`), so a
        // present-but-empty tensor is a malformed file — NOT an absent alpha. Returning `Ok(None)`
        // here would be indistinguishable from "no alpha key", silently falling back to the
        // `alpha == rank ⇒ scale 1.0` default and mis-scaling the adapter while reporting success.
        // Fail this one job with a typed error instead (F-031, matching the contract documented above).
        return Err(Error::Msg(
            "scalar_alpha: alpha tensor present but empty (size 0) — malformed adapter file".into(),
        ));
    }
    Ok(a.as_dtype(Dtype::Float32)?
        .try_as_slice::<f32>()
        .map_err(|e| Error::Msg(format!("scalar_alpha: not a readable scalar array: {e}")))?
        .first()
        .copied())
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
    let file = parse_lokr(w)?;
    let mut report = ApplyReport::default();
    for (path, factors) in &file.groups {
        let parts: Vec<&str> = path.split('.').collect();
        match host.adaptable_mut(&parts) {
            Some(lin) => {
                let base_shape = lin.base_shape();
                // Fork-parity residual path keeps the delta at bf16 (PARITY-BF16, sc-2609).
                let delta = file.delta(factors, &base_shape, Dtype::Bfloat16)?;
                lin.push(Adapter::Lokr { delta, scale });
                report.applied += 1;
            }
            None => report.unmatched_paths.push(path.clone()),
        }
    }
    Ok(report)
}

// ---- Third-party LyCORIS LoKr (sc-3642) ----------------------------------------------------------

/// Third-party LoKr factor suffixes — the PEFT set plus `lokr_t2` (the lycoris tucker/CP factor).
/// `.lokr_w1_a`/`_b` precede the bare `.lokr_w1` so exact-suffix matching never mis-binds. (gen-core.)
use gen_core::weightsmeta::LOKR_TP_SUFFIXES;

/// `true` if any key is a LoKr factor (`*.lokr_w…`), regardless of `networkType` metadata. This is
/// how a **third-party** LyCORIS LoKr (kohya / ai-toolkit / lycoris-lib) is recognized — those files
/// ship the Kronecker factors but NOT SceneWorks' peft `networkType=lokr` stamp that [`is_lokr`]
/// keys off. (A `lokr_t2` tucker factor always co-occurs with `lokr_w2_a`, so `.lokr_w` suffices.)
pub fn is_lokr_keys(w: &Weights) -> bool {
    wmeta::keys_contain_lokr(w.keys())
}

/// One module's third-party LoKr factors. Unlike the peft [`LokrFile`] (one global `(alpha, rank)`
/// from metadata), a third-party file carries **per-module** factor shapes + an optional per-module
/// `.alpha` scalar, so rank/alpha/scale are derived per module here.
#[derive(Default)]
pub struct ThirdPartyLokr {
    w1: Option<Array>,
    w1_a: Option<Array>,
    w1_b: Option<Array>,
    w2: Option<Array>,
    w2_a: Option<Array>,
    w2_b: Option<Array>,
    t2: Option<Array>,
    alpha: Option<f32>,
}

impl ThirdPartyLokr {
    /// The factorization rank (`lora_dim`). lycoris lays the factors out inconsistently, so derive in
    /// a fixed order from whichever decomposed factor is present: `lokr_w1_a` is `[shape0, dim]`
    /// (dim = `shape[1]`); the tucker `lokr_t2` is `[dim, dim, kH, kW]` (dim = `shape[0]`); the
    /// non-tucker `lokr_w2_a` is `[shape0, dim]` (dim = `shape[1]`). `None` when **both** factors are
    /// full — lycoris then forces `alpha = lora_dim` ⇒ scale 1, so rank is unused.
    fn rank(&self) -> Option<f32> {
        if let Some(a) = &self.w1_a {
            return Some(a.shape()[1] as f32);
        }
        if let Some(t) = &self.t2 {
            return Some(t.shape()[0] as f32);
        }
        self.w2_a.as_ref().map(|a| a.shape()[1] as f32)
    }

    /// LyCORIS `scale`: `alpha / lora_dim` (alpha defaulting to `lora_dim`), EXCEPT both-full forces
    /// scale 1 (mirrors `LokrModule.__init__`: `if use_w1 and use_w2: alpha = lora_dim`).
    fn scale(&self) -> f32 {
        match self.rank() {
            None => 1.0,
            Some(r) => self.alpha.unwrap_or(r) / r,
        }
    }

    /// Reconstruct this module's `ΔW` (lycoris per-module scale baked in) at `out_dtype`. `pub` so the
    /// merge-path providers (SDXL/Wan/LTX, sc-3671) reuse the exact derivation + reconstruction and
    /// differ only in how they install it (in-place merge vs forward residual).
    pub fn delta(&self, base_shape: &[i32], out_dtype: Dtype) -> Result<Array> {
        reconstruct_lokr_delta_scaled(
            self.scale(),
            base_shape,
            self.w1.as_ref(),
            self.w1_a.as_ref(),
            self.w1_b.as_ref(),
            self.w2.as_ref(),
            self.t2.as_ref(),
            self.w2_a.as_ref(),
            self.w2_b.as_ref(),
            out_dtype,
        )
    }
}

/// Group a third-party LoKr file's tensors by raw module key (the part before `.lokr_*`/`.alpha`).
/// The raw key is whatever the trainer wrote — a `<PREFIX>_<flattened.path>` (kohya/lycoris) or, more
/// rarely, a dotted path; resolution to the host's module map happens in [`apply_lokr_thirdparty`]
/// (or the merge-path providers' own tables, sc-3671).
pub fn parse_lokr_thirdparty(w: &Weights) -> Result<BTreeMap<String, ThirdPartyLokr>> {
    let mut groups: BTreeMap<String, ThirdPartyLokr> = BTreeMap::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        if let Some(raw) = key.strip_suffix(".alpha") {
            if let Some(a) = scalar_alpha(w.require(&key)?)? {
                groups.entry(raw.to_string()).or_default().alpha = Some(a);
            }
            continue;
        }
        for suffix in LOKR_TP_SUFFIXES {
            if let Some(raw) = key.strip_suffix(suffix) {
                let g = groups.entry(raw.to_string()).or_default();
                let t = w.require(&key)?.clone();
                match &suffix[1..] {
                    "lokr_w1" => g.w1 = Some(t),
                    "lokr_w1_a" => g.w1_a = Some(t),
                    "lokr_w1_b" => g.w1_b = Some(t),
                    "lokr_w2" => g.w2 = Some(t),
                    "lokr_w2_a" => g.w2_a = Some(t),
                    "lokr_w2_b" => g.w2_b = Some(t),
                    "lokr_t2" => g.t2 = Some(t),
                    _ => {}
                }
                break;
            }
        }
    }
    Ok(groups)
}

// `resolve_lokr_path` — resolve a third-party LoKr raw module key (`<PREFIX>_<stem>`, the diffusers
// path flattened with `.`→`_`) to a host dotted path, longest-stem-wins — is defined in gen-core
// (`weightsmeta`) and re-exported at the top of this module so the merge-path providers (sc-3671)
// resolve third-party keys against their own module tables.

/// Install a third-party LyCORIS **LoKr** file (LoHa is sc-3643) onto `host`. Reconstructs each
/// module's Kronecker delta from its per-module factors (full / low-rank / tucker) at the lycoris
/// scale and stacks it as an [`Adapter::Lokr`] residual at the user `scale` — the same install as
/// peft [`apply_lokr`], differing only in (a) per-module rank/alpha derivation and (b) resolving the
/// trainer's flattened key names to the host's dotted module map. Unresolved paths are surfaced in
/// `unmatched_paths`, never silently dropped.
pub fn apply_lokr_thirdparty(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
) -> Result<ApplyReport> {
    let table = kohya_table(&host.adaptable_paths());
    let groups = parse_lokr_thirdparty(w)?;
    let mut report = ApplyReport::default();
    for (raw, g) in &groups {
        // Flattened stem via the table (prefix-agnostic), else the raw key treated as already-dotted.
        let dotted = resolve_lokr_path(raw, &table).unwrap_or(raw.as_str());
        let parts: Vec<&str> = dotted.split('.').collect();
        match host.adaptable_mut(&parts) {
            Some(lin) => {
                let base_shape = lin.base_shape();
                // Fork-parity residual path keeps the delta at bf16 (PARITY-BF16, sc-2609) — same as
                // peft `apply_lokr`.
                let delta = g.delta(&base_shape, Dtype::Bfloat16)?;
                lin.push(Adapter::Lokr { delta, scale });
                report.applied += 1;
            }
            None => report.unmatched_paths.push(raw.clone()),
        }
    }
    Ok(report)
}

// ---- Third-party LyCORIS LoHa (sc-3643) ----------------------------------------------------------

/// Third-party LoHa factor suffixes (the two Hadamard low-rank pairs + optional tucker factors).
/// (gen-core.)
use gen_core::weightsmeta::LOHA_TP_SUFFIXES;

/// `true` if any key is a LoHa factor (`*.hada_w…`) — how a third-party LyCORIS LoHa (kohya /
/// ai-toolkit / lycoris-lib) is recognized. Mutually exclusive with [`is_lokr_keys`] (`lokr_*`).
pub fn is_loha_keys(w: &Weights) -> bool {
    wmeta::keys_contain_loha(w.keys())
}

/// One module's third-party LoHa factors — two low-rank Hadamard pairs, optional tucker `t1`/`t2`,
/// and an optional per-module `.alpha` (rank/scale derived per module, like [`ThirdPartyLokr`]).
#[derive(Default)]
pub struct ThirdPartyLoha {
    w1_a: Option<Array>,
    w1_b: Option<Array>,
    w2_a: Option<Array>,
    w2_b: Option<Array>,
    t1: Option<Array>,
    t2: Option<Array>,
    alpha: Option<f32>,
}

impl ThirdPartyLoha {
    /// rank (`lora_dim`) = `hada_w1_b.shape[0]` (lycoris stores `hada_w1_b` as `[lora_dim, …]` in
    /// both the tucker and non-tucker layouts).
    fn rank(&self) -> Option<f32> {
        self.w1_b.as_ref().map(|b| b.shape()[0] as f32)
    }

    /// LyCORIS `scale = alpha / lora_dim` (alpha defaulting to `lora_dim`). LoHa is always decomposed
    /// (no both-full case), so — unlike LoKr — there is no forced-1 branch.
    fn scale(&self) -> f32 {
        match self.rank() {
            None => 1.0,
            Some(r) => self.alpha.unwrap_or(r) / r,
        }
    }

    /// Reconstruct this module's LoHa `ΔW` (lycoris per-module scale baked in) at `out_dtype`. `pub`
    /// for the merge-path providers (sc-3671).
    pub fn delta(&self, base_shape: &[i32], out_dtype: Dtype) -> Result<Array> {
        let (w1_a, w1_b, w2_a, w2_b) = match (&self.w1_a, &self.w1_b, &self.w2_a, &self.w2_b) {
            (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
            _ => return Err("LoHa: a hada_w1/w2 a/b factor is missing".into()),
        };
        reconstruct_loha_delta(
            self.scale(),
            base_shape,
            w1_a,
            w1_b,
            w2_a,
            w2_b,
            self.t1.as_ref(),
            self.t2.as_ref(),
            out_dtype,
        )
    }
}

/// Group a third-party LoHa file's tensors by raw module key (the part before `.hada_*`/`.alpha`).
/// `pub` for the merge-path providers (sc-3671).
pub fn parse_loha_thirdparty(w: &Weights) -> Result<BTreeMap<String, ThirdPartyLoha>> {
    let mut groups: BTreeMap<String, ThirdPartyLoha> = BTreeMap::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        if let Some(raw) = key.strip_suffix(".alpha") {
            if let Some(a) = scalar_alpha(w.require(&key)?)? {
                groups.entry(raw.to_string()).or_default().alpha = Some(a);
            }
            continue;
        }
        for suffix in LOHA_TP_SUFFIXES {
            if let Some(raw) = key.strip_suffix(suffix) {
                let g = groups.entry(raw.to_string()).or_default();
                let t = w.require(&key)?.clone();
                match &suffix[1..] {
                    "hada_w1_a" => g.w1_a = Some(t),
                    "hada_w1_b" => g.w1_b = Some(t),
                    "hada_w2_a" => g.w2_a = Some(t),
                    "hada_w2_b" => g.w2_b = Some(t),
                    "hada_t1" => g.t1 = Some(t),
                    "hada_t2" => g.t2 = Some(t),
                    _ => {}
                }
                break;
            }
        }
    }
    Ok(groups)
}

/// Install a third-party LyCORIS **LoHa** file onto `host`. Reconstructs each module's Hadamard delta
/// and stacks it as an [`Adapter::Lokr`] residual (the reconstructed `ΔW` applies through the same
/// `scale · x·ΔWᵀ` forward path — no distinct adapter variant needed). Module-key resolution
/// (flattened-prefixed → dotted via [`kohya_table`]) and unmatched-path surfacing mirror
/// [`apply_lokr_thirdparty`].
pub fn apply_loha_thirdparty(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
) -> Result<ApplyReport> {
    let table = kohya_table(&host.adaptable_paths());
    let groups = parse_loha_thirdparty(w)?;
    let mut report = ApplyReport::default();
    for (raw, g) in &groups {
        let dotted = resolve_lokr_path(raw, &table).unwrap_or(raw.as_str());
        let parts: Vec<&str> = dotted.split('.').collect();
        match host.adaptable_mut(&parts) {
            Some(lin) => {
                let base_shape = lin.base_shape();
                let delta = g.delta(&base_shape, Dtype::Bfloat16)?;
                lin.push(Adapter::Lokr { delta, scale });
                report.applied += 1;
            }
            None => report.unmatched_paths.push(raw.clone()),
        }
    }
    Ok(report)
}

/// Install a PEFT/diffusers-format LoRA file onto `host`. The down/up factors carry the file's
/// namespace prefix on a **dotted** module path, in either of two interchangeable spellings:
/// - PEFT: `‹prefix›‹path›.lora_A.weight` / `.lora_B.weight`;
/// - diffusers/ComfyUI/ai-toolkit (e.g. the lightx2v Qwen-Image-Lightning LoRAs, sc-2909):
///   `‹prefix›‹path›.lora_down.weight` / `.lora_up.weight` — `lora_down`==`lora_A`, `lora_up`==`lora_B`
///   (identical shapes), differing from the kohya format only in that the path stays dotted (no
///   `lora_unet_` flattening), so it routes here rather than to [`apply_lora_kohya`].
///
/// Both store the down factor as `[r, in]` and the up factor as `[out, r]`; we transpose to the
/// residual form `x·A·B` (`A: [in, r]`, `B: [r, out]`) and fold `alpha/rank` into `B`, matching the
/// fork. `‹prefix›‹path›.alpha` is optional (and may be bare — see below). `strip_prefix` removes a
/// leading namespace such as `"base_model.model."` or `"transformer."`.
pub fn apply_lora_peft(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
    strip_prefix: Option<&str>,
) -> Result<ApplyReport> {
    let prefix = strip_prefix.unwrap_or("");
    let mut groups: BTreeMap<String, LoraParts> = BTreeMap::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        // The down/up factors always carry the file's namespace prefix. `lora_A`/`lora_B` (PEFT) and
        // `lora_down`/`lora_up` (diffusers/ComfyUI) are interchangeable spellings of the same role.
        if let Some(rest) = key.strip_prefix(prefix) {
            if let Some(path) = rest
                .strip_suffix(".lora_A.weight")
                .or_else(|| rest.strip_suffix(".lora_down.weight"))
            {
                groups.entry(path.to_string()).or_default().a = Some(w.require(&key)?.clone());
                continue;
            }
            if let Some(path) = rest
                .strip_suffix(".lora_B.weight")
                .or_else(|| rest.strip_suffix(".lora_up.weight"))
            {
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
            if let Some(new) = scalar_alpha(w.require(&key)?)? {
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

    // PEFT/diffusers `save_lora_adapter` files carry no per-target `.alpha` tensor — `lora_alpha`/`r`
    // (+ per-module overrides) live in the `lora_adapter_metadata` header blob (sc-5513). `None` for a
    // file without the blob, in which case the per-target `.alpha` or the factor rank is used exactly
    // as before. (kohya / BFL loaders ship a `.alpha` tensor and pass `None` here.)
    let cfg = wmeta::LoraAdapterMeta::from_metadata(w.metadata(wmeta::LORA_ADAPTER_METADATA_KEY));
    install_lora_groups(host, groups, scale, cfg.as_ref())
}

/// Install grouped `(down=A, up=B, alpha)` LoRA factors onto `host`, one residual per resolved module
/// path. Shared by the PEFT/diffusers loader ([`apply_lora_peft`]) and the kohya loader
/// ([`apply_lora_kohya`]): both conventions agree on the math (`A: [r,in]`, `B: [out,r]`, transpose to
/// the residual form `x·A·B`, fold `alpha/rank` into `B`) and differ only in how keys map to `path`.
/// A path with a missing `down` or `up` half is skipped (its partner targeted a non-LoRA key);
/// a path that resolves to no module is surfaced in `unmatched_paths`, never silently dropped.
fn install_lora_groups(
    host: &mut impl AdaptableHost,
    groups: BTreeMap<String, LoraParts>,
    scale: f32,
    meta: Option<&wmeta::LoraAdapterMeta>,
) -> Result<ApplyReport> {
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
                                       // A Linear LoRA's factors are 2-D; a malformed 1-D/scalar factor would panic on the
                                       // `a.shape()[1]` rank read below. Reject it with a typed error up front (F-034).
                if a.shape().len() != 2 || b.shape().len() != 2 {
                    return Err(Error::Msg(format!(
                        "lora adapter at '{path}' has non-2-D factors (down {:?}, up {:?})",
                        a_raw.shape(),
                        b_raw.shape()
                    )));
                }
                // Effective scaling. Precedence: per-target `.alpha` tensor (kohya / SceneWorks
                // trainer / BFL) → the PEFT/diffusers `lora_adapter_metadata` blob's
                // `alpha_pattern`/`lora_alpha` (sc-5513 — that format ships NO `.alpha` tensor) → no
                // fold (the pre-existing `alpha == rank ⇒ scale 1.0` default). The denominator honors
                // the blob `r`/`rank_pattern` when given (always `> 0`), else the factor's stored
                // leading dim (which equals it for a well-formed PEFT file).
                let (cfg_alpha, cfg_rank) = meta.map_or((None, None), |m| m.effective(&path));
                if let Some(alpha) = parts.alpha.or(cfg_alpha) {
                    let factor_rank = a.shape()[1] as f32; // r
                    if factor_rank == 0.0 {
                        // Zero rank (empty/malformed factor) → non-finite alpha/rank → a NaN residual
                        // folded into the linear, silently corrupting inference. Reject the adapter
                        // instead of installing it (sc-5252/F-002).
                        return Err(Error::Msg(format!(
                            "lora adapter at '{path}' has zero rank (empty down/up factor)"
                        )));
                    }
                    let rank = cfg_rank.unwrap_or(factor_rank);
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

// `KOHYA_PREFIX` (`lora_unet_`) is defined in gen-core and re-exported at the top of this module.

/// kohya factor suffixes mapped to a [`LoraParts`] role. `lora_down`==PEFT `lora_A`,
/// `lora_up`==PEFT `lora_B`; the optional `.default` infix is the peft-export form some kohya
/// converters emit. Order is irrelevant (exact-suffix match).
const KOHYA_SUFFIXES: [(&str, KohyaRole); 5] = [
    (".lora_down.weight", KohyaRole::Down),
    (".lora_up.weight", KohyaRole::Up),
    (".lora_down.default.weight", KohyaRole::Down),
    (".lora_up.default.weight", KohyaRole::Up),
    (".alpha", KohyaRole::Alpha),
];

#[derive(Clone, Copy)]
enum KohyaRole {
    Down,
    Up,
    Alpha,
}

/// `true` if `w` is a kohya-format LoRA — any key carries the `lora_unet_` prefix. (kohya files are
/// the only convention that flattens the module path; PEFT/diffusers keep dots, LoKr is bare.)
pub fn is_kohya(w: &Weights) -> bool {
    wmeta::keys_are_kohya(w.keys())
}

/// Build the kohya `flattened-stem → dotted-path` lookup from a host's routable target paths
/// (`AdaptableHost::adaptable_paths`). The stem is the dotted path with `.`→`_` (the kohya
/// flattening), WITHOUT the `lora_unet_` prefix. Mirrors the SDXL matcher (sc-2639) and the fork's
/// explicit `lora_unet_…` patterns, generalized over any [`AdaptableHost`].
fn kohya_table(paths: &[String]) -> BTreeMap<String, String> {
    wmeta::kohya_table(paths)
}

/// Install a kohya-format LoRA (`lora_unet_<flattened path>.lora_down/up.weight` + optional `.alpha`)
/// onto `host`. The flattened stem is resolved against `table` (built from
/// [`AdaptableHost::adaptable_paths`]) — blind `_`→`.` splitting is impossible because module names
/// contain underscores (`to_out.0`, `feed_forward.w1`, `img_mlp.net.0.proj`). Resolved factors are
/// installed through the same [`install_lora_groups`] path as PEFT, so a kohya file produces the
/// identical adapter to the equivalent PEFT file.
///
/// `lora_unet_` keys whose stem is NOT in the table (off-surface) are surfaced in `unmatched_paths`
/// so the strict policy fails loudly rather than silently dropping them. The BFL fused→split kohya
/// form (`lora_unet_double_blocks_*`, sc-2743) is routed to [`apply_lora_bfl`] *before* this loader
/// for a host that exposes [`AdaptableHost::bfl_targets`]; reaching here it has no table entry and is
/// likewise surfaced. Keys without the `lora_unet_` prefix (e.g. a bundled text-encoder `lora_te_…`)
/// are not denoiser targets and are ignored, matching the PEFT loader's treatment of out-of-namespace
/// keys.
pub fn apply_lora_kohya(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
    table: &BTreeMap<String, String>,
) -> Result<ApplyReport> {
    let mut groups: BTreeMap<String, LoraParts> = BTreeMap::new();
    let mut unresolved: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        let Some(rem) = key.strip_prefix(KOHYA_PREFIX) else {
            continue; // not a denoiser kohya key (e.g. text-encoder `lora_te_…`) — ignore.
        };
        let Some((stem, role)) = KOHYA_SUFFIXES
            .iter()
            .find_map(|(suf, role)| rem.strip_suffix(suf).map(|s| (s, *role)))
        else {
            continue; // a `lora_unet_` key with an unrecognized suffix — ignore.
        };
        let Some(path) = table.get(stem) else {
            unresolved.insert(stem.to_string());
            continue;
        };
        let parts = groups.entry(path.clone()).or_default();
        match role {
            KohyaRole::Down => parts.a = Some(w.require(&key)?.clone()),
            KohyaRole::Up => parts.b = Some(w.require(&key)?.clone()),
            KohyaRole::Alpha => parts.alpha = scalar_alpha(w.require(&key)?)?,
        }
    }

    // kohya / BFL files carry a per-target `.alpha` tensor, not the `lora_adapter_metadata` blob — no
    // blob to honor here (sc-5513).
    let mut report = install_lora_groups(host, groups, scale, None)?;
    report.unmatched_paths.extend(unresolved);
    Ok(report)
}

// ---- BFL / ComfyUI fused→split LoRA (sc-2743) ----------------------------------------------------

/// A row-slice over a raw LoRA factor (applied BEFORE the `[r,in]`/`[out,r]`→residual transpose),
/// porting the fork's `LoraTransforms` byte-for-byte (sc-2743). The BFL/ComfyUI checkpoints store a
/// block's q/k/v (and, for FLUX.1, the qkv+mlp) concatenated along dim-0 of a single *fused* linear;
/// each diffusers split target slices out its own rows. Indices/divisibility match `LoraTransforms`
/// exactly (verified against the fork venv).
#[derive(Clone, Debug)]
pub enum LoraRowSlice {
    /// Chunk `index` of `n` equal dim-0 chunks (`chunk = shape[0] / n`), ALWAYS sliced — the fork's
    /// `_split_qkv_up` (the up factor `[n·out, r]` → `[out, r]`). `n=3` for qkv.
    Chunk { n: i32, index: i32 },
    /// Chunk `index` of `n` equal dim-0 chunks IFF `shape[0] % n == 0`, else the whole tensor — the
    /// fork's `_split_qkv_down`/`_split_qkv_mlp_down` (the down factor is *shared* across q/k/v when
    /// the rank isn't divisible by `n`, which is the usual fused-qkv LoRA, and sliced when it is).
    ChunkIfDivisible { n: i32, index: i32 },
    /// The dim-0 slice `[Σdims[..index] .. Σdims[..=index]]` — the fork's `_split_qkv_mlp_up` with
    /// config-derived `dims` (FLUX.1 `linear1` = `[q,k,v,mlp]`, e.g. `[3072,3072,3072,12288]`). FLUX.2
    /// keeps qkv+mlp fused (`to_qkv_mlp_proj`) so it never uses this; FLUX.1 (sc-2657) will.
    Dims { dims: Vec<i32>, index: i32 },
}

impl LoraRowSlice {
    fn apply(&self, t: &Array) -> Result<Array> {
        let rows = t.shape()[0];
        // `n`/`index`/`dims` come from a static `bfl_targets()` table built in code, not the file —
        // but a miswritten entry (n<=0, index out of [0,n), or an out-of-range `dims` index) would
        // divide-by-zero / index out of bounds and panic. Reject it with a typed error (F-033).
        let (start, end) = match self {
            LoraRowSlice::Chunk { n, index } => {
                if *n <= 0 || *index < 0 || *index >= *n {
                    return Err(Error::Msg(format!(
                        "LoraRowSlice::Chunk: invalid chunk spec (n={n}, index={index})"
                    )));
                }
                let chunk = rows / n;
                (index * chunk, (index + 1) * chunk)
            }
            LoraRowSlice::ChunkIfDivisible { n, index } => {
                if *n <= 0 || *index < 0 || *index >= *n {
                    return Err(Error::Msg(format!(
                        "LoraRowSlice::ChunkIfDivisible: invalid chunk spec (n={n}, index={index})"
                    )));
                }
                if rows % n != 0 {
                    return Ok(t.clone());
                }
                let chunk = rows / n;
                (index * chunk, (index + 1) * chunk)
            }
            LoraRowSlice::Dims { dims, index } => {
                let i = *index as usize;
                if *index < 0 || i >= dims.len() {
                    return Err(Error::Msg(format!(
                        "LoraRowSlice::Dims: index {index} out of range for {} dims",
                        dims.len()
                    )));
                }
                let start: i32 = dims[..i].iter().sum();
                (start, start + dims[i])
            }
        };
        // `t[start:end, :]` — byte-identical to the fork's slicing.
        Ok(t.try_index((start..end, ..))?)
    }
}

/// One BFL/ComfyUI adapter target: a set of source key spellings (across the `lora_unet_` /
/// `diffusion_model.` / `base_model.model.` prefix conventions) mapping to a diffusers module
/// `target_path`, with an optional [`LoraRowSlice`] on the up/down factor. A *fused* source (BFL
/// `…img_attn.qkv`) is named by SEVERAL `BflTarget`s — one per split destination (`to_q`/`to_k`/`to_v`)
/// — that share its key spellings but slice different rows; the loader fans the one source tensor into
/// all of them. A plain rename (BFL `img_in` → `x_embedder`) is a `BflTarget` with no slice. Mirrors a
/// fork `LoRATarget` restricted to its BFL patterns + up/down transforms.
#[derive(Clone, Debug)]
pub struct BflTarget {
    /// Diffusers module path that [`AdaptableHost::adaptable_mut`] resolves (concrete, no `{block}`).
    pub target_path: String,
    /// Source up-factor (`lora_up`/`lora_B`) key spellings.
    pub up_keys: Vec<String>,
    /// Source down-factor (`lora_down`/`lora_A`) key spellings.
    pub down_keys: Vec<String>,
    /// Source `alpha` key spellings (no transform, no transpose).
    pub alpha_keys: Vec<String>,
    /// Row-slice applied to the up factor (the qkv split). `None` for a plain rename.
    pub up_slice: Option<LoraRowSlice>,
    /// Row-slice applied to the down factor (shared-or-split). `None` for a plain rename.
    pub down_slice: Option<LoraRowSlice>,
}

/// One contribution of a source key to a target: which target/role it feeds and how to slice it.
struct BflSlot {
    target: String,
    role: KohyaRole,
    slice: Option<LoraRowSlice>,
}

/// Invert `targets` into `source_key → [contribution, …]`. One fused source key (e.g. a qkv
/// `lora_up`) contributes to multiple targets (q/k/v) with different slices, so the value is a list.
fn bfl_index(targets: &[BflTarget]) -> BTreeMap<String, Vec<BflSlot>> {
    let mut index: BTreeMap<String, Vec<BflSlot>> = BTreeMap::new();
    let mut push = |key: &str, target: &str, role: KohyaRole, slice: Option<LoraRowSlice>| {
        index.entry(key.to_string()).or_default().push(BflSlot {
            target: target.to_string(),
            role,
            slice,
        });
    };
    for t in targets {
        for k in &t.up_keys {
            push(k, &t.target_path, KohyaRole::Up, t.up_slice.clone());
        }
        for k in &t.down_keys {
            push(k, &t.target_path, KohyaRole::Down, t.down_slice.clone());
        }
        for k in &t.alpha_keys {
            push(k, &t.target_path, KohyaRole::Alpha, None);
        }
    }
    index
}

/// `true` if any key in `w` is a known BFL source key for `targets` — i.e. the file uses the BFL /
/// ComfyUI naming (`double_blocks`/`single_blocks`/`img_in`/… across the three prefix conventions),
/// which the diffusers/peft/standard-kohya paths cannot resolve. Precise: a standard diffusers/peft
/// or standard-kohya file shares none of these spellings, so it is never misrouted here.
pub fn is_bfl(w: &Weights, targets: &[BflTarget]) -> bool {
    if targets.is_empty() {
        return false;
    }
    let index = bfl_index(targets);
    w.keys().any(|k| index.contains_key(k))
}

/// Recognized LoRA factor suffixes — a key ending in one of these is adapter-shaped (vs. a base
/// weight or some bundled extra). Used to surface BFL-named keys that resolve to no target.
const LORA_FACTOR_SUFFIXES: [&str; 5] = [
    ".lora_up.weight",
    ".lora_down.weight",
    ".lora_A.weight",
    ".lora_B.weight",
    ".alpha",
];

/// Install a BFL / ComfyUI fused→split LoRA onto `host` (sc-2743). Each file key is matched against
/// the inverted [`BflTarget`] index; a matched *fused* source is row-sliced per destination and fanned
/// into the diffusers split targets (`…img_attn.qkv` → `attn.to_q/to_k/to_v`), a plain rename is copied
/// through. Resolved factors feed the same [`install_lora_groups`] path as PEFT/kohya (transpose +
/// `alpha/rank` fold), so a BFL file yields the byte-identical adapter to the equivalent diffusers
/// split-target LoRA.
///
/// An adapter-shaped key that matches NO target — an off-surface BFL key (e.g. a block out of range)
/// — is surfaced in `unmatched_paths` (loud, never silently dropped). A bundled text-encoder key
/// (`lora_te_…`/`text_encoder.…`) is not a denoiser target and is ignored, matching the PEFT/kohya
/// loaders' treatment of out-of-namespace keys.
pub fn apply_lora_bfl(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
    targets: &[BflTarget],
) -> Result<ApplyReport> {
    let index = bfl_index(targets);
    let mut groups: BTreeMap<String, LoraParts> = BTreeMap::new();
    let mut unresolved: BTreeSet<String> = BTreeSet::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        let Some(slots) = index.get(&key) else {
            // Not a BFL source key: surface it if it's an adapter-shaped denoiser key (off-surface),
            // ignore a bundled text-encoder adapter (`lora_te_…`/`…text_encoder.…`).
            let adapter_shaped = LORA_FACTOR_SUFFIXES.iter().any(|s| key.ends_with(s));
            let foreign_te = key.starts_with("lora_te") || key.contains("text_encoder");
            if adapter_shaped && !foreign_te {
                unresolved.insert(key);
            }
            continue;
        };
        let v = w.require(&key)?;
        for slot in slots {
            let parts = groups.entry(slot.target.clone()).or_default();
            match slot.role {
                KohyaRole::Down => {
                    parts.a = Some(match &slot.slice {
                        Some(s) => s.apply(v)?,
                        None => v.clone(),
                    });
                }
                KohyaRole::Up => {
                    parts.b = Some(match &slot.slice {
                        Some(s) => s.apply(v)?,
                        None => v.clone(),
                    });
                }
                KohyaRole::Alpha => parts.alpha = scalar_alpha(v)?,
            }
        }
    }

    // kohya / BFL files carry a per-target `.alpha` tensor, not the `lora_adapter_metadata` blob — no
    // blob to honor here (sc-5513).
    let mut report = install_lora_groups(host, groups, scale, None)?;
    report.unmatched_paths.extend(unresolved);
    Ok(report)
}

// ---- BFL / ComfyUI fused→split LyCORIS LoKr / LoHa (sc-8345) --------------------------------------

/// One BFL/ComfyUI LyCORIS destination: a split diffusers `target_path` plus the row-slice that carves
/// this destination's rows out of the *fused* reconstructed `[out,in]` delta (`None` = a plain rename,
/// the whole delta). Unlike [`BflTarget`] — which slices the raw LoRA up/down *factors* before the
/// residual transpose — a LyCORIS adapter reconstructs the full Kronecker/Hadamard delta first and
/// then slices its rows, so only the out-dim (`up`) slice is relevant here.
#[derive(Clone, Debug)]
struct BflLycorisTarget {
    target_path: String,
    out_slice: Option<LoraRowSlice>,
}

/// Build a `prefixed-module-path → [split target, …]` map from a host's [`BflTarget`] list, keyed by
/// the module path *as a LyCORIS file spells it* — every `up_key` minus its `.lora_up.weight` /
/// `.lora_B.weight` factor suffix, i.e. the `lora_unet_<flat>` / `diffusion_model.<dotted>` /
/// `base_model.model.<dotted>` BFL spellings. A fused qkv source maps to its three split targets (each
/// carrying its own out-dim slice); a rename maps to one target with no slice. The LyCORIS analog of
/// [`bfl_index`] (which keys by full factor key + role because LoRA slices factors, not the delta).
fn bfl_lycoris_module_map(targets: &[BflTarget]) -> BTreeMap<String, Vec<BflLycorisTarget>> {
    let mut map: BTreeMap<String, Vec<BflLycorisTarget>> = BTreeMap::new();
    for t in targets {
        for up in &t.up_keys {
            let Some(module) = up
                .strip_suffix(".lora_up.weight")
                .or_else(|| up.strip_suffix(".lora_B.weight"))
            else {
                continue;
            };
            let entry = map.entry(module.to_string()).or_default();
            // The same module key appears under both the `lora_up` and `lora_B` spellings; keep one
            // entry per destination.
            if entry.iter().all(|e| e.target_path != t.target_path) {
                entry.push(BflLycorisTarget {
                    target_path: t.target_path.clone(),
                    out_slice: t.up_slice.clone(),
                });
            }
        }
    }
    map
}

/// The LyCORIS module path a key belongs to — `key` minus a trailing `.lokr_*` / `.hada_*` / `.alpha`
/// factor suffix — or `None` if `key` is not a LyCORIS factor key.
fn lycoris_module_of(key: &str) -> Option<&str> {
    if let Some(module) = key.strip_suffix(".alpha") {
        return Some(module);
    }
    LOKR_TP_SUFFIXES
        .iter()
        .chain(LOHA_TP_SUFFIXES.iter())
        .find_map(|suffix| key.strip_suffix(suffix))
}

/// `true` if any LyCORIS factor key in `w` names a module in the BFL map — i.e. the file uses the
/// BFL/ComfyUI fused naming a host's `bfl_targets()` covers. A diffusers/bare/standard-kohya LyCORIS
/// file (modules like `transformer.…` or a `lora_unet_<diffusers-flat>` that resolves through the
/// kohya table) shares none of these spellings, so it is never misrouted here and stays on the
/// existing third-party/peft path. Empty map (a host with no BFL surface — every engine but FLUX.1/
/// FLUX.2) ⇒ always `false`.
fn is_bfl_lycoris(w: &Weights, map: &BTreeMap<String, Vec<BflLycorisTarget>>) -> bool {
    if map.is_empty() {
        return false;
    }
    w.keys()
        .any(|k| lycoris_module_of(k).is_some_and(|module| map.contains_key(module)))
}

/// Install grouped LyCORIS deltas onto a host's BFL fused→split targets (sc-8345). For each source
/// module the `reconstruct` closure rebuilds the FULL fused `[out,in]` delta (the host-fused qkv shape,
/// with the format's `alpha/rank` scale already baked in); each destination then row-slices its share
/// out of that delta and stacks it as an [`Adapter::Lokr`] residual at the user `scale`. The fused
/// `out` is the SUM of the destinations' out dims (3·inner for a qkv split, Σdims for FLUX.1's qkv+mlp,
/// the target's own out for a rename), so it is derived from the resolved targets rather than parsed
/// from the slice. A module absent from `map` is surfaced in `unmatched_paths`, never silently dropped.
fn install_bfl_lycoris<I, F>(
    host: &mut impl AdaptableHost,
    map: &BTreeMap<String, Vec<BflLycorisTarget>>,
    groups: I,
    scale: f32,
) -> Result<ApplyReport>
where
    I: IntoIterator<Item = (String, F)>,
    F: FnOnce(&[i32]) -> Result<Array>,
{
    let mut report = ApplyReport::default();
    for (module, reconstruct) in groups {
        let Some(targets) = map.get(&module) else {
            report.unmatched_paths.push(module);
            continue;
        };
        // Fused reconstruction shape: rows = Σ destination out dims, cols = the shared in dim. Resolve
        // each destination's base shape up front (the mutable apply borrow comes after reconstruction).
        let mut fused_out = 0i32;
        let mut in_dim: Option<i32> = None;
        let mut resolvable = true;
        for tgt in targets {
            let parts: Vec<&str> = tgt.target_path.split('.').collect();
            match host.adaptable_mut(&parts).map(|lin| lin.base_shape()) {
                Some(shape) if shape.len() == 2 => {
                    fused_out += shape[0];
                    in_dim.get_or_insert(shape[1]);
                }
                _ => {
                    resolvable = false;
                    break;
                }
            }
        }
        let (Some(in_dim), true) = (in_dim, resolvable) else {
            // A destination that didn't resolve (or a non-2-D linear) — surface the module rather than
            // install a partial, mis-shaped delta.
            report.unmatched_paths.push(module);
            continue;
        };
        let delta = reconstruct(&[fused_out, in_dim])?;
        for tgt in targets {
            let parts: Vec<&str> = tgt.target_path.split('.').collect();
            let Some(lin) = host.adaptable_mut(&parts) else {
                report.unmatched_paths.push(tgt.target_path.clone());
                continue;
            };
            let piece = match &tgt.out_slice {
                Some(slice) => slice.apply(&delta)?,
                None => delta.clone(),
            };
            lin.push(Adapter::Lokr {
                delta: piece,
                scale,
            });
            report.applied += 1;
        }
    }
    Ok(report)
}

/// Install a metadata-stamped (peft) LoKr file in BFL/ComfyUI fused naming (sc-8345). Same Kronecker
/// reconstruction + `alpha/rank` fold as [`apply_lokr`], but the fused qkv source is rebuilt at the
/// host-fused shape and row-sliced into the split targets via [`install_bfl_lycoris`].
fn apply_lokr_bfl(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
    map: &BTreeMap<String, Vec<BflLycorisTarget>>,
) -> Result<ApplyReport> {
    let file = parse_lokr(w)?;
    let (alpha, rank) = (file.alpha, file.rank);
    let groups = file.groups.into_iter().map(|(module, factors)| {
        (module, move |base: &[i32]| {
            reconstruct_lokr_delta(
                alpha,
                rank,
                base,
                factors.get("lokr_w1"),
                factors.get("lokr_w1_a"),
                factors.get("lokr_w1_b"),
                factors.get("lokr_w2"),
                factors.get("lokr_w2_a"),
                factors.get("lokr_w2_b"),
                Dtype::Bfloat16,
            )
        })
    });
    install_bfl_lycoris(host, map, groups, scale)
}

/// Install a third-party LyCORIS **LoKr** file in BFL/ComfyUI fused naming (sc-8345). Per-module
/// lycoris scale + tucker-capable Kronecker reconstruction (same as [`apply_lokr_thirdparty`]), fused→
/// split via [`install_bfl_lycoris`].
fn apply_lokr_thirdparty_bfl(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
    map: &BTreeMap<String, Vec<BflLycorisTarget>>,
) -> Result<ApplyReport> {
    let groups = parse_lokr_thirdparty(w)?
        .into_iter()
        .map(|(module, g)| (module, move |base: &[i32]| g.delta(base, Dtype::Bfloat16)));
    install_bfl_lycoris(host, map, groups, scale)
}

/// Install a third-party LyCORIS **LoHa** file in BFL/ComfyUI fused naming (sc-8345). Hadamard
/// reconstruction (same as [`apply_loha_thirdparty`]), fused→split via [`install_bfl_lycoris`].
fn apply_loha_thirdparty_bfl(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
    map: &BTreeMap<String, Vec<BflLycorisTarget>>,
) -> Result<ApplyReport> {
    let groups = parse_loha_thirdparty(w)?
        .into_iter()
        .map(|(module, g)| (module, move |base: &[i32]| g.delta(base, Dtype::Bfloat16)));
    install_bfl_lycoris(host, map, groups, scale)
}

/// Load and install every adapter in `specs` onto `host`, stacking in order. Each spec's file is
/// read, dispatched to the LoKr or PEFT-LoRA loader by its [`AdapterKind`], applied at `spec.scale`,
/// and its [`ApplyReport`] merged into the combined result — unmatched target paths are surfaced,
/// never silently dropped. `lora_strip_prefix` is the per-family namespace stripped from PEFT-LoRA
/// keys (e.g. `"transformer."`); it does not apply to LoKr (whose keys are bare module paths).
///
/// This is the load-time seam (sc-2534): a provider calls it inside `load()` with its model's
/// [`AdaptableHost`] while the model is still mutable. Empty `specs` is a no-op (empty report).
///
/// **Scope (F-035):** this fixed-prefix variant routes PEFT/diffusers LoRA, metadata-stamped LoKr,
/// and keyless third-party LyCORIS LoKr/LoHa. Unlike [`apply_adapter_specs_autoprefix`] it does NOT
/// detect BFL/ComfyUI fused→split or kohya-flattened files — both require walking the host module
/// tree (`bfl_targets()` / `kohya_table()`), which only the autoprefix path does. Callers that may
/// receive BFL or kohya files must use the autoprefix variant; here such a file would resolve no
/// targets and be reported as fully unmatched rather than applied.
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
                // A third-party LyCORIS LoKr (sc-3642) / LoHa (sc-3643) carries `lokr_*` / `hada_*`
                // keys but no `networkType` stamp, so a caller can't know to set a non-Lora kind —
                // detect + route by keys.
                if is_lokr_keys(&w) {
                    apply_lokr_thirdparty(host, &w, spec.scale)?
                } else if is_loha_keys(&w) {
                    apply_loha_thirdparty(host, &w, spec.scale)?
                } else {
                    apply_lora_peft(host, &w, spec.scale, lora_strip_prefix)?
                }
            }
        };
        combined.applied += report.applied;
        combined.unmatched_paths.extend(report.unmatched_paths);
    }
    Ok(combined)
}

// `COMMON_LORA_PREFIXES` — the LoRA key namespace prefixes diffusers/peft adapter files use, tried
// in order; the first a key begins with is stripped. LoKr files are bare (no prefix); kohya
// `lora_unet_…` files flatten the module dots to underscores and resolve through a separate
// flattened→dotted table ([`apply_lora_kohya`], sc-2618), not this prefix strip. SceneWorks' trained
// LoRAs use `transformer.` (peft `save_lora_weights`) or `diffusion_model.` (sd-scripts export) —
// both observed on real files. Defined in gen-core (`weightsmeta`) and re-exported above.

/// The LoRA namespace prefix present in `w`'s keys, if any (see [`COMMON_LORA_PREFIXES`]).
pub fn detect_lora_prefix(w: &Weights) -> Option<&'static str> {
    wmeta::detect_lora_prefix(w.keys())
}

/// [`apply_adapter_specs`] with per-file LoRA-prefix **auto-detection** ([`detect_lora_prefix`])
/// instead of a fixed prefix — the common provider path, since LoRA files vary
/// (`transformer.` / `diffusion_model.` / bare) while LoKr keys are bare. The host's key→module map
/// must match the (prefix-stripped) diffusers module paths.
pub fn apply_adapter_specs_autoprefix(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
) -> Result<ApplyReport> {
    // The kohya `flattened → dotted` table and the BFL target list both walk the model tree, so
    // build each lazily and only once, the first time it is needed across `specs`.
    let mut kohya: Option<BTreeMap<String, String>> = None;
    let mut bfl: Option<Vec<BflTarget>> = None;
    let mut bfl_lyc: Option<BTreeMap<String, Vec<BflLycorisTarget>>> = None;
    let mut combined = ApplyReport::default();
    for spec in specs {
        // Load + classify the file once: the dispatch chain below (and the fallback that used to
        // delegate to `apply_adapter_specs`, re-reading the file) all key off `is_lokr`, so hoist both
        // the loaded `Weights` and `is_lokr(&w)` into locals rather than re-reading/re-evaluating
        // them up to four times per spec (F-004).
        let w = Weights::from_file(&spec.path)?;
        let is_lokr_w = is_lokr(&w);
        // A LyCORIS file (peft/metadata LoKr, keyed third-party LoKr, or LoHa) can ship in BFL/ComfyUI
        // fused naming on a FLUX host; detect that up front so the three LyCORIS arms below route to the
        // fused→split appliers (sc-8345). Non-FLUX hosts have an empty BFL surface ⇒ never matches.
        let is_lokr_keys_w = !is_lokr_w && is_lokr_keys(&w);
        let is_loha_keys_w = is_loha_keys(&w);
        let is_bfl_lycoris_file = if is_lokr_w || is_lokr_keys_w || is_loha_keys_w {
            if bfl.is_none() {
                bfl = Some(host.bfl_targets());
            }
            if bfl_lyc.is_none() {
                bfl_lyc = Some(bfl_lycoris_module_map(bfl.as_ref().unwrap()));
            }
            is_bfl_lycoris(&w, bfl_lyc.as_ref().unwrap())
        } else {
            false
        };
        // BFL / ComfyUI fused→split LoRA naming (sc-2743) is the orthogonal axis to kohya flattening and
        // shares the `lora_unet_` prefix, so it must be detected BEFORE `is_kohya`. (LoKr first.)
        let is_bfl_file = if is_lokr_w {
            false
        } else {
            if bfl.is_none() {
                bfl = Some(host.bfl_targets());
            }
            is_bfl(&w, bfl.as_ref().unwrap())
        };
        let report = if is_lokr_keys_w {
            // Third-party LyCORIS LoKr (sc-3642): `lokr_*` keys, no `networkType` stamp. Route to the
            // fused→split applier when BFL/ComfyUI-named (sc-8345), else the bare/diffusers/kohya path.
            // Detected BEFORE is_bfl/is_kohya — a kohya-flattened LoKr also carries the `lora_unet_`
            // prefix, so is_kohya would otherwise claim it and apply nothing.
            if is_bfl_lycoris_file {
                apply_lokr_thirdparty_bfl(host, &w, spec.scale, bfl_lyc.as_ref().unwrap())?
            } else {
                apply_lokr_thirdparty(host, &w, spec.scale)?
            }
        } else if is_loha_keys_w {
            // Third-party LyCORIS LoHa (sc-3643): `hada_*` keys. Same reasoning — BFL fused→split when
            // BFL-named (sc-8345), else the bare/diffusers/kohya path.
            if is_bfl_lycoris_file {
                apply_loha_thirdparty_bfl(host, &w, spec.scale, bfl_lyc.as_ref().unwrap())?
            } else {
                apply_loha_thirdparty(host, &w, spec.scale)?
            }
        } else if is_bfl_file {
            apply_lora_bfl(host, &w, spec.scale, bfl.as_ref().unwrap())?
        } else if !is_lokr_w && is_kohya(&w) {
            // kohya LoRA: dots are flattened to underscores, so keys resolve through the table
            // rather than the prefix-strip path. (LoKr keeps dotted paths; checked first.)
            if kohya.is_none() {
                kohya = Some(kohya_table(&host.adaptable_paths()));
            }
            apply_lora_kohya(host, &w, spec.scale, kohya.as_ref().unwrap())?
        } else {
            // Plain PEFT/diffusers LoRA (the common case) or a metadata-LoKr. The earlier branches
            // already excluded third-party LoKr/LoHa keys, so call the leaf appliers directly with the
            // already-loaded `w` instead of re-reading the file via `apply_adapter_specs` (F-004).
            match spec.kind {
                AdapterKind::Lokr => {
                    // metadata-stamped LoKr — BFL fused→split when BFL/ComfyUI-named (sc-8345), else
                    // the bare dotted-path applier.
                    if is_bfl_lycoris_file {
                        apply_lokr_bfl(host, &w, spec.scale, bfl_lyc.as_ref().unwrap())?
                    } else {
                        apply_lokr(host, &w, spec.scale)?
                    }
                }
                AdapterKind::Lora => {
                    if is_lokr_w {
                        // The file's metadata is authoritative; a kind/metadata mismatch is a caller
                        // error (matches `apply_adapter_specs`).
                        return Err(format!(
                            "adapter {} declared Lora but its metadata says networkType=lokr",
                            spec.path.display()
                        )
                        .into());
                    }
                    apply_lora_peft(host, &w, spec.scale, detect_lora_prefix(&w))?
                }
            }
        };
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
             format/prefix (expected diffusers/peft LoRA, kohya `lora_unet_` LoRA, BFL/ComfyUI \
             fused→split LoRA — for a host with a BFL surface — or LoKr keys)",
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
    use mlx_rs::ops::indexing::TryIndexOp;
    use mlx_rs::ops::{all_close, array_eq};
    use std::collections::HashMap;
    use std::path::PathBuf;

    /// A host whose modules live at arbitrary dotted paths — including segment names with internal
    /// underscores (`feed_forward`, `to_out.0`) so the kohya flattening is genuinely ambiguous and a
    /// blind `_`→`.` split would mis-route. `adaptable_paths` returns the registered paths, so it
    /// exercises the real `flattened → dotted` table path.
    struct MultiHost {
        mods: HashMap<String, AdaptableLinear>,
        paths: Vec<String>,
    }
    impl MultiHost {
        fn new(specs: &[(&str, Array)]) -> Self {
            let mut mods = HashMap::new();
            let mut paths = Vec::new();
            for (p, w) in specs {
                mods.insert((*p).to_string(), AdaptableLinear::dense(w.clone(), None));
                paths.push((*p).to_string());
            }
            Self { mods, paths }
        }
    }
    impl AdaptableHost for MultiHost {
        fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
            self.mods.get_mut(&path.join("."))
        }
        fn adaptable_paths(&self) -> Vec<String> {
            self.paths.clone()
        }
    }

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
    fn lora_peft_honors_lora_adapter_metadata_alpha() {
        // sc-5513: a diffusers / PEFT `save_lora_adapter` file carries NO per-target `.alpha` tensor —
        // the scaling lives in the `lora_adapter_metadata` header blob. With `lora_alpha = 16`, `r = 8`
        // the PEFT loader must fold `(16/8) = 2.0` (the metadata strength), not the pre-sc-5513
        // `alpha = rank` default (factor 1.0). Proves the blob is read and applied.
        let weight = Array::from_slice(
            &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &[4, 3],
        );
        // PEFT factors with a TRUE rank of 8 (matching the blob `r`): A [r=8, in=3], B [out=4, r=8].
        let a_raw = Array::from_slice(
            &(0..24).map(|i| i as f32 * 0.03 - 0.3).collect::<Vec<_>>(),
            &[8, 3],
        );
        let b_raw = Array::from_slice(
            &(0..32).map(|i| 0.4 - i as f32 * 0.02).collect::<Vec<_>>(),
            &[4, 8],
        );

        let path = tmp("lora_adapter_metadata.safetensors");
        let meta = HashMap::from([(
            "lora_adapter_metadata".to_string(),
            r#"{"lora_alpha": 16, "r": 8}"#.to_string(),
        )]);
        // Deliberately NO `lin.alpha` tensor — the scaling must come from the blob.
        Array::save_safetensors(
            vec![("lin.lora_A.weight", &a_raw), ("lin.lora_B.weight", &b_raw)],
            Some(&meta),
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();
        assert!(w.metadata("lora_adapter_metadata").is_some());

        let mut host = OneLinear {
            lin: AdaptableLinear::dense(weight.clone(), None),
        };
        let report = apply_lora_peft(&mut host, &w, 1.0, None).unwrap();
        assert_eq!(report.applied, 1);

        // Reference: alpha 16 over rank 8 ⇒ factor 2.0 folded into B (scale 1.0).
        let mut expected = AdaptableLinear::dense(weight.clone(), None);
        expected.push(Adapter::Lora {
            a: a_raw.t(),
            b: b_raw
                .t()
                .multiply(Array::from_slice(&[2.0f32], &[1]))
                .unwrap(),
            scale: 1.0,
        });
        // The pre-sc-5513 default (alpha = rank ⇒ factor 1.0) would diverge by a full factor of 2.
        let mut buggy = AdaptableLinear::dense(weight, None);
        buggy.push(Adapter::Lora {
            a: a_raw.t(),
            b: b_raw.t(),
            scale: 1.0,
        });

        let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
        let got = host.lin.forward(&x).unwrap();
        let want = expected.forward(&x).unwrap();
        let old = buggy.forward(&x).unwrap();
        assert!(
            all_close(&got, &want, 1e-5, 1e-5, false)
                .unwrap()
                .item::<bool>(),
            "metadata-alpha fold must match (16/8)·scale"
        );
        assert!(
            !all_close(&got, &old, 1e-4, 1e-4, false)
                .unwrap()
                .item::<bool>(),
            "metadata alpha must differ from the alpha=rank default"
        );
    }

    /// sc-5513 **live torch-PEFT A/B** (the epic 3641 / sc-3671 on-device harness; torch IS available
    /// on the Mac at `~/mlx-flux-venv`). `#[ignore]` — gated on a real diffusers `save_lora_adapter`
    /// file, generated with peft 0.19 + diffusers 0.37:
    /// ```text
    /// ~/mlx-flux-venv/bin/python - <<'PY'
    /// import os, torch
    /// from diffusers import UNet2DConditionModel
    /// from peft import LoraConfig
    /// torch.manual_seed(0)
    /// unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
    ///     block_out_channels=(16,32), down_block_types=("CrossAttnDownBlock2D","DownBlock2D"),
    ///     up_block_types=("UpBlock2D","CrossAttnUpBlock2D"), cross_attention_dim=16,
    ///     attention_head_dim=2, norm_num_groups=4)
    /// unet.add_adapter(LoraConfig(r=8, lora_alpha=16, target_modules=["to_q","to_k","to_v"],
    ///     alpha_pattern={"to_k":8}, rank_pattern={"to_k":16}, init_lora_weights=False))
    /// unet.save_lora_adapter("/tmp/sc5513_adapter")
    /// PY
    /// SC5513_PEFT_ADAPTER=/tmp/sc5513_adapter/pytorch_lora_weights.safetensors \
    ///   cargo test -p mlx-gen peft_lora_adapter_metadata_ab -- --ignored --nocapture
    /// ```
    /// peft's authoritative per-module scaling (`mod.scaling['default']`) is then `to_q`/`to_v` = 16/8 =
    /// 2.0 and `to_k` = 8/16 = 0.5 (the override is deliberately discriminating). The Rust core loader
    /// must install each residual at exactly that scaling — proving the `lora_adapter_metadata` blob is
    /// honored on a genuine torch file (which carries NO per-target `.alpha` tensor, the bug's premise).
    #[test]
    #[ignore = "needs a diffusers save_lora_adapter file via SC5513_PEFT_ADAPTER (see doc comment)"]
    fn peft_lora_adapter_metadata_ab() {
        let Ok(path) = std::env::var("SC5513_PEFT_ADAPTER") else {
            eprintln!("SC5513_PEFT_ADAPTER unset — skipping live torch A/B");
            return;
        };
        let w = Weights::from_file(&path).unwrap();
        // The whole premise: a real diffusers `save_lora_adapter` file ships NO per-target `.alpha`
        // tensor — the scaling lives in the `lora_adapter_metadata` blob.
        assert!(
            !w.keys().any(|k| k.ends_with(".alpha")),
            "diffusers save_lora_adapter must not ship a per-target .alpha tensor"
        );
        assert!(w.metadata(wmeta::LORA_ADAPTER_METADATA_KEY).is_some());

        let block = "down_blocks.0.attentions.0.transformer_blocks.0.attn1";
        // peft ground truth (confirmed independently via `mod.scaling`): global 2.0, `to_k` override 0.5.
        for (leaf, peft_scale) in [("to_q", 2.0f32), ("to_k", 0.5f32)] {
            let module = format!("{block}.{leaf}");
            let a_raw = w
                .require(&format!("{module}.lora_A.weight"))
                .unwrap()
                .clone();
            let b_raw = w
                .require(&format!("{module}.lora_B.weight"))
                .unwrap()
                .clone();
            let out = b_raw.shape()[0];
            let inp = a_raw.shape()[1];
            // Zero base so the forward IS the pure (scaled) residual.
            let zero = Array::from_slice(&vec![0f32; (out * inp) as usize], &[out, inp]);
            let mut host = MultiHost::new(&[(module.as_str(), zero.clone())]);
            let report = apply_lora_peft(&mut host, &w, 1.0, None).unwrap();
            assert!(report.applied >= 1, "{leaf}: not applied");

            // Reference at peft's ground-truth scaling: residual = (x·Aᵀ·Bᵀ)·peft_scale.
            let mut expect = AdaptableLinear::dense(zero, None);
            expect.push(Adapter::Lora {
                a: a_raw.t(),
                b: b_raw
                    .t()
                    .multiply(Array::from_slice(&[peft_scale], &[1]))
                    .unwrap(),
                scale: 1.0,
            });
            let x = Array::from_slice(
                &(0..inp).map(|i| (i as f32 * 0.3).sin()).collect::<Vec<_>>(),
                &[1, inp],
            );
            let segs: Vec<&str> = module.split('.').collect();
            let got = host.adaptable_mut(&segs).unwrap().forward(&x).unwrap();
            let want = expect.forward(&x).unwrap();
            assert!(
                all_close(&got, &want, 1e-4, 1e-4, false)
                    .unwrap()
                    .item::<bool>(),
                "{leaf}: Rust apply diverged from peft scaling {peft_scale}"
            );
            println!("OK {leaf}: Rust apply matches peft scaling {peft_scale}");
        }
    }

    #[test]
    fn lora_bf16_scalar_alpha_reads_without_panic() {
        // sc-2657: real kohya/BFL FLUX LoRAs ship `alpha` as a **bf16 scalar of shape []**. The alpha
        // read used `as_slice::<f32>()`, which `unwrap`s a dtype mismatch and would panic on bf16 — a
        // latent bug masked by every prior test synthesizing f32 alpha. The fix casts to f32 first.
        // Here a bf16 `[]`-shaped alpha must load AND fold identically to its f32 `[1]`-shaped twin.
        let weight = Array::from_slice(
            &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &[4, 3],
        );
        let a_raw = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]);
        let b_raw = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
        // alpha = 16 (exactly representable in bf16), stored bf16 and 0-d — like the real file.
        let alpha_bf16 = Array::from_slice(&[16.0f32], &[1])
            .reshape(&[] as &[i32])
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();

        let path = tmp("lora_bf16_alpha.safetensors");
        Array::save_safetensors(
            vec![
                ("lin.lora_A.weight", &a_raw),
                ("lin.lora_B.weight", &b_raw),
                ("lin.alpha", &alpha_bf16),
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
        assert_eq!(report.applied, 1, "bf16 alpha LoRA should apply, not panic");

        // Reference: identical fold with alpha=16, rank=2 → factor 8.
        let mut expected = AdaptableLinear::dense(weight, None);
        expected.push(Adapter::Lora {
            a: a_raw.t(),
            b: b_raw
                .t()
                .multiply(Array::from_slice(&[8.0f32], &[1]))
                .unwrap(),
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
    fn scalar_alpha_empty_tensor_errors_not_panic() {
        // sc-3959 added the no-panic guard for a malformed third-party adapter with a zero-length
        // `.alpha` (before it, `as_slice::<f32>()` panicked on the size-0 array, aborting the worker).
        // F-031 then tightened it from a silent `Ok(None)` to a typed error: the callers only reach
        // `scalar_alpha` for a *present* alpha key, so present-but-empty is a malformed file — returning
        // `None` would be indistinguishable from "no alpha" and silently mis-scale the adapter.
        let empty = Array::from_slice(&[] as &[f32], &[0]);
        assert_eq!(empty.size(), 0);
        let err = scalar_alpha(&empty)
            .expect_err("size-0 alpha must be a typed error, not Ok")
            .to_string();
        assert!(err.contains("empty"), "got: {err}");
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

    // ---- BFL/ComfyUI fused→split LyCORIS (sc-8345) ------------------------------------------------

    /// A host with a BFL surface: the three split attention projections (`to_q/to_k/to_v`, the fused-
    /// qkv destinations) plus a rename destination (`to_out`), and a `bfl_targets()` mapping the fused
    /// `diffusion_model.double_blocks.0.img_attn.{qkv,proj}` BFL names onto them — the minimal shape of
    /// a FLUX `Flux2Transformer` for the LyCORIS fused→split path.
    struct BflHost {
        mods: HashMap<String, AdaptableLinear>,
    }
    impl BflHost {
        fn new() -> Self {
            let mut mods = HashMap::new();
            // qkv splits: each [out=2, in=3] → the fused source is [6, 3].
            for dst in ["to_q", "to_k", "to_v"] {
                mods.insert(
                    format!("transformer_blocks.0.attn.{dst}"),
                    AdaptableLinear::dense(Array::from_slice(&[0.0f32; 6], &[2, 3]), None),
                );
            }
            // rename dest: [out=4, in=4].
            mods.insert(
                "transformer_blocks.0.attn.to_out".to_string(),
                AdaptableLinear::dense(Array::from_slice(&[0.0f32; 16], &[4, 4]), None),
            );
            Self { mods }
        }
    }
    impl AdaptableHost for BflHost {
        fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
            self.mods.get_mut(&path.join("."))
        }
        fn adaptable_paths(&self) -> Vec<String> {
            self.mods.keys().cloned().collect()
        }
        fn bfl_targets(&self) -> Vec<BflTarget> {
            let block_keys = |module: &str| {
                (
                    vec![
                        format!("diffusion_model.{module}.lora_B.weight"),
                        format!("diffusion_model.{module}.lora_up.weight"),
                    ],
                    vec![
                        format!("diffusion_model.{module}.lora_A.weight"),
                        format!("diffusion_model.{module}.lora_down.weight"),
                    ],
                    vec![format!("diffusion_model.{module}.alpha")],
                )
            };
            let mut out = Vec::new();
            let (up, down, alpha) = block_keys("double_blocks.0.img_attn.qkv");
            for (idx, dst) in ["to_q", "to_k", "to_v"].iter().enumerate() {
                out.push(BflTarget {
                    target_path: format!("transformer_blocks.0.attn.{dst}"),
                    up_keys: up.clone(),
                    down_keys: down.clone(),
                    alpha_keys: alpha.clone(),
                    up_slice: Some(LoraRowSlice::Chunk {
                        n: 3,
                        index: idx as i32,
                    }),
                    down_slice: Some(LoraRowSlice::ChunkIfDivisible {
                        n: 3,
                        index: idx as i32,
                    }),
                });
            }
            let (up, down, alpha) = block_keys("double_blocks.0.img_attn.proj");
            out.push(BflTarget {
                target_path: "transformer_blocks.0.attn.to_out".to_string(),
                up_keys: up,
                down_keys: down,
                alpha_keys: alpha,
                up_slice: None,
                down_slice: None,
            });
            out
        }
    }

    /// A `networkType=lokr` file in BFL/ComfyUI fused naming (`diffusion_model.…img_attn.qkv` fused,
    /// `…img_attn.proj` renamed) must apply onto a FLUX-shaped host: the fused qkv delta is rebuilt at
    /// `[6,3]` and row-sliced into `to_q/to_k/to_v`, and the proj rename lands whole on `to_out`. Before
    /// sc-8345 every target surfaced as unmatched (the strict apply errored). Exercises the full
    /// `apply_adapters_strict` dispatch, not just the leaf applier.
    #[test]
    fn bfl_named_lokr_fused_qkv_and_rename_resolve() {
        // Fused qkv LoKr: kron(w1[3,1], w2[2,3]) → [6,3].
        let qkv_w1 = Array::from_slice(&[1.0f32, 0.5, -0.25], &[3, 1]);
        let qkv_w2 = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
        // Proj rename LoKr: kron(w1[2,2], w2[2,2]) → [4,4].
        let proj_w1 = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]);
        let proj_w2 = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
        let mut meta = HashMap::new();
        meta.insert("networkType".to_string(), "lokr".to_string());
        meta.insert("alpha".to_string(), "1.0".to_string());
        meta.insert("rank".to_string(), "1".to_string());
        let path = tmp("bfl_lokr_fused.safetensors");
        Array::save_safetensors(
            vec![
                (
                    "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w1",
                    &qkv_w1,
                ),
                (
                    "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w2",
                    &qkv_w2,
                ),
                (
                    "diffusion_model.double_blocks.0.img_attn.proj.lokr_w1",
                    &proj_w1,
                ),
                (
                    "diffusion_model.double_blocks.0.img_attn.proj.lokr_w2",
                    &proj_w2,
                ),
            ],
            Some(&meta),
            &path,
        )
        .unwrap();

        let mut host = BflHost::new();
        let spec = AdapterSpec::new(path, 1.0, AdapterKind::Lokr);
        let report =
            apply_adapters_strict(&mut host, std::slice::from_ref(&spec), "flux2_klein_9b")
                .unwrap();
        assert_eq!(report.applied, 4);
        assert!(report.unmatched_paths.is_empty());

        // The fused qkv delta, reconstructed independently at the fused shape, row-sliced into thirds.
        let full = reconstruct_lokr_delta(
            1.0,
            1.0,
            &[6, 3],
            Some(&qkv_w1),
            None,
            None,
            Some(&qkv_w2),
            None,
            None,
            Dtype::Bfloat16,
        )
        .unwrap();
        for (idx, dst) in ["to_q", "to_k", "to_v"].iter().enumerate() {
            let lin = host
                .adaptable_mut(&["transformer_blocks", "0", "attn", dst])
                .unwrap();
            let Adapter::Lokr { delta, scale } = &lin.adapters()[0] else {
                panic!("expected a LoKr adapter on {dst}");
            };
            assert_eq!(*scale, 1.0);
            let start = idx as i32 * 2;
            let want = full.try_index((start..start + 2, ..)).unwrap();
            assert!(
                all_close(delta, &want, 1e-5, 1e-5, false)
                    .unwrap()
                    .item::<bool>(),
                "qkv split {dst} delta mismatch"
            );
        }

        // Proj rename lands whole on to_out.
        let proj_full = reconstruct_lokr_delta(
            1.0,
            1.0,
            &[4, 4],
            Some(&proj_w1),
            None,
            None,
            Some(&proj_w2),
            None,
            None,
            Dtype::Bfloat16,
        )
        .unwrap();
        let lin = host
            .adaptable_mut(&["transformer_blocks", "0", "attn", "to_out"])
            .unwrap();
        let Adapter::Lokr { delta, .. } = &lin.adapters()[0] else {
            panic!("expected a LoKr adapter on to_out");
        };
        assert!(all_close(delta, &proj_full, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>());
    }

    /// A bare diffusers-named LoKr (split `to_q` directly, no BFL fused name) on the SAME BFL host must
    /// still route through the ordinary `apply_lokr` path, NOT the fused→split one — `is_bfl_lycoris`
    /// keys only off the BFL spellings, so non-BFL LyCORIS is untouched by sc-8345.
    #[test]
    fn bare_diffusers_lokr_on_bfl_host_stays_on_plain_path() {
        // kron(w1[2,1], w2[1,3]) → [2,3], the shape of the split to_q.
        let w1 = Array::from_slice(&[1.0f32, 0.5], &[2, 1]);
        let w2 = Array::from_slice(&[0.1f32, 0.2, 0.3], &[1, 3]);
        let mut meta = HashMap::new();
        meta.insert("networkType".to_string(), "lokr".to_string());
        meta.insert("alpha".to_string(), "1.0".to_string());
        meta.insert("rank".to_string(), "1".to_string());
        let path = tmp("bare_lokr_on_bfl_host.safetensors");
        Array::save_safetensors(
            vec![
                ("transformer_blocks.0.attn.to_q.lokr_w1", &w1),
                ("transformer_blocks.0.attn.to_q.lokr_w2", &w2),
            ],
            Some(&meta),
            &path,
        )
        .unwrap();
        let mut host = BflHost::new();
        let spec = AdapterSpec::new(path, 1.0, AdapterKind::Lokr);
        let report =
            apply_adapter_specs_autoprefix(&mut host, std::slice::from_ref(&spec)).unwrap();
        assert_eq!(report.applied, 1);
        assert!(report.unmatched_paths.is_empty());
        let lin = host
            .adaptable_mut(&["transformer_blocks", "0", "attn", "to_q"])
            .unwrap();
        assert!(matches!(lin.adapters()[0], Adapter::Lokr { .. }));
    }

    /// A third-party LyCORIS LoKr (no `networkType` stamp — detected by `lokr_*` keys) in BFL fused
    /// naming routes through the fused→split applier too (sc-8345). Both-full factors ⇒ lycoris scale 1.
    #[test]
    fn bfl_named_thirdparty_lokr_fused_qkv_resolves() {
        // kron(w1[3,1], w2[2,3]) → [6,3]; both factors full ⇒ scale 1.0.
        let qkv_w1 = Array::from_slice(&[1.0f32, 0.5, -0.25], &[3, 1]);
        let qkv_w2 = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
        let proj_w1 = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]);
        let proj_w2 = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
        let path = tmp("bfl_tp_lokr_fused.safetensors");
        // NO `networkType` metadata → is_lokr() false, is_lokr_keys() true (the third-party path).
        Array::save_safetensors(
            vec![
                (
                    "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w1",
                    &qkv_w1,
                ),
                (
                    "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w2",
                    &qkv_w2,
                ),
                (
                    "diffusion_model.double_blocks.0.img_attn.proj.lokr_w1",
                    &proj_w1,
                ),
                (
                    "diffusion_model.double_blocks.0.img_attn.proj.lokr_w2",
                    &proj_w2,
                ),
            ],
            None,
            &path,
        )
        .unwrap();
        let mut host = BflHost::new();
        let spec = AdapterSpec::new(path, 1.0, AdapterKind::Lora); // kind irrelevant — keys route it
        let report =
            apply_adapters_strict(&mut host, std::slice::from_ref(&spec), "flux2_klein_9b")
                .unwrap();
        assert_eq!(report.applied, 4);
        assert!(report.unmatched_paths.is_empty());

        let full = reconstruct_lokr_delta_scaled(
            1.0,
            &[6, 3],
            Some(&qkv_w1),
            None,
            None,
            Some(&qkv_w2),
            None,
            None,
            None,
            Dtype::Bfloat16,
        )
        .unwrap();
        for (idx, dst) in ["to_q", "to_k", "to_v"].iter().enumerate() {
            let lin = host
                .adaptable_mut(&["transformer_blocks", "0", "attn", dst])
                .unwrap();
            let Adapter::Lokr { delta, .. } = &lin.adapters()[0] else {
                panic!("expected a LoKr adapter on {dst}");
            };
            let start = idx as i32 * 2;
            let want = full.try_index((start..start + 2, ..)).unwrap();
            assert!(all_close(delta, &want, 1e-5, 1e-5, false)
                .unwrap()
                .item::<bool>());
        }
    }

    /// A third-party LyCORIS LoHa (`hada_*` keys) in BFL fused naming routes through the fused→split
    /// applier (sc-8345); the Hadamard delta is rebuilt at the fused shape, then row-sliced.
    #[test]
    fn bfl_named_loha_fused_qkv_resolves() {
        // (w1_a@w1_b) ⊙ (w2_a@w2_b) at [6,3], rank r=1 ⇒ scale 1.0.
        let w1_a = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[6, 1]);
        let w1_b = Array::from_slice(&[1.0f32, -1.0, 0.5], &[1, 3]);
        let w2_a = Array::from_slice(&[0.6f32, 0.5, 0.4, 0.3, 0.2, 0.1], &[6, 1]);
        let w2_b = Array::from_slice(&[0.2f32, 0.4, -0.2], &[1, 3]);
        let path = tmp("bfl_loha_fused.safetensors");
        Array::save_safetensors(
            vec![
                (
                    "diffusion_model.double_blocks.0.img_attn.qkv.hada_w1_a",
                    &w1_a,
                ),
                (
                    "diffusion_model.double_blocks.0.img_attn.qkv.hada_w1_b",
                    &w1_b,
                ),
                (
                    "diffusion_model.double_blocks.0.img_attn.qkv.hada_w2_a",
                    &w2_a,
                ),
                (
                    "diffusion_model.double_blocks.0.img_attn.qkv.hada_w2_b",
                    &w2_b,
                ),
            ],
            None,
            &path,
        )
        .unwrap();
        let mut host = BflHost::new();
        let spec = AdapterSpec::new(path, 1.0, AdapterKind::Lora);
        let report =
            apply_adapters_strict(&mut host, std::slice::from_ref(&spec), "flux2_klein_9b")
                .unwrap();
        // Only the fused qkv is present → the three splits; proj/to_out untouched (no factors for it).
        assert_eq!(report.applied, 3);
        assert!(report.unmatched_paths.is_empty());

        let full = reconstruct_loha_delta(
            1.0,
            &[6, 3],
            &w1_a,
            &w1_b,
            &w2_a,
            &w2_b,
            None,
            None,
            Dtype::Bfloat16,
        )
        .unwrap();
        for (idx, dst) in ["to_q", "to_k", "to_v"].iter().enumerate() {
            let lin = host
                .adaptable_mut(&["transformer_blocks", "0", "attn", dst])
                .unwrap();
            let Adapter::Lokr { delta, .. } = &lin.adapters()[0] else {
                panic!("expected an installed delta on {dst}");
            };
            let start = idx as i32 * 2;
            let want = full.try_index((start..start + 2, ..)).unwrap();
            assert!(all_close(delta, &want, 1e-5, 1e-5, false)
                .unwrap()
                .item::<bool>());
        }
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
                pass_scales: None,
                moe_expert: None,
            },
            AdapterSpec {
                path: lokr_path.clone(),
                scale: 1.0,
                kind: AdapterKind::Lokr,
                pass_scales: None,
                moe_expert: None,
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
            pass_scales: None,
            moe_expert: None,
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
            pass_scales: None,
            moe_expert: None,
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
            pass_scales: None,
            moe_expert: None,
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
            pass_scales: None,
            moe_expert: None,
        }];
        assert!(apply_adapters_strict(&mut host2, &specs2, "test").is_err());
    }

    /// sc-2909: a diffusers/ComfyUI LoRA spelled with `lora_down`/`lora_up` factor suffixes on a
    /// **dotted, un-prefixed** path (the lightx2v Qwen-Image-Lightning format) routes through the
    /// PEFT loader (no `lora_unet_` prefix → not kohya) and installs the BYTE-IDENTICAL adapter to
    /// its `lora_A`/`lora_B` twin — and `apply_adapter_specs_autoprefix` resolves it end-to-end.
    #[test]
    fn diffusers_lora_down_up_equals_peft_ab() {
        let weight = Array::from_slice(
            &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &[4, 3],
        );
        let a_raw = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]); // [r, in]
        let b_raw = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]); // [out, r]
        let alpha = Array::from_slice(&[4.0f32], &[1]);

        // down==A, up==B, bare alpha, no namespace prefix — exactly the lightx2v Lightning spelling.
        let down_path = tmp("diffusers_down_up.safetensors");
        Array::save_safetensors(
            vec![
                ("lin.lora_down.weight", &a_raw),
                ("lin.lora_up.weight", &b_raw),
                ("lin.alpha", &alpha),
            ],
            None,
            &down_path,
        )
        .unwrap();
        // Detected as un-prefixed (not kohya, not BFL) and resolved through the strict seam.
        let w = Weights::from_file(&down_path).unwrap();
        assert!(!is_kohya(&w), "dotted-path lora_down is NOT kohya");
        assert_eq!(detect_lora_prefix(&w), None, "no namespace prefix");

        let mut via_down = OneLinear {
            lin: AdaptableLinear::dense(weight.clone(), None),
        };
        let report = apply_adapter_specs_autoprefix(
            &mut via_down,
            &[AdapterSpec::new(down_path, 0.5, AdapterKind::Lora)],
        )
        .unwrap();
        assert_eq!(report.applied, 1, "lora_down/up resolved to lin");
        assert!(report.unmatched_paths.is_empty());

        // The `lora_A`/`lora_B` twin must install the identical adapter.
        let ab_path = tmp("diffusers_ab_twin.safetensors");
        Array::save_safetensors(
            vec![
                ("lin.lora_A.weight", &a_raw),
                ("lin.lora_B.weight", &b_raw),
                ("lin.alpha", &alpha),
            ],
            None,
            &ab_path,
        )
        .unwrap();
        let mut via_ab = OneLinear {
            lin: AdaptableLinear::dense(weight, None),
        };
        apply_lora_peft(
            &mut via_ab,
            &Weights::from_file(&ab_path).unwrap(),
            0.5,
            None,
        )
        .unwrap();

        let pull = |h: &mut OneLinear| match h.adaptable_mut(&["lin"]).unwrap().adapters() {
            [Adapter::Lora { a, b, scale }] => (a.clone(), b.clone(), *scale),
            _ => panic!("expected one LoRA"),
        };
        let (da, db, ds) = pull(&mut via_down);
        let (pa, pb, ps) = pull(&mut via_ab);
        assert_eq!(ds, ps);
        assert!(
            array_eq(&da, &pa, false).unwrap().item::<bool>()
                && array_eq(&db, &pb, false).unwrap().item::<bool>(),
            "lora_down/up and lora_A/B installed different adapters"
        );
    }

    // ---- kohya `lora_unet_` LoRA (sc-2618) ----

    /// Two modules whose flattened kohya stems are ambiguous under a blind `_`→`.` split: the
    /// segment `to_out.0` and the segment name `feed_forward` both contain the separator char.
    fn kohya_two_module_host() -> MultiHost {
        let w_out = Array::from_slice(
            &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &[4, 3],
        );
        let w_ff = Array::from_slice(
            &(0..15).map(|i| i as f32 * 0.07).collect::<Vec<_>>(),
            &[5, 3],
        );
        MultiHost::new(&[
            ("blocks.0.attn.to_out.0", w_out),
            ("blocks.0.feed_forward.w1", w_ff),
        ])
    }

    /// The same (down, up, alpha) factors written in BOTH conventions and applied through the
    /// provider seam must yield byte-identical adapters — a kohya file is interchangeable with its
    /// PEFT twin. This is the sc-2618 gate at the core level (no model weights needed).
    #[test]
    fn kohya_equiv_to_peft_bit_exact() {
        // out=4/in=3 and out=5/in=3, rank=2; alpha=4 (≠ rank → exercises the alpha/rank fold).
        let a_out = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]); // [r,in]
        let b_out = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]); // [out,r]
        let a_ff = Array::from_slice(&[0.05f32, -0.15, 0.2, 0.3, -0.25, 0.1], &[2, 3]);
        let b_ff = Array::from_slice(
            &[0.2f32, -0.2, 0.1, 0.3, -0.1, 0.4, 0.15, -0.35, 0.05, 0.25],
            &[5, 2],
        );
        let alpha = Array::from_slice(&[4.0f32], &[1]);

        let kohya_path = tmp("equiv_kohya.safetensors");
        Array::save_safetensors(
            vec![
                ("lora_unet_blocks_0_attn_to_out_0.lora_down.weight", &a_out),
                ("lora_unet_blocks_0_attn_to_out_0.lora_up.weight", &b_out),
                ("lora_unet_blocks_0_attn_to_out_0.alpha", &alpha),
                ("lora_unet_blocks_0_feed_forward_w1.lora_down.weight", &a_ff),
                ("lora_unet_blocks_0_feed_forward_w1.lora_up.weight", &b_ff),
                ("lora_unet_blocks_0_feed_forward_w1.alpha", &alpha),
            ],
            None as Option<&HashMap<String, String>>,
            &kohya_path,
        )
        .unwrap();

        let peft_path = tmp("equiv_peft.safetensors");
        Array::save_safetensors(
            vec![
                ("transformer.blocks.0.attn.to_out.0.lora_A.weight", &a_out),
                ("transformer.blocks.0.attn.to_out.0.lora_B.weight", &b_out),
                ("transformer.blocks.0.attn.to_out.0.alpha", &alpha),
                ("transformer.blocks.0.feed_forward.w1.lora_A.weight", &a_ff),
                ("transformer.blocks.0.feed_forward.w1.lora_B.weight", &b_ff),
                ("transformer.blocks.0.feed_forward.w1.alpha", &alpha),
            ],
            None as Option<&HashMap<String, String>>,
            &peft_path,
        )
        .unwrap();

        let mut via_kohya = kohya_two_module_host();
        let rep_k = apply_adapters_strict(
            &mut via_kohya,
            &[AdapterSpec {
                path: kohya_path,
                scale: 0.75,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
            "test",
        )
        .unwrap();
        assert_eq!(rep_k.applied, 2, "both kohya modules resolve");

        let mut via_peft = kohya_two_module_host();
        apply_adapters_strict(
            &mut via_peft,
            &[AdapterSpec {
                path: peft_path,
                scale: 0.75,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
            "test",
        )
        .unwrap();

        let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
        for p in ["blocks.0.attn.to_out.0", "blocks.0.feed_forward.w1"] {
            let gk = via_kohya.mods.get(p).unwrap().forward(&x).unwrap();
            let gp = via_peft.mods.get(p).unwrap().forward(&x).unwrap();
            assert!(
                array_eq(&gk, &gp, false).unwrap().item::<bool>(),
                "kohya and peft adapters diverged at {p}"
            );
            // And both actually moved off the bare base.
            let base = AdaptableLinear::dense(
                via_kohya
                    .mods
                    .get(p)
                    .unwrap()
                    .dense_weight()
                    .unwrap()
                    .0
                    .clone(),
                None,
            )
            .forward(&x)
            .unwrap();
            assert!(
                !array_eq(&gk, &base, false).unwrap().item::<bool>(),
                "adapter at {p} was a no-op"
            );
        }
    }

    /// The flattened stem `blocks_0_feed_forward_w1` must resolve to `blocks.0.feed_forward.w1`
    /// (the table), NOT the blind split `blocks.0.feed.forward.w1` — proving the disambiguation does
    /// real work.
    #[test]
    fn kohya_table_disambiguates_underscore_segment_names() {
        let mut host = kohya_two_module_host();
        // The blind `_`→`.` split target does not exist; the correct dotted path does.
        assert!(host
            .adaptable_mut(&["blocks", "0", "feed", "forward", "w1"])
            .is_none());
        assert!(host
            .adaptable_mut(&["blocks", "0", "feed_forward", "w1"])
            .is_some());

        let table = kohya_table(&host.adaptable_paths());
        assert_eq!(
            table.get("blocks_0_feed_forward_w1").map(String::as_str),
            Some("blocks.0.feed_forward.w1")
        );
        assert_eq!(
            table.get("blocks_0_attn_to_out_0").map(String::as_str),
            Some("blocks.0.attn.to_out.0")
        );
    }

    /// A `lora_unet_` key whose stem is off-surface (e.g. FLUX.2 BFL `double_blocks_*`, sc-2743) is
    /// surfaced in `unmatched_paths` and fails the strict policy — loud, never silently dropped.
    #[test]
    fn kohya_offsurface_stem_surfaced_and_strict_errors() {
        let a = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]);
        let b = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
        let path = tmp("kohya_offsurface.safetensors");
        Array::save_safetensors(
            vec![
                (
                    "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight",
                    &a,
                ),
                ("lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight", &b),
            ],
            None as Option<&HashMap<String, String>>,
            &path,
        )
        .unwrap();

        let mut host = kohya_two_module_host();
        let table = kohya_table(&host.adaptable_paths());
        let report =
            apply_lora_kohya(&mut host, &Weights::from_file(&path).unwrap(), 1.0, &table).unwrap();
        assert_eq!(report.applied, 0);
        assert_eq!(
            report.unmatched_paths,
            vec!["double_blocks_0_img_attn_qkv".to_string()]
        );

        // Through the strict provider seam it is a hard error.
        let mut host2 = kohya_two_module_host();
        assert!(apply_adapters_strict(
            &mut host2,
            &[AdapterSpec {
                path,
                scale: 1.0,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
            "test",
        )
        .is_err());
    }

    /// A kohya adapter at `scale = 0` is a bit-exact no-op (the scale-0 invariant), and `is_kohya`
    /// detects the format.
    #[test]
    fn kohya_scale_zero_is_bit_exact_noop() {
        let a = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]);
        let b = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
        let path = tmp("kohya_scale0.safetensors");
        Array::save_safetensors(
            vec![
                ("lora_unet_blocks_0_attn_to_out_0.lora_down.weight", &a),
                ("lora_unet_blocks_0_attn_to_out_0.lora_up.weight", &b),
            ],
            None as Option<&HashMap<String, String>>,
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();
        assert!(is_kohya(&w));

        let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
        let mut host = kohya_two_module_host();
        let base = host
            .mods
            .get("blocks.0.attn.to_out.0")
            .unwrap()
            .forward(&x)
            .unwrap();
        let table = kohya_table(&host.adaptable_paths());
        apply_lora_kohya(&mut host, &w, 0.0, &table).unwrap();
        let out = host
            .mods
            .get("blocks.0.attn.to_out.0")
            .unwrap()
            .forward(&x)
            .unwrap();
        assert!(array_eq(&out, &base, false).unwrap().item::<bool>());
    }

    // ---- BFL / ComfyUI fused→split LoRA (sc-2743) ----

    /// The [`LoraRowSlice`] variants are byte-faithful to the fork's `LoraTransforms`. Expected values
    /// are pinned to the mflux venv (`LoraTransforms.split_*` on the same inputs, sc-2743): up always
    /// slices, down is shared unless the rank is divisible, and the qkv-mlp `dims` slice matches the
    /// `[3072,3072,3072,12288]` boundaries.
    #[test]
    fn lora_row_slice_matches_fork_transforms() {
        // arange(6,2): split_q_up = rows[0:2], split_v_up = rows[4:6].
        let t6 = Array::from_slice(&(0..12).map(|i| i as f32).collect::<Vec<_>>(), &[6, 2]);
        let q_up = LoraRowSlice::Chunk { n: 3, index: 0 }.apply(&t6).unwrap();
        assert_eq!(q_up.shape(), &[2, 2]);
        assert_eq!(q_up.as_slice::<f32>(), &[0.0, 1.0, 2.0, 3.0]);
        let v_up = LoraRowSlice::Chunk { n: 3, index: 2 }.apply(&t6).unwrap();
        assert_eq!(v_up.as_slice::<f32>(), &[8.0, 9.0, 10.0, 11.0]);

        // down: ChunkIfDivisible — whole when rank%3!=0 (the usual fused-qkv LoRA), sliced when ==0.
        let d4 = Array::from_slice(&(0..8).map(|i| i as f32).collect::<Vec<_>>(), &[4, 2]);
        let d4q = LoraRowSlice::ChunkIfDivisible { n: 3, index: 0 }
            .apply(&d4)
            .unwrap();
        assert_eq!(d4q.shape(), &[4, 2], "rank 4 not ÷3 → shared whole");
        assert_eq!(d4q.as_slice::<f32>(), d4.as_slice::<f32>());
        let d6q = LoraRowSlice::ChunkIfDivisible { n: 3, index: 0 }
            .apply(&t6)
            .unwrap();
        assert_eq!(
            d6q.as_slice::<f32>(),
            &[0.0, 1.0, 2.0, 3.0],
            "rank 6 ÷3 → sliced"
        );

        // qkv-mlp up `dims` (FLUX.1 `linear1`): q = rows[0:3072], mlp = rows[9216:21504].
        let dims = vec![3072, 3072, 3072, 12288];
        let total: i32 = dims.iter().sum();
        let big = Array::from_slice(
            &(0..total).map(|i| i as f32).collect::<Vec<_>>(),
            &[total, 1],
        );
        let q = LoraRowSlice::Dims {
            dims: dims.clone(),
            index: 0,
        }
        .apply(&big)
        .unwrap();
        assert_eq!(q.shape(), &[3072, 1]);
        assert_eq!(q.as_slice::<f32>()[0], 0.0);
        let mlp = LoraRowSlice::Dims {
            dims: dims.clone(),
            index: 3,
        }
        .apply(&big)
        .unwrap();
        assert_eq!(mlp.shape(), &[12288, 1]);
        assert_eq!(mlp.as_slice::<f32>()[0], 9216.0);
    }

    /// A host with three separate per-head linears at `blk.attn.to_{q,k,v}` (`[inner,in]` each).
    fn three_qkv_host(inner: i32, inp: i32) -> MultiHost {
        let zeros = || Array::from_slice(&vec![0.0f32; (inner * inp) as usize], &[inner, inp]);
        MultiHost::new(&[
            ("blk.attn.to_q", zeros()),
            ("blk.attn.to_k", zeros()),
            ("blk.attn.to_v", zeros()),
        ])
    }

    /// The sc-2743 gate at the core level: a BFL *fused* qkv LoRA, split via [`apply_lora_bfl`],
    /// installs the BYTE-IDENTICAL adapter at each of `to_q/to_k/to_v` as the equivalent *diffusers
    /// split-target* LoRA (the fork-verified PEFT path). The fused up `[3·inner, r]` is row-sliced into
    /// per-head `[inner, r]`; the down `[r, in]` (rank not ÷3) is shared. No model weights needed.
    #[test]
    fn bfl_fused_qkv_equals_diffusers_split() {
        let (inner, inp, r) = (4i32, 3i32, 2i32);
        // Per-head up factors, then the fused up = their dim-0 concat (row-major, so flat concat).
        let bq: Vec<f32> = (0..inner * r)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
            .collect();
        let bk: Vec<f32> = (0..inner * r)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.02)
            .collect();
        let bv: Vec<f32> = (0..inner * r)
            .map(|i| ((i % 3) as f32 - 1.0) * 0.03)
            .collect();
        let mut fused = Vec::new();
        fused.extend_from_slice(&bq);
        fused.extend_from_slice(&bk);
        fused.extend_from_slice(&bv);
        let b_fused = Array::from_slice(&fused, &[3 * inner, r]);
        let b_q = Array::from_slice(&bq, &[inner, r]);
        let b_k = Array::from_slice(&bk, &[inner, r]);
        let b_v = Array::from_slice(&bv, &[inner, r]);
        // Shared down [r, in] (rank 2 not ÷3 → shared across q/k/v) + alpha ≠ rank.
        let a = Array::from_slice(
            &(0..r * inp)
                .map(|i| ((i % 5) as f32 - 2.0) * 0.05)
                .collect::<Vec<_>>(),
            &[r, inp],
        );
        let alpha = Array::from_slice(&[4.0f32], &[1]);

        // BFL file: one fused qkv linear (kohya `lora_unet_` spelling).
        let up_key = "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight";
        let down_key = "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight";
        let alpha_key = "lora_unet_double_blocks_0_img_attn_qkv.alpha";
        let bfl_path = tmp("bfl_qkv.safetensors");
        Array::save_safetensors(
            vec![(up_key, &b_fused), (down_key, &a), (alpha_key, &alpha)],
            None as Option<&HashMap<String, String>>,
            &bfl_path,
        )
        .unwrap();
        let wb = Weights::from_file(&bfl_path).unwrap();

        let mk = |idx: i32, tgt: &str| BflTarget {
            target_path: tgt.to_string(),
            up_keys: vec![up_key.to_string()],
            down_keys: vec![down_key.to_string()],
            alpha_keys: vec![alpha_key.to_string()],
            up_slice: Some(LoraRowSlice::Chunk { n: 3, index: idx }),
            down_slice: Some(LoraRowSlice::ChunkIfDivisible { n: 3, index: idx }),
        };
        let targets = vec![
            mk(0, "blk.attn.to_q"),
            mk(1, "blk.attn.to_k"),
            mk(2, "blk.attn.to_v"),
        ];

        let mut host_bfl = three_qkv_host(inner, inp);
        let rep = apply_lora_bfl(&mut host_bfl, &wb, 0.7, &targets).unwrap();
        assert_eq!(rep.applied, 3, "all three split targets installed");
        assert!(rep.unmatched_paths.is_empty());

        // Equivalent diffusers split-target file: per-head up, SHARED down, same alpha.
        let peft_path = tmp("bfl_split_peft.safetensors");
        Array::save_safetensors(
            vec![
                ("transformer.blk.attn.to_q.lora_B.weight", &b_q),
                ("transformer.blk.attn.to_q.lora_A.weight", &a),
                ("transformer.blk.attn.to_q.alpha", &alpha),
                ("transformer.blk.attn.to_k.lora_B.weight", &b_k),
                ("transformer.blk.attn.to_k.lora_A.weight", &a),
                ("transformer.blk.attn.to_k.alpha", &alpha),
                ("transformer.blk.attn.to_v.lora_B.weight", &b_v),
                ("transformer.blk.attn.to_v.lora_A.weight", &a),
                ("transformer.blk.attn.to_v.alpha", &alpha),
            ],
            None as Option<&HashMap<String, String>>,
            &peft_path,
        )
        .unwrap();
        let wp = Weights::from_file(&peft_path).unwrap();
        let mut host_peft = three_qkv_host(inner, inp);
        apply_lora_peft(&mut host_peft, &wp, 0.7, Some("transformer.")).unwrap();

        for p in ["blk.attn.to_q", "blk.attn.to_k", "blk.attn.to_v"] {
            let pull = |h: &MultiHost| match h.mods.get(p).unwrap().adapters() {
                [Adapter::Lora { a, b, scale }] => (a.clone(), b.clone(), *scale),
                _ => panic!("expected one LoRA at {p}"),
            };
            let (ba, bb, bs) = pull(&host_bfl);
            let (pa, pb, ps) = pull(&host_peft);
            assert_eq!(bs, ps, "scale differs at {p}");
            assert!(
                array_eq(&ba, &pa, false).unwrap().item::<bool>()
                    && array_eq(&bb, &pb, false).unwrap().item::<bool>(),
                "BFL split and diffusers split installed different adapters at {p}"
            );
        }
    }

    /// `is_bfl` detects a BFL file; an off-surface adapter-shaped key is surfaced (not dropped) while a
    /// bundled text-encoder key is ignored; and a scale-0 BFL adapter is a bit-exact no-op.
    #[test]
    fn bfl_detection_unmatched_and_scale_zero() {
        let up = Array::from_slice(
            &(0..8).map(|i| i as f32 * 0.01).collect::<Vec<_>>(),
            &[4, 2],
        );
        let down = Array::from_slice(&[0.1f32, 0.2, -0.1, -0.2, 0.3, -0.3], &[2, 3]);
        let targets = vec![BflTarget {
            target_path: "blk.attn.to_out".to_string(),
            up_keys: vec!["lora_unet_double_blocks_0_img_attn_proj.lora_up.weight".to_string()],
            down_keys: vec!["lora_unet_double_blocks_0_img_attn_proj.lora_down.weight".to_string()],
            alpha_keys: vec![],
            up_slice: None,
            down_slice: None,
        }];

        let path = tmp("bfl_detect.safetensors");
        Array::save_safetensors(
            vec![
                (
                    "lora_unet_double_blocks_0_img_attn_proj.lora_up.weight",
                    &up,
                ),
                (
                    "lora_unet_double_blocks_0_img_attn_proj.lora_down.weight",
                    &down,
                ),
                // off-surface BFL key (no target) → surfaced, not silently dropped.
                (
                    "lora_unet_double_blocks_9_img_attn_proj.lora_up.weight",
                    &up,
                ),
                (
                    "lora_unet_double_blocks_9_img_attn_proj.lora_down.weight",
                    &down,
                ),
                // bundled text-encoder key → ignored (out of denoiser namespace).
                ("lora_te_text_model_layer_0.lora_up.weight", &up),
            ],
            None as Option<&HashMap<String, String>>,
            &path,
        )
        .unwrap();
        let w = Weights::from_file(&path).unwrap();
        assert!(is_bfl(&w, &targets), "a BFL source key marks the file BFL");

        let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
        let mut host = MultiHost::new(&[(
            "blk.attn.to_out",
            Array::from_slice(
                &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
                &[4, 3],
            ),
        )]);
        let base = host
            .mods
            .get("blk.attn.to_out")
            .unwrap()
            .forward(&x)
            .unwrap();

        // scale 0 → bit-exact no-op; the off-surface block-9 key is surfaced, the TE key ignored.
        let rep = apply_lora_bfl(&mut host, &w, 0.0, &targets).unwrap();
        assert_eq!(rep.applied, 1, "the on-surface proj target installed");
        // Both block-9 keys (up + down) are surfaced (sorted: down < up); the `lora_te_` key ignored.
        assert_eq!(
            rep.unmatched_paths,
            vec![
                "lora_unet_double_blocks_9_img_attn_proj.lora_down.weight".to_string(),
                "lora_unet_double_blocks_9_img_attn_proj.lora_up.weight".to_string(),
            ],
            "off-surface BFL keys surfaced; TE key ignored"
        );
        let out = host
            .mods
            .get("blk.attn.to_out")
            .unwrap()
            .forward(&x)
            .unwrap();
        assert!(
            array_eq(&out, &base, false).unwrap().item::<bool>(),
            "scale-0 BFL adapter must be a bit-exact no-op"
        );
    }

    /// sc-3642: a third-party (non-peft / lycoris) LoKr reconstructs the SAME per-module delta the
    /// `lycoris` library produces. Fixtures (real lycoris adapters + ground-truth deltas) come from
    /// `tools/sc3642_lokr_reference.py` via `~/mlx-flux-venv` — the on-device A/B. Covers the four
    /// shapes: full-w1 + decomposed-w2, both-decomposed, both-full (scale forced to 1), and conv
    /// `lokr_t2` tucker.
    #[test]
    fn thirdparty_lokr_matches_lycoris_reference() {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sc3642_lokr");
        for name in [
            "linear_w1full_w2lr",
            "linear_bothlr",
            "linear_bothfull",
            "conv_tucker",
        ] {
            let w = Weights::from_file(dir.join(format!("{name}.safetensors"))).unwrap();
            let exp = Weights::from_file(dir.join(format!("{name}.expected.safetensors"))).unwrap();
            assert!(is_lokr_keys(&w), "{name}: not detected as LoKr by keys");
            assert!(
                !is_lokr(&w),
                "{name}: a third-party file has no networkType metadata"
            );

            // Reconstruct the flattened→dotted table from the expected (dotted) module paths.
            let table: BTreeMap<String, String> = exp
                .keys()
                .map(|d| (d.replace('.', "_"), d.to_string()))
                .collect();
            let groups = parse_lokr_thirdparty(&w).unwrap();
            assert!(!groups.is_empty(), "{name}: parsed no LoKr modules");
            for (raw, g) in &groups {
                let dotted = resolve_lokr_path(raw, &table)
                    .unwrap_or_else(|| panic!("{name}: cannot resolve raw key {raw:?}"));
                let want = exp.require(dotted).unwrap();
                // Reconstruct in f32 (lycoris computes f32) and compare to the ground truth.
                let got = g.delta(want.shape(), Dtype::Float32).unwrap();
                assert_eq!(
                    got.shape(),
                    want.shape(),
                    "{name}/{dotted}: reconstructed delta shape mismatch"
                );
                assert!(
                    all_close(&got, want, 1e-4, 1e-5, false)
                        .unwrap()
                        .item::<bool>(),
                    "{name}/{dotted}: reconstructed LoKr delta diverged from the lycoris reference"
                );
            }
        }
    }

    /// sc-3642: the third-party LoKr installs through the autoprefix dispatch even when the caller
    /// labels the spec `AdapterKind::Lora` (a third-party file carries no `networkType` to tell the
    /// caller otherwise) — detection-by-keys routes it to `apply_lokr_thirdparty`, resolving the
    /// `lycoris_`-prefixed flattened key to the host's dotted module.
    #[test]
    fn thirdparty_lokr_routes_and_installs_via_autoprefix() {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sc3642_lokr");
        let exp = Weights::from_file(dir.join("linear_w1full_w2lr.expected.safetensors")).unwrap();
        // Host with the one Linear the fixture targets (dotted path "proj"), sized to the delta.
        let delta_shape = exp.require("proj").unwrap().shape().to_vec();
        let base = Array::zeros::<f32>(&delta_shape).unwrap();
        let mut host = MultiHost::new(&[("proj", base)]);
        let spec = AdapterSpec::new(
            dir.join("linear_w1full_w2lr.safetensors"),
            1.0,
            AdapterKind::Lora, // deliberately mislabeled — detection must override
        );
        let report =
            apply_adapter_specs_autoprefix(&mut host, std::slice::from_ref(&spec)).unwrap();
        assert_eq!(report.applied, 1, "third-party LoKr was not installed");
        assert!(
            report.unmatched_paths.is_empty(),
            "unexpected unmatched: {:?}",
            report.unmatched_paths
        );
    }

    /// sc-3643: a third-party (non-peft / lycoris) LoHa reconstructs the SAME per-module delta the
    /// `lycoris` library produces. Fixtures from `tools/sc3643_loha_reference.py` via `~/mlx-flux-venv`.
    /// Covers linear, conv (kernel folded into the factors), and conv `hada_t1/t2` tucker.
    #[test]
    fn thirdparty_loha_matches_lycoris_reference() {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sc3643_loha");
        for name in ["linear", "conv_notucker", "conv_tucker"] {
            let w = Weights::from_file(dir.join(format!("{name}.safetensors"))).unwrap();
            let exp = Weights::from_file(dir.join(format!("{name}.expected.safetensors"))).unwrap();
            assert!(is_loha_keys(&w), "{name}: not detected as LoHa by keys");
            assert!(!is_lokr_keys(&w), "{name}: must not look like LoKr");
            assert!(
                !is_lokr(&w),
                "{name}: a third-party file has no networkType metadata"
            );

            let table: BTreeMap<String, String> = exp
                .keys()
                .map(|d| (d.replace('.', "_"), d.to_string()))
                .collect();
            let groups = parse_loha_thirdparty(&w).unwrap();
            assert!(!groups.is_empty(), "{name}: parsed no LoHa modules");
            for (raw, g) in &groups {
                let dotted = resolve_lokr_path(raw, &table)
                    .unwrap_or_else(|| panic!("{name}: cannot resolve raw key {raw:?}"));
                let want = exp.require(dotted).unwrap();
                let got = g.delta(want.shape(), Dtype::Float32).unwrap();
                assert_eq!(
                    got.shape(),
                    want.shape(),
                    "{name}/{dotted}: reconstructed delta shape mismatch"
                );
                assert!(
                    all_close(&got, want, 1e-4, 1e-5, false)
                        .unwrap()
                        .item::<bool>(),
                    "{name}/{dotted}: reconstructed LoHa delta diverged from the lycoris reference"
                );
            }
        }
    }

    /// sc-3643: a third-party LoHa installs through the autoprefix dispatch even when the caller
    /// labels the spec `AdapterKind::Lora` — detection-by-keys routes it to `apply_loha_thirdparty`.
    #[test]
    fn thirdparty_loha_routes_and_installs_via_autoprefix() {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sc3643_loha");
        let exp = Weights::from_file(dir.join("linear.expected.safetensors")).unwrap();
        let delta_shape = exp.require("proj").unwrap().shape().to_vec();
        let base = Array::zeros::<f32>(&delta_shape).unwrap();
        let mut host = MultiHost::new(&[("proj", base)]);
        let spec = AdapterSpec::new(dir.join("linear.safetensors"), 1.0, AdapterKind::Lora);
        let report =
            apply_adapter_specs_autoprefix(&mut host, std::slice::from_ref(&spec)).unwrap();
        assert_eq!(report.applied, 1, "third-party LoHa was not installed");
        assert!(
            report.unmatched_paths.is_empty(),
            "unexpected unmatched: {:?}",
            report.unmatched_paths
        );
    }
}
