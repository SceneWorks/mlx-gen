//! Backend-neutral **weights-metadata** layer (sc-3722): a safetensors header/byte-view reader plus
//! the LoRA / LoKr / LoHa / kohya **string + metadata parsing** that decides *what* an adapter file
//! is and *where* each factor binds — all with zero tensor deps.
//!
//! Two halves:
//! 1. [`CheckpointMeta`] — opens one `.safetensors` file (or a sharded dir) via the neutral
//!    `safetensors` crate and exposes keys, dtypes, shapes, the `__metadata__` map, and raw byte
//!    views, **without materializing tensors**. Candle reads torch safetensors straight through this;
//!    mlx-gen keeps its mlx-rs full-checkpoint loader and uses this for adapter/metadata inspection.
//! 2. The format predicates / factor-suffix tables / rank-alpha parsing / key-alias resolution that
//!    were inline in mlx-gen's `adapters/loader.rs`. The *factor-reconstruction math* (`kron`,
//!    `matmul`) stays in mlx-gen; only the string/metadata logic lives here so a candle adapter
//!    loader reuses it verbatim.
//!
//! Reference: PEFT (`networkType=lokr`, `rank`/`alpha` metadata, `‹path›.lokr_*` factors), LyCORIS
//! third-party LoKr/LoHa (`lokr_*`/`hada_*` factors, optional per-module `.alpha`), and kohya
//! (`lora_unet_<flattened path>.lora_down/up.weight` + `.alpha`).

use std::collections::BTreeMap;
use std::path::Path;

pub use safetensors::Dtype;
use safetensors::SafeTensors;

use crate::{Error, Result};

// =================================================================================================
// CheckpointMeta — neutral safetensors header / byte-view reader.
// =================================================================================================

/// One tensor's neutral description: dtype, shape, and a borrowed view of its raw little-endian bytes
/// (row-major, exactly as stored). The backend lifts these bytes into its own array type.
#[derive(Clone, Copy)]
pub struct TensorView<'a> {
    pub dtype: Dtype,
    pub shape: &'a [usize],
    pub data: &'a [u8],
}

struct TensorLoc {
    shard: usize,
    dtype: Dtype,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// A safetensors checkpoint's **metadata** — keys, dtypes, shapes, byte ranges, and the file's
/// `__metadata__` map — without allocating any tensor. Backed by the owned file buffer(s), so byte
/// views borrow from `self`.
pub struct CheckpointMeta {
    buffers: Vec<Vec<u8>>,
    index: BTreeMap<String, TensorLoc>,
    file_metadata: BTreeMap<String, String>,
}

impl CheckpointMeta {
    /// Open one `.safetensors` file, reading its header (and the whole file buffer) but not parsing
    /// tensors into a tensor library.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let mut me = Self {
            buffers: Vec::new(),
            index: BTreeMap::new(),
            file_metadata: BTreeMap::new(),
        };
        me.add_file(path.as_ref())?;
        Ok(me)
    }

    /// Open and merge every `.safetensors` file under `dir` (sharded checkpoints). Keys are unioned;
    /// on a duplicate key the later file (sorted by path) wins — the same merge semantics as
    /// mlx-gen's `Weights::from_dir`.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let mut files: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(Error::Msg(format!(
                "no .safetensors files in {}",
                dir.display()
            )));
        }
        let mut me = Self {
            buffers: Vec::new(),
            index: BTreeMap::new(),
            file_metadata: BTreeMap::new(),
        };
        for f in files {
            me.add_file(&f)?;
        }
        Ok(me)
    }

    fn add_file(&mut self, path: &Path) -> Result<()> {
        // Reads the WHOLE file into `self.buffers` and keeps it — `TensorLoc` indexes into the buffer
        // for lazy random-access tensor reads. This is intended for adapter-sized checkpoints; calling
        // `from_dir` on a multi-GB *base-model* dir would hold every file resident (host RAM). An mmap
        // backing (page on demand) is the efficiency follow-up if base-model use is ever needed (F-038).
        let buf = std::fs::read(path)?;
        // `read_metadata` returns (header_json_len, Metadata); the data region begins at 8 + n and
        // each tensor's data_offsets are relative to it.
        let (n, meta) = SafeTensors::read_metadata(&buf)
            .map_err(|e| Error::Msg(format!("safetensors header in {}: {e}", path.display())))?;
        let data_base = 8 + n;
        let shard = self.buffers.len();
        for (key, info) in meta.tensors() {
            self.index.insert(
                key,
                TensorLoc {
                    shard,
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                    start: data_base + info.data_offsets.0,
                    end: data_base + info.data_offsets.1,
                },
            );
        }
        if let Some(kv) = meta.metadata() {
            for (k, v) in kv {
                self.file_metadata.insert(k.clone(), v.clone());
            }
        }
        self.buffers.push(buf);
        Ok(())
    }

    /// Tensor keys, sorted.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(String::as_str)
    }

    /// `true` if `key` is present.
    pub fn contains(&self, key: &str) -> bool {
        self.index.contains_key(key)
    }

    /// A `__metadata__` value (e.g. `networkType`, `rank`, `alpha`), if present.
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.file_metadata.get(key).map(String::as_str)
    }

    /// A tensor's dtype/shape/raw byte view, or `None` if the key is absent.
    pub fn tensor(&self, key: &str) -> Option<TensorView<'_>> {
        self.index.get(key).map(|loc| TensorView {
            dtype: loc.dtype,
            shape: &loc.shape,
            data: &self.buffers[loc.shard][loc.start..loc.end],
        })
    }
}

// =================================================================================================
// LoRA / LoKr / LoHa / kohya format parsing (string + metadata only).
// =================================================================================================

/// PEFT LoKr per-module factor suffixes; each factor is full (`lokr_w1`/`lokr_w2`) or low-rank
/// (`_a`/`_b`). `.lokr_w1_a`/`_b` precede the bare `.lokr_w1` so exact-suffix matching never mis-binds.
pub const LOKR_SUFFIXES: [&str; 6] = [
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w1",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_w2",
];

/// Third-party LyCORIS LoKr factor suffixes — the PEFT set plus `lokr_t2` (the tucker/CP factor).
pub const LOKR_TP_SUFFIXES: [&str; 7] = [
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w1",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_w2",
    ".lokr_t2",
];

/// Third-party LyCORIS LoHa factor suffixes — two low-rank Hadamard pairs + optional tucker `t1`/`t2`.
pub const LOHA_TP_SUFFIXES: [&str; 6] = [
    ".hada_w1_a",
    ".hada_w1_b",
    ".hada_w2_a",
    ".hada_w2_b",
    ".hada_t1",
    ".hada_t2",
];

/// The kohya flattened-path namespace prefix (`lora_unet_<dotted-path-with-dots→underscores>`).
pub const KOHYA_PREFIX: &str = "lora_unet_";

/// Common LoRA namespace prefixes a PEFT/diffusers file may carry on its keys (LoKr keys are bare).
pub const COMMON_LORA_PREFIXES: [&str; 2] = ["transformer.", "diffusion_model."];

/// `true` if the file's `networkType` metadata marks it a (PEFT) LoKr adapter.
pub fn is_lokr_network_type(network_type: Option<&str>) -> bool {
    network_type
        .map(|s| s.trim().eq_ignore_ascii_case("lokr"))
        .unwrap_or(false)
}

/// `true` if any key is a LoKr factor (`*.lokr_w…`), regardless of `networkType` metadata — how a
/// **third-party** LyCORIS LoKr is recognized (those files ship the factors but not the PEFT stamp).
pub fn keys_contain_lokr<'a>(mut keys: impl Iterator<Item = &'a str>) -> bool {
    keys.any(|k| k.contains(".lokr_w"))
}

/// `true` if any key is a LoHa factor (`*.hada_w…`). Mutually exclusive with [`keys_contain_lokr`].
pub fn keys_contain_loha<'a>(mut keys: impl Iterator<Item = &'a str>) -> bool {
    keys.any(|k| k.contains(".hada_w"))
}

/// `true` if any key carries the kohya `lora_unet_` prefix (the only convention that flattens the
/// module path; PEFT/diffusers keep dots, LoKr is bare).
pub fn keys_are_kohya<'a>(mut keys: impl Iterator<Item = &'a str>) -> bool {
    keys.any(|k| k.starts_with(KOHYA_PREFIX))
}

/// The [`COMMON_LORA_PREFIXES`] namespace present in `keys`, if any.
pub fn detect_lora_prefix<'a>(keys: impl IntoIterator<Item = &'a str>) -> Option<&'static str> {
    let keys: Vec<&str> = keys.into_iter().collect();
    COMMON_LORA_PREFIXES
        .into_iter()
        .find(|&p| keys.iter().any(|k| k.starts_with(p)))
}

/// Parse the PEFT `(rank, alpha)` from safetensors metadata. `rank` defaults to `1.0`; `alpha`
/// defaults to `rank` (scale 1.0), matching PEFT.
pub fn parse_rank_alpha(rank: Option<&str>, alpha: Option<&str>) -> (f32, f32) {
    // Treat a parsed rank <= 0 the same as absent (→ 1.0): a zero rank would make the downstream
    // `alpha/rank` scale non-finite and NaN-poison the adapter merge (sc-5252/F-002).
    let rank = rank
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|&r| r > 0.0)
        .unwrap_or(1.0);
    let alpha = alpha.and_then(|s| s.parse::<f32>().ok()).unwrap_or(rank);
    (rank, alpha)
}

/// The safetensors `__metadata__` key under which PEFT / diffusers `save_lora_adapter` store the
/// LoRA config blob (sc-5513). Callers pass `meta(LORA_ADAPTER_METADATA_KEY)` to [`LoraAdapterMeta`].
pub const LORA_ADAPTER_METADATA_KEY: &str = "lora_adapter_metadata";

/// Parsed view of the PEFT / diffusers `lora_adapter_metadata` config blob (sc-5513 — the MLX sibling
/// of candle's sc-5374 `LoraAdapterMeta`).
///
/// `peft.save_pretrained()` and diffusers `save_lora_adapter` do **not** write a per-target `.alpha`
/// tensor — the kohya / SceneWorks-trainer convention every inference adapter loader reads first. They
/// store the LoRA scaling inside the safetensors header `__metadata__["lora_adapter_metadata"]`: a JSON
/// blob carrying `lora_alpha`, `r`, and the optional per-module `alpha_pattern` / `rank_pattern`
/// overrides. With no `.alpha` tensor the loaders would otherwise fall back to `alpha = rank` (scale
/// 1.0) and apply such a file at the WRONG strength whenever `lora_alpha ≠ r` (the common `alpha = 2r`
/// / `r/2` / fixed-16 cases). Parsing this blob lets the merge recover the true `(alpha/rank)` scaling.
///
/// Scope is the **LoRA** path: LoKr carries its `rank`/`alpha` as top-level `__metadata__` strings the
/// LoKr loaders already read via [`parse_rank_alpha`], so it is unaffected. The MLX/SceneWorks trainer's
/// own PEFT output writes a per-target `.alpha` tensor (and top-level `rank`/`alpha`, not this blob), so
/// its round-trip is also unaffected — this is purely external/community adapter coverage (same flavor
/// as sc-3671 / the candle sc-5225 / sc-5374 lineage).
#[derive(Debug, Default, Clone)]
pub struct LoraAdapterMeta {
    lora_alpha: Option<f32>,
    r: Option<f32>,
    alpha_pattern: BTreeMap<String, f32>,
    rank_pattern: BTreeMap<String, f32>,
}

impl LoraAdapterMeta {
    /// Parse the `lora_adapter_metadata` JSON `blob` (the value of [`LORA_ADAPTER_METADATA_KEY`] in a
    /// safetensors `__metadata__` map). Returns `None` when the blob is absent (`None` — kohya /
    /// trainer files, which carry a per-target `.alpha` tensor instead) or unparseable — treated as
    /// absent so a malformed blob can never poison the merge (the caller keeps today's `alpha = rank`
    /// default).
    ///
    /// A non-positive `r` / `rank_pattern` value is dropped (treated as absent): it is the scaling
    /// denominator, and a `≤ 0` rank would make `alpha/rank` non-finite and NaN-poison the residual —
    /// same guard as [`parse_rank_alpha`] (sc-5252/F-002). A `0` *alpha* is kept (a legitimate scale-0,
    /// disabled-adapter value).
    pub fn from_metadata(blob: Option<&str>) -> Option<Self> {
        let v: serde_json::Value = serde_json::from_str(blob?).ok()?;
        let num = |x: &serde_json::Value| x.as_f64().map(|f| f as f32);
        let pos = |x: &serde_json::Value| x.as_f64().map(|f| f as f32).filter(|&f| f > 0.0);
        // A `{module: number}` JSON object → `BTreeMap`, applying `keep` per value and skipping any
        // non-numeric / filtered-out entry.
        let pattern = |x: Option<&serde_json::Value>,
                       keep: &dyn Fn(&serde_json::Value) -> Option<f32>|
         -> BTreeMap<String, f32> {
            x.and_then(|v| v.as_object())
                .map(|o| {
                    o.iter()
                        .filter_map(|(k, val)| keep(val).map(|f| (k.clone(), f)))
                        .collect()
                })
                .unwrap_or_default()
        };
        Some(Self {
            lora_alpha: v.get("lora_alpha").and_then(num),
            r: v.get("r").and_then(pos),
            alpha_pattern: pattern(v.get("alpha_pattern"), &num),
            rank_pattern: pattern(v.get("rank_pattern"), &pos),
        })
    }

    /// PEFT per-module override resolution. A single `target_name_key` is the first pattern key (across
    /// `rank_pattern` ∪ `alpha_pattern`) that equals `module_path` or that `module_path` ends with as
    /// `.{key}` — PEFT's `re.match(r".*\.{key}$")` in `LoraModel._create_and_replace`. Deterministic on
    /// overlap (`BTreeMap` key order). `None` ⇒ no per-module override, use the globals.
    fn target_name_key(&self, module_path: &str) -> Option<&str> {
        self.rank_pattern
            .keys()
            .chain(self.alpha_pattern.keys())
            .map(String::as_str)
            .find(|&k| module_path == k || module_path.ends_with(&format!(".{k}")))
    }

    /// The effective `(alpha, rank)` for `module_path`: `alpha_pattern[key] → lora_alpha` and
    /// `rank_pattern[key] → r`, each `None` when unset. The caller uses `alpha` (falling back to the
    /// factor rank) as the numerator; `rank` is honored as the scaling denominator (a well-formed PEFT
    /// file stores `A` as `[r, in]`, so the factor's leading dim already equals this — the metadata
    /// value is used for faithfulness and as the source of truth PEFT itself scales by). Any returned
    /// `rank` is `> 0` ([`from_metadata`] drops non-positive ranks).
    pub fn effective(&self, module_path: &str) -> (Option<f32>, Option<f32>) {
        let key = self.target_name_key(module_path);
        let alpha = key
            .and_then(|k| self.alpha_pattern.get(k))
            .copied()
            .or(self.lora_alpha);
        let rank = key
            .and_then(|k| self.rank_pattern.get(k))
            .copied()
            .or(self.r);
        (alpha, rank)
    }
}

/// Split a factor key into `(module_path, factor_name)` using `suffixes` (exact-suffix match, in
/// order — list `_a`/`_b` before the bare factor). `factor_name` has the leading `.` dropped (e.g.
/// `blk.0.lokr_w1_a` → `("blk.0", "lokr_w1_a")`). `None` if no suffix matches.
pub fn split_factor_key<'a>(key: &'a str, suffixes: &[&str]) -> Option<(&'a str, &'a str)> {
    for suffix in suffixes {
        if let Some(path) = key.strip_suffix(suffix) {
            // Slice the factor name out of `key` (drop the leading '.') so both halves borrow `key`.
            return Some((path, &key[path.len() + 1..]));
        }
    }
    None
}

/// Resolve a third-party flattened module key to a host dotted path. The key is `<PREFIX>_<stem>`
/// where `stem` is the diffusers path with dots flattened to underscores and `PREFIX` varies by
/// trainer (`lora_unet`, `lycoris`, …). Matched prefix-agnostically: `stem` (a `flattened → dotted`
/// table entry) must equal `raw` or be an `_`-delimited suffix of it; the longest such stem wins.
pub fn resolve_lokr_path<'a>(raw: &str, table: &'a BTreeMap<String, String>) -> Option<&'a str> {
    let mut best: Option<(&str, usize)> = None;
    for (stem, dotted) in table {
        let is_match = raw == stem
            || (raw.len() > stem.len()
                && raw.ends_with(stem.as_str())
                && raw.as_bytes()[raw.len() - stem.len() - 1] == b'_');
        let longer = match best {
            None => true,
            Some((_, l)) => stem.len() > l,
        };
        if is_match && longer {
            best = Some((dotted.as_str(), stem.len()));
        }
    }
    best.map(|(d, _)| d)
}

/// Build the kohya `flattened-stem → dotted-path` lookup from a host's routable target paths. The
/// stem is the dotted path with `.`→`_` (the kohya flattening), WITHOUT the `lora_unet_` prefix.
pub fn kohya_table(paths: &[String]) -> BTreeMap<String, String> {
    paths
        .iter()
        .map(|p| (p.replace('.', "_"), p.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView as StTensorView;

    #[test]
    fn lokr_network_type_predicate() {
        assert!(is_lokr_network_type(Some("lokr")));
        assert!(is_lokr_network_type(Some("  LoKr ")));
        assert!(!is_lokr_network_type(Some("loha")));
        assert!(!is_lokr_network_type(None));
    }

    #[test]
    fn rank_alpha_defaults() {
        assert_eq!(parse_rank_alpha(Some("16"), Some("8")), (16.0, 8.0));
        // alpha defaults to rank (scale 1.0).
        assert_eq!(parse_rank_alpha(Some("16"), None), (16.0, 16.0));
        // rank defaults to 1.0.
        assert_eq!(parse_rank_alpha(None, None), (1.0, 1.0));
        // A parsed rank <= 0 is treated as absent (→ 1.0) so the downstream alpha/rank scale stays
        // finite rather than NaN-poisoning the merge (sc-5252/F-002). alpha then defaults to 1.0.
        assert_eq!(parse_rank_alpha(Some("0"), None), (1.0, 1.0));
        assert_eq!(parse_rank_alpha(Some("0"), Some("8")), (1.0, 8.0));
        assert_eq!(parse_rank_alpha(Some("-4"), None), (1.0, 1.0));
    }

    /// sc-5513: a diffusers / PEFT `lora_adapter_metadata` blob with `lora_alpha ≠ r` parses to the
    /// global pair, so the loaders can recover the true `(alpha/rank)` scaling (the story example:
    /// `lora_alpha = 16`, `r = 8` ⇒ scale 2.0). The pair applies to any module with no override.
    #[test]
    fn lora_adapter_meta_parses_global_alpha_rank() {
        let blob = r#"{"lora_alpha": 16, "r": 8, "target_modules": ["to_q", "to_v"]}"#;
        let cfg = LoraAdapterMeta::from_metadata(Some(blob)).expect("blob must parse");
        assert_eq!(
            cfg.effective("transformer_blocks.0.attn.to_q"),
            (Some(16.0), Some(8.0))
        );
    }

    /// sc-5513: PEFT `alpha_pattern` / `rank_pattern` override the globals for a module whose dotted
    /// path ends with the pattern key (`re.match(r".*\.{key}$")`); a non-matching module keeps the
    /// globals. Mirrors PEFT's single-`target_name_key` resolution.
    #[test]
    fn lora_adapter_meta_honors_per_module_patterns() {
        let blob = r#"{"lora_alpha": 8, "r": 8,
                       "alpha_pattern": {"to_q": 32},
                       "rank_pattern": {"to_q": 16}}"#;
        let cfg = LoraAdapterMeta::from_metadata(Some(blob)).unwrap();
        // `…attn.to_q` ends with `.to_q` → the override pair.
        assert_eq!(
            cfg.effective("transformer_blocks.0.attn.to_q"),
            (Some(32.0), Some(16.0))
        );
        // `…attn.to_k` matches no pattern → the globals.
        assert_eq!(
            cfg.effective("transformer_blocks.0.attn.to_k"),
            (Some(8.0), Some(8.0))
        );
    }

    /// sc-5513: an absent blob (`None` — kohya / trainer files) and a malformed blob both yield `None`,
    /// so the caller falls back to today's per-target-`.alpha`-or-`rank` behavior rather than erroring.
    #[test]
    fn lora_adapter_meta_absent_or_malformed_is_none() {
        assert!(LoraAdapterMeta::from_metadata(None).is_none());
        assert!(LoraAdapterMeta::from_metadata(Some("{not valid json")).is_none());
    }

    /// sc-5513: a non-positive `r` / `rank_pattern` value is dropped (treated as absent) so the scaling
    /// denominator stays `> 0` and `alpha/rank` can never NaN-poison the merge (sc-5252/F-002); the
    /// caller then falls back to the factor's leading dim. A `0` *alpha* is a legitimate scale-0 value
    /// and is kept.
    #[test]
    fn lora_adapter_meta_drops_nonpositive_rank() {
        let cfg = LoraAdapterMeta::from_metadata(Some(
            r#"{"lora_alpha": 16, "r": 0, "rank_pattern": {"to_q": -4}}"#,
        ))
        .unwrap();
        // r = 0 dropped → rank None (caller uses the factor leading dim); alpha still recovered.
        assert_eq!(cfg.effective("blocks.0.to_k"), (Some(16.0), None));
        // rank_pattern[to_q] = -4 dropped → falls back to the (absent) global r → None.
        assert_eq!(cfg.effective("blocks.0.to_q"), (Some(16.0), None));
        // A scale-0 (disabled) adapter alpha is preserved.
        let z = LoraAdapterMeta::from_metadata(Some(r#"{"lora_alpha": 0, "r": 8}"#)).unwrap();
        assert_eq!(z.effective("blocks.0.to_q"), (Some(0.0), Some(8.0)));
    }

    #[test]
    fn key_predicates() {
        let lokr = ["blk.0.lokr_w1_a", "blk.0.lokr_w1_b"];
        let loha = ["blk.0.hada_w1_a"];
        let kohya = ["lora_unet_down_blocks_0.lora_down.weight"];
        assert!(keys_contain_lokr(lokr.iter().copied()));
        assert!(!keys_contain_loha(lokr.iter().copied()));
        assert!(keys_contain_loha(loha.iter().copied()));
        assert!(keys_are_kohya(kohya.iter().copied()));
        assert_eq!(
            detect_lora_prefix(["transformer.blk.0.attn"].into_iter()),
            Some("transformer.")
        );
        assert_eq!(detect_lora_prefix(["bare.key"].into_iter()), None);
    }

    #[test]
    fn factor_key_split() {
        // `_a`/`_b` precede the bare factor → never mis-binds.
        assert_eq!(
            split_factor_key("a.b.lokr_w1_a", &LOKR_SUFFIXES),
            Some(("a.b", "lokr_w1_a"))
        );
        assert_eq!(
            split_factor_key("a.b.lokr_w2", &LOKR_SUFFIXES),
            Some(("a.b", "lokr_w2"))
        );
        assert_eq!(split_factor_key("a.b.weight", &LOKR_SUFFIXES), None);
    }

    #[test]
    fn lokr_path_resolution_longest_stem_wins() {
        let mut table = BTreeMap::new();
        table.insert("blocks_0_attn".to_string(), "blocks.0.attn".to_string());
        table.insert("attn".to_string(), "attn".to_string());
        // `<PREFIX>_blocks_0_attn` matches the longer stem, not the short `attn` suffix.
        assert_eq!(
            resolve_lokr_path("lora_unet_blocks_0_attn", &table),
            Some("blocks.0.attn")
        );
        assert_eq!(resolve_lokr_path("lycoris_attn", &table), Some("attn"));
        assert_eq!(resolve_lokr_path("lora_unet_unknown", &table), None);
    }

    #[test]
    fn checkpoint_meta_reads_keys_dtype_shape_and_bytes() {
        // Serialize a tiny safetensors file, reopen it through CheckpointMeta, and assert the header
        // view + byte slice round-trip without a tensor library.
        let data: Vec<u8> = (0u8..16).collect(); // 4×i32 = 16 bytes
        let tv = StTensorView::new(Dtype::I32, vec![2, 2], &data).unwrap();
        let bytes = safetensors::serialize([("blk.weight", tv)], &None).unwrap();

        let dir = std::env::temp_dir().join(format!("gencore_meta_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("w.safetensors");
        std::fs::write(&path, &bytes).unwrap();

        let meta = CheckpointMeta::from_file(&path).unwrap();
        assert_eq!(meta.keys().collect::<Vec<_>>(), vec!["blk.weight"]);
        let t = meta.tensor("blk.weight").unwrap();
        assert_eq!(t.dtype, Dtype::I32);
        assert_eq!(t.shape, &[2, 2]);
        assert_eq!(t.data, &data[..]);
        assert!(meta.tensor("missing").is_none());

        std::fs::remove_file(&path).ok();
    }
}
