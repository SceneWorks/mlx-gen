//! Weight loading — safetensors → MLX arrays by dotted key, plus file metadata, with a
//! dtype helper. No torch dependency: reads safetensors directly via mlx-rs.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{Array, Dtype};

use crate::{Error, Result};

/// A loaded set of named tensors (dotted keys, e.g. `"layers.0.attention.to_q.weight"`)
/// plus the file's string metadata (e.g. a LoKr adapter's `networkType` / `alpha` / `rank`).
pub struct Weights {
    tensors: HashMap<String, Array>,
    metadata: HashMap<String, String>,
}

impl Weights {
    /// An empty weights container — for building small in-memory fixtures (e.g. checkpoint-remap unit
    /// tests that assert `remap_*_keys` produces the aliased names without a real snapshot).
    pub fn empty() -> Self {
        Self {
            tensors: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Load a single `.safetensors` file (tensors + metadata).
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let (tensors, metadata) = Array::load_safetensors_with_metadata(path.as_ref())?;
        Ok(Self { tensors, metadata })
    }

    /// Load and merge every `.safetensors` file under `dir` (sharded checkpoints). Keys
    /// across shards are disjoint, so a plain merge reconstructs the full tensor set
    /// without parsing the index — no torch, no shard map needed.
    ///
    /// Hidden entries are skipped: macOS AppleDouble sidecars (`._model.safetensors`) carry the
    /// `.safetensors` extension but are not shards, and sort ahead of the real file — see
    /// [`gen_core::weightsmeta::is_hidden_file`].
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let mut files: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .filter(|p| !gen_core::weightsmeta::is_hidden_file(p))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(format!("no .safetensors files in {}", dir.display()).into());
        }
        let mut tensors = HashMap::new();
        let mut metadata = HashMap::new();
        for f in files {
            // Name the offending file: the underlying mlx-c error carries only its own C++ source
            // location, so a corrupt/foreign file in a shard dir was previously undiagnosable.
            let (t, m) = Array::load_safetensors_with_metadata(&f)
                .map_err(|e| Error::from(format!("loading shard {}: {e}", f.display())))?;
            // Shards are expected to be disjoint; a key collision means the shard set is wrong (e.g.
            // a stray extra file in the dir) and a plain `extend` would silently let the later shard
            // win, loading a partially-wrong tensor set. Surface it instead (F-032). Metadata is
            // descriptive (per-shard `__metadata__`), so a later-wins merge there is benign.
            for (k, v) in t {
                if tensors.insert(k.clone(), v).is_some() {
                    return Err(format!(
                        "duplicate tensor key `{k}` across shards in {} (non-disjoint shard set)",
                        dir.display()
                    )
                    .into());
                }
            }
            metadata.extend(m);
        }
        Ok(Self { tensors, metadata })
    }

    pub fn get(&self, key: &str) -> Option<&Array> {
        self.tensors.get(key)
    }

    /// Get a tensor by key, returning an error (not panicking) when it is absent.
    pub fn require(&self, key: &str) -> Result<&Array> {
        self.tensors
            .get(key)
            .ok_or_else(|| Error::MissingTensor(key.to_string()))
    }

    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    /// Insert (or overwrite) a tensor under `key`. Used by checkpoint remapping (diffusers →
    /// internal names + conv-weight transposes) when loading real weights.
    pub fn insert(&mut self, key: impl Into<String>, tensor: Array) {
        self.tensors.insert(key.into(), tensor);
    }

    /// Copy the tensor at `from` to the new key `to` (no-op if `from` is absent). A convenience
    /// for the identity-but-renamed entries in a checkpoint→internal mapping.
    pub fn alias(&mut self, from: &str, to: &str) {
        if let Some(t) = self.tensors.get(from).cloned() {
            self.tensors.insert(to.to_string(), t);
        }
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Cast every tensor to `dtype` in place — mirrors the way a reference loader downcasts the
    /// whole checkpoint at load (e.g. the vendored SDXL `_load_safetensor_weights(..., float16=True)`
    /// applies `v.astype(mx.float16)` to every tensor). A no-op when `dtype` already matches.
    pub fn cast_all(&mut self, dtype: Dtype) -> Result<()> {
        // Per tensor this transiently holds both the source and the cast copy until the reassignment
        // drops the source (~2× that tensor's bytes at the moment of cast). Fine for the per-tensor
        // granularity here; a chunked cast+eval would be the lever if a full-checkpoint downcast ever
        // shows up as a peak-memory problem (F-089).
        for v in self.tensors.values_mut() {
            if v.dtype() != dtype {
                *v = v.as_dtype(dtype)?;
            }
        }
        Ok(())
    }
}

/// Cast to a target compute dtype (e.g. bf16, mirroring mflux's torch_convert downcast). A thin
/// `as_dtype` passthrough — kept as a named helper so call sites read as an explicit downcast (and
/// `as_dtype` itself no-ops when the dtype already matches) (F-089).
pub fn to_dtype(a: &Array, dtype: Dtype) -> Result<Array> {
    Ok(a.as_dtype(dtype)?)
}

/// Upcast to `f32`. The common case of [`to_dtype`] — providers that pin reductions/preprocessing
/// to f32 (e.g. scail2's CLIP/preprocess islands) call this so the intent reads at the use site.
pub fn to_f32(a: &Array) -> Result<Array> {
    Ok(a.as_dtype(Dtype::Float32)?)
}

/// Build a dotted tensor key from a `prefix` and a leaf `name`, collapsing the empty-prefix case
/// (so a root module addresses `"weight"`, a nested one `"blocks.0.weight"`). The Rust analogue of
/// the fork's `f"{prefix}.{name}"` key assembly — shared so provider loaders/text-encoders that
/// walk `tree_flatten`-style names don't each re-spell it.
pub fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_file_with_metadata() {
        let dir = std::env::temp_dir().join("mlx_gen_weights_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("w.safetensors");

        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let mut meta = HashMap::new();
        meta.insert("networkType".to_string(), "lokr".to_string());
        Array::save_safetensors(vec![("blk.weight", &a)], Some(&meta), &path).unwrap();

        let w = Weights::from_file(&path).unwrap();
        assert_eq!(w.len(), 1);
        assert!(w.get("blk.weight").is_some());
        assert!(w.require("blk.weight").is_ok());
        assert!(w.require("missing").is_err());
        assert_eq!(w.metadata("networkType"), Some("lokr"));
    }

    #[test]
    fn to_dtype_casts_to_bf16() {
        let a = Array::from_slice(&[1.0f32, 2.0], &[2]);
        assert_eq!(
            to_dtype(&a, Dtype::Bfloat16).unwrap().dtype(),
            Dtype::Bfloat16
        );
    }

    /// SceneWorks#1333 regression: `boogu_image` failed to load because its `mllm/` dir held a macOS
    /// AppleDouble sidecar next to the real shard. `._model.safetensors` has extension `safetensors`
    /// and sorts first, so `from_dir` opened it and mlx-c rejected its magic bytes with
    /// `[load_safetensors] Invalid json header length`. The sidecar must be skipped.
    #[test]
    fn from_dir_skips_appledouble_sidecar() {
        let dir = std::env::temp_dir().join(format!("mlx_gen_appledouble_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        Array::save_safetensors(
            vec![("blk.weight", &a)],
            None,
            dir.join("model.safetensors"),
        )
        .unwrap();
        // Real AppleDouble header (magic 0x00051607, version 0x00020000).
        std::fs::write(
            dir.join("._model.safetensors"),
            [0x00, 0x05, 0x16, 0x07, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00],
        )
        .unwrap();

        let w = Weights::from_dir(&dir).expect("sidecar must be skipped, not loaded");
        assert_eq!(w.len(), 1);
        assert!(w.get("blk.weight").is_some());

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A genuinely corrupt *shard* still errors — and the message now names the file, which the bare
    /// mlx-c error (a C++ source location) did not.
    #[test]
    fn from_dir_error_names_the_offending_shard() {
        let dir = std::env::temp_dir().join(format!("mlx_gen_bad_shard_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), [0x00, 0x05, 0x16, 0x07]).unwrap();

        let err = match Weights::from_dir(&dir) {
            Ok(_) => panic!("a corrupt shard must not load"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("model.safetensors"), "unexpected: {err}");

        std::fs::remove_dir_all(&dir).ok();
    }
}
