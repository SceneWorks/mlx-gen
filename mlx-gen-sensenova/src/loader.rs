//! Weight-map foundation for the SenseNova-U1 (NEO-Unify) checkpoint.
//!
//! The 8B-MoT checkpoint is a **flat** sharded safetensors set (8 shards plus an index) — no
//! diffusers component sub-directories. [`load_raw`] merges the shards via the
//! core [`Weights`] loader; [`expected_keys`] enumerates the **canonical** tensor names the dense
//! dual-transformer architecture defines from a [`NeoChatConfig`]; and [`check_coverage`] diffs
//! those against a checkpoint's actual keys, so the downstream module loaders (sc-3182 …) can rely
//! on the key layout being exactly what they expect — nothing missing, nothing unaccounted for.
//!
//! The per-layer key shape (validated against the real checkpoint, 1116 tensors total) is two
//! parallel **dense** stacks per layer, the generation path carrying a `_mot_gen` suffix. The
//! suffix attaches to different name segments per group (a quirk of the reference module names):
//! `input_layernorm_mot_gen`, `mlp_mot_gen.gate_proj`, `self_attn.q_proj_mot_gen`,
//! `self_attn.q_norm_mot_gen`, `self_attn.q_norm_hw_mot_gen`.

use std::collections::BTreeSet;
use std::path::Path;

use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::NeoChatConfig;

/// The generation-path module-name suffix (understanding path = no suffix).
const GEN: &str = "_mot_gen";

/// Load and merge every shard under a SenseNova-U1 snapshot directory into one [`Weights`] map.
///
/// The checkpoint is flat (all `*.safetensors` shards live directly under `root`), so the core
/// directory loader reconstructs the full tensor set without parsing the index. Validate the result
/// with [`check_coverage`] before building modules.
pub fn load_raw(root: impl AsRef<Path>) -> Result<Weights> {
    Weights::from_dir(root)
}

/// The canonical set of tensor keys the NEO-Unify dense dual-transformer defines for `cfg`.
///
/// Built purely from the config (layer count, `tie_word_embeddings`, `fm_head_layers`,
/// `add_noise_scale_embedding`) so it is the single source of truth the module loaders share with
/// [`check_coverage`].
pub fn expected_keys(cfg: &NeoChatConfig) -> BTreeSet<String> {
    let mut keys = BTreeSet::new();
    let lm = "language_model.model";

    // Embeddings + final norms (dual-path norm) + the untied lm_head.
    keys.insert(format!("{lm}.embed_tokens.weight"));
    keys.insert(format!("{lm}.norm.weight"));
    keys.insert(format!("{lm}.norm{GEN}.weight"));
    if !cfg.tie_word_embeddings {
        keys.insert("language_model.lm_head.weight".to_string());
    }

    // 42 decoder layers × two dense paths (understanding "", generation `_mot_gen`).
    for i in 0..cfg.llm.num_hidden_layers {
        let layer = format!("{lm}.layers.{i}");
        for gen in [false, true] {
            let s = if gen { GEN } else { "" };
            keys.insert(format!("{layer}.input_layernorm{s}.weight"));
            keys.insert(format!("{layer}.post_attention_layernorm{s}.weight"));
            // SwiGLU MLP — suffix attaches to the `mlp` segment.
            for proj in ["gate", "up", "down"] {
                keys.insert(format!("{layer}.mlp{s}.{proj}_proj.weight"));
            }
            // Attention projections — suffix attaches to the proj segment.
            for proj in ["q", "k", "v", "o"] {
                keys.insert(format!("{layer}.self_attn.{proj}_proj{s}.weight"));
            }
            // QK-norms: temporal (`q_norm`/`k_norm`) + spatial (`q_norm_hw`/`k_norm_hw`).
            for n in ["q_norm", "k_norm", "q_norm_hw", "k_norm_hw"] {
                keys.insert(format!("{layer}.self_attn.{n}{s}.weight"));
            }
        }
    }

    // Vision embedders: the understanding-path tower + the generation-path tower under fm_modules.
    // Each is a Conv `patch_embedding` + Conv `dense_embedding` (both with bias).
    for emb in [
        "vision_model.embeddings",
        "fm_modules.vision_model_mot_gen.embeddings",
    ] {
        for conv in ["patch_embedding", "dense_embedding"] {
            keys.insert(format!("{emb}.{conv}.weight"));
            keys.insert(format!("{emb}.{conv}.bias"));
        }
    }

    // Flow-matching head: a Linear/GELU `Sequential` → Linear weights at even indices 0, 2, … .
    for j in 0..cfg.fm_head_layers {
        let idx = j * 2;
        keys.insert(format!("fm_modules.fm_head.{idx}.weight"));
        keys.insert(format!("fm_modules.fm_head.{idx}.bias"));
    }

    // Timestep embedder (always) + noise-scale embedder (only when enabled): a 2-Linear MLP each,
    // weights at indices 0 and 2.
    let mut embedders = vec!["timestep_embedder"];
    if cfg.add_noise_scale_embedding {
        embedders.push("noise_scale_embedder");
    }
    for emb in embedders {
        for idx in [0, 2] {
            keys.insert(format!("fm_modules.{emb}.mlp.{idx}.weight"));
            keys.insert(format!("fm_modules.{emb}.mlp.{idx}.bias"));
        }
    }

    keys
}

/// The result of diffing a checkpoint's keys against [`expected_keys`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Coverage {
    /// Keys the architecture expects but the checkpoint does not provide.
    pub missing: Vec<String>,
    /// Keys present in the checkpoint that the architecture does not account for.
    pub unexpected: Vec<String>,
}

impl Coverage {
    /// `true` when every expected key is present and no checkpoint key is unaccounted for.
    pub fn is_complete(&self) -> bool {
        self.missing.is_empty() && self.unexpected.is_empty()
    }
}

/// Diff a checkpoint's tensor keys against the canonical [`expected_keys`] for `cfg`.
pub fn check_coverage<'a>(
    present: impl IntoIterator<Item = &'a str>,
    cfg: &NeoChatConfig,
) -> Coverage {
    let expected = expected_keys(cfg);
    let present: BTreeSet<String> = present.into_iter().map(str::to_string).collect();
    Coverage {
        missing: expected.difference(&present).cloned().collect(),
        unexpected: present.difference(&expected).cloned().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::mot_8b;

    #[test]
    fn expected_keys_match_8b_mot_layout() {
        let keys = expected_keys(&mot_8b());
        // 4 top-level (embed + norm + norm_mot_gen + lm_head)
        // + 42 layers × 26 per-layer (13 per path × 2 paths)
        // + 8 vision (2 towers × 2 convs × {weight,bias})
        // + 4 fm_head (2 linears × {weight,bias})
        // + 8 embedders (timestep + noise_scale, 2 idx × {weight,bias} each)
        assert_eq!(
            keys.len(),
            4 + 42 * 26 + 8 + 4 + 8,
            "1116 canonical tensors"
        );

        // Spot-check the suffix placement on the generation path.
        for k in [
            "language_model.model.embed_tokens.weight",
            "language_model.lm_head.weight",
            "language_model.model.norm_mot_gen.weight",
            "language_model.model.layers.0.input_layernorm_mot_gen.weight",
            "language_model.model.layers.41.mlp_mot_gen.gate_proj.weight",
            "language_model.model.layers.0.self_attn.q_proj_mot_gen.weight",
            "language_model.model.layers.0.self_attn.q_norm_hw_mot_gen.weight",
            "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.bias",
            "fm_modules.fm_head.2.weight",
            "fm_modules.noise_scale_embedder.mlp.0.bias",
        ] {
            assert!(keys.contains(k), "missing canonical key {k}");
        }
    }

    #[test]
    fn coverage_flags_missing_and_unexpected() {
        let cfg = mot_8b();
        let expected = expected_keys(&cfg);

        // Identical sets → complete.
        let present: Vec<&str> = expected.iter().map(String::as_str).collect();
        assert!(check_coverage(present.iter().copied(), &cfg).is_complete());

        // Drop one + add a stray → both surfaced.
        let mut trimmed: Vec<String> = expected.iter().cloned().collect();
        let dropped = trimmed.pop().unwrap();
        trimmed.push("language_model.model.layers.0.bogus.weight".to_string());
        let cov = check_coverage(trimmed.iter().map(String::as_str), &cfg);
        assert!(!cov.is_complete());
        assert_eq!(cov.missing, vec![dropped]);
        assert_eq!(
            cov.unexpected,
            vec!["language_model.model.layers.0.bogus.weight".to_string()]
        );
    }

    #[test]
    fn tied_embeddings_drops_lm_head() {
        let mut cfg = mot_8b();
        cfg.tie_word_embeddings = true;
        assert!(!expected_keys(&cfg).contains("language_model.lm_head.weight"));
    }
}
