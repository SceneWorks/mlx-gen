//! Krea 2's **Qwen3-VL-4B-Instruct** condition encoder (text path only — the vision tower is unused
//! for text-to-image). A 36-layer decoder-only LM; the hidden states at the 12 evenly-spaced indices
//! `text_encoder_select_layers = [2,5,…,35]` are **stacked** (not aggregated here) into
//! `[B, L, 12, 2560]` — the exact contract the DiT's `TextFusionTransformer` consumes (sc-7568). The
//! learned aggregation lives in the DiT, NOT here: this module just runs Qwen3-VL-4B and collects the
//! 12 layers.
//!
//! Mirrors the `mlx-gen-ideogram` / `mlx-gen-flux2` Qwen3 assembly over the shared `mlx-gen` core
//! primitives (`TextRope`, `TokenEmbedding`, `AdaptableLinear`, `rms_norm`, masked SDPA). GQA
//! (32 query / 8 kv heads), bias-less q/k/v/o, **per-head q/k RMSNorm**, HF half-split RoPE
//! (θ = 5e6), SwiGLU MLP, pre-norm residual blocks, causal mask. The text-only path uses plain 1-D
//! RoPE: Qwen3-VL's interleaved MRoPE sections all index the same sequential text position when there
//! are no image tokens, so it reduces exactly to standard RoPE. Weights live under `language_model.*`.

pub mod attention;
pub mod encoder;
pub mod layer;
pub mod mlp;
pub mod tokenizer;

pub use attention::Qwen3Attention;
pub use encoder::KreaTextEncoder;
pub use layer::Qwen3DecoderLayer;
pub use mlp::Qwen3Mlp;
pub use tokenizer::KreaTokenizer;

// The HF half-split text RoPE is identical across families and lives in core.
pub use mlx_gen::nn::TextRope;

use std::path::Path;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::TokenEmbedding;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

/// Qwen3-VL-4B text-tower architecture (verified from the published `text_encoder/config.json`:
/// `qwen3_vl_text`, hidden 2560, 36 layers, GQA 32/8, head_dim 128, FFN 9728, eps 1e-6) + the Krea
/// conditioning policy (which hidden-state layers to stack, how many template-prefix tokens to drop).
#[derive(Debug, Clone, PartialEq)]
pub struct KreaTeConfig {
    pub hidden_size: i32,
    pub num_layers: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub intermediate_size: i32,
    pub rms_norm_eps: f32,
    /// `rope_theta` is null in the published `text_config`; the effective default for `qwen3_vl_text`
    /// is **5e6** (the `rope_parameters.rope_theta` transformers fills in).
    pub rope_theta: f32,
    /// HF `output_hidden_states` indices the pipeline stacks (`model_index.json`
    /// `text_encoder_select_layers`): `hidden_states[i]` = the LM state after running `i` layers.
    pub select_hidden: Vec<usize>,
    /// Leading template-prefix tokens dropped from the conditioning (`Qwen3VLConditioner`'s
    /// `prompt_template_encode_start_idx`); the system-instruction prefix tokenizes to this many.
    pub prefix_tokens: usize,
}

impl KreaTeConfig {
    pub fn qwen3_vl_4b() -> Self {
        Self {
            hidden_size: 2560,
            num_layers: 36,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 9728,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            select_hidden: vec![2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35],
            prefix_tokens: 34,
        }
    }

    /// Parse `<root>/text_encoder/config.json` (`text_config`) + `<root>/model_index.json`
    /// (`text_encoder_select_layers`); missing scalars fall back to [`Self::qwen3_vl_4b`].
    pub fn from_snapshot(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        let path = root.join("text_encoder").join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| Error::Msg(format!("krea te: read {}: {e}", path.display())))?;
        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| Error::Msg(format!("krea te: parse {}: {e}", path.display())))?;
        let tc = v.get("text_config").unwrap_or(&v);
        let d = Self::qwen3_vl_4b();
        let u = |k: &str, dflt: i32| {
            tc.get(k)
                .and_then(serde_json::Value::as_i64)
                .map(|n| n as i32)
                .unwrap_or(dflt)
        };

        let mut cfg = Self {
            hidden_size: u("hidden_size", d.hidden_size),
            num_layers: u("num_hidden_layers", d.num_layers),
            num_heads: u("num_attention_heads", d.num_heads),
            num_kv_heads: u("num_key_value_heads", d.num_kv_heads),
            head_dim: u("head_dim", d.head_dim),
            intermediate_size: u("intermediate_size", d.intermediate_size),
            rms_norm_eps: tc
                .get("rms_norm_eps")
                .and_then(serde_json::Value::as_f64)
                .map(|n| n as f32)
                .unwrap_or(d.rms_norm_eps),
            // `text_config.rope_theta` is null on disk; honor `rope_parameters`/`rope_scaling` if set,
            // else the qwen3_vl_text default (5e6).
            rope_theta: tc
                .get("rope_parameters")
                .or_else(|| tc.get("rope_scaling"))
                .and_then(|r| r.get("rope_theta"))
                .or_else(|| tc.get("rope_theta"))
                .and_then(serde_json::Value::as_f64)
                .map(|n| n as f32)
                .unwrap_or(d.rope_theta),
            select_hidden: d.select_hidden.clone(),
            prefix_tokens: d.prefix_tokens,
        };

        // `text_encoder_select_layers` lives in the pipeline manifest.
        if let Ok(t) = std::fs::read_to_string(root.join("model_index.json")) {
            if let Ok(mv) = serde_json::from_str::<serde_json::Value>(&t) {
                if let Some(arr) = mv
                    .get("text_encoder_select_layers")
                    .and_then(|a| a.as_array())
                {
                    let sel: Vec<usize> = arr
                        .iter()
                        .filter_map(|x| x.as_u64().map(|n| n as usize))
                        .collect();
                    if !sel.is_empty() {
                        cfg.select_hidden = sel;
                    }
                }
            }
        }
        Ok(cfg)
    }
}

/// Load a bias-less Qwen3 projection from its `{base}.weight` `key`, auto-detecting a packed snapshot.
pub(crate) fn lin(w: &Weights, key: &str) -> Result<AdaptableLinear> {
    let base = key.strip_suffix(".weight").unwrap_or(key);
    crate::quant::lin(w, base, false)
}

/// Load a token embedding, auto-detecting a packed snapshot.
pub(crate) fn embedding(w: &Weights, base: &str) -> Result<TokenEmbedding> {
    crate::quant::embedding(w, base)
}

/// Join a module prefix with a leaf name, tolerating an empty prefix.
pub(crate) fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}
