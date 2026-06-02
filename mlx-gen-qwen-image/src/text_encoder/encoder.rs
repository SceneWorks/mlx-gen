//! Full Qwen2.5-VL text encoder: token embedding → 28 pre-norm decoder layers → final RMSNorm.
//! `encode` then applies the fork's prompt post-processing (drop the leading system-prompt tokens).

use mlx_rs::fast::rms_norm;
use mlx_rs::{Array, Dtype};

use mlx_gen::array::host_i32;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, QwenEncoderLayer, TextRope};

/// Qwen2.5-VL LM dimensions (the `text_encoder` of `Qwen/Qwen-Image`).
pub struct QwenTextEncoderConfig {
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub n_layers: usize,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub intermediate_size: i32,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    /// Tokens to drop from the front of the prompt (the chat-template system prefix).
    pub prompt_drop_idx: i32,
}

impl QwenTextEncoderConfig {
    pub fn qwen_image() -> Self {
        Self {
            vocab_size: 152064,
            hidden_size: 3584,
            n_layers: 28,
            n_heads: 28,
            n_kv_heads: 4,
            head_dim: 128,
            intermediate_size: 18944,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            prompt_drop_idx: 34,
        }
    }
}

pub struct QwenTextEncoder {
    embed_tokens: Array,
    layers: Vec<QwenEncoderLayer>,
    norm: Array,
    rope: TextRope,
    eps: f32,
    drop_idx: i32,
}

impl QwenTextEncoder {
    /// Loads from the fork's internal tree under `prefix` (`"encoder"`):
    /// `{prefix}.embed_tokens.weight`, `{prefix}.layers.{i}.…`, `{prefix}.norm.weight`.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &QwenTextEncoderConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(QwenEncoderLayer::from_weights(
                w,
                &join(prefix, &format!("layers.{i}")),
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.head_dim,
                cfg.rms_norm_eps,
            )?);
        }
        Ok(Self {
            embed_tokens: w.require(&join(prefix, "embed_tokens.weight"))?.clone(),
            layers,
            norm: w.require(&join(prefix, "norm.weight"))?.clone(),
            rope: TextRope::new(cfg.head_dim, cfg.rope_theta),
            eps: cfg.rms_norm_eps,
            drop_idx: cfg.prompt_drop_idx,
        })
    }

    /// Token embedding (f32): `input_ids` `[b, s]` int32 → `[b, s, hidden]`. Exposed so the VL
    /// encoder can splice vision embeds into the stream before running the layers.
    pub fn embed(&self, input_ids: &Array) -> Result<Array> {
        Ok(self
            .embed_tokens
            .take_axis(input_ids, 0)?
            .as_dtype(Dtype::Float32)?)
    }

    /// Run the decoder stack (RoPE + 28 layers + final RMSNorm) over pre-embedded `[b, s, hidden]`
    /// hidden states. Shared by [`forward`](Self::forward) and the VL encoder's spliced path.
    pub fn forward_from_embeds(&self, embeds: &Array, attention_mask: &Array) -> Result<Array> {
        let sh = embeds.shape();
        let (b, s) = (sh[0], sh[1]);
        let (cos, sin) = self.rope.forward(s)?;
        let mask = build_mask(attention_mask, b, s)?;

        let mut hidden = embeds.clone();
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
        }
        Ok(rms_norm(&hidden, &self.norm, self.eps)?)
    }

    /// `input_ids` / `attention_mask`: `[b, s]` int32. Returns the final-normed hidden states
    /// `[b, s, hidden]` (f32).
    pub fn forward(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        self.forward_from_embeds(&self.embed(input_ids)?, attention_mask)
    }

    /// Prompt conditioning: final-normed hidden states with the leading `drop_idx` system-prompt
    /// tokens removed. Assumes a single un-padded sequence per row (the per-prompt pipeline case);
    /// variable-length batch padding is handled at the pipeline level.
    pub fn encode(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        let hidden = self.forward(input_ids, attention_mask)?;
        let s = hidden.shape()[1];
        let idx: Vec<i32> = (self.drop_idx..s).collect();
        let idx = Array::from_slice(&idx, &[idx.len() as i32]);
        Ok(hidden.take_axis(&idx, 1)?)
    }
}

/// Additive attention mask `[b, 1, s, s]`: `0` where a query may attend (key is causal **and**
/// not padding), `-inf` otherwise.
///
/// Built host-side (a one-time `O(b·s²)` fill per prompt encode, **not** per denoise step).
/// Deliberately kept on the host rather than constructed with on-device broadcast ops: at realistic
/// prompt lengths this is negligible against the denoise loop, and a plain fill is the simplest way
/// to stay bit-exact with the fork (sc-2583). Revisit only if profiling ever flags it.
fn build_mask(attention_mask: &Array, b: i32, s: i32) -> Result<Array> {
    let am = host_i32(attention_mask)?;
    let (b, s) = (b as usize, s as usize);
    let mut data = vec![0f32; b * s * s];
    for bi in 0..b {
        for i in 0..s {
            for j in 0..s {
                let allowed = j <= i && am[bi * s + j] == 1;
                if !allowed {
                    data[(bi * s + i) * s + j] = f32::NEG_INFINITY;
                }
            }
        }
    }
    Ok(Array::from_slice(&data, &[b as i32, 1, s as i32, s as i32]))
}
