//! Boogu Qwen3-VL condition encoder: token embedding → all `num_layers` causal decoder layers →
//! final RMSNorm → **last_hidden_state** `[B, L, 4096]` (the per-token instruction features the DiT
//! caption embedder consumes). Differs from the ideogram TE only in the head: Boogu applies the
//! final norm and returns a single layer, vs ideogram's 13-layer pre-final-norm interleave.

use mlx_rs::fast::rms_norm;
use mlx_rs::Array;

use mlx_gen::array::host_i32;
use mlx_gen::nn::{TextRope, TokenEmbedding};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, BooguTextEncoderConfig, Qwen3DecoderLayer};

pub struct BooguTextEncoder {
    embed_tokens: TokenEmbedding,
    layers: Vec<Qwen3DecoderLayer>,
    rope: TextRope,
    final_norm: Array,
    eps: f32,
}

impl BooguTextEncoder {
    /// Load from the `mllm` weights under `prefix` (`"model.language_model"`):
    /// `{prefix}.embed_tokens.weight`, `{prefix}.layers.{i}.…`, `{prefix}.norm.weight`.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &BooguTextEncoderConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_layers as usize);
        for i in 0..cfg.num_layers {
            layers.push(Qwen3DecoderLayer::from_weights(
                w,
                &join(prefix, &format!("layers.{i}")),
                cfg.num_heads,
                cfg.num_kv_heads,
                cfg.head_dim,
                cfg.rms_norm_eps,
            )?);
        }
        Ok(Self {
            embed_tokens: crate::quant::embedding(w, &join(prefix, "embed_tokens"))?,
            layers,
            rope: TextRope::new(cfg.head_dim, cfg.rope_theta),
            final_norm: w.require(&join(prefix, "norm.weight"))?.clone(),
            eps: cfg.rms_norm_eps,
        })
    }

    /// Quantize the embedding + every decoder-layer projection in place (group-wise Q4/Q8). The
    /// per-layer norms + final norm stay dense.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.embed_tokens.quantize(bits, true)?;
        for layer in &mut self.layers {
            layer.quantize(bits)?;
        }
        Ok(())
    }

    /// `input_ids` / `attention_mask`: `[b, s]` int32. Returns `last_hidden_state` `[b, s, 4096]`
    /// (f32) — all layers run, final norm applied.
    pub fn last_hidden(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        let (cos, sin) = self.rope.forward(s)?;
        let mask = build_mask(attention_mask, b, s)?;

        let mut hidden = self.embed_tokens.forward(input_ids)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
        }
        Ok(rms_norm(&hidden, &self.final_norm, self.eps)?)
    }
}

/// Additive attention mask `[b, 1, s, s]`: `0` where a query may attend (key is causal **and** not
/// padding), `-inf` otherwise. The Qwen3 LM is causal.
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
