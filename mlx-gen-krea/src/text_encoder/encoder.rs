//! The Krea Qwen3-VL-4B text encoder forward: token embedding → causal Qwen3 decoder layers,
//! capturing the hidden states at `select_hidden` and **stacking** them on a new axis →
//! `[B, L, num_select, hidden]`, then dropping the leading template-prefix tokens. This is the exact
//! `context` the DiT's `TextFusionTransformer` consumes (sc-7568) — the aggregation happens there, not
//! here.
//!
//! HF `hidden_states` indexing: `hidden_states[i]` is the state after running `i` decoder layers
//! (`hidden_states[0]` = the raw embedding). So the reference's `select_hidden = [2,5,…,35]` capture
//! the OUTPUT of 0-indexed layers `[1,4,…,34]`. The final `language_model.norm` is never applied (all
//! selected layers are pre-final-norm), and only `max+1` layers are run (later layers can't matter).

use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

use mlx_gen::nn::{build_mask, TextRope, TokenEmbedding};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use super::{embedding, join, KreaTeConfig, Qwen3DecoderLayer};

pub struct KreaTextEncoder {
    embed_tokens: TokenEmbedding,
    layers: Vec<Qwen3DecoderLayer>,
    rope: TextRope,
    /// 0-indexed decoder-layer OUTPUT indices to capture (= `select_hidden[i] - 1`), in stack order.
    out_layers: Vec<usize>,
    prefix_tokens: i32,
}

impl KreaTextEncoder {
    /// Load from the `text_encoder` weights under `prefix` (`"language_model"`):
    /// `{prefix}.embed_tokens.weight`, `{prefix}.layers.{i}.…`. The final `{prefix}.norm.weight` is
    /// intentionally not loaded.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &KreaTeConfig) -> Result<Self> {
        let out_layers: Vec<usize> = cfg
            .select_hidden
            .iter()
            .map(|&s| {
                s.checked_sub(1).ok_or_else(|| {
                    Error::Msg("krea te: select_hidden index 0 has no layer output".into())
                })
            })
            .collect::<Result<_>>()?;
        let max_layer = *out_layers.iter().max().unwrap_or(&0);
        if max_layer as i32 >= cfg.num_layers {
            return Err(Error::Msg(format!(
                "krea te: select_hidden needs layer {max_layer} but the encoder has {} layers",
                cfg.num_layers
            )));
        }

        let mut layers = Vec::with_capacity(max_layer + 1);
        for i in 0..=max_layer {
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
            embed_tokens: embedding(w, &join(prefix, "embed_tokens"))?,
            layers,
            rope: TextRope::new(cfg.head_dim, cfg.rope_theta),
            out_layers,
            prefix_tokens: cfg.prefix_tokens as i32,
        })
    }

    /// Quantize the token table + every decoder-layer projection in place (group-wise affine Q4/Q8).
    /// `cast_to_bf16=true` for the embedding matches the Qwen3 TE convention; the norms stay dense.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.embed_tokens.quantize(bits, true)?;
        for layer in &mut self.layers {
            layer.quantize(bits)?;
        }
        Ok(())
    }

    /// `input_ids` / `attention_mask`: `[b, s]` int32. Returns the stacked conditioning
    /// `[b, s - prefix_tokens, num_select, hidden]` (the DiT's `context`). The final norm is never
    /// applied; only layers up to `max(out_layers)` are run.
    pub fn forward(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        let (cos, sin) = self.rope.forward(s)?;
        let mask = build_mask(attention_mask, b, s)?;

        let mut hidden = self.embed_tokens.forward(input_ids)?;
        let mut saved: Vec<(usize, Array)> = Vec::with_capacity(self.out_layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
            if self.out_layers.contains(&i) {
                saved.push((i, hidden.clone()));
            }
        }

        // Stack the captured layers (in `out_layers` order) on a NEW axis 2 → [b, s, n, hidden],
        // matching the reference `torch.stack([hidden_states[i] for i in select], dim=2)`.
        let pick = |idx: usize| -> Result<&Array> {
            saved
                .iter()
                .find(|(k, _)| *k == idx)
                .map(|(_, v)| v)
                .ok_or_else(|| Error::Msg(format!("krea te: hidden state {idx} not captured")))
        };
        let expanded: Vec<Array> = self
            .out_layers
            .iter()
            .map(|&idx| Ok(pick(idx)?.expand_dims(2)?))
            .collect::<Result<_>>()?;
        let refs: Vec<&Array> = expanded.iter().collect();
        let stacked = concatenate_axis(&refs, 2)?; // [b, s, n, hidden]

        // Drop the leading template-prefix tokens (the system instruction).
        let n = stacked.shape()[1];
        let idx: Vec<i32> = (self.prefix_tokens..n).collect();
        Ok(stacked.take_axis(Array::from_slice(&idx, &[idx.len() as i32]), 1)?)
    }
}
