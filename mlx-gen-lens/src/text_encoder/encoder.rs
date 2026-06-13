//! The Lens **encoder-only** gpt-oss-20b stack (sc-3171): `embed_tokens` → 24 decoder layers with
//! per-layer sliding/full masks → capture the hidden states at the selected layers `[5, 11, 17, 23]`
//! → early-exit after the last selected layer. A faithful port of
//! `_vendor/lens/text_encoder.py::LensGptOssEncoder.forward` (the Lens feature-extraction path).
//!
//! ## Parity-critical details (from the reference)
//! - **Captured = layer *output*.** `captured[pos] = hidden_states` is taken *after* running decoder
//!   layer `i` (not the embedding-offset `hidden_states[i]` of HF's stock `output_hidden_states`). So
//!   the default selection `[5, 11, 17, 23]` is the output of decoder indices 5/11/17/23.
//! - **Per-layer mask by `layer_types[i]`.** Even layers are sliding-window (window 128), odd layers
//!   are full causal ([`GptOssConfig::is_sliding`]). Both masks are built once for the sequence and
//!   reused; for the un-padded single prompt the Lens encoder runs this is pure causal ±the window.
//! - **`position_ids = arange(L)`**, RoPE computed once (the YaRN `inv_freq` + `attention_scaling`).
//! - **No final `norm`, no LM head, no KV cache, no generation** — the feature path stops at the max
//!   selected layer.
//!
//! The encoder runs the *whole* token sequence (the 97-token harmony preamble is real causal
//! context); the DiT later consumes the captured features sliced at `txt_offset = 97` (sc-3173).

use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Quant, Result};

use crate::config::GptOssConfig;
use crate::text_encoder::gpt_oss::{attention_mask, GptOssDecoderLayer};

/// The Lens default multi-layer capture indices (`selected_layer_index` in the DiT config /
/// `set_selected_layers` default).
pub const DEFAULT_SELECTED_LAYERS: [usize; 4] = [5, 11, 17, 23];

/// The Lens gpt-oss-20b text encoder, run encoder-only with multi-layer hidden capture.
pub struct LensTextEncoder {
    /// `model.embed_tokens.weight`, `[vocab, hidden]`.
    embed_tokens: Array,
    /// Decoder layers `0..=max_selected` (the stack is truncated at the last captured layer — the
    /// remaining layers, the final `norm`, and the LM head are never built).
    layers: Vec<GptOssDecoderLayer>,
    /// YaRN RoPE frequencies `[head_dim/2]` and the `attention_scaling` (mscale), computed once.
    inv_freq: Array,
    attn_scaling: f32,
    /// Layer indices whose outputs are captured, in the order the DiT expects them.
    selected_layers: Vec<usize>,
    sliding_window: i32,
    dtype: Dtype,
    /// `is_sliding` mapping is config-driven; kept for the per-layer mask choice.
    cfg: GptOssConfig,
}

impl LensTextEncoder {
    /// Load the encoder from the full `text_encoder` weights at `dtype` (bf16 production / f32 gate),
    /// capturing the [`DEFAULT_SELECTED_LAYERS`]. Only layers `0..=max(selected)` are constructed —
    /// for the default selection that is all 24, but a smaller selection loads (and dequantizes the
    /// MXFP4 experts of) only the needed prefix.
    pub fn from_weights(w: &Weights, cfg: &GptOssConfig, dtype: Dtype) -> Result<Self> {
        Self::with_selected_layers(w, cfg, dtype, DEFAULT_SELECTED_LAYERS.to_vec(), None)
    }

    /// As [`from_weights`](Self::from_weights) but quantizes the MoE experts to Q4/Q8 (sc-3172) so the
    /// encoder loads at `~12 GB` instead of `~40 GB` bf16. Attention / router / embedding stay dense.
    pub fn from_weights_quant(
        w: &Weights,
        cfg: &GptOssConfig,
        dtype: Dtype,
        quant: Option<Quant>,
    ) -> Result<Self> {
        Self::with_selected_layers(w, cfg, dtype, DEFAULT_SELECTED_LAYERS.to_vec(), quant)
    }

    /// As [`from_weights`](Self::from_weights) but with an explicit (non-empty, unique, in-range)
    /// capture-index list (`set_selected_layers`) and optional MoE-expert quantization.
    pub fn with_selected_layers(
        w: &Weights,
        cfg: &GptOssConfig,
        dtype: Dtype,
        selected_layers: Vec<usize>,
        quant: Option<Quant>,
    ) -> Result<Self> {
        // Reachable from `Result`-returning public APIs, so error rather than panic the worker on a
        // bad capture-index list (F-014).
        let max_layer = *selected_layers
            .iter()
            .max()
            .ok_or_else(|| Error::Msg("lens encoder: selected_layers must be non-empty".into()))?;
        if max_layer >= cfg.num_layers {
            return Err(Error::Msg(format!(
                "lens encoder: selected layer {max_layer} out of range (model has {} layers)",
                cfg.num_layers
            )));
        }

        let embed_tokens = w.require("model.embed_tokens.weight")?.as_dtype(dtype)?;
        let mut layers = Vec::with_capacity(max_layer + 1);
        for i in 0..=max_layer {
            layers.push(GptOssDecoderLayer::from_weights(
                w,
                &format!("model.layers.{i}"),
                cfg,
                dtype,
                quant,
            )?);
        }

        let (inv_freq, attn_scaling) = cfg.yarn_rope();
        Ok(Self {
            embed_tokens,
            layers,
            inv_freq: Array::from_slice(&inv_freq, &[inv_freq.len() as i32]),
            attn_scaling,
            selected_layers,
            sliding_window: cfg.sliding_window,
            dtype,
            cfg: *cfg,
        })
    }

    /// The capture indices, in DiT order.
    pub fn selected_layers(&self) -> &[usize] {
        &self.selected_layers
    }

    /// Encode `input_ids` `[B, L]` (int32) → the captured hidden states, one `[B, L, hidden]` per
    /// selected layer in selection order (== `LensGptOssEncoder.forward`'s returned list). Runs
    /// `position_ids = arange(L)` and stops after the max selected layer.
    pub fn encode(&self, input_ids: &Array) -> Result<Vec<Array>> {
        let l = input_ids.shape()[1];

        // Both per-layer masks, built once for the sequence (full causal + sliding-window causal).
        let full_mask = attention_mask(l, None, self.dtype)?;
        let sliding_mask = attention_mask(l, Some(self.sliding_window), self.dtype)?;

        let mut hidden = self.embed_tokens.take_axis(input_ids, 0)?; // [B, L, hidden]

        // Capture slots, filled in selection order (matches the reference's `index_lookup`).
        let mut captured: Vec<Option<Array>> = vec![None; self.selected_layers.len()];
        for (i, layer) in self.layers.iter().enumerate() {
            let mask = if self.cfg.is_sliding(i) {
                &sliding_mask
            } else {
                &full_mask
            };
            hidden = layer.forward(&hidden, &self.inv_freq, self.attn_scaling, mask)?;
            if let Some(pos) = self.selected_layers.iter().position(|&s| s == i) {
                captured[pos] = Some(hidden.clone());
            }
        }

        Ok(captured
            .into_iter()
            .map(|c| c.expect("every selected layer captured"))
            .collect())
    }
}
