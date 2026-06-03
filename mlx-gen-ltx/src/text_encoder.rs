//! LTX-2.3 text encoder — the full S1 path producing `video_embeddings` from token ids.
//!
//! Port of `text_encoder.py::LTX2TextEncoder.encode` (the 2.3 "v2" / per-token-RMS feature path):
//!   Gemma-3-12B (49 hidden states) → `norm_and_concat_per_token_rms` (3840×49 = 188160)
//!   → `rescale_norm(√(out/hidden))` → `video_aggregate_embed` Linear (188160 → 4096)
//!   → `Embeddings1DConnector` → `video_embeddings` (1, L, 4096).
//!
//! Runs **bf16** end-to-end to match the reference (gemma-3-12b-it-bf16 + bf16 activations).
//! `video_aggregate_embed.{weight,bias}` and the connector both live in `connector.safetensors`
//! (`text_embedding_projection.video_aggregate_embed.*`, `video_embeddings_connector.*`).

use mlx_rs::ops::{add, mean_axes, multiply, rsqrt, stack_axis};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::linear;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::LtxConfig;
use crate::connector::Connector;
use crate::gemma::{GemmaConfig, GemmaModel};

const RMS_EPS: f32 = 1e-6;

/// The LTX-2.3 video text encoder (Gemma backbone + feature extractor + connector).
pub struct LtxTextEncoder {
    gemma: GemmaModel,
    aggregate_w: Array, // (out_dim, 188160)
    aggregate_b: Array, // (out_dim,)
    connector: Connector,
    rescale: Array, // √(out_dim / hidden) as a scalar in `dtype`
    dtype: Dtype,
}

impl LtxTextEncoder {
    /// Build from the Gemma weights + the LTX `connector.safetensors` (which carries both the
    /// `video_aggregate_embed` feature-extractor Linear and the video connector). `dtype` = the
    /// compute dtype (bf16 to match the reference).
    pub fn from_weights(
        gemma_w: &Weights,
        connector_w: &Weights,
        gemma_cfg: GemmaConfig,
        ltx_cfg: &LtxConfig,
        dtype: Dtype,
    ) -> Result<Self> {
        let gemma = GemmaModel::from_weights(gemma_w, gemma_cfg)?;
        let load = |key: &str| -> Result<Array> {
            connector_w
                .get(key)
                .ok_or_else(|| Error::MissingTensor(key.into()))?
                .as_dtype(dtype)
                .map_err(Error::from)
        };
        let aggregate_w = load("text_embedding_projection.video_aggregate_embed.weight")?;
        let aggregate_b = load("text_embedding_projection.video_aggregate_embed.bias")?;
        let out_dim = aggregate_w.shape()[0];
        let hidden = gemma_cfg.hidden_size;
        let rescale =
            Array::from_slice(&[(out_dim as f32 / hidden as f32).sqrt()], &[1]).as_dtype(dtype)?;
        let connector =
            Connector::from_weights(connector_w, "video_embeddings_connector.", ltx_cfg, dtype)?;
        Ok(Self {
            gemma,
            aggregate_w,
            aggregate_b,
            connector,
            rescale,
            dtype,
        })
    }

    /// Encode `(1, L)` token ids + `(1, L)` attention mask → `video_embeddings` `(1, L, 4096)`.
    pub fn encode(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        Ok(self.encode_with_features(input_ids, attention_mask)?.1)
    }

    /// Like [`encode`](Self::encode) but also returns the pre-connector `video_features` (the
    /// feature-extractor output) for stage localization.
    pub fn encode_with_features(
        &self,
        input_ids: &Array,
        attention_mask: &Array,
    ) -> Result<(Array, Array)> {
        let hiddens = self.gemma.forward(input_ids, attention_mask)?; // 49 × (1, L, 3840)
        let video_features = self.feature_extract(&hiddens, attention_mask)?;
        let video_embeddings = self.connector.forward(&video_features, attention_mask)?;
        Ok((video_features, video_embeddings))
    }

    /// `norm_and_concat_per_token_rms` + `rescale_norm` + `video_aggregate_embed`.
    /// Each `(token, layer)` slice over the 3840 hidden dim is RMS-normalized independently, the 49
    /// layers are concatenated **dim-major / layer-minor** (`d*49 + layer`, via stack+reshape),
    /// padded positions zeroed, scaled by `√(out/hidden)`, then projected to `out_dim`.
    fn feature_extract(&self, hiddens: &[Array], attention_mask: &Array) -> Result<Array> {
        let refs: Vec<&Array> = hiddens.iter().collect();
        let encoded = stack_axis(&refs, 3)?; // (1, L, 3840, 49)
        let sh = encoded.shape();
        let (b, l) = (sh[0], sh[1]);
        // per-token RMS over the hidden dim (axis 2), per layer.
        let var = mean_axes(&multiply(&encoded, &encoded)?, &[2], true)?; // (1, L, 1, 49)
        let eps = Array::from_slice(&[RMS_EPS], &[1]).as_dtype(self.dtype)?;
        let normed = multiply(&encoded, &rsqrt(&add(&var, &eps)?)?)?;
        let normed = normed.reshape(&[b, l, -1])?; // (1, L, 188160), dim-major/layer-minor
                                                   // zero padded token positions (multiply by the 0/1 mask == where(mask, x, 0)).
        let mask = attention_mask.reshape(&[b, l, 1])?.as_dtype(self.dtype)?;
        let normed = multiply(&normed, &mask)?;
        let rescaled = multiply(&normed, &self.rescale)?;
        linear(&rescaled, &self.aggregate_w, &self.aggregate_b)
    }
}
