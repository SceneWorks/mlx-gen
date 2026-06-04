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
//!
//! The **AudioVideo** path (sc-2684) reuses the shared Gemma hiddens + per-token-RMS `normed_hidden`
//! and adds a parallel **audio** head: `text_embedding_projection.audio_aggregate_embed` (→ 2048) +
//! `audio_embeddings_connector` (8 layers, dim 2048 = 32×64). Built only by [`from_weights_av`];
//! the video-only [`from_weights`] leaves it `None`.

use mlx_rs::ops::{add, mean_axes, multiply, rsqrt, stack_axis};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::linear;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::LtxConfig;
use crate::connector::Connector;
use crate::gemma::{GemmaConfig, GemmaModel};

const RMS_EPS: f32 = 1e-6;

/// One modality's feature-extractor head: the `aggregate_embed` Linear (188160 → out_dim) + its
/// `rescale_norm` scalar + the `Embeddings1DConnector`.
struct FeatureHead {
    aggregate_w: Array, // (out_dim, 188160)
    aggregate_b: Array, // (out_dim,)
    rescale: Array,     // √(out_dim / hidden) scalar in `dtype`
    connector: Connector,
}

impl FeatureHead {
    /// `rescale_norm(normed) → aggregate_embed → connector`. `normed` is the shared masked
    /// per-token-RMS `(1, L, 188160)`; `mask01` the `(1, L)` 1/0 attention mask.
    fn forward(&self, normed: &Array, mask01: &Array) -> Result<(Array, Array)> {
        let features = linear(
            &multiply(normed, &self.rescale)?,
            &self.aggregate_w,
            &self.aggregate_b,
        )?;
        let embeddings = self.connector.forward(&features, mask01)?;
        Ok((features, embeddings))
    }
}

/// The LTX-2.3 text encoder (Gemma backbone + per-token-RMS feature extractor + connector). Carries
/// a video head always and an optional audio head (sc-2684 AudioVideo path).
pub struct LtxTextEncoder {
    gemma: GemmaModel,
    video: FeatureHead,
    audio: Option<FeatureHead>,
    dtype: Dtype,
}

impl LtxTextEncoder {
    /// Build the **video-only** encoder from the Gemma weights + the LTX `connector.safetensors`.
    /// `dtype` = the compute dtype (bf16 to match the reference).
    pub fn from_weights(
        gemma_w: &Weights,
        connector_w: &Weights,
        gemma_cfg: GemmaConfig,
        ltx_cfg: &LtxConfig,
        dtype: Dtype,
    ) -> Result<Self> {
        let gemma = GemmaModel::from_weights(gemma_w, gemma_cfg)?;
        let video = Self::video_head(connector_w, gemma_cfg, ltx_cfg, dtype)?;
        Ok(Self {
            gemma,
            video,
            audio: None,
            dtype,
        })
    }

    /// Build the **AudioVideo** encoder (sc-2684): the video head + the audio head
    /// (`audio_aggregate_embed` + `audio_embeddings_connector`, dim 2048 = 32 × 64).
    pub fn from_weights_av(
        gemma_w: &Weights,
        connector_w: &Weights,
        gemma_cfg: GemmaConfig,
        ltx_cfg: &LtxConfig,
        dtype: Dtype,
    ) -> Result<Self> {
        let gemma = GemmaModel::from_weights(gemma_w, gemma_cfg)?;
        let video = Self::video_head(connector_w, gemma_cfg, ltx_cfg, dtype)?;
        let audio = Self::audio_head(connector_w, gemma_cfg, ltx_cfg, dtype)?;
        Ok(Self {
            gemma,
            video,
            audio: Some(audio),
            dtype,
        })
    }

    fn aggregate(
        connector_w: &Weights,
        key_prefix: &str,
        gemma_cfg: GemmaConfig,
        dtype: Dtype,
    ) -> Result<(Array, Array, Array)> {
        let load = |key: &str| -> Result<Array> {
            connector_w
                .get(key)
                .ok_or_else(|| Error::MissingTensor(key.into()))?
                .as_dtype(dtype)
                .map_err(Error::from)
        };
        let aggregate_w = load(&format!("{key_prefix}.weight"))?;
        let aggregate_b = load(&format!("{key_prefix}.bias"))?;
        let out_dim = aggregate_w.shape()[0];
        let rescale = Array::from_slice(
            &[(out_dim as f32 / gemma_cfg.hidden_size as f32).sqrt()],
            &[1],
        )
        .as_dtype(dtype)?;
        Ok((aggregate_w, aggregate_b, rescale))
    }

    fn video_head(
        connector_w: &Weights,
        gemma_cfg: GemmaConfig,
        ltx_cfg: &LtxConfig,
        dtype: Dtype,
    ) -> Result<FeatureHead> {
        let (aggregate_w, aggregate_b, rescale) = Self::aggregate(
            connector_w,
            "text_embedding_projection.video_aggregate_embed",
            gemma_cfg,
            dtype,
        )?;
        let connector =
            Connector::from_weights(connector_w, "video_embeddings_connector.", ltx_cfg, dtype)?;
        Ok(FeatureHead {
            aggregate_w,
            aggregate_b,
            rescale,
            connector,
        })
    }

    fn audio_head(
        connector_w: &Weights,
        gemma_cfg: GemmaConfig,
        ltx_cfg: &LtxConfig,
        dtype: Dtype,
    ) -> Result<FeatureHead> {
        let (aggregate_w, aggregate_b, rescale) = Self::aggregate(
            connector_w,
            "text_embedding_projection.audio_aggregate_embed",
            gemma_cfg,
            dtype,
        )?;
        // The audio connector shares the checkpoint's layer count / theta / register max-pos but
        // runs at the audio connector dims (32 × 64 = 2048).
        let connector = Connector::from_weights_dims(
            connector_w,
            "audio_embeddings_connector.",
            ltx_cfg.connector_num_layers,
            ltx_cfg.audio_connector_num_attention_heads,
            ltx_cfg.audio_connector_attention_head_dim,
            ltx_cfg.positional_embedding_theta,
            ltx_cfg.connector_positional_embedding_max_pos,
            dtype,
        )?;
        Ok(FeatureHead {
            aggregate_w,
            aggregate_b,
            rescale,
            connector,
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
        let normed = self.normed_hidden(&hiddens, attention_mask)?;
        self.video.forward(&normed, attention_mask)
    }

    /// AudioVideo encode (sc-2684): `(video_embeddings (1,L,4096), audio_embeddings (1,L,2048))`.
    /// Errors if this encoder was not built with [`from_weights_av`].
    pub fn encode_av(&self, input_ids: &Array, attention_mask: &Array) -> Result<(Array, Array)> {
        let (_, _, ve, ae) = self.encode_av_with_features(input_ids, attention_mask)?;
        Ok((ve, ae))
    }

    /// AudioVideo encode returning `(video_features, audio_features, video_embeddings,
    /// audio_embeddings)` — the pre-connector features included for stage localization.
    pub fn encode_av_with_features(
        &self,
        input_ids: &Array,
        attention_mask: &Array,
    ) -> Result<(Array, Array, Array, Array)> {
        let audio = self.audio.as_ref().ok_or_else(|| {
            Error::Msg("ltx_2_3: text encoder built without the audio head".into())
        })?;
        let hiddens = self.gemma.forward(input_ids, attention_mask)?;
        let normed = self.normed_hidden(&hiddens, attention_mask)?;
        let (vf, ve) = self.video.forward(&normed, attention_mask)?;
        let (af, ae) = audio.forward(&normed, attention_mask)?;
        Ok((vf, af, ve, ae))
    }

    /// `norm_and_concat_per_token_rms` — the shared masked per-token RMS `(1, L, 188160)`.
    /// Each `(token, layer)` slice over the 3840 hidden dim is RMS-normalized independently, the 49
    /// layers are concatenated **dim-major / layer-minor** (`d*49 + layer`, via stack+reshape), and
    /// padded positions are zeroed. The video / audio heads then rescale + aggregate this off it.
    fn normed_hidden(&self, hiddens: &[Array], attention_mask: &Array) -> Result<Array> {
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
        Ok(multiply(&normed, &mask)?)
    }
}
