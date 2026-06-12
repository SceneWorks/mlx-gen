//! SAM3 text encoder — the CLIP-H text tower (`Sam3Config.text_config`) + the SAM3
//! `text_projection` (1024→256), porting `Sam3Model.get_text_features` (epic 4910, sc-4920).
//!
//! A standard CLIP text transformer: token + learned position embeddings → 24 pre-norm layers
//! (causal **and** key-padding masked) → final LayerNorm, giving `last_hidden_state[1, N, 1024]`;
//! SAM3 then projects every token to 256 to form the prompt conditioning the DETR encoder consumes.
//! Activation is exact GELU; LayerNorm eps is **1e-5** (the vision encoder's is 1e-6). The tokenizer
//! is the shipped CLIP `tokenizer.json` (lowercased word-BPE, BOS 49406 / EOS 49407, padded to 32).

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::ops::add;
use mlx_rs::Array;

use mlx_gen::nn::{gelu_exact, linear};
use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::Sam3TextConfig;

use std::path::Path;

/// `"{prefix}.{leaf}"`.
fn join(prefix: &str, leaf: &str) -> String {
    format!("{prefix}.{leaf}")
}

/// One CLIP encoder layer: pre-norm self-attention + pre-norm GELU MLP, both residual.
struct ClipLayer {
    ln1_w: Array,
    ln1_b: Array,
    ln2_w: Array,
    ln2_b: Array,
    q_w: Array,
    q_b: Array,
    k_w: Array,
    k_b: Array,
    v_w: Array,
    v_b: Array,
    o_w: Array,
    o_b: Array,
    fc1_w: Array,
    fc1_b: Array,
    fc2_w: Array,
    fc2_b: Array,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
    eps: f32,
}

impl ClipLayer {
    fn from_weights(w: &Weights, prefix: &str, cfg: &Sam3TextConfig) -> Result<Self> {
        let g = |n: &str| -> Result<Array> { Ok(w.require(&join(prefix, n))?.clone()) };
        let head_dim = cfg.head_dim();
        Ok(Self {
            ln1_w: g("layer_norm1.weight")?,
            ln1_b: g("layer_norm1.bias")?,
            ln2_w: g("layer_norm2.weight")?,
            ln2_b: g("layer_norm2.bias")?,
            q_w: g("self_attn.q_proj.weight")?,
            q_b: g("self_attn.q_proj.bias")?,
            k_w: g("self_attn.k_proj.weight")?,
            k_b: g("self_attn.k_proj.bias")?,
            v_w: g("self_attn.v_proj.weight")?,
            v_b: g("self_attn.v_proj.bias")?,
            o_w: g("self_attn.out_proj.weight")?,
            o_b: g("self_attn.out_proj.bias")?,
            fc1_w: g("mlp.fc1.weight")?,
            fc1_b: g("mlp.fc1.bias")?,
            fc2_w: g("mlp.fc2.weight")?,
            fc2_b: g("mlp.fc2.bias")?,
            num_heads: cfg.num_attention_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            eps: cfg.layer_norm_eps,
        })
    }

    fn forward(&self, x: &Array, mask: &Array) -> Result<Array> {
        let y = layer_norm(x, Some(&self.ln1_w), Some(&self.ln1_b), self.eps)?;
        let y = self.attention(&y, mask)?;
        let x = add(x, &y)?;
        let y = layer_norm(&x, Some(&self.ln2_w), Some(&self.ln2_b), self.eps)?;
        let y = gelu_exact(&linear(&y, &self.fc1_w, &self.fc1_b)?)?;
        let y = linear(&y, &self.fc2_w, &self.fc2_b)?;
        Ok(add(&x, &y)?)
    }

    fn attention(&self, x: &Array, mask: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, n) = (sh[0], sh[1]);
        let to_heads = |a: Array| -> Result<Array> {
            Ok(a.reshape(&[b, n, self.num_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = to_heads(linear(x, &self.q_w, &self.q_b)?)?;
        let k = to_heads(linear(x, &self.k_w, &self.k_b)?)?;
        let v = to_heads(linear(x, &self.v_w, &self.v_b)?)?;
        let o = scaled_dot_product_attention(&q, &k, &v, self.scale, mask, None)?;
        let o =
            o.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, n, self.num_heads * self.head_dim])?;
        linear(&o, &self.o_w, &self.o_b)
    }
}

/// SAM3 text encoder: CLIP text tower → final LayerNorm → 1024→256 projection.
pub struct Sam3TextEncoder {
    token_embedding: Array,
    position_embedding: Array,
    layers: Vec<ClipLayer>,
    final_ln_w: Array,
    final_ln_b: Array,
    proj_w: Array,
    proj_b: Array,
    eps: f32,
}

impl Sam3TextEncoder {
    /// Load from a `facebook/sam3` weight map. `clip_prefix` is typically
    /// `"detector_model.text_encoder.text_model"`; `proj_prefix` is `"detector_model.text_projection"`.
    pub fn from_weights(
        w: &Weights,
        clip_prefix: &str,
        proj_prefix: &str,
        cfg: &Sam3TextConfig,
    ) -> Result<Self> {
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                ClipLayer::from_weights(w, &join(clip_prefix, &format!("encoder.layers.{i}")), cfg)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            token_embedding: w
                .require(&join(clip_prefix, "embeddings.token_embedding.weight"))?
                .clone(),
            position_embedding: w
                .require(&join(clip_prefix, "embeddings.position_embedding.weight"))?
                .clone(),
            layers,
            final_ln_w: w
                .require(&join(clip_prefix, "final_layer_norm.weight"))?
                .clone(),
            final_ln_b: w
                .require(&join(clip_prefix, "final_layer_norm.bias"))?
                .clone(),
            proj_w: w.require(&join(proj_prefix, "weight"))?.clone(),
            proj_b: w.require(&join(proj_prefix, "bias"))?.clone(),
            eps: cfg.layer_norm_eps,
        })
    }

    /// Encode `input_ids` `[1, N]` (int32) with a key-padding `attention_mask` (`1` = real token).
    /// Returns the projected text features `[1, N, 256]` (the DETR-stack conditioning).
    pub fn forward(&self, input_ids: &Array, attention_mask: &[i32]) -> Result<Array> {
        let sh = input_ids.shape();
        let (b, n) = (sh[0], sh[1]);
        let dim = self.token_embedding.shape()[1];

        // token + position embeddings
        let ids_flat = input_ids.reshape(&[b * n])?;
        let tok = self
            .token_embedding
            .take_axis(&ids_flat, 0)?
            .reshape(&[b, n, dim])?;
        let pos_idx = Array::from_slice(&(0..n).collect::<Vec<i32>>(), &[n]);
        let pos = self.position_embedding.take_axis(&pos_idx, 0)?; // [N, D]
        let mut x = add(&tok, &pos.reshape(&[1, n, dim])?)?;

        let mask = causal_padding_mask(n, attention_mask);
        for layer in &self.layers {
            x = layer.forward(&x, &mask)?;
        }
        let last_hidden_state =
            layer_norm(&x, Some(&self.final_ln_w), Some(&self.final_ln_b), self.eps)?;
        // SAM3 projection: every token 1024 → 256.
        linear(&last_hidden_state, &self.proj_w, &self.proj_b)
    }
}

/// Additive attention mask `[1, 1, N, N]` (f32): position `i` may attend to key `j` iff `j <= i`
/// (causal) **and** `attention_mask[j] == 1` (key-padding); otherwise `-1e9`. Matches HF
/// `CLIPTextTransformer` combining its causal mask with the passed `attention_mask`.
fn causal_padding_mask(n: i32, attention_mask: &[i32]) -> Array {
    let nu = n as usize;
    let mut m = vec![0f32; nu * nu];
    for i in 0..nu {
        for (j, slot) in m[i * nu..(i + 1) * nu].iter_mut().enumerate() {
            let padded = attention_mask.get(j).copied().unwrap_or(1) == 0;
            if j > i || padded {
                *slot = -1.0e9;
            }
        }
    }
    Array::from_slice(&m, &[1, 1, n, n])
}

/// CLIP tokenizer for SAM3 concept prompts — the shipped `tokenizer.json` (lowercased word-BPE,
/// BOS 49406 / EOS 49407), padded to `max_position_embeddings` (32) with the EOS/pad token.
pub struct Sam3Tokenizer {
    inner: TextTokenizer,
}

impl Sam3Tokenizer {
    /// Load from the `facebook/sam3` `tokenizer.json`.
    pub fn from_file(tokenizer_json: impl AsRef<Path>, cfg: &Sam3TextConfig) -> Result<Self> {
        let config = TokenizerConfig {
            max_length: cfg.max_position_embeddings as usize,
            pad_token_id: cfg.pad_token_id,
            chat_template: ChatTemplate::None,
            pad_to_max_length: true,
        };
        Ok(Self {
            inner: TextTokenizer::from_file(tokenizer_json, config)?,
        })
    }

    /// Tokenize a concept phrase → `input_ids[1, 32]` (int32) + the key-padding `attention_mask`.
    pub fn encode(&self, text: &str) -> Result<(Array, Vec<i32>)> {
        let out = self.inner.tokenize(text)?;
        let n = out.ids.len() as i32;
        Ok((Array::from_slice(&out.ids, &[1, n]), out.mask))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_padding_mask_blocks_future_and_padding() {
        // N=3, last token padded → key 2 blocked everywhere; upper triangle blocked.
        let m = causal_padding_mask(3, &[1, 1, 0]);
        let s = m.reshape(&[9]).unwrap();
        let v = s.as_slice::<f32>();
        assert_eq!(v[0], 0.0); // (0,0) ok
        assert_eq!(v[1], -1e9); // (0,1) future
        assert_eq!(v[3], 0.0); // (1,0) ok
        assert_eq!(v[4], 0.0); // (1,1) ok
        assert_eq!(v[2], -1e9); // (0,2) future+pad
        assert_eq!(v[8], -1e9); // (2,2) padded key
        assert_eq!(v[6], 0.0); // (2,0) ok
    }
}
