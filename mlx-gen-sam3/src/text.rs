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

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::gelu_exact;
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
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    o: AdaptableLinear,
    fc1: AdaptableLinear,
    fc2: AdaptableLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
    eps: f32,
}

impl ClipLayer {
    fn from_weights(w: &Weights, prefix: &str, cfg: &Sam3TextConfig) -> Result<Self> {
        let g = |n: &str| -> Result<Array> { Ok(w.require(&join(prefix, n))?.clone()) };
        let l = |n: &str| crate::load_linear(w, &join(prefix, n));
        let head_dim = cfg.head_dim();
        Ok(Self {
            ln1_w: g("layer_norm1.weight")?,
            ln1_b: g("layer_norm1.bias")?,
            ln2_w: g("layer_norm2.weight")?,
            ln2_b: g("layer_norm2.bias")?,
            q: l("self_attn.q_proj")?,
            k: l("self_attn.k_proj")?,
            v: l("self_attn.v_proj")?,
            o: l("self_attn.out_proj")?,
            fc1: l("mlp.fc1")?,
            fc2: l("mlp.fc2")?,
            num_heads: cfg.num_attention_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            eps: cfg.layer_norm_eps,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        for l in [
            &mut self.q,
            &mut self.k,
            &mut self.v,
            &mut self.o,
            &mut self.fc1,
            &mut self.fc2,
        ] {
            crate::quantize_linear(l, bits)?;
        }
        Ok(())
    }

    fn forward(&self, x: &Array, mask: &Array) -> Result<Array> {
        let y = layer_norm(x, Some(&self.ln1_w), Some(&self.ln1_b), self.eps)?;
        let y = self.attention(&y, mask)?;
        let x = add(x, &y)?;
        let y = layer_norm(&x, Some(&self.ln2_w), Some(&self.ln2_b), self.eps)?;
        let y = gelu_exact(&self.fc1.forward(&y)?)?;
        let y = self.fc2.forward(&y)?;
        Ok(add(&x, &y)?)
    }

    fn attention(&self, x: &Array, mask: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, n) = (sh[0], sh[1]);
        let to_heads = |a: Array| -> Result<Array> {
            Ok(a.reshape(&[b, n, self.num_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = to_heads(self.q.forward(x)?)?;
        let k = to_heads(self.k.forward(x)?)?;
        let v = to_heads(self.v.forward(x)?)?;
        let o = scaled_dot_product_attention(&q, &k, &v, self.scale, mask, None)?;
        let o =
            o.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, n, self.num_heads * self.head_dim])?;
        self.o.forward(&o)
    }
}

/// SAM3 text encoder: CLIP text tower → final LayerNorm → 1024→256 projection.
pub struct Sam3TextEncoder {
    token_embedding: Array,
    position_embedding: Array,
    layers: Vec<ClipLayer>,
    final_ln_w: Array,
    final_ln_b: Array,
    proj: AdaptableLinear,
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
            proj: crate::load_linear(w, proj_prefix)?,
            eps: cfg.layer_norm_eps,
        })
    }

    /// Quantize the CLIP attention/MLP projections + the 1024→256 text projection (Q8/Q4). Token
    /// and position embeddings stay dense (sc-4925).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for layer in &mut self.layers {
            layer.quantize(bits)?;
        }
        crate::quantize_linear(&mut self.proj, bits)
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
        self.proj.forward(&last_hidden_state)
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
