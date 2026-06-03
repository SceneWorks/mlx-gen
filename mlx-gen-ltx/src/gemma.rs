//! Gemma-3-12B-IT language-model forward — the LTX-2.3 text encoder's backbone (S1).
//!
//! Port of `mlx_vlm/models/gemma3/language.py` (`Gemma3Model`) as driven by the LTX
//! `LanguageModel` wrapper (`text_encoder.py`): 48 decoder layers, hidden 3840, 16 query / 8 KV
//! heads (GQA), head_dim 256, intermediate 15360. Key Gemma specifics:
//! - **RMSNorm uses `(1 + weight)`** (`fast.rms_norm(x, 1+w, eps)`), eps 1e-6.
//! - Token embeddings scaled by **√hidden_size** (computed in bf16, matching the reference).
//! - **Per-layer RoPE base**: local 1e4 on sliding layers `(i+1) % 6 != 0`, global 1e6 otherwise
//!   (via `fast::rope`, the same op the reference's `nn.RoPE` wraps; `rope_scaling` is in the HF
//!   config but the reference does NOT apply it, so we match by ignoring it).
//! - **q/k RMSNorm over head_dim** (256), applied post-reshape.
//! - attention scale = `query_pre_attn_scalar^-0.5` (= 256^-0.5).
//! - MLP = `down(gelu_approx(gate(x)) * up(x))`.
//! - norm-sandwich block: input_ln → attn → post_attn_ln → +res → pre_ff_ln → mlp → post_ff_ln → +res.
//!
//! Runs **bf16** to match the reference (the gemma-3-12b-it-bf16 checkpoint + bf16 activations);
//! all GEMMs have K>512 so the pmetal bf16-GEMM regime doesn't apply (and sc-2714 fixed it anyway).
//!
//! [`forward`](GemmaModel::forward) returns the **49 hidden states** the LTX feature extractor
//! consumes: `[embed·√d] + layers[0..46] outputs + norm(layer47 output)`. For sequence lengths
//! ≤ sliding_window (1024) the sliding mask equals the full causal+padding mask, so a single
//! additive causal+padding mask is used for all layers (only the RoPE base differs per layer).

use mlx_rs::fast::{rms_norm, rope, scaled_dot_product_attention};
use mlx_rs::nn::gelu_approximate;
use mlx_rs::ops::{add, matmul, multiply};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

/// Gemma-3 text-config (the gemma-3-12b-it values).
#[derive(Clone, Copy, Debug)]
pub struct GemmaConfig {
    pub hidden_size: i32,
    pub num_layers: usize,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub intermediate: i32,
    pub rms_eps: f32,
    pub query_pre_attn_scalar: f32,
    pub rope_local_base: f32,
    pub rope_global_base: f32,
    pub sliding_window_pattern: usize,
}

impl GemmaConfig {
    pub fn gemma_3_12b() -> Self {
        Self {
            hidden_size: 3840,
            num_layers: 48,
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 256,
            intermediate: 15360,
            rms_eps: 1e-6,
            query_pre_attn_scalar: 256.0,
            rope_local_base: 10_000.0,
            rope_global_base: 1_000_000.0,
            sliding_window_pattern: 6,
        }
    }
}

struct GemmaLayer {
    input_ln: Array,     // (1 + weight)
    post_attn_ln: Array, // (1 + weight)
    pre_ff_ln: Array,    // (1 + weight)
    post_ff_ln: Array,   // (1 + weight)
    q_proj: Array,
    k_proj: Array,
    v_proj: Array,
    o_proj: Array,
    q_norm: Array, // (1 + weight), head_dim
    k_norm: Array, // (1 + weight), head_dim
    gate_proj: Array,
    up_proj: Array,
    down_proj: Array,
    rope_base: f32,
}

/// The Gemma-3 backbone used as the LTX text encoder.
pub struct GemmaModel {
    embed: Array, // (vocab, hidden) bf16
    layers: Vec<GemmaLayer>,
    norm: Array, // (1 + weight)
    cfg: GemmaConfig,
    embed_scale: Array, // √hidden_size as a bf16 scalar
}

/// `y = x · Wᵀ` for a stored `[out, in]` weight, no bias (Gemma's projections).
fn linear_nb(x: &Array, w: &Array) -> Result<Array> {
    Ok(matmul(x, w.t())?)
}

impl GemmaModel {
    /// Build from a `Weights` map holding the `language_model.model.*` Gemma tensors (bf16).
    pub fn from_weights(w: &Weights, cfg: GemmaConfig) -> Result<Self> {
        let p = "language_model.model.";
        let get = |key: &str| -> Result<Array> {
            w.get(key)
                .ok_or_else(|| Error::MissingTensor(key.into()))?
                .as_dtype(Dtype::Bfloat16)
                .map_err(Error::from)
        };
        // RMSNorm weight + 1.0 (Gemma scales by 1+w), kept bf16.
        let norm_w = |key: &str| -> Result<Array> {
            Ok(add(
                &get(key)?,
                &Array::from_slice(&[1.0f32], &[1]).as_dtype(Dtype::Bfloat16)?,
            )?)
        };

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let b = format!("{p}layers.{i}.");
            let is_sliding = (i + 1) % cfg.sliding_window_pattern != 0;
            layers.push(GemmaLayer {
                input_ln: norm_w(&format!("{b}input_layernorm.weight"))?,
                post_attn_ln: norm_w(&format!("{b}post_attention_layernorm.weight"))?,
                pre_ff_ln: norm_w(&format!("{b}pre_feedforward_layernorm.weight"))?,
                post_ff_ln: norm_w(&format!("{b}post_feedforward_layernorm.weight"))?,
                q_proj: get(&format!("{b}self_attn.q_proj.weight"))?,
                k_proj: get(&format!("{b}self_attn.k_proj.weight"))?,
                v_proj: get(&format!("{b}self_attn.v_proj.weight"))?,
                o_proj: get(&format!("{b}self_attn.o_proj.weight"))?,
                q_norm: norm_w(&format!("{b}self_attn.q_norm.weight"))?,
                k_norm: norm_w(&format!("{b}self_attn.k_norm.weight"))?,
                gate_proj: get(&format!("{b}mlp.gate_proj.weight"))?,
                up_proj: get(&format!("{b}mlp.up_proj.weight"))?,
                down_proj: get(&format!("{b}mlp.down_proj.weight"))?,
                rope_base: if is_sliding {
                    cfg.rope_local_base
                } else {
                    cfg.rope_global_base
                },
            });
        }

        // Embedding scale = √hidden_size, rounded to bf16 like the reference.
        let embed_scale = Array::from_slice(&[(cfg.hidden_size as f32).sqrt()], &[1])
            .as_dtype(Dtype::Bfloat16)?;

        Ok(Self {
            embed: get(&format!("{p}embed_tokens.weight"))?,
            layers,
            norm: norm_w(&format!("{p}norm.weight"))?,
            cfg,
            embed_scale,
        })
    }

    /// Additive causal + left-padding mask `(1, 1, L, L)` in bf16. `valid(i,j) = j<=i && mask01[j]`.
    fn causal_padding_mask(&self, mask01: &Array, l: usize) -> Result<Array> {
        let m = mask01.as_slice::<i32>(); // (1, L)
        let neg = half_min_bf16();
        let mut data = vec![0f32; l * l];
        for i in 0..l {
            for j in 0..l {
                let valid = j <= i && m[j] != 0;
                data[i * l + j] = if valid { 0.0 } else { neg };
            }
        }
        Array::from_slice(&data, &[1, 1, l as i32, l as i32])
            .as_dtype(Dtype::Bfloat16)
            .map_err(Error::from)
    }

    fn attn(&self, layer: &GemmaLayer, x: &Array, mask: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, l) = (sh[0], sh[1]);
        let (h, kv, d) = (self.cfg.num_heads, self.cfg.num_kv_heads, self.cfg.head_dim);
        let q = linear_nb(x, &layer.q_proj)?
            .reshape(&[b, l, h, d])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = linear_nb(x, &layer.k_proj)?
            .reshape(&[b, l, kv, d])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = linear_nb(x, &layer.v_proj)?
            .reshape(&[b, l, kv, d])?
            .transpose_axes(&[0, 2, 1, 3])?;
        // q/k RMSNorm over head_dim, then RoPE (per-layer base).
        let q = rms_norm(&q, &layer.q_norm, self.cfg.rms_eps)?;
        let k = rms_norm(&k, &layer.k_norm, self.cfg.rms_eps)?;
        let q = rope(&q, d, false, Some(layer.rope_base), 1.0, 0, None)?;
        let k = rope(&k, d, false, Some(layer.rope_base), 1.0, 0, None)?;
        let scale = self.cfg.query_pre_attn_scalar.powf(-0.5);
        let out = scaled_dot_product_attention(&q, &k, &v, scale, mask, None)?; // GQA-aware
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, l, h * d])?;
        linear_nb(&out, &layer.o_proj)
    }

    fn mlp(&self, layer: &GemmaLayer, x: &Array) -> Result<Array> {
        let gate = gelu_approximate(&linear_nb(x, &layer.gate_proj)?)?;
        let up = linear_nb(x, &layer.up_proj)?;
        linear_nb(&multiply(&gate, &up)?, &layer.down_proj)
    }

    fn layer_forward(&self, layer: &GemmaLayer, x: &Array, mask: &Array) -> Result<Array> {
        let r = self.attn(
            layer,
            &rms_norm(x, &layer.input_ln, self.cfg.rms_eps)?,
            mask,
        )?;
        let h = add(x, &rms_norm(&r, &layer.post_attn_ln, self.cfg.rms_eps)?)?;
        let r = self.mlp(layer, &rms_norm(&h, &layer.pre_ff_ln, self.cfg.rms_eps)?)?;
        Ok(add(
            &h,
            &rms_norm(&r, &layer.post_ff_ln, self.cfg.rms_eps)?,
        )?)
    }

    /// Run the Gemma forward, returning the **49 hidden states** the LTX feature extractor consumes.
    /// `input_ids` and `mask01` are `(1, L)` (i32); `mask01` is 1 for valid tokens (left-padded).
    pub fn forward(&self, input_ids: &Array, mask01: &Array) -> Result<Vec<Array>> {
        let sh = input_ids.shape();
        let (b, l) = (sh[0], sh[1]);
        let ids = input_ids.reshape(&[-1])?;
        let mut h = self
            .embed
            .take_axis(&ids, 0)?
            .reshape(&[b, l, self.cfg.hidden_size])?;
        h = multiply(&h, &self.embed_scale)?;

        let mask = self.causal_padding_mask(mask01, l as usize)?;
        let mut hiddens = Vec::with_capacity(self.cfg.num_layers + 1);
        hiddens.push(h.clone()); // hidden state 0 = scaled embedding
        for (i, layer) in self.layers.iter().enumerate() {
            h = self.layer_forward(layer, &h, &mask)?;
            if i < self.cfg.num_layers - 1 {
                hiddens.push(h.clone());
            }
        }
        hiddens.push(rms_norm(&h, &self.norm, self.cfg.rms_eps)?); // final norm = 49th state
        Ok(hiddens)
    }
}

/// bf16 smallest (most-negative) finite value, as f32 — matches `mx.finfo(bfloat16).min`.
fn half_min_bf16() -> f32 {
    // bf16 max magnitude = (2 - 2^-7) * 2^127 ≈ 3.3895314e38.
    -3.389_531_4e38
}
