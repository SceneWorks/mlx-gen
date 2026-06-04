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
//! **Quant (sc-2686).** The reference `utils.apply_quantization` quantizes the LM Linears iff the
//! **Gemma snapshot's** `config.json` carries a `quantization` block (group_size/bits) — i.e. only
//! when a *quantized* Gemma snapshot (e.g. `mlx-community/gemma-3-12b-it-8bit`) is used; the default
//! `…-bf16` has no such block, so the TE stays dense. When present, each `nn.Linear` is quantized iff
//! its weights carry `.scales` **and** its output dim is divisible by 64 (the reference's class
//! predicate; every Gemma-3-12b dim is, so the skip never fires here). `embed_tokens` is dequantized
//! to a dense bf16 table at load — affine dequant is per-group, so gather-then-dequant == dequant-
//! then-gather, numerically identical to the reference's `QuantizedEmbedding`. Pass [`GemmaQuant`] to
//! [`GemmaModel::from_weights`] (`None` ⇒ the dense bf16 default).
//!
//! [`forward`](GemmaModel::forward) returns the **49 hidden states** the LTX feature extractor
//! consumes: `[embed·√d] + layers[0..46] outputs + norm(layer47 output)`. For sequence lengths
//! ≤ sliding_window (1024) the sliding mask equals the full causal+padding mask, so a single
//! additive causal+padding mask is used for all layers (only the RoPE base differs per layer).

use mlx_rs::fast::{rms_norm, rope, scaled_dot_product_attention};
use mlx_rs::nn::gelu_approximate;
use mlx_rs::ops::{add, dequantize, matmul, multiply, quantized_matmul};
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

/// Quantization geometry for the Gemma backbone, read from the **Gemma snapshot's** `config.json`
/// `quantization` block (`group_size`, `bits`) — the reference `utils.apply_quantization` source.
/// `None` for the default `gemma-3-12b-it-bf16` (no block → dense bf16 TE).
#[derive(Clone, Copy, Debug)]
pub struct GemmaQuant {
    pub group: i32,
    pub bits: i32,
}

/// A Gemma projection — dense bf16 or affine-quantized. Gemma projections are **bias-free**, so the
/// `Quant` `biases` is the affine-quant zero-point (not a Linear bias). Quantized iff a quant config
/// is present *and* the weights carry `.scales` (the reference predicate; the `÷64` skip never fires
/// for Gemma-3-12b, whose every output dim is a multiple of 64).
enum GemmaLinear {
    Dense {
        w: Array, // [out, in] bf16
    },
    Quant {
        q: Array,      // [out, in_packed] U32
        scales: Array, // [out, in/group] bf16
        biases: Array, // [out, in/group] bf16
        group: i32,
        bits: i32,
    },
}

impl GemmaLinear {
    /// Load `{key}.weight` (+ `.scales`/`.biases` when quantized) at `key`. Quantized iff `quant` is
    /// `Some` and `{key}.scales` is present.
    fn load(w: &Weights, key: &str, quant: Option<GemmaQuant>) -> Result<Self> {
        let req =
            |k: &str| -> Result<&Array> { w.get(k).ok_or_else(|| Error::MissingTensor(k.into())) };
        match (quant, w.get(&format!("{key}.scales"))) {
            (Some(qz), Some(scales)) => Ok(GemmaLinear::Quant {
                q: req(&format!("{key}.weight"))?.clone(),
                scales: scales.as_dtype(Dtype::Bfloat16)?,
                biases: req(&format!("{key}.biases"))?.as_dtype(Dtype::Bfloat16)?,
                group: qz.group,
                bits: qz.bits,
            }),
            _ => Ok(GemmaLinear::Dense {
                w: req(&format!("{key}.weight"))?.as_dtype(Dtype::Bfloat16)?,
            }),
        }
    }

    /// `y = x · Wᵀ` (no bias). Dense → `matmul(x, Wᵀ)`; quant → `quantized_matmul` (transpose, fp32
    /// accumulation), bit-identical to the reference `QuantizedLinear.__call__`.
    fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            GemmaLinear::Dense { w } => Ok(matmul(x, w.t())?),
            GemmaLinear::Quant {
                q,
                scales,
                biases,
                group,
                bits,
            } => Ok(quantized_matmul(x, q, scales, biases, true, *group, *bits)?),
        }
    }
}

struct GemmaLayer {
    input_ln: Array,     // (1 + weight)
    post_attn_ln: Array, // (1 + weight)
    pre_ff_ln: Array,    // (1 + weight)
    post_ff_ln: Array,   // (1 + weight)
    q_proj: GemmaLinear,
    k_proj: GemmaLinear,
    v_proj: GemmaLinear,
    o_proj: GemmaLinear,
    q_norm: Array, // (1 + weight), head_dim
    k_norm: Array, // (1 + weight), head_dim
    gate_proj: GemmaLinear,
    up_proj: GemmaLinear,
    down_proj: GemmaLinear,
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

impl GemmaModel {
    /// Build from a `Weights` map holding the `language_model.model.*` Gemma tensors (bf16). `quant`
    /// (the Gemma snapshot's `config.json` `quantization` block) selectively quantizes the LM Linears;
    /// `None` is the dense bf16 default.
    pub fn from_weights(w: &Weights, cfg: GemmaConfig, quant: Option<GemmaQuant>) -> Result<Self> {
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
        let lin = |key: &str| -> Result<GemmaLinear> { GemmaLinear::load(w, key, quant) };

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let b = format!("{p}layers.{i}.");
            let is_sliding = (i + 1) % cfg.sliding_window_pattern != 0;
            layers.push(GemmaLayer {
                input_ln: norm_w(&format!("{b}input_layernorm.weight"))?,
                post_attn_ln: norm_w(&format!("{b}post_attention_layernorm.weight"))?,
                pre_ff_ln: norm_w(&format!("{b}pre_feedforward_layernorm.weight"))?,
                post_ff_ln: norm_w(&format!("{b}post_feedforward_layernorm.weight"))?,
                q_proj: lin(&format!("{b}self_attn.q_proj"))?,
                k_proj: lin(&format!("{b}self_attn.k_proj"))?,
                v_proj: lin(&format!("{b}self_attn.v_proj"))?,
                o_proj: lin(&format!("{b}self_attn.o_proj"))?,
                q_norm: norm_w(&format!("{b}self_attn.q_norm.weight"))?,
                k_norm: norm_w(&format!("{b}self_attn.k_norm.weight"))?,
                gate_proj: lin(&format!("{b}mlp.gate_proj"))?,
                up_proj: lin(&format!("{b}mlp.up_proj"))?,
                down_proj: lin(&format!("{b}mlp.down_proj"))?,
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
            embed: load_embedding(w, &format!("{p}embed_tokens"), quant)?,
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
        let q = layer
            .q_proj
            .forward(x)?
            .reshape(&[b, l, h, d])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = layer
            .k_proj
            .forward(x)?
            .reshape(&[b, l, kv, d])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = layer
            .v_proj
            .forward(x)?
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
        layer.o_proj.forward(&out)
    }

    fn mlp(&self, layer: &GemmaLayer, x: &Array) -> Result<Array> {
        let gate = gelu_approximate(&layer.gate_proj.forward(x)?)?;
        let up = layer.up_proj.forward(x)?;
        layer.down_proj.forward(&multiply(&gate, &up)?)
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

/// Load the token-embedding table as a dense bf16 matrix. When the snapshot is quantized,
/// `embed_tokens` is a `QuantizedEmbedding`; dequantize the whole table to bf16 (per-group affine
/// dequant is row-independent, so dequant-then-gather == the reference's gather-then-dequant — and the
/// dense footprint matches the bf16-snapshot case). `None` / no `.scales` → load the bf16 table.
fn load_embedding(w: &Weights, key: &str, quant: Option<GemmaQuant>) -> Result<Array> {
    let req =
        |k: &str| -> Result<&Array> { w.get(k).ok_or_else(|| Error::MissingTensor(k.into())) };
    match (quant, w.get(&format!("{key}.scales"))) {
        (Some(qz), Some(scales)) => {
            let q = req(&format!("{key}.weight"))?;
            let biases = req(&format!("{key}.biases"))?;
            dequantize(q, scales, Some(biases), Some(qz.group), Some(qz.bits))?
                .as_dtype(Dtype::Bfloat16)
                .map_err(Error::from)
        }
        _ => req(&format!("{key}.weight"))?
            .as_dtype(Dtype::Bfloat16)
            .map_err(Error::from),
    }
}

/// bf16 smallest (most-negative) finite value, as f32 — matches `mx.finfo(bfloat16).min`.
fn half_min_bf16() -> f32 {
    // bf16 max magnitude = (2 - 2^-7) * 2^127 ≈ 3.3895314e38.
    -3.389_531_4e38
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{abs, max as max_op, quantize, subtract};

    /// The TE-quant consumption mechanics (sc-2686): a [`GemmaLinear::Quant`] forward reproduces the
    /// explicitly-dequantized dense forward (the same packed weights both sides, fp32 accumulation).
    /// Validates the `quantized_matmul` arg order + transpose flag + output shape without needing a
    /// quantized Gemma snapshot (the e2e TE-quant gate, which does, is surfaced on the story).
    #[test]
    fn quant_linear_matches_dequantized_dense() {
        // A Gemma-shaped projection: out=64 (÷64), in=128 (2 groups of 64) — quantizable by mlx.
        let wv: Vec<f32> = (0..64 * 128).map(|i| (i as f32 * 0.01).sin()).collect();
        let w = Array::from_slice(&wv, &[64, 128])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let (q, scales, biases) = quantize(&w, 64, 8).unwrap();

        let dense = GemmaLinear::Dense {
            w: dequantize(&q, &scales, Some(&biases), Some(64), Some(8))
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap(),
        };
        let quant = GemmaLinear::Quant {
            q,
            scales: scales.as_dtype(Dtype::Float32).unwrap(),
            biases: biases.as_dtype(Dtype::Float32).unwrap(),
            group: 64,
            bits: 8,
        };

        let xv: Vec<f32> = (0..3 * 128).map(|i| (i as f32 * 0.017).cos()).collect();
        let x = Array::from_slice(&xv, &[3, 128]);
        let yd = dense.forward(&x).unwrap();
        let yq = quant.forward(&x).unwrap();
        assert_eq!(yq.shape(), &[3, 64], "x[3,128] · wᵀ[128,64]");
        let delta = abs(subtract(&yd, &yq).unwrap()).unwrap();
        let mag = abs(&yd).unwrap();
        let diff = max_op(&delta, None).unwrap().item::<f32>();
        let denom = max_op(&mag, None).unwrap().item::<f32>().max(1e-6);
        assert!(
            diff / denom < 1e-2,
            "quant forward rel {:.3e}",
            diff / denom
        );
    }
}
