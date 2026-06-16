//! FLUX.1 prompt text path: CLIP pooled prompt embedding + T5 sequence prompt embedding.
//! Ports the fork's `flux_text_encoder` modules directly.

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::array::{host_i32, scalar};
use mlx_gen::nn::gelu_tanh;
use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_rs::fast::{layer_norm, scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::ops::{add, dequantize, matmul, multiply, power, quantize, sigmoid, softmax_axis};
use mlx_rs::{Array, Dtype};

pub struct FluxTextEncoders {
    pub t5: T5TextEncoder,
    pub clip: ClipTextEncoder,
}

impl FluxTextEncoders {
    pub fn encode(&self, t5_ids: &Array, clip_ids: &Array) -> Result<(Array, Array)> {
        Ok((self.t5.forward(t5_ids)?, self.clip.forward(clip_ids)?))
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.t5.quantize(bits)?;
        self.clip.quantize(bits)?;
        Ok(())
    }
}

enum TokenEmbedding {
    Dense(Array),
    Quantized {
        wq: Array,
        scales: Array,
        biases: Array,
        group_size: i32,
        bits: i32,
    },
}

impl TokenEmbedding {
    fn dense(weight: Array) -> Self {
        Self::Dense(weight)
    }

    fn forward(&self, ids: &Array) -> Result<Array> {
        let out = match self {
            Self::Dense(w) => w.take_axis(ids, 0)?,
            Self::Quantized {
                wq,
                scales,
                biases,
                group_size,
                bits,
            } => {
                let pw = wq.take_axis(ids, 0)?;
                let sc = scales.take_axis(ids, 0)?;
                let bi = biases.take_axis(ids, 0)?;
                dequantize(&pw, &sc, &bi, *group_size, *bits)?
            }
        };
        // Return the native (bf16) embedding to match the mflux reference (sc-2787). CLIP genuinely
        // runs bf16 (its `nn.LayerNorm` fast kernel returns bf16); T5 immediately upcasts to f32 in
        // its `T5LayerNorm` (variance `astype(f32)`, which MLX promotion propagates through the whole
        // encoder), so T5 stays f32-internally either way — the FLUX checkpoint is bf16-native, so the
        // bf16↔f32 cast is lossless here. The old MANDATORY-f32 comment was bug-forced: T5/CLIP
        // attention is bf16×bf16 K≤512 (the [[pmetal-mlx-bf16-matmul-bug]] dense 16-bit GEMM), now
        // fixed by sc-2772 (NAX metal target ≥26.2) — so bf16 is correct AND the parity dtype.
        Ok(out)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        if let Self::Dense(w) = self {
            let (wq, scales, biases) = quantize(&w.as_dtype(Dtype::Bfloat16)?, 64, bits)?;
            *self = Self::Quantized {
                wq,
                scales,
                biases,
                group_size: 64,
                bits,
            };
        }
        Ok(())
    }
}

pub struct ClipTextEncoder {
    token_embedding: TokenEmbedding,
    position_embedding: TokenEmbedding,
    layers: Vec<ClipEncoderLayer>,
    final_ln_w: Array,
    final_ln_b: Array,
}

impl ClipTextEncoder {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let p = |suffix: &str| join(prefix, suffix);
        let mut layers = Vec::with_capacity(12);
        for i in 0..12 {
            layers.push(ClipEncoderLayer::from_weights(
                w,
                &p(&format!("text_model.encoder.layers.{i}")),
            )?);
        }
        Ok(Self {
            token_embedding: TokenEmbedding::dense(
                w.require(&p("text_model.embeddings.token_embedding.weight"))?
                    .clone(),
            ),
            position_embedding: TokenEmbedding::dense(
                w.require(&p("text_model.embeddings.position_embedding.weight"))?
                    .clone(),
            ),
            layers,
            final_ln_w: w.require(&p("text_model.final_layer_norm.weight"))?.clone(),
            final_ln_b: w.require(&p("text_model.final_layer_norm.bias"))?.clone(),
        })
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.token_embedding.quantize(bits)?;
        self.position_embedding.quantize(bits)?;
        for layer in &mut self.layers {
            layer.quantize(bits)?;
        }
        Ok(())
    }

    /// `tokens`: `[1, 77]` int32. Returns pooled CLIP embedding `[1, 768]`, selected at the
    /// highest token id (the fork's `mx.argmax(tokens, axis=-1)`).
    pub fn forward(&self, tokens: &Array) -> Result<Array> {
        let s = tokens.shape()[1];
        let token = self.token_embedding.forward(tokens)?;
        let pos_ids: Vec<i32> = (0..s).collect();
        let pos_ids = Array::from_slice(&pos_ids, &[1, s]);
        let pos = self.position_embedding.forward(&pos_ids)?;
        let mut hidden = add(&token, &pos)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        let hidden = layer_norm(
            &hidden,
            Some(&self.final_ln_w),
            Some(&self.final_ln_b),
            1e-5,
        )?;
        let token_ids = host_i32(tokens)?;
        // Pooled output is the hidden state at the *first* argmax of the token ids — the fork's
        // `mx.argmax(tokens, axis=-1)` (first occurrence on ties). CLIP pads to 77 with the EOS id
        // (49407), so the EOS and every pad token tie; `Iterator::max_by_key` would return the
        // LAST tie (a pad position) instead of the EOS, picking the wrong pooled vector.
        let max_id = token_ids.iter().copied().max().unwrap_or(0);
        let idx = token_ids.iter().position(|&id| id == max_id).unwrap_or(0) as i32;
        let flat = hidden.reshape(&[s, 768])?;
        let idx = Array::from_slice(&[idx], &[1]);
        Ok(flat.take_axis(&idx, 0)?)
    }
}

struct ClipEncoderLayer {
    ln1_w: Array,
    ln1_b: Array,
    attn: ClipAttention,
    ln2_w: Array,
    ln2_b: Array,
    mlp: ClipMlp,
}

impl ClipEncoderLayer {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            ln1_w: w.require(&join(prefix, "layer_norm1.weight"))?.clone(),
            ln1_b: w.require(&join(prefix, "layer_norm1.bias"))?.clone(),
            attn: ClipAttention::from_weights(w, &join(prefix, "self_attn"))?,
            ln2_w: w.require(&join(prefix, "layer_norm2.weight"))?.clone(),
            ln2_b: w.require(&join(prefix, "layer_norm2.bias"))?.clone(),
            mlp: ClipMlp::from_weights(w, &join(prefix, "mlp"))?,
        })
    }

    fn forward(&self, hidden: &Array) -> Result<Array> {
        let residual = hidden;
        let normed = layer_norm(hidden, Some(&self.ln1_w), Some(&self.ln1_b), 1e-5)?;
        let hidden = add(residual, &self.attn.forward(&normed)?)?;
        let residual = hidden.clone();
        let normed = layer_norm(&hidden, Some(&self.ln2_w), Some(&self.ln2_b), 1e-5)?;
        Ok(add(&residual, &self.mlp.forward(&normed)?)?)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attn.quantize(bits)?;
        self.mlp.quantize(bits)?;
        Ok(())
    }
}

struct ClipAttention {
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    out: AdaptableLinear,
}

impl ClipAttention {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let linear = |name: &str| -> Result<AdaptableLinear> {
            Ok(AdaptableLinear::dense(
                w.require(&join(prefix, &format!("{name}.weight")))?.clone(),
                Some(w.require(&join(prefix, &format!("{name}.bias")))?.clone()),
            ))
        };
        Ok(Self {
            q: linear("q_proj")?,
            k: linear("k_proj")?,
            v: linear("v_proj")?,
            out: linear("out_proj")?,
        })
    }

    fn forward(&self, hidden: &Array) -> Result<Array> {
        let b = hidden.shape()[0];
        let s = hidden.shape()[1];
        // Read the batch from the input instead of hardcoding 1, so a B>1 CLIP encode reshapes
        // correctly rather than shape-erroring / mis-shaping (F-061).
        let q = self
            .q
            .forward(hidden)?
            .reshape(&[b, s, 12, 64])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = self
            .k
            .forward(hidden)?
            .reshape(&[b, s, 12, 64])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = self
            .v
            .forward(hidden)?
            .reshape(&[b, s, 12, 64])?
            .transpose_axes(&[0, 2, 1, 3])?;
        // CLIP text attention is purely causal (no key-padding term — pads are attended causally),
        // so use the implicit causal mode instead of materializing an `s·s` additive mask host-side
        // each encode (F-040). q_len == k_len here, so the modes are equivalent.
        let y = scaled_dot_product_attention(
            &q,
            &k,
            &v,
            (64.0_f32).powf(-0.5),
            ScaledDotProductAttentionMask::Causal,
            None,
        )?;
        let y = y.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, 768])?;
        self.out.forward(&y)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.q.quantize(bits, None)?;
        self.k.quantize(bits, None)?;
        self.v.quantize(bits, None)?;
        self.out.quantize(bits, None)?;
        Ok(())
    }
}

struct ClipMlp {
    fc1: AdaptableLinear,
    fc2: AdaptableLinear,
}

impl ClipMlp {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let linear = |name: &str| -> Result<AdaptableLinear> {
            Ok(AdaptableLinear::dense(
                w.require(&join(prefix, &format!("{name}.weight")))?.clone(),
                Some(w.require(&join(prefix, &format!("{name}.bias")))?.clone()),
            ))
        };
        Ok(Self {
            fc1: linear("fc1")?,
            fc2: linear("fc2")?,
        })
    }

    fn forward(&self, hidden: &Array) -> Result<Array> {
        let hidden = self.fc1.forward(hidden)?;
        let hidden = quick_gelu(&hidden)?;
        self.fc2.forward(&hidden)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.fc1.quantize(bits, None)?;
        self.fc2.quantize(bits, None)?;
        Ok(())
    }
}

pub struct T5TextEncoder {
    shared: TokenEmbedding,
    blocks: Vec<T5Block>,
    final_ln_w: Array,
}

impl T5TextEncoder {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let p = |suffix: &str| join(prefix, suffix);
        let mut blocks = Vec::with_capacity(24);
        for i in 0..24 {
            blocks.push(T5Block::from_weights(w, &p(&format!("encoder.block.{i}")))?);
        }
        Ok(Self {
            shared: TokenEmbedding::dense(w.require(&p("shared.weight"))?.clone()),
            blocks,
            final_ln_w: w.require(&p("encoder.final_layer_norm.weight"))?.clone(),
        })
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.shared.quantize(bits)?;
        for block in &mut self.blocks {
            block.quantize(bits)?;
        }
        Ok(())
    }

    /// `tokens`: `[1, L]` int32. Returns T5 sequence embeddings `[1, L, 4096]`.
    pub fn forward(&self, tokens: &Array) -> Result<Array> {
        self.forward_masked(tokens, None)
    }

    /// As [`forward`](Self::forward), but with an optional **additive** key-padding mask (broadcastable
    /// to the attention scores `[1, heads, L, L]`, e.g. `[1, 1, 1, L]` with a large negative at padded
    /// keys). Chroma (epic 3531) runs T5 with the tokenizer padding mask — unlike FLUX, which runs T5
    /// unmasked. `mask = None` is **byte-identical** to [`forward`](Self::forward).
    pub fn forward_masked(&self, tokens: &Array, mask: Option<&Array>) -> Result<Array> {
        let mut hidden = self.shared.forward(tokens)?;
        // The relative-position bias depends only on seq_len and is identical across all blocks (only
        // block 0 carries the table; every other block clones it), so compute it once here and share
        // it instead of rebuilding the O(L²) gather inside each of the 24 blocks (F-099).
        if let Some(block0) = self.blocks.first() {
            let bias = block0.attn.position_bias(hidden.shape()[1])?;
            for block in &self.blocks {
                hidden = block.forward(&hidden, mask, &bias)?;
            }
        }
        t5_rms_norm(&hidden, &self.final_ln_w, 1e-6)
    }
}

struct T5Block {
    attn: T5Attention,
    ff: T5FeedForward,
}

impl T5Block {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            attn: T5Attention::from_weights(w, &join(prefix, "layer.0"))?,
            ff: T5FeedForward::from_weights(w, &join(prefix, "layer.1"))?,
        })
    }

    fn forward(&self, hidden: &Array, mask: Option<&Array>, bias: &Array) -> Result<Array> {
        let hidden = self.attn.forward(hidden, mask, bias)?;
        self.ff.forward(&hidden)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attn.quantize(bits)?;
        self.ff.quantize(bits)?;
        Ok(())
    }
}

struct T5Attention {
    ln_w: Array,
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    o: AdaptableLinear,
    rel_bias: TokenEmbedding,
}

impl T5Attention {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let linear = |name: &str| -> Result<AdaptableLinear> {
            Ok(AdaptableLinear::dense(
                w.require(&join(prefix, &format!("SelfAttention.{name}.weight")))?
                    .clone(),
                None,
            ))
        };
        Ok(Self {
            ln_w: w.require(&join(prefix, "layer_norm.weight"))?.clone(),
            q: linear("q")?,
            k: linear("k")?,
            v: linear("v")?,
            o: linear("o")?,
            rel_bias: TokenEmbedding::dense(
                w.require(&join(
                    prefix,
                    "SelfAttention.relative_attention_bias.weight",
                ))
                .or_else(|_| {
                    w.require(
                        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
                    )
                })?
                .clone(),
            ),
        })
    }

    /// `bias` is the shared relative-position bias for this seq_len, precomputed once per forward in
    /// [`T5TextEncoder::forward_masked`] (it is identical across all blocks — F-099).
    fn forward(&self, hidden: &Array, mask: Option<&Array>, bias: &Array) -> Result<Array> {
        let normed = t5_rms_norm(hidden, &self.ln_w, 1e-6)?;
        let q = shape_t5(&self.q.forward(&normed)?)?;
        let k = shape_t5(&self.k.forward(&normed)?)?;
        let v = shape_t5(&self.v.forward(&normed)?)?;
        let scores = matmul(&q, &k.transpose_axes(&[0, 1, 3, 2])?)?;
        // Chroma key-padding mask (epic 3531): additive, broadcast over query/heads. `None` for FLUX.
        let biased = match mask {
            Some(m) => add(&add(&scores, bias)?, m)?,
            None => add(&scores, bias)?,
        };
        let weights = softmax_axis(&biased, -1, false)?;
        let attn = unshape_t5(&matmul(&weights, &v)?)?;
        Ok(add(hidden, &self.o.forward(&attn)?)?)
    }

    fn position_bias(&self, seq_len: i32) -> Result<Array> {
        let buckets = relative_position_buckets(seq_len);
        let idx = Array::from_slice(&buckets, &[seq_len, seq_len]);
        let values = self.rel_bias.forward(&idx)?;
        Ok(values.transpose_axes(&[2, 0, 1])?.expand_dims(0)?)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.q.quantize(bits, None)?;
        self.k.quantize(bits, None)?;
        self.v.quantize(bits, None)?;
        self.o.quantize(bits, None)?;
        self.rel_bias.quantize(bits)?;
        Ok(())
    }
}

struct T5FeedForward {
    ln_w: Array,
    wi0: AdaptableLinear,
    wi1: AdaptableLinear,
    wo: AdaptableLinear,
}

impl T5FeedForward {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let linear = |name: &str| -> Result<AdaptableLinear> {
            Ok(AdaptableLinear::dense(
                w.require(&join(prefix, &format!("DenseReluDense.{name}.weight")))?
                    .clone(),
                None,
            ))
        };
        Ok(Self {
            ln_w: w.require(&join(prefix, "layer_norm.weight"))?.clone(),
            wi0: linear("wi_0")?,
            wi1: linear("wi_1")?,
            wo: linear("wo")?,
        })
    }

    fn forward(&self, hidden: &Array) -> Result<Array> {
        let normed = t5_rms_norm(hidden, &self.ln_w, 1e-6)?;
        // Shared dtype-preserving tanh-GELU (sc-2779). Replaces the local `new_gelu`, whose f32
        // `√(2/π)` constant was 1 ULP off the fork's f64-host value (see [[mlx-rs-gelu-approx-f64-constant]]);
        // `gelu_tanh` computes the constant in f64 and preserves the input dtype.
        let gelu = gelu_tanh(&self.wi0.forward(&normed)?)?;
        let linear = self.wi1.forward(&normed)?;
        let ff = self.wo.forward(&multiply(&gelu, &linear)?)?;
        Ok(add(hidden, &ff)?)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.wi0.quantize(bits, None)?;
        self.wi1.quantize(bits, None)?;
        self.wo.quantize(bits, None)?;
        Ok(())
    }
}

fn quick_gelu(x: &Array) -> Result<Array> {
    // Dtype-preserving (sc-2787): the fork's `1.702 * input_array` is a weak python scalar, so a bf16
    // input stays bf16. A strong f32 `scalar(1.702)` would promote bf16→f32 and break CLIP bf16 parity.
    let c = scalar(1.702).as_dtype(x.dtype())?;
    Ok(multiply(x, &sigmoid(&multiply(x, &c)?)?)?)
}

/// T5's `T5LayerNorm` — RMS-normalize over the last axis with NO mean subtraction.
///
/// This is deliberately the fork's hand-rolled primitive sequence (`weight * x *
/// rsqrt(mean(x^2) + eps)`), NOT `mlx_rs::fast::rms_norm`. The fused kernel differs from the fork's
/// primitives by ~1e-7 per call; T5-xxl applies it 49×, so on the wheel that grows to ~3e-3 in
/// `prompt_embeds` (this exact form is BIT-EXACT to the fork on the wheel — verified sc-2345 review,
/// 2026-06-02). On the pinned NAX build it removes the fast-vs-manual share of the T5 drift
/// (dev@512²: 2.66e-3 → 1.87e-3 mean_rel); the rest is irreducible NAX-vs-wheel f32 accumulation over
/// the 24 layers (block-0 bit-exact, grows monotonically with depth — not a code bug, the deferred
/// cross-build delta). CLIP is unaffected because it uses `LayerNorm`, whose fused kernel DOES match
/// the fork. `power(x, 2)` (not `square`) matches the fork's `mx.power(_, 2)` — they differ by 1 ULP.
fn t5_rms_norm(x: &Array, weight: &Array, eps: f32) -> Result<Array> {
    let var = power(x, Array::from_slice(&[2.0_f32], &[1]))?.mean_axis(-1, true)?;
    let normed = multiply(x, &add(&var, scalar(eps))?.rsqrt()?)?;
    Ok(multiply(weight, &normed)?)
}

fn shape_t5(x: &Array) -> Result<Array> {
    Ok(x.reshape(&[1, -1, 64, 64])?.transpose_axes(&[0, 2, 1, 3])?)
}

fn unshape_t5(x: &Array) -> Result<Array> {
    Ok(x.transpose_axes(&[0, 2, 1, 3])?.reshape(&[1, -1, 4096])?)
}

fn relative_position_buckets(seq_len: i32) -> Vec<i32> {
    let mut buckets = Vec::with_capacity((seq_len * seq_len) as usize);
    for context in 0..seq_len {
        for memory in 0..seq_len {
            let relative = memory - context;
            buckets.push(relative_position_bucket(relative));
        }
    }
    buckets
}

fn relative_position_bucket(relative_position: i32) -> i32 {
    let num_buckets = 32;
    let max_distance = 128.0_f32;
    let mut bucket = 0;
    let mut n = relative_position;
    let half = num_buckets / 2;
    if n > 0 {
        bucket += half;
    }
    n = n.abs();
    let max_exact = half / 2;
    let val = if n < max_exact {
        n
    } else {
        let n_float = n as f32;
        let log_ratio = (n_float / max_exact as f32).ln() / (max_distance / max_exact as f32).ln();
        let large = max_exact + (log_ratio * (half - max_exact) as f32).floor() as i32;
        large.min(half - 1)
    };
    bucket + val
}

fn join(prefix: &str, suffix: &str) -> String {
    if prefix.is_empty() {
        suffix.to_string()
    } else {
        format!("{prefix}.{suffix}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t5_relative_position_buckets_match_known_edges() {
        assert_eq!(relative_position_bucket(0), 0);
        assert_eq!(relative_position_bucket(1), 17);
        assert_eq!(relative_position_bucket(-1), 1);
        assert_eq!(relative_position_bucket(128), 31);
        assert_eq!(relative_position_bucket(-128), 15);
    }
}
