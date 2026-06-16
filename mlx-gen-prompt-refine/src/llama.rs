//! A small **config-driven** Llama-3.2 causal decoder + sampler on MLX.
//!
//! Modeled on the parity-proven JoyCaption MLX Llama path
//! (`mlx_gen::caption::joycaption::language`), but (a) reads dims / GQA / the Llama-3 `rope_scaling`
//! block from the snapshot's `config.json` instead of hardcoding Llama-3.1-8B, (b) handles **tied
//! embeddings** (Llama-3.2-1B/3B share the LM head with `embed_tokens` — no separate
//! `lm_head.weight`), and (c) drops the image projector / token-splice (this is a text-only LLM). The
//! sampler is the JoyCaption one **without** the repetition penalty, matching the candle backend's
//! plain temperature/top-p `LogitsProcessor` so both backends advertise the same behavior.

use mlx_rs::fast::{rms_norm, scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::ops::{add, broadcast_to, concatenate_axis, cos, matmul, multiply, sin, split};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

/// Llama-3.2 decoder configuration, parsed from a snapshot's `config.json`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LlamaConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_layers: usize,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_factor: f32,
    pub rope_low_freq_factor: f32,
    pub rope_high_freq_factor: f32,
    pub rope_original_context: f32,
    /// Llama-3.2-1B/3B tie `lm_head.weight` to `embed_tokens.weight`; the snapshot then omits a
    /// separate `lm_head.weight` and the LM head reuses the embedding table.
    pub tie_word_embeddings: bool,
}

impl LlamaConfig {
    /// Parse the relevant fields out of an HF Llama `config.json`. The Llama-3 `rope_scaling` block
    /// is optional: absent (or factor 1) yields standard RoPE.
    pub fn from_json(path: &std::path::Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            Error::Msg(format!(
                "prompt_refine: read config.json {}: {e}",
                path.display()
            ))
        })?;
        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| Error::Msg(format!("prompt_refine: parse config.json: {e}")))?;

        let req_i = |key: &str| -> Result<i64> {
            v.get(key)
                .and_then(|x| x.as_i64())
                .ok_or_else(|| Error::Msg(format!("prompt_refine: config.json missing `{key}`")))
        };
        let hidden_size = req_i("hidden_size")? as i32;
        let num_heads = req_i("num_attention_heads")? as i32;
        let head_dim = v
            .get("head_dim")
            .and_then(|x| x.as_i64())
            .map(|x| x as i32)
            .unwrap_or(hidden_size / num_heads);
        let rs = v.get("rope_scaling");
        let rope_f = |key: &str, default: f32| -> f32 {
            rs.and_then(|r| r.get(key))
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .unwrap_or(default)
        };

        Ok(Self {
            hidden_size,
            intermediate_size: req_i("intermediate_size")? as i32,
            num_layers: req_i("num_hidden_layers")? as usize,
            num_heads,
            num_kv_heads: v
                .get("num_key_value_heads")
                .and_then(|x| x.as_i64())
                .map(|x| x as i32)
                .unwrap_or(num_heads),
            head_dim,
            vocab_size: req_i("vocab_size")? as i32,
            rms_norm_eps: v
                .get("rms_norm_eps")
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .unwrap_or(1e-5),
            rope_theta: v
                .get("rope_theta")
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .unwrap_or(500_000.0),
            rope_factor: rope_f("factor", 1.0),
            rope_low_freq_factor: rope_f("low_freq_factor", 1.0),
            rope_high_freq_factor: rope_f("high_freq_factor", 1.0),
            rope_original_context: rope_f("original_max_position_embeddings", 8192.0),
            tie_word_embeddings: v
                .get("tie_word_embeddings")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
        })
    }
}

/// The Llama causal decoder: token embedding, N transformer layers, final norm, and an LM head
/// (tied to the embedding table when `tie_word_embeddings`).
pub struct LlamaModel {
    embed_tokens: Array,
    layers: Vec<LlamaLayer>,
    norm: Array,
    lm_head: Array,
    rope: Llama3Rope,
    cfg: LlamaConfig,
}

impl LlamaModel {
    /// Load an HF Llama snapshot. `prefix` is "" for a plain `LlamaForCausalLM` (keys
    /// `model.embed_tokens.weight`, `model.layers.{i}.*`, `model.norm.weight`, `lm_head.weight`).
    pub fn from_weights(w: &Weights, prefix: &str, cfg: LlamaConfig) -> Result<Self> {
        let model_prefix = join(prefix, "model");
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(LlamaLayer::from_weights(
                w,
                &join(&model_prefix, &format!("layers.{i}")),
                &cfg,
            )?);
        }
        let embed_tokens = req_bf16(w, &join(&model_prefix, "embed_tokens.weight"))?;
        // Tied embeddings (Llama-3.2-1B/3B): the snapshot has no `lm_head.weight`; the LM head reuses
        // the embedding table (`logits = normed @ embed_tokens.T`).
        let lm_head = if cfg.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            req_bf16(w, &join(prefix, "lm_head.weight"))?
        };
        Ok(Self {
            embed_tokens,
            layers,
            norm: req_bf16(w, &join(&model_prefix, "norm.weight"))?,
            lm_head,
            rope: Llama3Rope::new(&cfg),
            cfg,
        })
    }

    pub fn config(&self) -> &LlamaConfig {
        &self.cfg
    }

    pub fn new_cache(&self) -> LlamaKvCache {
        LlamaKvCache {
            layers: (0..self.layers.len()).map(|_| None).collect(),
        }
    }

    fn embed(&self, input_ids: &Array) -> Result<Array> {
        let sh = input_ids.shape();
        let ids = input_ids.reshape(&[-1])?;
        Ok(self
            .embed_tokens
            .take_axis(&ids, 0)?
            .reshape(&[sh[0], sh[1], self.cfg.hidden_size])?
            .as_dtype(Dtype::Bfloat16)?)
    }

    /// Run `input_ids` `[1, seq]` at absolute `offset`, append K/V to `cache`, and return logits for
    /// the **last** query position `[batch, vocab]`.
    pub fn decode_logits(
        &self,
        input_ids: &Array,
        cache: &mut LlamaKvCache,
        offset: i32,
    ) -> Result<Array> {
        let embeds = self.embed(input_ids)?;
        let sh = embeds.shape();
        let (b, q_len) = (sh[0], sh[1]);
        let (cos_t, sin_t) = self.rope.forward(q_len, offset)?;

        let mut hidden = embeds;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward_step(&hidden, &cos_t, &sin_t, cache, i)?;
        }

        let last_idx = Array::from_slice(&[q_len - 1], &[1]);
        let last = hidden
            .take_axis(&last_idx, 1)?
            .reshape(&[b, self.cfg.hidden_size])?;
        let normed = rms_norm(&last, &self.norm, self.cfg.rms_norm_eps)?;
        Ok(matmul(&normed, self.lm_head.t())?)
    }
}

struct LlamaLayer {
    input_ln: Array,
    post_ln: Array,
    attn: LlamaAttention,
    mlp: LlamaMlp,
    eps: f32,
}

impl LlamaLayer {
    fn from_weights(w: &Weights, prefix: &str, cfg: &LlamaConfig) -> Result<Self> {
        Ok(Self {
            input_ln: req_bf16(w, &join(prefix, "input_layernorm.weight"))?,
            post_ln: req_bf16(w, &join(prefix, "post_attention_layernorm.weight"))?,
            attn: LlamaAttention::from_weights(w, &join(prefix, "self_attn"), cfg)?,
            mlp: LlamaMlp::from_weights(w, &join(prefix, "mlp"))?,
            eps: cfg.rms_norm_eps,
        })
    }

    fn forward_step(
        &self,
        x: &Array,
        cos_t: &Array,
        sin_t: &Array,
        cache: &mut LlamaKvCache,
        layer_idx: usize,
    ) -> Result<Array> {
        let normed = rms_norm(x, &self.input_ln, self.eps)?;
        let h = add(
            x,
            &self
                .attn
                .forward_step(&normed, cos_t, sin_t, cache, layer_idx)?,
        )?;
        let normed2 = rms_norm(&h, &self.post_ln, self.eps)?;
        Ok(add(&h, &self.mlp.forward(&normed2)?)?)
    }
}

struct LlamaAttention {
    q_w: Array,
    k_w: Array,
    v_w: Array,
    o_w: Array,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl LlamaAttention {
    fn from_weights(w: &Weights, prefix: &str, cfg: &LlamaConfig) -> Result<Self> {
        Ok(Self {
            q_w: req_bf16(w, &join(prefix, "q_proj.weight"))?,
            k_w: req_bf16(w, &join(prefix, "k_proj.weight"))?,
            v_w: req_bf16(w, &join(prefix, "v_proj.weight"))?,
            o_w: req_bf16(w, &join(prefix, "o_proj.weight"))?,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            scale: (cfg.head_dim as f32).powf(-0.5),
        })
    }

    fn forward_step(
        &self,
        x: &Array,
        cos_t: &Array,
        sin_t: &Array,
        cache: &mut LlamaKvCache,
        layer_idx: usize,
    ) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        let q = matmul(x, self.q_w.t())?.reshape(&[b, s, self.num_heads, self.head_dim])?;
        let k = matmul(x, self.k_w.t())?.reshape(&[b, s, self.num_kv_heads, self.head_dim])?;
        let v = matmul(x, self.v_w.t())?.reshape(&[b, s, self.num_kv_heads, self.head_dim])?;

        let q = apply_rope(&q, cos_t, sin_t)?;
        let k = apply_rope(&k, cos_t, sin_t)?;

        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;
        let (k_all, v_all) = cache.append(layer_idx, k, v)?;

        let groups = self.num_heads / self.num_kv_heads;
        let k_all = repeat_kv_cache(&k_all, groups)?;
        let v_all = repeat_kv_cache(&v_all, groups)?;
        // Implicit causal decode mask: MLX aligns the `q_len` queries to the last positions of the
        // `k_len` cached keys, reproducing the old host-built `decode_mask` exactly (F-040) while
        // dropping a per-step host→device transfer in the autoregressive loop.
        let out = scaled_dot_product_attention(
            &q,
            &k_all,
            &v_all,
            self.scale,
            ScaledDotProductAttentionMask::Causal,
            None,
        )?;
        let out =
            out.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, s, self.num_heads * self.head_dim])?;
        Ok(matmul(&out, self.o_w.t())?)
    }
}

struct LlamaMlp {
    gate_w: Array,
    up_w: Array,
    down_w: Array,
}

impl LlamaMlp {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate_w: req_bf16(w, &join(prefix, "gate_proj.weight"))?,
            up_w: req_bf16(w, &join(prefix, "up_proj.weight"))?,
            down_w: req_bf16(w, &join(prefix, "down_proj.weight"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let gate = silu(&matmul(x, self.gate_w.t())?)?;
        let up = matmul(x, self.up_w.t())?;
        Ok(matmul(&multiply(&gate, &up)?, self.down_w.t())?)
    }
}

/// Per-layer growable K/V cache (immutable-array MLX: each step `concat`s the new K/V onto the prior).
pub struct LlamaKvCache {
    layers: Vec<Option<(Array, Array)>>,
}

impl LlamaKvCache {
    fn append(&mut self, i: usize, k: Array, v: Array) -> Result<(Array, Array)> {
        let merged = match self.layers[i].take() {
            Some((pk, pv)) => (
                concatenate_axis(&[&pk, &k], 2)?,
                concatenate_axis(&[&pv, &v], 2)?,
            ),
            None => (k, v),
        };
        self.layers[i] = Some((merged.0.clone(), merged.1.clone()));
        Ok(merged)
    }
}

/// Llama-3 RoPE with the `rope_scaling` frequency adjustment (`_compute_llama3_parameters`): low
/// frequencies are divided by `factor`, high frequencies left alone, and a smooth interpolation in
/// between. `factor == 1` (or no `rope_scaling`) reduces to standard RoPE.
struct Llama3Rope {
    inv_freq: Vec<f32>,
    dim: i32,
}

impl Llama3Rope {
    fn new(cfg: &LlamaConfig) -> Self {
        let half = (cfg.head_dim / 2) as usize;
        let low_freq_wavelen = cfg.rope_original_context / cfg.rope_low_freq_factor;
        let high_freq_wavelen = cfg.rope_original_context / cfg.rope_high_freq_factor;
        let inv_freq = (0..half)
            .map(|i| {
                let inv = 1.0 / cfg.rope_theta.powf((2 * i) as f32 / cfg.head_dim as f32);
                let wavelen = 2.0 * std::f32::consts::PI / inv;
                if wavelen > low_freq_wavelen {
                    inv / cfg.rope_factor
                } else if wavelen < high_freq_wavelen {
                    inv
                } else {
                    let smooth = (cfg.rope_original_context / wavelen - cfg.rope_low_freq_factor)
                        / (cfg.rope_high_freq_factor - cfg.rope_low_freq_factor);
                    (1.0 - smooth) * inv / cfg.rope_factor + smooth * inv
                }
            })
            .collect();
        Self {
            inv_freq,
            dim: cfg.head_dim,
        }
    }

    fn forward(&self, seq_len: i32, offset: i32) -> Result<(Array, Array)> {
        let half = self.inv_freq.len();
        let mut freqs = Vec::with_capacity(seq_len as usize * half);
        for s in 0..seq_len {
            let pos = offset + s;
            for &f in &self.inv_freq {
                freqs.push(pos as f32 * f);
            }
        }
        let freqs = Array::from_slice(&freqs, &[seq_len, half as i32]);
        let emb = concatenate_axis(&[&freqs, &freqs], 1)?;
        let cos_t = cos(&emb)?
            .reshape(&[1, seq_len, self.dim])?
            .as_dtype(Dtype::Bfloat16)?;
        let sin_t = sin(&emb)?
            .reshape(&[1, seq_len, self.dim])?
            .as_dtype(Dtype::Bfloat16)?;
        Ok((cos_t, sin_t))
    }
}

fn apply_rope(x: &Array, cos_t: &Array, sin_t: &Array) -> Result<Array> {
    let cos_t = cos_t.expand_dims(2)?;
    let sin_t = sin_t.expand_dims(2)?;
    let parts = split(x, 2, 3)?;
    let rot = concatenate_axis(&[&parts[1].negative()?, &parts[0]], 3)?;
    Ok(add(&multiply(x, &cos_t)?, &multiply(&rot, &sin_t)?)?)
}

fn repeat_kv_cache(x: &Array, groups: i32) -> Result<Array> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let sh = x.shape();
    let (b, hkv, s, hd) = (sh[0], sh[1], sh[2], sh[3]);
    let x = x.expand_dims(2)?;
    let x = broadcast_to(&x, &[b, hkv, groups, s, hd])?;
    Ok(x.reshape(&[b, hkv * groups, s, hd])?)
}

fn req_bf16(w: &Weights, key: &str) -> Result<Array> {
    Ok(w.require(key)?.as_dtype(Dtype::Bfloat16)?)
}

fn join(prefix: &str, key: &str) -> String {
    if prefix.is_empty() {
        key.to_owned()
    } else {
        format!("{prefix}.{key}")
    }
}

// --- sampling -------------------------------------------------------------------------------------

/// Sample the next token from logits `[.., vocab]`. `temperature <= 0` is greedy `argmax` (seed-free,
/// deterministic); otherwise softmax-with-temperature, optional top-p nucleus filtering, then a
/// categorical draw from `rng`. No repetition penalty — matching the candle backend's plain
/// `LogitsProcessor` so the two backends behave the same.
pub fn sample_token(
    logits: &Array,
    temperature: f32,
    top_p: f32,
    rng: &mut SplitMix64,
) -> Result<i32> {
    let lf = logits.as_dtype(Dtype::Float32)?;
    let v: Vec<f32> = lf.as_slice::<f32>().to_vec();
    let vocab = v.len();

    if temperature <= 0.0 {
        return Ok(argmax_f32(&v));
    }

    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let inv_t = 1.0 / temperature;
    let mut probs: Vec<(usize, f32)> = (0..vocab)
        .map(|i| (i, ((v[i] - max) * inv_t).exp()))
        .collect();

    if top_p < 1.0 {
        probs = nucleus_select(&probs, top_p);
    }

    let total: f32 = probs.iter().map(|x| x.1).sum();
    if total <= 0.0 || !total.is_finite() {
        return Ok(argmax_f32(&v));
    }
    let mut target = rng.next_f32() * total;
    for (i, prob) in &probs {
        target -= prob;
        if target <= 0.0 {
            return Ok(*i as i32);
        }
    }
    Ok(probs.last().map(|x| x.0).unwrap_or(0) as i32)
}

/// Top-p (nucleus) selection: the highest-probability `(token, weight)` pairs in descending order
/// whose cumulative weight first reaches `top_p · total`, via a partial max-heap (popped only until
/// the threshold) instead of a full `vocab` sort. At least one token is always kept.
fn nucleus_select(probs: &[(usize, f32)], top_p: f32) -> Vec<(usize, f32)> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    struct ByWeight(usize, f32);
    impl PartialEq for ByWeight {
        fn eq(&self, o: &Self) -> bool {
            self.1.total_cmp(&o.1) == Ordering::Equal
        }
    }
    impl Eq for ByWeight {}
    impl PartialOrd for ByWeight {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for ByWeight {
        fn cmp(&self, o: &Self) -> Ordering {
            self.1.total_cmp(&o.1)
        }
    }

    let total: f32 = probs.iter().map(|x| x.1).sum();
    let threshold = top_p.max(0.0) * total;
    let mut heap: BinaryHeap<ByWeight> = probs.iter().map(|&(i, p)| ByWeight(i, p)).collect();
    let mut kept: Vec<(usize, f32)> = Vec::new();
    let mut cum = 0.0f32;
    while let Some(ByWeight(i, p)) = heap.pop() {
        kept.push((i, p));
        cum += p;
        if cum >= threshold {
            break;
        }
    }
    kept
}

fn argmax_f32(v: &[f32]) -> i32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best as i32
}

/// Deterministic SplitMix64 PRNG for the categorical draw (seedable + reproducible).
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax_at_temperature_zero() {
        let logits = Array::from_slice(&[0.1f32, 4.0, 2.0], &[1, 3]);
        let mut rng = SplitMix64::new(0);
        assert_eq!(sample_token(&logits, 0.0, 1.0, &mut rng).unwrap(), 1);
    }

    #[test]
    fn top_p_keeps_at_least_one_token() {
        let logits = Array::from_slice(&[5.0f32, 4.0, 1.0], &[1, 3]);
        let mut rng = SplitMix64::new(0);
        // top_p 0 still keeps the single most-probable token.
        assert_eq!(sample_token(&logits, 0.7, 0.0, &mut rng).unwrap(), 0);
    }

    #[test]
    fn sampling_is_seed_reproducible_and_varies() {
        let logits = Array::from_slice(&[0.0f32; 64], &[1, 64]);
        let draw = |seed: u64| -> Vec<i32> {
            let mut rng = SplitMix64::new(seed);
            (0..32)
                .map(|_| sample_token(&logits, 1.0, 1.0, &mut rng).unwrap())
                .collect()
        };
        assert_eq!(draw(7), draw(7), "same seed reproduces the same samples");
        assert_ne!(draw(7), draw(99), "different seeds differ");
    }

    #[test]
    fn nucleus_select_matches_full_sort() {
        fn reference(probs: &[(usize, f32)], top_p: f32) -> Vec<(usize, f32)> {
            let mut p = probs.to_vec();
            p.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            let total: f32 = p.iter().map(|x| x.1).sum();
            let threshold = top_p.max(0.0) * total;
            let mut cum = 0.0;
            let mut keep = p.len();
            for (n, x) in p.iter().enumerate() {
                cum += x.1;
                if cum >= threshold {
                    keep = n + 1;
                    break;
                }
            }
            p.truncate(keep.max(1));
            p
        }
        let probs: Vec<(usize, f32)> = [3.1, 0.2, 9.4, 1.7, 0.05, 5.5, 2.2, 0.9]
            .iter()
            .enumerate()
            .map(|(i, &w)| (i, w))
            .collect();
        for top_p in [0.0_f32, 0.3, 0.5, 0.8, 0.95, 0.999] {
            assert_eq!(
                nucleus_select(&probs, top_p),
                reference(&probs, top_p),
                "top_p={top_p}"
            );
        }
    }

    #[test]
    fn config_from_json_parses_llama_3_2_3b_fields() {
        let dir = std::env::temp_dir().join("mlx_gen_prompt_refine_cfg_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(
            &path,
            r#"{
                "hidden_size": 3072,
                "intermediate_size": 8192,
                "num_hidden_layers": 28,
                "num_attention_heads": 24,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 128256,
                "rms_norm_eps": 1e-05,
                "rope_theta": 500000.0,
                "tie_word_embeddings": true,
                "rope_scaling": {
                    "factor": 32.0,
                    "high_freq_factor": 4.0,
                    "low_freq_factor": 1.0,
                    "original_max_position_embeddings": 8192,
                    "rope_type": "llama3"
                }
            }"#,
        )
        .unwrap();
        let cfg = LlamaConfig::from_json(&path).unwrap();
        assert_eq!(cfg.hidden_size, 3072);
        assert_eq!(cfg.intermediate_size, 8192);
        assert_eq!(cfg.num_layers, 28);
        assert_eq!(cfg.num_heads, 24);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.vocab_size, 128256);
        assert!(cfg.tie_word_embeddings);
        assert_eq!(cfg.rope_factor, 32.0);
        assert_eq!(cfg.rope_high_freq_factor, 4.0);
        assert_eq!(cfg.rope_original_context, 8192.0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn config_from_json_defaults_rope_to_standard_when_absent() {
        let dir = std::env::temp_dir().join("mlx_gen_prompt_refine_cfg_test2");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(
            &path,
            r#"{
                "hidden_size": 2048,
                "intermediate_size": 5632,
                "num_hidden_layers": 22,
                "num_attention_heads": 32,
                "num_key_value_heads": 4,
                "vocab_size": 32000
            }"#,
        )
        .unwrap();
        let cfg = LlamaConfig::from_json(&path).unwrap();
        // head_dim falls back to hidden/num_heads; rope factors default to 1.0 (standard RoPE).
        assert_eq!(cfg.head_dim, 2048 / 32);
        assert_eq!(cfg.rope_factor, 1.0);
        assert!(!cfg.tie_word_embeddings);
        let _ = std::fs::remove_file(&path);
    }
}
