//! LLaVA projector, image-token splice, and Llama decoder for JoyCaption.
//!
//! JoyCaption is a `LlavaForConditionalGeneration`: SigLIP image features are projected into the
//! Llama hidden size, expanded image-token placeholders are replaced by those projected features,
//! and a Llama-3.1 8B causal decoder generates the caption text.

use mlx_rs::fast::{rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, broadcast_to, concatenate_axis, matmul, multiply, split};
use mlx_rs::{Array, Dtype};

use crate::array::host_i32;
use crate::caption::{CaptionFinishReason, CaptionSampling};
use crate::generator::default_seed;
use crate::nn::{gelu_exact, linear, silu};
use crate::runtime::CancelFlag;
use crate::weights::Weights;
use crate::{Error, Result};

use super::{END_OF_TEXT_TOKEN_ID, EOM_TOKEN_ID, EOT_TOKEN_ID, IMAGE_SEQ_LENGTH, IMAGE_TOKEN_ID};

pub const LLAMA_HIDDEN_SIZE: i32 = 4096;
pub const LLAMA_INTERMEDIATE_SIZE: i32 = 14336;
pub const LLAMA_NUM_LAYERS: usize = 32;
pub const LLAMA_NUM_HEADS: i32 = 32;
pub const LLAMA_NUM_KV_HEADS: i32 = 8;
pub const LLAMA_HEAD_DIM: i32 = 128;
pub const LLAMA_VOCAB_SIZE: i32 = 128256;
pub const LLAMA_RMS_NORM_EPS: f32 = 1e-5;
pub const LLAMA_ROPE_THETA: f32 = 500_000.0;
pub const LLAMA_ROPE_FACTOR: f32 = 8.0;
pub const LLAMA_ROPE_LOW_FREQ_FACTOR: f32 = 1.0;
pub const LLAMA_ROPE_HIGH_FREQ_FACTOR: f32 = 4.0;
pub const LLAMA_ORIGINAL_MAX_POSITION_EMBEDDINGS: f32 = 8192.0;

pub const PROJECTOR_IN_SIZE: i32 = 1152;
pub const PROJECTOR_HIDDEN_SIZE: i32 = 4096;
pub const PROJECTOR_OUT_SIZE: i32 = 4096;

pub const STOP_TOKENS: &[i32] = &[END_OF_TEXT_TOKEN_ID, EOM_TOKEN_ID, EOT_TOKEN_ID];

/// Repetition-penalty strength used in [`sample_token`] — the classic CTRL / HF
/// `RepetitionPenaltyLogitsProcessor` formulation (Keskar et al. 2019): a recently-emitted token's
/// logit is **divided** by this when positive and **multiplied** by it when negative, mildly
/// discouraging the model from repeating it. `1.05` is a gentle penalty.
///
/// NOTE: this is a deliberate, port-time deviation from the reference sampler — the HF JoyCaption
/// demo (`fancyfeast/joy-caption-*`) generates with plain temperature/top-p and applies **no**
/// repetition penalty. It was added here to curb JoyCaption's tendency to loop on long captions; it
/// is documented (rather than removed or made configurable) to keep current outputs stable. If
/// reference parity is ever the goal, set the penalty to `1.0` (a no-op) or lift it into
/// [`CaptionSampling`](crate::caption::CaptionSampling).
const REPETITION_PENALTY: f32 = 1.05;
/// How many of the most-recently-emitted history tokens [`sample_token`]'s repetition penalty looks
/// back over (the CTRL-style sliding window).
const REPETITION_PENALTY_WINDOW: usize = 256;

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
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            hidden_size: LLAMA_HIDDEN_SIZE,
            intermediate_size: LLAMA_INTERMEDIATE_SIZE,
            num_layers: LLAMA_NUM_LAYERS,
            num_heads: LLAMA_NUM_HEADS,
            num_kv_heads: LLAMA_NUM_KV_HEADS,
            head_dim: LLAMA_HEAD_DIM,
            vocab_size: LLAMA_VOCAB_SIZE,
            rms_norm_eps: LLAMA_RMS_NORM_EPS,
            rope_theta: LLAMA_ROPE_THETA,
            rope_factor: LLAMA_ROPE_FACTOR,
            rope_low_freq_factor: LLAMA_ROPE_LOW_FREQ_FACTOR,
            rope_high_freq_factor: LLAMA_ROPE_HIGH_FREQ_FACTOR,
            rope_original_context: LLAMA_ORIGINAL_MAX_POSITION_EMBEDDINGS,
        }
    }
}

impl LlamaConfig {
    pub fn head_count_matches_hidden(&self) -> bool {
        self.hidden_size == self.num_heads * self.head_dim
    }
}

pub struct LlavaProjector {
    linear1_w: Array,
    linear1_b: Array,
    linear2_w: Array,
    linear2_b: Array,
}

impl LlavaProjector {
    /// Load HF `multi_modal_projector.{linear_1,linear_2}.{weight,bias}`.
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            linear1_w: req_bf16(w, &join(prefix, "linear_1.weight"))?,
            linear1_b: req_bf16(w, &join(prefix, "linear_1.bias"))?,
            linear2_w: req_bf16(w, &join(prefix, "linear_2.weight"))?,
            linear2_b: req_bf16(w, &join(prefix, "linear_2.bias"))?,
        })
    }

    /// Project SigLIP features `[b, image_seq, 1152]` to Llama hidden features `[b, image_seq, 4096]`.
    pub fn forward(&self, vision_features: &Array) -> Result<Array> {
        let sh = vision_features.shape();
        if sh.len() != 3 || sh[2] != PROJECTOR_IN_SIZE {
            return Err(Error::Msg(format!(
                "joycaption projector: expected [batch, seq, {PROJECTOR_IN_SIZE}], got {sh:?}"
            )));
        }
        let hidden = gelu_exact(&linear(vision_features, &self.linear1_w, &self.linear1_b)?)?;
        linear(&hidden, &self.linear2_w, &self.linear2_b)
    }
}

pub struct LlamaDecoder {
    embed_tokens: Array,
    layers: Vec<LlamaLayer>,
    norm: Array,
    lm_head: Array,
    rope: Llama3Rope,
    cfg: LlamaConfig,
}

impl LlamaDecoder {
    /// Load HF `language_model.model.*` and `language_model.lm_head.weight`.
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
        Ok(Self {
            embed_tokens: w
                .require(&join(&model_prefix, "embed_tokens.weight"))?
                .as_dtype(Dtype::Bfloat16)?,
            layers,
            norm: w
                .require(&join(&model_prefix, "norm.weight"))?
                .as_dtype(Dtype::Bfloat16)?,
            lm_head: w
                .require(&join(prefix, "lm_head.weight"))?
                .as_dtype(Dtype::Bfloat16)?,
            rope: Llama3Rope::new(&cfg),
            cfg,
        })
    }

    pub fn new_cache(&self) -> LlamaKvCache {
        LlamaKvCache {
            layers: (0..self.layers.len()).map(|_| None).collect(),
        }
    }

    pub fn embed(&self, input_ids: &Array) -> Result<Array> {
        let sh = input_ids.shape();
        let ids = input_ids.reshape(&[-1])?;
        Ok(self
            .embed_tokens
            .take_axis(&ids, 0)?
            .reshape(&[sh[0], sh[1], self.cfg.hidden_size])?
            .as_dtype(Dtype::Bfloat16)?)
    }

    pub fn decode_logits(
        &self,
        input_ids: &Array,
        cache: &mut LlamaKvCache,
        offset: i32,
    ) -> Result<Array> {
        let embeds = self.embed(input_ids)?;
        self.decode_logits_from_embeds(&embeds, cache, offset)
    }

    /// Run pre-embedded tokens at absolute `offset`, append K/V to `cache`, and return logits for
    /// the last query position `[batch, vocab]`.
    pub fn decode_logits_from_embeds(
        &self,
        input_embeds: &Array,
        cache: &mut LlamaKvCache,
        offset: i32,
    ) -> Result<Array> {
        let sh = input_embeds.shape();
        if sh.len() != 3 || sh[2] != self.cfg.hidden_size {
            return Err(Error::Msg(format!(
                "joycaption llama: expected input embeds [batch, seq, {}], got {sh:?}",
                self.cfg.hidden_size
            )));
        }
        let (b, q_len) = (sh[0], sh[1]);
        let k_len = offset + q_len;
        let mask = decode_mask(q_len, k_len, offset)?;
        let (cos, sin) = self.rope.forward(q_len, offset)?;

        let mut hidden = input_embeds.as_dtype(Dtype::Bfloat16)?;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward_step(&hidden, &cos, &sin, &mask, cache, i)?;
        }

        let last_idx = Array::from_slice(&[q_len - 1], &[1]);
        let last = hidden
            .take_axis(&last_idx, 1)?
            .reshape(&[b, self.cfg.hidden_size])?;
        let normed = rms_norm(&last, &self.norm, self.cfg.rms_norm_eps)?;
        Ok(matmul(&normed, self.lm_head.t())?)
    }

    /// Generate token IDs from already-spliced prompt embeddings. Stop tokens are not included in
    /// the returned token list, matching the decode boundary callers expect.
    pub fn generate_from_embeds(
        &self,
        prompt_ids: &[i32],
        prompt_embeds: &Array,
        sampling: CaptionSampling,
        cancel: &CancelFlag,
    ) -> Result<LanguageGeneration> {
        if prompt_ids.is_empty() {
            return Err(Error::Msg("joycaption: prompt ids are empty".to_owned()));
        }
        let sh = prompt_embeds.shape();
        if sh.len() != 3 || sh[0] != 1 || sh[1] as usize != prompt_ids.len() {
            return Err(Error::Msg(format!(
                "joycaption: prompt ids length {} must match prompt embeds [1, seq, hidden], got {sh:?}",
                prompt_ids.len()
            )));
        }
        if cancel.is_cancelled() {
            return Ok(LanguageGeneration {
                token_ids: Vec::new(),
                finish_reason: CaptionFinishReason::Cancelled,
            });
        }

        let mut history = prompt_ids.to_vec();
        let mut generated = Vec::new();
        let mut cache = self.new_cache();
        // `seed: None` draws a fresh per-call seed so repeated captions vary; an explicit seed
        // reproduces an exact sample (F-002 — previously hardcoded to 0, silently deterministic).
        let mut rng = SplitMix64::new(sampling.seed.unwrap_or_else(default_seed));
        let prompt_len = prompt_embeds.shape()[1];
        let mut logits = self.decode_logits_from_embeds(prompt_embeds, &mut cache, 0)?;

        for step in 0..sampling.max_new_tokens {
            if cancel.is_cancelled() {
                return Ok(LanguageGeneration {
                    token_ids: generated,
                    finish_reason: CaptionFinishReason::Cancelled,
                });
            }

            let next = sample_token(&logits, &history, sampling, &mut rng)?;
            if is_stop_token(next) {
                return Ok(LanguageGeneration {
                    token_ids: generated,
                    finish_reason: CaptionFinishReason::StopToken,
                });
            }
            generated.push(next);
            history.push(next);

            if step + 1 == sampling.max_new_tokens {
                break;
            }

            let token = Array::from_slice(&[next], &[1, 1]);
            logits = self.decode_logits(&token, &mut cache, prompt_len + step as i32)?;
        }

        Ok(LanguageGeneration {
            token_ids: generated,
            finish_reason: CaptionFinishReason::MaxTokens,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LanguageGeneration {
    pub token_ids: Vec<i32>,
    pub finish_reason: CaptionFinishReason,
}

pub fn is_stop_token(token_id: i32) -> bool {
    STOP_TOKENS.contains(&token_id)
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
            input_ln: w
                .require(&join(prefix, "input_layernorm.weight"))?
                .as_dtype(Dtype::Bfloat16)?,
            post_ln: w
                .require(&join(prefix, "post_attention_layernorm.weight"))?
                .as_dtype(Dtype::Bfloat16)?,
            attn: LlamaAttention::from_weights(w, &join(prefix, "self_attn"), cfg)?,
            mlp: LlamaMlp::from_weights(w, &join(prefix, "mlp"))?,
            eps: cfg.rms_norm_eps,
        })
    }

    fn forward_step(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        mask: &Array,
        cache: &mut LlamaKvCache,
        layer_idx: usize,
    ) -> Result<Array> {
        let normed = rms_norm(x, &self.input_ln, self.eps)?;
        let h = add(
            x,
            &self
                .attn
                .forward_step(&normed, cos, sin, mask, cache, layer_idx)?,
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
        cos: &Array,
        sin: &Array,
        mask: &Array,
        cache: &mut LlamaKvCache,
        layer_idx: usize,
    ) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        let q = matmul(x, self.q_w.t())?.reshape(&[b, s, self.num_heads, self.head_dim])?;
        let k = matmul(x, self.k_w.t())?.reshape(&[b, s, self.num_kv_heads, self.head_dim])?;
        let v = matmul(x, self.v_w.t())?.reshape(&[b, s, self.num_kv_heads, self.head_dim])?;

        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;

        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;
        let (k_all, v_all) = cache.append(layer_idx, k, v)?;

        let groups = self.num_heads / self.num_kv_heads;
        let k_all = repeat_kv_cache(&k_all, groups)?;
        let v_all = repeat_kv_cache(&v_all, groups)?;
        let mask = mask.as_dtype(q.dtype())?;
        let out = scaled_dot_product_attention(&q, &k_all, &v_all, self.scale, &mask, None)?;
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
        let cos = mlx_rs::ops::cos(&emb)?
            .reshape(&[1, seq_len, self.dim])?
            .as_dtype(Dtype::Bfloat16)?;
        let sin = mlx_rs::ops::sin(&emb)?
            .reshape(&[1, seq_len, self.dim])?
            .as_dtype(Dtype::Bfloat16)?;
        Ok((cos, sin))
    }
}

/// HF LLaVA prompt expansion: each image marker token becomes `image_seq_length` placeholders so
/// projected image rows can replace them one-for-one.
pub fn expand_image_tokens(ids: &[i32], image_token_id: i32, image_seq_length: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(ids.len() + image_seq_length.saturating_sub(1));
    for &id in ids {
        if id == image_token_id {
            out.extend(std::iter::repeat_n(image_token_id, image_seq_length));
        } else {
            out.push(id);
        }
    }
    out
}

pub fn expand_joycaption_image_tokens(ids: &[i32]) -> Vec<i32> {
    expand_image_tokens(ids, IMAGE_TOKEN_ID, IMAGE_SEQ_LENGTH)
}

/// Splice projected image features into token embeddings. `input_ids` must already be expanded so
/// the number of image-token positions equals the number of projected image rows.
pub fn splice_image_features(
    token_embeds: &Array,
    input_ids: &Array,
    projected_features: &Array,
    image_token_id: i32,
) -> Result<Array> {
    let sh = token_embeds.shape();
    if sh.len() != 3 {
        return Err(Error::Msg(format!(
            "joycaption splice: token embeddings must be [batch, seq, hidden], got {sh:?}"
        )));
    }
    let (b, s, h) = (sh[0], sh[1], sh[2]);
    let n_text = b * s;
    let fsh = projected_features.shape();
    let features = match fsh {
        [fb, fs, fh] if *fb == b && *fh == h => projected_features.reshape(&[fb * fs, h])?,
        [fs, fh] if *fh == h => projected_features.reshape(&[*fs, h])?,
        _ => {
            return Err(Error::Msg(format!(
                "joycaption splice: projected features must be [batch, image_seq, {h}] or [image_seq, {h}], got {fsh:?}"
            )));
        }
    };
    let n_vis = features.shape()[0];
    let ids = host_i32(input_ids)?;
    let gather = image_gather_index_exact(&ids, image_token_id, n_vis, n_text)?;
    let embeds_flat = token_embeds.reshape(&[n_text, h])?;
    let src = concatenate_axis(&[&embeds_flat, &features], 0)?;
    let idx = Array::from_slice(&gather, &[n_text]);
    Ok(src.take_axis(&idx, 0)?.reshape(&[b, s, h])?)
}

pub fn image_gather_index_exact(
    ids: &[i32],
    image_token_id: i32,
    n_vis: i32,
    n_text: i32,
) -> Result<Vec<i32>> {
    if ids.len() != n_text as usize {
        return Err(Error::Msg(format!(
            "joycaption splice: input_ids length {} does not match embedding rows {n_text}",
            ids.len()
        )));
    }
    let count = ids.iter().filter(|&&id| id == image_token_id).count() as i32;
    if count != n_vis {
        return Err(Error::Msg(format!(
            "joycaption splice: image token count {count} does not match projected image rows {n_vis}"
        )));
    }
    let mut out = Vec::with_capacity(n_text as usize);
    let mut vi = 0i32;
    for (p, &id) in ids.iter().enumerate() {
        if id == image_token_id {
            out.push(n_text + vi);
            vi += 1;
        } else {
            out.push(p as i32);
        }
    }
    Ok(out)
}

/// Convert expanded prompt IDs to `[1, seq]` ids and all-ones attention mask.
pub fn input_arrays_from_ids(ids: &[i32]) -> (Array, Array) {
    let seq = ids.len() as i32;
    let mask = vec![1i32; ids.len()];
    (
        Array::from_slice(ids, &[1, seq]),
        Array::from_slice(&mask, &[1, seq]),
    )
}

fn sample_token(
    logits: &Array,
    history: &[i32],
    sampling: CaptionSampling,
    rng: &mut SplitMix64,
) -> Result<i32> {
    let lf = logits.as_dtype(Dtype::Float32)?;
    let mut v: Vec<f32> = lf.as_slice::<f32>().to_vec();
    let vocab = v.len();

    // CTRL/HF-style repetition penalty over the recent history (see REPETITION_PENALTY) — a
    // documented, port-time deviation from the plain temperature/top-p reference sampler (F-012).
    for &token in history.iter().rev().take(REPETITION_PENALTY_WINDOW) {
        let idx = token as usize;
        if idx < vocab {
            v[idx] = if v[idx] < 0.0 {
                v[idx] * REPETITION_PENALTY
            } else {
                v[idx] / REPETITION_PENALTY
            };
        }
    }

    if sampling.temperature <= 0.0 {
        return Ok(argmax_f32(&v));
    }

    let mut idx: Vec<usize> = (0..vocab).collect();
    let max = idx.iter().map(|&i| v[i]).fold(f32::NEG_INFINITY, f32::max);
    let inv_t = 1.0 / sampling.temperature;
    let mut probs: Vec<(usize, f32)> = idx
        .drain(..)
        .map(|i| (i, ((v[i] - max) * inv_t).exp()))
        .collect();

    if sampling.top_p < 1.0 {
        probs = nucleus_select(&probs, sampling.top_p);
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
/// whose cumulative weight first reaches `top_p · total`. Found with a partial **max-heap** selection
/// — popped only until the threshold — instead of sorting all `vocab` entries every token (F-011); the
/// finding's "partial-select before the host sort". For distinct weights (the universal case for a
/// real softmax) the kept set and its order are identical to a full descending `sort_unstable_by`.
/// At least one token is always kept.
fn nucleus_select(probs: &[(usize, f32)], top_p: f32) -> Vec<(usize, f32)> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    /// `(token, weight)` ordered by weight for a max-heap (`total_cmp` is a total order over f32).
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

fn apply_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let cos = cos.expand_dims(2)?;
    let sin = sin.expand_dims(2)?;
    let parts = split(x, 2, 3)?;
    let rot = concatenate_axis(&[&parts[1].negative()?, &parts[0]], 3)?;
    Ok(add(&multiply(x, &cos)?, &multiply(&rot, &sin)?)?)
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

fn decode_mask(q_len: i32, k_len: i32, q_offset: i32) -> Result<Array> {
    let neg = half_min_bf16();
    let mut data = vec![0f32; (q_len * k_len) as usize];
    for r in 0..q_len {
        let pos = q_offset + r;
        for j in 0..k_len {
            if j > pos {
                data[(r * k_len + j) as usize] = neg;
            }
        }
    }
    Ok(Array::from_slice(&data, &[1, 1, q_len, k_len]).as_dtype(Dtype::Bfloat16)?)
}

fn req_bf16(w: &Weights, key: &str) -> Result<Array> {
    Ok(w.require(key)?.as_dtype(Dtype::Bfloat16)?)
}

use super::join;

fn half_min_bf16() -> f32 {
    -3.389_531_4e38
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
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
    use std::env;

    use mlx_rs::ops::zeros;

    use super::super::vision::{
        joycaption_vision_features, SiglipImageProcessor, SiglipVisionConfig, SiglipVisionTower,
    };
    use super::super::{decode_generated, encode_chat_prompt, load_tokenizer};
    use super::*;
    use crate::media::Image;

    #[test]
    fn nucleus_select_matches_full_sort() {
        // F-011: the partial max-heap nucleus selection must return exactly the same kept set and
        // descending order as the previous full-sort-then-truncate, for distinct weights.
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
        // A single dominating weight keeps exactly one token even at high top_p.
        let spike: Vec<(usize, f32)> = [(0usize, 1000.0f32), (1, 0.001), (2, 0.001)].to_vec();
        assert_eq!(nucleus_select(&spike, 0.9).len(), 1);
    }

    #[test]
    fn default_llama_config_matches_joycaption() {
        let cfg = LlamaConfig::default();
        assert!(cfg.head_count_matches_hidden());
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.intermediate_size, 14336);
        assert_eq!(cfg.num_layers, 32);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.vocab_size, 128256);
        assert_eq!(STOP_TOKENS, &[128001, 128008, 128009]);
    }

    #[test]
    fn image_token_expansion_matches_llava_seq_length() {
        let ids = [1, IMAGE_TOKEN_ID, 2];
        let expanded = expand_image_tokens(&ids, IMAGE_TOKEN_ID, 4);
        assert_eq!(
            expanded,
            vec![
                1,
                IMAGE_TOKEN_ID,
                IMAGE_TOKEN_ID,
                IMAGE_TOKEN_ID,
                IMAGE_TOKEN_ID,
                2
            ]
        );
    }

    #[test]
    fn gather_replaces_image_tokens_exactly_in_order() {
        let ids = [10, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 11];
        let got = image_gather_index_exact(&ids, IMAGE_TOKEN_ID, 2, 4).unwrap();
        assert_eq!(got, vec![0, 4, 5, 3]);
    }

    #[test]
    fn gather_errors_when_image_feature_count_mismatches() {
        let ids = [IMAGE_TOKEN_ID, 7];
        let err = image_gather_index_exact(&ids, IMAGE_TOKEN_ID, 2, 2).unwrap_err();
        assert!(err.to_string().contains("image token count 1"));
    }

    #[test]
    fn splice_replaces_projected_rows() {
        let embeds =
            Array::from_slice(&[1.0f32, 1.0, 10.0, 10.0, 20.0, 20.0, 2.0, 2.0], &[1, 4, 2]);
        let ids = Array::from_slice(&[5, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 6], &[1, 4]);
        let features = Array::from_slice(&[100.0f32, 101.0, 200.0, 201.0], &[1, 2, 2]);
        let got = splice_image_features(&embeds, &ids, &features, IMAGE_TOKEN_ID).unwrap();
        assert_eq!(
            got.as_slice::<f32>(),
            &[1.0, 1.0, 100.0, 101.0, 200.0, 201.0, 2.0, 2.0]
        );
    }

    #[test]
    fn sampling_respects_greedy_temperature_zero() {
        let logits = Array::from_slice(&[0.1f32, 4.0, 2.0], &[1, 3]);
        let mut rng = SplitMix64::new(0);
        let next = sample_token(
            &logits,
            &[],
            CaptionSampling {
                temperature: 0.0,
                top_p: 1.0,
                max_new_tokens: 1,
                seed: None,
            },
            &mut rng,
        )
        .unwrap();
        assert_eq!(next, 1);
    }

    #[test]
    fn sampling_top_p_keeps_at_least_one_token() {
        let logits = Array::from_slice(&[5.0f32, 4.0, 1.0], &[1, 3]);
        let mut rng = SplitMix64::new(0);
        let next = sample_token(
            &logits,
            &[],
            CaptionSampling {
                temperature: 0.7,
                top_p: 0.0,
                max_new_tokens: 1,
                seed: None,
            },
            &mut rng,
        )
        .unwrap();
        assert_eq!(next, 0);
    }

    #[test]
    fn caption_sampling_seed_is_reproducible_and_varies() {
        // F-002: stochastic sampling must be reproducible with a chosen seed and vary across seeds
        // (previously the RNG was hardcoded to 0, so every "sampled" caption was identical forever).
        assert_eq!(
            CaptionSampling::default().seed,
            None,
            "default seed is None"
        );

        // A flat distribution makes the categorical draw rng-dominated, so the seed steers it.
        let logits = Array::from_slice(&[0.0f32; 64], &[1, 64]);
        let sampling = CaptionSampling {
            temperature: 1.0,
            top_p: 1.0,
            max_new_tokens: 32,
            seed: None,
        };
        let draw = |seed: u64| -> Vec<i32> {
            let mut rng = SplitMix64::new(seed);
            (0..32)
                .map(|_| sample_token(&logits, &[], sampling, &mut rng).unwrap())
                .collect()
        };

        assert_eq!(draw(7), draw(7), "same seed reproduces the same samples");
        assert_ne!(
            draw(7),
            draw(99),
            "different seeds produce different samples"
        );
    }

    #[test]
    fn stop_token_set_matches_llama_generation_config() {
        assert!(is_stop_token(END_OF_TEXT_TOKEN_ID));
        assert!(is_stop_token(EOM_TOKEN_ID));
        assert!(is_stop_token(EOT_TOKEN_ID));
        assert!(!is_stop_token(IMAGE_TOKEN_ID));
    }

    #[test]
    fn cancelled_generation_returns_cancelled_without_decode() {
        let cancel = CancelFlag::new();
        cancel.cancel();
        let gen = LanguageGeneration {
            token_ids: Vec::new(),
            finish_reason: CaptionFinishReason::Cancelled,
        };
        assert!(cancel.is_cancelled());
        assert_eq!(gen.finish_reason, CaptionFinishReason::Cancelled);
    }

    #[test]
    fn projector_rejects_wrong_feature_width() {
        let fake = LlavaProjector {
            linear1_w: zeros::<f32>(&[PROJECTOR_HIDDEN_SIZE, PROJECTOR_IN_SIZE]).unwrap(),
            linear1_b: zeros::<f32>(&[PROJECTOR_HIDDEN_SIZE]).unwrap(),
            linear2_w: zeros::<f32>(&[PROJECTOR_OUT_SIZE, PROJECTOR_HIDDEN_SIZE]).unwrap(),
            linear2_b: zeros::<f32>(&[PROJECTOR_OUT_SIZE]).unwrap(),
        };
        let bad = zeros::<f32>(&[1, 2, PROJECTOR_IN_SIZE - 1]).unwrap();
        assert!(fake.forward(&bad).is_err());
    }

    #[test]
    #[ignore = "needs the JoyCaption snapshot; set MLX_GEN_JOYCAPTION_SNAPSHOT"]
    fn real_weights_short_generation_produces_text() {
        let root =
            env::var("MLX_GEN_JOYCAPTION_SNAPSHOT").expect("set MLX_GEN_JOYCAPTION_SNAPSHOT");
        let weights = Weights::from_dir(&root).expect("weights load");
        let tokenizer = load_tokenizer(&root).expect("tokenizer loads");
        let vision = SiglipVisionTower::from_weights(
            &weights,
            "vision_tower.vision_model",
            SiglipVisionConfig::default(),
        )
        .expect("vision tower loads");
        let projector = LlavaProjector::from_weights(&weights, "multi_modal_projector")
            .expect("projector loads");
        let llama = LlamaDecoder::from_weights(&weights, "language_model", LlamaConfig::default())
            .expect("llama loads");

        let image = Image {
            width: 384,
            height: 384,
            pixels: vec![127u8; 384 * 384 * 3],
        };
        let pixels = SiglipImageProcessor::default()
            .preprocess(&image)
            .expect("image preprocesses");
        let vision_features =
            joycaption_vision_features(&vision.forward(&pixels).expect("vision forward"))
                .expect("vision features");
        let projected = projector
            .forward(&vision_features)
            .expect("projector forward");
        let ids =
            encode_chat_prompt(&tokenizer, "Write a very short caption.").expect("prompt encodes");
        let ids = expand_joycaption_image_tokens(&ids);
        let (input_ids, _) = input_arrays_from_ids(&ids);
        let embeds = llama.embed(&input_ids).expect("token embeds");
        let spliced =
            splice_image_features(&embeds, &input_ids, &projected, IMAGE_TOKEN_ID).expect("splice");
        let out = llama
            .generate_from_embeds(
                &ids,
                &spliced,
                CaptionSampling {
                    max_new_tokens: 8,
                    ..Default::default()
                },
                &CancelFlag::default(),
            )
            .expect("generate");
        let toks: Vec<u32> = out.token_ids.iter().map(|&id| id as u32).collect();
        let text = decode_generated(&tokenizer, &toks).expect("decode");
        assert!(
            !text.trim().is_empty(),
            "short generation should produce text"
        );
    }
}
