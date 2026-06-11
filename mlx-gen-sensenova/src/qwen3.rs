//! The NEO-Unify **dense dual-path Qwen3** backbone (sc-3182) — port of `modeling_qwen3.py`.
//!
//! Each of the 42 decoder layers carries two parallel **dense** transformer stacks: an
//! *understanding* path and a *generation* path (the `_mot_gen` weights). A forward runs on one
//! path at a time, selected by the `image_gen_indicators` mask — the reference's `forward_und`
//! (all-understanding tokens) and `forward_gen` (all-generation tokens). Those are the paths T2I
//! generation and text/understanding actually use, and the ones validated bit-near here.
//!
//! Per-layer attention (head_dim 128) splits each head into a **temporal** half (normed by
//! `q_norm`/`k_norm`) and a **spatial** half (normed by `q_norm_hw`/`k_norm_hw`); the spatial half
//! splits again into height + width. Three independent RoPE rotations are applied — temporal
//! (`rope_theta`), height and width (`rope_theta_hw`) — then concatenated back to 128. K/V are
//! shared GQA (8 KV heads → 32 query heads). Attention is the reference's eager path
//! (matmul → f32 softmax → matmul) under a block-causal mask, so understanding tokens attend
//! causally while a generation image-block (tokens sharing one temporal index) attends
//! bidirectionally within the block.
//!
//! NOTE — the reference's *mixed*-token attention (both understanding and generation tokens in one
//! forward) references undefined locals (`query_states_h`/`key_states_h`) and never norms the
//! temporal half, so it cannot run; only the pure paths are real. Mixed-token handling (for
//! interleaved generation) is resolved in sc-3190, where the correct behaviour is defined.

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::{add, broadcast_to, concatenate_axis, matmul, multiply, softmax_axis, split};
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::NeoChatConfig;
use crate::distill::lora_delta;

/// Which transformer path a forward runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Path {
    /// Understanding path (text / image input). The reference `forward_und`.
    Und,
    /// Image-generation path (the `_mot_gen` weights). The reference `forward_gen`.
    Gen,
}

/// Per-layer key/value cache for incremental decode (the reference's HF `DynamicCache`). Each entry
/// holds the **already-RoPE'd** keys and the raw values in `[B, Hkv, S, D]` layout (kv-head count,
/// pre-GQA-expansion — matching what the reference appends before `repeat_kv`). The flash-attn
/// `[B,S,H,D]` repack (`prepare_flash_kv_cache`) is a torch-kernel micro-opt with no MLX analogue;
/// the plain `[B,Hkv,S,D]` concat here is its functional equivalent.
pub struct KvCache {
    layers: Vec<Option<(Array, Array)>>,
    seq_len: i32,
}

impl KvCache {
    /// Total cached sequence length (the reference cache's `get_seq_length()`).
    pub fn len(&self) -> i32 {
        self.seq_len
    }

    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Persisting append (`update_cache=True`): concat the new K/V onto layer `i` and store the
    /// result back. Returns the full `[B,Hkv,S_all,D]` K/V for this forward.
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

    /// Non-persisting use (`update_cache=False`): concat past + current for this forward only, but
    /// do **not** store the current K/V. This is the denoise-loop path — each diffusion step runs a
    /// fresh image block against the frozen text prefix without polluting the cache.
    fn extend(&self, i: usize, k: &Array, v: &Array) -> Result<(Array, Array)> {
        match &self.layers[i] {
            Some((pk, pv)) => Ok((
                concatenate_axis(&[pk, k], 2)?,
                concatenate_axis(&[pv, v], 2)?,
            )),
            None => Ok((k.clone(), v.clone())),
        }
    }
}

/// `y = x · Wᵀ` for a stored `[out, in]` bias-less Linear.
fn matmul_t(x: &Array, w: &Array) -> Result<Array> {
    Ok(matmul(x, w.t())?)
}

fn require(w: &Weights, key: &str) -> Result<Array> {
    Ok(w.require(key)?.clone())
}

/// A bias-less Linear from a stored `[out, in]` weight, as a quantizable [`AdaptableLinear`]
/// (dense bf16 forward == the previous `matmul_t`; `quantize` swaps in a `quantized_matmul` base).
fn linear(w: &Weights, key: &str) -> Result<AdaptableLinear> {
    Ok(AdaptableLinear::dense(require(w, key)?, None))
}

/// The per-path attention weights (one of understanding / generation). The projections are
/// quantizable; the QK-norms stay dense (small, precision-sensitive).
struct AttnPath {
    q_proj: AdaptableLinear,
    k_proj: AdaptableLinear,
    v_proj: AdaptableLinear,
    o_proj: AdaptableLinear,
    q_norm: Array,
    k_norm: Array,
    q_norm_hw: Array,
    k_norm_hw: Array,
}

impl AttnPath {
    /// `attn_prefix` = `…layers.{i}.self_attn`, `s` = the path suffix (`""` or `"_mot_gen"`).
    fn from_weights(w: &Weights, attn_prefix: &str, s: &str) -> Result<Self> {
        Ok(Self {
            q_proj: linear(w, &format!("{attn_prefix}.q_proj{s}.weight"))?,
            k_proj: linear(w, &format!("{attn_prefix}.k_proj{s}.weight"))?,
            v_proj: linear(w, &format!("{attn_prefix}.v_proj{s}.weight"))?,
            o_proj: linear(w, &format!("{attn_prefix}.o_proj{s}.weight"))?,
            q_norm: require(w, &format!("{attn_prefix}.q_norm{s}.weight"))?,
            k_norm: require(w, &format!("{attn_prefix}.k_norm{s}.weight"))?,
            q_norm_hw: require(w, &format!("{attn_prefix}.q_norm_hw{s}.weight"))?,
            k_norm_hw: require(w, &format!("{attn_prefix}.k_norm_hw{s}.weight"))?,
        })
    }

    /// Quantize the four projections in place (Q4/Q8, group 64).
    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.q_proj.quantize(bits, None)?;
        self.k_proj.quantize(bits, None)?;
        self.v_proj.quantize(bits, None)?;
        self.o_proj.quantize(bits, None)
    }

    /// Merge the distill LoRA (sc-3192) into the four projections. `attn_prefix` =
    /// `…layers.{i}.self_attn`, `s` = the path suffix (`"_mot_gen"` — the LoRA touches only the
    /// generation path). Returns the number of projections merged (≤ 4; absent targets are skipped).
    fn merge_distill_lora(&mut self, lora: &Weights, attn_prefix: &str, s: &str) -> Result<usize> {
        let mut n = 0;
        for (proj, lin) in [
            ("q", &mut self.q_proj),
            ("k", &mut self.k_proj),
            ("v", &mut self.v_proj),
            ("o", &mut self.o_proj),
        ] {
            if let Some(delta) = lora_delta(lora, &format!("{attn_prefix}.{proj}_proj{s}"))? {
                lin.merge_dense_delta(&delta)?;
                n += 1;
            }
        }
        Ok(n)
    }
}

struct Mlp {
    gate: AdaptableLinear,
    up: AdaptableLinear,
    down: AdaptableLinear,
}

impl Mlp {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate: linear(w, &format!("{prefix}.gate_proj.weight"))?,
            up: linear(w, &format!("{prefix}.up_proj.weight"))?,
            down: linear(w, &format!("{prefix}.down_proj.weight"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let gated = multiply(&silu(&self.gate.forward(x)?)?, &self.up.forward(x)?)?;
        self.down.forward(&gated)
    }

    /// Quantize the SwiGLU's three linears in place.
    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.gate.quantize(bits, None)?;
        self.up.quantize(bits, None)?;
        self.down.quantize(bits, None)
    }

    /// Merge the distill LoRA (sc-3192) into the SwiGLU's three linears. `prefix` =
    /// `…layers.{i}.mlp_mot_gen` (the LoRA touches only the generation-path MLP). Returns the number
    /// of linears merged (≤ 3).
    fn merge_distill_lora(&mut self, lora: &Weights, prefix: &str) -> Result<usize> {
        let mut n = 0;
        for (proj, lin) in [
            ("gate", &mut self.gate),
            ("up", &mut self.up),
            ("down", &mut self.down),
        ] {
            if let Some(delta) = lora_delta(lora, &format!("{prefix}.{proj}_proj"))? {
                lin.merge_dense_delta(&delta)?;
                n += 1;
            }
        }
        Ok(n)
    }
}

struct Layer {
    input_ln: Array,
    input_ln_gen: Array,
    post_ln: Array,
    post_ln_gen: Array,
    attn_und: AttnPath,
    attn_gen: AttnPath,
    mlp_und: Mlp,
    mlp_gen: Mlp,
}

impl Layer {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let attn = format!("{prefix}.self_attn");
        Ok(Self {
            input_ln: require(w, &format!("{prefix}.input_layernorm.weight"))?,
            input_ln_gen: require(w, &format!("{prefix}.input_layernorm_mot_gen.weight"))?,
            post_ln: require(w, &format!("{prefix}.post_attention_layernorm.weight"))?,
            post_ln_gen: require(
                w,
                &format!("{prefix}.post_attention_layernorm_mot_gen.weight"),
            )?,
            attn_und: AttnPath::from_weights(w, &attn, "")?,
            attn_gen: AttnPath::from_weights(w, &attn, "_mot_gen")?,
            mlp_und: Mlp::from_weights(w, &format!("{prefix}.mlp"))?,
            mlp_gen: Mlp::from_weights(w, &format!("{prefix}.mlp_mot_gen"))?,
        })
    }

    /// Quantize both paths' attention projections + SwiGLU linears in place.
    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attn_und.quantize(bits)?;
        self.attn_gen.quantize(bits)?;
        self.mlp_und.quantize(bits)?;
        self.mlp_gen.quantize(bits)
    }
}

/// RoPE cos/sin for arbitrary integer positions over `dim` rotary dims (f32), shaped `[1, S, dim]`.
/// Mirrors `Qwen3RotaryEmbedding`: `inv_freq[j] = theta^(-2j/dim)`, `emb = cat(freqs, freqs)`.
fn rope_cos_sin(positions: &[i32], dim: usize, theta: f32) -> Result<(Array, Array)> {
    let half = dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|j| 1.0f32 / theta.powf((2 * j) as f32 / dim as f32))
        .collect();
    let pos: Vec<f32> = positions.iter().map(|&p| p as f32).collect();
    let s = positions.len() as i32;
    let pos = Array::from_slice(&pos, &[s, 1]);
    let inv = Array::from_slice(&inv_freq, &[1, half as i32]);
    let freqs = matmul(&pos, &inv)?; // [S, half]
    let emb = concatenate_axis(&[&freqs, &freqs], 1)?; // [S, dim]
    let cos = emb.cos()?.expand_dims(0)?; // [1, S, dim]
    let sin = emb.sin()?.expand_dims(0)?;
    Ok((cos, sin))
}

/// HF half-split rotary: `x*cos + rotate_half(x)*sin`, with `cos`/`sin` `[1,S,dim]` broadcast over
/// the head axis of `x` `[B,S,H,dim]`.
fn apply_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let cos = cos.expand_dims(2)?; // [1,S,1,dim]
    let sin = sin.expand_dims(2)?;
    let parts = split(x, 2, 3)?;
    let rot = concatenate_axis(&[&parts[1].negative()?, &parts[0]], 3)?;
    Ok(add(&multiply(x, &cos)?, &multiply(&rot, &sin)?)?)
}

/// Expand `[B,S,Hkv,D]` → `[B,S,Hkv*groups,D]` (GQA), repeating each kv head `groups` times.
fn repeat_kv(x: &Array, groups: i32) -> Result<Array> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let sh = x.shape();
    let (b, s, hkv, d) = (sh[0], sh[1], sh[2], sh[3]);
    let x = x.expand_dims(3)?;
    let x = broadcast_to(&x, &[b, s, hkv, groups, d])?;
    Ok(x.reshape(&[b, s, hkv * groups, d])?)
}

/// The dense dual-path Qwen3 backbone: embeddings, the decoder stack, the dual final norm, and the
/// `lm_head` (its own tensor, or the tied `embed_tokens` weight when `tie_word_embeddings`).
pub struct Qwen3Backbone {
    embed_tokens: Array,
    layers: Vec<Layer>,
    norm: Array,
    norm_gen: Array,
    lm_head: Array,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    eps: f32,
    rope_theta: f32,
    rope_theta_hw: f32,
}

/// Precomputed tri-axis RoPE tables (`(cos, sin)` per temporal/H/W axis) and the additive block
/// mask for a fixed set of position indexes and cache prefix length. These are invariant across the
/// denoise steps of a given cache (the indexes and `past` don't change in use-only mode), so the
/// gen-path denoise loops build one per cache via [`Qwen3Backbone::prepare_rope_mask`] and reuse it
/// for every step instead of rebuilding inside each `predict_v` call — the mask alone is a CPU
/// `q × (past+q)` f32 fill, ~1–5M entries at 1024² (F-139).
pub struct RopeMask {
    cos_t: Array,
    sin_t: Array,
    cos_h: Array,
    sin_h: Array,
    cos_w: Array,
    sin_w: Array,
    mask: Array,
    n_tokens: i32,
}

impl Qwen3Backbone {
    /// Build from a checkpoint, `prefix` = the `language_model` namespace (e.g. `"language_model"`).
    pub fn from_weights(w: &Weights, cfg: &NeoChatConfig, prefix: &str) -> Result<Self> {
        let model = format!("{prefix}.model");
        let layers = (0..cfg.llm.num_hidden_layers)
            .map(|i| Layer::from_weights(w, &format!("{model}.layers.{i}")))
            .collect::<Result<Vec<_>>>()?;
        let embed_tokens = require(w, &format!("{model}.embed_tokens.weight"))?;
        // Tied embeddings share the token-embedding matrix as the output projection (`logits =
        // hidden @ embed_tokens.T`), so no `lm_head` tensor exists in the checkpoint — matching what
        // `config`/`expected_keys` already model (F-138). The 8B-MoT ships an untied `lm_head`; this
        // branch keeps the backbone in sync with the other two layers for a tied NEO checkpoint.
        let lm_head = if cfg.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            require(w, &format!("{prefix}.lm_head.weight"))?
        };
        Ok(Self {
            embed_tokens,
            layers,
            norm: require(w, &format!("{model}.norm.weight"))?,
            norm_gen: require(w, &format!("{model}.norm_mot_gen.weight"))?,
            lm_head,
            num_heads: cfg.llm.num_attention_heads as i32,
            num_kv_heads: cfg.llm.num_key_value_heads as i32,
            head_dim: cfg.llm.head_dim() as i32,
            eps: cfg.llm.rms_norm_eps,
            rope_theta: cfg.llm.rope_theta,
            rope_theta_hw: cfg.llm.rope_theta_hw,
        })
    }

    /// Token embedding: `input_ids` `[B,S]` int32 → `[B,S,hidden]`.
    pub fn embed(&self, input_ids: &Array) -> Result<Array> {
        let sh = input_ids.shape().to_vec();
        let flat = input_ids.reshape(&[-1])?;
        let g = self.embed_tokens.take_axis(&flat, 0)?;
        let h = self.embed_tokens.shape()[1];
        Ok(g.reshape(&[sh[0], sh[1], h])?)
    }

    /// Project final hidden states `[B,S,hidden]` → logits `[B,S,vocab]`.
    pub fn lm_head(&self, hidden: &Array) -> Result<Array> {
        matmul_t(hidden, &self.lm_head)
    }

    /// Quantize the decoder stack to Q4/Q8 (group 64) — every layer's attention projections and
    /// SwiGLU linears on **both** the understanding and generation paths (the bulk of the 8B
    /// params). The token embedding, `lm_head`, RMSNorms, and QK-norms stay dense (precision-
    /// sensitive / not Linears). Weights are bf16-native on disk, so the packing byte-matches the
    /// reference `nn.quantize`.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for layer in &mut self.layers {
            layer.quantize(bits)?;
        }
        Ok(())
    }

    /// Merge the 8-step distill LoRA (sc-3192) into every layer's **generation-path** attention
    /// projections + SwiGLU (`*_mot_gen`); the understanding path is untouched. `prefix` is the
    /// `language_model` namespace (matching [`Qwen3Backbone::from_weights`]). Must run before
    /// [`Qwen3Backbone::quantize`] (the merge seam errors on a quantized base). Returns the total
    /// number of linears merged (`7 · layers` when the LoRA carries every target).
    pub fn merge_distill_lora(&mut self, lora: &Weights, prefix: &str) -> Result<usize> {
        let mut n = 0;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let attn = format!("{prefix}.model.layers.{i}.self_attn");
            n += layer.attn_gen.merge_distill_lora(lora, &attn, "_mot_gen")?;
            let mlp = format!("{prefix}.model.layers.{i}.mlp_mot_gen");
            n += layer.mlp_gen.merge_distill_lora(lora, &mlp)?;
        }
        Ok(n)
    }

    /// Run the full stack on a single path. `embeds` `[B,S,hidden]`; `temporal`/`height`/`width`
    /// are the three position rows (each length `S`).
    pub fn forward_path(
        &self,
        embeds: &Array,
        temporal: &[i32],
        height: &[i32],
        width: &[i32],
        path: Path,
    ) -> Result<Array> {
        // Three RoPE bases over the head_dim split: temporal (head_dim/2), height & width
        // (head_dim/4 each).
        let dt = (self.head_dim / 2) as usize;
        let dhw = (self.head_dim / 4) as usize;
        let (cos_t, sin_t) = rope_cos_sin(temporal, dt, self.rope_theta)?;
        let (cos_h, sin_h) = rope_cos_sin(height, dhw, self.rope_theta_hw)?;
        let (cos_w, sin_w) = rope_cos_sin(width, dhw, self.rope_theta_hw)?;
        let mask = block_causal_mask(temporal)?;

        let mut hidden = embeds.clone();
        for layer in &self.layers {
            let (input_ln, post_ln, attn, mlp) = match path {
                Path::Und => (
                    &layer.input_ln,
                    &layer.post_ln,
                    &layer.attn_und,
                    &layer.mlp_und,
                ),
                Path::Gen => (
                    &layer.input_ln_gen,
                    &layer.post_ln_gen,
                    &layer.attn_gen,
                    &layer.mlp_gen,
                ),
            };
            // Attention sub-block.
            let normed = rms_norm(&hidden, input_ln, self.eps)?;
            let attn_out = self.attention(
                &normed, attn, &cos_t, &sin_t, &cos_h, &sin_h, &cos_w, &sin_w, &mask,
            )?;
            hidden = add(&hidden, &attn_out)?;
            // MLP sub-block.
            let normed = rms_norm(&hidden, post_ln, self.eps)?;
            hidden = add(&hidden, &mlp.forward(&normed)?)?;
        }

        let final_norm = match path {
            Path::Und => &self.norm,
            Path::Gen => &self.norm_gen,
        };
        Ok(rms_norm(&hidden, final_norm, self.eps)?)
    }

    /// A fresh empty cache (one slot per decoder layer).
    pub fn new_cache(&self) -> KvCache {
        KvCache {
            layers: (0..self.layers.len()).map(|_| None).collect(),
            seq_len: 0,
        }
    }

    /// The cached counterpart of [`Qwen3Backbone::forward_path`] — the incremental-decode forward
    /// that backs the AR runtime (sc-3187) and the denoise loop (sc-3188).
    ///
    /// `embeds` `[B, S_new, hidden]` are the **new** tokens; `temporal`/`height`/`width` are their
    /// `(t,h,w)` positions (each length `S_new`). Attention runs the `S_new` queries against the
    /// `cache.len() + S_new` cached-plus-new keys under a mask that lets every new token see all
    /// cached context (the reference's all-zero past block) and applies block-causal masking within
    /// the new tokens (same temporal index → bidirectional, else causal). When `append` is true the
    /// new K/V is persisted (text decode); when false it is used for this forward only (the
    /// `update_cache=False` denoise path). Returns the final-normed hidden states `[B, S_new,
    /// hidden]`; the caller applies [`Qwen3Backbone::lm_head`] for logits.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_cached(
        &self,
        embeds: &Array,
        temporal: &[i32],
        height: &[i32],
        width: &[i32],
        path: Path,
        cache: &mut KvCache,
        append: bool,
    ) -> Result<Array> {
        let rm = self.prepare_rope_mask(temporal, height, width, cache.len())?;
        self.forward_prepared(embeds, &rm, path, cache, append)
    }

    /// Build the tri-axis RoPE tables + block mask for `(temporal, height, width)` position indexes
    /// at cache prefix length `past`. Hoisted out of [`forward_cached`] so the denoise loops can
    /// build it once per cache and reuse it across all steps (F-139); see [`RopeMask`].
    pub fn prepare_rope_mask(
        &self,
        temporal: &[i32],
        height: &[i32],
        width: &[i32],
        past: i32,
    ) -> Result<RopeMask> {
        let dt = (self.head_dim / 2) as usize;
        let dhw = (self.head_dim / 4) as usize;
        let (cos_t, sin_t) = rope_cos_sin(temporal, dt, self.rope_theta)?;
        let (cos_h, sin_h) = rope_cos_sin(height, dhw, self.rope_theta_hw)?;
        let (cos_w, sin_w) = rope_cos_sin(width, dhw, self.rope_theta_hw)?;
        let mask = cached_block_mask(past, temporal)?;
        Ok(RopeMask {
            cos_t,
            sin_t,
            cos_h,
            sin_h,
            cos_w,
            sin_w,
            mask,
            n_tokens: temporal.len() as i32,
        })
    }

    /// The decoder stack over `embeds` using a prebuilt [`RopeMask`]. Identical to [`forward_cached`]
    /// once the RoPE/mask are built; split out so the per-step builds can be hoisted (F-139). The
    /// `RopeMask`'s `past` must match `cache.len()` — true within a denoise run (use-only `append =
    /// false` leaves the cache length fixed).
    pub fn forward_prepared(
        &self,
        embeds: &Array,
        rm: &RopeMask,
        path: Path,
        cache: &mut KvCache,
        append: bool,
    ) -> Result<Array> {
        let mut hidden = embeds.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            let (input_ln, post_ln, attn, mlp) = match path {
                Path::Und => (
                    &layer.input_ln,
                    &layer.post_ln,
                    &layer.attn_und,
                    &layer.mlp_und,
                ),
                Path::Gen => (
                    &layer.input_ln_gen,
                    &layer.post_ln_gen,
                    &layer.attn_gen,
                    &layer.mlp_gen,
                ),
            };
            let normed = rms_norm(&hidden, input_ln, self.eps)?;
            let attn_out = self.attention_cached(
                &normed, attn, &rm.cos_t, &rm.sin_t, &rm.cos_h, &rm.sin_h, &rm.cos_w, &rm.sin_w,
                &rm.mask, cache, i, append,
            )?;
            hidden = add(&hidden, &attn_out)?;
            let normed = rms_norm(&hidden, post_ln, self.eps)?;
            hidden = add(&hidden, &mlp.forward(&normed)?)?;
        }
        if append {
            cache.seq_len += rm.n_tokens;
        }

        let final_norm = match path {
            Path::Und => &self.norm,
            Path::Gen => &self.norm_gen,
        };
        Ok(rms_norm(&hidden, final_norm, self.eps)?)
    }

    /// Cached attention: project the new tokens, RoPE q/k, merge with the cache, GQA-expand, attend.
    #[allow(clippy::too_many_arguments)]
    fn attention_cached(
        &self,
        x: &Array,
        a: &AttnPath,
        cos_t: &Array,
        sin_t: &Array,
        cos_h: &Array,
        sin_h: &Array,
        cos_w: &Array,
        sin_w: &Array,
        mask: &Array,
        cache: &mut KvCache,
        layer_idx: usize,
        append: bool,
    ) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        let hd = self.head_dim;

        // q/k: project + reshape + temporal/spatial norm + tri-axis RoPE, then to [B,H,S,D].
        let q = self
            .qk_rope(
                &a.q_proj.forward(x)?,
                b,
                s,
                self.num_heads,
                &a.q_norm,
                &a.q_norm_hw,
                cos_t,
                sin_t,
                cos_h,
                sin_h,
                cos_w,
                sin_w,
            )?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = self
            .qk_rope(
                &a.k_proj.forward(x)?,
                b,
                s,
                self.num_kv_heads,
                &a.k_norm,
                &a.k_norm_hw,
                cos_t,
                sin_t,
                cos_h,
                sin_h,
                cos_w,
                sin_w,
            )?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = a
            .v_proj
            .forward(x)?
            .reshape(&[b, s, self.num_kv_heads, hd])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Merge with the cache (persist or use-only), then GQA-expand the full K/V.
        let (k_all, v_all) = if append {
            cache.append(layer_idx, k, v)?
        } else {
            cache.extend(layer_idx, &k, &v)?
        };
        let groups = self.num_heads / self.num_kv_heads;
        let k_all = repeat_kv_bhsd(&k_all, groups)?;
        let v_all = repeat_kv_bhsd(&v_all, groups)?;

        let scale = (hd as f32).powf(-0.5);
        let scores = multiply(
            &matmul(&q, &k_all.transpose_axes(&[0, 1, 3, 2])?)?,
            Array::from_f32(scale),
        )?;
        let scores = add(&scores, mask)?;
        let weights = softmax_axis(&scores, -1, true)?;
        let out = matmul(&weights, &v_all)?;
        let out = out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, s, self.num_heads * hd])?;
        a.o_proj.forward(&out)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        x: &Array,
        a: &AttnPath,
        cos_t: &Array,
        sin_t: &Array,
        cos_h: &Array,
        sin_h: &Array,
        cos_w: &Array,
        sin_w: &Array,
        mask: &Array,
    ) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        let hd = self.head_dim;

        // q/k: project, reshape to heads, split temporal/spatial, norm each half, rope, concat.
        let q = self.qk_rope(
            &a.q_proj.forward(x)?,
            b,
            s,
            self.num_heads,
            &a.q_norm,
            &a.q_norm_hw,
            cos_t,
            sin_t,
            cos_h,
            sin_h,
            cos_w,
            sin_w,
        )?;
        let k = self.qk_rope(
            &a.k_proj.forward(x)?,
            b,
            s,
            self.num_kv_heads,
            &a.k_norm,
            &a.k_norm_hw,
            cos_t,
            sin_t,
            cos_h,
            sin_h,
            cos_w,
            sin_w,
        )?;
        // v: project + reshape only (no norm, no rope).
        let v = a
            .v_proj
            .forward(x)?
            .reshape(&[b, s, self.num_kv_heads, hd])?;

        // GQA expand, then [B,H,S,D].
        let groups = self.num_heads / self.num_kv_heads;
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = repeat_kv(&k, groups)?.transpose_axes(&[0, 2, 1, 3])?;
        let v = repeat_kv(&v, groups)?.transpose_axes(&[0, 2, 1, 3])?;

        // Eager attention (matmul → f32 softmax → matmul), matching the reference's eager path.
        let scale = (hd as f32).powf(-0.5);
        let scores = multiply(
            &matmul(&q, &k.transpose_axes(&[0, 1, 3, 2])?)?,
            Array::from_f32(scale),
        )?;
        let scores = add(&scores, mask)?;
        // `precise=true` → softmax accumulated in f32 (matches the reference's f32 softmax).
        let weights = softmax_axis(&scores, -1, true)?;
        let out = matmul(&weights, &v)?; // [B,H,S,D]
        let out = out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, s, self.num_heads * hd])?;
        a.o_proj.forward(&out)
    }

    /// Project→reshape→(temporal/spatial split)→norm halves→rope(t,h,w)→concat. `proj` is the
    /// already-projected `[B, S, H*head_dim]` tensor.
    #[allow(clippy::too_many_arguments)]
    fn qk_rope(
        &self,
        proj: &Array,
        b: i32,
        s: i32,
        heads: i32,
        norm_t: &Array,
        norm_hw: &Array,
        cos_t: &Array,
        sin_t: &Array,
        cos_h: &Array,
        sin_h: &Array,
        cos_w: &Array,
        sin_w: &Array,
    ) -> Result<Array> {
        let x = proj.reshape(&[b, s, heads, self.head_dim])?;
        let halves = split(&x, 2, 3)?; // temporal | spatial, each head_dim/2
        let t = rms_norm(&halves[0], norm_t, self.eps)?;
        let hw = rms_norm(&halves[1], norm_hw, self.eps)?;
        let hw_parts = split(&hw, 2, 3)?; // height | width, each head_dim/4
        let t = apply_rope(&t, cos_t, sin_t)?;
        let h = apply_rope(&hw_parts[0], cos_h, sin_h)?;
        let w = apply_rope(&hw_parts[1], cos_w, sin_w)?;
        concatenate_axis(&[&t, &h, &w], 3).map_err(Error::from)
    }
}

/// Expand `[B,Hkv,S,D]` → `[B,Hkv*groups,S,D]` (GQA), repeating each kv head `groups` times. The
/// cached-attention counterpart of [`repeat_kv`] (which operates on the pre-transpose `[B,S,Hkv,D]`
/// layout); both insert the repeat axis immediately after the head axis, so the resulting head
/// ordering is identical.
fn repeat_kv_bhsd(x: &Array, groups: i32) -> Result<Array> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let sh = x.shape();
    let (b, hkv, s, d) = (sh[0], sh[1], sh[2], sh[3]);
    let x = x.expand_dims(2)?;
    let x = broadcast_to(&x, &[b, hkv, groups, s, d])?;
    Ok(x.reshape(&[b, hkv * groups, s, d])?)
}

/// Cached additive mask `[1,1,S_new, past+S_new]` (0 / -inf). New query row `r` attends to: every
/// cached column (`j < past` — the reference's all-zero past block), and new column `c` iff
/// `temporal[r] == temporal[c]` (same image block → bidirectional) **or** `c <= r` (causal). With
/// `past == 0` this is exactly [`block_causal_mask`].
fn cached_block_mask(past: i32, temporal: &[i32]) -> Result<Array> {
    let q = temporal.len();
    let past = past as usize;
    let k = past + q;
    let mut data = vec![0f32; q * k];
    for r in 0..q {
        for j in 0..k {
            let allowed = if j < past {
                true
            } else {
                let c = j - past;
                temporal[r] == temporal[c] || c <= r
            };
            if !allowed {
                data[r * k + j] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Array::from_slice(&data, &[1, 1, q as i32, k as i32]))
}

/// Block-causal additive mask `[1,1,S,S]` (0 / -inf): token `i` attends to `j` iff
/// `temporal[i] == temporal[j]` (same image block → bidirectional) **or** `j <= i` (causal).
fn block_causal_mask(temporal: &[i32]) -> Result<Array> {
    let s = temporal.len();
    let mut data = vec![0f32; s * s];
    for i in 0..s {
        for j in 0..s {
            let allowed = temporal[i] == temporal[j] || j <= i;
            if !allowed {
                data[i * s + j] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Array::from_slice(&data, &[1, 1, s as i32, s as i32]))
}
