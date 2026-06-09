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

use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::NeoChatConfig;

/// Which transformer path a forward runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Path {
    /// Understanding path (text / image input). The reference `forward_und`.
    Und,
    /// Image-generation path (the `_mot_gen` weights). The reference `forward_gen`.
    Gen,
}

/// `y = x · Wᵀ` for a stored `[out, in]` bias-less Linear.
fn matmul_t(x: &Array, w: &Array) -> Result<Array> {
    Ok(matmul(x, w.t())?)
}

fn require(w: &Weights, key: &str) -> Result<Array> {
    Ok(w.require(key)?.clone())
}

/// The per-path attention weights (one of understanding / generation).
struct AttnPath {
    q_proj: Array,
    k_proj: Array,
    v_proj: Array,
    o_proj: Array,
    q_norm: Array,
    k_norm: Array,
    q_norm_hw: Array,
    k_norm_hw: Array,
}

impl AttnPath {
    /// `attn_prefix` = `…layers.{i}.self_attn`, `s` = the path suffix (`""` or `"_mot_gen"`).
    fn from_weights(w: &Weights, attn_prefix: &str, s: &str) -> Result<Self> {
        Ok(Self {
            q_proj: require(w, &format!("{attn_prefix}.q_proj{s}.weight"))?,
            k_proj: require(w, &format!("{attn_prefix}.k_proj{s}.weight"))?,
            v_proj: require(w, &format!("{attn_prefix}.v_proj{s}.weight"))?,
            o_proj: require(w, &format!("{attn_prefix}.o_proj{s}.weight"))?,
            q_norm: require(w, &format!("{attn_prefix}.q_norm{s}.weight"))?,
            k_norm: require(w, &format!("{attn_prefix}.k_norm{s}.weight"))?,
            q_norm_hw: require(w, &format!("{attn_prefix}.q_norm_hw{s}.weight"))?,
            k_norm_hw: require(w, &format!("{attn_prefix}.k_norm_hw{s}.weight"))?,
        })
    }
}

struct Mlp {
    gate: Array,
    up: Array,
    down: Array,
}

impl Mlp {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate: require(w, &format!("{prefix}.gate_proj.weight"))?,
            up: require(w, &format!("{prefix}.up_proj.weight"))?,
            down: require(w, &format!("{prefix}.down_proj.weight"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let gated = multiply(&silu(&matmul_t(x, &self.gate)?)?, &matmul_t(x, &self.up)?)?;
        matmul_t(&gated, &self.down)
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
            post_ln_gen: require(w, &format!("{prefix}.post_attention_layernorm_mot_gen.weight"))?,
            attn_und: AttnPath::from_weights(w, &attn, "")?,
            attn_gen: AttnPath::from_weights(w, &attn, "_mot_gen")?,
            mlp_und: Mlp::from_weights(w, &format!("{prefix}.mlp"))?,
            mlp_gen: Mlp::from_weights(w, &format!("{prefix}.mlp_mot_gen"))?,
        })
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
/// untied `lm_head`.
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

impl Qwen3Backbone {
    /// Build from a checkpoint, `prefix` = the `language_model` namespace (e.g. `"language_model"`).
    pub fn from_weights(w: &Weights, cfg: &NeoChatConfig, prefix: &str) -> Result<Self> {
        let model = format!("{prefix}.model");
        let layers = (0..cfg.llm.num_hidden_layers)
            .map(|i| Layer::from_weights(w, &format!("{model}.layers.{i}")))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens: require(w, &format!("{model}.embed_tokens.weight"))?,
            layers,
            norm: require(w, &format!("{model}.norm.weight"))?,
            norm_gen: require(w, &format!("{model}.norm_mot_gen.weight"))?,
            lm_head: require(w, &format!("{prefix}.lm_head.weight"))?,
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
                Path::Und => (&layer.input_ln, &layer.post_ln, &layer.attn_und, &layer.mlp_und),
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
            &matmul_t(x, &a.q_proj)?,
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
            &matmul_t(x, &a.k_proj)?,
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
        let v = matmul_t(x, &a.v_proj)?.reshape(&[b, s, self.num_kv_heads, hd])?;

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
        matmul_t(&out, &a.o_proj)
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
