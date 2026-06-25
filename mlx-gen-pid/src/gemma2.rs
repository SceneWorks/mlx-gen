//! Gemma-2-2B-IT decoder — PiD's caption text encoder. Port of HF `Gemma2Model` (the
//! `.get_decoder()` stack PiD loads: embedding → 26 norm-sandwich decoder layers → final RMSNorm →
//! last-hidden `[B, L, 2304]`; no lm_head / final-logit-softcap needed).
//!
//! Gemma-2 specifics (vs the Gemma-3 LTX port): **attention logit soft-capping** `50·tanh(s/50)`
//! pre-softmax (so no fused SDPA — explicit attention), **no q/k norm**, RoPE is the standard HF
//! **rotate_half** convention (not PiD's interleaved), attention scale `query_pre_attn_scalar^-0.5`,
//! GQA (8 query / 4 KV heads, head_dim 256 — independent of hidden 2304), gelu-tanh MLP, and the
//! norm-sandwich block (input → attn → post_attn → +res → pre_ff → mlp → post_ff → +res). RMSNorm is
//! Gemma's `x·rsqrt(mean(x²)+eps)·(1+w)`; token embeddings are scaled by `√hidden_size`.
//!
//! PiD captions are ≤300 tokens ≪ the 4096 sliding window, so every layer is plain full-causal —
//! the local/global distinction collapses and a single causal (+ optional padding) mask suffices.

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::{
    add, broadcast_to, concatenate_axis, matmul, multiply, negative, softmax_axis, split, tanh,
};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::array::scalar;
use mlx_gen::nn::gelu_tanh;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// Gemma-2 decoder configuration.
#[derive(Debug, Clone)]
pub struct Gemma2Config {
    pub hidden_size: i32,
    pub num_layers: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub intermediate_size: i32,
    pub rope_theta: f32,
    pub attn_softcap: f32,
    pub query_pre_attn_scalar: f32,
    pub rms_eps: f32,
}

impl Gemma2Config {
    /// The released `gemma-2-2b-it` config.
    pub fn gemma_2_2b() -> Self {
        Self {
            hidden_size: 2304,
            num_layers: 26,
            num_heads: 8,
            num_kv_heads: 4,
            head_dim: 256,
            intermediate_size: 9216,
            rope_theta: 10000.0,
            attn_softcap: 50.0,
            query_pre_attn_scalar: 256.0,
            rms_eps: 1e-6,
        }
    }
}

fn lin(w: &Weights, key: &str) -> Result<AdaptableLinear> {
    Ok(AdaptableLinear::dense(w.require(key)?.clone(), None))
}

/// Gemma RMSNorm with the precomputed `(1 + weight)`. The reference computes the normalization in
/// **fp32** (`_norm(x.float()) · (1 + w.float())`) then casts back to the input dtype — load-bearing
/// on bf16: a bf16-internal reduction drifts hugely over Gemma-2's 104 norms with O(50) activations
/// (measured: 18% peak-rel). We upcast x + weight to f32, normalize, then cast back (f32 path is a
/// no-op cast, so the tiny f32 fixture is unaffected).
fn rms(x: &Array, one_plus_w: &Array, eps: f32) -> Result<Array> {
    let xf = x.as_dtype(Dtype::Float32)?;
    let wf = one_plus_w.as_dtype(Dtype::Float32)?;
    Ok(rms_norm(&xf, &wf, eps)?.as_dtype(x.dtype())?)
}

/// Host `(cos, sin)` `[seq, head_dim]` for HF rotate_half RoPE: `emb = cat(freqs, freqs)`.
fn rope_tables(head_dim: i32, seq: i32, theta: f32) -> (Array, Array) {
    let half = (head_dim / 2) as usize;
    let inv: Vec<f64> = (0..half)
        .map(|i| 1.0 / (theta as f64).powf((2 * i) as f64 / head_dim as f64))
        .collect();
    let hd = head_dim as usize;
    let s = seq as usize;
    let mut cos = vec![0f32; s * hd];
    let mut sin = vec![0f32; s * hd];
    for p in 0..s {
        for j in 0..hd {
            let f = inv[j % half]; // emb = cat(freqs, freqs) -> index wraps at half
            let a = p as f64 * f;
            cos[p * hd + j] = a.cos() as f32;
            sin[p * hd + j] = a.sin() as f32;
        }
    }
    (
        Array::from_slice(&cos, &[seq, head_dim]),
        Array::from_slice(&sin, &[seq, head_dim]),
    )
}

/// `rotate_half(x) = cat(-x[..,h:], x[..,:h])` for `[B,H,L,D]`.
fn rotate_half(x: &Array) -> Result<Array> {
    let p = split(x, 2, 3)?; // [x1, x2] halves along the head-dim axis
    Ok(concatenate_axis(&[&negative(&p[1])?, &p[0]], 3)?)
}

/// `q·cos + rotate_half(q)·sin` with `cos`/`sin` `[L, D]` broadcast over `[B,H,L,D]`.
fn apply_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let (l, d) = (cos.shape()[0], cos.shape()[1]);
    let cos = cos.reshape(&[1, 1, l, d])?;
    let sin = sin.reshape(&[1, 1, l, d])?;
    let xc = multiply(x, &cos.as_dtype(x.dtype())?)?;
    let xs = multiply(&rotate_half(x)?, &sin.as_dtype(x.dtype())?)?;
    Ok(add(&xc, &xs)?)
}

/// Repeat KV heads `n_rep×` along the head axis (`[B,nkv,L,D]` → `[B,nkv·n_rep,L,D]`), matching HF
/// `repeat_kv` (kv head j → q heads `[j·n_rep .. (j+1)·n_rep)`).
fn repeat_kv(x: &Array, n_rep: i32) -> Result<Array> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let sh = x.shape();
    let (b, nkv, l, d) = (sh[0], sh[1], sh[2], sh[3]);
    let expanded = broadcast_to(&x.reshape(&[b, nkv, 1, l, d])?, &[b, nkv, n_rep, l, d])?;
    Ok(expanded.reshape(&[b, nkv * n_rep, l, d])?)
}

struct Attention {
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    o: AdaptableLinear,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    softcap: f32,
}

impl Attention {
    fn from_weights(w: &Weights, prefix: &str, cfg: &Gemma2Config) -> Result<Self> {
        Ok(Self {
            q: lin(w, &format!("{prefix}.q_proj.weight"))?,
            k: lin(w, &format!("{prefix}.k_proj.weight"))?,
            v: lin(w, &format!("{prefix}.v_proj.weight"))?,
            o: lin(w, &format!("{prefix}.o_proj.weight"))?,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            scale: cfg.query_pre_attn_scalar.powf(-0.5),
            softcap: cfg.attn_softcap,
        })
    }

    /// `x`: `[B,L,hidden]`; `cos`/`sin`: `[L,head_dim]`; `mask`: additive `[1,1,L,L]` (or broadcast).
    fn forward(&self, x: &Array, cos: &Array, sin: &Array, mask: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, l) = (sh[0], sh[1]);
        let hd = self.head_dim;
        let to_heads = |a: Array, nh: i32| -> Result<Array> {
            Ok(a.reshape(&[b, l, nh, hd])?.transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = apply_rope(&to_heads(self.q.forward(x)?, self.num_heads)?, cos, sin)?;
        let k = apply_rope(&to_heads(self.k.forward(x)?, self.num_kv_heads)?, cos, sin)?;
        let v = to_heads(self.v.forward(x)?, self.num_kv_heads)?;
        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;

        // explicit attention (logit soft-cap blocks fused SDPA)
        let scores = multiply(
            &matmul(&q, &k.transpose_axes(&[0, 1, 3, 2])?)?,
            &scalar(self.scale).as_dtype(q.dtype())?,
        )?;
        // softcap·tanh(scores/softcap)
        let cap = scalar(self.softcap).as_dtype(scores.dtype())?;
        let scores = multiply(
            &tanh(&multiply(
                &scores,
                &scalar(1.0 / self.softcap).as_dtype(scores.dtype())?,
            )?)?,
            &cap,
        )?;
        // softmax in f32 then back to the input dtype (HF eager: `softmax(..., dtype=float32)`).
        let scores = add(
            &scores.as_dtype(Dtype::Float32)?,
            &mask.as_dtype(Dtype::Float32)?,
        )?;
        let attn = softmax_axis(&scores, -1, false)?.as_dtype(x.dtype())?;
        let out = matmul(&attn, &v)?; // [B,H,L,D]
        let out = out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, l, self.num_heads * hd])?;
        self.o.forward(&out)
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
            gate: lin(w, &format!("{prefix}.gate_proj.weight"))?,
            up: lin(w, &format!("{prefix}.up_proj.weight"))?,
            down: lin(w, &format!("{prefix}.down_proj.weight"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let g = gelu_tanh(&self.gate.forward(x)?)?;
        self.down.forward(&multiply(&g, &self.up.forward(x)?)?)
    }
}

struct Layer {
    input_ln: Array,
    attn: Attention,
    post_attn_ln: Array,
    pre_ff_ln: Array,
    mlp: Mlp,
    post_ff_ln: Array,
    eps: f32,
}

impl Layer {
    fn from_weights(w: &Weights, prefix: &str, cfg: &Gemma2Config) -> Result<Self> {
        let onep = |key: &str| -> Result<Array> {
            Ok(add(
                w.require(key)?,
                &scalar(1.0).as_dtype(w.require(key)?.dtype())?,
            )?)
        };
        Ok(Self {
            input_ln: onep(&format!("{prefix}.input_layernorm.weight"))?,
            attn: Attention::from_weights(w, &format!("{prefix}.self_attn"), cfg)?,
            post_attn_ln: onep(&format!("{prefix}.post_attention_layernorm.weight"))?,
            pre_ff_ln: onep(&format!("{prefix}.pre_feedforward_layernorm.weight"))?,
            mlp: Mlp::from_weights(w, &format!("{prefix}.mlp"))?,
            post_ff_ln: onep(&format!("{prefix}.post_feedforward_layernorm.weight"))?,
            eps: cfg.rms_eps,
        })
    }

    fn forward(&self, x: &Array, cos: &Array, sin: &Array, mask: &Array) -> Result<Array> {
        let h = self
            .attn
            .forward(&rms(x, &self.input_ln, self.eps)?, cos, sin, mask)?;
        let x = add(x, &rms(&h, &self.post_attn_ln, self.eps)?)?;
        let h = self.mlp.forward(&rms(&x, &self.pre_ff_ln, self.eps)?)?;
        Ok(add(&x, &rms(&h, &self.post_ff_ln, self.eps)?)?)
    }
}

/// The Gemma-2 decoder (caption encoder).
pub struct Gemma2 {
    embed: Array, // [vocab, hidden]
    layers: Vec<Layer>,
    norm: Array,
    cfg: Gemma2Config,
}

impl Gemma2 {
    /// `prefix` is `"model."` for the HF checkpoint layout.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &Gemma2Config) -> Result<Self> {
        let layers = (0..cfg.num_layers)
            .map(|i| Layer::from_weights(w, &format!("{prefix}layers.{i}"), cfg))
            .collect::<Result<Vec<_>>>()?;
        let norm = add(
            w.require(&format!("{prefix}norm.weight"))?,
            &scalar(1.0).as_dtype(w.require(&format!("{prefix}norm.weight"))?.dtype())?,
        )?;
        Ok(Self {
            embed: w.require(&format!("{prefix}embed_tokens.weight"))?.clone(),
            layers,
            norm,
            cfg: cfg.clone(),
        })
    }

    /// `ids`: `[B, L]` (i32). `pad_mask`: optional `[B, L]` (1 = real, 0 = pad). Returns the
    /// last-hidden states `[B, L, hidden]` in the embedding dtype.
    pub fn forward(&self, ids: &Array, pad_mask: Option<&Array>) -> Result<Array> {
        let sh = ids.shape();
        let (b, l) = (sh[0], sh[1]);
        let hidden = self.cfg.hidden_size;

        // embed + √hidden scale (cast to embedding dtype, per the reference)
        let flat = ids.reshape(&[b * l])?;
        let emb = self.embed.take_axis(&flat, 0)?.reshape(&[b, l, hidden])?;
        let normalizer = scalar((hidden as f32).sqrt()).as_dtype(emb.dtype())?;
        let mut x = multiply(&emb, &normalizer)?;

        let (cos, sin) = rope_tables(self.cfg.head_dim, l, self.cfg.rope_theta);
        let mask = causal_mask(b, l, pad_mask)?;
        for layer in &self.layers {
            x = layer.forward(&x, &cos, &sin, &mask)?;
        }
        rms(&x, &self.norm, self.cfg.rms_eps)
    }
}

/// Additive `[B,1,L,L]` causal mask (0 where a query may attend, large-negative otherwise),
/// optionally also masking padding keys (`pad_mask[b,j]==0`).
fn causal_mask(b: i32, l: i32, pad_mask: Option<&Array>) -> Result<Array> {
    let neg = -1e9f32;
    let mut m = vec![0f32; (l * l) as usize];
    for i in 0..l as usize {
        for j in 0..l as usize {
            if j > i {
                m[i * l as usize + j] = neg;
            }
        }
    }
    let causal = Array::from_slice(&m, &[1, 1, l, l]);
    match pad_mask {
        None => Ok(causal),
        Some(pm) => {
            // pad_mask [B,L] (1 real / 0 pad) -> additive key mask [B,1,1,L]
            let pad = pm.as_dtype(Dtype::Float32)?.reshape(&[b, 1, 1, l])?;
            let one = scalar(1.0);
            let key_add = multiply(&mlx_rs::ops::subtract(&one, &pad)?, scalar(neg))?;
            Ok(add(&causal, &key_add)?)
        }
    }
}
