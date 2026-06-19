//! Boogu DiT building blocks: GQA self-attention, the dual-stream joint attention, the SwiGLU FFN,
//! the `LuminaRMSNormZero` modulation, and the three block flavours (plain/context, modulated
//! single-stream, double-stream).
//!
//! All attention is **bidirectional** (no causal mask) and, for the per-sample `B = 1` path, fully
//! unmasked (every token valid) — so SDPA takes no mask. Per-head q/k RMSNorm runs over the head dim
//! before the interleaved RoPE; GQA repeats each kv head to match the query heads (matching the
//! reference's explicit `repeat_interleave`).

use mlx_rs::fast::{rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, concatenate_axis, multiply, tanh};
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::rope::apply_interleaved_rope;
use super::{join, repeat_kv, slice_axis1};
use crate::quant::lin;

/// diffusers `Attention(eps=1e-5)` — the per-head q/k RMSNorm epsilon (distinct from the block
/// RMSNorm `norm_eps`, which is also 1e-5 here but conceptually separate).
const QK_EPS: f32 = 1e-5;

/// `1.0 + a`, broadcasting the scalar (used for the `(1 + scale)` modulation factors).
fn plus1(a: &Array) -> Result<Array> {
    Ok(add(a, Array::from_f32(1.0))?)
}

// ── GQA self-attention (standard `BooguImageAttnProcessor`) ─────────────────────────────────
pub struct SelfAttention {
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    o: AdaptableLinear,
    norm_q: Array,
    norm_k: Array,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl SelfAttention {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
    ) -> Result<Self> {
        Ok(Self {
            q: lin(w, &join(prefix, "to_q"), false)?,
            k: lin(w, &join(prefix, "to_k"), false)?,
            v: lin(w, &join(prefix, "to_v"), false)?,
            o: lin(w, &join(prefix, "to_out.0"), false)?,
            norm_q: w.require(&join(prefix, "norm_q.weight"))?.clone(),
            norm_k: w.require(&join(prefix, "norm_k.weight"))?.clone(),
            heads,
            kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// `x`: `[b, s, hidden]`; `cos`/`sin`: `[1, s, head_dim/2]`. Unmasked (B=1 full sequence).
    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        let q = self
            .q
            .forward(x)?
            .reshape(&[b, s, self.heads, self.head_dim])?;
        let k = self
            .k
            .forward(x)?
            .reshape(&[b, s, self.kv_heads, self.head_dim])?;
        let v = self
            .v
            .forward(x)?
            .reshape(&[b, s, self.kv_heads, self.head_dim])?;

        let q = rms_norm(&q, &self.norm_q, QK_EPS)?;
        let k = rms_norm(&k, &self.norm_k, QK_EPS)?;
        let q = apply_interleaved_rope(&q, cos, sin)?;
        let k = apply_interleaved_rope(&k, cos, sin)?;

        let groups = self.heads / self.kv_heads;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;

        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;
        let o = scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?;
        let o = o
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, s, self.heads * self.head_dim])?;
        self.o.forward(&o)
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.q.quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        self.k.quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        self.v.quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        self.o.quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        Ok(())
    }
}

// ── Dual-stream joint attention (`BooguImageDoubleStreamSelfAttnProcessor`) ──────────────────
/// Separate img/instruct QKV projections; the streams are concatenated **instruct-first**, attended
/// jointly, split back, projected by separate `img_out`/`instruct_out`, re-merged, and run through
/// the shared `to_out.0`. The block then re-splits the result into its instruct/img halves.
pub struct JointAttention {
    img_q: AdaptableLinear,
    img_k: AdaptableLinear,
    img_v: AdaptableLinear,
    instruct_q: AdaptableLinear,
    instruct_k: AdaptableLinear,
    instruct_v: AdaptableLinear,
    img_out: AdaptableLinear,
    instruct_out: AdaptableLinear,
    to_out: AdaptableLinear,
    norm_q: Array,
    norm_k: Array,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl JointAttention {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
    ) -> Result<Self> {
        let p = |s: &str| join(prefix, s);
        Ok(Self {
            img_q: lin(w, &p("processor.img_to_q"), false)?,
            img_k: lin(w, &p("processor.img_to_k"), false)?,
            img_v: lin(w, &p("processor.img_to_v"), false)?,
            instruct_q: lin(w, &p("processor.instruct_to_q"), false)?,
            instruct_k: lin(w, &p("processor.instruct_to_k"), false)?,
            instruct_v: lin(w, &p("processor.instruct_to_v"), false)?,
            img_out: lin(w, &p("processor.img_out"), false)?,
            instruct_out: lin(w, &p("processor.instruct_out"), false)?,
            to_out: lin(w, &p("to_out.0"), false)?,
            norm_q: w.require(&p("norm_q.weight"))?.clone(),
            norm_k: w.require(&p("norm_k.weight"))?.clone(),
            heads,
            kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// `img`: `[b, Li, D]`, `instruct`: `[b, Lt, D]`, joint `cos`/`sin`: `[1, Lt+Li, head_dim/2]`.
    /// Returns the joint attention output `[b, Lt+Li, D]` (instruct-first).
    pub fn forward(
        &self,
        img: &Array,
        instruct: &Array,
        cos: &Array,
        sin: &Array,
    ) -> Result<Array> {
        let b = img.shape()[0];
        let (li, lt) = (img.shape()[1], instruct.shape()[1]);
        let to_heads = |x: &Array, proj: &AdaptableLinear, n: i32, l: i32| -> Result<Array> {
            Ok(proj.forward(x)?.reshape(&[b, l, n, self.head_dim])?)
        };

        // Concatenate instruct-first along the sequence axis, then split into heads.
        let q = concatenate_axis(
            &[
                &to_heads(instruct, &self.instruct_q, self.heads, lt)?,
                &to_heads(img, &self.img_q, self.heads, li)?,
            ],
            1,
        )?;
        let k = concatenate_axis(
            &[
                &to_heads(instruct, &self.instruct_k, self.kv_heads, lt)?,
                &to_heads(img, &self.img_k, self.kv_heads, li)?,
            ],
            1,
        )?;
        let v = concatenate_axis(
            &[
                &to_heads(instruct, &self.instruct_v, self.kv_heads, lt)?,
                &to_heads(img, &self.img_v, self.kv_heads, li)?,
            ],
            1,
        )?;

        let q = rms_norm(&q, &self.norm_q, QK_EPS)?;
        let k = rms_norm(&k, &self.norm_k, QK_EPS)?;
        let q = apply_interleaved_rope(&q, cos, sin)?;
        let k = apply_interleaved_rope(&k, cos, sin)?;

        let groups = self.heads / self.kv_heads;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;

        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;
        let o = scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?;
        let o =
            o.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, lt + li, self.heads * self.head_dim])?;

        // Split → separate output projections → re-merge → shared output projection.
        let instruct_part = slice_axis1(&o, 0, lt)?;
        let img_part = slice_axis1(&o, lt, lt + li)?;
        let merged = concatenate_axis(
            &[
                &self.instruct_out.forward(&instruct_part)?,
                &self.img_out.forward(&img_part)?,
            ],
            1,
        )?;
        self.to_out.forward(&merged)
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for p in [
            &mut self.img_q,
            &mut self.img_k,
            &mut self.img_v,
            &mut self.instruct_q,
            &mut self.instruct_k,
            &mut self.instruct_v,
            &mut self.img_out,
            &mut self.instruct_out,
            &mut self.to_out,
        ] {
            p.quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        }
        Ok(())
    }
}

// ── SwiGLU feed-forward (`LuminaFeedForward`) ───────────────────────────────────────────────
pub struct SwiGlu {
    w1: AdaptableLinear,
    w2: AdaptableLinear,
    w3: AdaptableLinear,
}

impl SwiGlu {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w1: lin(w, &join(prefix, "linear_1"), false)?,
            w2: lin(w, &join(prefix, "linear_2"), false)?,
            w3: lin(w, &join(prefix, "linear_3"), false)?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let gated = multiply(&silu(&self.w1.forward(x)?)?, &self.w3.forward(x)?)?;
        self.w2.forward(&gated)
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.w1.quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        self.w2.quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        self.w3.quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        Ok(())
    }
}

// ── LuminaRMSNormZero modulation ────────────────────────────────────────────────────────────
/// `emb = linear(silu(temb))` (`1024 → 4·D`), chunked into `(scale_msa, gate_msa, scale_mlp,
/// gate_mlp)`; the returned hidden is `RMSNorm(x)·(1 + scale_msa)`. The caller reuses the other three
/// chunks per its modulation pattern (different blocks read different chunk slots).
pub struct ModNorm {
    linear: AdaptableLinear,
    norm: Array,
    eps: f32,
}

impl ModNorm {
    pub fn from_weights(w: &Weights, prefix: &str, eps: f32) -> Result<Self> {
        Ok(Self {
            linear: lin(w, &join(prefix, "linear"), true)?,
            norm: w.require(&join(prefix, "norm.weight"))?.clone(),
            eps,
        })
    }

    /// `x`: `[b, s, D]`, `temb`: `[b, 1, 1024]`. Returns `(normed, c2, c3, c4)`, each `[b, 1, D]`
    /// except `normed` which is `[b, s, D]`.
    pub fn forward(&self, x: &Array, temb: &Array) -> Result<(Array, Array, Array, Array)> {
        let emb = self.linear.forward(&silu(temb)?)?; // [b, 1, 4D]
        let chunks = mlx_rs::ops::split(&emb, 4, 2)?;
        let normed = multiply(&rms_norm(x, &self.norm, self.eps)?, &plus1(&chunks[0])?)?;
        Ok((
            normed,
            chunks[1].clone(),
            chunks[2].clone(),
            chunks[3].clone(),
        ))
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.linear.quantize(bits, Some(crate::quant::GROUP_SIZE))
    }
}

// ── Plain (non-modulated) block — context refiner ───────────────────────────────────────────
pub struct PlainBlock {
    attn: SelfAttention,
    ff: SwiGlu,
    norm1: Array,
    norm2: Array,
    ffn_norm1: Array,
    ffn_norm2: Array,
    eps: f32,
}

impl PlainBlock {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
        eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            attn: SelfAttention::from_weights(w, &join(prefix, "attn"), heads, kv_heads, head_dim)?,
            ff: SwiGlu::from_weights(w, &join(prefix, "feed_forward"))?,
            norm1: w.require(&join(prefix, "norm1.weight"))?.clone(),
            norm2: w.require(&join(prefix, "norm2.weight"))?.clone(),
            ffn_norm1: w.require(&join(prefix, "ffn_norm1.weight"))?.clone(),
            ffn_norm2: w.require(&join(prefix, "ffn_norm2.weight"))?.clone(),
            eps,
        })
    }

    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let attn = self
            .attn
            .forward(&rms_norm(x, &self.norm1, self.eps)?, cos, sin)?;
        let x = add(x, &rms_norm(&attn, &self.norm2, self.eps)?)?;
        let mlp = self.ff.forward(&rms_norm(&x, &self.ffn_norm1, self.eps)?)?;
        Ok(add(&x, &rms_norm(&mlp, &self.ffn_norm2, self.eps)?)?)
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attn.quantize(bits)?;
        self.ff.quantize(bits)
    }
}

// ── Modulated single-stream / noise-refiner block ───────────────────────────────────────────
pub struct ModBlock {
    attn: SelfAttention,
    ff: SwiGlu,
    norm1: ModNorm,
    norm2: Array,
    ffn_norm1: Array,
    ffn_norm2: Array,
    eps: f32,
}

impl ModBlock {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
        eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            attn: SelfAttention::from_weights(w, &join(prefix, "attn"), heads, kv_heads, head_dim)?,
            ff: SwiGlu::from_weights(w, &join(prefix, "feed_forward"))?,
            norm1: ModNorm::from_weights(w, &join(prefix, "norm1"), eps)?,
            norm2: w.require(&join(prefix, "norm2.weight"))?.clone(),
            ffn_norm1: w.require(&join(prefix, "ffn_norm1.weight"))?.clone(),
            ffn_norm2: w.require(&join(prefix, "ffn_norm2.weight"))?.clone(),
            eps,
        })
    }

    /// `x`: `[b, s, D]`, `temb`: `[b, 1, 1024]`.
    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array, temb: &Array) -> Result<Array> {
        let (normed, gate_msa, scale_mlp, gate_mlp) = self.norm1.forward(x, temb)?;
        let attn = self.attn.forward(&normed, cos, sin)?;
        let x = add(
            x,
            &multiply(&tanh(&gate_msa)?, &rms_norm(&attn, &self.norm2, self.eps)?)?,
        )?;
        let mlp_in = multiply(
            &rms_norm(&x, &self.ffn_norm1, self.eps)?,
            &plus1(&scale_mlp)?,
        )?;
        let mlp = self.ff.forward(&mlp_in)?;
        Ok(add(
            &x,
            &multiply(
                &tanh(&gate_mlp)?,
                &rms_norm(&mlp, &self.ffn_norm2, self.eps)?,
            )?,
        )?)
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attn.quantize(bits)?;
        self.ff.quantize(bits)?;
        self.norm1.quantize(bits)
    }
}

// ── Double-stream block ─────────────────────────────────────────────────────────────────────
pub struct DoubleBlock {
    joint_attn: JointAttention,
    self_attn: SelfAttention,
    img_ff: SwiGlu,
    instruct_ff: SwiGlu,
    img_norm1: ModNorm,
    img_norm2: ModNorm,
    img_norm3: ModNorm,
    instruct_norm1: ModNorm,
    instruct_norm2: ModNorm,
    img_attn_norm: Array,
    img_self_attn_norm: Array,
    img_ffn_norm1: Array,
    img_ffn_norm2: Array,
    instruct_attn_norm: Array,
    instruct_ffn_norm1: Array,
    instruct_ffn_norm2: Array,
    eps: f32,
}

impl DoubleBlock {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
        eps: f32,
    ) -> Result<Self> {
        let req = |s: &str| -> Result<Array> { Ok(w.require(&join(prefix, s))?.clone()) };
        Ok(Self {
            joint_attn: JointAttention::from_weights(
                w,
                &join(prefix, "img_instruct_attn"),
                heads,
                kv_heads,
                head_dim,
            )?,
            self_attn: SelfAttention::from_weights(
                w,
                &join(prefix, "img_self_attn"),
                heads,
                kv_heads,
                head_dim,
            )?,
            img_ff: SwiGlu::from_weights(w, &join(prefix, "img_feed_forward"))?,
            instruct_ff: SwiGlu::from_weights(w, &join(prefix, "instruct_feed_forward"))?,
            img_norm1: ModNorm::from_weights(w, &join(prefix, "img_norm1"), eps)?,
            img_norm2: ModNorm::from_weights(w, &join(prefix, "img_norm2"), eps)?,
            img_norm3: ModNorm::from_weights(w, &join(prefix, "img_norm3"), eps)?,
            instruct_norm1: ModNorm::from_weights(w, &join(prefix, "instruct_norm1"), eps)?,
            instruct_norm2: ModNorm::from_weights(w, &join(prefix, "instruct_norm2"), eps)?,
            img_attn_norm: req("img_attn_norm.weight")?,
            img_self_attn_norm: req("img_self_attn_norm.weight")?,
            img_ffn_norm1: req("img_ffn_norm1.weight")?,
            img_ffn_norm2: req("img_ffn_norm2.weight")?,
            instruct_attn_norm: req("instruct_attn_norm.weight")?,
            instruct_ffn_norm1: req("instruct_ffn_norm1.weight")?,
            instruct_ffn_norm2: req("instruct_ffn_norm2.weight")?,
            eps,
        })
    }

    /// `img`: `[b, Li, D]`, `instruct`: `[b, Lt, D]`; joint `cos`/`sin`: `[1, Lt+Li, head_dim/2]`;
    /// image `img_cos`/`img_sin`: `[1, Li, head_dim/2]`; `temb`: `[b, 1, 1024]`.
    /// Returns the updated `(img, instruct)`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        img: &Array,
        instruct: &Array,
        cos: &Array,
        sin: &Array,
        img_cos: &Array,
        img_sin: &Array,
        temb: &Array,
    ) -> Result<(Array, Array)> {
        let lt = instruct.shape()[1];
        let li = img.shape()[1];

        let (img_n1, img_gate_msa, img_scale_mlp, img_gate_mlp) =
            self.img_norm1.forward(img, temb)?;
        let (img_n2, img_shift_mlp, _, _) = self.img_norm2.forward(img, temb)?;
        let (img_n3, img_gate_self, _, _) = self.img_norm3.forward(img, temb)?;
        let (ins_n1, ins_gate_msa, ins_scale_mlp, ins_gate_mlp) =
            self.instruct_norm1.forward(instruct, temb)?;
        let (ins_n2, ins_shift_mlp, _, _) = self.instruct_norm2.forward(instruct, temb)?;

        // Joint instruct↔img attention, then split back to the two streams.
        let joint = self.joint_attn.forward(&img_n1, &ins_n1, cos, sin)?;
        let instruct_attn_out = slice_axis1(&joint, 0, lt)?;
        let img_attn_out = slice_axis1(&joint, lt, lt + li)?;

        // Image self-attention.
        let img_self_out = self.self_attn.forward(&img_n3, img_cos, img_sin)?;

        // Image residual updates.
        let img = add(
            img,
            &multiply(
                &tanh(&img_gate_msa)?,
                &rms_norm(&img_attn_out, &self.img_attn_norm, self.eps)?,
            )?,
        )?;
        let img = add(
            &img,
            &multiply(
                &tanh(&img_gate_self)?,
                &rms_norm(&img_self_out, &self.img_self_attn_norm, self.eps)?,
            )?,
        )?;
        let img_mlp_in = add(&multiply(&img_n2, &plus1(&img_scale_mlp)?)?, &img_shift_mlp)?;
        let img_mlp =
            self.img_ff
                .forward(&rms_norm(&img_mlp_in, &self.img_ffn_norm1, self.eps)?)?;
        let img = add(
            &img,
            &multiply(
                &tanh(&img_gate_mlp)?,
                &rms_norm(&img_mlp, &self.img_ffn_norm2, self.eps)?,
            )?,
        )?;

        // Instruction residual updates.
        let instruct = add(
            instruct,
            &multiply(
                &tanh(&ins_gate_msa)?,
                &rms_norm(&instruct_attn_out, &self.instruct_attn_norm, self.eps)?,
            )?,
        )?;
        let ins_mlp_in = add(&multiply(&ins_n2, &plus1(&ins_scale_mlp)?)?, &ins_shift_mlp)?;
        let ins_mlp = self.instruct_ff.forward(&rms_norm(
            &ins_mlp_in,
            &self.instruct_ffn_norm1,
            self.eps,
        )?)?;
        let instruct = add(
            &instruct,
            &multiply(
                &tanh(&ins_gate_mlp)?,
                &rms_norm(&ins_mlp, &self.instruct_ffn_norm2, self.eps)?,
            )?,
        )?;

        Ok((img, instruct))
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.joint_attn.quantize(bits)?;
        self.self_attn.quantize(bits)?;
        self.img_ff.quantize(bits)?;
        self.instruct_ff.quantize(bits)?;
        self.img_norm1.quantize(bits)?;
        self.img_norm2.quantize(bits)?;
        self.img_norm3.quantize(bits)?;
        self.instruct_norm1.quantize(bits)?;
        self.instruct_norm2.quantize(bits)?;
        Ok(())
    }
}
