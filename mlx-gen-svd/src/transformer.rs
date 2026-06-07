//! SVD `TransformerSpatioTemporalModel` (sc-3374) — a spatial `BasicTransformerBlock` (self-attn +
//! cross-attn to the CLIP `image_embeds` + GEGLU ff) and a `TemporalBasicTransformerBlock` (ff_in +
//! self-attn over the frame axis + cross-attn + ff), blended per layer by an `AlphaBlender`
//! (`merge_strategy="learned_with_images"`, no switch → `σ(mix)·spatial + (1−σ)·temporal` at
//! inference). Port of diffusers `transformer_temporal.TransformerSpatioTemporalModel` +
//! `attention.{BasicTransformerBlock,TemporalBasicTransformerBlock}`. NHWC I/O.

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, broadcast_to, matmul, multiply, sigmoid, subtract};
use mlx_rs::Array;

use mlx_gen::array::scalar;
use mlx_gen::nn::{gelu_exact, group_norm, linear};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::embeddings::{sinusoidal_timestep, TimestepEmbedding};

const GN_GROUPS: i32 = 32;
const GN_EPS: f32 = 1e-6;
const LN_EPS: f32 = 1e-5;

/// Load a `(weight, bias)` Linear.
fn lin(w: &Weights, name: &str) -> Result<(Array, Array)> {
    Ok((
        w.require(&format!("{name}.weight"))?.clone(),
        w.require(&format!("{name}.bias"))?.clone(),
    ))
}

/// GEGLU feed-forward: `proj` is `ff.net.0.proj` (`[2·inner, D]`), split into value/gate halves;
/// `out` is `ff.net.2`. `value · gelu(gate) → out`.
fn geglu(x: &Array, proj: &(Array, Array), out: &(Array, Array)) -> Result<Array> {
    let p = linear(x, &proj.0, &proj.1)?;
    let last = (p.shape().len() - 1) as i32;
    let two_inner = p.shape()[p.shape().len() - 1];
    let inner = two_inner / 2;
    let v_idx = Array::from_slice(&(0..inner).collect::<Vec<i32>>(), &[inner]);
    let g_idx = Array::from_slice(&(inner..two_inner).collect::<Vec<i32>>(), &[inner]);
    let value = p.take_axis(&v_idx, last)?;
    let gate = p.take_axis(&g_idx, last)?;
    let y = multiply(&value, &gelu_exact(&gate)?)?;
    linear(&y, &out.0, &out.1)
}

/// Multi-head attention: bias-free q/k/v, biased `to_out.0`, no mask. `head_dim = inner/heads`,
/// `scale = head_dim^-0.5`. Self-attn passes `context == x`; cross-attn passes the memory.
struct Attention {
    q: Array,
    k: Array,
    v: Array,
    out: (Array, Array),
    heads: i32,
}

impl Attention {
    fn from_weights(w: &Weights, prefix: &str, heads: i32) -> Result<Self> {
        let req = |n: &str| w.require(&format!("{prefix}.{n}.weight")).cloned();
        Ok(Self {
            q: req("to_q")?,
            k: req("to_k")?,
            v: req("to_v")?,
            out: lin(w, &format!("{prefix}.to_out.0"))?,
            heads,
        })
    }

    /// `x`: `[B, Lq, Dq]`; `context`: `[B, Lk, Dkv]`.
    fn forward(&self, x: &Array, context: &Array) -> Result<Array> {
        let (b, lq) = (x.shape()[0], x.shape()[1]);
        let lk = context.shape()[1];
        let q = matmul(x, self.q.t())?; // [B, Lq, inner]
        let inner = q.shape()[2];
        let head_dim = inner / self.heads;
        let scale = (head_dim as f32).powf(-0.5);
        let k = matmul(context, self.k.t())?;
        let v = matmul(context, self.v.t())?;
        let to_heads = |a: Array, n: i32| -> Result<Array> {
            Ok(a.reshape(&[b, n, self.heads, head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = to_heads(q, lq)?;
        let k = to_heads(k, lk)?;
        let v = to_heads(v, lk)?;
        let o = scaled_dot_product_attention(&q, &k, &v, scale, None, None)?;
        let o = o.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, lq, inner])?;
        linear(&o, &self.out.0, &self.out.1)
    }
}

/// Spatial `BasicTransformerBlock`: pre-norm self-attn → pre-norm cross-attn → pre-norm GEGLU ff.
struct BasicBlock {
    norm1: (Array, Array),
    attn1: Attention,
    norm2: (Array, Array),
    attn2: Attention,
    norm3: (Array, Array),
    ff_proj: (Array, Array),
    ff_out: (Array, Array),
}

impl BasicBlock {
    fn from_weights(w: &Weights, prefix: &str, heads: i32) -> Result<Self> {
        Ok(Self {
            norm1: lin(w, &format!("{prefix}.norm1"))?,
            attn1: Attention::from_weights(w, &format!("{prefix}.attn1"), heads)?,
            norm2: lin(w, &format!("{prefix}.norm2"))?,
            attn2: Attention::from_weights(w, &format!("{prefix}.attn2"), heads)?,
            norm3: lin(w, &format!("{prefix}.norm3"))?,
            ff_proj: lin(w, &format!("{prefix}.ff.net.0.proj"))?,
            ff_out: lin(w, &format!("{prefix}.ff.net.2"))?,
        })
    }

    fn forward(&self, x: &Array, context: &Array) -> Result<Array> {
        let y = layer_norm(x, Some(&self.norm1.0), Some(&self.norm1.1), LN_EPS)?;
        let x = add(x, &self.attn1.forward(&y, &y)?)?;
        let y = layer_norm(&x, Some(&self.norm2.0), Some(&self.norm2.1), LN_EPS)?;
        let x = add(&x, &self.attn2.forward(&y, context)?)?;
        let y = layer_norm(&x, Some(&self.norm3.0), Some(&self.norm3.1), LN_EPS)?;
        Ok(add(&x, &geglu(&y, &self.ff_proj, &self.ff_out)?)?)
    }
}

/// `TemporalBasicTransformerBlock`: reshape to attend over the frame axis, then ff_in (+residual,
/// `is_res`) → self-attn → cross-attn → ff (+residual), then reshape back.
struct TemporalBlock {
    norm_in: (Array, Array),
    ffin_proj: (Array, Array),
    ffin_out: (Array, Array),
    norm1: (Array, Array),
    attn1: Attention,
    norm2: (Array, Array),
    attn2: Attention,
    norm3: (Array, Array),
    ff_proj: (Array, Array),
    ff_out: (Array, Array),
}

impl TemporalBlock {
    fn from_weights(w: &Weights, prefix: &str, heads: i32) -> Result<Self> {
        Ok(Self {
            norm_in: lin(w, &format!("{prefix}.norm_in"))?,
            ffin_proj: lin(w, &format!("{prefix}.ff_in.net.0.proj"))?,
            ffin_out: lin(w, &format!("{prefix}.ff_in.net.2"))?,
            norm1: lin(w, &format!("{prefix}.norm1"))?,
            attn1: Attention::from_weights(w, &format!("{prefix}.attn1"), heads)?,
            norm2: lin(w, &format!("{prefix}.norm2"))?,
            attn2: Attention::from_weights(w, &format!("{prefix}.attn2"), heads)?,
            norm3: lin(w, &format!("{prefix}.norm3"))?,
            ff_proj: lin(w, &format!("{prefix}.ff.net.0.proj"))?,
            ff_out: lin(w, &format!("{prefix}.ff.net.2"))?,
        })
    }

    /// `x`: `[B·F, seq, C]`; `context`: `[B·seq, ctx, Dkv]` (frame-0 memory, broadcast over seq).
    fn forward(&self, x: &Array, num_frames: i32, context: &Array) -> Result<Array> {
        let sh = x.shape();
        let (bf, seq, c) = (sh[0], sh[1], sh[2]);
        let b = bf / num_frames;
        // [B·F, seq, C] → [B, F, seq, C] → [B, seq, F, C] → [B·seq, F, C] (attend over frames).
        let h = x
            .reshape(&[b, num_frames, seq, c])?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b * seq, num_frames, c])?;
        let residual = h.clone();
        let n = layer_norm(&h, Some(&self.norm_in.0), Some(&self.norm_in.1), LN_EPS)?;
        let h = add(&geglu(&n, &self.ffin_proj, &self.ffin_out)?, &residual)?; // is_res
        let n = layer_norm(&h, Some(&self.norm1.0), Some(&self.norm1.1), LN_EPS)?;
        let h = add(&self.attn1.forward(&n, &n)?, &h)?;
        let n = layer_norm(&h, Some(&self.norm2.0), Some(&self.norm2.1), LN_EPS)?;
        let h = add(&self.attn2.forward(&n, context)?, &h)?;
        let n = layer_norm(&h, Some(&self.norm3.0), Some(&self.norm3.1), LN_EPS)?;
        let h = add(&geglu(&n, &self.ff_proj, &self.ff_out)?, &h)?;
        // back to [B·F, seq, C].
        Ok(h.reshape(&[b, seq, num_frames, c])?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[bf, seq, c])?)
    }
}

/// The full spatio-temporal transformer at one resolution.
pub struct TransformerSpatioTemporal {
    norm: (Array, Array),
    proj_in: (Array, Array),
    blocks: Vec<BasicBlock>,
    temporal_blocks: Vec<TemporalBlock>,
    time_pos_embed: TimestepEmbedding,
    mix_factor: Array,
    proj_out: (Array, Array),
    in_channels: i32,
}

impl TransformerSpatioTemporal {
    /// `prefix` addresses the `attentions.{i}` module; `heads` is the block's head count.
    pub fn from_weights(w: &Weights, prefix: &str, heads: i32, num_layers: usize) -> Result<Self> {
        let norm = lin(w, &format!("{prefix}.norm"))?;
        let in_channels = norm.0.shape()[0];
        let blocks = (0..num_layers)
            .map(|i| {
                BasicBlock::from_weights(w, &format!("{prefix}.transformer_blocks.{i}"), heads)
            })
            .collect::<Result<Vec<_>>>()?;
        let temporal_blocks = (0..num_layers)
            .map(|i| {
                TemporalBlock::from_weights(
                    w,
                    &format!("{prefix}.temporal_transformer_blocks.{i}"),
                    heads,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            norm,
            proj_in: lin(w, &format!("{prefix}.proj_in"))?,
            blocks,
            temporal_blocks,
            time_pos_embed: TimestepEmbedding::from_weights(
                w,
                &format!("{prefix}.time_pos_embed"),
            )?,
            mix_factor: w
                .require(&format!("{prefix}.time_mixer.mix_factor"))?
                .clone(),
            proj_out: lin(w, &format!("{prefix}.proj_out"))?,
            in_channels,
        })
    }

    /// `x`: NHWC `[B·F, H, W, C]`; `context`: CLIP image memory `[B·F, ctx, Dkv]`.
    pub fn forward(&self, x: &Array, context: &Array, num_frames: i32) -> Result<Array> {
        let sh = x.shape();
        let (bf, h_, w_, c) = (sh[0], sh[1], sh[2], sh[3]);
        let b = bf / num_frames;
        let seq = h_ * w_;

        // Temporal cross-attn memory: take frame-0's context, broadcast over the H·W tokens.
        let cs = context.shape();
        let (ctx_seq, cd) = (cs[1], cs[2]);
        let tctx = context.reshape(&[b, num_frames, ctx_seq, cd])?;
        let tctx = tctx.take_axis(Array::from_int(0), 1)?; // [B, ctx, cd] (frame 0)
        let tctx = broadcast_to(&tctx.reshape(&[b, 1, ctx_seq, cd])?, &[b, seq, ctx_seq, cd])?;
        let tctx = tctx.reshape(&[b * seq, ctx_seq, cd])?;

        let residual = x.clone();
        let n = group_norm(x, &self.norm.0, &self.norm.1, GN_GROUPS, GN_EPS)?;
        let mut tokens = linear(&n.reshape(&[bf, seq, c])?, &self.proj_in.0, &self.proj_in.1)?;

        // Per-frame position embedding: arange(F) tiled over the batch.
        let nframes: Vec<f32> = (0..b)
            .flat_map(|_| (0..num_frames).map(|f| f as f32))
            .collect();
        let nframes = Array::from_slice(&nframes, &[bf]);
        let emb = self
            .time_pos_embed
            .forward(&sinusoidal_timestep(&nframes, self.in_channels)?)?;
        let emb = emb.reshape(&[bf, 1, c])?; // [B·F, 1, C]

        let alpha = sigmoid(&self.mix_factor)?;
        let one_minus = subtract(scalar(1.0), &alpha)?;
        for (block, temporal) in self.blocks.iter().zip(&self.temporal_blocks) {
            tokens = block.forward(&tokens, context)?;
            let mix = add(&tokens, &emb)?;
            let mix = temporal.forward(&mix, num_frames, &tctx)?;
            tokens = add(&multiply(&tokens, &alpha)?, &multiply(&mix, &one_minus)?)?;
        }

        let tokens = linear(&tokens, &self.proj_out.0, &self.proj_out.1)?;
        Ok(add(&tokens.reshape(&[bf, h_, w_, c])?, &residual)?)
    }
}
