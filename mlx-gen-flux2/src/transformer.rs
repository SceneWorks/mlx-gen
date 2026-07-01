//! FLUX.2 MMDiT transformer — 8 double (joint img+txt) blocks + 24 single (fused parallel
//! attention+SwiGLU) blocks, shared per-stream modulation, 4-axis interleaved RoPE, and an
//! `AdaLayerNormContinuous` output. Port of `models/flux2/model/flux2_transformer/`.
//!
//! Runs f32 activations (matmul(f32 act, bf16 weight)→f32): the `x_embedder` (K=128, M=seq≥2) is
//! the dense 16-bit Metal GEMM bug shape, so the whole stack must run f32 — which is also the
//! quality target. Linears are bias-less core [`AdaptableLinear`]s so `spec.quantize` can pack
//! every projection to Q4/Q8 in place (sc-2643; the fork quantizes every transformer `nn.Linear`).
//! RMSNorm/LayerNorm weights stay full precision. With f32 activations the quantized forward feeds
//! `quantized_matmul` f32 inputs (no bf16 upcast needed). LoRA over these bases = sc-2646.

use mlx_rs::error::Exception;
use mlx_rs::fast::{layer_norm, rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, concatenate_axis, multiply, sigmoid, split};
use mlx_rs::transforms::compile::compile;
use mlx_rs::{Array, Dtype};
use std::f32::consts::LN_10;

use mlx_gen::adapters::loader::{BflTarget, LoraRowSlice};
use mlx_gen::adapters::{prefixed_paths, AdaptableHost, AdaptableLinear};
use mlx_gen::array::scalar;
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::chunk::{map_seq_chunks, MemoryConfig};
use crate::config::{Flux2Config, Flux2Quant};
use crate::kv_cache::{Flux2KvCache, Stream};
use crate::pos_embed::Flux2PosEmbed;
// The quant-aware bias-less Linear loader is shared with the text encoder (its canonical home, 6937).
use crate::text_encoder::lin;

/// Per-call KV-cache binding handed to an attention layer: `(cache, layer_idx_within_stream)`.
/// `None` on the dense path (txt2img, plain edit) and inside parity tests.
type CacheSlot<'a> = Option<(&'a Flux2KvCache, usize)>;

const LN_EPS: f32 = 1e-6;
const RMS_EPS: f32 = 1e-5;

// sc-2963 compiled-glue toggle + the `modulate`/`gated`/`rope_rotate` helpers are hoisted into core
// (F-101) so FLUX.1/FLUX.2 share one implementation. Re-export the toggle as this crate's public API;
// FLUX.2's modulate keeps a strong f32 `1` via `one_matches_scale = false`. SwiGLU stays crate-specific.
use mlx_gen::nn::compile_glue;
pub use mlx_gen::nn::{set_compile_glue, CompileGlueGuard};

fn require_f32_input(x: &Array) -> Result<Array> {
    Ok(x.as_dtype(Dtype::Float32)?)
}

/// `[B,S,H·D]` → `[B,H,S,D]`, with per-head q/k RMSNorm (f32). Port of `AttentionUtils.process_qkv`.
#[allow(clippy::too_many_arguments)]
fn process_qkv(
    x: &Array,
    q_w: &AdaptableLinear,
    k_w: &AdaptableLinear,
    v_w: &AdaptableLinear,
    norm_q: &Array,
    norm_k: &Array,
    heads: i32,
    head_dim: i32,
) -> Result<(Array, Array, Array)> {
    let sh = x.shape();
    let (b, s) = (sh[0], sh[1]);
    let to_bhsd = |a: Array| -> Result<Array> {
        Ok(a.reshape(&[b, s, heads, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?)
    };
    let q = to_bhsd(q_w.forward(x)?)?;
    let k = to_bhsd(k_w.forward(x)?)?;
    let v = to_bhsd(v_w.forward(x)?)?;
    let q = rms_norm(&q, norm_q, RMS_EPS)?;
    let k = rms_norm(&k, norm_k, RMS_EPS)?;
    Ok((q, k, v))
}

/// The complex RoPE rotation `(real + imag·i)·(cos + sin·i)` → `(out_real, out_imag)`. Forwards to
/// the shared [`mlx_gen::nn::rope_rotate`].
fn rope_rotate(real: &Array, imag: &Array, cos: &Array, sin: &Array) -> Result<(Array, Array)> {
    mlx_gen::nn::rope_rotate(real, imag, cos, sin)
}

/// Interleaved RoPE (`AttentionUtils.apply_rope_bshd`): pairs `(x[2i], x[2i+1])` rotated by
/// `cos/sin[i]`. `cos`/`sin`: `[S, head_dim/2]`; `q`/`k`: `[B,H,S,head_dim]`.
fn apply_rope(q: &Array, k: &Array, cos: &Array, sin: &Array) -> Result<(Array, Array)> {
    let s = cos.shape()[0];
    let half = cos.shape()[1];
    let cos = cos.reshape(&[1, 1, s, half])?;
    let sin = sin.reshape(&[1, 1, s, half])?;
    let one = |x: &Array| -> Result<Array> {
        let sh = x.shape();
        let (b, h, seq, hd) = (sh[0], sh[1], sh[2], sh[3]);
        let x5 = x.reshape(&[b, h, seq, hd / 2, 2])?;
        let p = split(&x5, 2, 4)?;
        let real = p[0].reshape(&[b, h, seq, hd / 2])?;
        let imag = p[1].reshape(&[b, h, seq, hd / 2])?;
        let (out0, out1) = rope_rotate(&real, &imag, &cos, &sin)?;
        Ok(
            concatenate_axis(&[&out0.expand_dims(4)?, &out1.expand_dims(4)?], 4)?
                .reshape(&[b, h, seq, hd])?,
        )
    };
    Ok((one(q)?, one(k)?))
}

/// SDPA over `[B,H,S,D]` → `[B,S,H·D]`.
fn attention(q: &Array, k: &Array, v: &Array, head_dim: i32) -> Result<Array> {
    let b = q.shape()[0];
    let scale = (head_dim as f32).powf(-0.5);
    let o = scaled_dot_product_attention(q, k, v, scale, None, None)?;
    Ok(o.transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[b, -1, q.shape()[1] * head_dim])?)
}

/// SwiGLU: split last axis in half, `silu(x1) · x2`. The `split` runs eagerly (a shapeless
/// `mx.compile` can't infer a split's output shapes); the fusable `silu(x1)·x2` arithmetic is
/// compiled into one kernel when the sc-2963 glue toggle is on. Bit-exact to the eager
/// `multiply(silu(x1), x2)` — the inline `a·sigmoid(a)` mirrors [`mlx_gen::nn::silu`] op-for-op.
fn swiglu(x: &Array) -> Result<Array> {
    let p = split(x, 2, -1)?;
    let f = |(a, b): (&Array, &Array)| -> std::result::Result<Array, Exception> {
        multiply(&multiply(a, &sigmoid(a)?)?, b) // silu(a)·b
    };
    if compile_glue() {
        Ok(compile(f, true)((&p[0], &p[1]))?)
    } else {
        Ok(f((&p[0], &p[1]))?)
    }
}

/// `(1 + scale) · norm(x) + shift` — FLUX.2 keeps a strong f32 `1`. Forwards to the shared
/// [`mlx_gen::nn::modulate`] with `one_matches_scale = false`.
fn modulate(norm: &Array, scale: &Array, shift: &Array) -> Result<Array> {
    mlx_gen::nn::modulate(norm, scale, shift, false)
}

/// Gated residual `x + gate·y`. Forwards to the shared [`mlx_gen::nn::gated`].
fn gated(x: &Array, gate: &Array, y: &Array) -> Result<Array> {
    mlx_gen::nn::gated(x, gate, y)
}

/// Sinusoidal timestep embedding (diffusers `_timestep_embedding`, flip_sin_to_cos): `[B]` → `[B,
/// dim]` = `concat([cos(args), sin(args)])`.
fn timestep_embedding(t: &Array, dim: usize) -> Result<Array> {
    let half = dim / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-LN_10 * 4.0 * i as f32 / half as f32).exp())
        .collect();
    // ln(10000) = 4·ln(10).
    let freqs = Array::from_slice(&freqs, &[1, half as i32]);
    let t = t.reshape(&[t.shape()[0], 1])?.as_dtype(Dtype::Float32)?;
    let args = multiply(&t, &freqs)?;
    Ok(concatenate_axis(&[&args.cos()?, &args.sin()?], 1)?)
}

struct FeedForward {
    linear_in: AdaptableLinear,
    linear_out: AdaptableLinear,
}

impl FeedForward {
    fn from_weights(w: &Weights, prefix: &str, quant: Option<Flux2Quant>) -> Result<Self> {
        Ok(Self {
            linear_in: lin(w, &format!("{prefix}.linear_in.weight"), quant)?,
            linear_out: lin(w, &format!("{prefix}.linear_out.weight"), quant)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let x = self.linear_in.forward(x)?;
        let x = swiglu(&x)?;
        self.linear_out.forward(&x)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.linear_in.quantize(bits, None)?;
        self.linear_out.quantize(bits, None)?;
        Ok(())
    }
}

struct DoubleBlock {
    attn: DoubleAttention,
    ff: FeedForward,
    ff_context: FeedForward,
}

struct DoubleAttention {
    to_q: AdaptableLinear,
    to_k: AdaptableLinear,
    to_v: AdaptableLinear,
    to_out: AdaptableLinear,
    norm_q: Array,
    norm_k: Array,
    add_q: AdaptableLinear,
    add_k: AdaptableLinear,
    add_v: AdaptableLinear,
    to_add_out: AdaptableLinear,
    norm_added_q: Array,
    norm_added_k: Array,
    heads: i32,
    head_dim: i32,
}

impl DoubleAttention {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        heads: i32,
        head_dim: i32,
        quant: Option<Flux2Quant>,
    ) -> Result<Self> {
        let g = |n: &str| w.require(&format!("{prefix}.{n}.weight")).cloned();
        let l = |n: &str| lin(w, &format!("{prefix}.{n}.weight"), quant);
        Ok(Self {
            to_q: l("to_q")?,
            to_k: l("to_k")?,
            to_v: l("to_v")?,
            to_out: l("to_out")?,
            norm_q: g("norm_q")?,
            norm_k: g("norm_k")?,
            add_q: l("add_q_proj")?,
            add_k: l("add_k_proj")?,
            add_v: l("add_v_proj")?,
            to_add_out: l("to_add_out")?,
            norm_added_q: g("norm_added_q")?,
            norm_added_k: g("norm_added_k")?,
            heads,
            head_dim,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        for p in [
            &mut self.to_q,
            &mut self.to_k,
            &mut self.to_v,
            &mut self.to_out,
            &mut self.add_q,
            &mut self.add_k,
            &mut self.add_v,
            &mut self.to_add_out,
        ] {
            p.quantize(bits, None)?;
        }
        Ok(())
    }

    /// Joint attention. Returns `(img_attn_out, txt_attn_out)`. `cache` (the 9b-kv edit path)
    /// stores/splices the trailing reference K/V for this double-stream layer post-RoPE.
    fn forward(
        &self,
        img: &Array,
        txt: &Array,
        cos: &Array,
        sin: &Array,
        cache: CacheSlot<'_>,
    ) -> Result<(Array, Array)> {
        let (iq, ik, iv) = process_qkv(
            img,
            &self.to_q,
            &self.to_k,
            &self.to_v,
            &self.norm_q,
            &self.norm_k,
            self.heads,
            self.head_dim,
        )?;
        let (tq, tk, tv) = process_qkv(
            txt,
            &self.add_q,
            &self.add_k,
            &self.add_v,
            &self.norm_added_q,
            &self.norm_added_k,
            self.heads,
            self.head_dim,
        )?;
        // [txt, img] order along the sequence (axis 2 in BHSD).
        let q = concatenate_axis(&[&tq, &iq], 2)?;
        let k = concatenate_axis(&[&tk, &ik], 2)?;
        let v = concatenate_axis(&[&tv, &iv], 2)?;
        let (q, k) = apply_rope(&q, &k, cos, sin)?;
        // KV-cache hook (post-RoPE, pre-SDPA): extract stores the trailing ref K/V; cached splices
        // it back so the `[txt, target]` queries attend over `[txt, target, ref]`.
        let (k, v) = match cache {
            Some((c, idx)) => c.apply(Stream::Double, idx, k, v)?,
            None => (k, v),
        };
        let o = attention(&q, &k, &v, self.head_dim)?;
        let txt_seq = txt.shape()[1];
        let txt_idx = Array::from_slice(&(0..txt_seq).collect::<Vec<i32>>(), &[txt_seq]);
        let img_idx = Array::from_slice(
            &(txt_seq..o.shape()[1]).collect::<Vec<i32>>(),
            &[o.shape()[1] - txt_seq],
        );
        let txt_out = self.to_add_out.forward(&o.take_axis(&txt_idx, 1)?)?;
        let img_out = self.to_out.forward(&o.take_axis(&img_idx, 1)?)?;
        Ok((img_out, txt_out))
    }
}

impl DoubleBlock {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        heads: i32,
        head_dim: i32,
        quant: Option<Flux2Quant>,
    ) -> Result<Self> {
        Ok(Self {
            attn: DoubleAttention::from_weights(
                w,
                &format!("{prefix}.attn"),
                heads,
                head_dim,
                quant,
            )?,
            ff: FeedForward::from_weights(w, &format!("{prefix}.ff"), quant)?,
            ff_context: FeedForward::from_weights(w, &format!("{prefix}.ff_context"), quant)?,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attn.quantize(bits)?;
        self.ff.quantize(bits)?;
        self.ff_context.quantize(bits)?;
        Ok(())
    }

    /// `img_mod` / `txt_mod`: `[(shift_msa,scale_msa,gate_msa),(shift_mlp,scale_mlp,gate_mlp)]`.
    /// `ffn_chunk` (sc-6266) bounds the image FFN's SwiGLU intermediate over sequence row-blocks;
    /// `None` ⇒ the whole sequence at once, byte-identical to the shipped path.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        mut img: Array,
        mut txt: Array,
        img_mod: &[(Array, Array, Array); 2],
        txt_mod: &[(Array, Array, Array); 2],
        cos: &Array,
        sin: &Array,
        cache: CacheSlot<'_>,
        ffn_chunk: Option<usize>,
    ) -> Result<(Array, Array)> {
        let (shift_msa, scale_msa, gate_msa) = &img_mod[0];
        let (shift_mlp, scale_mlp, gate_mlp) = &img_mod[1];
        let (c_shift_msa, c_scale_msa, c_gate_msa) = &txt_mod[0];
        let (c_shift_mlp, c_scale_mlp, c_gate_mlp) = &txt_mod[1];

        let norm_img = modulate(&layer_norm(&img, None, None, LN_EPS)?, scale_msa, shift_msa)?;
        let norm_txt = modulate(
            &layer_norm(&txt, None, None, LN_EPS)?,
            c_scale_msa,
            c_shift_msa,
        )?;

        let (img_attn, txt_attn) = self.attn.forward(&norm_img, &norm_txt, cos, sin, cache)?;
        img = gated(&img, gate_msa, &img_attn)?;
        txt = gated(&txt, c_gate_msa, &txt_attn)?;

        let norm_img2 = modulate(&layer_norm(&img, None, None, LN_EPS)?, scale_mlp, shift_mlp)?;
        // sc-6266: chunk the (largest) image FFN intermediate over sequence row-blocks on the gated
        // long-sequence multi-reference edit path. `ffn_chunk == None` ⇒ a single `ff.forward` call,
        // byte-identical to the shipped forward.
        let img_ff = map_seq_chunks(&norm_img2, ffn_chunk, |c| self.ff.forward(c))?;
        img = gated(&img, gate_mlp, &img_ff)?;

        let norm_txt2 = modulate(
            &layer_norm(&txt, None, None, LN_EPS)?,
            c_scale_mlp,
            c_shift_mlp,
        )?;
        let txt_ff = self.ff_context.forward(&norm_txt2)?;
        txt = gated(&txt, c_gate_mlp, &txt_ff)?;

        Ok((txt, img))
    }
}

struct SingleBlock {
    to_qkv_mlp: AdaptableLinear,
    to_out: AdaptableLinear,
    norm_q: Array,
    norm_k: Array,
    heads: i32,
    head_dim: i32,
    inner: i32,
}

impl SingleBlock {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        heads: i32,
        head_dim: i32,
        quant: Option<Flux2Quant>,
    ) -> Result<Self> {
        Ok(Self {
            to_qkv_mlp: lin(w, &format!("{prefix}.attn.to_qkv_mlp_proj.weight"), quant)?,
            to_out: lin(w, &format!("{prefix}.attn.to_out.weight"), quant)?,
            norm_q: w.require(&format!("{prefix}.attn.norm_q.weight"))?.clone(),
            norm_k: w.require(&format!("{prefix}.attn.norm_k.weight"))?.clone(),
            heads,
            head_dim,
            inner: heads * head_dim,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.to_qkv_mlp.quantize(bits, None)?;
        self.to_out.quantize(bits, None)?;
        Ok(())
    }

    /// `mod`: `(shift, scale, gate)`. `cache` (9b-kv edit) stores/splices the trailing reference
    /// K/V for this single-stream layer post-RoPE.
    fn forward(
        &self,
        hidden: &Array,
        m: &(Array, Array, Array),
        cos: &Array,
        sin: &Array,
        cache: CacheSlot<'_>,
    ) -> Result<Array> {
        let (shift, scale, gate) = m;
        let norm = modulate(&layer_norm(hidden, None, None, LN_EPS)?, scale, shift)?;
        let proj = self.to_qkv_mlp.forward(&norm)?;

        let sh = proj.shape();
        let (b, s) = (sh[0], sh[1]);
        let take = |start: i32, end: i32| -> Result<Array> {
            let idx = Array::from_slice(&(start..end).collect::<Vec<i32>>(), &[end - start]);
            Ok(proj.take_axis(&idx, 2)?)
        };
        let q = take(0, self.inner)?;
        let k = take(self.inner, 2 * self.inner)?;
        let v = take(2 * self.inner, 3 * self.inner)?;
        let mlp = take(3 * self.inner, sh[2])?;

        let to_bhsd = |a: Array| -> Result<Array> {
            Ok(a.reshape(&[b, s, self.heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = rms_norm(&to_bhsd(q)?, &self.norm_q, RMS_EPS)?;
        let k = rms_norm(&to_bhsd(k)?, &self.norm_k, RMS_EPS)?;
        let v = to_bhsd(v)?;
        let (q, k) = apply_rope(&q, &k, cos, sin)?;
        let (k, v) = match cache {
            Some((c, idx)) => c.apply(Stream::Single, idx, k, v)?,
            None => (k, v),
        };
        let attn = attention(&q, &k, &v, self.head_dim)?;

        let mlp = swiglu(&mlp)?;
        let cat = concatenate_axis(&[&attn, &mlp], -1)?;
        let attn_output = self.to_out.forward(&cat)?;
        gated(hidden, gate, &attn_output)
    }
}

/// Per-stream modulation producer: `silu(temb) → linear → split into `sets` × (shift,scale,gate)`.
struct Modulation {
    linear: AdaptableLinear,
    sets: usize,
}

impl Modulation {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        sets: usize,
        quant: Option<Flux2Quant>,
    ) -> Result<Self> {
        Ok(Self {
            linear: lin(w, &format!("{prefix}.linear.weight"), quant)?,
            sets,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.linear.quantize(bits, None)
    }

    /// `temb`: `[B, dim]` → `Vec<(shift,scale,gate)>` of length `sets`, each `[B,1,dim]`.
    fn forward(&self, temb: &Array) -> Result<Vec<(Array, Array, Array)>> {
        let mod_ = self.linear.forward(&silu(temb)?)?.expand_dims(1)?;
        let chunks = split(&mod_, (3 * self.sets) as i32, -1)?;
        if chunks.len() != 3 * self.sets {
            return Err(Error::Msg(format!(
                "flux2 modulation: expected {} chunks (3×{} sets), got {}",
                3 * self.sets,
                self.sets,
                chunks.len()
            )));
        }
        Ok((0..self.sets)
            .map(|i| {
                (
                    chunks[3 * i].clone(),
                    chunks[3 * i + 1].clone(),
                    chunks[3 * i + 2].clone(),
                )
            })
            .collect())
    }
}

/// The FLUX.2 MMDiT transformer.
pub struct Flux2Transformer {
    pos_embed: Flux2PosEmbed,
    time_linear1: AdaptableLinear,
    time_linear2: AdaptableLinear,
    /// The embedded-guidance branch (`time_guidance_embed.guidance_embedder.linear_{1,2}`) — `Some`
    /// for the guidance-distilled **dev** (sc-2365), `None` for the CFG-free klein. When present,
    /// `temb` adds a guidance embedding to the timestep embedding (the FLUX.1-dev pattern).
    guidance_linear1: Option<AdaptableLinear>,
    guidance_linear2: Option<AdaptableLinear>,
    mod_img: Modulation,
    mod_txt: Modulation,
    mod_single: Modulation,
    x_embedder: AdaptableLinear,
    context_embedder: AdaptableLinear,
    double_blocks: Vec<DoubleBlock>,
    single_blocks: Vec<SingleBlock>,
    norm_out_linear: AdaptableLinear,
    proj_out: AdaptableLinear,
    time_channels: usize,
}

/// The conditioning inputs every [`Flux2Transformer`] forward shares, grouped so the four entry
/// points (`forward` / `forward_with_cache` / `forward_with_mem` / `forward_with_control`) each carry
/// one `&Flux2ForwardInputs` plus only their distinguishing extra (cache / mem / control) instead of
/// 7-9 positional args — and so the two same-typed `&Array` id tensors are named at every call site
/// (F-072). `hidden_states`: `[B, seq_img, in_channels]`; `encoder_hidden_states`: `[B, seq_txt,
/// joint_attention_dim]`; `img_ids`/`txt_ids`: `[seq, 4]` (or `[1, seq, 4]`); `timestep` is the scaled
/// sigma (×1000); `guidance` is the embedded-guidance scale (`Some` for dev, `None` for klein).
#[derive(Clone, Copy)]
pub struct Flux2ForwardInputs<'a> {
    pub hidden_states: &'a Array,
    pub encoder_hidden_states: &'a Array,
    pub img_ids: &'a Array,
    pub txt_ids: &'a Array,
    pub timestep: f32,
    pub guidance: Option<f32>,
}

impl Flux2Transformer {
    /// Load the transformer from a **dense** weight map (the parity-test + dense-snapshot path).
    pub fn from_weights(w: &Weights, cfg: &Flux2Config) -> Result<Self> {
        Self::from_weights_quant(w, cfg, None)
    }

    /// Load the transformer, building each Linear from packed Q4/Q8 parts when `quant` is `Some`
    /// AND the on-disk weights carry the packed `.scales`/`.biases` (a pre-quantized snapshot,
    /// sc-5917). `quant == None` ⇒ the dense path, identical to [`from_weights`](Self::from_weights).
    /// The packed path never materializes a dense bf16 weight, so the dev 32B DiT loads at its Q4
    /// resident footprint (~17 GB) rather than the 60 GB bf16 load transient.
    pub fn from_weights_quant(
        w: &Weights,
        cfg: &Flux2Config,
        quant: Option<Flux2Quant>,
    ) -> Result<Self> {
        let heads = cfg.num_heads as i32;
        let head_dim = cfg.head_dim as i32;
        let double_blocks = (0..cfg.num_double_layers)
            .map(|i| {
                DoubleBlock::from_weights(
                    w,
                    &format!("transformer_blocks.{i}"),
                    heads,
                    head_dim,
                    quant,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let single_blocks = (0..cfg.num_single_layers)
            .map(|i| {
                SingleBlock::from_weights(
                    w,
                    &format!("single_transformer_blocks.{i}"),
                    heads,
                    head_dim,
                    quant,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        // The guidance embedder is present only on the guidance-distilled dev checkpoint; klein has
        // no `guidance_embedder.*` keys, so this is `None` there (the `.weight` key gates both the
        // dense and the pre-quantized snapshot — the packed codes also live at `.weight`).
        let guidance_key = |n: &str| format!("time_guidance_embed.guidance_embedder.{n}.weight");
        let (guidance_linear1, guidance_linear2) = if w.get(&guidance_key("linear_1")).is_some() {
            (
                Some(lin(w, &guidance_key("linear_1"), quant)?),
                Some(lin(w, &guidance_key("linear_2"), quant)?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            pos_embed: Flux2PosEmbed::new(cfg.rope_theta, cfg.axes_dim),
            time_linear1: lin(w, "time_guidance_embed.linear_1.weight", quant)?,
            time_linear2: lin(w, "time_guidance_embed.linear_2.weight", quant)?,
            guidance_linear1,
            guidance_linear2,
            mod_img: Modulation::from_weights(w, "double_stream_modulation_img", 2, quant)?,
            mod_txt: Modulation::from_weights(w, "double_stream_modulation_txt", 2, quant)?,
            mod_single: Modulation::from_weights(w, "single_stream_modulation", 1, quant)?,
            x_embedder: lin(w, "x_embedder.weight", quant)?,
            context_embedder: lin(w, "context_embedder.weight", quant)?,
            double_blocks,
            single_blocks,
            norm_out_linear: lin(w, "norm_out.linear.weight", quant)?,
            proj_out: lin(w, "proj_out.weight", quant)?,
            time_channels: cfg.timestep_channels,
        })
    }

    /// Quantize every transformer `nn.Linear` to Q4/Q8 (group_size 64) in place — the mlx-rs
    /// equivalent of the fork's `nn.quantize(transformer, predicate=hasattr to_quantized, bits)`.
    /// RMSNorm/LayerNorm weights are not Linears, so they stay full precision (as in the fork).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.time_linear1.quantize(bits, None)?;
        self.time_linear2.quantize(bits, None)?;
        for g in [&mut self.guidance_linear1, &mut self.guidance_linear2]
            .into_iter()
            .flatten()
        {
            g.quantize(bits, None)?;
        }
        self.mod_img.quantize(bits)?;
        self.mod_txt.quantize(bits)?;
        self.mod_single.quantize(bits)?;
        self.x_embedder.quantize(bits, None)?;
        self.context_embedder.quantize(bits, None)?;
        for b in &mut self.double_blocks {
            b.quantize(bits)?;
        }
        for b in &mut self.single_blocks {
            b.quantize(bits)?;
        }
        self.norm_out_linear.quantize(bits, None)?;
        self.proj_out.quantize(bits, None)?;
        Ok(())
    }

    /// Test-only (sc-2643 byte-parity gate): the quantized `(wq, scales, biases, group_size, bits)`
    /// of `transformer_blocks.0.attn.to_q` — a representative bias-less, bf16-native Linear. `None`
    /// if the transformer is still dense.
    #[doc(hidden)]
    pub fn probe_quant_to_q(&self) -> Option<(&Array, &Array, &Array, i32, i32)> {
        let (wq, sc, bi, _bias, gs, b) = self.double_blocks[0].attn.to_q.quantized_params()?;
        Some((wq, sc, bi, gs, b))
    }

    /// `timestep` is fed as sigma·1000 (the caller scales it). `guidance` is the raw guidance scale
    /// (e.g. 4.0) for the guidance-distilled dev path, or `None` for klein. Mirrors diffusers
    /// `Flux2TimestepGuidanceEmbeddings`: `timestep_emb + guidance_emb` (no pooled-CLIP term), each
    /// `time_proj → linear_1 → silu → linear_2`, with guidance scaled ×1000 here (the
    /// `transformer_flux2.py` `guidance = guidance * 1000` step) before the shared sinusoidal proj.
    fn temb(&self, timestep: f32, guidance: Option<f32>) -> Result<Array> {
        let embed = |scalar: f32, l1: &AdaptableLinear, l2: &AdaptableLinear| -> Result<Array> {
            let t = Array::from_slice(&[scalar], &[1]);
            let emb = timestep_embedding(&t, self.time_channels)?;
            l2.forward(&silu(&l1.forward(&emb)?)?)
        };
        let mut temb = embed(timestep, &self.time_linear1, &self.time_linear2)?;
        if let (Some(g), Some(g1), Some(g2)) =
            (guidance, &self.guidance_linear1, &self.guidance_linear2)
        {
            // diffusers scales guidance ×1000 at the transformer boundary (the timestep is already
            // ×1000 by the caller). A `Some(guidance)` on a klein transformer (no guidance embedder)
            // is silently ignored — the embedded-guidance path is dev-only.
            temb = add(&temb, &embed(g * 1000.0, g1, g2)?)?;
        }
        Ok(temb)
    }

    fn norm_out(&self, x: &Array, temb: &Array) -> Result<Array> {
        let p = self.norm_out_linear.forward(&silu(temb)?)?; // [B, 2·dim]
        let parts = split(&p, 2, 1)?;
        let scale = parts[0].expand_dims(1)?; // [B,1,dim]
        let shift = parts[1].expand_dims(1)?;
        let normed = layer_norm(x, None, None, LN_EPS)?;
        Ok(add(
            &multiply(&normed, &add(&scale, scalar(1.0))?)?,
            &shift,
        )?)
    }

    /// `hidden_states`: `[B, seq_img, in_channels]`; `encoder_hidden_states`: `[B, seq_txt,
    /// joint_attention_dim]`; `img_ids`/`txt_ids`: `[seq, 4]` (or `[1, seq, 4]`). `timestep` is the
    /// scaled sigma (×1000). Returns the velocity `[B, seq_img, out_channels]`. Dense path: no cache,
    /// no embedded guidance (klein). The richer entry points take a [`Flux2ForwardInputs`]:
    /// [`Self::forward_with_cache`] / [`Self::forward_with_mem`] / [`Self::forward_with_control`].
    pub fn forward(
        &self,
        hidden_states: &Array,
        encoder_hidden_states: &Array,
        img_ids: &Array,
        txt_ids: &Array,
        timestep: f32,
    ) -> Result<Array> {
        self.forward_inner(
            hidden_states,
            encoder_hidden_states,
            img_ids,
            txt_ids,
            timestep,
            None,
            None,
            None,
            &MemoryConfig::OFF,
        )
    }

    /// As [`Self::forward`] with an optional 9b-kv [`Flux2KvCache`] threaded through every attention
    /// layer (the double + single stacks indexed independently from 0). On the
    /// [`crate::kv_cache::CacheMode::Extract`] step the `img_ids` carry the reference tokens
    /// (`[target, ref]`); on [`crate::kv_cache::CacheMode::Cached`] steps they carry `[target]` only
    /// and the cached ref K/V are spliced back inside each attention. This is the **cache** entry
    /// point — `cache` and `control` (see [`Self::forward_with_control`]) are mutually exclusive.
    pub fn forward_with_cache(
        &self,
        inputs: &Flux2ForwardInputs,
        cache: Option<&Flux2KvCache>,
    ) -> Result<Array> {
        self.forward_inner(
            inputs.hidden_states,
            inputs.encoder_hidden_states,
            inputs.img_ids,
            inputs.txt_ids,
            inputs.timestep,
            inputs.guidance,
            cache,
            None,
            &MemoryConfig::OFF,
        )
    }

    /// As [`Self::forward_with_cache`], with an explicit [`MemoryConfig`] that bounds the per-step
    /// activation high-water (sc-6266). The generate loop passes [`MemoryConfig::LONG_SEQ`] only on
    /// the gated long-sequence multi-reference edit path; every other path uses [`MemoryConfig::OFF`]
    /// (this method with `OFF` is byte-identical to [`Self::forward_with_cache`]).
    pub fn forward_with_mem(
        &self,
        inputs: &Flux2ForwardInputs,
        cache: Option<&Flux2KvCache>,
        mem: &MemoryConfig,
    ) -> Result<Array> {
        self.forward_inner(
            inputs.hidden_states,
            inputs.encoder_hidden_states,
            inputs.img_ids,
            inputs.txt_ids,
            inputs.timestep,
            inputs.guidance,
            cache,
            None,
            mem,
        )
    }

    /// FLUX.2-dev Fun-Controlnet-Union forward (sc-2292): [`Self::forward_with_cache`] plus a VACE
    /// control branch. `control = (branch, control_context, scale)` — the branch's per-block hints are
    /// computed once from the post-embedder image+caption streams and added to the base image stream
    /// after each base double block in `branch.places`, scaled by `scale`. At `scale = 0` the result
    /// is byte-identical to the base forward (the parity self-check). This is the **control** entry
    /// point — it takes no KV cache (`cache` XOR `control`; dev control is a single embedded-guidance
    /// forward).
    pub fn forward_with_control(
        &self,
        inputs: &Flux2ForwardInputs,
        control: (&Flux2ControlBranch, &Array, f32),
    ) -> Result<Array> {
        self.forward_inner(
            inputs.hidden_states,
            inputs.encoder_hidden_states,
            inputs.img_ids,
            inputs.txt_ids,
            inputs.timestep,
            inputs.guidance,
            None,
            Some(control),
            &MemoryConfig::OFF,
        )
    }

    /// Shared body behind [`forward_with_cache`](Self::forward_with_cache) and
    /// [`forward_with_control`](Self::forward_with_control). `cache` threads the 9b-kv reference K/V
    /// (klein edit); `control` injects VACE control hints (dev pose). The two are mutually exclusive
    /// in practice (kv = klein edit; control = dev pose).
    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        hidden_states: &Array,
        encoder_hidden_states: &Array,
        img_ids: &Array,
        txt_ids: &Array,
        timestep: f32,
        guidance: Option<f32>,
        cache: Option<&Flux2KvCache>,
        control: Option<(&Flux2ControlBranch, &Array, f32)>,
        mem: &MemoryConfig,
    ) -> Result<Array> {
        let temb = self.temb(timestep, guidance)?;
        let mut img = self
            .x_embedder
            .forward(&require_f32_input(hidden_states)?)?;
        let mut txt = self
            .context_embedder
            .forward(&require_f32_input(encoder_hidden_states)?)?;

        let drop_batch = |ids: &Array| -> Result<Array> {
            let sh = ids.shape();
            if sh.len() == 3 {
                // The pos-embed table is built per-position for a single batch row; a B>1 ids tensor
                // would silently produce the wrong RoPE (and the reshape only works for B==1) (F-063).
                if sh[0] != 1 {
                    return Err(Error::Msg(format!(
                        "flux2 pos-embed ids: batch dim must be 1, got shape {sh:?}"
                    )));
                }
                Ok(ids.reshape(&[sh[1], sh[2]])?)
            } else {
                Ok(ids.clone())
            }
        };
        let (img_cos, img_sin) = self.pos_embed.forward(&drop_batch(img_ids)?)?;
        let (txt_cos, txt_sin) = self.pos_embed.forward(&drop_batch(txt_ids)?)?;
        let cos = concatenate_axis(&[&txt_cos, &img_cos], 0)?;
        let sin = concatenate_axis(&[&txt_sin, &img_sin], 0)?;

        let mi = self.mod_img.forward(&temb)?;
        let mt = self.mod_txt.forward(&temb)?;
        let img_mod = [mi[0].clone(), mi[1].clone()];
        let txt_mod = [mt[0].clone(), mt[1].clone()];

        // VACE control hints (sc-2292): computed once from the post-embedder image+caption streams,
        // before the base double-block loop (the fork's `forward_control`), then injected per block.
        let hints = match control {
            Some((branch, cc, _)) => {
                Some(branch.forward_control(&img, &txt, cc, &img_mod, &txt_mod, &cos, &sin)?)
            }
            None => None,
        };

        for (idx, block) in self.double_blocks.iter().enumerate() {
            (txt, img) = block.forward(
                img,
                txt,
                &img_mod,
                &txt_mod,
                &cos,
                &sin,
                cache.map(|c| (c, idx)),
                mem.ffn_seq_chunk,
            )?;
            // Add the control hint into the base image stream (`img + hints[n]·scale`) at the mapped
            // base double blocks. `scale = 0` → `+0` → byte-identical to the base forward.
            if let (Some(hints), Some((branch, _, scale))) = (&hints, &control) {
                if let Some(n) = branch.hint_index(idx) {
                    img = add(&img, &multiply(&hints[n], scalar(*scale))?)?;
                }
            }
            // sc-6266: cap the per-step lazy-graph peak at ~one block's transients (bit-exact). Gated
            // off (`mem.eval_per_block == false`) for every shipped path → no extra evals there.
            if mem.eval_per_block {
                mlx_rs::transforms::eval([&img, &txt])?;
            }
        }

        let txt_seq = txt.shape()[1];
        let mut hidden = concatenate_axis(&[&txt, &img], 1)?;
        let ms = self.mod_single.forward(&temb)?;
        for (idx, block) in self.single_blocks.iter().enumerate() {
            hidden = block.forward(&hidden, &ms[0], &cos, &sin, cache.map(|c| (c, idx)))?;
            // sc-6266: per-block eval-to-free (bit-exact), gated off for shipped paths.
            if mem.eval_per_block {
                mlx_rs::transforms::eval([&hidden])?;
            }
        }

        // Keep only the image tokens.
        let total_seq = hidden.shape()[1];
        if total_seq < txt_seq {
            return Err(Error::Msg(format!(
                "flux2: combined sequence length {total_seq} is shorter than the text sequence {txt_seq}"
            )));
        }
        let img_seq = total_seq - txt_seq;
        let img_idx = Array::from_slice(
            &(txt_seq..hidden.shape()[1]).collect::<Vec<i32>>(),
            &[img_seq],
        );
        let hidden = hidden.take_axis(&img_idx, 1)?;
        let hidden = self.norm_out(&hidden, &temb)?;
        self.proj_out.forward(&hidden)
    }
}

// ---- sc-2292: FLUX.2-dev Fun-Controlnet-Union (VACE-style strict pose) -------------------------
//
// Port of `alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union` (`videox_fun/models/flux2_transformer2d_control.py`):
// a VACE-style ControlNet (modified from `vace/models/wan/wan_vace.py`) added on the FIRST 4 of dev's
// 8 base double blocks (`control_layers = range(0, num_double_layers, 2) = [0, 2, 4, 6]`). A
// `control_img_in` patch embedder maps the packed control context (control latent 128 + mask 4 +
// inpaint latent 128 = 260) into the inner dim; N control double blocks thread an internal
// `(c_image, txt)` pair — block 0 seeds `c = before_proj(c) + img_embed`, each runs a full base
// double-block forward and emits `after_proj(c)` as its hint — and the hints are added into the base
// image stream after the matching base double blocks, scaled by `control_context_scale`. The control
// blocks reuse the base `double_stream_modulation_{img,txt}` + RoPE (the fork passes them through);
// the threaded `txt` is local to the control stack (the base caption stream is untouched).

/// In-features of `control_img_in`: the packed control context = control latent (128) + mask (4) +
/// inpaint latent (128) = 260, per `pipeline_flux2_control.py`
/// (`torch.concat([control_latents, mask_condition, inpaint_latent], dim=2)`).
pub const CONTROL_IN_DIM: i32 = 260;

/// One VACE control block: a full FLUX.2 double block (its own attn / ff / ff_context weights) plus
/// the `after_proj` hint projection (every block) and `before_proj` (block 0 only) seeding the
/// control branch from the base image embedding. Port of `Flux2ControlTransformerBlock`.
struct Flux2ControlBlock {
    base: DoubleBlock,
    /// `before_proj(c) + img_embed` seeds block 0 (`None` for the rest). Bias-carrying.
    before_proj: Option<AdaptableLinear>,
    /// `after_proj(c)` — the per-block hint added into the base image stream. Bias-carrying.
    after_proj: AdaptableLinear,
}

impl Flux2ControlBlock {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        heads: i32,
        head_dim: i32,
        has_before_proj: bool,
    ) -> Result<Self> {
        // The control block's attn/ff/ff_context keys match a base double block 1:1; load dense
        // (the bf16 control overlay is small enough to quantize in place after load).
        let base = DoubleBlock::from_weights(w, prefix, heads, head_dim, None)?;
        let after_proj = AdaptableLinear::dense(
            w.require(&format!("{prefix}.after_proj.weight"))?.clone(),
            Some(w.require(&format!("{prefix}.after_proj.bias"))?.clone()),
        );
        let before_proj = if has_before_proj {
            Some(AdaptableLinear::dense(
                w.require(&format!("{prefix}.before_proj.weight"))?.clone(),
                Some(w.require(&format!("{prefix}.before_proj.bias"))?.clone()),
            ))
        } else {
            None
        };
        Ok(Self {
            base,
            before_proj,
            after_proj,
        })
    }

    /// Quantize the block's base double block + the `after_proj`/`before_proj` projections (all
    /// `% 64 == 0`). The only control Linear left dense is `control_img_in` (260 in-features), handled
    /// in [`Flux2ControlBranch::quantize`].
    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.base.quantize(bits)?;
        self.after_proj.quantize(bits, None)?;
        if let Some(bp) = &mut self.before_proj {
            bp.quantize(bits, None)?;
        }
        Ok(())
    }
}

/// The FLUX.2-dev Fun-Controlnet-Union control branch (sc-2292): the `control_img_in` patch embedder
/// plus the N control blocks injecting hints into the base double blocks at `control_layers`. Built
/// from the Fun-Controlnet-Union checkpoint and driven by [`Flux2Transformer::forward_with_control`].
pub struct Flux2ControlBranch {
    /// `control_img_in`: 260 → inner. Kept **dense** (its 260 in-features is not a multiple of the
    /// quant group size 64), matching the fork's `nn.quantize` predicate. Bias-carrying.
    control_img_in: AdaptableLinear,
    blocks: Vec<Flux2ControlBlock>,
    /// Base double-block indices each control block injects into (`control_layers`); `places[n]` is
    /// the base index for hint `n` (`[0, 2, 4, 6]` for dev's 8 double blocks).
    places: Vec<usize>,
}

impl Flux2ControlBranch {
    /// Build from the Fun-Controlnet-Union checkpoint (`control` Weights). Keys are un-prefixed for a
    /// real checkpoint (`control_img_in.*`, `control_transformer_blocks.{i}.*`); `prefix` is e.g.
    /// `"w"` for a synthetic fixture. `control_layers = range(0, num_double_layers, 2)`.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &Flux2Config) -> Result<Self> {
        let p = |s: &str| {
            if prefix.is_empty() {
                s.to_string()
            } else {
                format!("{prefix}.{s}")
            }
        };
        let heads = cfg.num_heads as i32;
        let head_dim = cfg.head_dim as i32;
        let places = cfg.control_layer_places();
        let control_img_in = AdaptableLinear::dense(
            w.require(&p("control_img_in.weight"))?.clone(),
            Some(w.require(&p("control_img_in.bias"))?.clone()),
        );
        let blocks = (0..places.len())
            .map(|i| {
                Flux2ControlBlock::from_weights(
                    w,
                    &p(&format!("control_transformer_blocks.{i}")),
                    heads,
                    head_dim,
                    i == 0,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            control_img_in,
            blocks,
            places,
        })
    }

    /// Quantize the control blocks (+ their `after_proj`/`before_proj`); `control_img_in` stays dense.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for b in &mut self.blocks {
            b.quantize(bits)?;
        }
        Ok(())
    }

    /// The hint index injected at base double-block `idx`, or `None`.
    fn hint_index(&self, idx: usize) -> Option<usize> {
        self.places.iter().position(|&p| p == idx)
    }

    /// The number of per-block control hints this branch emits (= the number of control blocks =
    /// `control_layers.len()`). Exposed for the sc-8978 numeric golden.
    pub fn num_hints(&self) -> usize {
        self.blocks.len()
    }

    /// Run the control stack → per-block hints (the fork's `forward_control`). `img_embed`/`txt_embed`
    /// are the post-embedder base streams; `control_context` is the packed 260-ch control context;
    /// `img_mod`/`txt_mod`/`cos`/`sin` are the shared base double-stream modulation + RoPE (the control
    /// blocks reuse the base modulation, per the fork). The threaded `txt` is local to the control
    /// stack — only the image-stream hints leave. `pub` so the sc-8978 numeric golden can drive it
    /// directly against the authoritative VideoX-Fun `Flux2ControlTransformer2DModel.forward_control`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_control(
        &self,
        img_embed: &Array,
        txt_embed: &Array,
        control_context: &Array,
        img_mod: &[(Array, Array, Array); 2],
        txt_mod: &[(Array, Array, Array); 2],
        cos: &Array,
        sin: &Array,
    ) -> Result<Vec<Array>> {
        let mut c = self
            .control_img_in
            .forward(&require_f32_input(control_context)?)?;
        let mut txt = txt_embed.clone();
        let mut hints = Vec::with_capacity(self.blocks.len());
        for (i, block) in self.blocks.iter().enumerate() {
            if i == 0 {
                let bp = block.before_proj.as_ref().ok_or_else(|| {
                    Error::Msg("flux2 control block 0 is missing before_proj".into())
                })?;
                c = add(&bp.forward(&c)?, img_embed)?;
            }
            let (new_txt, new_c) = block
                .base
                .forward(c, txt, img_mod, txt_mod, cos, sin, None, None)?;
            hints.push(block.after_proj.forward(&new_c)?);
            c = new_c;
            txt = new_txt;
        }
        Ok(hints)
    }
}

/// The FLUX.2-dev base MMDiT + its Fun-Controlnet-Union control branch (sc-2292). Composes the
/// parity-proven [`Flux2Transformer`] with a [`Flux2ControlBranch`]; [`forward`](Self::forward)
/// threads the control context + scale, and [`quantize`](Self::quantize) packs both (the base
/// no-ops if it was loaded pre-quantized; the dense control overlay packs here).
pub struct Flux2ControlTransformer {
    base: Flux2Transformer,
    branch: Flux2ControlBranch,
}

impl Flux2ControlTransformer {
    pub fn new(base: Flux2Transformer, branch: Flux2ControlBranch) -> Self {
        Self { base, branch }
    }

    /// Quantize the base + the control branch. On a dev pre-quantized snapshot (sc-5917) the base is
    /// already packed, so `base.quantize` is a no-op; the bf16 control overlay packs here.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.base.quantize(bits)?;
        self.branch.quantize(bits)?;
        Ok(())
    }

    /// Adapter host = the base DiT (LoRA/LoKr target; the control branch is never an adapter target,
    /// mirroring the Z-Image control port).
    pub fn base_mut(&mut self) -> &mut Flux2Transformer {
        &mut self.base
    }

    /// Control forward: latent `[B, seq, in]` + text embeds + ids + timestep + embedded `guidance` +
    /// packed `control_context` (260ch, same image seq as the latent) + `control_context_scale`.
    /// Returns the velocity `[B, seq_img, out]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Array,
        encoder_hidden_states: &Array,
        img_ids: &Array,
        txt_ids: &Array,
        timestep: f32,
        guidance: Option<f32>,
        control_context: &Array,
        control_context_scale: f32,
    ) -> Result<Array> {
        self.base.forward_with_control(
            &Flux2ForwardInputs {
                hidden_states,
                encoder_hidden_states,
                img_ids,
                txt_ids,
                timestep,
                guidance,
            },
            (&self.branch, control_context, control_context_scale),
        )
    }
}

// ---- LoRA/LoKr adapter routing (sc-2646) ------------------------------------------------------
//
// The Rust analog of the fork's `Flux2LoRAMapping`: map the trained-file (diffusers) module paths
// to the crate's `AdaptableLinear` fields, across the FULL transformer-only surface (globals +
// double + single blocks). VAE + Qwen3 TE are NOT LoRA targets. The fork's standard/diffusers
// naming is what these resolve (bare / `transformer.` / `diffusion_model.` prefixes are stripped by
// the core loader before the path reaches here); the BFL/ComfyUI fused-qkv-split + kohya `lora_unet_`
// namings are a separate cross-family format (sc-2618), not handled here.

impl AdaptableHost for Modulation {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["linear"] => Some(&mut self.linear),
            _ => None,
        }
    }
}

impl AdaptableHost for FeedForward {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["linear_in"] => Some(&mut self.linear_in),
            ["linear_out"] => Some(&mut self.linear_out),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        ["linear_in", "linear_out"]
            .into_iter()
            .map(String::from)
            .collect()
    }
}

impl AdaptableHost for DoubleAttention {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        // Trained-file (diffusers) naming → fields: image stream `to_q/k/v`/`to_out`; text stream
        // `add_{q,k,v}_proj` → `add_{q,k,v}` and `to_add_out`.
        match path {
            ["to_q"] => Some(&mut self.to_q),
            ["to_k"] => Some(&mut self.to_k),
            ["to_v"] => Some(&mut self.to_v),
            // The fork accepts both the bare `to_out` and the HF-style `to_out.0` (diffusers wraps
            // the output projection in a `Sequential[Linear, Dropout]`); both address this Linear.
            ["to_out"] | ["to_out", "0"] => Some(&mut self.to_out),
            ["add_q_proj"] => Some(&mut self.add_q),
            ["add_k_proj"] => Some(&mut self.add_k),
            ["add_v_proj"] => Some(&mut self.add_v),
            ["to_add_out"] => Some(&mut self.to_add_out),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        // Both `to_out` and the HF-style `to_out.0` alias resolve to the output projection, and the
        // fork carries a `lora_unet_…_attn_to_out` *and* `…_attn_to_out_0` kohya pattern — emit both
        // so either flattened spelling resolves.
        [
            "to_q",
            "to_k",
            "to_v",
            "to_out",
            "to_out.0",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }
}

impl AdaptableHost for DoubleBlock {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["attn", rest @ ..] => self.attn.adaptable_mut(rest),
            ["ff", rest @ ..] => self.ff.adaptable_mut(rest),
            ["ff_context", rest @ ..] => self.ff_context.adaptable_mut(rest),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        let mut out = prefixed_paths("attn", &self.attn);
        out.extend(prefixed_paths("ff", &self.ff));
        out.extend(prefixed_paths("ff_context", &self.ff_context));
        out
    }
}

impl AdaptableHost for SingleBlock {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        // The fused `to_qkv_mlp_proj` takes a single LoRA covering q/k/v/mlp jointly (the fork maps
        // it as one target); `to_out` is the output projection.
        match path {
            ["attn", "to_qkv_mlp_proj"] => Some(&mut self.to_qkv_mlp),
            ["attn", "to_out"] => Some(&mut self.to_out),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        ["attn.to_qkv_mlp_proj", "attn.to_out"]
            .into_iter()
            .map(String::from)
            .collect()
    }
}

impl AdaptableHost for Flux2Transformer {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            // Globals.
            ["x_embedder"] => Some(&mut self.x_embedder),
            ["context_embedder"] => Some(&mut self.context_embedder),
            ["proj_out"] => Some(&mut self.proj_out),
            ["norm_out", "linear"] => Some(&mut self.norm_out_linear),
            ["double_stream_modulation_img", rest @ ..] => self.mod_img.adaptable_mut(rest),
            ["double_stream_modulation_txt", rest @ ..] => self.mod_txt.adaptable_mut(rest),
            ["single_stream_modulation", rest @ ..] => self.mod_single.adaptable_mut(rest),
            ["time_guidance_embed", "linear_1"] => Some(&mut self.time_linear1),
            ["time_guidance_embed", "linear_2"] => Some(&mut self.time_linear2),
            // The embedded-guidance branch (`time_guidance_embed.guidance_embedder.linear_{1,2}`)
            // exists only on FLUX.2-**dev** (sc-5920); klein is guidance-distilled so these fields
            // are `None` → `as_mut()` yields `None` → unmatched, exactly as before. Routed for
            // symmetry with the timestep branch above so a dev "all-linear" LoRA that includes the
            // guidance embedder resolves instead of failing the strict no-silent-drop apply.
            ["time_guidance_embed", "guidance_embedder", "linear_1"] => {
                self.guidance_linear1.as_mut()
            }
            ["time_guidance_embed", "guidance_embedder", "linear_2"] => {
                self.guidance_linear2.as_mut()
            }
            ["transformer_blocks", n, rest @ ..] => self
                .double_blocks
                .get_mut(n.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            ["single_transformer_blocks", n, rest @ ..] => self
                .single_blocks
                .get_mut(n.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            _ => None,
        }
    }

    /// kohya-reachable targets (sc-2618): the diffusers-named double + single block linears. Globals
    /// (`x_embedder`/`context_embedder`/`proj_out`/`norm_out`/the modulations/`time_guidance_embed`)
    /// carry no `lora_unet_` pattern in the fork mapping, so they are excluded (reachable via the
    /// dotted form). The fused→split BFL convention (`double_blocks_*`/`single_blocks_*`) is a
    /// different format (sc-2743) and is intentionally NOT enumerated → such keys surface as unmatched.
    fn adaptable_paths(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (i, b) in self.double_blocks.iter().enumerate() {
            out.extend(prefixed_paths(&format!("transformer_blocks.{i}"), b));
        }
        for (i, b) in self.single_blocks.iter().enumerate() {
            out.extend(prefixed_paths(&format!("single_transformer_blocks.{i}"), b));
        }
        out
    }

    /// BFL / ComfyUI fused→split targets (sc-2743), the Rust analog of the fork's
    /// `Flux2LoRAMapping._get_bfl_*` + the `base_model.model.` global renames. Three things sc-2618's
    /// diffusers/peft/kohya paths can't do, all here:
    /// - **fused-qkv split**: the BFL `double_blocks.{n}.{img,txt}_attn.qkv` linear is one fused
    ///   `[3·inner, …]` projection; FLUX.2's model keeps q/k/v SEPARATE (`attn.to_q/to_k/to_v`,
    ///   `add_{q,k,v}_proj`), so each destination row-slices its third (equal 3-way; `inner`-independent).
    /// - **BFL module renames**: `img_attn.proj`→`to_out`, `txt_attn.proj`→`to_add_out`,
    ///   `{img,txt}_mlp.{0,2}`→`ff{_context}.linear_{in,out}`, `single_blocks.{n}.linear{1,2}`→
    ///   `attn.{to_qkv_mlp_proj,to_out}` (linear1 stays FUSED in FLUX.2 → no split), and the global
    ///   `base_model.model.` renames (`img_in`→`x_embedder`, `final_layer.linear`→`proj_out`, …).
    /// - **the `diffusion_model.` / `base_model.model.` dotted prefixes** carrying BFL module names.
    ///
    /// klein-absent globals (`norm_out`, `guidance_linear_*`) have no `base_model.model.` BFL spelling
    /// in the fork, so they're omitted; their diffusers-named forms stay peft-reachable. The 4-way
    /// qkv-mlp split (`_split_qkv_mlp_up`) is FLUX.1's (separate `proj_mlp`) and lands with sc-2657.
    fn bfl_targets(&self) -> Vec<BflTarget> {
        let mut out = Vec::new();

        // Globals: `base_model.model.` BFL renames only (the diffusers-named globals — bare /
        // `transformer.` / `diffusion_model.` — are already covered by the peft loader).
        for (bfl, tgt) in [
            ("img_in", "x_embedder"),
            ("txt_in", "context_embedder"),
            ("time_in.in_layer", "time_guidance_embed.linear_1"),
            ("time_in.out_layer", "time_guidance_embed.linear_2"),
            (
                "double_stream_modulation_img.lin",
                "double_stream_modulation_img.linear",
            ),
            (
                "double_stream_modulation_txt.lin",
                "double_stream_modulation_txt.linear",
            ),
            (
                "single_stream_modulation.lin",
                "single_stream_modulation.linear",
            ),
            ("final_layer.linear", "proj_out"),
        ] {
            let (up, down, alpha) = bfl_global_keys(bfl);
            out.push(rename_target(tgt, up, down, alpha));
        }

        // Double blocks.
        for i in 0..self.double_blocks.len() {
            // Fused qkv → split: img → to_{q,k,v}; txt → add_{q,k,v}_proj.
            for (stream, dst) in [
                ("img", ["to_q", "to_k", "to_v"]),
                ("txt", ["add_q_proj", "add_k_proj", "add_v_proj"]),
            ] {
                let flat = format!("double_blocks_{i}_{stream}_attn_qkv");
                let dotted = format!("double_blocks.{i}.{stream}_attn.qkv");
                let (up, down, alpha) = bfl_block_keys(&flat, &dotted);
                for idx in 0..3i32 {
                    out.push(BflTarget {
                        target_path: format!("transformer_blocks.{i}.attn.{}", dst[idx as usize]),
                        up_keys: up.clone(),
                        down_keys: down.clone(),
                        alpha_keys: alpha.clone(),
                        up_slice: Some(LoraRowSlice::Chunk { n: 3, index: idx }),
                        down_slice: Some(LoraRowSlice::ChunkIfDivisible { n: 3, index: idx }),
                    });
                }
            }
            // attn output proj (rename, no split): img.proj → to_out; txt.proj → to_add_out.
            for (stream, tgt) in [("img", "to_out"), ("txt", "to_add_out")] {
                let flat = format!("double_blocks_{i}_{stream}_attn_proj");
                let dotted = format!("double_blocks.{i}.{stream}_attn.proj");
                let (up, down, alpha) = bfl_block_keys(&flat, &dotted);
                out.push(rename_target(
                    &format!("transformer_blocks.{i}.attn.{tgt}"),
                    up,
                    down,
                    alpha,
                ));
            }
            // MLP (rename): img_mlp.{0,2} → ff.linear_{in,out}; txt_mlp.{0,2} → ff_context.linear_{in,out}.
            for (stream, ff) in [("img", "ff"), ("txt", "ff_context")] {
                for (n, lin) in [("0", "linear_in"), ("2", "linear_out")] {
                    let flat = format!("double_blocks_{i}_{stream}_mlp_{n}");
                    let dotted = format!("double_blocks.{i}.{stream}_mlp.{n}");
                    let (up, down, alpha) = bfl_block_keys(&flat, &dotted);
                    out.push(rename_target(
                        &format!("transformer_blocks.{i}.{ff}.{lin}"),
                        up,
                        down,
                        alpha,
                    ));
                }
            }
        }

        // Single blocks (rename, FUSED — no split): linear1 → attn.to_qkv_mlp_proj; linear2 → attn.to_out.
        for i in 0..self.single_blocks.len() {
            for (which, tgt) in [("linear1", "to_qkv_mlp_proj"), ("linear2", "to_out")] {
                let flat = format!("single_blocks_{i}_{which}");
                let dotted = format!("single_blocks.{i}.{which}");
                let (up, down, alpha) = bfl_block_keys(&flat, &dotted);
                out.push(rename_target(
                    &format!("single_transformer_blocks.{i}.attn.{tgt}"),
                    up,
                    down,
                    alpha,
                ));
            }
        }

        out
    }
}

/// A non-split BFL target (a plain module rename): the source factors are copied through, no slice.
fn rename_target(
    target_path: &str,
    up_keys: Vec<String>,
    down_keys: Vec<String>,
    alpha_keys: Vec<String>,
) -> BflTarget {
    BflTarget {
        target_path: target_path.to_string(),
        up_keys,
        down_keys,
        alpha_keys,
        up_slice: None,
        down_slice: None,
    }
}

/// Every BFL source-key spelling for one *block* linear, across the three prefix conventions: kohya
/// `lora_unet_<flat>` (flattened module path) and the dotted `diffusion_model.<dotted>` /
/// `base_model.model.<dotted>` (both BFL-named for block layers), partitioned into (up, down, alpha)
/// — `lora_up`≡`lora_B`, `lora_down`≡`lora_A`. Mirrors the fork's BFL `possible_*_patterns`.
fn bfl_block_keys(flat: &str, dotted: &str) -> (Vec<String>, Vec<String>, Vec<String>) {
    let up = vec![
        format!("lora_unet_{flat}.lora_up.weight"),
        format!("diffusion_model.{dotted}.lora_B.weight"),
        format!("diffusion_model.{dotted}.lora_up.weight"),
        format!("base_model.model.{dotted}.lora_B.weight"),
        format!("base_model.model.{dotted}.lora_up.weight"),
    ];
    let down = vec![
        format!("lora_unet_{flat}.lora_down.weight"),
        format!("diffusion_model.{dotted}.lora_A.weight"),
        format!("diffusion_model.{dotted}.lora_down.weight"),
        format!("base_model.model.{dotted}.lora_A.weight"),
        format!("base_model.model.{dotted}.lora_down.weight"),
    ];
    let alpha = vec![
        format!("lora_unet_{flat}.alpha"),
        format!("diffusion_model.{dotted}.alpha"),
        format!("base_model.model.{dotted}.alpha"),
    ];
    (up, down, alpha)
}

/// BFL source-key spellings for a *global* linear: only the `base_model.model.<bfl_name>` form adds
/// new coverage (the diffusers-named globals are peft-reachable), so the fork carries no `lora_unet_`
/// or `diffusion_model.` BFL-named global pattern. (up, down, alpha).
fn bfl_global_keys(bfl: &str) -> (Vec<String>, Vec<String>, Vec<String>) {
    let up = vec![
        format!("base_model.model.{bfl}.lora_B.weight"),
        format!("base_model.model.{bfl}.lora_up.weight"),
    ];
    let down = vec![
        format!("base_model.model.{bfl}.lora_A.weight"),
        format!("base_model.model.{bfl}.lora_down.weight"),
    ];
    let alpha = vec![format!("base_model.model.{bfl}.alpha")];
    (up, down, alpha)
}

/// Configuration glue so callers can keep the transformer's dims in one place.
pub type Flux2TransformerConfig = Flux2Config;

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::adapters::{install_adapter, Adapter};

    #[test]
    fn timestep_embedding_shape_and_flip() {
        let t = Array::from_slice(&[1000.0f32], &[1]);
        let emb = timestep_embedding(&t, 256).unwrap();
        assert_eq!(emb.shape(), &[1, 256]);
    }

    // ---- sc-2646 adapter routing (diffusers-name → field translation) -------------------------

    fn dummy_lin() -> AdaptableLinear {
        AdaptableLinear::dense(Array::from_slice(&[0.0f32], &[1, 1]), None)
    }
    fn dummy_arr() -> Array {
        Array::from_slice(&[1.0f32], &[1])
    }
    fn noop_adapter() -> Adapter {
        Adapter::Lora {
            a: Array::from_slice(&[0.0f32], &[1, 1]),
            b: Array::from_slice(&[0.0f32], &[1, 1]),
            scale: 0.0,
        }
    }
    /// Path resolves iff installing a no-op adapter there succeeds.
    fn resolves(host: &mut impl AdaptableHost, path: &str) -> bool {
        install_adapter(host, path, noop_adapter()).is_ok()
    }

    fn double_attn() -> DoubleAttention {
        DoubleAttention {
            to_q: dummy_lin(),
            to_k: dummy_lin(),
            to_v: dummy_lin(),
            to_out: dummy_lin(),
            norm_q: dummy_arr(),
            norm_k: dummy_arr(),
            add_q: dummy_lin(),
            add_k: dummy_lin(),
            add_v: dummy_lin(),
            to_add_out: dummy_lin(),
            norm_added_q: dummy_arr(),
            norm_added_k: dummy_arr(),
            heads: 1,
            head_dim: 1,
        }
    }

    #[test]
    fn double_attention_routes_diffusers_names() {
        let mut attn = double_attn();
        for p in [
            "to_q",
            "to_k",
            "to_v",
            "to_out",
            "to_out.0", // HF-style diffusers Sequential alias (the fork accepts both)
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
        ] {
            assert!(resolves(&mut attn, p), "{p} should resolve");
        }
        // Internal field names + off-surface must not resolve.
        for p in ["add_q", "add_k", "add_v", "to_add_out.0", "qkv"] {
            assert!(!resolves(&mut attn, p), "{p} must not resolve");
        }
    }

    #[test]
    fn double_block_routes_attn_and_ffs() {
        let mut block = DoubleBlock {
            attn: double_attn(),
            ff: FeedForward {
                linear_in: dummy_lin(),
                linear_out: dummy_lin(),
            },
            ff_context: FeedForward {
                linear_in: dummy_lin(),
                linear_out: dummy_lin(),
            },
        };
        for p in [
            "attn.to_q",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.linear_in",
            "ff.linear_out",
            "ff_context.linear_in",
            "ff_context.linear_out",
        ] {
            assert!(resolves(&mut block, p), "{p} should resolve");
        }
        for p in ["ff.net.0.proj", "mlp.linear_in", "attn.to_qkv_mlp_proj"] {
            assert!(!resolves(&mut block, p), "{p} must not resolve");
        }
    }

    #[test]
    fn single_block_routes_fused_qkv_mlp() {
        let mut block = SingleBlock {
            to_qkv_mlp: dummy_lin(),
            to_out: dummy_lin(),
            norm_q: dummy_arr(),
            norm_k: dummy_arr(),
            heads: 1,
            head_dim: 1,
            inner: 1,
        };
        // The fused projection is addressed by its checkpoint name `attn.to_qkv_mlp_proj`.
        assert!(resolves(&mut block, "attn.to_qkv_mlp_proj"));
        assert!(resolves(&mut block, "attn.to_out"));
        // The internal field name + split q/k/v must NOT resolve (single LoRA covers them jointly).
        for p in ["to_qkv_mlp", "attn.to_q", "attn.to_qkv_mlp_proj.0"] {
            assert!(!resolves(&mut block, p), "{p} must not resolve");
        }
    }

    #[test]
    fn modulation_and_feed_forward_route_leaf_names() {
        let mut m = Modulation {
            linear: dummy_lin(),
            sets: 1,
        };
        assert!(resolves(&mut m, "linear"));
        assert!(!resolves(&mut m, "weight"));

        let mut ff = FeedForward {
            linear_in: dummy_lin(),
            linear_out: dummy_lin(),
        };
        assert!(resolves(&mut ff, "linear_in"));
        assert!(resolves(&mut ff, "linear_out"));
        assert!(!resolves(&mut ff, "net.0.proj"));
    }

    fn ff() -> FeedForward {
        FeedForward {
            linear_in: dummy_lin(),
            linear_out: dummy_lin(),
        }
    }
    fn modulation(sets: usize) -> Modulation {
        Modulation {
            linear: dummy_lin(),
            sets,
        }
    }

    /// A minimal transformer (1 double + 1 single block) for the top-level key→module routing —
    /// the globals' diffusers-name translations + the block-index parse.
    fn tiny_transformer() -> Flux2Transformer {
        Flux2Transformer {
            pos_embed: Flux2PosEmbed::new(2000.0, [32, 32, 32, 32]),
            time_linear1: dummy_lin(),
            time_linear2: dummy_lin(),
            guidance_linear1: None,
            guidance_linear2: None,
            mod_img: modulation(2),
            mod_txt: modulation(2),
            mod_single: modulation(1),
            x_embedder: dummy_lin(),
            context_embedder: dummy_lin(),
            double_blocks: vec![DoubleBlock {
                attn: double_attn(),
                ff: ff(),
                ff_context: ff(),
            }],
            single_blocks: vec![SingleBlock {
                to_qkv_mlp: dummy_lin(),
                to_out: dummy_lin(),
                norm_q: dummy_arr(),
                norm_k: dummy_arr(),
                heads: 1,
                head_dim: 1,
                inner: 1,
            }],
            norm_out_linear: dummy_lin(),
            proj_out: dummy_lin(),
            time_channels: 256,
        }
    }

    #[test]
    fn transformer_routes_full_diffusers_surface() {
        let mut t = tiny_transformer();
        // Globals (diffusers names → internal fields).
        for p in [
            "x_embedder",
            "context_embedder",
            "proj_out",
            "norm_out.linear",
            "double_stream_modulation_img.linear",
            "double_stream_modulation_txt.linear",
            "single_stream_modulation.linear",
            "time_guidance_embed.linear_1",
            "time_guidance_embed.linear_2",
            // Double block 0.
            "transformer_blocks.0.attn.to_q",
            "transformer_blocks.0.attn.add_k_proj",
            "transformer_blocks.0.attn.to_add_out",
            "transformer_blocks.0.ff.linear_in",
            "transformer_blocks.0.ff_context.linear_out",
            // Single block 0.
            "single_transformer_blocks.0.attn.to_qkv_mlp_proj",
            "single_transformer_blocks.0.attn.to_out",
        ] {
            assert!(resolves(&mut t, p), "{p} should resolve");
        }
        // Off-surface / wrong index / klein-absent guidance linears must NOT resolve.
        for p in [
            "norm_out_linear",
            "time_guidance_embed.guidance_linear_1",
            "transformer_blocks.1.attn.to_q", // only 1 double block here
            "single_transformer_blocks.5.attn.to_out",
            "transformer_blocks.0.attn.qkv",
            "vae.encoder",
        ] {
            assert!(!resolves(&mut t, p), "{p} must not resolve");
        }
    }

    // ---- sc-2618 kohya `lora_unet_` routing (no real weights) ---------------------------------

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("mlx_gen_flux2_kohya_test");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    /// `adaptable_paths()` is the kohya-reachable surface; every entry must resolve via
    /// `adaptable_mut` (drift guard) and flatten to a collision-free stem (so the table is 1:1).
    #[test]
    fn adaptable_paths_resolve_and_flatten_uniquely() {
        let mut t = tiny_transformer();
        let paths = t.adaptable_paths();
        assert!(!paths.is_empty());
        // Drift guard: each enumerated path resolves through the matcher.
        for p in &paths {
            assert!(
                resolves(&mut t, p),
                "enumerated {p} does not resolve via adaptable_mut"
            );
        }
        // Globals are excluded from the kohya surface.
        for g in [
            "x_embedder",
            "proj_out",
            "norm_out.linear",
            "time_guidance_embed.linear_1",
        ] {
            assert!(
                !paths.iter().any(|p| p == g),
                "global {g} must be excluded from kohya"
            );
        }
        // Collision-free flattening.
        let flat: std::collections::BTreeSet<String> =
            paths.iter().map(|p| p.replace('.', "_")).collect();
        assert_eq!(
            flat.len(),
            paths.len(),
            "two paths flattened to the same kohya stem"
        );
        // The `to_out` / `to_out.0` aliases both appear (the fork emits both kohya spellings).
        assert!(paths
            .iter()
            .any(|p| p == "transformer_blocks.0.attn.to_out"));
        assert!(paths
            .iter()
            .any(|p| p == "transformer_blocks.0.attn.to_out.0"));
    }

    /// A diffusers-named kohya file applies through the strict provider seam (every stem resolves).
    #[test]
    fn kohya_diffusers_applies() {
        use crate::adapters::apply_flux2_adapters;
        use mlx_gen::runtime::{AdapterKind, AdapterSpec};

        let small = Array::from_slice(&[0.01f32], &[1, 1]); // [r=1,in=1] / [out=1,r=1]
        let meta = None as Option<&std::collections::HashMap<String, String>>;

        // One kohya key pair per reachable stem.
        let mut t = tiny_transformer();
        let n = t.adaptable_paths().len();
        let mut arrays: Vec<(String, &Array)> = Vec::new();
        for stem in t.adaptable_paths().iter().map(|p| p.replace('.', "_")) {
            arrays.push((format!("lora_unet_{stem}.lora_down.weight"), &small));
            arrays.push((format!("lora_unet_{stem}.lora_up.weight"), &small));
        }
        let refs: Vec<(&str, &Array)> = arrays.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let path = tmp("flux2_kohya_diffusers.safetensors");
        Array::save_safetensors(refs, meta, &path).unwrap();
        let report = apply_flux2_adapters(
            &mut t,
            &[AdapterSpec {
                path,
                scale: 1.0,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
        )
        .unwrap();
        assert_eq!(
            report.applied, n,
            "every diffusers-named kohya stem should resolve"
        );
        assert!(report.unmatched_paths.is_empty());
    }

    // ---- sc-5920 FLUX.2-dev adapters (wider/deeper graph + the dev guidance embedder) ----------
    //
    // dev shares the `Flux2Transformer` (so the klein adapter engine + key→module map serve it), but
    // its DiT is wider/deeper — 8 double + **48** single blocks (vs klein's 24) — and it carries the
    // embedded-guidance embedder klein lacks. These no-real-weight tests pin that the path-addressed
    // install covers the full dev graph; `dev_adapter_real_weights.rs` is the on-Mac render check.

    /// A dev-shaped transformer: 8 double + 48 single blocks and the embedded-guidance embedder
    /// present (`Some`) — the structural deltas from `tiny_transformer` (klein) that this story pins.
    fn dev_transformer() -> Flux2Transformer {
        Flux2Transformer {
            pos_embed: Flux2PosEmbed::new(2000.0, [32, 32, 32, 32]),
            time_linear1: dummy_lin(),
            time_linear2: dummy_lin(),
            // dev's embedded distilled-guidance branch (klein is `None` here).
            guidance_linear1: Some(dummy_lin()),
            guidance_linear2: Some(dummy_lin()),
            mod_img: modulation(2),
            mod_txt: modulation(2),
            mod_single: modulation(1),
            x_embedder: dummy_lin(),
            context_embedder: dummy_lin(),
            double_blocks: (0..8)
                .map(|_| DoubleBlock {
                    attn: double_attn(),
                    ff: ff(),
                    ff_context: ff(),
                })
                .collect(),
            single_blocks: (0..48)
                .map(|_| SingleBlock {
                    to_qkv_mlp: dummy_lin(),
                    to_out: dummy_lin(),
                    norm_q: dummy_arr(),
                    norm_k: dummy_arr(),
                    heads: 1,
                    head_dim: 1,
                    inner: 1,
                })
                .collect(),
            norm_out_linear: dummy_lin(),
            proj_out: dummy_lin(),
            time_channels: 256,
        }
    }

    /// The dev key→module map resolves the WIDER/DEEPER graph (every one of the 8 double + 48 single
    /// blocks, incl. the last) plus the dev-only embedded-guidance embedder, and still rejects
    /// out-of-range indices.
    #[test]
    fn dev_routes_wider_graph_and_guidance_embedder() {
        let mut t = dev_transformer();

        // The dev embedded-guidance embedder resolves (on klein, `guidance_linear*` is `None` →
        // `as_mut()` is `None` → these would NOT resolve — the structural delta this story pins).
        for p in [
            "time_guidance_embed.guidance_embedder.linear_1",
            "time_guidance_embed.guidance_embedder.linear_2",
        ] {
            assert!(
                resolves(&mut t, p),
                "dev guidance embedder {p} should resolve"
            );
        }

        // Every double block (0..8) and EVERY single block (0..48), including the deepest indices that
        // klein's 24-block graph never reaches — the path-addressed install scales with the config.
        for i in 0..8 {
            for tgt in [
                "attn.to_q",
                "attn.to_add_out",
                "ff.linear_out",
                "ff_context.linear_in",
            ] {
                let p = format!("transformer_blocks.{i}.{tgt}");
                assert!(resolves(&mut t, &p), "{p} should resolve");
            }
        }
        for i in 0..48 {
            for tgt in ["attn.to_qkv_mlp_proj", "attn.to_out"] {
                let p = format!("single_transformer_blocks.{i}.{tgt}");
                assert!(resolves(&mut t, &p), "{p} should resolve");
            }
        }

        // Out of range for dev (8 double: 0..7; 48 single: 0..47) and the klein-spelled guidance
        // linears (not the dev `guidance_embedder.*` path) must NOT resolve.
        for p in [
            "transformer_blocks.8.attn.to_q",
            "single_transformer_blocks.48.attn.to_out",
            "time_guidance_embed.guidance_linear_1",
        ] {
            assert!(!resolves(&mut t, p), "{p} must not resolve");
        }
    }

    /// The full dev kohya surface applies through the strict provider: one `lora_unet_` key-pair per
    /// enumerated stem resolves, none unmatched. Pins the dev surface count = 8 double × 13 + 48
    /// single × 2 = 200 (klein's is 8×13 + 24×2 = 152), proving the wider/deeper graph is covered.
    #[test]
    fn dev_kohya_full_surface_applies() {
        use crate::adapters::apply_flux2_adapters;
        use mlx_gen::runtime::{AdapterKind, AdapterSpec};

        let small = Array::from_slice(&[0.01f32], &[1, 1]);
        let meta = None as Option<&std::collections::HashMap<String, String>>;

        let mut t = dev_transformer();
        let n = t.adaptable_paths().len();
        assert_eq!(n, 200, "dev kohya surface (8×13 double + 48×2 single)");

        let mut arrays: Vec<(String, &Array)> = Vec::new();
        for stem in t.adaptable_paths().iter().map(|p| p.replace('.', "_")) {
            arrays.push((format!("lora_unet_{stem}.lora_down.weight"), &small));
            arrays.push((format!("lora_unet_{stem}.lora_up.weight"), &small));
        }
        let refs: Vec<(&str, &Array)> = arrays.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let path = tmp("flux2_dev_kohya_full.safetensors");
        Array::save_safetensors(refs, meta, &path).unwrap();
        let report = apply_flux2_adapters(
            &mut t,
            &[AdapterSpec {
                path,
                scale: 1.0,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
        )
        .unwrap();
        assert_eq!(report.applied, n, "every dev kohya stem should resolve");
        assert!(report.unmatched_paths.is_empty());
    }

    /// A peft LoKr file (bare module paths, `networkType=lokr`) resolves + installs across a span of
    /// the dev graph (first/last single block, a deep double block) — LoKr falls out of the same
    /// family-agnostic engine as LoRA, on the wider graph.
    #[test]
    fn dev_lokr_resolves_on_wider_graph() {
        use crate::adapters::apply_flux2_adapters;
        use mlx_gen::runtime::{AdapterKind, AdapterSpec};

        // Minimal valid peft LoKr for a [1,1] base: w1=[1,1], low-rank w2 = w2_a@w2_b = [1,1] →
        // kron(w1, w2) reshapes to [1,1]. (The real-weight test exercises true block shapes.)
        let w1 = Array::from_slice(&[1.0f32], &[1, 1]);
        let w2a = Array::from_slice(&[0.5f32], &[1, 1]);
        let w2b = Array::from_slice(&[0.5f32], &[1, 1]);
        let targets = [
            "single_transformer_blocks.0.attn.to_qkv_mlp_proj",
            "single_transformer_blocks.47.attn.to_out",
            "transformer_blocks.7.attn.to_q",
            "transformer_blocks.7.ff.linear_in",
        ];
        let mut arrays: Vec<(String, &Array)> = Vec::new();
        for tgt in targets {
            arrays.push((format!("{tgt}.lokr_w1"), &w1));
            arrays.push((format!("{tgt}.lokr_w2_a"), &w2a));
            arrays.push((format!("{tgt}.lokr_w2_b"), &w2b));
        }
        let refs: Vec<(&str, &Array)> = arrays.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let mut md = std::collections::HashMap::new();
        md.insert("networkType".to_string(), "lokr".to_string());
        md.insert("rank".to_string(), "1".to_string());
        md.insert("alpha".to_string(), "1".to_string());
        let path = tmp("flux2_dev_lokr.safetensors");
        Array::save_safetensors(refs, Some(&md), &path).unwrap();

        let mut t = dev_transformer();
        let report = apply_flux2_adapters(
            &mut t,
            &[AdapterSpec {
                path,
                scale: 1.0,
                kind: AdapterKind::Lokr,
                pass_scales: None,
                moe_expert: None,
            }],
        )
        .unwrap();
        assert_eq!(
            report.applied,
            targets.len(),
            "every dev LoKr target should resolve"
        );
        assert!(report.unmatched_paths.is_empty());
        // Each target carries one real LoKr delta (not silently dropped / turned into a no-op).
        for tgt in targets {
            let segs: Vec<&str> = tgt.split('.').collect();
            let installed = AdaptableHost::adaptable_mut(&mut t, &segs)
                .unwrap()
                .adapters();
            assert!(
                matches!(installed, [Adapter::Lokr { .. }]),
                "expected exactly one LoKr adapter at {tgt}"
            );
        }
    }

    // ---- sc-2743 BFL / ComfyUI fused→split routing (no real weights) --------------------------

    /// The full `bfl_targets()` surface: drift guard (every target resolves), count, collision-free
    /// target paths, the fused-qkv 3-way fan-out, and the FLUX.2 single-block `linear1` staying FUSED.
    #[test]
    fn bfl_targets_resolve_full_surface() {
        let mut t = tiny_transformer();
        let targets = t.bfl_targets();
        // 8 globals + (1 double block × 12) + (1 single block × 2) = 22.
        assert_eq!(
            targets.len(),
            22,
            "BFL target count for a 1-double + 1-single tiny transformer"
        );
        // Drift guard: every BFL target path resolves through the matcher.
        for tg in &targets {
            let segs: Vec<&str> = tg.target_path.split('.').collect();
            assert!(
                AdaptableHost::adaptable_mut(&mut t, &segs).is_some(),
                "BFL target {} does not resolve via adaptable_mut",
                tg.target_path
            );
        }
        // Distinct destinations (the qkv fan-out is across DIFFERENT targets, never a collision).
        let distinct: std::collections::BTreeSet<&String> =
            targets.iter().map(|tg| &tg.target_path).collect();
        assert_eq!(distinct.len(), targets.len(), "two BFL targets collide");

        // Fused img-qkv up key feeds exactly to_q/to_k/to_v with Chunk index 0/1/2.
        let qkv_up = "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight";
        let mut fanned: Vec<(String, i32)> = targets
            .iter()
            .filter(|tg| tg.up_keys.iter().any(|k| k == qkv_up))
            .map(|tg| {
                let idx = match &tg.up_slice {
                    Some(LoraRowSlice::Chunk { index, .. }) => *index,
                    _ => panic!("qkv target {} lacks a Chunk up-slice", tg.target_path),
                };
                (tg.target_path.clone(), idx)
            })
            .collect();
        fanned.sort();
        assert_eq!(
            fanned,
            vec![
                ("transformer_blocks.0.attn.to_k".to_string(), 1),
                ("transformer_blocks.0.attn.to_q".to_string(), 0),
                ("transformer_blocks.0.attn.to_v".to_string(), 2),
            ]
        );

        // FLUX.2 single-block `linear1` stays FUSED → maps to `to_qkv_mlp_proj` with NO slice.
        let l1 = targets
            .iter()
            .find(|tg| tg.target_path == "single_transformer_blocks.0.attn.to_qkv_mlp_proj")
            .expect("single linear1 target");
        assert!(
            l1.up_slice.is_none() && l1.down_slice.is_none(),
            "FLUX.2 single linear1 must not split (it is fused in the model)"
        );
    }

    /// sc-2743 gate at the FLUX.2 dispatch level: a BFL *fused* qkv kohya file resolves and installs
    /// the BYTE-IDENTICAL `to_q/to_k/to_v` adapters as the equivalent diffusers split-target file
    /// (the diffusers path is fork-verified, sc-2646 → transitively the BFL path matches the fork).
    #[test]
    fn bfl_fused_qkv_resolves_and_splits_like_diffusers() {
        use crate::adapters::apply_flux2_adapters;
        use mlx_gen::adapters::Adapter;
        use mlx_gen::runtime::{AdapterKind, AdapterSpec};
        let meta = None as Option<&std::collections::HashMap<String, String>>;

        // out=2 per head, 3 heads → fused up [6,1]; r=1 (not ÷3) → shared down [1,in=2]; alpha=4.
        let (inner, inp, r) = (2i32, 2i32, 1i32);
        let bq = [0.10f32, 0.11];
        let bk = [0.20f32, 0.21];
        let bv = [0.30f32, 0.31];
        let mut fused = bq.to_vec();
        fused.extend_from_slice(&bk);
        fused.extend_from_slice(&bv);
        let b_fused = Array::from_slice(&fused, &[3 * inner, r]);
        let b_q = Array::from_slice(&bq, &[inner, r]);
        let b_k = Array::from_slice(&bk, &[inner, r]);
        let b_v = Array::from_slice(&bv, &[inner, r]);
        let a = Array::from_slice(&[0.5f32, -0.5], &[r, inp]);
        let alpha = Array::from_slice(&[4.0f32], &[1]);

        let bpath = tmp("flux2_bfl_qkv.safetensors");
        Array::save_safetensors(
            vec![
                (
                    "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight",
                    &b_fused,
                ),
                (
                    "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight",
                    &a,
                ),
                ("lora_unet_double_blocks_0_img_attn_qkv.alpha", &alpha),
            ],
            meta,
            &bpath,
        )
        .unwrap();
        let mut tb = tiny_transformer();
        let rb = apply_flux2_adapters(
            &mut tb,
            &[AdapterSpec {
                path: bpath,
                scale: 0.8,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
        )
        .unwrap();
        assert_eq!(rb.applied, 3, "one fused qkv → three split targets");
        assert!(rb.unmatched_paths.is_empty());

        // Equivalent diffusers split-target file: per-head up, SHARED down, same alpha.
        let ppath = tmp("flux2_bfl_split_peft.safetensors");
        Array::save_safetensors(
            vec![
                (
                    "transformer.transformer_blocks.0.attn.to_q.lora_B.weight",
                    &b_q,
                ),
                (
                    "transformer.transformer_blocks.0.attn.to_q.lora_A.weight",
                    &a,
                ),
                ("transformer.transformer_blocks.0.attn.to_q.alpha", &alpha),
                (
                    "transformer.transformer_blocks.0.attn.to_k.lora_B.weight",
                    &b_k,
                ),
                (
                    "transformer.transformer_blocks.0.attn.to_k.lora_A.weight",
                    &a,
                ),
                ("transformer.transformer_blocks.0.attn.to_k.alpha", &alpha),
                (
                    "transformer.transformer_blocks.0.attn.to_v.lora_B.weight",
                    &b_v,
                ),
                (
                    "transformer.transformer_blocks.0.attn.to_v.lora_A.weight",
                    &a,
                ),
                ("transformer.transformer_blocks.0.attn.to_v.alpha", &alpha),
            ],
            meta,
            &ppath,
        )
        .unwrap();
        let mut tp = tiny_transformer();
        apply_flux2_adapters(
            &mut tp,
            &[AdapterSpec {
                path: ppath,
                scale: 0.8,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
        )
        .unwrap();

        for tgt in ["to_q", "to_k", "to_v"] {
            let segs = ["transformer_blocks", "0", "attn", tgt];
            let pull = |t: &mut Flux2Transformer| match AdaptableHost::adaptable_mut(t, &segs)
                .unwrap()
                .adapters()
            {
                [Adapter::Lora { a, b, .. }] => (a.clone(), b.clone()),
                _ => panic!("expected one LoRA at {tgt}"),
            };
            let (ba, bb) = pull(&mut tb);
            let (pa, pb) = pull(&mut tp);
            assert!(
                mlx_rs::ops::array_eq(&ba, &pa, false)
                    .unwrap()
                    .item::<bool>()
                    && mlx_rs::ops::array_eq(&bb, &pb, false)
                        .unwrap()
                        .item::<bool>(),
                "BFL split and diffusers split installed different adapters at {tgt}"
            );
        }
    }

    /// sc-8345 regression: a metadata LoKr in BFL/ComfyUI `diffusion_model.` fused naming — the exact
    /// convention from the reported `flux2_klein_9b_edit` failure ("112 adapter target(s) matched no
    /// module") — must now resolve on the real `Flux2Transformer`. The fused img-qkv reconstructs at the
    /// fused shape and row-slices into to_q/to_k/to_v; the `img_attn.proj` rename lands whole on to_out.
    /// `apply_flux2_adapters` goes through the strict no-silent-drop policy, so the `.unwrap()` itself
    /// proves zero unmatched (pre-fix it would Err); correctness of each slice is checked against an
    /// independent reconstruct-then-slice (LoKr can't be expressed as a per-split file the way LoRA can).
    #[test]
    fn bfl_named_lokr_resolves_on_real_transformer() {
        use crate::adapters::apply_flux2_adapters;
        use mlx_gen::adapters::{reconstruct_lokr_delta, Adapter};
        use mlx_gen::runtime::{AdapterKind, AdapterSpec};
        use mlx_rs::ops::indexing::TryIndexOp;
        use mlx_rs::Dtype;

        // tiny_transformer linears are all [1,1] → the fused img-qkv source is [3,1], proj is [1,1].
        let qkv_w1 = Array::from_slice(&[1.0f32, 0.5, -0.25], &[3, 1]);
        let qkv_w2 = Array::from_slice(&[2.0f32], &[1, 1]);
        let proj_w1 = Array::from_slice(&[0.3f32], &[1, 1]);
        let proj_w2 = Array::from_slice(&[1.5f32], &[1, 1]);
        let meta = std::collections::HashMap::from([
            ("networkType".to_string(), "lokr".to_string()),
            ("alpha".to_string(), "1.0".to_string()),
            ("rank".to_string(), "1".to_string()),
        ]);
        let path = tmp("flux2_bfl_lokr_diffmodel.safetensors");
        Array::save_safetensors(
            vec![
                (
                    "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w1",
                    &qkv_w1,
                ),
                (
                    "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w2",
                    &qkv_w2,
                ),
                (
                    "diffusion_model.double_blocks.0.img_attn.proj.lokr_w1",
                    &proj_w1,
                ),
                (
                    "diffusion_model.double_blocks.0.img_attn.proj.lokr_w2",
                    &proj_w2,
                ),
            ],
            Some(&meta),
            &path,
        )
        .unwrap();

        let mut t = tiny_transformer();
        let report = apply_flux2_adapters(
            &mut t,
            &[AdapterSpec {
                path,
                scale: 0.7,
                kind: AdapterKind::Lokr,
                pass_scales: None,
                moe_expert: None,
            }],
        )
        .unwrap();
        assert_eq!(report.applied, 4, "3 qkv splits + 1 proj rename");
        assert!(report.unmatched_paths.is_empty());

        let full = reconstruct_lokr_delta(
            1.0,
            1.0,
            &[3, 1],
            Some(&qkv_w1),
            None,
            None,
            Some(&qkv_w2),
            None,
            None,
            Dtype::Bfloat16,
        )
        .unwrap();
        for (idx, dst) in ["to_q", "to_k", "to_v"].iter().enumerate() {
            let segs = ["transformer_blocks", "0", "attn", dst];
            let lin = AdaptableHost::adaptable_mut(&mut t, &segs).unwrap();
            let Adapter::Lokr { delta, scale } = &lin.adapters()[0] else {
                panic!("expected a LoKr adapter at {dst}");
            };
            assert_eq!(*scale, 0.7);
            let start = idx as i32;
            let want = full.try_index((start..start + 1, ..)).unwrap();
            assert!(
                mlx_rs::ops::all_close(delta, &want, 1e-5, 1e-5, false)
                    .unwrap()
                    .item::<bool>(),
                "qkv split {dst} delta mismatch"
            );
        }
    }

    /// sc-2743: BFL plain renames resolve across all three prefix conventions — `base_model.model.`
    /// globals (`img_in`→`x_embedder`, `final_layer.linear`→`proj_out`), a `diffusion_model.` dotted
    /// block (`…img_attn.proj`→`to_out`), and a `base_model.model.` dotted single block
    /// (`…linear1`→`to_qkv_mlp_proj`).
    #[test]
    fn bfl_renames_and_prefixes_resolve() {
        use crate::adapters::apply_flux2_adapters;
        use mlx_gen::runtime::{AdapterKind, AdapterSpec};
        let meta = None as Option<&std::collections::HashMap<String, String>>;
        let s = Array::from_slice(&[0.01f32], &[1, 1]);
        let path = tmp("flux2_bfl_renames.safetensors");
        Array::save_safetensors(
            vec![
                ("base_model.model.img_in.lora_A.weight", &s),
                ("base_model.model.img_in.lora_B.weight", &s),
                ("base_model.model.final_layer.linear.lora_A.weight", &s),
                ("base_model.model.final_layer.linear.lora_B.weight", &s),
                (
                    "diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight",
                    &s,
                ),
                (
                    "diffusion_model.double_blocks.0.img_attn.proj.lora_B.weight",
                    &s,
                ),
                ("base_model.model.single_blocks.0.linear1.lora_A.weight", &s),
                ("base_model.model.single_blocks.0.linear1.lora_B.weight", &s),
            ],
            meta,
            &path,
        )
        .unwrap();
        let mut t = tiny_transformer();
        let rep = apply_flux2_adapters(
            &mut t,
            &[AdapterSpec {
                path,
                scale: 1.0,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
        )
        .unwrap();
        assert_eq!(
            rep.applied, 4,
            "all four BFL renames across the prefix conventions resolve"
        );
        assert!(rep.unmatched_paths.is_empty());
    }
}
