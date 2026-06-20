//! The Chroma DiT (`ChromaTransformer2DModel`).
//!
//! The FLUX MMDiT skeleton (19 dual + 38 single blocks, FluxPosEmbed RoPE, gelu-tanh FFN) with the
//! Chroma deltas:
//! - **sc-3836:** the distilled-guidance modulation generator — `ChromaCombinedTimestepTextProjEmbeddings`
//!   + `ChromaApproximator` → `pooled_temb [B, mod_index_len, inner]`.
//! - **sc-3837 (this slice):** the forward pass — `x_embedder`/`context_embedder`, RoPE over
//!   `cat(txt_ids, img_ids)`, the double/single blocks with **pruned adaLN** (modulation *sliced* from
//!   `pooled_temb`, no per-block linear), **MMDiT attention masking** (the 0/1 mask is added to the
//!   scores, the reference's literal behavior), QK-norm RMS eps **1e-6**, and the pruned `norm_out` +
//!   `proj_out`.
//!
//! The transformer runs f32 activations (parity is to the torch-`diffusers` reference; the cross-
//! backend f32 floor is ~1e-3, see the parity tests). The masked T5 encode that *builds* the
//! sequence mask is sc-3838; the generate path is sc-3839.

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::nn::{gated, gelu_tanh, silu};
/// Re-exported so the model's denoise loop can enable the shared `mx.compile` fusion of the DiT's
/// elementwise glue (adaLN modulate + gated residuals), matching FLUX.1/FLUX.2 (F-101/F-102).
/// [`CompileGlueGuard`] is the RAII form the production denoise binds so the toggle is restored on
/// drop (F-007) instead of leaking the process-global on.
pub use mlx_gen::nn::{set_compile_glue, CompileGlueGuard};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::fast::{layer_norm, rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, broadcast_to, concatenate_axis, cos, divide, exp, multiply, power, sin};
use mlx_rs::{Array, Dtype};

use crate::config::ChromaTransformerConfig;

/// Sinusoid / RoPE frequency base (diffusers `get_timestep_embedding` `max_period` and
/// `FluxPosEmbed(theta=10000)`).
const MAX_PERIOD: f64 = 10000.0;
const ROPE_THETA: f32 = 10000.0;
/// RMSNorm epsilon for the Approximator norms — torch `nn.RMSNorm(hidden)` with `eps=None` resolves
/// to `torch.finfo(float32).eps` (the f32 path).
const APPROX_RMS_EPS: f32 = 1.192_092_9e-7;
/// QK-norm RMS epsilon. Chroma's `FluxAttention(eps=1e-6)` — **NOT** FLUX's 1e-5.
const QK_RMS_EPS: f32 = 1e-6;
/// AdaLayerNorm LayerNorm epsilon (all pruned norms + `norm_out`, `elementwise_affine=False`).
const LN_EPS: f32 = 1e-6;

// ============================ leaf helpers ============================

/// `get_timestep_embedding(timesteps, dim, flip_sin_to_cos=True, downscale_freq_shift)` (diffusers),
/// in f32. `dim` even. `flip_sin_to_cos=True` ⇒ output order `[cos, sin]`.
fn timestep_embedding(timesteps: &Array, dim: usize, downscale_freq_shift: f64) -> Result<Array> {
    let half = (dim / 2) as i32;
    let factor = -MAX_PERIOD.ln() / (half as f64 - downscale_freq_shift);
    let exponent: Vec<f32> = (0..half).map(|i| (i as f64 * factor) as f32).collect();
    let freqs = exp(Array::from_slice(&exponent, &[1, half]))?; // [1, half]
    let t = timesteps.as_dtype(Dtype::Float32)?.reshape(&[-1, 1])?; // [N, 1]
    let emb = multiply(&t, &freqs)?; // [N, half]
    Ok(concatenate_axis(&[cos(&emb)?, sin(&emb)?], -1)?) // flip ⇒ [cos, sin]
}

/// A dense `nn.Linear` (`[out, in]` weight + bias) wrapping the core [`AdaptableLinear`] — so it can
/// be quantized (sc-3841) and carry LoRA/LoKr adapters (sc-3842). The forward runs f32 activations
/// over the bf16 (or quantized) weight; mlx promotes.
struct Lin(AdaptableLinear);

impl Lin {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self(AdaptableLinear::dense(
            w.require(&format!("{prefix}.weight"))?.clone(),
            Some(w.require(&format!("{prefix}.bias"))?.clone()),
        )))
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        self.0.forward(x)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.0.quantize(bits, None)
    }

    fn inner_mut(&mut self) -> &mut AdaptableLinear {
        &mut self.0
    }
}

/// adaLN affine `normed·(1+scale) + shift`. `scale`/`shift` are `[B,1,inner]` (broadcast over seq).
/// Delegates to the shared [`mlx_gen::nn::modulate`] with `one_matches_scale=false` (strong-f32 `1`,
/// matching the previous hand-rolled affine bit-for-bit) so it fuses under `compile_glue` (F-102).
fn modulate(normed: &Array, scale: &Array, shift: &Array) -> Result<Array> {
    mlx_gen::nn::modulate(normed, scale, shift, false)
}

/// The `j`-th modulation row of a `[B,K,inner]` slice, as `[B,1,inner]` (broadcastable over seq).
fn row(block: &Array, j: i32) -> Result<Array> {
    Ok(block.take_axis(Array::from_int(j), 1)?.expand_dims(1)?)
}

/// `len` contiguous modulation rows of `pooled_temb` from `start`, as `[B,len,inner]`.
fn rows(t: &Array, start: i32, len: i32) -> Result<Array> {
    let idx: Vec<i32> = (start..start + len).collect();
    Ok(t.take_axis(Array::from_slice(&idx, &[len]), 1)?)
}

/// `len` contiguous sequence positions of `[B,S,inner]` from `start`, as `[B,len,inner]`.
fn seq_slice(t: &Array, start: i32, len: i32) -> Result<Array> {
    let idx: Vec<i32> = (start..start + len).collect();
    Ok(t.take_axis(Array::from_slice(&idx, &[len]), 1)?)
}

// ============================ RoPE ============================

pub(crate) struct RopeTable {
    cos: Array,
    sin: Array,
}

/// FluxPosEmbed: per-axis sinusoid tables from position ids `[N,3]`, concatenated to `[N, head_dim/2]`.
/// Mirrors the (bit-exact) flux port: `omega = theta^-(2k/dim)`, `out = pos·omega`, then `cos`/`sin`.
fn build_rope(ids: &Array, axes: [usize; 3]) -> Result<RopeTable> {
    let ids = ids.as_dtype(Dtype::Float32)?;
    let n = ids.shape()[0];
    let mut coss = Vec::with_capacity(3);
    let mut sins = Vec::with_capacity(3);
    for (a, &dim) in axes.iter().enumerate() {
        let dim = dim as i32;
        let half = dim / 2;
        let pos = ids
            .take_axis(Array::from_int(a as i32), 1)?
            .reshape(&[n, 1])?; // [N,1]
        let scale: Vec<f32> = (0..half).map(|k| (2 * k) as f32 / dim as f32).collect();
        let omega = divide(
            mlx_gen::array::scalar(1.0),
            &power(
                mlx_gen::array::scalar(ROPE_THETA),
                Array::from_slice(&scale, &[1, half]),
            )?,
        )?; // [1, half]
        let out = multiply(&pos, &omega)?; // [N, half]
        coss.push(cos(&out)?);
        sins.push(sin(&out)?);
    }
    let cref: Vec<&Array> = coss.iter().collect();
    let sref: Vec<&Array> = sins.iter().collect();
    Ok(RopeTable {
        cos: concatenate_axis(&cref, 1)?,
        sin: concatenate_axis(&sref, 1)?,
    })
}

/// Apply RoPE to `x [B,H,S,hd]` (adjacent-pair / interleaved convention), in f32.
fn apply_rope_one(x: &Array, rope: &RopeTable) -> Result<Array> {
    let sh = x.shape();
    let (b, heads, seq, hd) = (sh[0], sh[1], sh[2], sh[3]);
    let half = hd / 2;
    let x5 = x
        .as_dtype(Dtype::Float32)?
        .reshape(&[b, heads, seq, half, 2])?;
    let p = mlx_rs::ops::split(&x5, 2, 4)?;
    let real = p[0].reshape(&[b, heads, seq, half])?;
    let imag = p[1].reshape(&[b, heads, seq, half])?;
    let c = rope.cos.reshape(&[1, 1, seq, half])?;
    let s = rope.sin.reshape(&[1, 1, seq, half])?;
    let out0 = mlx_rs::ops::subtract(&multiply(&real, &c)?, &multiply(&imag, &s)?)?;
    let out1 = add(&multiply(&imag, &c)?, &multiply(&real, &s)?)?;
    Ok(
        concatenate_axis(&[&out0.expand_dims(4)?, &out1.expand_dims(4)?], 4)?
            .reshape(&[b, heads, seq, hd])?,
    )
}

/// Project `x [B,S,inner]` to heads `[B,H,S,hd]`, optionally RMS-normed (QK-norm) over `hd` (f32).
fn proj_heads(x: &Array, lin: &Lin, heads: i32, hd: i32, norm: Option<&Array>) -> Result<Array> {
    let b = x.shape()[0];
    let s = x.shape()[1];
    let y = lin
        .forward(x)?
        .reshape(&[b, s, heads, hd])?
        .transpose_axes(&[0, 2, 1, 3])?;
    match norm {
        Some(w) => Ok(rms_norm(&y.as_dtype(Dtype::Float32)?, w, QK_RMS_EPS)?),
        None => Ok(y.as_dtype(Dtype::Float32)?),
    }
}

/// Scaled-dot-product attention over `[B,H,S,hd]` → `[B,S,inner]`. `mask` is the additive `[B,1,S,S]`
/// MMDiT mask (Chroma adds the 0/1 mask to the scores) or `None`.
fn sdpa(q: &Array, k: &Array, v: &Array, hd: i32, mask: Option<&Array>) -> Result<Array> {
    let b = q.shape()[0];
    let scale = (hd as f32).powf(-0.5);
    // `&Array` is taken as an *additive* mask (Chroma's 0/1 mask is added to the scores).
    let y = match mask {
        Some(m) => scaled_dot_product_attention(q, k, v, scale, m, None)?,
        None => scaled_dot_product_attention(q, k, v, scale, None, None)?,
    };
    Ok(y.transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[b, -1, q.shape()[1] * hd])?)
}

// ============================ embeddings + Approximator (sc-3836) ============================

/// `ChromaCombinedTimestepTextProjEmbeddings` — builds the Approximator input vector (parameter-free).
struct TimestepTextProj {
    num_channels: usize,
    mod_proj: Array,
}

impl TimestepTextProj {
    fn new(cfg: &ChromaTransformerConfig) -> Result<Self> {
        let num_channels = cfg.approximator_num_channels / 4;
        let n = cfg.mod_index_len();
        let idx: Vec<f32> = (0..n).map(|i| (i as f32) * 1000.0).collect();
        let idx = Array::from_slice(&idx, &[n as i32]);
        let mod_proj = timestep_embedding(&idx, 2 * num_channels, 0.0)?;
        Ok(Self {
            num_channels,
            mod_proj,
        })
    }

    /// `timestep` already scaled (`t*1000`), shape `[B]`. Returns `input_vec [B, mod_index_len, 4*nc]`.
    fn forward(&self, timestep: &Array) -> Result<Array> {
        let b = timestep.shape()[0];
        let n = self.mod_proj.shape()[0];
        let nc = 2 * self.num_channels as i32;
        let time = timestep_embedding(timestep, self.num_channels, 0.0)?;
        let zeros = Array::from_slice(&vec![0.0_f32; b as usize], &[b]);
        let guid = timestep_embedding(&zeros, self.num_channels, 0.0)?;
        let tg = concatenate_axis(&[time, guid], -1)?.reshape(&[b, 1, nc])?;
        let tg = broadcast_to(&tg, &[b, n, nc])?;
        let mp = broadcast_to(&self.mod_proj.reshape(&[1, n, nc])?, &[b, n, nc])?;
        Ok(concatenate_axis(&[tg, mp], -1)?)
    }
}

/// `ChromaApproximator` — `in_proj` then `n_layers` residual blocks
/// `x = x + linear_2(silu(linear_1(rms_norm(x))))`, then `out_proj`.
struct Approximator {
    in_proj: Lin,
    layers: Vec<(Lin, Lin)>,
    norms: Vec<Array>,
    out_proj: Lin,
}

impl Approximator {
    fn load(w: &Weights, cfg: &ChromaTransformerConfig) -> Result<Self> {
        let p = "distilled_guidance_layer";
        let mut layers = Vec::with_capacity(cfg.approximator_layers);
        let mut norms = Vec::with_capacity(cfg.approximator_layers);
        for i in 0..cfg.approximator_layers {
            layers.push((
                Lin::load(w, &format!("{p}.layers.{i}.linear_1"))?,
                Lin::load(w, &format!("{p}.layers.{i}.linear_2"))?,
            ));
            norms.push(w.require(&format!("{p}.norms.{i}.weight"))?.clone());
        }
        Ok(Self {
            in_proj: Lin::load(w, &format!("{p}.in_proj"))?,
            layers,
            norms,
            out_proj: Lin::load(w, &format!("{p}.out_proj"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = self.in_proj.forward(x)?;
        for ((lin1, lin2), norm) in self.layers.iter().zip(self.norms.iter()) {
            let n = rms_norm(&x, norm, APPROX_RMS_EPS)?;
            let h = lin2.forward(&silu(&lin1.forward(&n)?)?)?;
            x = add(&x, &h)?;
        }
        self.out_proj.forward(&x)
    }

    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        Some(match path {
            ["in_proj"] => self.in_proj.inner_mut(),
            ["out_proj"] => self.out_proj.inner_mut(),
            ["layers", n, "linear_1"] => {
                self.layers.get_mut(n.parse::<usize>().ok()?)?.0.inner_mut()
            }
            ["layers", n, "linear_2"] => {
                self.layers.get_mut(n.parse::<usize>().ok()?)?.1.inner_mut()
            }
            _ => return None,
        })
    }
}

// ============================ blocks ============================

struct DoubleAttn {
    to_q: Lin,
    to_k: Lin,
    to_v: Lin,
    to_out: Lin,
    add_q: Lin,
    add_k: Lin,
    add_v: Lin,
    to_add_out: Lin,
    norm_q: Array,
    norm_k: Array,
    norm_added_q: Array,
    norm_added_k: Array,
    heads: i32,
    head_dim: i32,
}

impl DoubleAttn {
    fn load(w: &Weights, p: &str, cfg: &ChromaTransformerConfig) -> Result<Self> {
        Ok(Self {
            to_q: Lin::load(w, &format!("{p}.to_q"))?,
            to_k: Lin::load(w, &format!("{p}.to_k"))?,
            to_v: Lin::load(w, &format!("{p}.to_v"))?,
            to_out: Lin::load(w, &format!("{p}.to_out.0"))?,
            add_q: Lin::load(w, &format!("{p}.add_q_proj"))?,
            add_k: Lin::load(w, &format!("{p}.add_k_proj"))?,
            add_v: Lin::load(w, &format!("{p}.add_v_proj"))?,
            to_add_out: Lin::load(w, &format!("{p}.to_add_out"))?,
            norm_q: w.require(&format!("{p}.norm_q.weight"))?.clone(),
            norm_k: w.require(&format!("{p}.norm_k.weight"))?.clone(),
            norm_added_q: w.require(&format!("{p}.norm_added_q.weight"))?.clone(),
            norm_added_k: w.require(&format!("{p}.norm_added_k.weight"))?.clone(),
            heads: cfg.num_attention_heads as i32,
            head_dim: cfg.attention_head_dim as i32,
        })
    }

    /// Joint attention. Returns `(image_attn [B,Si,inner], text_attn [B,St,inner])`. The concatenated
    /// sequence order is `[text, image]` (matches the mask order built in the forward).
    fn forward(
        &self,
        hidden: &Array,
        encoder: &Array,
        rope: &RopeTable,
        mask: Option<&Array>,
    ) -> Result<(Array, Array)> {
        let (h, hd) = (self.heads, self.head_dim);
        let q = proj_heads(hidden, &self.to_q, h, hd, Some(&self.norm_q))?;
        let k = proj_heads(hidden, &self.to_k, h, hd, Some(&self.norm_k))?;
        let v = proj_heads(hidden, &self.to_v, h, hd, None)?;
        let eq = proj_heads(encoder, &self.add_q, h, hd, Some(&self.norm_added_q))?;
        let ek = proj_heads(encoder, &self.add_k, h, hd, Some(&self.norm_added_k))?;
        let ev = proj_heads(encoder, &self.add_v, h, hd, None)?;
        let q = concatenate_axis(&[&eq, &q], 2)?;
        let k = concatenate_axis(&[&ek, &k], 2)?;
        let v = concatenate_axis(&[&ev, &v], 2)?;
        let q = apply_rope_one(&q, rope)?;
        let k = apply_rope_one(&k, rope)?;
        let out = sdpa(&q, &k, &v, hd, mask)?; // [B, S, inner]
        let st = encoder.shape()[1];
        let txt = seq_slice(&out, 0, st)?;
        let img = seq_slice(&out, st, hidden.shape()[1])?;
        Ok((self.to_out.forward(&img)?, self.to_add_out.forward(&txt)?))
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        for l in [
            &mut self.to_q,
            &mut self.to_k,
            &mut self.to_v,
            &mut self.to_out,
            &mut self.add_q,
            &mut self.add_k,
            &mut self.add_v,
            &mut self.to_add_out,
        ] {
            l.quantize(bits)?;
        }
        Ok(())
    }

    /// Resolve a diffusers adapter sub-path (within `…attn.`) to its linear (sc-3842).
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        Some(match path {
            ["to_q"] => self.to_q.inner_mut(),
            ["to_k"] => self.to_k.inner_mut(),
            ["to_v"] => self.to_v.inner_mut(),
            ["to_out", "0"] => self.to_out.inner_mut(),
            ["add_q_proj"] => self.add_q.inner_mut(),
            ["add_k_proj"] => self.add_k.inner_mut(),
            ["add_v_proj"] => self.add_v.inner_mut(),
            ["to_add_out"] => self.to_add_out.inner_mut(),
            _ => return None,
        })
    }
}

struct FeedForward {
    lin1: Lin,
    lin2: Lin,
}

impl FeedForward {
    fn load(w: &Weights, p: &str) -> Result<Self> {
        Ok(Self {
            lin1: Lin::load(w, &format!("{p}.net.0.proj"))?,
            lin2: Lin::load(w, &format!("{p}.net.2"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        self.lin2.forward(&gelu_tanh(&self.lin1.forward(x)?)?)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.lin1.quantize(bits)?;
        self.lin2.quantize(bits)
    }
}

struct DoubleBlock {
    attn: DoubleAttn,
    ff: FeedForward,
    ff_context: FeedForward,
}

impl DoubleBlock {
    fn load(w: &Weights, i: usize, cfg: &ChromaTransformerConfig) -> Result<Self> {
        let p = format!("transformer_blocks.{i}");
        Ok(Self {
            attn: DoubleAttn::load(w, &format!("{p}.attn"), cfg)?,
            ff: FeedForward::load(w, &format!("{p}.ff"))?,
            ff_context: FeedForward::load(w, &format!("{p}.ff_context"))?,
        })
    }

    /// `temb` is the 12-row modulation slice `[B,12,inner]` (`[:6]` image, `[6:]` text). Each stream's
    /// rows are `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)`.
    fn forward(
        &self,
        hidden: &Array,
        encoder: &Array,
        temb: &Array,
        rope: &RopeTable,
        mask: Option<&Array>,
    ) -> Result<(Array, Array)> {
        let norm_hidden = modulate(
            &layer_norm(hidden, None, None, LN_EPS)?,
            &row(temb, 1)?,
            &row(temb, 0)?,
        )?;
        let norm_encoder = modulate(
            &layer_norm(encoder, None, None, LN_EPS)?,
            &row(temb, 7)?,
            &row(temb, 6)?,
        )?;

        let (attn_img, attn_txt) = self.attn.forward(&norm_hidden, &norm_encoder, rope, mask)?;

        // image stream.
        let hidden = gated(hidden, &row(temb, 2)?, &attn_img)?;
        let nh = modulate(
            &layer_norm(&hidden, None, None, LN_EPS)?,
            &row(temb, 4)?,
            &row(temb, 3)?,
        )?;
        let hidden = gated(&hidden, &row(temb, 5)?, &self.ff.forward(&nh)?)?;

        // text stream.
        let encoder = gated(encoder, &row(temb, 8)?, &attn_txt)?;
        let ne = modulate(
            &layer_norm(&encoder, None, None, LN_EPS)?,
            &row(temb, 10)?,
            &row(temb, 9)?,
        )?;
        let encoder = gated(&encoder, &row(temb, 11)?, &self.ff_context.forward(&ne)?)?;

        Ok((encoder, hidden))
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attn.quantize(bits)?;
        self.ff.quantize(bits)?;
        self.ff_context.quantize(bits)
    }

    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        Some(match path {
            ["attn", rest @ ..] => return self.attn.adaptable_mut(rest),
            ["ff", "net", "0", "proj"] => self.ff.lin1.inner_mut(),
            ["ff", "net", "2"] => self.ff.lin2.inner_mut(),
            ["ff_context", "net", "0", "proj"] => self.ff_context.lin1.inner_mut(),
            ["ff_context", "net", "2"] => self.ff_context.lin2.inner_mut(),
            _ => return None,
        })
    }
}

struct SingleAttn {
    to_q: Lin,
    to_k: Lin,
    to_v: Lin,
    norm_q: Array,
    norm_k: Array,
    heads: i32,
    head_dim: i32,
}

impl SingleAttn {
    fn load(w: &Weights, p: &str, cfg: &ChromaTransformerConfig) -> Result<Self> {
        Ok(Self {
            to_q: Lin::load(w, &format!("{p}.to_q"))?,
            to_k: Lin::load(w, &format!("{p}.to_k"))?,
            to_v: Lin::load(w, &format!("{p}.to_v"))?,
            norm_q: w.require(&format!("{p}.norm_q.weight"))?.clone(),
            norm_k: w.require(&format!("{p}.norm_k.weight"))?.clone(),
            heads: cfg.num_attention_heads as i32,
            head_dim: cfg.attention_head_dim as i32,
        })
    }

    fn forward(&self, x: &Array, rope: &RopeTable, mask: Option<&Array>) -> Result<Array> {
        let (h, hd) = (self.heads, self.head_dim);
        let q = apply_rope_one(&proj_heads(x, &self.to_q, h, hd, Some(&self.norm_q))?, rope)?;
        let k = apply_rope_one(&proj_heads(x, &self.to_k, h, hd, Some(&self.norm_k))?, rope)?;
        let v = proj_heads(x, &self.to_v, h, hd, None)?;
        sdpa(&q, &k, &v, hd, mask)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.to_q.quantize(bits)?;
        self.to_k.quantize(bits)?;
        self.to_v.quantize(bits)
    }

    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        Some(match path {
            ["to_q"] => self.to_q.inner_mut(),
            ["to_k"] => self.to_k.inner_mut(),
            ["to_v"] => self.to_v.inner_mut(),
            _ => return None,
        })
    }
}

struct SingleBlock {
    attn: SingleAttn,
    proj_mlp: Lin,
    proj_out: Lin,
}

impl SingleBlock {
    fn load(w: &Weights, i: usize, cfg: &ChromaTransformerConfig) -> Result<Self> {
        let p = format!("single_transformer_blocks.{i}");
        Ok(Self {
            attn: SingleAttn::load(w, &format!("{p}.attn"), cfg)?,
            proj_mlp: Lin::load(w, &format!("{p}.proj_mlp"))?,
            proj_out: Lin::load(w, &format!("{p}.proj_out"))?,
        })
    }

    /// `temb` is the 3-row modulation slice `[B,3,inner]` (shift, scale, gate). `hidden` is the joint
    /// `[text|image]` stream.
    fn forward(
        &self,
        hidden: &Array,
        temb: &Array,
        rope: &RopeTable,
        mask: Option<&Array>,
    ) -> Result<Array> {
        let norm_hidden = modulate(
            &layer_norm(hidden, None, None, LN_EPS)?,
            &row(temb, 1)?,
            &row(temb, 0)?,
        )?;
        let mlp = gelu_tanh(&self.proj_mlp.forward(&norm_hidden)?)?;
        let attn = self.attn.forward(&norm_hidden, rope, mask)?;
        let proj = self
            .proj_out
            .forward(&concatenate_axis(&[&attn, &mlp], 2)?)?;
        gated(hidden, &row(temb, 2)?, &proj)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attn.quantize(bits)?;
        self.proj_mlp.quantize(bits)?;
        self.proj_out.quantize(bits)
    }

    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        Some(match path {
            ["attn", rest @ ..] => return self.attn.adaptable_mut(rest),
            ["proj_mlp"] => self.proj_mlp.inner_mut(),
            ["proj_out"] => self.proj_out.inner_mut(),
            _ => return None,
        })
    }
}

// ============================ the transformer ============================

pub struct ChromaTransformer {
    pub cfg: ChromaTransformerConfig,
    x_embedder: Lin,
    context_embedder: Lin,
    time_text_embed: TimestepTextProj,
    approximator: Approximator,
    double_blocks: Vec<DoubleBlock>,
    single_blocks: Vec<SingleBlock>,
    proj_out: Lin,
}

impl ChromaTransformer {
    /// Load from a diffusers `transformer/` weight map. Validates the Chroma key surface + the
    /// pruned-adaLN invariant, then materializes the typed modules.
    pub fn from_weights(w: Weights, cfg: ChromaTransformerConfig) -> Result<Self> {
        // Pruned-adaLN invariant: Chroma blocks have NO `.norm*.linear` weights.
        if let Some(k) = w
            .keys()
            .find(|k| k.contains(".norm1.linear") || k.contains(".norm.linear"))
        {
            return Err(Error::Msg(format!(
                "chroma transformer: unexpected per-block modulation linear {k:?} — Chroma uses \
                 pruned adaLN (modulation comes from distilled_guidance_layer)"
            )));
        }

        let n_double = (0..)
            .take_while(|i| {
                w.get(&format!("transformer_blocks.{i}.attn.to_q.weight"))
                    .is_some()
            })
            .count();
        let n_single = (0..)
            .take_while(|i| {
                w.get(&format!("single_transformer_blocks.{i}.proj_out.weight"))
                    .is_some()
            })
            .count();
        if n_double != cfg.num_layers || n_single != cfg.num_single_layers {
            return Err(Error::Msg(format!(
                "chroma transformer: block counts {n_double} double / {n_single} single != config \
                 {} / {}",
                cfg.num_layers, cfg.num_single_layers
            )));
        }

        let double_blocks = (0..cfg.num_layers)
            .map(|i| DoubleBlock::load(&w, i, &cfg))
            .collect::<Result<Vec<_>>>()?;
        let single_blocks = (0..cfg.num_single_layers)
            .map(|i| SingleBlock::load(&w, i, &cfg))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            x_embedder: Lin::load(&w, "x_embedder")?,
            context_embedder: Lin::load(&w, "context_embedder")?,
            time_text_embed: TimestepTextProj::new(&cfg)?,
            approximator: Approximator::load(&w, &cfg)?,
            double_blocks,
            single_blocks,
            proj_out: Lin::load(&w, "proj_out")?,
            cfg,
        })
    }

    /// Quantize the matmul-heavy block linears (double/single attention + FFN) to Q4/Q8 (sc-3841).
    /// The small/sensitive modules — `x_embedder`/`context_embedder`/`proj_out` and the
    /// distilled-guidance Approximator (which drives all modulation) — stay dense, mirroring the
    /// "quantize the big GEMMs" convention. T5/VAE are quantized separately by the loader (if at all).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for b in &mut self.double_blocks {
            b.quantize(bits)?;
        }
        for b in &mut self.single_blocks {
            b.quantize(bits)?;
        }
        Ok(())
    }

    /// `pooled_temb [B, mod_index_len, inner]` for a **raw** (unscaled) timestep `[B]`.
    pub fn pooled_temb(&self, timestep: &Array) -> Result<Array> {
        let scaled = multiply(
            &timestep.as_dtype(Dtype::Float32)?,
            mlx_gen::array::scalar(1000.0),
        )?;
        self.approximator
            .forward(&self.time_text_embed.forward(&scaled)?)
    }

    /// The Chroma DiT forward.
    ///
    /// - `hidden [B, Si, in_channels]` — packed image latent tokens.
    /// - `encoder [B, St, joint_attention_dim]` — T5 prompt embeddings.
    /// - `timestep [B]` — raw denoise timestep (scaled `*1000` internally).
    /// - `img_ids [Si,3]` / `txt_ids [St,3]` — RoPE position ids.
    /// - `attention_mask [B, St+Si]` (0/1) or `None` — the **full-sequence** MMDiT mask in `[text,
    ///   image]` order. The 0/1 mask is added to the attention scores (the reference's behavior). The
    ///   mask that *builds* this from the T5 padding is sc-3838.
    ///
    /// Returns the predicted velocity `[B, Si, out_channels]`.
    ///
    /// Convenience wrapper that builds the step-invariant tensors (`pooled_temb`, the RoPE table, the
    /// `[B,1,S,S]` mask) and calls [`Self::forward_prepared`]. The denoise loop prefers the prepared
    /// form so those tensors are computed once per step / per branch rather than per call (F-102).
    pub fn forward(
        &self,
        hidden: &Array,
        encoder: &Array,
        timestep: &Array,
        img_ids: &Array,
        txt_ids: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Array> {
        let pooled = self.pooled_temb(timestep)?;
        let rope = self.build_rope_table(txt_ids, img_ids)?;
        let mask2d = Self::attention_mask2d(attention_mask)?;
        self.forward_prepared(hidden, encoder, &pooled, &rope, mask2d.as_ref())
    }

    /// The RoPE table over `cat(txt_ids, img_ids)` — depends only on the token positions, so the
    /// denoise loop builds it once per branch instead of every step (F-102).
    pub(crate) fn build_rope_table(&self, txt_ids: &Array, img_ids: &Array) -> Result<RopeTable> {
        let ids = concatenate_axis(&[txt_ids, img_ids], 0)?;
        build_rope(&ids, self.cfg.axes_dims_rope)
    }

    /// `[B,S]` 0/1 mask → additive `[B,1,S,S] = m[:,None,None,:]·m[:,None,:,None]`. Depends only on the
    /// per-request padding, so the denoise loop builds it once per branch (F-102).
    pub(crate) fn attention_mask2d(attention_mask: Option<&Array>) -> Result<Option<Array>> {
        match attention_mask {
            Some(m) => {
                let m = m.as_dtype(Dtype::Float32)?;
                let b = m.shape()[0];
                let s = m.shape()[1];
                let a = m.reshape(&[b, 1, 1, s])?;
                let bt = m.reshape(&[b, 1, s, 1])?;
                Ok(Some(multiply(&a, &bt)?))
            }
            None => Ok(None),
        }
    }

    /// Run the MMDiT given the pre-built step-invariant tensors: `pooled` (the Approximator modulation
    /// table — shared by both CFG branches at a step), `rope`, and the additive `mask2d`. `hidden`
    /// (latents) and `encoder` (text) are per-branch. Bit-identical to [`Self::forward`].
    pub(crate) fn forward_prepared(
        &self,
        hidden: &Array,
        encoder: &Array,
        pooled: &Array,
        rope: &RopeTable,
        mask_ref: Option<&Array>,
    ) -> Result<Array> {
        let hidden = self.x_embedder.forward(hidden)?;
        let encoder = self.context_embedder.forward(encoder)?;

        let st = encoder.shape()[1];
        let n_single = self.cfg.num_single_layers as i32;
        let img_offset = 3 * n_single;
        let txt_offset = img_offset + 6 * self.cfg.num_layers as i32;

        let mut hidden = hidden;
        let mut encoder = encoder;
        for (i, block) in self.double_blocks.iter().enumerate() {
            let i = i as i32;
            let img = rows(pooled, img_offset + 6 * i, 6)?;
            let txt = rows(pooled, txt_offset + 6 * i, 6)?;
            let temb = concatenate_axis(&[&img, &txt], 1)?; // [B,12,inner]
            let (e, h) = block.forward(&hidden, &encoder, &temb, rope, mask_ref)?;
            encoder = e;
            hidden = h;
        }

        let mut joint = concatenate_axis(&[&encoder, &hidden], 1)?; // [B, S, inner]
        for (i, block) in self.single_blocks.iter().enumerate() {
            let temb = rows(pooled, 3 * i as i32, 3)?;
            joint = block.forward(&joint, &temb, rope, mask_ref)?;
        }

        // Drop the text tokens; pruned `norm_out` (shift, scale = pooled[-2:]); proj_out.
        let hidden = seq_slice(&joint, st, joint.shape()[1] - st)?;
        let n = self.cfg.mod_index_len() as i32;
        let no = rows(pooled, n - 2, 2)?;
        let hidden = modulate(
            &layer_norm(&hidden, None, None, LN_EPS)?,
            &row(&no, 1)?,
            &row(&no, 0)?,
        )?;
        self.proj_out.forward(&hidden)
    }

    /// Test hook: the Approximator input vector for a raw timestep `[B]` (pure elementwise — isolates
    /// the embedding build from the matmul floor).
    #[doc(hidden)]
    pub fn input_vec_for_tests(&self, timestep: &Array) -> Result<Array> {
        let scaled = multiply(
            &timestep.as_dtype(Dtype::Float32)?,
            mlx_gen::array::scalar(1000.0),
        )?;
        self.time_text_embed.forward(&scaled)
    }
}

impl AdaptableHost for ChromaTransformer {
    /// Resolve a trained-file (diffusers/peft) dotted adapter path to its [`AdaptableLinear`].
    /// Covers the double/single block attention + FFN linears, the global embedders/`proj_out`, and
    /// the distilled-guidance Approximator (some community Chroma LoRAs train it).
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["transformer_blocks", n, rest @ ..] => self
                .double_blocks
                .get_mut(n.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            ["single_transformer_blocks", n, rest @ ..] => self
                .single_blocks
                .get_mut(n.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            ["x_embedder"] => Some(self.x_embedder.inner_mut()),
            ["context_embedder"] => Some(self.context_embedder.inner_mut()),
            ["proj_out"] => Some(self.proj_out.inner_mut()),
            ["distilled_guidance_layer", rest @ ..] => self.approximator.adaptable_mut(rest),
            _ => None,
        }
    }

    /// kohya `lora_unet_`-reachable targets: the block-indexed attention + FFN linears in trained-file
    /// naming. Globals (`x_embedder`/`context_embedder`/`proj_out`/`distilled_guidance_layer`) are
    /// excluded — they stay reachable via the dotted peft form (every path here must resolve via
    /// [`adaptable_mut`](Self::adaptable_mut); guarded by `tests/adapter_routing.rs`).
    fn adaptable_paths(&self) -> Vec<String> {
        const DOUBLE: [&str; 12] = [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.add_q_proj",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ];
        const SINGLE: [&str; 5] = [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "proj_mlp",
            "proj_out",
        ];
        let mut out = Vec::new();
        for i in 0..self.double_blocks.len() {
            for leaf in DOUBLE {
                out.push(format!("transformer_blocks.{i}.{leaf}"));
            }
        }
        for i in 0..self.single_blocks.len() {
            for leaf in SINGLE {
                out.push(format!("single_transformer_blocks.{i}.{leaf}"));
            }
        }
        out
    }
}
