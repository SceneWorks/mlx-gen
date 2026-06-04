//! S3 — the LTX-2.3 **DiT** (video stack): the preprocessor (patchify + adaLN-single) → 48 ×
//! `BasicAVTransformerBlock` (video-only path) → output projection → velocity. Port of the
//! `mlx_video` reference `models/ltx/{transformer,attention,adaln,feed_forward,ltx}.py`.
//!
//! Per-block math (S3a): **gated attention** (`to_gate_logits → 2·sigmoid`, zero-init identity), q/k
//! **RMSNorm** over the full inner_dim (pre-head, learned), **SPLIT 3-D RoPE** on q/k (reusing the S0
//! [`crate::rope`]), SDPA, `to_out`; **adaLN-single** with the 9-row `scale_shift_table` (gated 2.3
//! family: MSA rows 0..3, FF rows 3..6, text-cross-attn rows 6..9); **prompt adaLN** modulating the
//! text context; **FeedForward** = `proj_in → gelu(tanh) → proj_out`.
//!
//! Full forward (S3b): patchify_proj → adaLN-single (timestep → 9·dim) + prompt-adaLN (→ 2·dim) →
//! caption projection (Identity for 2.3) → SPLIT RoPE from the position grid → 48 blocks → output
//! (`LayerNorm` affine-false + final 2-row `scale_shift_table` modulated by the embedded timestep) →
//! `proj_out → 128` velocity. `denoised = latent − σ·velocity`.
//!
//! **Quant.** The shipped transformer stores the attn/ff Linears selectively quantized (U32 +
//! `scales` + `biases`) — there is no dense bf16 checkpoint. The **bits/group ride on the checkpoint's
//! `split_model.json`** ([`crate::config::SplitModel`]): `base_q8`/`eros`-style at 8 bits, `base_q4`
//! at 4 bits, group 64 — read into [`Precision`], never hardcoded (sc-2686). The per-Linear predicate
//! (quantize iff the weights carry `.scales`) mirrors `generate_av.py`'s `_should_quantize`.
//!
//! [`Precision::quant_f32`] is the production quality target: **f32 activations × `quantized_matmul`**
//! (a single block is bit-exact to the reference at matched mlx 0.31.2). [`Precision::quant_bf16`]
//! mirrors the reference's own bf16 compute (the production-speed path). [`Precision::dense_f32`]
//! additionally dequantizes the weights to dense f32 — the S3a block-math gate.

use mlx_rs::fast::{layer_norm, rms_norm as fast_rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{
    add, concatenate_axis, dequantize, divide, multiply, quantized_matmul, sigmoid, subtract,
};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::{gelu_tanh, linear};
use mlx_gen::weights::{to_dtype, Weights};
use mlx_gen::Result;

use crate::config::LtxConfig;
use crate::rope::{apply_split_rotary_emb, precompute_split_freqs_cis};

/// adaLN-single sinusoidal timestep projection width (PixArt `Timesteps`).
const TIME_PROJ_DIM: i32 = 256;

/// How to run the (selectively quantized) DiT: the activation/compute dtype, whether quantized
/// weights stay packed (`quantized_matmul`) or are dequantized to dense, and the **checkpoint's**
/// quant geometry (`bits`/`group` from `split_model.json` — so Q4 and Q8 both load without a code
/// change; sc-2686). Construct via [`quant_f32`](Self::quant_f32) / [`quant_bf16`](Self::quant_bf16)
/// / [`dense_f32`](Self::dense_f32).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Precision {
    mode: Mode,
    bits: i32,
    group: i32,
}

/// The compute mode (independent of the quant bit-width, which rides alongside in [`Precision`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// f32 activations, quantized weights **dequantized** to dense f32 — the S3a block-math gate.
    DenseF32,
    /// **f32 activations × `quantized_matmul`** — the production path / quality target. The full
    /// 48-layer velocity is bit-exact to the reference (mlx 0.31.2), required because the distilled
    /// stage-1 sampler is chaos-sensitive (sc-2842).
    QuantF32,
    /// bf16 activations × `quantized_matmul` — the reference's own compute dtype (production speed).
    QuantBf16,
}

impl Precision {
    /// f32 activations, quantized weights dequantized to dense f32 (the block-math gate).
    pub fn dense_f32(bits: i32, group: i32) -> Self {
        Self {
            mode: Mode::DenseF32,
            bits,
            group,
        }
    }

    /// f32 activations × `quantized_matmul` (the production quality target).
    pub fn quant_f32(bits: i32, group: i32) -> Self {
        Self {
            mode: Mode::QuantF32,
            bits,
            group,
        }
    }

    /// bf16 activations × `quantized_matmul` (the reference's native production-speed path).
    pub fn quant_bf16(bits: i32, group: i32) -> Self {
        Self {
            mode: Mode::QuantBf16,
            bits,
            group,
        }
    }

    fn dtype(self) -> Dtype {
        match self.mode {
            Mode::DenseF32 | Mode::QuantF32 => Dtype::Float32,
            Mode::QuantBf16 => Dtype::Bfloat16,
        }
    }

    /// Whether quantized weights are kept packed (`quantized_matmul`) vs dequantized to dense f32.
    fn keep_quant(self) -> bool {
        matches!(self.mode, Mode::QuantF32 | Mode::QuantBf16)
    }
}

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// Load a non-Linear param (norm weight, scale-shift table) cast to the compute dtype.
fn param(w: &Weights, key: &str, prec: Precision) -> Result<Array> {
    to_dtype(w.require(key)?, prec.dtype())
}

/// `x · (1 + scale) + shift` (adaLN modulation), broadcasting `scale`/`shift` `(B, S', dim)` over the
/// token axis.
fn modulate(x: &Array, scale: &Array, shift: &Array) -> Result<Array> {
    Ok(add(
        &multiply(x, &add(scale, scalar(1.0).as_dtype(scale.dtype())?)?)?,
        shift,
    )?)
}

/// A Linear — dense or Q8-quantized, selected by [`Precision`] at load.
enum Linear {
    Dense {
        w: Array, // [out, in]
        b: Array, // [out]
    },
    Quant {
        q: Array,      // [out, in_packed] U32
        scales: Array, // [out, in/group]
        biases: Array,
        b: Array,
        group: i32,
        bits: i32,
    },
}

impl Linear {
    fn load(w: &Weights, prefix: &str, prec: Precision) -> Result<Self> {
        let dt = prec.dtype();
        let b = to_dtype(w.require(&format!("{prefix}.bias"))?, dt)?;
        match w.get(&format!("{prefix}.scales")) {
            Some(scales) => {
                let q = w.require(&format!("{prefix}.weight"))?;
                let biases = w.require(&format!("{prefix}.biases"))?;
                if prec.keep_quant() {
                    // Keep the weights packed; `quantized_matmul` dequantizes on the fly with fp32
                    // accumulation and is correct for f32 *or* bf16 activations (the Z-Image/Qwen Q8
                    // path) at either bit-width. Scales / biases are cast to the compute dtype so the
                    // on-the-fly dequant matches the reference's (f32 for the quant_f32 path — a
                    // lossless upcast of the bf16 file scales). bits/group come from the checkpoint's
                    // split_model.json via `prec`, so Q4 and Q8 both load unchanged.
                    Ok(Linear::Quant {
                        q: q.clone(),
                        scales: to_dtype(scales, dt)?,
                        biases: to_dtype(biases, dt)?,
                        b,
                        group: prec.group,
                        bits: prec.bits,
                    })
                } else {
                    // Dequantize to dense f32 (bit-identical to the reference's mx.dequantize).
                    let dense =
                        dequantize(q, scales, Some(biases), Some(prec.group), Some(prec.bits))?;
                    Ok(Linear::Dense {
                        w: to_dtype(&dense, Dtype::Float32)?,
                        b,
                    })
                }
            }
            None => Ok(Linear::Dense {
                w: to_dtype(w.require(&format!("{prefix}.weight"))?, dt)?,
                b,
            }),
        }
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Linear::Dense { w, b } => linear(x, w, b),
            Linear::Quant {
                q,
                scales,
                biases,
                b,
                group,
                bits,
            } => Ok(add(
                &quantized_matmul(x, q, scales, biases, true, *group, *bits)?,
                b,
            )?),
        }
    }
}

/// `mx.fast.rms_norm(x, ones, eps)` — the block's weightless pre-norm (feature RMS over the last axis).
fn rms_norm_noweight(x: &Array, eps: f32) -> Result<Array> {
    let dim = *x.shape().last().unwrap();
    let ones = Array::ones::<f32>(&[dim])?.as_dtype(x.dtype())?;
    Ok(fast_rms_norm(x, &ones, eps)?)
}

/// Multi-head attention with q/k RMSNorm, optional SPLIT RoPE, optional per-head gating. Self-attn
/// when `context` is `None`; cross-attn otherwise. RoPE `(cos, sin)` applies to q **and** k (self-attn
/// only; cross-attn passes `pe = None`).
struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    q_norm: Array,
    k_norm: Array,
    to_out: Linear,
    gate: Option<Linear>,
    heads: i32,
    dim_head: i32,
    eps: f32,
}

impl Attention {
    /// Load an attention with explicit `heads`/`dim_head` (the *inner* dims = `heads·dim_head`,
    /// which the q/k/v project to; cross-modal attns project a different query/context dim into the
    /// same inner). `eps` is the q/k-RMSNorm epsilon.
    fn load(
        w: &Weights,
        prefix: &str,
        heads: i32,
        dim_head: i32,
        eps: f32,
        prec: Precision,
    ) -> Result<Self> {
        let gate = if w.get(&format!("{prefix}.to_gate_logits.weight")).is_some() {
            Some(Linear::load(w, &format!("{prefix}.to_gate_logits"), prec)?)
        } else {
            None
        };
        Ok(Self {
            to_q: Linear::load(w, &format!("{prefix}.to_q"), prec)?,
            to_k: Linear::load(w, &format!("{prefix}.to_k"), prec)?,
            to_v: Linear::load(w, &format!("{prefix}.to_v"), prec)?,
            q_norm: param(w, &format!("{prefix}.q_norm.weight"), prec)?,
            k_norm: param(w, &format!("{prefix}.k_norm.weight"), prec)?,
            to_out: Linear::load(w, &format!("{prefix}.to_out"), prec)?,
            gate,
            heads,
            dim_head,
            eps,
        })
    }

    /// `(B, S, inner)` → `(B, H, S, head_dim)`.
    fn to_heads(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        Ok(x.reshape(&[b, s, self.heads, self.dim_head])?
            .transpose_axes(&[0, 2, 1, 3])?)
    }

    /// `pe` rotates the query (and the key if `k_pe` is `None`); `k_pe` rotates the key separately
    /// (cross-modal: video-positioned q, audio-positioned k, or vice-versa). `pe == None` ⇒ no RoPE
    /// on either (text cross-attention). Mirrors `attention.py::Attention.__call__`.
    fn forward(
        &self,
        x: &Array,
        context: Option<&Array>,
        mask: Option<&Array>,
        pe: Option<(&Array, &Array)>,
        k_pe: Option<(&Array, &Array)>,
    ) -> Result<Array> {
        let ctx = context.unwrap_or(x);
        let q = fast_rms_norm(&self.to_q.forward(x)?, &self.q_norm, self.eps)?;
        let k = fast_rms_norm(&self.to_k.forward(ctx)?, &self.k_norm, self.eps)?;
        let v = self.to_v.forward(ctx)?;

        let mut qh = self.to_heads(&q)?;
        let mut kh = self.to_heads(&k)?;
        let vh = self.to_heads(&v)?;
        if let Some((cos, sin)) = pe {
            qh = apply_split_rotary_emb(&qh, cos, sin)?;
            let (kc, ks) = k_pe.unwrap_or((cos, sin));
            kh = apply_split_rotary_emb(&kh, kc, ks)?;
        }

        // Match the reference's Python `1.0 / math.sqrt(dim_head)` (f64 → f32), not `d^-0.5` in f32.
        let scale = (1.0f64 / (self.dim_head as f64).sqrt()) as f32;
        let out = match mask {
            Some(m) => scaled_dot_product_attention(&qh, &kh, &vh, scale, m, None)?,
            None => scaled_dot_product_attention(&qh, &kh, &vh, scale, None, None)?,
        };
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        let inner = self.heads * self.dim_head;
        let mut out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, inner])?;

        if let Some(gate) = &self.gate {
            // Per-head gate: 2·sigmoid(logits) (zero-init → identity), broadcast over head_dim.
            let logits = gate.forward(x)?;
            let gates = multiply(&sigmoid(&logits)?, scalar(2.0).as_dtype(logits.dtype())?)?;
            let gates = gates.reshape(&[b, s, self.heads, 1])?;
            out = multiply(&out.reshape(&[b, s, self.heads, self.dim_head])?, &gates)?
                .reshape(&[b, s, inner])?;
        }
        self.to_out.forward(&out)
    }
}

/// `proj_in → gelu(tanh) → proj_out`.
struct FeedForward {
    proj_in: Linear,
    proj_out: Linear,
}

impl FeedForward {
    fn load(w: &Weights, prefix: &str, prec: Precision) -> Result<Self> {
        Ok(Self {
            proj_in: Linear::load(w, &format!("{prefix}.proj_in"), prec)?,
            proj_out: Linear::load(w, &format!("{prefix}.proj_out"), prec)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        self.proj_out
            .forward(&gelu_tanh(&self.proj_in.forward(x)?)?)
    }
}

/// `table[row] + timestep_proj[row]` for `row ∈ [lo, hi)`. `table` is `(num_ada, dim)`; `timestep` is
/// `(B, S', num_ada·dim)`. Returns the `hi−lo` modulation tensors, each `(B, S', dim)`.
fn ada_values(table: &Array, timestep: &Array, lo: i32, hi: i32) -> Result<Vec<Array>> {
    let num_ada = table.shape()[0];
    let dim = table.shape()[1];
    let ts = timestep.shape();
    let (b, s) = (ts[0], ts[1]);
    let ts4 = timestep.reshape(&[b, s, num_ada, dim])?;
    let mut out = Vec::with_capacity((hi - lo) as usize);
    for row in lo..hi {
        let trow = table.index_axis(row, 0)?.reshape(&[1, 1, dim])?;
        let tsrow = ts4.index_axis(row, 2)?;
        out.push(add(&trow, &tsrow)?);
    }
    Ok(out)
}

/// Index a single position `i` along `axis`, dropping that axis.
trait IndexAxis {
    fn index_axis(&self, i: i32, axis: i32) -> Result<Array>;
}
impl IndexAxis for Array {
    fn index_axis(&self, i: i32, axis: i32) -> Result<Array> {
        Ok(self.take_axis(Array::from_int(i), axis)?)
    }
}

/// One video transformer block (`BasicAVTransformerBlock`, video-only / gated 2.3 path).
pub struct VideoBlock {
    attn1: Attention,
    attn2: Attention,
    ff: FeedForward,
    scale_shift_table: Array,        // (9, inner)
    prompt_scale_shift_table: Array, // (2, inner)
    eps: f32,
}

impl VideoBlock {
    /// Load a block (`prefix` e.g. `transformer_blocks.0`) at the given [`Precision`].
    pub fn load(w: &Weights, prefix: &str, cfg: &LtxConfig, prec: Precision) -> Result<Self> {
        let (h, dh, eps) = (
            cfg.num_attention_heads,
            cfg.attention_head_dim,
            cfg.norm_eps as f32,
        );
        Ok(Self {
            attn1: Attention::load(w, &format!("{prefix}.attn1"), h, dh, eps, prec)?,
            attn2: Attention::load(w, &format!("{prefix}.attn2"), h, dh, eps, prec)?,
            ff: FeedForward::load(w, &format!("{prefix}.ff"), prec)?,
            scale_shift_table: param(w, &format!("{prefix}.scale_shift_table"), prec)?,
            prompt_scale_shift_table: param(
                w,
                &format!("{prefix}.prompt_scale_shift_table"),
                prec,
            )?,
            eps: cfg.norm_eps as f32,
        })
    }

    /// Forward (gated, 9-row table): MSA(self, RoPE) → text cross-attn (prompt-modulated context) →
    /// FeedForward, each adaLN-modulated and gated.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Array,
        timesteps: &Array,
        prompt_timestep: Option<&Array>,
        context: &Array,
        mask: Option<&Array>,
        cos: &Array,
        sin: &Array,
    ) -> Result<Array> {
        // --- MSA (self-attention) ---
        let msa = ada_values(&self.scale_shift_table, timesteps, 0, 3)?;
        let norm = modulate(&rms_norm_noweight(x, self.eps)?, &msa[1], &msa[0])?;
        let attn = self
            .attn1
            .forward(&norm, None, None, Some((cos, sin)), None)?;
        let mut x = add(x, &multiply(&attn, &msa[2])?)?;

        // --- prompt-adaLN on the text context ---
        let v_context = {
            let (p_shift, p_scale) = match prompt_timestep {
                Some(pt) => {
                    let p = ada_values(&self.prompt_scale_shift_table, pt, 0, 2)?;
                    (p[0].clone(), p[1].clone())
                }
                None => (
                    self.prompt_scale_shift_table.index_axis(0, 0)?,
                    self.prompt_scale_shift_table.index_axis(1, 0)?,
                ),
            };
            modulate(context, &p_scale, &p_shift)?
        };

        // --- text cross-attention (adaLN rows 6..9) ---
        let ca = ada_values(&self.scale_shift_table, timesteps, 6, 9)?;
        let norm_ca = modulate(&rms_norm_noweight(&x, self.eps)?, &ca[1], &ca[0])?;
        let cross = self
            .attn2
            .forward(&norm_ca, Some(&v_context), mask, None, None)?;
        x = add(&x, &multiply(&cross, &ca[2])?)?;

        // --- FeedForward (adaLN rows 3..6) ---
        let mlp = ada_values(&self.scale_shift_table, timesteps, 3, 6)?;
        let norm_mlp = modulate(&rms_norm_noweight(&x, self.eps)?, &mlp[1], &mlp[0])?;
        let ff = self.ff.forward(&norm_mlp)?;
        x = add(&x, &multiply(&ff, &mlp[2])?)?;

        Ok(x)
    }
}

/// PixArt sinusoidal timestep embedding (`flip_sin_to_cos`, `downscale_freq_shift = 0`, max_period
/// 10000): `concat([cos(t·f), sin(t·f)])` with `f[i] = exp(−ln(10000)·i/half)`. `timesteps` is `(N,)`
/// f32; returns `(N, TIME_PROJ_DIM)` f32.
///
/// The log-spaced freqs are computed in **MLX float32** (`arange → ×(−ln θ) → ÷half → exp`), mirroring
/// the reference `get_timestep_embedding` op-for-op. A host-f64 table (the obvious shortcut) diverges
/// ~1e-7 per element from the MLX-f32 kernels (88/128 freqs differ; the projection differs up to
/// ~5e-5 after ×1000 + cos/sin) — invisible in bf16 but, in the F32Q8 path, this adaLN timestep
/// embedding modulates **every** block and the sub-ULP seed compounds across the 48-layer residual
/// stream into a percent-level velocity divergence that the distilled stage-1 sampler then amplifies.
/// (RoPE, by contrast, follows the reference's own numpy-f64 path — see [`crate::rope`].)
fn timestep_embedding(timesteps: &Array) -> Result<Array> {
    let half = TIME_PROJ_DIM / 2; // 128
    let neg_ln = -(10000f64).ln() as f32;
    let exponent = divide(
        &multiply(&Array::arange::<_, f32>(None, half, None)?, scalar(neg_ln))?,
        scalar(half as f32),
    )?;
    let freq = exponent.exp()?.reshape(&[1, half])?; // (1, half)
    let emb = multiply(&timesteps.reshape(&[-1, 1])?, &freq)?; // (N, half)
    Ok(concatenate_axis(&[&emb.cos()?, &emb.sin()?], 1)?) // (N, dim), cos first
}

/// adaLN-single (`AdaLayerNormSingle`): `timestep → sinusoidal(256) → MLP(silu) → embedded`, then
/// `linear(silu(embedded)) → coeff·dim` scale-shift parameters.
struct AdaLayerNormSingle {
    ts_lin1: Linear, // 256 → dim
    ts_lin2: Linear, // dim → dim
    linear: Linear,  // dim → coeff·dim
}

impl AdaLayerNormSingle {
    fn load(w: &Weights, prefix: &str, prec: Precision) -> Result<Self> {
        Ok(Self {
            ts_lin1: Linear::load(w, &format!("{prefix}.emb.timestep_embedder.linear1"), prec)?,
            ts_lin2: Linear::load(w, &format!("{prefix}.emb.timestep_embedder.linear2"), prec)?,
            linear: Linear::load(w, &format!("{prefix}.linear"), prec)?,
        })
    }

    /// `timestep` is the already-scaled `(N,)` f32. Returns `(scale_shift (N, coeff·dim), embedded
    /// (N, dim))` in `dt`.
    fn forward(&self, timestep: &Array, dt: Dtype) -> Result<(Array, Array)> {
        let proj = timestep_embedding(timestep)?.as_dtype(dt)?;
        let h = mlx_gen::nn::silu(&self.ts_lin1.forward(&proj)?)?;
        let embedded = self.ts_lin2.forward(&h)?;
        let scale_shift = self.linear.forward(&mlx_gen::nn::silu(&embedded)?)?;
        Ok((scale_shift, embedded))
    }
}

/// The LTX-2.3 video DiT: preprocessor + 48 blocks + output projection. Predicts velocity.
pub struct LtxDiT {
    patchify_proj: Linear,
    adaln: AdaLayerNormSingle,
    prompt_adaln: Option<AdaLayerNormSingle>,
    blocks: Vec<VideoBlock>,
    scale_shift_table: Array, // (2, inner)
    proj_out: Linear,
    cfg: LtxConfig,
    prec: Precision,
}

impl LtxDiT {
    pub fn from_weights(w: &Weights, cfg: &LtxConfig, prec: Precision) -> Result<Self> {
        let blocks = (0..cfg.num_layers)
            .map(|i| VideoBlock::load(w, &format!("transformer_blocks.{i}"), cfg, prec))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patchify_proj: Linear::load(w, "patchify_proj", prec)?,
            adaln: AdaLayerNormSingle::load(w, "adaln_single", prec)?,
            prompt_adaln: if cfg.apply_gated_attention {
                Some(AdaLayerNormSingle::load(w, "prompt_adaln_single", prec)?)
            } else {
                None
            },
            blocks,
            scale_shift_table: param(w, "scale_shift_table", prec)?,
            proj_out: Linear::load(w, "proj_out", prec)?,
            cfg: cfg.clone(),
            prec,
        })
    }

    /// The preprocessor (mirrors the reference `TransformerArgsPreprocessor.prepare`): patchify_proj →
    /// adaLN-single timestep projection + prompt-adaLN → caption_projection (Identity, 2.3) → SPLIT
    /// RoPE tables. Shared by [`forward`](Self::forward) and [`block_hidden`](Self::block_hidden).
    fn preprocess(
        &self,
        latent: &Array,
        timestep: &Array,
        context: &Array,
        positions: &Array,
    ) -> Result<Preprocessed> {
        let dt = self.prec.dtype();
        let b = latent.shape()[0];
        let inner = self.cfg.inner_dim();
        let coeff = self.cfg.adaln_embedding_coefficient;

        let x = self.patchify_proj.forward(&latent.as_dtype(dt)?)?;

        // adaLN-single timestep projection. The `× timestep_scale_multiplier` runs in the **input
        // dtype** (matching `denoise_av`, which feeds a latent-dtype timestep): the adaLN sinusoid
        // upcasts to f32 internally, but a bf16 timestep must round `bf16(σ·1000)` *first* — pre-
        // upcasting to f32 would change the high-frequency sinusoid phase (~33% velocity divergence
        // in the bf16 path). f32 input is unaffected (`f32(σ)·1000` either way).
        let mult = scalar(self.cfg.timestep_scale_multiplier as f32).as_dtype(timestep.dtype())?;
        let ts_flat = multiply(timestep, &mult)?.reshape(&[-1])?;
        let (ts_emb, emb_ts) = self.adaln.forward(&ts_flat, dt)?;
        let ts_emb = ts_emb.reshape(&[b, -1, coeff * inner])?;
        let emb_ts = emb_ts.reshape(&[b, -1, inner])?;

        // prompt-adaLN (gated family): one shared modulation per sample.
        let prompt_ts = match &self.prompt_adaln {
            Some(padaln) => {
                let src = if timestep.ndim() > 1 {
                    timestep.index_axis(0, 1)?.reshape(&[b, 1])?
                } else {
                    timestep.clone()
                };
                let src = multiply(&src, &mult)?.reshape(&[-1])?;
                let (pts, _) = padaln.forward(&src, dt)?;
                Some(pts.reshape(&[b, -1, 2 * inner])?)
            }
            None => None,
        };

        // caption_projection = Identity (2.3): context enters cross-attn as-is.
        let context = context.as_dtype(dt)?;

        // SPLIT RoPE from the position grid (f32 tables; the block casts per input dtype).
        let (cos, sin) = precompute_split_freqs_cis(
            positions,
            inner,
            self.cfg.positional_embedding_theta,
            &self.cfg.positional_embedding_max_pos,
            self.cfg.num_attention_heads,
        )?;

        Ok(Preprocessed {
            x,
            ts_emb,
            emb_ts,
            prompt_ts,
            context,
            cos,
            sin,
        })
    }

    /// Velocity forward.
    ///
    /// * `latent` — `(B, S, in_channels=128)` patchified latent tokens.
    /// * `timestep` — `(B, 1)` (or `(B,)`) per-sample sigma (T2V; broadcast over tokens).
    /// * `context` — `(B, ctx, inner)` text embeddings (connector output); `mask` its additive mask.
    /// * `positions` — `(B, 3, S, 2)` position grid (from [`crate::positions`]).
    pub fn forward(
        &self,
        latent: &Array,
        timestep: &Array,
        context: &Array,
        mask: Option<&Array>,
        positions: &Array,
    ) -> Result<Array> {
        let p = self.preprocess(latent, timestep, context, positions)?;
        let mut h = p.x;
        for block in &self.blocks {
            h = block.forward(
                &h,
                &p.ts_emb,
                p.prompt_ts.as_ref(),
                &p.context,
                mask,
                &p.cos,
                &p.sin,
            )?;
        }
        self.output_head(&h, &p.emb_ts)
    }

    /// Diagnostic: run the preprocessor + the first `n` blocks and return the hidden state (for the
    /// per-block bisection of the e2e residual). `n == blocks.len()` is the full pre-output hidden.
    #[doc(hidden)]
    pub fn block_hidden(
        &self,
        latent: &Array,
        timestep: &Array,
        context: &Array,
        mask: Option<&Array>,
        positions: &Array,
        n: usize,
    ) -> Result<Array> {
        let p = self.preprocess(latent, timestep, context, positions)?;
        let mut h = p.x;
        for block in self.blocks.iter().take(n) {
            h = block.forward(
                &h,
                &p.ts_emb,
                p.prompt_ts.as_ref(),
                &p.context,
                mask,
                &p.cos,
                &p.sin,
            )?;
        }
        Ok(h)
    }
}

/// The [`LtxDiT::preprocess`] outputs threaded into the block stack + output head.
struct Preprocessed {
    x: Array,
    ts_emb: Array,
    emb_ts: Array,
    prompt_ts: Option<Array>,
    context: Array,
    cos: Array,
    sin: Array,
}

impl LtxDiT {
    /// The output head in isolation (LayerNorm-affine-false → final scale-shift → proj_out), for the
    /// S3b bisection: feed the reference post-block hidden + embedded timestep, compare the velocity.
    pub fn output_head(&self, h: &Array, emb_ts: &Array) -> Result<Array> {
        let b = h.shape()[0];
        let inner = self.cfg.inner_dim();
        let table = self.scale_shift_table.reshape(&[1, 1, 2, inner])?;
        let ss = add(&table, &emb_ts.reshape(&[b, -1, 1, inner])?)?;
        let shift = ss.index_axis(0, 2)?;
        let scale = ss.index_axis(1, 2)?;
        let normed = layer_norm(h, None, None, self.cfg.norm_eps as f32)?;
        let out = modulate(&normed, &scale, &shift)?;
        self.proj_out.forward(&out)
    }
}

/// `denoised = latent − σ·velocity` (`to_denoised`): velocity → x₀.
pub fn to_denoised(latent: &Array, velocity: &Array, sigma: &Array) -> Result<Array> {
    Ok(subtract(latent, &multiply(velocity, sigma)?)?)
}

// ===================================================================================================
// AudioVideo DiT (sc-2684) — the dual-modality `BasicAVTransformerBlock` / `LTXModel`.
// ===================================================================================================

/// `positions[:, 0:1, :, :]` — the time axis as a `(B, 1, T, 2)` grid (the cross-modal RoPE input;
/// `MultiModalTransformerArgsPreprocessor.prepare`). For the audio grid (already `(B, 1, T, 2)`) this
/// is a no-op slice.
fn time_axis(positions: &Array) -> Result<Array> {
    let sh = positions.shape();
    let (b, t) = (sh[0], sh[2]);
    Ok(positions
        .take_axis(Array::from_int(0), 1)? // (B, T, 2)
        .reshape(&[b, 1, t, 2])?)
}

/// One modality's non-block modules + dims — the video or audio half of the AV DiT. Carries the
/// patchify projection, adaLN-single (timestep → coeff·dim) + prompt-adaLN, the two cross-modal
/// adaLN-single modules (4-coeff scale-shift + 1-coeff gate), the output scale-shift table, and the
/// output projection, plus the dims that drive RoPE.
struct Stream {
    patchify: Linear,
    adaln: AdaLayerNormSingle,
    prompt_adaln: AdaLayerNormSingle,
    cross_ss_adaln: AdaLayerNormSingle,
    cross_gate_adaln: AdaLayerNormSingle,
    scale_shift_table: Array, // (2, inner) output head
    proj_out: Linear,
    inner: i32,
    heads: i32,
    coeff: i32, // adaLN row count (9 gated)
    self_max_pos: Vec<i32>,
    cross_inner: i32, // audio_cross_attention_dim (2048) — the cross-modal RoPE inner
    cross_max_pos: i32,
    theta: f64,
    ts_mult: i32,
    av_ca_ts_mult: i32,
    eps: f32,
    prec: Precision,
}

/// The per-modality preprocessed args threaded into the block stack + output head.
struct StreamPrep {
    x: Array,
    ts_emb: Array,
    emb_ts: Array,
    prompt_ts: Array,
    context: Array,
    cos: Array,
    sin: Array,
    cross_cos: Array,
    cross_sin: Array,
    cross_ss_ts: Array,
    cross_gate_ts: Array,
}

/// Borrowed view of a [`StreamPrep`] passed to [`AvBlock::forward`].
struct StreamArgs<'a> {
    ts_emb: &'a Array,
    prompt_ts: &'a Array,
    context: &'a Array,
    mask: Option<&'a Array>,
    cos: &'a Array,
    sin: &'a Array,
    cross_cos: &'a Array,
    cross_sin: &'a Array,
    cross_ss_ts: &'a Array,
    cross_gate_ts: &'a Array,
}

impl StreamPrep {
    fn args<'a>(&'a self, mask: Option<&'a Array>) -> StreamArgs<'a> {
        StreamArgs {
            ts_emb: &self.ts_emb,
            prompt_ts: &self.prompt_ts,
            context: &self.context,
            mask,
            cos: &self.cos,
            sin: &self.sin,
            cross_cos: &self.cross_cos,
            cross_sin: &self.cross_sin,
            cross_ss_ts: &self.cross_ss_ts,
            cross_gate_ts: &self.cross_gate_ts,
        }
    }
}

impl Stream {
    /// `latent` `(B, S, in)`, per-token `timestep` `(B, S)`, text `context`, `positions` grid.
    /// Reproduces `TransformerArgsPreprocessor.prepare` + the multimodal cross-PE / cross-timesteps.
    fn prepare(
        &self,
        latent: &Array,
        timestep: &Array,
        context: &Array,
        positions: &Array,
    ) -> Result<StreamPrep> {
        let dt = self.prec.dtype();
        let b = latent.shape()[0];
        let (inner, coeff) = (self.inner, self.coeff);

        let x = self.patchify.forward(&latent.as_dtype(dt)?)?;

        // adaLN-single timestep projection (the `× ts_mult` runs in the input dtype; see the
        // video-only path's note — bf16 must round `bf16(σ·1000)` first).
        let mult = scalar(self.ts_mult as f32).as_dtype(timestep.dtype())?;
        let ts_flat = multiply(timestep, &mult)?.reshape(&[-1])?;
        let (ts_emb, emb_ts) = self.adaln.forward(&ts_flat, dt)?;
        let ts_emb = ts_emb.reshape(&[b, -1, coeff * inner])?;
        let emb_ts = emb_ts.reshape(&[b, -1, inner])?;

        // prompt-adaLN: one shared modulation per sample (timestep[:, :1]).
        let src = if timestep.ndim() > 1 {
            timestep.index_axis(0, 1)?.reshape(&[b, 1])?
        } else {
            timestep.clone()
        };
        let src = multiply(&src, &mult)?.reshape(&[-1])?;
        let (pts, _) = self.prompt_adaln.forward(&src, dt)?;
        let prompt_ts = pts.reshape(&[b, -1, 2 * inner])?;

        // Cross-modal scale-shift (4·dim) + gate (1·dim) timesteps. The gate timestep carries the
        // extra `av_ca_factor = av_ca_ts_mult / ts_mult` (1.0 for 2.3, an exact f32 no-op).
        let (cross_ss, _) = self.cross_ss_adaln.forward(&ts_flat, dt)?;
        let cross_ss_ts = cross_ss.reshape(&[b, -1, 4 * inner])?;
        let factor =
            scalar(self.av_ca_ts_mult as f32 / self.ts_mult as f32).as_dtype(ts_flat.dtype())?;
        let gate_in = multiply(&ts_flat, &factor)?;
        let (cross_gate, _) = self.cross_gate_adaln.forward(&gate_in, dt)?;
        let cross_gate_ts = cross_gate.reshape(&[b, -1, inner])?;

        // caption_projection = Identity (2.3): context enters cross-attn as-is.
        let context = context.as_dtype(dt)?;

        // Self-attention SPLIT RoPE (modality inner dim, modality max_pos).
        let (cos, sin) = precompute_split_freqs_cis(
            positions,
            inner,
            self.theta,
            &self.self_max_pos,
            self.heads,
        )?;
        // Cross-modal SPLIT RoPE: the time axis only, at the cross inner dim (2048) / cross max_pos.
        let (cross_cos, cross_sin) = precompute_split_freqs_cis(
            &time_axis(positions)?,
            self.cross_inner,
            self.theta,
            &[self.cross_max_pos],
            self.heads,
        )?;

        Ok(StreamPrep {
            x,
            ts_emb,
            emb_ts,
            prompt_ts,
            context,
            cos,
            sin,
            cross_cos,
            cross_sin,
            cross_ss_ts,
            cross_gate_ts,
        })
    }

    /// Output head (LayerNorm-affine-false → final 2-row scale-shift → proj_out). Mirrors
    /// `LTXModel._process_output`.
    fn output_head(&self, h: &Array, emb_ts: &Array) -> Result<Array> {
        let b = h.shape()[0];
        let table = self.scale_shift_table.reshape(&[1, 1, 2, self.inner])?;
        let ss = add(&table, &emb_ts.reshape(&[b, -1, 1, self.inner])?)?;
        let shift = ss.index_axis(0, 2)?;
        let scale = ss.index_axis(1, 2)?;
        let normed = layer_norm(h, None, None, self.eps)?;
        self.proj_out.forward(&modulate(&normed, &scale, &shift)?)
    }
}

/// `4·scale-shift + 1·gate` cross-modal adaLN values from the pre-split tables. Returns
/// `(scale_a2v, shift_a2v, scale_v2a, shift_v2a, gate)` — the row layout of
/// `scale_shift_table_a2v_ca_{audio,video}` (`get_av_ca_ada_values`).
fn av_ca_ada(
    ss_table: &Array,
    gate_table: &Array,
    ss_ts: &Array,
    gate_ts: &Array,
) -> Result<(Array, Array, Array, Array, Array)> {
    let ss = ada_values(ss_table, ss_ts, 0, 4)?;
    let g = ada_values(gate_table, gate_ts, 0, 1)?;
    Ok((
        ss[0].clone(),
        ss[1].clone(),
        ss[2].clone(),
        ss[3].clone(),
        g[0].clone(),
    ))
}

/// One AudioVideo transformer block: the video stack + the audio stack + bidirectional cross-modal
/// attention (`BasicAVTransformerBlock`). Per-block order: video self+text-CA → audio self+text-CA →
/// cross-modal (a2v updates video, v2a updates audio) → video FF → audio FF.
struct AvBlock {
    // Video.
    attn1: Attention,
    attn2: Attention,
    ff: FeedForward,
    v_sst: Array, // (9, 4096)
    v_pst: Array, // (2, 4096)
    // Audio.
    a_attn1: Attention,
    a_attn2: Attention,
    a_ff: FeedForward,
    a_sst: Array, // (9, 2048)
    a_pst: Array, // (2, 2048)
    // Cross-modal.
    a2v: Attention,       // audio_to_video_attn (Q video, K/V audio)
    v2a: Attention,       // video_to_audio_attn (Q audio, K/V video)
    ca_audio_ss: Array,   // (4, 2048)
    ca_audio_gate: Array, // (1, 2048)
    ca_video_ss: Array,   // (4, 4096)
    ca_video_gate: Array, // (1, 4096)
    eps: f32,
}

impl AvBlock {
    fn load(w: &Weights, prefix: &str, cfg: &LtxConfig, prec: Precision) -> Result<Self> {
        let eps = cfg.norm_eps as f32;
        let (vh, vdh) = (cfg.num_attention_heads, cfg.attention_head_dim);
        let (ah, adh) = (cfg.audio_num_attention_heads, cfg.audio_attention_head_dim);
        // Split a (5, dim) cross table into the 4-row scale-shift block + the 1-row gate.
        let split = |key: &str| -> Result<(Array, Array)> {
            let t = param(w, &format!("{prefix}.{key}"), prec)?;
            let ss = t.take_axis(Array::from_slice(&[0, 1, 2, 3], &[4]), 0)?;
            let gate = t.take_axis(Array::from_slice(&[4], &[1]), 0)?;
            Ok((ss, gate))
        };
        let (ca_audio_ss, ca_audio_gate) = split("scale_shift_table_a2v_ca_audio")?;
        let (ca_video_ss, ca_video_gate) = split("scale_shift_table_a2v_ca_video")?;
        Ok(Self {
            attn1: Attention::load(w, &format!("{prefix}.attn1"), vh, vdh, eps, prec)?,
            attn2: Attention::load(w, &format!("{prefix}.attn2"), vh, vdh, eps, prec)?,
            ff: FeedForward::load(w, &format!("{prefix}.ff"), prec)?,
            v_sst: param(w, &format!("{prefix}.scale_shift_table"), prec)?,
            v_pst: param(w, &format!("{prefix}.prompt_scale_shift_table"), prec)?,
            a_attn1: Attention::load(w, &format!("{prefix}.audio_attn1"), ah, adh, eps, prec)?,
            a_attn2: Attention::load(w, &format!("{prefix}.audio_attn2"), ah, adh, eps, prec)?,
            a_ff: FeedForward::load(w, &format!("{prefix}.audio_ff"), prec)?,
            a_sst: param(w, &format!("{prefix}.audio_scale_shift_table"), prec)?,
            a_pst: param(w, &format!("{prefix}.audio_prompt_scale_shift_table"), prec)?,
            // Cross-modal attns run at the audio inner dim (heads 32 × head_dim 64 = 2048).
            a2v: Attention::load(
                w,
                &format!("{prefix}.audio_to_video_attn"),
                ah,
                adh,
                eps,
                prec,
            )?,
            v2a: Attention::load(
                w,
                &format!("{prefix}.video_to_audio_attn"),
                ah,
                adh,
                eps,
                prec,
            )?,
            ca_audio_ss,
            ca_audio_gate,
            ca_video_ss,
            ca_video_gate,
            eps,
        })
    }

    /// Self+text-CA for one modality (`run_vx`/`run_ax` body, sans FF): MSA (RoPE) → prompt-modulated
    /// text cross-attention. Returns the updated stream hidden.
    #[allow(clippy::too_many_arguments)]
    fn self_and_text(
        &self,
        x: &Array,
        attn1: &Attention,
        attn2: &Attention,
        sst: &Array,
        pst: &Array,
        a: &StreamArgs,
    ) -> Result<Array> {
        let msa = ada_values(sst, a.ts_emb, 0, 3)?;
        let norm = modulate(&rms_norm_noweight(x, self.eps)?, &msa[1], &msa[0])?;
        let attn = attn1.forward(&norm, None, None, Some((a.cos, a.sin)), None)?;
        let mut x = add(x, &multiply(&attn, &msa[2])?)?;

        let p = ada_values(pst, a.prompt_ts, 0, 2)?;
        let context = modulate(a.context, &p[1], &p[0])?;

        let ca = ada_values(sst, a.ts_emb, 6, 9)?;
        let norm_ca = modulate(&rms_norm_noweight(&x, self.eps)?, &ca[1], &ca[0])?;
        let cross = attn2.forward(&norm_ca, Some(&context), a.mask, None, None)?;
        x = add(&x, &multiply(&cross, &ca[2])?)?;
        Ok(x)
    }

    /// FeedForward (adaLN rows 3..6) for one modality.
    fn feed_forward(
        &self,
        x: &Array,
        ff: &FeedForward,
        sst: &Array,
        ts_emb: &Array,
    ) -> Result<Array> {
        let mlp = ada_values(sst, ts_emb, 3, 6)?;
        let norm = modulate(&rms_norm_noweight(x, self.eps)?, &mlp[1], &mlp[0])?;
        Ok(add(x, &multiply(&ff.forward(&norm)?, &mlp[2])?)?)
    }

    /// Joint forward: `(vx, ax)` in, `(vx, ax)` out.
    fn forward(
        &self,
        vx: &Array,
        ax: &Array,
        v: &StreamArgs,
        a: &StreamArgs,
    ) -> Result<(Array, Array)> {
        // Video / audio self-attention + text cross-attention.
        let mut vx =
            self.self_and_text(vx, &self.attn1, &self.attn2, &self.v_sst, &self.v_pst, v)?;
        let mut ax = self.self_and_text(
            ax,
            &self.a_attn1,
            &self.a_attn2,
            &self.a_sst,
            &self.a_pst,
            a,
        )?;

        // Cross-modal attention — both directions read the pre-update rms_norm snapshots.
        let vx_n3 = rms_norm_noweight(&vx, self.eps)?;
        let ax_n3 = rms_norm_noweight(&ax, self.eps)?;
        let (sca_a2v, sha_a2v, sca_v2a, sha_v2a, gate_v2a) = av_ca_ada(
            &self.ca_audio_ss,
            &self.ca_audio_gate,
            a.cross_ss_ts,
            a.cross_gate_ts,
        )?;
        let (scv_a2v, shv_a2v, scv_v2a, shv_v2a, gate_a2v) = av_ca_ada(
            &self.ca_video_ss,
            &self.ca_video_gate,
            v.cross_ss_ts,
            v.cross_gate_ts,
        )?;

        // Audio-to-Video: Q from video (video cross-PE), K/V from audio (audio cross-PE).
        let a2v = self.a2v.forward(
            &modulate(&vx_n3, &scv_a2v, &shv_a2v)?,
            Some(&modulate(&ax_n3, &sca_a2v, &sha_a2v)?),
            None,
            Some((v.cross_cos, v.cross_sin)),
            Some((a.cross_cos, a.cross_sin)),
        )?;
        vx = add(&vx, &multiply(&a2v, &gate_a2v)?)?;

        // Video-to-Audio: Q from audio (audio cross-PE), K/V from video (video cross-PE).
        let v2a = self.v2a.forward(
            &modulate(&ax_n3, &sca_v2a, &sha_v2a)?,
            Some(&modulate(&vx_n3, &scv_v2a, &shv_v2a)?),
            None,
            Some((a.cross_cos, a.cross_sin)),
            Some((v.cross_cos, v.cross_sin)),
        )?;
        ax = add(&ax, &multiply(&v2a, &gate_v2a)?)?;

        // FeedForward.
        vx = self.feed_forward(&vx, &self.ff, &self.v_sst, v.ts_emb)?;
        ax = self.feed_forward(&ax, &self.a_ff, &self.a_sst, a.ts_emb)?;
        Ok((vx, ax))
    }
}

/// The LTX-2.3 **AudioVideo** DiT (`LTXModel` with both stacks). Predicts `(video_velocity,
/// audio_velocity)` from the two latent token streams + shared text conditioning.
pub struct AvDiT {
    video: Stream,
    audio: Stream,
    blocks: Vec<AvBlock>,
}

impl AvDiT {
    pub fn from_weights(w: &Weights, cfg: &LtxConfig, prec: Precision) -> Result<Self> {
        let video = Stream {
            patchify: Linear::load(w, "patchify_proj", prec)?,
            adaln: AdaLayerNormSingle::load(w, "adaln_single", prec)?,
            prompt_adaln: AdaLayerNormSingle::load(w, "prompt_adaln_single", prec)?,
            cross_ss_adaln: AdaLayerNormSingle::load(
                w,
                "av_ca_video_scale_shift_adaln_single",
                prec,
            )?,
            cross_gate_adaln: AdaLayerNormSingle::load(w, "av_ca_a2v_gate_adaln_single", prec)?,
            scale_shift_table: param(w, "scale_shift_table", prec)?,
            proj_out: Linear::load(w, "proj_out", prec)?,
            inner: cfg.inner_dim(),
            heads: cfg.num_attention_heads,
            coeff: cfg.adaln_embedding_coefficient,
            self_max_pos: cfg.positional_embedding_max_pos.to_vec(),
            cross_inner: cfg.audio_cross_attention_dim,
            cross_max_pos: cfg.cross_pe_max_pos(),
            theta: cfg.positional_embedding_theta,
            ts_mult: cfg.timestep_scale_multiplier,
            av_ca_ts_mult: cfg.av_ca_timestep_scale_multiplier,
            eps: cfg.norm_eps as f32,
            prec,
        };
        let audio = Stream {
            patchify: Linear::load(w, "audio_patchify_proj", prec)?,
            adaln: AdaLayerNormSingle::load(w, "audio_adaln_single", prec)?,
            prompt_adaln: AdaLayerNormSingle::load(w, "audio_prompt_adaln_single", prec)?,
            cross_ss_adaln: AdaLayerNormSingle::load(
                w,
                "av_ca_audio_scale_shift_adaln_single",
                prec,
            )?,
            cross_gate_adaln: AdaLayerNormSingle::load(w, "av_ca_v2a_gate_adaln_single", prec)?,
            scale_shift_table: param(w, "audio_scale_shift_table", prec)?,
            proj_out: Linear::load(w, "audio_proj_out", prec)?,
            inner: cfg.audio_inner_dim(),
            heads: cfg.audio_num_attention_heads,
            coeff: cfg.adaln_embedding_coefficient,
            self_max_pos: vec![cfg.audio_positional_embedding_max_pos],
            cross_inner: cfg.audio_cross_attention_dim,
            cross_max_pos: cfg.cross_pe_max_pos(),
            theta: cfg.positional_embedding_theta,
            ts_mult: cfg.timestep_scale_multiplier,
            av_ca_ts_mult: cfg.av_ca_timestep_scale_multiplier,
            eps: cfg.norm_eps as f32,
            prec,
        };
        let blocks = (0..cfg.num_layers)
            .map(|i| AvBlock::load(w, &format!("transformer_blocks.{i}"), cfg, prec))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            video,
            audio,
            blocks,
        })
    }

    /// Joint velocity forward.
    ///
    /// * `*_latent` — `(B, S, in_channels)` patchified tokens (video 128, audio 128).
    /// * `*_timestep` — `(B, S)` per-token sigma.
    /// * `*_context` — text embeddings (video 4096, audio 2048); `*_mask` their additive masks.
    /// * `*_positions` — the position grids (video `(B,3,T,2)`, audio `(B,1,T,2)`).
    ///
    /// Returns `(video_velocity (B, S_v, 128), audio_velocity (B, S_a, 128))`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        video_latent: &Array,
        video_timestep: &Array,
        video_context: &Array,
        video_mask: Option<&Array>,
        video_positions: &Array,
        audio_latent: &Array,
        audio_timestep: &Array,
        audio_context: &Array,
        audio_mask: Option<&Array>,
        audio_positions: &Array,
    ) -> Result<(Array, Array)> {
        let vp =
            self.video
                .prepare(video_latent, video_timestep, video_context, video_positions)?;
        let ap =
            self.audio
                .prepare(audio_latent, audio_timestep, audio_context, audio_positions)?;
        let (mut vx, mut ax) = (vp.x.clone(), ap.x.clone());
        let (va, aa) = (vp.args(video_mask), ap.args(audio_mask));
        for block in &self.blocks {
            let (nv, na) = block.forward(&vx, &ax, &va, &aa)?;
            vx = nv;
            ax = na;
        }
        let v_vel = self.video.output_head(&vx, &vp.emb_ts)?;
        let a_vel = self.audio.output_head(&ax, &ap.emb_ts)?;
        Ok((v_vel, a_vel))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modulate_closed_form() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let scale = Array::from_slice(&[0.0f32, 1.0, 0.0, 1.0], &[1, 1, 4]);
        let shift = Array::from_slice(&[1.0f32, 0.0, -1.0, 0.0], &[1, 1, 4]);
        let got = modulate(&x, &scale, &shift).unwrap();
        assert_eq!(got.as_slice::<f32>(), &[2.0, 4.0, 2.0, 8.0]);
    }

    #[test]
    fn ada_values_splits_rows() {
        let table = Array::from_slice(&(0..18).map(|v| v as f32).collect::<Vec<_>>(), &[9, 2]);
        let ts = Array::zeros::<f32>(&[1, 1, 18]).unwrap();
        let vals = ada_values(&table, &ts, 0, 3).unwrap();
        assert_eq!(vals.len(), 3);
        assert_eq!(vals[0].as_slice::<f32>(), &[0.0, 1.0]);
        assert_eq!(vals[2].as_slice::<f32>(), &[4.0, 5.0]);
    }

    #[test]
    fn timestep_embedding_shape_and_pad() {
        // (N=2,) → (2, 256), cos-first (t=0 → cos=1, sin=0).
        let t = Array::from_slice(&[0.0f32, 1.0], &[2]);
        let emb = timestep_embedding(&t).unwrap();
        assert_eq!(emb.shape(), &[2, 256]);
        let row0 = emb.index_axis(0, 0).unwrap();
        let s = row0.as_slice::<f32>();
        assert!((s[0] - 1.0).abs() < 1e-6); // cos(0)
        assert!(s[128].abs() < 1e-6); // sin(0)
    }
}
