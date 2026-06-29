//! SANA **Linear Diffusion Transformer trunk** — faithful mlx-rs port of diffusers
//! `SanaTransformer2DModel` / `SanaTransformerBlock` (epic 8485, story sc-8487).
//!
//! Port target: `Efficient-Large-Model/Sana_1600M_1024px_diffusers` (the 1.6B model Clark Labs
//! ported to MLX). We write the **bf16/fp16** path (the checkpoint dtype is preserved through the
//! forward — every op is dtype-preserving); we do NOT copy Clark Labs' 2-bit ternary quant (that was
//! a small-machine fit, not a fidelity requirement).
//!
//! ## Architecture (the four story pillars)
//!
//!  - **ReLU linear self-attention** (`attn1`, `SanaLinearAttnProcessor2_0`) — O(N) attention:
//!    `ReLU(Q),ReLU(K)`, then the `value`-padded-with-a-ones-row trick collapsed to the algebraically
//!    identical numerator/denominator split `num = (Vᵀ·K)·Q`, `den = (Σ_n K)·Q`, divided with a
//!    `1/(·+1e-15)` normalizer — the SAME f32 linear-attention kernel the DC-AE spike
//!    ([`crate::dc_ae::LinearAttn`]) uses, minus the multiscale QKV projections (the trunk's `attn1`
//!    is plain single-scale). `attention_bias=false` for SANA-1.6B → `to_q/k/v` have no bias;
//!    `to_out.0` carries a bias.
//!  - **Cross-attention** (`attn2`, standard softmax SDPA) to the caption embeddings — `to_q/k/v` all
//!    bias-carrying, KV from the projected+normed caption.
//!  - **Mix-FFN** (`ff`, `GLUMBConv`) — `conv_inverted(1×1) → SiLU → conv_depth(3×3 depthwise) → gated
//!    SiLU → conv_point(1×1, no bias)`. The 3×3 depthwise conv is the token-mixer; the FFN runs over
//!    the un-flattened `[B, inner, H, W]` grid (channels-first in the reference; channels-last here).
//!    No residual/norm inside the block's `ff` (the block owns the residual + gate).
//!  - **NoPE** — `interpolation_scale=None` ⇒ `patch_embed` has no `pos_embed`; the conv patchify
//!    (here `patch_size=1`, a 1×1 conv) plus the Mix-FFN depthwise conv provide all locality.
//!
//! Per-block adaLN-single modulation `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
//! gate_mlp)` comes from `block.scale_shift_table[6,dim] + timestep_emb.reshape(B,6,-1)`; the
//! timestep path is `Timesteps(256) → timestep_embedder(MLP) = embedded_timestep`, then
//! `time_embed.linear(SiLU(embedded_timestep)) → [B, 6·dim]`. Output: `SanaModulatedNorm`
//! (affine-free LayerNorm + `top.scale_shift_table[2,dim] + embedded_timestep`) → `proj_out` →
//! unpatchify to `[B, out_channels, H, W]` (32 channels = the DC-AE f32c32 latent, so the trunk's
//! output feeds [`crate::dc_ae::DcAeDecoder::decode`] directly — sc-8489 composition).
//!
//! Tensor keys are the diffusers `SanaTransformer2DModel` names exactly, so a converted checkpoint
//! loads unchanged. Layout convention follows [`crate::dc_ae`]: channels-last NHWC for the conv ops,
//! `[B, N, C]` token layout for the attention/Linear ops (diffusers' `flatten/permute` between the
//! two is mirrored explicitly).

use mlx_rs::fast::layer_norm;
use mlx_rs::ops::{add, clip, divide, matmul, multiply, softmax_axis, split_sections, sum_axes};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::{gelu_tanh, silu, timestep_sincos};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::SanaTransformerConfig;

const F32: Dtype = Dtype::Float32;

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

fn relu(x: &Array) -> Result<Array> {
    Ok(mlx_rs::nn::relu(x)?)
}

// ----------------------------------------------------------------------------------------------
// Linear / norm primitives (dtype-preserving; bf16/fp16 weights flow through unchanged).
// ----------------------------------------------------------------------------------------------

/// `nn.Linear`: stored weight is `[out, in]`; applies `x · Wᵀ (+ b)` over the last axis.
struct Linear {
    w_t: Array, // pre-transposed [in, out]
    b: Option<Array>,
}

impl Linear {
    fn load(w: &Weights, prefix: &str, bias: bool) -> Result<Self> {
        let w_t = w
            .require(&format!("{prefix}.weight"))?
            .transpose_axes(&[1, 0])?;
        let b = if bias {
            Some(w.require(&format!("{prefix}.bias"))?.clone())
        } else {
            None
        };
        Ok(Self { w_t, b })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let inn = sh[sh.len() - 1];
        let out = self.w_t.shape()[1];
        let n: i32 = sh[..sh.len() - 1].iter().product();
        let y = matmul(&x.reshape(&[n, inn])?, &self.w_t)?;
        let mut outsh: Vec<i32> = sh[..sh.len() - 1].to_vec();
        outsh.push(out);
        let y = y.reshape(&outsh)?;
        match &self.b {
            Some(b) => Ok(add(&y, b)?),
            None => Ok(y),
        }
    }
}

/// `RMSNorm(elementwise_affine=True, bias=False)` over the last axis, f32 reduction (diffusers
/// `caption_norm`). `weight` is `[C]`.
fn rms_norm(x: &Array, weight: &Array, eps: f32) -> Result<Array> {
    let dt = x.dtype();
    let rank = x.shape().len();
    let ax = (rank - 1) as i32;
    let xf = x.as_dtype(F32)?;
    let var = mlx_rs::ops::mean_axes(&multiply(&xf, &xf)?, &[ax], true)?;
    let normed = multiply(&xf, &add(&var, scalar(eps))?.rsqrt()?)?;
    Ok(multiply(&normed.as_dtype(dt)?, weight)?)
}

/// adaLN-single affine `norm·(1 + scale) + shift` (diffusers `hidden * (1 + scale) + shift`).
fn modulate(norm: &Array, scale: &Array, shift: &Array) -> Result<Array> {
    let one = scalar(1.0).as_dtype(scale.dtype())?;
    Ok(add(&multiply(norm, &add(scale, &one)?)?, shift)?)
}

// ----------------------------------------------------------------------------------------------
// Conv (channels-last NHWC; PyTorch [O, I/groups, H, W] → mlx [O, H, W, I/groups] at load).
// ----------------------------------------------------------------------------------------------

struct Conv {
    w: Array,
    b: Option<Array>,
    stride: i32,
    padding: i32,
    groups: i32,
}

impl Conv {
    fn load(
        w: &Weights,
        prefix: &str,
        stride: i32,
        padding: i32,
        groups: i32,
        bias: bool,
    ) -> Result<Self> {
        let weight = w
            .require(&format!("{prefix}.weight"))?
            .transpose_axes(&[0, 2, 3, 1])?;
        let b = if bias {
            Some(w.require(&format!("{prefix}.bias"))?.clone())
        } else {
            None
        };
        Ok(Self {
            w: weight,
            b,
            stride,
            padding,
            groups,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let y = mlx_rs::ops::conv2d(
            x,
            &self.w,
            (self.stride, self.stride),
            (self.padding, self.padding),
            (1, 1),
            self.groups,
        )?;
        match &self.b {
            Some(b) => Ok(add(&y, b)?),
            None => Ok(y),
        }
    }
}

// ----------------------------------------------------------------------------------------------
// ReLU linear self-attention (attn1).
// ----------------------------------------------------------------------------------------------

/// `SanaLinearAttnProcessor2_0`: ReLU linear attention over the token axis. Input/output `[B, N, C]`.
struct LinearSelfAttn {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    /// Sprint `qk_norm = "rms_norm_across_heads"` (sc-8490): RMSNorm over the full projected query /
    /// key (the whole `inner_dim`), applied BEFORE the head split and the ReLU. `None` for base SANA.
    norm_q: Option<Array>,
    norm_k: Option<Array>,
    heads: i32,
    attn_eps: f32,
    /// qk-norm RMSNorm eps (`1e-5`, diffusers `Attention.__init__` default). NOT `cfg.norm_eps`
    /// (`1e-6`), which governs only the affine-free LayerNorms.
    qk_norm_eps: f32,
}

impl LinearSelfAttn {
    fn load(w: &Weights, prefix: &str, cfg: &SanaTransformerConfig) -> Result<Self> {
        let (norm_q, norm_k) = if cfg.qk_norm {
            (
                Some(w.require(&format!("{prefix}.norm_q.weight"))?.clone()),
                Some(w.require(&format!("{prefix}.norm_k.weight"))?.clone()),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            // attention_bias=false → q/k/v bias-free; to_out.0 carries a bias.
            to_q: Linear::load(w, &format!("{prefix}.to_q"), false)?,
            to_k: Linear::load(w, &format!("{prefix}.to_k"), false)?,
            to_v: Linear::load(w, &format!("{prefix}.to_v"), false)?,
            to_out: Linear::load(w, &format!("{prefix}.to_out.0"), true)?,
            norm_q,
            norm_k,
            heads: cfg.num_attention_heads,
            attn_eps: cfg.attn_eps,
            qk_norm_eps: cfg.attn_qk_norm_eps,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, n) = (sh[0], sh[1]);
        let inner = self.to_q.w_t.shape()[1];
        let hd = inner / self.heads;

        // qk_norm = "rms_norm_across_heads": RMSNorm over the full `inner_dim`, BEFORE the head split
        // (diffusers applies `attn.norm_q(query)` / `attn.norm_k(key)` to the `[B,N,inner]` projection).
        let q_proj = self.to_q.forward(x)?;
        let q_proj = match &self.norm_q {
            Some(g) => rms_norm(&q_proj, g, self.qk_norm_eps)?,
            None => q_proj,
        };
        let k_proj = self.to_k.forward(x)?;
        let k_proj = match &self.norm_k {
            Some(g) => rms_norm(&k_proj, g, self.qk_norm_eps)?,
            None => k_proj,
        };

        // [B,N,inner] → [B, heads, hd, N]  (diffusers: transpose(1,2).unflatten(1,(heads,-1)))
        let to_bh_d_n = |a: Array| -> Result<Array> {
            Ok(a.reshape(&[b, n, self.heads, hd])?
                .transpose_axes(&[0, 2, 3, 1])?)
        };
        let q = relu(&to_bh_d_n(q_proj)?)?.as_dtype(F32)?; // [B,H,hd,N]
        let k = relu(&to_bh_d_n(k_proj)?)?.as_dtype(F32)?; // [B,H,hd,N]
        let v = to_bh_d_n(self.to_v.forward(x)?)?.as_dtype(F32)?; // [B,H,hd,N]

        // Reference pads value with a ones-row then divides by it. Algebraically identical f32 split:
        //   num = (V·Kᵀ)·Q : [B,H,hd,N]   den = (Σ_n K)·Q : [B,H,1,N]
        let k_t = k.transpose_axes(&[0, 1, 3, 2])?; // [B,H,N,hd]
        let num = matmul(&matmul(&v, &k_t)?, &q)?; // [B,H,hd,N]
        let k_sum = sum_axes(&k, &[3], true)?; // [B,H,hd,1]
        let den = matmul(&k_sum.transpose_axes(&[0, 1, 3, 2])?, &q)?; // [B,H,1,N]
        let out = divide(&num, &add(&den, scalar(self.attn_eps))?)?; // [B,H,hd,N]

        // [B,H,hd,N] → [B,N,inner]
        let out = out
            .transpose_axes(&[0, 3, 1, 2])?
            .reshape(&[b, n, inner])?
            .as_dtype(x.dtype())?;
        let out = self.to_out.forward(&out)?;

        // Reference (`SanaLinearAttnProcessor2_0`) clips `to_out` to fp16's representable range as an
        // overflow guard — but only when the *input* dtype was fp16 (`if original_dtype ==
        // torch.float16: hidden_states.clip(-65504, 65504)`). bf16/f32 are left unchanged.
        if x.dtype() == Dtype::Float16 {
            Ok(clip(&out, (-65504.0, 65504.0))?)
        } else {
            Ok(out)
        }
    }
}

// ----------------------------------------------------------------------------------------------
// Standard cross-attention (attn2) to the caption embedding.
// ----------------------------------------------------------------------------------------------

struct CrossAttn {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    /// Sprint `qk_norm = "rms_norm_across_heads"` (sc-8490): RMSNorm over the full projected query /
    /// key (the whole cross `inner_dim`), applied BEFORE the head split. `None` for base SANA.
    norm_q: Option<Array>,
    norm_k: Option<Array>,
    heads: i32,
    /// qk-norm RMSNorm eps (`1e-5`, diffusers `Attention.__init__` default). NOT `cfg.norm_eps`
    /// (`1e-6`), which governs only the affine-free LayerNorms.
    qk_norm_eps: f32,
}

impl CrossAttn {
    fn load(w: &Weights, prefix: &str, cfg: &SanaTransformerConfig) -> Result<Self> {
        let (norm_q, norm_k) = if cfg.qk_norm {
            (
                Some(w.require(&format!("{prefix}.norm_q.weight"))?.clone()),
                Some(w.require(&format!("{prefix}.norm_k.weight"))?.clone()),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            to_q: Linear::load(w, &format!("{prefix}.to_q"), true)?,
            to_k: Linear::load(w, &format!("{prefix}.to_k"), true)?,
            to_v: Linear::load(w, &format!("{prefix}.to_v"), true)?,
            to_out: Linear::load(w, &format!("{prefix}.to_out.0"), true)?,
            norm_q,
            norm_k,
            heads: cfg.num_cross_attention_heads,
            qk_norm_eps: cfg.attn_qk_norm_eps,
        })
    }

    /// `x` (query) `[B, N, C]`, `kv` (caption) `[B, M, C]`.
    fn forward(&self, x: &Array, kv: &Array) -> Result<Array> {
        let xsh = x.shape();
        let (b, n) = (xsh[0], xsh[1]);
        let m = kv.shape()[1];
        let inner = self.to_q.w_t.shape()[1];
        let hd = inner / self.heads;
        let scale = scalar(1.0 / (hd as f32).sqrt());

        // qk_norm = "rms_norm_across_heads": RMSNorm over the full cross `inner_dim`, BEFORE the head
        // split (diffusers `attn.norm_q(query)` / `attn.norm_k(key)` on the `[B,*,inner]` projection).
        let q_proj = self.to_q.forward(x)?;
        let q_proj = match &self.norm_q {
            Some(g) => rms_norm(&q_proj, g, self.qk_norm_eps)?,
            None => q_proj,
        };
        let k_proj = self.to_k.forward(kv)?;
        let k_proj = match &self.norm_k {
            Some(g) => rms_norm(&k_proj, g, self.qk_norm_eps)?,
            None => k_proj,
        };

        let split_heads = |a: Array, len: i32| -> Result<Array> {
            // [B,len,inner] → [B,heads,len,hd]
            Ok(a.reshape(&[b, len, self.heads, hd])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = split_heads(q_proj, n)?; // [B,H,N,hd]
        let k = split_heads(k_proj, m)?; // [B,H,M,hd]
        let v = split_heads(self.to_v.forward(kv)?, m)?; // [B,H,M,hd]

        // Softmax SDPA in f32 (caption seq is short; full attention).
        let qf = q.as_dtype(F32)?;
        let kf = k.as_dtype(F32)?;
        let scores = multiply(&matmul(&qf, &kf.transpose_axes(&[0, 1, 3, 2])?)?, &scale)?; // [B,H,N,M]
        let probs = softmax_axis(&scores, -1, None)?;
        let ctx = matmul(&probs, &v.as_dtype(F32)?)?; // [B,H,N,hd]

        let ctx = ctx
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, n, inner])?
            .as_dtype(x.dtype())?;
        self.to_out.forward(&ctx)
    }
}

// ----------------------------------------------------------------------------------------------
// GLUMBConv Mix-FFN (block `ff`: norm_type=None, residual_connection=False).
// ----------------------------------------------------------------------------------------------

struct GluMbConv {
    conv_inverted: Conv, // 1×1, in → 2·hidden  (+bias)
    conv_depth: Conv,    // 3×3 depthwise, 2·hidden → 2·hidden (+bias)
    conv_point: Conv,    // 1×1, hidden → out (no bias)
    hidden: i32,
}

impl GluMbConv {
    fn load(w: &Weights, prefix: &str, cfg: &SanaTransformerConfig) -> Result<Self> {
        let inner = cfg.inner_dim();
        let hidden = (cfg.mlp_ratio * inner as f32) as i32;
        Ok(Self {
            conv_inverted: Conv::load(w, &format!("{prefix}.conv_inverted"), 1, 0, 1, true)?,
            conv_depth: Conv::load(w, &format!("{prefix}.conv_depth"), 1, 1, 2 * hidden, true)?,
            conv_point: Conv::load(w, &format!("{prefix}.conv_point"), 1, 0, 1, false)?,
            hidden,
        })
    }

    /// `x` is NHWC `[B, H, W, inner]`. Returns NHWC `[B, H, W, out]`.
    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.conv_inverted.forward(x)?;
        let h = silu(&h)?;
        let h = self.conv_depth.forward(&h)?;
        let parts = split_sections(&h, &[self.hidden], 3)?; // chunk(2) over the channel (NHWC) axis
        let h = multiply(&parts[0], &silu(&parts[1])?)?;
        self.conv_point.forward(&h)
    }
}

// ----------------------------------------------------------------------------------------------
// SanaTransformerBlock.
// ----------------------------------------------------------------------------------------------

struct SanaBlock {
    scale_shift_table: Array, // [6, dim]
    attn1: LinearSelfAttn,
    attn2: CrossAttn,
    ff: GluMbConv,
    norm_eps: f32,
}

impl SanaBlock {
    fn load(w: &Weights, prefix: &str, cfg: &SanaTransformerConfig) -> Result<Self> {
        Ok(Self {
            scale_shift_table: w.require(&format!("{prefix}.scale_shift_table"))?.clone(),
            attn1: LinearSelfAttn::load(w, &format!("{prefix}.attn1"), cfg)?,
            attn2: CrossAttn::load(w, &format!("{prefix}.attn2"), cfg)?,
            ff: GluMbConv::load(w, &format!("{prefix}.ff"), cfg)?,
            norm_eps: cfg.norm_eps,
        })
    }

    /// `hidden` `[B, N, dim]` (N = H·W tokens), `caption` `[B, M, dim]`, `temb` `[B, 6·dim]`.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hidden: &Array,
        caption: &Array,
        temb: &Array,
        h: i32,
        w: i32,
    ) -> Result<Array> {
        let dim = self.scale_shift_table.shape()[1];
        let b = hidden.shape()[0];
        // 1. Modulation: scale_shift_table[None] + temb.reshape(B,6,-1)  → chunk(6) along axis 1.
        let ss = self.scale_shift_table.reshape(&[1, 6, dim])?;
        let modg = add(&ss, &temb.reshape(&[b, 6, dim])?)?; // [B,6,dim]
        let mc = split_sections(&modg, &[1, 2, 3, 4, 5], 1)?; // 6 × [B,1,dim]
        let chunk = |i: usize| -> Result<Array> { Ok(mc[i].reshape(&[b, 1, dim])?) };
        let (shift_msa, scale_msa, gate_msa) = (chunk(0)?, chunk(1)?, chunk(2)?);
        let (shift_mlp, scale_mlp, gate_mlp) = (chunk(3)?, chunk(4)?, chunk(5)?);

        // 2. Self linear-attention.
        let norm_h = layer_norm(hidden, None, None, self.norm_eps)?;
        let norm_h = modulate(&norm_h, &scale_msa, &shift_msa)?;
        let attn_out = self.attn1.forward(&norm_h)?;
        let hidden = add(hidden, &multiply(&gate_msa, &attn_out)?)?;

        // 3. Cross-attention (no pre-norm in SANA — attn2 reads `hidden` directly).
        let cross = self.attn2.forward(&hidden, caption)?;
        let hidden = add(&cross, &hidden)?;

        // 4. Mix-FFN. norm2 → modulate → un-flatten to [B,H,W,dim] → GLUMBConv → flatten → gate.
        let norm_h = layer_norm(&hidden, None, None, self.norm_eps)?;
        let norm_h = modulate(&norm_h, &scale_mlp, &shift_mlp)?;
        let grid = norm_h.reshape(&[b, h, w, dim])?; // [B,N,dim] → NHWC (channels-last)
        let ff = self.ff.forward(&grid)?;
        let ff = ff.reshape(&[b, h * w, dim])?;
        Ok(add(&hidden, &multiply(&gate_mlp, &ff)?)?)
    }
}

// ----------------------------------------------------------------------------------------------
// Full trunk.
// ----------------------------------------------------------------------------------------------

/// SANA Linear-DiT trunk (`SanaTransformer2DModel`).
pub struct SanaTransformer {
    cfg: SanaTransformerConfig,
    patch_embed: Conv, // proj: in → inner (kernel/stride = patch_size)
    // timestep path (AdaLayerNormSingle.emb + .linear, or — Sprint — the combined
    // timestep+guidance embedder, see `guidance_embedder`)
    ts_embedder_1: Linear,
    ts_embedder_2: Linear,
    time_linear: Linear, // → 6·inner
    /// Sprint (sc-8490): the extra guidance embedder (`SanaCombinedTimestepGuidanceEmbeddings`). The
    /// embedded guidance scalar runs through the same `Timesteps(256)` sincos projection as the
    /// timestep, then this two-linear MLP, and is summed into the timestep conditioning. `None` for
    /// base SANA (`AdaLayerNormSingle`).
    guidance_embedder: Option<(Linear, Linear)>,
    // caption path
    caption_proj_1: Linear,
    caption_proj_2: Linear,
    caption_norm: Array, // RMSNorm weight [inner]
    blocks: Vec<SanaBlock>,
    scale_shift_table: Array, // [2, inner] (output modulated norm)
    proj_out: Linear,
}

impl SanaTransformer {
    pub fn from_weights(w: &Weights, cfg: SanaTransformerConfig) -> Result<Self> {
        let p = cfg.patch_size;
        let patch_embed = Conv::load(w, "patch_embed.proj", p, 0, 1, true)?;
        let mut blocks = Vec::with_capacity(cfg.num_layers as usize);
        for i in 0..cfg.num_layers {
            blocks.push(SanaBlock::load(
                w,
                &format!("transformer_blocks.{i}"),
                &cfg,
            )?);
        }
        // Sprint's guidance variant (`SanaCombinedTimestepGuidanceEmbeddings`) drops the `.emb.`
        // nesting AdaLayerNormSingle introduces and adds a parallel `guidance_embedder`.
        let (ts1_key, ts2_key, guidance_embedder) = if cfg.guidance_embeds {
            (
                "time_embed.timestep_embedder.linear_1",
                "time_embed.timestep_embedder.linear_2",
                Some((
                    Linear::load(w, "time_embed.guidance_embedder.linear_1", true)?,
                    Linear::load(w, "time_embed.guidance_embedder.linear_2", true)?,
                )),
            )
        } else {
            (
                "time_embed.emb.timestep_embedder.linear_1",
                "time_embed.emb.timestep_embedder.linear_2",
                None,
            )
        };
        Ok(Self {
            patch_embed,
            ts_embedder_1: Linear::load(w, ts1_key, true)?,
            ts_embedder_2: Linear::load(w, ts2_key, true)?,
            time_linear: Linear::load(w, "time_embed.linear", true)?,
            guidance_embedder,
            caption_proj_1: Linear::load(w, "caption_projection.linear_1", true)?,
            caption_proj_2: Linear::load(w, "caption_projection.linear_2", true)?,
            caption_norm: w.require("caption_norm.weight")?.clone(),
            blocks,
            scale_shift_table: w.require("scale_shift_table")?.clone(),
            proj_out: Linear::load(w, "proj_out", true)?,
            cfg,
        })
    }

    /// Forward one denoise step.
    ///
    /// * `latent_nchw` — `[B, in_channels, H, W]` (channels-first, diffusers-native).
    /// * `caption` — `[B, M, caption_channels]` caption embedding (M = 300 for SANA-1.6B).
    /// * `timestep` — `[B]` (or `[1]`) scalar timestep(s).
    ///
    /// Returns the noise prediction `[B, out_channels, H, W]` (channels-first), where
    /// `out_channels == 32` matches the DC-AE f32c32 latent so the output feeds
    /// [`crate::dc_ae::DcAeDecoder::decode`] directly (sc-8489 composition).
    pub fn forward(&self, latent_nchw: &Array, caption: &Array, timestep: &Array) -> Result<Array> {
        self.forward_with_guidance(latent_nchw, caption, timestep, None)
    }

    /// [`Self::forward`] with an optional **embedded guidance scalar** (SANA-Sprint, sc-8490).
    ///
    /// * `guidance` — `[B]` (or `[1]`) the CFG-free guidance scalar (already multiplied by the
    ///   `guidance_embeds_scale` by the caller). `Some` only for a Sprint-config trunk
    ///   (`guidance_embeds = true`); `None` runs the base AdaLN-single path. Sprint feeds the scale
    ///   as an embedded conditioning input — it is NOT classifier-free guidance (no uncond forward).
    pub fn forward_with_guidance(
        &self,
        latent_nchw: &Array,
        caption: &Array,
        timestep: &Array,
        guidance: Option<&Array>,
    ) -> Result<Array> {
        let cfg = &self.cfg;
        let dim = cfg.inner_dim();
        let lsh = latent_nchw.shape();
        let (b, height, width) = (lsh[0], lsh[2], lsh[3]);
        let p = cfg.patch_size;
        let (ph, pw) = (height / p, width / p);
        let dt = latent_nchw.dtype();

        // 1. Patch embed (NHWC). [B,C,H,W] → NHWC → conv → [B,ph,pw,dim] → tokens [B,N,dim].
        let x = latent_nchw.transpose_axes(&[0, 2, 3, 1])?; // NHWC
        let x = self.patch_embed.forward(&x)?; // [B,ph,pw,dim]
        let mut hidden = x.reshape(&[b, ph * pw, dim])?;

        // 2. Timestep embedding → embedded_timestep [B,dim] and modulation temb [B,6·dim].
        let ts_proj = timestep_sincos(timestep, 256, 10_000.0, 0.0)?.as_dtype(dt)?; // [B,256]
        let timesteps_emb = self
            .ts_embedder_2
            .forward(&silu(&self.ts_embedder_1.forward(&ts_proj)?)?)?; // [B,dim]
                                                                       // Sprint: conditioning = timesteps_emb + guidance_emb (the guidance scalar through the same
                                                                       // sincos(256) projection + a parallel MLP). embedded_timestep (the output-modnorm input) is
                                                                       // this combined conditioning, exactly as diffusers `SanaCombinedTimestepGuidanceEmbeddings`.
        let emb = match (&self.guidance_embedder, guidance) {
            (Some((g1, g2)), Some(g)) => {
                let g_proj = timestep_sincos(g, 256, 10_000.0, 0.0)?.as_dtype(dt)?;
                let guidance_emb = g2.forward(&silu(&g1.forward(&g_proj)?)?)?;
                add(&timesteps_emb, &guidance_emb)?
            }
            _ => timesteps_emb,
        };
        let temb = self.time_linear.forward(&silu(&emb)?)?; // [B,6·dim]

        // 3. Caption projection + RMSNorm.
        let cap = self.caption_proj_1.forward(caption)?;
        let cap = self.caption_proj_2.forward(&gelu_tanh(&cap)?)?;
        let cap = cap.reshape(&[b, -1, dim])?;
        let caption = rms_norm(&cap, &self.caption_norm, cfg.caption_norm_eps)?;

        // 4. Transformer blocks.
        for block in &self.blocks {
            hidden = block.forward(&hidden, &caption, &temb, ph, pw)?;
        }

        // 5. Output: SanaModulatedNorm(embedded_timestep) → proj_out → unpatchify.
        let ss = self.scale_shift_table.reshape(&[1, 2, dim])?;
        let modg = add(&ss, &emb.reshape(&[b, 1, dim])?)?; // [B,2,dim]
        let parts = split_sections(&modg, &[1], 1)?; // 2 × [B,1,dim]
        let shift = parts[0].reshape(&[b, 1, dim])?;
        let scale = parts[1].reshape(&[b, 1, dim])?;
        let normed = layer_norm(&hidden, None, None, 1e-6)?;
        let one = scalar(1.0).as_dtype(scale.dtype())?;
        let hidden = add(&multiply(&normed, &add(&scale, &one)?)?, &shift)?;

        let out = self.proj_out.forward(&hidden)?; // [B,N, p·p·out_channels]
                                                   // unpatchify: [B,ph,pw,p,p,out_c] → permute(0,5,1,3,2,4) → [B,out_c,ph·p,pw·p].
        let oc = cfg.out_channels;
        let out = out.reshape(&[b, ph, pw, p, p, oc])?;
        let out = out.transpose_axes(&[0, 5, 1, 3, 2, 4])?;
        Ok(out.reshape(&[b, oc, ph * p, pw * p])?)
    }
}
