//! The SCAIL-2 DiT forward (`SCAIL2Model.forward`, upstream `wan/modules/model_scail2.py`).
//!
//! SCAIL-2 is a Wan2.1-14B **I2V** diffusion backbone with Bernini-family **packed-token**
//! conditioning. Each denoise step assembles one self-attention sequence from four token chunks —
//! `[additional_ref | ref | video | pose]` — embeds them through three Conv3d patch stems (latent /
//! pose / 28-channel color-coded mask, the mask & pose embeds *added* onto the latent embeds), applies
//! a per-chunk 3-axis RoPE (the [`crate::rope::ScailRope`] shifts; `replace_flag` toggles the
//! reference H-shift between animation and cross-identity replacement), runs the Wan blocks with
//! **I2V image cross-attention** (CLIP image tokens via `k_img`/`v_img` alongside the UMT5 text
//! tokens), and finally keeps only the video tokens (`unpatchify` at `offset = additional_ref + ref`).
//!
//! This module owns the DiT only. It reuses mlx-gen primitives — [`AdaptableLinear`],
//! [`mlx_gen_wan::rope::rope_apply`], [`mlx_gen_wan::patchify`], `mlx_rs::fast` norms/SDPA — and loads
//! the raw `SCAIL2Model` PyTorch parameter names directly from the converted snapshot
//! (`patch_embedding{,_pose,_mask}`, `blocks.{i}.{self_attn,cross_attn,ffn,...}`, `img_emb.proj.*`,
//! `time_*`, `text_embedding.*`, `head.*`). The Conv3d patch weights `[out, in, 1, 2, 2]` are read as
//! `[out, in·4]` Linears (the stride==kernel patch embed is exactly a patchify + linear; the patchify
//! feature order `(c, pt, ph, pw)` matches the Conv weight flatten order).
//!
//! Activations flow in **f32** through the norms / adaLN modulation / gated residuals (the upstream
//! `amp.autocast(float32)` islands); the matmul-heavy projections run in `compute_dtype` (f32 for the
//! parity gate, bf16 in production). RoPE is applied in f32.

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::array::scalar;
use mlx_gen::nn::{gelu_exact, gelu_tanh};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_gen_wan::config::WanQuant;
use mlx_gen_wan::patchify::{patchify, unpatchify};
use mlx_gen_wan::rope::rope_apply;
use mlx_rs::fast::{layer_norm, rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, concatenate_axis, multiply, sigmoid, split};
use mlx_rs::{Array, Dtype};

use crate::config::Scail2Config;
use crate::rope::ScailRope;

/// `nn.LayerNorm` default eps (the `img_emb` MLPProj LayerNorms). The DiT's own `WanLayerNorm` uses
/// `cfg.eps` (1e-6) instead.
const IMG_LN_EPS: f32 = 1e-5;

// ---- small dtype/elementwise helpers (f32 islands) ----------------------------------------------

fn to_f32(x: &Array) -> Result<Array> {
    Ok(x.as_dtype(Dtype::Float32)?)
}

fn silu(x: &Array) -> Result<Array> {
    Ok(multiply(x, &sigmoid(x)?)?)
}

/// Non-affine LayerNorm — `WanLayerNorm(elementwise_affine=False)`.
fn ln(x: &Array, eps: f32) -> Result<Array> {
    Ok(layer_norm(x, None, None, eps)?)
}

/// adaLN affine `m·(1+scale)+shift`.
fn modulate(m: &Array, scale: &Array, shift: &Array) -> Result<Array> {
    Ok(add(&multiply(m, &add(scale, scalar(1.0))?)?, shift)?)
}

/// Gated residual `x + y·gate`.
fn gated(x: &Array, y: &Array, gate: &Array) -> Result<Array> {
    Ok(add(x, &multiply(y, gate)?)?)
}

// ---- weight loaders -----------------------------------------------------------------------------

/// A biased `[out, in]` Linear (`nn.Linear`) → dense [`AdaptableLinear`].
fn load_lin(w: &Weights, prefix: &str) -> Result<AdaptableLinear> {
    Ok(AdaptableLinear::dense(
        w.require(&format!("{prefix}.weight"))?.clone(),
        Some(w.require(&format!("{prefix}.bias"))?.clone()),
    ))
}

/// Like [`load_lin`] but **pre-quantized-snapshot aware** (sc-5445), mirroring the Wan DiT
/// `load_linear`. When the snapshot carries a `quantization` manifest (`quant` is `Some`) *and* this
/// Linear's packed `{prefix}.scales` is present on disk, build a quantized base **directly** from the
/// packed `weight` (u32) / `scales` / `biases` (+ the unpacked dense `bias`) — no dense bf16 weight is
/// ever materialized, which is what keeps the load-time memory floor low (the whole point of shipping
/// a pre-quantized snapshot vs. quantizing at load). Otherwise (dense snapshot, or a Linear the
/// `convert::quantize_scail2_transformer` predicate left dense) fall back to the dense path. `.scales`
/// presence is the per-Linear signal — only the predicate Linears are packed.
fn load_lin_q(w: &Weights, prefix: &str, quant: Option<WanQuant>) -> Result<AdaptableLinear> {
    if let (Some(q), Some(scales)) = (quant, w.get(&format!("{prefix}.scales"))) {
        return Ok(AdaptableLinear::from_quantized_parts(
            w.require(&format!("{prefix}.weight"))?.clone(),
            scales.clone(),
            w.require(&format!("{prefix}.biases"))?.clone(),
            w.get(&format!("{prefix}.bias")).cloned(),
            q.group_size,
            q.bits,
        ));
    }
    load_lin(w, prefix)
}

/// A `nn.Conv3d(in, out, kernel=stride=patch)` patch embed → a `[out, in·∏patch]` dense Linear (the
/// stride==kernel non-overlapping conv is a patchify + linear; the patchify feature order
/// `(c, pt, ph, pw)` matches the Conv weight flatten `(in, kt, kh, kw)`).
fn load_conv(w: &Weights, prefix: &str) -> Result<AdaptableLinear> {
    let wgt = w.require(&format!("{prefix}.weight"))?;
    let s = wgt.shape(); // [out, in, kt, kh, kw]
    let out = s[0];
    let infeat = s[1] * s[2] * s[3] * s[4];
    let wgt2 = wgt.reshape(&[out, infeat])?;
    let bias = w.require(&format!("{prefix}.bias"))?.clone();
    Ok(AdaptableLinear::dense(wgt2, Some(bias)))
}

fn req(w: &Weights, name: &str) -> Result<Array> {
    Ok(w.require(name)?.clone())
}

/// A weight used inside an f32 island (modulation tables, affine-LN weights) — kept f32.
fn req_f32(w: &Weights, name: &str) -> Result<Array> {
    to_f32(w.require(name)?)
}

// ---- attention ----------------------------------------------------------------------------------

/// Push a linear's quantized packs (`wq`/`scales`/`biases` + optional `bias`) into `out` for the
/// eval-to-free pass (mirrors the Wan DiT `push_quant_arrays`); a dense linear contributes nothing.
fn push_quant_arrays<'a>(lin: &'a AdaptableLinear, out: &mut Vec<&'a Array>) {
    if let Some((wq, scales, biases, bias, _, _)) = lin.quantized_params() {
        out.push(wq);
        out.push(scales);
        out.push(biases);
        if let Some(bias) = bias {
            out.push(bias);
        }
    }
}

/// Wan self-attention with qk-RMSNorm and 3-axis RoPE, over the full packed sequence.
struct SelfAttn {
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    o: AdaptableLinear,
    norm_q: Array,
    norm_k: Array,
    n: i32,
    d: i32,
    scale: f32,
    eps: f32,
}

impl SelfAttn {
    fn load(w: &Weights, prefix: &str, cfg: &Scail2Config) -> Result<Self> {
        let head_dim = cfg.wan.head_dim();
        let q = cfg.wan.quantization;
        Ok(Self {
            q: load_lin_q(w, &format!("{prefix}.q"), q)?,
            k: load_lin_q(w, &format!("{prefix}.k"), q)?,
            v: load_lin_q(w, &format!("{prefix}.v"), q)?,
            o: load_lin_q(w, &format!("{prefix}.o"), q)?,
            norm_q: req(w, &format!("{prefix}.norm_q.weight"))?,
            norm_k: req(w, &format!("{prefix}.norm_k.weight"))?,
            n: cfg.wan.num_heads as i32,
            d: head_dim as i32,
            scale: (head_dim as f32).powf(-0.5),
            eps: cfg.wan.eps as f32,
        })
    }

    /// Quantize the four projections (q/k/v/o) to Q4/Q8 in place (mirrors the Wan DiT). The
    /// qk-RMSNorm weights stay dense (small + precision-sensitive).
    fn quantize(&mut self, bits: i32, group: Option<i32>) -> Result<()> {
        self.q.quantize(bits, group)?;
        self.k.quantize(bits, group)?;
        self.v.quantize(bits, group)?;
        self.o.quantize(bits, group)?;
        Ok(())
    }

    /// Collect this attention's quantized packs for the eval-to-free pass.
    fn push_quant_arrays<'a>(&'a self, out: &mut Vec<&'a Array>) {
        push_quant_arrays(&self.q, out);
        push_quant_arrays(&self.k, out);
        push_quant_arrays(&self.v, out);
        push_quant_arrays(&self.o, out);
    }

    /// Resolve a LoRA target projection (`q`/`k`/`v`/`o`) to its [`AdaptableLinear`] (sc-5451).
    fn adaptable_mut(&mut self, proj: &str) -> Option<&mut AdaptableLinear> {
        match proj {
            "q" => Some(&mut self.q),
            "k" => Some(&mut self.k),
            "v" => Some(&mut self.v),
            "o" => Some(&mut self.o),
            _ => None,
        }
    }

    /// `x`: `[1, L, dim]` (f32, already adaLN-modulated). `cos`/`sin`: `[L, 1, half_d]` (f32).
    fn forward(&self, x: &Array, cos: &Array, sin: &Array, cdt: Dtype) -> Result<Array> {
        let (b, s) = (x.shape()[0], x.shape()[1]);
        let (n, d) = (self.n, self.d);
        let xw = x.as_dtype(cdt)?;
        let q = rms_norm(&self.q.forward(&xw)?, &self.norm_q, self.eps)?;
        let k = rms_norm(&self.k.forward(&xw)?, &self.norm_k, self.eps)?;
        // RoPE in f32, then back to compute dtype; transpose to [b, n, s, d] for SDPA.
        let q = rope_apply(&to_f32(&q)?.reshape(&[b, s, n, d])?, cos, sin)?
            .as_dtype(cdt)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = rope_apply(&to_f32(&k)?.reshape(&[b, s, n, d])?, cos, sin)?
            .as_dtype(cdt)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = self
            .v
            .forward(&xw)?
            .reshape(&[b, s, n, d])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let out = scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?;
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, n * d])?;
        self.o.forward(&out)
    }
}

/// Wan **I2V** cross-attention: text tokens through `k`/`v`, CLIP image tokens through
/// `k_img`/`v_img`; the two attention outputs are summed before the output projection.
struct CrossAttnI2V {
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    o: AdaptableLinear,
    k_img: AdaptableLinear,
    v_img: AdaptableLinear,
    norm_q: Array,
    norm_k: Array,
    norm_k_img: Array,
    n: i32,
    d: i32,
    scale: f32,
    eps: f32,
}

impl CrossAttnI2V {
    fn load(w: &Weights, prefix: &str, cfg: &Scail2Config) -> Result<Self> {
        let head_dim = cfg.wan.head_dim();
        let q = cfg.wan.quantization;
        Ok(Self {
            q: load_lin_q(w, &format!("{prefix}.q"), q)?,
            k: load_lin_q(w, &format!("{prefix}.k"), q)?,
            v: load_lin_q(w, &format!("{prefix}.v"), q)?,
            o: load_lin_q(w, &format!("{prefix}.o"), q)?,
            k_img: load_lin_q(w, &format!("{prefix}.k_img"), q)?,
            v_img: load_lin_q(w, &format!("{prefix}.v_img"), q)?,
            norm_q: req(w, &format!("{prefix}.norm_q.weight"))?,
            norm_k: req(w, &format!("{prefix}.norm_k.weight"))?,
            norm_k_img: req(w, &format!("{prefix}.norm_k_img.weight"))?,
            n: cfg.wan.num_heads as i32,
            d: head_dim as i32,
            scale: (head_dim as f32).powf(-0.5),
            eps: cfg.wan.eps as f32,
        })
    }

    /// Quantize all six projections (text q/k/v/o + I2V image k_img/v_img) to Q4/Q8 in place. The
    /// three qk-RMSNorm weights stay dense.
    fn quantize(&mut self, bits: i32, group: Option<i32>) -> Result<()> {
        self.q.quantize(bits, group)?;
        self.k.quantize(bits, group)?;
        self.v.quantize(bits, group)?;
        self.o.quantize(bits, group)?;
        self.k_img.quantize(bits, group)?;
        self.v_img.quantize(bits, group)?;
        Ok(())
    }

    /// Collect this cross-attention's quantized packs for the eval-to-free pass.
    fn push_quant_arrays<'a>(&'a self, out: &mut Vec<&'a Array>) {
        push_quant_arrays(&self.q, out);
        push_quant_arrays(&self.k, out);
        push_quant_arrays(&self.v, out);
        push_quant_arrays(&self.o, out);
        push_quant_arrays(&self.k_img, out);
        push_quant_arrays(&self.v_img, out);
    }

    /// Resolve a LoRA target projection (the text `q`/`k`/`v`/`o` or the I2V image `k_img`/`v_img`)
    /// to its [`AdaptableLinear`] (sc-5451).
    fn adaptable_mut(&mut self, proj: &str) -> Option<&mut AdaptableLinear> {
        match proj {
            "q" => Some(&mut self.q),
            "k" => Some(&mut self.k),
            "v" => Some(&mut self.v),
            "o" => Some(&mut self.o),
            "k_img" => Some(&mut self.k_img),
            "v_img" => Some(&mut self.v_img),
            _ => None,
        }
    }

    /// `x`: `[1, L, dim]` (f32). `text_ctx`: `[1, L_text, dim]`. `img_ctx`: `[1, L_img, dim]`.
    fn forward(&self, x: &Array, text_ctx: &Array, img_ctx: &Array, cdt: Dtype) -> Result<Array> {
        let (b, s) = (x.shape()[0], x.shape()[1]);
        let (n, d) = (self.n, self.d);
        let heads = |t: &Array| -> Result<Array> {
            Ok(t.reshape(&[b, -1, n, d])?.transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = heads(&rms_norm(
            &self.q.forward(&x.as_dtype(cdt)?)?,
            &self.norm_q,
            self.eps,
        )?)?;
        let tc = text_ctx.as_dtype(cdt)?;
        let ic = img_ctx.as_dtype(cdt)?;
        let k = heads(&rms_norm(&self.k.forward(&tc)?, &self.norm_k, self.eps)?)?;
        let v = heads(&self.v.forward(&tc)?)?;
        let k_img = heads(&rms_norm(
            &self.k_img.forward(&ic)?,
            &self.norm_k_img,
            self.eps,
        )?)?;
        let v_img = heads(&self.v_img.forward(&ic)?)?;
        let flat = |o: Array| -> Result<Array> {
            Ok(o.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, n * d])?)
        };
        let x_txt = flat(scaled_dot_product_attention(
            &q, &k, &v, self.scale, None, None,
        )?)?;
        let x_img = flat(scaled_dot_product_attention(
            &q, &k_img, &v_img, self.scale, None, None,
        )?)?;
        self.o.forward(&add(&x_txt, &x_img)?)
    }
}

/// One Wan attention block: adaLN-6vec modulation → self-attn (gated residual) → affine-LN +
/// I2V cross-attn → adaLN FFN (gated residual).
struct Block {
    modulation: Array, // [1, 6, dim] f32
    self_attn: SelfAttn,
    cross: CrossAttnI2V,
    norm3_w: Array,
    norm3_b: Array,
    ffn0: AdaptableLinear,
    ffn2: AdaptableLinear,
    eps: f32,
}

impl Block {
    fn load(w: &Weights, i: usize, cfg: &Scail2Config) -> Result<Self> {
        let p = format!("blocks.{i}");
        Ok(Self {
            modulation: req_f32(w, &format!("{p}.modulation"))?,
            self_attn: SelfAttn::load(w, &format!("{p}.self_attn"), cfg)?,
            cross: CrossAttnI2V::load(w, &format!("{p}.cross_attn"), cfg)?,
            norm3_w: req_f32(w, &format!("{p}.norm3.weight"))?,
            norm3_b: req_f32(w, &format!("{p}.norm3.bias"))?,
            ffn0: load_lin_q(w, &format!("{p}.ffn.0"), cfg.wan.quantization)?,
            ffn2: load_lin_q(w, &format!("{p}.ffn.2"), cfg.wan.quantization)?,
            eps: cfg.wan.eps as f32,
        })
    }

    /// Quantize this block's self/cross-attention projections + FFN (`ffn.0`/`ffn.2`) to Q4/Q8 in
    /// place. The modulation table and the affine `norm3` LayerNorm stay dense.
    fn quantize(&mut self, bits: i32, group: Option<i32>) -> Result<()> {
        self.self_attn.quantize(bits, group)?;
        self.cross.quantize(bits, group)?;
        self.ffn0.quantize(bits, group)?;
        self.ffn2.quantize(bits, group)?;
        Ok(())
    }

    /// Collect this block's quantized packs for the eval-to-free pass.
    fn push_quant_arrays<'a>(&'a self, out: &mut Vec<&'a Array>) {
        self.self_attn.push_quant_arrays(out);
        self.cross.push_quant_arrays(out);
        push_quant_arrays(&self.ffn0, out);
        push_quant_arrays(&self.ffn2, out);
    }

    /// Resolve a LoRA target under this block (`self_attn.*` / `cross_attn.*` / `ffn.0` / `ffn.2`,
    /// the path tail after `blocks.{i}.`) to its [`AdaptableLinear`] (sc-5451).
    fn adaptable_mut(&mut self, sub: &[&str]) -> Option<&mut AdaptableLinear> {
        match sub {
            ["self_attn", proj] => self.self_attn.adaptable_mut(proj),
            ["cross_attn", proj] => self.cross.adaptable_mut(proj),
            ["ffn", "0"] => Some(&mut self.ffn0),
            ["ffn", "2"] => Some(&mut self.ffn2),
            _ => None,
        }
    }

    /// `x`: `[1, L, dim]` (f32). `e0`: `[1, 6, dim]` (f32, time modulation).
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Array,
        e0: &Array,
        text_ctx: &Array,
        img_ctx: &Array,
        cos: &Array,
        sin: &Array,
        cdt: Dtype,
    ) -> Result<Array> {
        let m = add(&self.modulation, e0)?; // [1, 6, dim]
        let p = split(&m, 6, 1)?; // 6 × [1, 1, dim]
                                  // self-attention
        let x_mod = modulate(&ln(x, self.eps)?, &p[1], &p[0])?;
        let y = self.self_attn.forward(&x_mod, cos, sin, cdt)?;
        let x = gated(x, &to_f32(&y)?, &p[2])?;
        // cross-attention (affine LN, no modulation)
        let x_cross = layer_norm(&x, Some(&self.norm3_w), Some(&self.norm3_b), self.eps)?;
        let cx = self.cross.forward(&x_cross, text_ctx, img_ctx, cdt)?;
        let x = add(&x, &to_f32(&cx)?)?;
        // feed-forward
        let x_mod = modulate(&ln(&x, self.eps)?, &p[4], &p[3])?;
        let y = self.ffn0.forward(&x_mod.as_dtype(cdt)?)?;
        let y = self.ffn2.forward(&gelu_tanh(&y)?)?;
        gated(&x, &to_f32(&y)?, &p[5])
    }
}

/// The loaded SCAIL-2 DiT.
pub struct Scail2Dit {
    patch_embedding: AdaptableLinear,
    patch_embedding_pose: AdaptableLinear,
    patch_embedding_mask: AdaptableLinear,
    text_embedding_0: AdaptableLinear,
    text_embedding_2: AdaptableLinear,
    time_embedding_0: AdaptableLinear,
    time_embedding_2: AdaptableLinear,
    time_projection: AdaptableLinear,
    img_ln0_w: Array,
    img_ln0_b: Array,
    img_emb_1: AdaptableLinear,
    img_emb_3: AdaptableLinear,
    img_ln4_w: Array,
    img_ln4_b: Array,
    blocks: Vec<Block>,
    head_modulation: Array, // [1, 2, dim] f32
    head: AdaptableLinear,
    rope: ScailRope,
    cfg: Scail2Config,
    compute_dtype: Dtype,
}

/// All conditioning tensors for one denoise-step forward. Spatial dims are latent (`vae_stride`-down)
/// dims; channel counts are `vae_z_dim` (16) for latents and `mask_dim` (28) for masks.
pub struct Scail2Inputs<'a> {
    /// Noisy video latent `[16, T, H, W]`.
    pub x: &'a Array,
    /// Reference-character latent `[16, 1, H, W]`.
    pub ref_latent: &'a Array,
    /// Reference mask latent `[28, 1+T, H, W]`.
    pub ref_masks: &'a Array,
    /// Driving-pose latent `[16, T, H/2, W/2]` (half spatial res).
    pub pose_latent: &'a Array,
    /// Driving-mask latent `[28, T, H/2, W/2]`.
    pub driving_masks: &'a Array,
    /// Clean-history mask `[4, T, H, W]` (segment > 0); `None` appends the i2v zero-mask.
    pub history_mask: Option<&'a Array>,
    /// Extra-character latents `[16, n, H, W]` (multi-reference); `None` for single reference.
    pub additional_ref_latent: Option<&'a Array>,
    /// Extra-character mask latents `[28, n, H, W]`; required iff `additional_ref_latent` is set.
    pub additional_ref_masks: Option<&'a Array>,
    /// CLIP image features `[1, 257, 1280]` from the open-CLIP XLM-RoBERTa ViT-H/14 visual tower.
    pub clip_fea: &'a Array,
    /// UMT5 text embeddings `[L_text, 4096]`.
    pub context: &'a Array,
    /// Diffusion timestep.
    pub t: f32,
    /// `true` = cross-identity replacement (ref H-shift 120), `false` = animation (H-shift 0).
    pub replace_flag: bool,
}

impl Scail2Dit {
    /// Load the DiT from a `Weights` view of the converted `dit.safetensors` (raw `SCAIL2Model`
    /// parameter names). `compute_dtype` defaults to f32; call [`Scail2Dit::set_compute_dtype`] for
    /// bf16 production inference.
    pub fn from_weights(w: &Weights, cfg: &Scail2Config) -> Result<Self> {
        let mut blocks = Vec::with_capacity(cfg.wan.num_layers);
        for i in 0..cfg.wan.num_layers {
            blocks.push(Block::load(w, i, cfg)?);
        }
        Ok(Self {
            patch_embedding: load_conv(w, "patch_embedding")?,
            patch_embedding_pose: load_conv(w, "patch_embedding_pose")?,
            patch_embedding_mask: load_conv(w, "patch_embedding_mask")?,
            text_embedding_0: load_lin(w, "text_embedding.0")?,
            text_embedding_2: load_lin(w, "text_embedding.2")?,
            time_embedding_0: load_lin(w, "time_embedding.0")?,
            time_embedding_2: load_lin(w, "time_embedding.2")?,
            time_projection: load_lin(w, "time_projection.1")?,
            img_ln0_w: req_f32(w, "img_emb.proj.0.weight")?,
            img_ln0_b: req_f32(w, "img_emb.proj.0.bias")?,
            img_emb_1: load_lin(w, "img_emb.proj.1")?,
            img_emb_3: load_lin(w, "img_emb.proj.3")?,
            img_ln4_w: req_f32(w, "img_emb.proj.4.weight")?,
            img_ln4_b: req_f32(w, "img_emb.proj.4.bias")?,
            blocks,
            head_modulation: req_f32(w, "head.modulation")?,
            head: load_lin(w, "head.head")?,
            rope: ScailRope::new(cfg.wan.head_dim()),
            cfg: cfg.clone(),
            compute_dtype: Dtype::Float32,
        })
    }

    /// Set the matmul compute dtype (f32 for parity, bf16 for production).
    pub fn set_compute_dtype(&mut self, dt: Dtype) {
        self.compute_dtype = dt;
    }

    /// Quantize the transformer-only attention + FFN Linears to Q4/Q8 **in place** (sc-5445),
    /// mirroring the Wan DiT's `_quantize_predicate` surface: every block's self/cross-attention
    /// `q/k/v/o` (+ the I2V `k_img`/`v_img`) and `ffn.0`/`ffn.2`. The patch/text/time/image
    /// embeddings, `time_projection`, modulation tables, qk/`norm3`/LayerNorm norms, and the output
    /// head stay dense (small + precision-sensitive — the reference skips them). `group` is the
    /// quantization group size (`None` ⇒ the mflux/reference default of 64). Compute then runs
    /// `quantized_matmul` (fp32-accumulate) on the bf16 activations the blocks already feed.
    pub fn quantize(&mut self, bits: i32, group: Option<i32>) -> Result<()> {
        for block in &mut self.blocks {
            block.quantize(bits, group)?;
        }
        // Force-materialize the quantized packs now so the per-weight bf16 dequant transient frees
        // here instead of lazily at the first forward (matches the Wan DiT; the win is the load-time
        // peak — the packed Q4/Q8 arrays are what stays resident, the bf16 source is released).
        let mut arrays: Vec<&Array> = Vec::new();
        for block in &self.blocks {
            block.push_quant_arrays(&mut arrays);
        }
        if !arrays.is_empty() {
            mlx_rs::transforms::eval(arrays)?;
        }
        Ok(())
    }

    /// Number of transformer blocks (40 for the 14B).
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Sinusoidal timestep embedding → `(e, e0)`: `e` `[1, dim]` (head modulation), `e0` `[1, 6, dim]`
    /// (block modulation). Built in f64 then cast to f32, matching upstream `sinusoidal_embedding_1d`.
    fn time_embed(&self, t: f32) -> Result<(Array, Array)> {
        let freq_dim = self.cfg.wan.freq_dim;
        let half = freq_dim / 2;
        let mut emb = vec![0f32; freq_dim];
        for j in 0..half {
            let ang = (t as f64) * 10000f64.powf(-(j as f64) / half as f64);
            emb[j] = ang.cos() as f32;
            emb[half + j] = ang.sin() as f32;
        }
        let sin_emb = Array::from_slice(&emb, &[1, freq_dim as i32]);
        let e = self
            .time_embedding_2
            .forward(&silu(&self.time_embedding_0.forward(&sin_emb)?)?)?;
        let dim = self.cfg.wan.dim as i32;
        let e0 = self
            .time_projection
            .forward(&silu(&e)?)?
            .reshape(&[1, 6, dim])?;
        Ok((e, e0))
    }

    /// UMT5 text embeddings `[L, 4096]` → `[1, text_len, dim]` (zero-padded to `text_len`).
    fn embed_text(&self, context: &Array) -> Result<Array> {
        let cdt = self.compute_dtype;
        let text_len = self.cfg.wan.text_len as i32;
        let l = context.shape()[0];
        let td = context.shape()[1];
        let ctx = if l < text_len {
            let pad = Array::zeros::<f32>(&[text_len - l, td])?.as_dtype(context.dtype())?;
            concatenate_axis(&[context, &pad], 0)?
        } else {
            context.clone()
        };
        let ctx = ctx.reshape(&[1, text_len, td])?;
        let h = gelu_tanh(&self.text_embedding_0.forward(&ctx.as_dtype(cdt)?)?)?;
        self.text_embedding_2.forward(&h)
    }

    /// CLIP features `[1, 257, 1280]` → image context `[1, 257, dim]` via the MLPProj
    /// (LayerNorm → Linear → exact-GELU → Linear → LayerNorm).
    fn embed_img(&self, clip_fea: &Array) -> Result<Array> {
        let cdt = self.compute_dtype;
        let h = layer_norm(
            &to_f32(clip_fea)?,
            Some(&self.img_ln0_w),
            Some(&self.img_ln0_b),
            IMG_LN_EPS,
        )?;
        let h = gelu_exact(&self.img_emb_1.forward(&h.as_dtype(cdt)?)?)?;
        let h = self.img_emb_3.forward(&h)?;
        let h = layer_norm(
            &to_f32(&h)?,
            Some(&self.img_ln4_w),
            Some(&self.img_ln4_b),
            IMG_LN_EPS,
        )?;
        h.as_dtype(cdt).map_err(Into::into)
    }

    /// Modulated output head: `[1, L, dim]` → `[1, L, out_dim·∏patch]`.
    fn apply_head(&self, x: &Array, e: &Array) -> Result<Array> {
        let dim = self.cfg.wan.dim as i32;
        let m = add(&self.head_modulation, &e.reshape(&[1, 1, dim])?)?; // [1, 2, dim]
        let p = split(&m, 2, 1)?;
        let x_mod = modulate(&ln(x, self.cfg.wan.eps as f32)?, &p[1], &p[0])?;
        self.head.forward(&x_mod)
    }

    /// One denoise-step velocity prediction → `[16, T, H, W]` (the upstream `forward` minus the list
    /// batching, single sample).
    pub fn forward(&self, inp: &Scail2Inputs) -> Result<Array> {
        let cdt = self.compute_dtype;
        let cfg = &self.cfg;
        let ps = cfg.wan.patch_size;
        let i2v = cfg.i2v_mask_dim as i32;
        let dim = cfg.wan.dim as i32;

        let tt = inp.x.shape()[1];
        let hh = inp.x.shape()[2];
        let ww = inp.x.shape()[3];

        // --- append the i2v binary-mask channels (in_dim 20 = 16 + 4) ---
        let x20 = match inp.history_mask {
            Some(hm) => concatenate_axis(&[inp.x, hm], 0)?,
            None => concatenate_axis(&[inp.x, &Array::zeros::<f32>(&[i2v, tt, hh, ww])?], 0)?,
        };
        let ref20 = concatenate_axis(
            &[inp.ref_latent, &Array::ones::<f32>(&[i2v, 1, hh, ww])?],
            0,
        )?;
        let pose_t = inp.pose_latent.shape()[1];
        let pose_h = inp.pose_latent.shape()[2];
        let pose_w = inp.pose_latent.shape()[3];
        let pose20 = concatenate_axis(
            &[
                inp.pose_latent,
                &Array::ones::<f32>(&[i2v, pose_t, pose_h, pose_w])?,
            ],
            0,
        )?;

        // --- patch grids / chunk lengths (patch (1,2,2)) ---
        let rope_t = (tt / ps.0 as i32) as usize;
        let rope_h = (hh / ps.1 as i32) as usize;
        let rope_w = (ww / ps.2 as i32) as usize;
        let ref_length = rope_h * rope_w;
        let seq_length = rope_t * rope_h * rope_w;
        let h_shift = if inp.replace_flag {
            cfg.replace_h_shift
        } else {
            0
        };
        let base_video_shift = 1usize;

        // --- patch-embed stems (ref+video share patch_embedding; mask/pose added) ---
        let refvid = concatenate_axis(&[&ref20, &x20], 1)?; // [20, 1+T, H, W]
        let (rv_tok, _) = patchify(&refvid, ps)?;
        let (rm_tok, _) = patchify(inp.ref_masks, ps)?;
        let refvid_emb = add(
            &self.patch_embedding.forward(&rv_tok.as_dtype(cdt)?)?,
            &self.patch_embedding_mask.forward(&rm_tok.as_dtype(cdt)?)?,
        )?;
        let (pose_tok, _) = patchify(&pose20, ps)?;
        let (dm_tok, _) = patchify(inp.driving_masks, ps)?;
        let pose_emb = add(
            &self
                .patch_embedding_pose
                .forward(&pose_tok.as_dtype(cdt)?)?,
            &self.patch_embedding_mask.forward(&dm_tok.as_dtype(cdt)?)?,
        )?;

        // --- assemble packed tokens + per-chunk RoPE: [additional_ref | ref | video | pose] ---
        let mut tok_list: Vec<Array> = Vec::new();
        let mut cos_list: Vec<Array> = Vec::new();
        let mut sin_list: Vec<Array> = Vec::new();
        let mut addref_count = 0usize;

        if let Some(ar) = inp.additional_ref_latent {
            let arm = inp.additional_ref_masks.ok_or_else(|| {
                Error::Msg(
                    "scail2: additional_ref_masks required with additional_ref_latent".into(),
                )
            })?;
            let ar_n = ar.shape()[1];
            let ar20 = concatenate_axis(&[ar, &Array::ones::<f32>(&[i2v, ar_n, hh, ww])?], 0)?;
            let (ar_tok, _) = patchify(&ar20, ps)?;
            let (arm_tok, _) = patchify(arm, ps)?;
            let ar_emb = add(
                &self.patch_embedding.forward(&ar_tok.as_dtype(cdt)?)?,
                &self.patch_embedding_mask.forward(&arm_tok.as_dtype(cdt)?)?,
            )?;
            addref_count = ar_n as usize;
            let (c, s) = self
                .rope
                .chunk((addref_count, rope_h, rope_w), (0, h_shift, 0), false)?;
            tok_list.push(ar_emb);
            cos_list.push(c);
            sin_list.push(s);
        }

        // ref+video tokens (one block); RoPE splits ref (1 frame) and video (rope_t frames).
        tok_list.push(refvid_emb);
        let (rc, rs) = self
            .rope
            .chunk((1, rope_h, rope_w), (addref_count, h_shift, 0), false)?;
        let (vc, vs) = self.rope.chunk(
            (rope_t, rope_h, rope_w),
            (base_video_shift + addref_count, 0, 0),
            false,
        )?;
        cos_list.push(rc);
        cos_list.push(vc);
        sin_list.push(rs);
        sin_list.push(vs);

        // pose tokens (W-shifted, freq avg-pool downsampled).
        tok_list.push(pose_emb);
        let (pc, psn) = self.rope.chunk(
            (rope_t, rope_h, rope_w),
            (base_video_shift + addref_count, 0, cfg.pose_w_shift),
            true,
        )?;
        cos_list.push(pc);
        sin_list.push(psn);

        let tok_refs: Vec<&Array> = tok_list.iter().collect();
        let cos_refs: Vec<&Array> = cos_list.iter().collect();
        let sin_refs: Vec<&Array> = sin_list.iter().collect();
        let tokens = concatenate_axis(&tok_refs, 0)?; // [L_total, dim]
        let l_total = tokens.shape()[0];
        let tokens = tokens.reshape(&[1, l_total, dim])?;
        let cos = concatenate_axis(&cos_refs, 0)?; // [L_total, 1, half_d]
        let sin = concatenate_axis(&sin_refs, 0)?;

        // --- time / text / image conditioning ---
        let (e, e0) = self.time_embed(inp.t)?;
        let text_ctx = self.embed_text(inp.context)?;
        let img_ctx = self.embed_img(inp.clip_fea)?;

        // --- transformer blocks (f32 activations) ---
        let mut x = to_f32(&tokens)?;
        for block in &self.blocks {
            x = block.forward(&x, &e0, &text_ctx, &img_ctx, &cos, &sin, cdt)?;
        }
        let xh = self.apply_head(&x, &e)?; // [1, L_total, out_dim·∏patch]

        // --- keep only the video tokens, unpatchify back to [16, T, H, W] ---
        let addref_length = addref_count * ref_length;
        let offset = (addref_length + ref_length) as i32;
        let l_video = seq_length as i32;
        let idx = Array::from_slice(
            &(offset..offset + l_video).collect::<Vec<i32>>(),
            &[l_video],
        );
        let op = xh.shape()[2];
        let vid_tok = xh.take_axis(&idx, 1)?.reshape(&[l_video, op])?;
        unpatchify(&vid_tok, (rope_t, rope_h, rope_w), cfg.wan.out_dim, ps)
    }
}

/// Every LoRA-adaptable target in the SCAIL-2 DiT as a dotted `SCAIL2Model` parameter path (the
/// naming a diffusers/PEFT/kohya LoRA file carries, once its namespace prefix is stripped), for a
/// model with `num_layers` blocks. SCAIL-2 *is* Wan2.1-14B I2V, so its raw module names are exactly
/// the targets a Wan-I2V LoRA (the lightx2v step-distill lightning LoRA) or SCAIL-2's own Bias-Aware
/// DPO LoRA names. This is the single source of truth for [`AdaptableHost::adaptable_paths`] (the
/// kohya `flattened → dotted` table) and is kept in lock-step with [`Scail2Dit::adaptable_mut`] by
/// tests (`adaptable_paths_unique_and_kohya_collision_free` here + a real-weight resolution guard).
pub(crate) fn scail2_adaptable_paths(num_layers: usize) -> Vec<String> {
    // Globals (the whole-model Linears outside the transformer blocks). `patch_embedding` and the two
    // SCAIL-2-specific stems (`_pose`/`_mask`) are SCAIL-2-shaped — only SCAIL-2's own DPO LoRA can
    // name them; an external Wan-I2V LoRA never does, so it never trips the strict installer here.
    let mut paths: Vec<String> = [
        "patch_embedding",
        "patch_embedding_pose",
        "patch_embedding_mask",
        "text_embedding.0",
        "text_embedding.2",
        "time_embedding.0",
        "time_embedding.2",
        "time_projection.1",
        "img_emb.proj.1",
        "img_emb.proj.3",
        "head.head",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    for i in 0..num_layers {
        for leaf in [
            "self_attn.q",
            "self_attn.k",
            "self_attn.v",
            "self_attn.o",
            "cross_attn.q",
            "cross_attn.k",
            "cross_attn.v",
            "cross_attn.o",
            "cross_attn.k_img",
            "cross_attn.v_img",
            "ffn.0",
            "ffn.2",
        ] {
            paths.push(format!("blocks.{i}.{leaf}"));
        }
    }
    paths
}

/// Install inference LoRA(s) onto the SCAIL-2 DiT as forward-time residuals (sc-5451). SCAIL-2 is
/// Wan2.1-14B I2V, so the family-agnostic [`mlx_gen::adapters::loader`] path resolves a diffusers /
/// PEFT / kohya / LoKr / LoHa file directly against the raw module names — the same residual install
/// the Z-Image / Qwen-Image providers use. Because adapters apply *over* the (possibly Q4/Q8) base
/// rather than merging into it, they stack cleanly on the pre-quantized packed weights (sc-5445).
impl AdaptableHost for Scail2Dit {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["patch_embedding"] => Some(&mut self.patch_embedding),
            ["patch_embedding_pose"] => Some(&mut self.patch_embedding_pose),
            ["patch_embedding_mask"] => Some(&mut self.patch_embedding_mask),
            ["text_embedding", "0"] => Some(&mut self.text_embedding_0),
            ["text_embedding", "2"] => Some(&mut self.text_embedding_2),
            ["time_embedding", "0"] => Some(&mut self.time_embedding_0),
            ["time_embedding", "2"] => Some(&mut self.time_embedding_2),
            ["time_projection", "1"] => Some(&mut self.time_projection),
            ["img_emb", "proj", "1"] => Some(&mut self.img_emb_1),
            ["img_emb", "proj", "3"] => Some(&mut self.img_emb_3),
            ["head", "head"] => Some(&mut self.head),
            ["blocks", idx, rest @ ..] => {
                let i: usize = idx.parse().ok()?;
                self.blocks.get_mut(i)?.adaptable_mut(rest)
            }
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        scail2_adaptable_paths(self.blocks.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// After the loader strips the `diffusion_model.`/`transformer.` namespace, a Wan2.1-14B-I2V
    /// step-distill (lightx2v lightning) LoRA and SCAIL-2's own Bias-Aware DPO LoRA name exactly these
    /// dotted targets — every one must be an adaptable SCAIL-2 path or the strict installer would
    /// reject the file. Mirrors the wan `lightning_lora_keys_normalize_to_wan_dit_targets` guard.
    #[test]
    fn lightx2v_and_dpo_target_keys_are_adaptable() {
        let paths: BTreeSet<String> = scail2_adaptable_paths(40).into_iter().collect();
        let must = [
            "blocks.0.self_attn.q",
            "blocks.0.self_attn.k",
            "blocks.0.self_attn.v",
            "blocks.0.self_attn.o",
            "blocks.0.cross_attn.q",
            "blocks.0.cross_attn.k",
            "blocks.0.cross_attn.v",
            "blocks.0.cross_attn.o",
            "blocks.0.cross_attn.k_img",
            "blocks.0.cross_attn.v_img",
            "blocks.0.ffn.0",
            "blocks.0.ffn.2",
            "blocks.39.self_attn.q",
            "blocks.39.ffn.2",
            "head.head",
        ];
        for k in must {
            assert!(
                paths.contains(k),
                "`{k}` is not an adaptable SCAIL-2 LoRA target"
            );
        }
    }

    /// The path set must be duplicate-free AND stay collision-free under the kohya `.`→`_` flattening
    /// (the [`AdaptableHost::adaptable_paths`] contract — the `flattened → dotted` table would
    /// otherwise lose a target). 11 globals + 12 Linears × `num_layers` blocks.
    #[test]
    fn adaptable_paths_unique_and_kohya_collision_free() {
        let paths = scail2_adaptable_paths(40);
        let n = paths.len();
        assert_eq!(n, 11 + 40 * 12);
        let uniq: BTreeSet<&String> = paths.iter().collect();
        assert_eq!(uniq.len(), n, "duplicate adaptable path");
        let flat: BTreeSet<String> = paths.iter().map(|p| p.replace('.', "_")).collect();
        assert_eq!(flat.len(), n, "kohya-flattened path collision");
    }
}
