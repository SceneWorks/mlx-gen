//! The Krea 2 dense single-stream DiT (`Krea2Transformer2DModel` / reference `mmdit.py`
//! `SingleStreamDiT`) forward.
//!
//! ```text
//!   img_in:        img tokens = Linear(patchify(latent, p=2))          [b, img_len, 6144]
//!   time_embed:    t   = Linear(GELU(Linear(sinusoid(timestep))))      [b, 1, 6144]
//!   time_mod_proj: tvec = Linear(GELU(t))                              [b, 1, 6·6144]   (shared modulation)
//!   text_fusion:   ctx = aggregate(stacked 12 Qwen3-VL layers)         [b, cap, 2560]
//!   txt_in:        ctx = Linear(GELU(Linear(RMSNorm(ctx))))            [b, cap, 6144]
//!   combined = [ctx ; img]                                            [b, cap+img_len, 6144]
//!   28× transformer_blocks (gated single-stream, DoubleSharedModulation, 3-axis RoPE)
//!   final_layer:   (1+scale)·RMSNorm(x) + shift → Linear(6144→64)      [b, cap+img_len, 64]
//!   slice image tokens → unpatchify                                   → velocity [b, 16, H, W]
//! ```
//!
//! Per-sample `B = 1`: the text stream is trimmed to its valid length (the encoder's padding mask) and
//! the whole sequence runs **unmasked** — numerically exact for the image-velocity output (the
//! reference's pad-to-256 + key/query mask only zeroes tokens that are then discarded).

pub mod block;
pub mod rope;

use mlx_rs::ops::{concatenate_axis, cos, divide, exp, multiply, sin, split, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::array::scalar;
use mlx_gen::nn::gelu_tanh;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::Krea2Config;
use crate::quant::lin;
use block::{RmsScale, SingleStreamBlock, TextFusionTransformer};
use mlx_gen::adapters::AdaptableLinear;
use rope::RopeTables;

/// The Krea 2 single-stream DiT.
pub struct Krea2Transformer {
    cfg: Krea2Config,
    dtype: Dtype,
    img_in: AdaptableLinear,
    time_embed_l1: AdaptableLinear,
    time_embed_l2: AdaptableLinear,
    time_mod_proj: AdaptableLinear,
    txt_in_norm: RmsScale,
    txt_in_l1: AdaptableLinear,
    txt_in_l2: AdaptableLinear,
    text_fusion: TextFusionTransformer,
    blocks: Vec<SingleStreamBlock>,
    final_norm: RmsScale,
    final_linear: AdaptableLinear,
    final_sstable: Array, // [1, 2, hidden]
}

impl Krea2Transformer {
    /// Build from a loaded `transformer/` weight set (already validated by [`crate::convert`]).
    pub fn from_weights(w: &Weights, cfg: &Krea2Config) -> Result<Self> {
        let (heads, kv, hd, eps) = (
            cfg.num_attention_heads as i32,
            cfg.num_kv_heads as i32,
            cfg.attention_head_dim as i32,
            cfg.norm_eps,
        );
        let (theads, tkv) = (
            cfg.text_num_attention_heads as i32,
            cfg.text_num_kv_heads as i32,
        );
        let hidden = cfg.hidden_size as i32;

        // The dense `img_in.bias` is always present and in the compute dtype (bf16 real / f32 fixture);
        // the quantized snapshot only packs the attn/FFN Linears, so this never reads u32 codes.
        let dtype = w.require("img_in.bias")?.dtype();

        let final_sstable = w
            .require("final_layer.scale_shift_table")?
            .reshape(&[1, 2, hidden])?;

        Ok(Self {
            cfg: cfg.clone(),
            dtype,
            img_in: lin(w, "img_in", true)?,
            time_embed_l1: lin(w, "time_embed.linear_1", true)?,
            time_embed_l2: lin(w, "time_embed.linear_2", true)?,
            time_mod_proj: lin(w, "time_mod_proj", true)?,
            txt_in_norm: RmsScale::from_weights(w, "txt_in.norm.weight", eps)?,
            txt_in_l1: lin(w, "txt_in.linear_1", true)?,
            txt_in_l2: lin(w, "txt_in.linear_2", true)?,
            text_fusion: TextFusionTransformer::from_weights(
                w,
                cfg.num_layerwise_text_blocks,
                cfg.num_refiner_text_blocks,
                theads,
                tkv,
                hd,
                eps,
            )?,
            blocks: (0..cfg.num_layers)
                .map(|i| {
                    SingleStreamBlock::from_weights(
                        w,
                        &format!("transformer_blocks.{i}"),
                        heads,
                        kv,
                        hd,
                        hidden,
                        eps,
                    )
                })
                .collect::<Result<_>>()?,
            final_norm: RmsScale::from_weights(w, "final_layer.norm.weight", eps)?,
            final_linear: lin(w, "final_layer.linear", true)?,
            final_sstable,
        })
    }

    /// Velocity prediction.
    ///
    /// - `latent`: `[b, 16, H, W]` (H, W multiples of `patch_size`),
    /// - `timestep`: `[b]` f32 (raw flow time in `[0, 1]`),
    /// - `context`: `[b, n_tokens, num_text_layers, text_hidden]` — the stacked Qwen3-VL select-layer
    ///   hidden states (sc-7569),
    /// - `mask`: `Some([b, n_tokens])` to trim the text stream to its valid length (B = 1), or `None`
    ///   (all tokens valid).
    ///
    /// Returns the velocity `[b, 16, H, W]`.
    pub fn forward(
        &self,
        latent: &Array,
        timestep: &Array,
        context: &Array,
        mask: Option<&Array>,
    ) -> Result<Array> {
        let cfg = &self.cfg;
        let p = cfg.patch_size as i32;
        let dt = self.dtype;
        let sh = latent.shape();
        let (h, w) = (sh[2], sh[3]);
        let (ht, wt) = (h / p, w / p);
        let img_len = ht * wt;
        let latent_ch = cfg.in_channels as i32 / (p * p);

        // Trim the text stream to its valid length (B = 1).
        let n_tok = context.shape()[1];
        let cap_len = match mask {
            Some(m) => sum(&m.as_dtype(Dtype::Float32)?, false)?.item::<f32>() as i32,
            None => n_tok,
        };
        let context = slice_axis1(context, 0, cap_len)?.as_dtype(dt)?;

        // Image patch embed.
        let img = self.img_in.forward(&patchify(&latent.as_dtype(dt)?, p)?)?; // [b, img_len, hidden]

        // Timestep embed → `t`; shared modulation `tvec = time_mod_proj(GELU(t))`.
        let t_sin = temb(timestep, cfg.timestep_embed_dim as i32)?.as_dtype(dt)?; // [b, 1, tdim]
        let t = self
            .time_embed_l2
            .forward(&gelu_tanh(&self.time_embed_l1.forward(&t_sin)?)?)?; // [b, 1, hidden]
        let tvec = self.time_mod_proj.forward(&gelu_tanh(&t)?)?; // [b, 1, 6·hidden]

        // Text fusion (12 layers → 1) then the text input projection.
        let ctx = self.text_fusion.forward(&context)?; // [b, cap, text_hidden]
        let ctx = self.txt_in_norm.forward(&ctx)?;
        let ctx = self
            .txt_in_l2
            .forward(&gelu_tanh(&self.txt_in_l1.forward(&ctx)?)?)?; // [b, cap, hidden]

        // Fuse to the joint sequence and run the single-stream stack under the joint RoPE.
        let mut combined = concatenate_axis(&[&ctx, &img], 1)?; // [b, cap+img_len, hidden]
        let rope = RopeTables::build_t2i(
            cap_len as usize,
            ht as usize,
            wt as usize,
            cfg.axes_dims_rope,
            cfg.rope_theta as f64,
        );
        let (rcos, rsin) = rope.joint();
        for blk in &self.blocks {
            combined = blk.forward(&combined, &tvec, &rcos, &rsin)?;
        }

        // Continuous-AdaLN output (SimpleModulation on `t`), then slice the image tokens + unpatchify.
        let out = self.final_layer(&combined, &t)?; // [b, cap+img_len, in_channels]
        let img_out = slice_axis1(&out, cap_len, cap_len + img_len)?;
        unpatchify(&img_out, ht, wt, p, latent_ch)
    }

    /// Reference `LastLayer`: `SimpleModulation(t) = t + scale_shift_table` → `(scale, shift)`;
    /// `Linear((1+scale)·RMSNorm(x) + shift)`.
    fn final_layer(&self, x: &Array, t: &Array) -> Result<Array> {
        let m = mlx_rs::ops::add(t, &self.final_sstable)?; // [b, 2, hidden] (t broadcasts 1→2)
        let parts = split(&m, 2, 1)?;
        let (scale, shift) = (&parts[0], &parts[1]); // each [b, 1, hidden]
        let normed = mlx_rs::ops::add(
            &multiply(
                &self.final_norm.forward(x)?,
                &mlx_rs::ops::add(scale, Array::from_f32(1.0))?,
            )?,
            shift,
        )?;
        self.final_linear.forward(&normed)
    }

    /// Quantize the DiT's Linear projections — the attn/FFN of every single-stream and text-fusion
    /// block (the 256 targets [`crate::convert::transformer_quant_targets`] packs). The embedders,
    /// `time_mod_proj`, `txt_in`, `projector`, and `final_layer` stay dense, matching the converter.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.text_fusion.quantize(bits)?;
        for b in &mut self.blocks {
            b.quantize(bits)?;
        }
        Ok(())
    }
}

// ── Shared helpers ──────────────────────────────────────────────────────────────────────────

/// Join a module prefix with a leaf name, tolerating an empty prefix.
pub(crate) fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

/// Slice `[b, L, ...]` along the sequence axis (axis 1) to `[start, end)`.
pub(crate) fn slice_axis1(x: &Array, start: i32, end: i32) -> Result<Array> {
    let idx: Vec<i32> = (start..end).collect();
    Ok(x.take_axis(Array::from_slice(&idx, &[end - start]), 1)?)
}

/// Expand `[b, s, hkv, hd]` → `[b, s, hkv·groups, hd]`, repeating each kv head `groups` times
/// consecutively (`repeat_interleave` over the head axis, matching the reference's `enable_gqa`).
pub(crate) fn repeat_kv(x: &Array, groups: i32) -> Result<Array> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let sh = x.shape();
    let (b, s, hkv, hd) = (sh[0], sh[1], sh[2], sh[3]);
    let x = x.expand_dims(3)?; // [b, s, hkv, 1, hd]
    let x = mlx_rs::ops::broadcast_to(&x, &[b, s, hkv, groups, hd])?;
    Ok(x.reshape(&[b, s, hkv * groups, hd])?)
}

/// Reference `temb`: `freqs = exp(−ln(1e4)·arange(half)/half)`, `args = (timestep·1e3)·freqs`,
/// `concat([cos, sin], −1)` (cos-first). `timestep`: `[b]` → `[b, 1, dim]` (a per-sample vector that
/// broadcasts over the sequence). Built in f32 (the reference upcasts).
fn temb(timestep: &Array, dim: i32) -> Result<Array> {
    let half = dim / 2;
    let arange: Vec<f32> = (0..half).map(|i| i as f32).collect();
    let arange = Array::from_slice(&arange, &[half]);
    let neg_ln = -(10000f64.ln()) as f32;
    let exponent = divide(&multiply(&arange, scalar(neg_ln))?, scalar(half as f32))?;
    let freqs = exp(&exponent)?; // [half]

    let t = timestep.as_dtype(Dtype::Float32)?;
    let b = t.shape()[0];
    let scaled = multiply(&t.reshape(&[b, 1, 1])?, scalar(1000.0))?; // [b, 1, 1]
    let args = multiply(&scaled, &freqs)?; // [b, 1, half]
    Ok(concatenate_axis(&[&cos(&args)?, &sin(&args)?], -1)?) // [b, 1, dim]
}

/// Reference `rearrange("b c (h ph) (w pw) -> b (h w) (c ph pw)")`: `[b, C, H, W] →
/// [b, (H/p)(W/p), C·p·p]` with **channel-major** patch flattening (NOT boogu's `(ph pw c)`).
fn patchify(latent: &Array, p: i32) -> Result<Array> {
    let sh = latent.shape();
    let (b, c, h, w) = (sh[0], sh[1], sh[2], sh[3]);
    let (ht, wt) = (h / p, w / p);
    let x = latent.reshape(&[b, c, ht, p, wt, p])?; // b, c, ht, ph, wt, pw
    let x = x.transpose_axes(&[0, 2, 4, 1, 3, 5])?; // b, ht, wt, c, ph, pw
    Ok(x.reshape(&[b, ht * wt, c * p * p])?)
}

/// Inverse of [`patchify`] (`"b (h w) (c ph pw) -> b c (h ph) (w pw)"`): `[b, (h)(w), C·p·p] →
/// [b, C, h·p, w·p]`.
fn unpatchify(tokens: &Array, ht: i32, wt: i32, p: i32, c: i32) -> Result<Array> {
    let b = tokens.shape()[0];
    let x = tokens.reshape(&[b, ht, wt, c, p, p])?; // b, ht, wt, c, ph, pw
    let x = x.transpose_axes(&[0, 3, 1, 4, 2, 5])?; // b, c, ht, ph, wt, pw
    Ok(x.reshape(&[b, c, ht * p, wt * p])?)
}
