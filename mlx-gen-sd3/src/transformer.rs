//! Stable Diffusion 3.5 **MMDiT-Large** forward pass — the SD3.5 **E3** slice (sc-7862).
//!
//! This is the genuinely novel core of the native SD3.5 port: the `SD3Transformer2DModel` forward.
//! It is a faithful mirror of diffusers `SD3Transformer2DModel` / `JointTransformerBlock`, built by
//! REUSING the joint-attention double-stream pattern from `mlx-gen-flux2`'s `DoubleBlock` /
//! `DoubleAttention` (per-head q/k RMSNorm via the `process_qkv` placement, joint concat → SDPA →
//! split) with the SD3.5 deltas (spike sc-7850, real-weight confirmed):
//!
//!   * **NO RoPE.** SD3.5 positions image tokens with a LEARNED 2D positional embedding added at
//!     patchify (`pos_embed.pos_embed [1, 192*192, 2432]`, cropped/centered to the actual patch grid
//!     exactly as diffusers `PatchEmbed.cropped_pos_embed`). FLUX.2's axial RoPE is dropped entirely.
//!   * **38 all-double-stream (joint) blocks.** FLUX.2's hybrid 8-double + 24-single topology is
//!     replaced by 38 joint blocks; the single-stream path does not exist here.
//!   * **Per-block adaLN modulation** from the `(timestep + pooled-text)` embedding. Each block's
//!     `norm1` (image, AdaLayerNormZero, 6×hidden) and `norm1_context` (text) produce
//!     `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)`. The **LAST** block is
//!     `context_pre_only`: its text stream uses AdaLayerNormContinuous (`norm1_context` 2×hidden, no
//!     `attn.to_add_out`, no `ff_context`), matching E1's converter.
//!   * **qk-RMSNorm on BOTH streams** (`norm_q`/`norm_k` image, `norm_added_q`/`norm_added_k` text),
//!     each `[head_dim]`, eps [`crate::config::RMS_EPS`].
//!   * **GELU-approx FFN** (diffusers `FeedForward(activation_fn="gelu-approximate")`), NOT SwiGLU.
//!   * Output: `norm_out` (AdaLayerNormContinuous from the time/text embed) + `proj_out` →
//!     unpatchify → `[B, 16, H/8, W/8]` noise prediction.
//!
//! Tensor keys are the diffusers `SD3Transformer2DModel` names exactly (E1's converter is a 1:1
//! identity rename, [`crate::convert`]), so a converted/quantized checkpoint loads unchanged.
//!
//! All Linears are bias-carrying [`AdaptableLinear`]s so the offline-quantized Q4/Q8 transformer
//! ([`crate::convert::quantize_sd3_dir`]) loads via [`AdaptableLinear::from_quantized_parts`] and a
//! future LoRA/LoKr (epic T-stories) composes over the dense base for free. Activations run f32
//! (the mlx-gen DiT convention; the 16-bit Metal GEMM bug + the quality target), so the quantized
//! forward feeds `quantized_matmul` f32 inputs.

use mlx_rs::fast::{layer_norm, rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{concatenate_axis, split};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::{conv2d, gelu_tanh, modulate, silu, timestep_sincos};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::{Sd3Arch, RMS_EPS};

/// Group-wise affine quantization group size (the mlx-gen / mflux default; matches the offline
/// packer [`crate::convert::quantize_sd3_dir`]).
const QUANT_GROUP_SIZE: i32 = 64;

/// diffusers `SD3Transformer2DModel` LayerNorm epsilon (the elementwise-affine-free norms inside the
/// joint blocks and the AdaLN-continuous output norm).
const LN_EPS: f32 = 1e-6;
/// Sinusoidal timestep embedding: diffusers `get_timestep_embedding(..., downscale_freq_shift=0)`.
const TIME_MAX_PERIOD: f64 = 10_000.0;
const TIME_DOWNSCALE_FREQ_SHIFT: f64 = 0.0;

// ----------------------------------------------------------------------------------------------
// Linear loading (dense or pre-quantized), bias-carrying.
// ----------------------------------------------------------------------------------------------

/// Load a bias-carrying [`AdaptableLinear`] at `{prefix}` from `w`. Delegates to
/// [`mlx_gen::quant::lin`], which **auto-detects** dense vs pre-quantized: if `{prefix}.scales` is
/// present it builds the pre-quantized base (consume-side of
/// [`crate::convert::quantize_sd3_transformer`], bit-width inferred from the packed shapes at
/// [`QUANT_GROUP_SIZE`]); otherwise the dense base. Every SD3.5 transformer Linear carries a bias
/// (diffusers `nn.Linear` default), so `bias = true`.
fn lin(w: &Weights, prefix: &str) -> Result<AdaptableLinear> {
    mlx_gen::quant::lin(w, prefix, true, QUANT_GROUP_SIZE)
}

// ----------------------------------------------------------------------------------------------
// Per-head q/k RMSNorm reshape, mirroring flux2's `process_qkv`.
// ----------------------------------------------------------------------------------------------

/// `[B,S,H·D] → [B,H,S,D]` with per-head q/k RMSNorm. Identical placement to
/// `mlx_gen_flux2::transformer::process_qkv` (QK-norm AFTER the per-head reshape, BEFORE attention),
/// but with no RoPE applied to the result.
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

/// SDPA over `[B,H,S,D] → [B,S,H·D]` (no mask; full bidirectional attention over the joint sequence).
fn attention(q: &Array, k: &Array, v: &Array, head_dim: i32) -> Result<Array> {
    let b = q.shape()[0];
    let scale = (head_dim as f32).powf(-0.5);
    let o = scaled_dot_product_attention(q, k, v, scale, None, None)?;
    Ok(o.transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[b, -1, q.shape()[1] * head_dim])?)
}

// ----------------------------------------------------------------------------------------------
// Joint (double-stream) attention.
// ----------------------------------------------------------------------------------------------

struct JointAttention {
    // image stream
    to_q: AdaptableLinear,
    to_k: AdaptableLinear,
    to_v: AdaptableLinear,
    to_out: AdaptableLinear,
    norm_q: Array,
    norm_k: Array,
    // text stream
    add_q: AdaptableLinear,
    add_k: AdaptableLinear,
    add_v: AdaptableLinear,
    /// `None` on the final `context_pre_only` block (the text attention output is discarded).
    to_add_out: Option<AdaptableLinear>,
    norm_added_q: Array,
    norm_added_k: Array,
    heads: i32,
    head_dim: i32,
}

impl JointAttention {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        heads: i32,
        head_dim: i32,
        context_pre_only: bool,
    ) -> Result<Self> {
        let g = |n: &str| w.require(&format!("{prefix}.{n}.weight")).cloned();
        let l = |n: &str| lin(w, &format!("{prefix}.{n}"));
        Ok(Self {
            to_q: l("to_q")?,
            to_k: l("to_k")?,
            to_v: l("to_v")?,
            to_out: l("to_out.0")?,
            norm_q: g("norm_q")?,
            norm_k: g("norm_k")?,
            add_q: l("add_q_proj")?,
            add_k: l("add_k_proj")?,
            add_v: l("add_v_proj")?,
            to_add_out: if context_pre_only {
                None
            } else {
                Some(l("to_add_out")?)
            },
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
        ] {
            p.quantize(bits, None)?;
        }
        if let Some(o) = self.to_add_out.as_mut() {
            o.quantize(bits, None)?;
        }
        Ok(())
    }

    /// Joint attention over `[img ; txt]` (diffusers concatenates the text AFTER the image along the
    /// sequence). Returns `(img_attn_out, Option<txt_attn_out>)`; the text output is `None` on the
    /// final `context_pre_only` block.
    fn forward(&self, img: &Array, txt: &Array) -> Result<(Array, Option<Array>)> {
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
        // [img, txt] order along the sequence (axis 2 in BHSD) — diffusers `cat([sample, context])`.
        let q = concatenate_axis(&[&iq, &tq], 2)?;
        let k = concatenate_axis(&[&ik, &tk], 2)?;
        let v = concatenate_axis(&[&iv, &tv], 2)?;
        let o = attention(&q, &k, &v, self.head_dim)?;

        let img_seq = img.shape()[1];
        let img_idx = Array::from_slice(&(0..img_seq).collect::<Vec<i32>>(), &[img_seq]);
        let img_part = o.take_axis(&img_idx, 1)?;
        let img_out = self.to_out.forward(&img_part)?;
        let txt_out = match &self.to_add_out {
            Some(to_add_out) => {
                let txt_idx = Array::from_slice(
                    &(img_seq..o.shape()[1]).collect::<Vec<i32>>(),
                    &[o.shape()[1] - img_seq],
                );
                Some(to_add_out.forward(&o.take_axis(&txt_idx, 1)?)?)
            }
            None => None,
        };
        Ok((img_out, txt_out))
    }
}

// ----------------------------------------------------------------------------------------------
// FeedForward (GELU-approx).
// ----------------------------------------------------------------------------------------------

struct FeedForward {
    net0: AdaptableLinear, // net.0.proj  [4·hidden, hidden]
    net2: AdaptableLinear, // net.2       [hidden, 4·hidden]
}

impl FeedForward {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            net0: lin(w, &format!("{prefix}.net.0.proj"))?,
            net2: lin(w, &format!("{prefix}.net.2"))?,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.net0.quantize(bits, None)?;
        self.net2.quantize(bits, None)?;
        Ok(())
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.net0.forward(x)?;
        let h = gelu_tanh(&h)?;
        self.net2.forward(&h)
    }
}

// ----------------------------------------------------------------------------------------------
// adaLN modulation producers.
// ----------------------------------------------------------------------------------------------

/// AdaLayerNormZero: `silu(emb) → linear → [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
/// gate_mlp]`, each `[B, hidden]`. Mirrors diffusers `AdaLayerNormZero` (which packs the 6 chunks in
/// exactly this order). Used by `norm1` (image) and the non-final blocks' `norm1_context`.
struct AdaLnZero {
    linear: AdaptableLinear,
}

/// The 6 AdaLayerNormZero modulation tensors, each `[B, 1, hidden]` (ready to broadcast over `[B, S,
/// hidden]`): `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)`.
type ZeroMod = (Array, Array, Array, Array, Array, Array);

impl AdaLnZero {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            linear: lin(w, &format!("{prefix}.linear"))?,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.linear.quantize(bits, None)
    }

    fn forward(&self, emb: &Array) -> Result<ZeroMod> {
        let e = self.linear.forward(&silu(emb)?)?; // [B, 6·hidden]
        let parts = split(&e, 6, -1)?;
        let u = |a: &Array| -> Result<Array> { Ok(a.expand_dims(1)?) }; // [B, hidden] -> [B, 1, hidden]
        Ok((
            u(&parts[0])?,
            u(&parts[1])?,
            u(&parts[2])?,
            u(&parts[3])?,
            u(&parts[4])?,
            u(&parts[5])?,
        ))
    }
}

/// AdaLayerNormContinuous: `silu(emb) → linear → [scale, shift]`, each `[B, hidden]`. Used by the
/// final block's `norm1_context` (`context_pre_only`) and the model-level `norm_out`.
///
/// NOTE the diffusers chunk order: `AdaLayerNormContinuous` does `scale, shift =
/// emb.chunk(2, dim=1)` — i.e. the FIRST chunk is `scale`, the SECOND is `shift` (the opposite of
/// `AdaLayerNormZero`, which is shift-first). The net norm it applies is `norm(x)·(1+scale) +
/// shift`. See the sibling crate `mlx-gen-flux2/src/transformer.rs::norm_out` for the same module.
struct AdaLnContinuous {
    linear: AdaptableLinear,
}

impl AdaLnContinuous {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            linear: lin(w, &format!("{prefix}.linear"))?,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.linear.quantize(bits, None)
    }

    /// Returns `(shift, scale)`, each `[B, 1, hidden]`.
    ///
    /// diffusers chunks the projection as `scale, shift = emb.chunk(2)`, so `parts[0]` is the
    /// **scale** and `parts[1]` is the **shift**. We return them in `(shift, scale)` order so the
    /// downstream `modulated_layer_norm(x, shift, scale)` applies `(1+scale)·norm(x) + shift`,
    /// matching diffusers `AdaLayerNormContinuous`.
    fn forward(&self, emb: &Array) -> Result<(Array, Array)> {
        let e = self.linear.forward(&silu(emb)?)?; // [B, 2·hidden]
        let parts = split(&e, 2, -1)?;
        let scale = parts[0].expand_dims(1)?;
        let shift = parts[1].expand_dims(1)?;
        Ok((shift, scale))
    }
}

/// `(1 + scale)·layernorm(x) + shift` over the affine-free LayerNorm (diffusers'
/// `nn.LayerNorm(elementwise_affine=False)`). f32 `1` (strong) — `one_matches_scale = false`.
fn modulated_layer_norm(x: &Array, shift: &Array, scale: &Array) -> Result<Array> {
    let n = layer_norm(x, None, None, LN_EPS)?;
    modulate(&n, scale, shift, false)
}

/// Gated residual `x + gate·y` (diffusers `x = x + gate.unsqueeze(1) * y`).
fn gated(x: &Array, gate: &Array, y: &Array) -> Result<Array> {
    mlx_gen::nn::gated(x, gate, y)
}

// ----------------------------------------------------------------------------------------------
// Joint transformer block.
// ----------------------------------------------------------------------------------------------

struct JointBlock {
    norm1: AdaLnZero,
    /// Non-final: AdaLN-zero (6 chunks). Final (`context_pre_only`): AdaLN-continuous (2 chunks).
    norm1_context: ContextNorm,
    attn: JointAttention,
    ff: FeedForward,
    /// Absent on the final `context_pre_only` block.
    ff_context: Option<FeedForward>,
}

enum ContextNorm {
    Zero(AdaLnZero),
    Continuous(AdaLnContinuous),
}

impl JointBlock {
    fn from_weights(
        w: &Weights,
        idx: usize,
        is_last: bool,
        heads: i32,
        head_dim: i32,
    ) -> Result<Self> {
        let p = format!("transformer_blocks.{idx}");
        let norm1 = AdaLnZero::from_weights(w, &format!("{p}.norm1"))?;
        let norm1_context = if is_last {
            ContextNorm::Continuous(AdaLnContinuous::from_weights(
                w,
                &format!("{p}.norm1_context"),
            )?)
        } else {
            ContextNorm::Zero(AdaLnZero::from_weights(w, &format!("{p}.norm1_context"))?)
        };
        Ok(Self {
            norm1,
            norm1_context,
            attn: JointAttention::from_weights(w, &format!("{p}.attn"), heads, head_dim, is_last)?,
            ff: FeedForward::from_weights(w, &format!("{p}.ff"))?,
            ff_context: if is_last {
                None
            } else {
                Some(FeedForward::from_weights(w, &format!("{p}.ff_context"))?)
            },
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.norm1.quantize(bits)?;
        match &mut self.norm1_context {
            ContextNorm::Zero(z) => z.quantize(bits)?,
            ContextNorm::Continuous(c) => c.quantize(bits)?,
        }
        self.attn.quantize(bits)?;
        self.ff.quantize(bits)?;
        if let Some(ff) = self.ff_context.as_mut() {
            ff.quantize(bits)?;
        }
        Ok(())
    }

    /// `(hidden_states, encoder_hidden_states, temb)` → updated `(hidden_states,
    /// encoder_hidden_states)`. On the final block `encoder_hidden_states` is returned unchanged
    /// (`context_pre_only`: the text stream is read-only after attention). Faithful mirror of
    /// diffusers `JointTransformerBlock.forward`.
    fn forward(&self, img: &Array, txt: &Array, temb: &Array) -> Result<(Array, Array)> {
        let (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) =
            self.norm1.forward(temb)?;
        let norm_img = modulated_layer_norm(img, &shift_msa, &scale_msa)?;

        match &self.norm1_context {
            // ---- non-final block: full joint update of both streams --------------------------
            ContextNorm::Zero(zero) => {
                let (c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp) =
                    zero.forward(temb)?;
                let norm_txt = modulated_layer_norm(txt, &c_shift_msa, &c_scale_msa)?;

                let (img_attn, txt_attn) = self.attn.forward(&norm_img, &norm_txt)?;
                let txt_attn = txt_attn.ok_or_else(|| {
                    Error::Msg("sd3 non-final block: missing text attention output".into())
                })?;

                let mut img = gated(img, &gate_msa, &img_attn)?;
                let norm_img2 = modulated_layer_norm(&img, &shift_mlp, &scale_mlp)?;
                let img_ff = self.ff.forward(&norm_img2)?;
                img = gated(&img, &gate_mlp, &img_ff)?;

                let mut txt = gated(txt, &c_gate_msa, &txt_attn)?;
                let norm_txt2 = modulated_layer_norm(&txt, &c_shift_mlp, &c_scale_mlp)?;
                let ff_context = self
                    .ff_context
                    .as_ref()
                    .ok_or_else(|| Error::Msg("sd3 non-final block: missing ff_context".into()))?;
                let txt_ff = ff_context.forward(&norm_txt2)?;
                txt = gated(&txt, &c_gate_mlp, &txt_ff)?;

                Ok((img, txt))
            }
            // ---- final block (context_pre_only): only the image stream is updated --------------
            ContextNorm::Continuous(cont) => {
                let (c_shift, c_scale) = cont.forward(temb)?;
                let norm_txt = modulated_layer_norm(txt, &c_shift, &c_scale)?;

                let (img_attn, _txt_attn) = self.attn.forward(&norm_img, &norm_txt)?;

                let mut img = gated(img, &gate_msa, &img_attn)?;
                let norm_img2 = modulated_layer_norm(&img, &shift_mlp, &scale_mlp)?;
                let img_ff = self.ff.forward(&norm_img2)?;
                img = gated(&img, &gate_mlp, &img_ff)?;

                // Text stream is read-only after this block — return it untouched.
                Ok((img, txt.clone()))
            }
        }
    }
}

// ----------------------------------------------------------------------------------------------
// Patch embed (learned 2D pos_embed, NO RoPE).
// ----------------------------------------------------------------------------------------------

struct PatchEmbed {
    /// NHWC `[hidden, patch, patch, in_channels]` patchify conv.
    proj_w: Array,
    proj_b: Array,
    /// The learned positional table, reshaped to `[1, max, max, hidden]` for cropping.
    pos_embed: Array,
    patch: i32,
    pos_embed_max_size: i32,
    hidden: i32,
}

impl PatchEmbed {
    fn from_weights(w: &Weights, arch: &Sd3Arch) -> Result<Self> {
        let hidden = arch.hidden() as i32;
        let patch = arch.patch_size as i32;
        let in_ch = arch.in_channels as i32;
        let max = arch.pos_embed_max_size as i32;
        // diffusers conv weight is NCHW `[hidden, in, patch, patch]`; mlx conv2d wants NHWC
        // `[out, kH, kW, in]`.
        let proj_w = w
            .require("pos_embed.proj.weight")?
            .transpose_axes(&[0, 2, 3, 1])?;
        let proj_b = w.require("pos_embed.proj.bias")?.clone();
        // `[1, max*max, hidden]` → `[1, max, max, hidden]` for the centered crop.
        let pos_embed = w
            .require("pos_embed.pos_embed")?
            .reshape(&[1, max, max, hidden])?;
        let _ = in_ch; // shape is asserted by the converter/validator, not re-checked here.
        Ok(Self {
            proj_w,
            proj_b,
            pos_embed,
            patch,
            pos_embed_max_size: max,
            hidden,
        })
    }

    /// Crop the learned positional table to a `(ph, pw)` patch grid, centered, exactly as diffusers
    /// `PatchEmbed.cropped_pos_embed`: `top = (max - ph)//2`, `left = (max - pw)//2`, slice
    /// `[top:top+ph, left:left+pw]`, flatten to `[1, ph*pw, hidden]`.
    fn cropped_pos_embed(&self, ph: i32, pw: i32) -> Result<Array> {
        if ph > self.pos_embed_max_size || pw > self.pos_embed_max_size {
            return Err(Error::Msg(format!(
                "sd3 pos_embed: patch grid {ph}x{pw} exceeds pos_embed_max_size {}",
                self.pos_embed_max_size
            )));
        }
        let top = (self.pos_embed_max_size - ph) / 2;
        let left = (self.pos_embed_max_size - pw) / 2;
        let rows = Array::from_slice(&(top..top + ph).collect::<Vec<i32>>(), &[ph]);
        let cols = Array::from_slice(&(left..left + pw).collect::<Vec<i32>>(), &[pw]);
        let cropped = self
            .pos_embed
            .take_axis(&rows, 1)? // [1, ph, max, hidden]
            .take_axis(&cols, 2)?; // [1, ph, pw, hidden]
        Ok(cropped.reshape(&[1, ph * pw, self.hidden])?)
    }

    /// `latent [B, in_ch, H, W]` (f32) → `(tokens [B, ph*pw, hidden], ph, pw)`.
    fn forward(&self, latent: &Array) -> Result<(Array, i32, i32)> {
        let sh = latent.shape();
        let (h, ww) = (sh[2], sh[3]);
        let (ph, pw) = (h / self.patch, ww / self.patch);
        // NCHW → NHWC for the conv.
        let x = latent.transpose_axes(&[0, 2, 3, 1])?;
        // conv2d, stride = patch, padding 0 → `[B, ph, pw, hidden]`.
        let conv = conv2d(&x, &self.proj_w, Some(&self.proj_b), self.patch, 0)?;
        let b = conv.shape()[0];
        let tokens = conv.reshape(&[b, ph * pw, self.hidden])?; // flatten(2).transpose -> NHWC flatten is row-major (h,w)
        let pos = self.cropped_pos_embed(ph, pw)?;
        Ok((mlx_rs::ops::add(&tokens, &pos)?, ph, pw))
    }
}

// ----------------------------------------------------------------------------------------------
// Combined timestep + pooled-text embedder.
// ----------------------------------------------------------------------------------------------

struct TimeTextEmbed {
    ts_l1: AdaptableLinear,
    ts_l2: AdaptableLinear,
    txt_l1: AdaptableLinear,
    txt_l2: AdaptableLinear,
    time_proj_dim: usize,
}

impl TimeTextEmbed {
    fn from_weights(w: &Weights, arch: &Sd3Arch) -> Result<Self> {
        Ok(Self {
            ts_l1: lin(w, "time_text_embed.timestep_embedder.linear_1")?,
            ts_l2: lin(w, "time_text_embed.timestep_embedder.linear_2")?,
            txt_l1: lin(w, "time_text_embed.text_embedder.linear_1")?,
            txt_l2: lin(w, "time_text_embed.text_embedder.linear_2")?,
            time_proj_dim: arch.time_proj_dim,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        for p in [
            &mut self.ts_l1,
            &mut self.ts_l2,
            &mut self.txt_l1,
            &mut self.txt_l2,
        ] {
            p.quantize(bits, None)?;
        }
        Ok(())
    }

    /// `(timestep [B], pooled [B, pooled_dim])` → `temb [B, hidden]`.
    fn forward(&self, timestep: &Array, pooled: &Array) -> Result<Array> {
        // Sinusoidal timestep proj (diffusers `Timesteps(256, flip_sin_to_cos=True, shift=0)`).
        let t_proj = timestep_sincos(
            timestep,
            self.time_proj_dim,
            TIME_MAX_PERIOD,
            TIME_DOWNSCALE_FREQ_SHIFT,
        )?;
        // timestep_embedder: linear_1 → SiLU → linear_2.
        let t = self.ts_l2.forward(&silu(&self.ts_l1.forward(&t_proj)?)?)?;
        // text_embedder (PixArtAlphaTextProjection): linear_1 → SiLU → linear_2.
        let pooled = pooled.as_dtype(Dtype::Float32)?;
        let p = self
            .txt_l2
            .forward(&silu(&self.txt_l1.forward(&pooled)?)?)?;
        Ok(mlx_rs::ops::add(&t, &p)?)
    }
}

// ----------------------------------------------------------------------------------------------
// The MMDiT-Large transformer.
// ----------------------------------------------------------------------------------------------

/// SD3.5-Large / Large-Turbo `SD3Transformer2DModel`.
pub struct Sd3Transformer {
    arch: Sd3Arch,
    patch_embed: PatchEmbed,
    time_text_embed: TimeTextEmbed,
    context_embedder: AdaptableLinear,
    blocks: Vec<JointBlock>,
    norm_out: AdaLnContinuous,
    proj_out: AdaptableLinear,
}

impl Sd3Transformer {
    /// Load from a weight map (diffusers `SD3Transformer2DModel` keys, the identity-converted MLX
    /// set). Every Linear is loaded via [`lin`], which auto-detects a pre-quantized packed snapshot
    /// (`{key}.scales` present → [`AdaptableLinear::from_quantized_parts`]) vs a dense bf16 one.
    pub fn from_weights(w: &Weights, arch: &Sd3Arch) -> Result<Self> {
        let heads = arch.num_heads as i32;
        let head_dim = arch.head_dim as i32;
        let mut blocks = Vec::with_capacity(arch.num_layers);
        for i in 0..arch.num_layers {
            let is_last = i + 1 == arch.num_layers;
            blocks.push(JointBlock::from_weights(w, i, is_last, heads, head_dim)?);
        }
        Ok(Self {
            arch: *arch,
            patch_embed: PatchEmbed::from_weights(w, arch)?,
            time_text_embed: TimeTextEmbed::from_weights(w, arch)?,
            context_embedder: lin(w, "context_embedder")?,
            blocks,
            norm_out: AdaLnContinuous::from_weights(w, "norm_out")?,
            proj_out: lin(w, "proj_out")?,
        })
    }

    /// Convenience: load directly from a `transformer/` dir (dense or pre-quantized). Packed-vs-dense
    /// is auto-detected per Linear from the on-disk tensor set (no `quantization` manifest needed).
    pub fn from_dir(transformer_dir: &std::path::Path, arch: &Sd3Arch) -> Result<Self> {
        let w = Weights::from_dir(transformer_dir)?;
        Self::from_weights(&w, arch)
    }

    /// Quantize every Linear in the transformer to Q4/Q8 in place (group_size 64) — the consume-side
    /// equivalent of loading an already-packed checkpoint. The four per-block qk-RMSNorms and the
    /// learned `pos_embed` table stay dense (matching [`crate::convert::quantize_sd3_transformer`]).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.time_text_embed.quantize(bits)?;
        self.context_embedder.quantize(bits, None)?;
        for b in &mut self.blocks {
            b.quantize(bits)?;
        }
        self.norm_out.quantize(bits)?;
        self.proj_out.quantize(bits, None)?;
        Ok(())
    }

    pub fn arch(&self) -> &Sd3Arch {
        &self.arch
    }

    /// MMDiT forward (noise prediction). Inputs:
    ///   * `latent`   `[B, in_channels, H, W]` (the 16-ch noisy latent),
    ///   * `context`  `[B, ctx_seq, joint_attention_dim]` (= `[B, 333, 4096]` from E2),
    ///   * `pooled`   `[B, pooled_projection_dim]` (= `[B, 2048]` from E2),
    ///   * `timestep` `[B]` (the continuous flow-match timestep, 0..1000 scale).
    ///
    /// Returns `[B, out_channels, H, W]` (the predicted velocity/noise latent). Activations run f32.
    pub fn forward(
        &self,
        latent: &Array,
        context: &Array,
        pooled: &Array,
        timestep: &Array,
    ) -> Result<Array> {
        let latent = latent.as_dtype(Dtype::Float32)?;
        let context = context.as_dtype(Dtype::Float32)?;

        // 1. patchify + learned pos_embed (NO RoPE).
        let (mut img, ph, pw) = self.patch_embed.forward(&latent)?;
        // 2. time + pooled-text embed → temb [B, hidden].
        let temb = self.time_text_embed.forward(timestep, pooled)?;
        // 3. project the joint context to hidden width.
        let mut txt = self.context_embedder.forward(&context)?;

        // 4. 38 joint blocks.
        for block in &self.blocks {
            let (i, t) = block.forward(&img, &txt, &temb)?;
            img = i;
            txt = t;
        }

        // 5. AdaLN-continuous output norm + proj_out.
        let (shift, scale) = self.norm_out.forward(&temb)?;
        let img = modulated_layer_norm(&img, &shift, &scale)?;
        let img = self.proj_out.forward(&img)?; // [B, ph*pw, patch*patch*out_ch]

        // 6. unpatchify → [B, out_ch, H, W].
        self.unpatchify(&img, ph, pw)
    }

    /// `[B, ph*pw, patch*patch*out_ch]` → `[B, out_ch, ph*patch, pw*patch]`. Mirrors diffusers
    /// `SD3Transformer2DModel`'s unpatchify (`reshape → einsum nhwpqc→nchpwq → reshape`).
    fn unpatchify(&self, x: &Array, ph: i32, pw: i32) -> Result<Array> {
        let p = self.arch.patch_size as i32;
        let c = self.arch.out_channels as i32;
        let b = x.shape()[0];
        // [B, ph, pw, p, p, c]
        let x = x.reshape(&[b, ph, pw, p, p, c])?;
        // nhwpqc -> nchpwq : axes (0, 5, 1, 3, 2, 4)
        let x = x.transpose_axes(&[0, 5, 1, 3, 2, 4])?;
        // [B, c, ph*p, pw*p]
        Ok(x.reshape(&[b, c, ph * p, pw * p])?)
    }
}

// ----------------------------------------------------------------------------------------------
// Unit tests.
// ----------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{add, multiply};
    use mlx_rs::transforms::eval;

    /// Build an [`AdaLnContinuous`] whose projection is a pure bias: a zero weight matrix `[2H, E]`
    /// plus a bias `[2H]` equal to `concat(scale_vals, shift_vals)`. With a zero weight the projection
    /// output is exactly the bias regardless of `emb` (and of the inner `silu`), so `forward` must
    /// return that bias chunked in the diffusers `scale, shift = chunk(2)` order — which lets us pin
    /// the (shift, scale) labelling with asymmetric known values.
    fn ada_ln_continuous_pure_bias(
        hidden: i32,
        scale_vals: &[f32],
        shift_vals: &[f32],
    ) -> AdaLnContinuous {
        assert_eq!(scale_vals.len() as i32, hidden);
        assert_eq!(shift_vals.len() as i32, hidden);
        let mut w = Weights::empty();
        // dense Linear: `{prefix}.weight` is [out, in] = [2*hidden, emb_dim], `{prefix}.bias` [2*hidden].
        let emb_dim = hidden; // arbitrary; weight is zero so the value is irrelevant.
        w.insert(
            "norm.linear.weight",
            Array::zeros::<f32>(&[2 * hidden, emb_dim]).unwrap(),
        );
        let mut bias = Vec::with_capacity(2 * hidden as usize);
        bias.extend_from_slice(scale_vals); // diffusers FIRST chunk = scale
        bias.extend_from_slice(shift_vals); // diffusers SECOND chunk = shift
        w.insert("norm.linear.bias", Array::from_slice(&bias, &[2 * hidden]));
        AdaLnContinuous::from_weights(&w, "norm").unwrap()
    }

    /// Guards the diffusers `AdaLayerNormContinuous` chunk order: the FIRST projection chunk is
    /// `scale`, the SECOND is `shift` (the opposite of `AdaLayerNormZero`). A regression that swaps
    /// them — e.g. returning `(parts[0], parts[1])` as `(shift, scale)` — produces the wrong applied
    /// norm and corrupts every channel before `proj_out`, so this asserts BOTH the returned tuple
    /// AND the net application `norm(x)·(1+scale) + shift`.
    #[test]
    fn ada_ln_continuous_chunk_order_is_scale_then_shift() {
        let hidden = 4;
        // Deliberately asymmetric so a scale/shift swap can never coincidentally pass.
        let scale_vals = [0.5f32, -1.0, 2.0, 0.0];
        let shift_vals = [10.0f32, -20.0, 30.0, -40.0];
        let ada = ada_ln_continuous_pure_bias(hidden, &scale_vals, &shift_vals);

        let emb = Array::from_slice(&[0.3f32, -0.7, 1.1, 0.0], &[1, hidden]);
        let (shift, scale) = ada.forward(&emb).unwrap();
        eval([&shift, &scale]).unwrap();

        // Tuple is (shift, scale): shift == second chunk, scale == first chunk.
        let shift_v: Vec<f32> = shift.as_slice::<f32>().to_vec();
        let scale_v: Vec<f32> = scale.as_slice::<f32>().to_vec();
        for i in 0..hidden as usize {
            assert!(
                (shift_v[i] - shift_vals[i]).abs() < 1e-4,
                "shift[{i}] = {} expected {}",
                shift_v[i],
                shift_vals[i]
            );
            assert!(
                (scale_v[i] - scale_vals[i]).abs() < 1e-4,
                "scale[{i}] = {} expected {}",
                scale_v[i],
                scale_vals[i]
            );
        }

        // Net application must match diffusers: norm(x)·(1 + scale) + shift.
        let x = Array::from_slice(&[1.0f32, 2.0, -3.0, 4.0], &[1, 1, hidden]);
        let got = modulated_layer_norm(&x, &shift, &scale).unwrap();
        let normed = layer_norm(&x, None, None, LN_EPS).unwrap();
        let one = Array::from_slice(&[1.0f32], &[1]);
        let one_plus_scale = add(&scale, &one).unwrap();
        let scaled = multiply(&normed, &one_plus_scale).unwrap();
        let expected = add(&scaled, &shift).unwrap();
        eval([&got, &expected]).unwrap();
        let got_v: Vec<f32> = got.as_slice::<f32>().to_vec();
        let exp_v: Vec<f32> = expected.as_slice::<f32>().to_vec();
        for i in 0..got_v.len() {
            assert!(
                (got_v[i] - exp_v[i]).abs() < 1e-4,
                "modulated[{i}] = {} expected {}",
                got_v[i],
                exp_v[i]
            );
        }

        // Sanity: the SWAPPED application (the bug) would NOT match — proves the test discriminates.
        let one_plus_shift = add(&shift, &one).unwrap();
        let swap_scaled = multiply(&normed, &one_plus_shift).unwrap();
        let swapped = add(&swap_scaled, &scale).unwrap();
        eval([&swapped]).unwrap();
        let swap_v: Vec<f32> = swapped.as_slice::<f32>().to_vec();
        let mut differs = false;
        for i in 0..got_v.len() {
            if (got_v[i] - swap_v[i]).abs() > 1e-3 {
                differs = true;
            }
        }
        assert!(
            differs,
            "test cannot distinguish correct from swapped scale/shift"
        );
    }
}
