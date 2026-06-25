//! Attention + transformer blocks: `RotaryAttention` (pixel stream), `MMDiTJointAttention` +
//! `MMDiTBlockT2I` (dual-stream patch blocks), and `PiTBlock` (per-pixel block). Faithful port of
//! the corresponding classes in `pixeldit_official.py`.

use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::{concatenate_axis, pad, split};
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::{gated, modulate};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::layers::{lin, rms, FeedForward, Mlp};
use super::rope::apply_rope;

/// `[B,S,3·H·Dh]` → q,k,v each `[B,S,H,Dh]` via the reference's `reshape(B,S,3,H,Dh).permute(2,...)`.
fn split_qkv(qkv: &Array, heads: i32, head_dim: i32) -> Result<(Array, Array, Array)> {
    let sh = qkv.shape();
    let (b, s) = (sh[0], sh[1]);
    let q5 = qkv.reshape(&[b, s, 3, heads, head_dim])?;
    let parts = split(&q5, 3, 2)?;
    let take = |a: &Array| -> Result<Array> { Ok(a.reshape(&[b, s, heads, head_dim])?) };
    Ok((take(&parts[0])?, take(&parts[1])?, take(&parts[2])?))
}

/// `[B,H,S,Dh]` → `[B,S,H·Dh]`.
fn merge_heads(x: &Array) -> Result<Array> {
    let sh = x.shape();
    let (b, h, s, d) = (sh[0], sh[1], sh[2], sh[3]);
    Ok(x.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, h * d])?)
}

fn to_bhsd(x: &Array) -> Result<Array> {
    Ok(x.transpose_axes(&[0, 2, 1, 3])?) // [B,S,H,Dh] -> [B,H,S,Dh]
}

/// Flash-attention entry that stays on MLX's *fused* full-attention kernel for any head_dim.
///
/// MLX only flashes (never materializes the `[B,H,S,S]` scores) when `head_dim ∈ {64, 80, 128}`
/// (`ScaledDotProductAttention::use_fallback`). The pixel stream's head_dim is **72**, which falls to
/// the dense path and OOMs at SR-4K — pixel-stream attention is over `L = Hs·Ws` tokens (65 536 at
/// 4096²), so the dense scores are `[B, 16, 65536, 65536] ≈ 274 GB`. Zero-pad q/k/v's head_dim up to
/// the next supported size, flash, then slice the output back. The padded dims are zero in both q and
/// k so `QKᵀ` is unchanged, and `scale` stays the caller's original `head_dim^-0.5` — exact, not an
/// approximation. A head_dim already in the supported set (the patch stream's 64) takes the direct
/// path with no copy.
fn flash_sdpa(q: &Array, k: &Array, v: &Array, scale: f32) -> Result<Array> {
    let hd = q.shape()[3];
    match [64, 80, 128].into_iter().find(|&t| t >= hd) {
        Some(t) if t != hd => {
            let w = [(0, 0), (0, 0), (0, 0), (0, t - hd)];
            let qp = pad(q, &w[..], None, None)?;
            let kp = pad(k, &w[..], None, None)?;
            let vp = pad(v, &w[..], None, None)?;
            let o = scaled_dot_product_attention(&qp, &kp, &vp, scale, None, None)?;
            let idx = Array::from_slice(&(0..hd).collect::<Vec<i32>>(), &[hd]);
            Ok(o.take_axis(&idx, 3)?)
        }
        _ => Ok(scaled_dot_product_attention(q, k, v, scale, None, None)?),
    }
}

/// `RotaryAttention` — single-stream qk-normed rotary attention (the PiT pixel block's attention).
pub struct RotaryAttention {
    qkv: AdaptableLinear,
    q_norm: Array,
    k_norm: Array,
    proj: AdaptableLinear,
    heads: i32,
    head_dim: i32,
}

impl RotaryAttention {
    pub fn from_weights(w: &Weights, prefix: &str, dim: i32, heads: i32) -> Result<Self> {
        Ok(Self {
            qkv: lin(w, &format!("{prefix}.qkv"))?,
            q_norm: w.require(&format!("{prefix}.q_norm.weight"))?.clone(),
            k_norm: w.require(&format!("{prefix}.k_norm.weight"))?.clone(),
            proj: lin(w, &format!("{prefix}.proj"))?,
            heads,
            head_dim: dim / heads,
        })
    }

    /// `x`: `[B, N, dim]`; `cos`/`sin`: `[N, head_dim/2]`.
    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let (q, k, v) = split_qkv(&self.qkv.forward(x)?, self.heads, self.head_dim)?;
        let q = rms(&q, &self.q_norm)?;
        let k = rms(&k, &self.k_norm)?;
        let (q, k, v) = (to_bhsd(&q)?, to_bhsd(&k)?, to_bhsd(&v)?);
        let (q, k) = apply_rope(&q, &k, cos, sin)?;
        let scale = (self.head_dim as f32).powf(-0.5);
        let o = flash_sdpa(&q, &k, &v, scale)?;
        self.proj.forward(&merge_heads(&o)?)
    }
}

/// `MMDiTJointAttention` — separate img/txt QKV with per-stream qk-norm, RoPE (2-D on img, 1-D on
/// txt), a single joint SDPA over `[txt, img]`, then per-stream output projections.
pub struct MMDiTJointAttention {
    qkv_x: AdaptableLinear,
    qkv_y: AdaptableLinear,
    q_norm_x: Array,
    k_norm_x: Array,
    q_norm_y: Array,
    k_norm_y: Array,
    proj_x: AdaptableLinear,
    proj_y: AdaptableLinear,
    heads: i32,
    head_dim: i32,
}

impl MMDiTJointAttention {
    pub fn from_weights(w: &Weights, prefix: &str, dim: i32, heads: i32) -> Result<Self> {
        Ok(Self {
            qkv_x: lin(w, &format!("{prefix}.qkv_x"))?,
            qkv_y: lin(w, &format!("{prefix}.qkv_y"))?,
            q_norm_x: w.require(&format!("{prefix}.q_norm_x.weight"))?.clone(),
            k_norm_x: w.require(&format!("{prefix}.k_norm_x.weight"))?.clone(),
            q_norm_y: w.require(&format!("{prefix}.q_norm_y.weight"))?.clone(),
            k_norm_y: w.require(&format!("{prefix}.k_norm_y.weight"))?.clone(),
            proj_x: lin(w, &format!("{prefix}.proj_x"))?,
            proj_y: lin(w, &format!("{prefix}.proj_y"))?,
            heads,
            head_dim: dim / heads,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Array,
        y: &Array,
        cos_img: &Array,
        sin_img: &Array,
        cos_txt: &Array,
        sin_txt: &Array,
    ) -> Result<(Array, Array)> {
        let ny = y.shape()[1];
        let nx = x.shape()[1];

        let (qx, kx, vx) = split_qkv(&self.qkv_x.forward(x)?, self.heads, self.head_dim)?;
        let qx = rms(&qx, &self.q_norm_x)?;
        let kx = rms(&kx, &self.k_norm_x)?;
        let (qy, ky, vy) = split_qkv(&self.qkv_y.forward(y)?, self.heads, self.head_dim)?;
        let qy = rms(&qy, &self.q_norm_y)?;
        let ky = rms(&ky, &self.k_norm_y)?;

        let (qx, kx, vx) = (to_bhsd(&qx)?, to_bhsd(&kx)?, to_bhsd(&vx)?);
        let (qy, ky, vy) = (to_bhsd(&qy)?, to_bhsd(&ky)?, to_bhsd(&vy)?);
        let (qx, kx) = apply_rope(&qx, &kx, cos_img, sin_img)?;
        let (qy, ky) = apply_rope(&qy, &ky, cos_txt, sin_txt)?;

        // joint sequence [txt, img] along the token axis (axis 2 of [B,H,S,Dh])
        let q = concatenate_axis(&[&qy, &qx], 2)?;
        let k = concatenate_axis(&[&ky, &kx], 2)?;
        let v = concatenate_axis(&[&vy, &vx], 2)?;
        let scale = (self.head_dim as f32).powf(-0.5);
        let out = flash_sdpa(&q, &k, &v, scale)?;

        let txt_idx = Array::from_slice(&(0..ny).collect::<Vec<i32>>(), &[ny]);
        let img_idx = Array::from_slice(&(ny..ny + nx).collect::<Vec<i32>>(), &[nx]);
        let out_y = merge_heads(&out.take_axis(&txt_idx, 2)?)?;
        let out_x = merge_heads(&out.take_axis(&img_idx, 2)?)?;
        Ok((self.proj_x.forward(&out_x)?, self.proj_y.forward(&out_y)?))
    }
}

/// `MMDiTBlockT2I` — dual-stream block: joint attention + per-stream SwiGLU FFN, each gated by an
/// AdaLN modulation of the shared (already-SiLU'd) condition.
pub struct MMDiTBlockT2I {
    norm_x1: Array,
    norm_y1: Array,
    attn: MMDiTJointAttention,
    norm_x2: Array,
    norm_y2: Array,
    mlp_x: FeedForward,
    mlp_y: FeedForward,
    adaln_img: AdaptableLinear,
    adaln_txt: AdaptableLinear,
}

impl MMDiTBlockT2I {
    pub fn from_weights(w: &Weights, prefix: &str, dim: i32, heads: i32) -> Result<Self> {
        Ok(Self {
            norm_x1: w.require(&format!("{prefix}.norm_x1.weight"))?.clone(),
            norm_y1: w.require(&format!("{prefix}.norm_y1.weight"))?.clone(),
            attn: MMDiTJointAttention::from_weights(w, &format!("{prefix}.attn"), dim, heads)?,
            norm_x2: w.require(&format!("{prefix}.norm_x2.weight"))?.clone(),
            norm_y2: w.require(&format!("{prefix}.norm_y2.weight"))?.clone(),
            mlp_x: FeedForward::from_weights(w, &format!("{prefix}.mlp_x"))?,
            mlp_y: FeedForward::from_weights(w, &format!("{prefix}.mlp_y"))?,
            adaln_img: lin(w, &format!("{prefix}.adaLN_modulation_img.0"))?,
            adaln_txt: lin(w, &format!("{prefix}.adaLN_modulation_txt.0"))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Array,
        y: &Array,
        c: &Array,
        cos_img: &Array,
        sin_img: &Array,
        cos_txt: &Array,
        sin_txt: &Array,
    ) -> Result<(Array, Array)> {
        let mx = split(&self.adaln_img.forward(c)?, 6, -1)?;
        let my = split(&self.adaln_txt.forward(c)?, 6, -1)?;

        let x_norm = modulate(&rms(x, &self.norm_x1)?, &mx[1], &mx[0], false)?;
        let y_norm = modulate(&rms(y, &self.norm_y1)?, &my[1], &my[0], false)?;
        let (attn_x, attn_y) = self
            .attn
            .forward(&x_norm, &y_norm, cos_img, sin_img, cos_txt, sin_txt)?;
        let x = gated(x, &mx[2], &attn_x)?;
        let y = gated(y, &my[2], &attn_y)?;

        let x_mlp =
            self.mlp_x
                .forward(&modulate(&rms(&x, &self.norm_x2)?, &mx[4], &mx[3], false)?)?;
        let x = gated(&x, &mx[5], &x_mlp)?;
        let y_mlp =
            self.mlp_y
                .forward(&modulate(&rms(&y, &self.norm_y2)?, &my[4], &my[3], false)?)?;
        let y = gated(&y, &my[5], &y_mlp)?;
        Ok((x, y))
    }
}

/// `PiTBlock` — per-pixel block: compress the per-patch pixels to one attention token, rotary
/// attention across patch tokens, expand back, GELU MLP; both stages AdaLN-gated per pixel.
pub struct PiTBlock {
    compress_to_attn: AdaptableLinear,
    expand_from_attn: AdaptableLinear,
    norm1: Array,
    attn: RotaryAttention,
    norm2: Array,
    mlp: Mlp,
    adaln: AdaptableLinear,
    pixel_dim: i32,
    attn_dim: i32,
}

impl PiTBlock {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        pixel_dim: i32,
        attn_dim: i32,
        attn_heads: i32,
    ) -> Result<Self> {
        Ok(Self {
            compress_to_attn: lin(w, &format!("{prefix}.compress_to_attn"))?,
            expand_from_attn: lin(w, &format!("{prefix}.expand_from_attn"))?,
            norm1: w.require(&format!("{prefix}.norm1.weight"))?.clone(),
            attn: RotaryAttention::from_weights(
                w,
                &format!("{prefix}.attn"),
                attn_dim,
                attn_heads,
            )?,
            norm2: w.require(&format!("{prefix}.norm2.weight"))?.clone(),
            mlp: Mlp::from_weights(w, &format!("{prefix}.mlp"))?,
            adaln: lin(w, &format!("{prefix}.adaLN_modulation.0"))?,
            pixel_dim,
            attn_dim,
        })
    }

    /// `x`: `[B·L, P², pixel_dim]`; `s_cond`: `[B·L, context_dim]`; `cos`/`sin`: pixel-stream
    /// 2-D RoPE for the `(Hs, Ws)` patch grid. `b`/`l` are the batch and patch-grid token counts.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Array,
        s_cond: &Array,
        cos: &Array,
        sin: &Array,
        b: i32,
        l: i32,
    ) -> Result<Array> {
        let sh = x.shape();
        let (bl, p2) = (sh[0], sh[1]);
        let cond = self
            .adaln
            .forward(s_cond)?
            .reshape(&[bl, p2, 6 * self.pixel_dim])?;
        let m = split(&cond, 6, -1)?; // shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

        let x_norm = modulate(&rms(x, &self.norm1)?, &m[1], &m[0], false)?;
        let x_flat = x_norm.reshape(&[bl, p2 * self.pixel_dim])?;
        let x_comp = self
            .compress_to_attn
            .forward(&x_flat)?
            .reshape(&[b, l, self.attn_dim])?;
        let attn_out = self.attn.forward(&x_comp, cos, sin)?;
        let attn_exp = self
            .expand_from_attn
            .forward(&attn_out.reshape(&[b * l, self.attn_dim])?)?
            .reshape(&[bl, p2, self.pixel_dim])?;
        let x = gated(x, &m[2], &attn_exp)?;

        let mlp_out = self
            .mlp
            .forward(&modulate(&rms(&x, &self.norm2)?, &m[4], &m[3], false)?)?;
        gated(&x, &m[5], &mlp_out)
    }
}
