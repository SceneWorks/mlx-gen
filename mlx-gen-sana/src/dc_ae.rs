//! DC-AE (deep compression autoencoder) **decoder** — faithful mlx-rs port of diffusers
//! `AutoencoderDC` (`mit-han-lab/dc-ae-f32c32-sana-1.0`), epic 8485 spike sc-8486.
//!
//! Scope is the decode path only — the spike's GO/NO-GO question is whether the f32 deep-compression
//! decode reproduces cleanly on Metal. The whole decoder runs in **f32** (the checkpoint is f32 and
//! the linear-attention normalizer is f32 in the reference regardless).
//!
//! Layout: everything is **channels-last NHWC** (mlx-native); diffusers' `movedim(1,-1)` before its
//! channel-wise Linear/RMSNorm ops is therefore a no-op here, and the conv weights (PyTorch
//! `[O, I/groups, H, W]`) are transposed to mlx `[O, H, W, I/groups]` at load. The one place that
//! needs the channels-first view is the multi-head reshape inside the linear attention, handled
//! locally by transposing to NCHW for that block of math and back.
//!
//! Block fidelity (vs the exact diffusers source):
//!  - `ResBlock`: `conv1 → SiLU → conv2(no-bias) → RMSNorm(channel) → + residual`.
//!  - `EfficientViTBlock`: `SanaMultiscaleLinearAttention → GLUMBConv`.
//!  - `SanaMultiscaleLinearAttention`: per-pixel `to_q/k/v` (no bias) → multiscale depthwise+grouped
//!    QKV projections → per-head `ReLU(Q),ReLU(K)` linear attention with a `1/(Σ+eps)` normalizer,
//!    computed here without the reference's ones-row `F.pad` via the algebraically identical
//!    numerator/denominator split (same f32 sums) → `to_out`(no bias) → RMSNorm → + residual.
//!  - `GLUMBConv`: `conv_inverted(1×1) → SiLU → conv_depth(3×3 depthwise) → gated SiLU → conv_point
//!    (1×1 no-bias) → RMSNorm → + residual`.
//!  - `DCUpBlock2d` (interpolate): `nearest-upsample → conv`, plus a channel shortcut
//!    `repeat_interleave → pixel_shuffle`.

use mlx_rs::ops::{
    add, broadcast_to, concatenate_axis, conv2d as conv2d_op, divide, matmul, mean_axes, multiply,
    split_sections, sum_axes,
};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::{silu, upsample_nearest};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::{BlockType, DcAeConfig};

const F32: Dtype = Dtype::Float32;

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

fn relu(x: &Array) -> Result<Array> {
    Ok(mlx_rs::nn::relu(x)?)
}

/// `PixelShuffle2D(r)` over NHWC: `(N, H, W, C·r²) → (N, H·r, W·r, C)`. (Mirrors the LTX upsampler.)
fn pixel_shuffle_2d(x: &Array, r: i32) -> Result<Array> {
    let sh = x.shape();
    let (n, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
    let out_c = c / (r * r);
    let x = x.reshape(&[n, h, w, out_c, r, r])?;
    let x = x.transpose_axes(&[0, 1, 4, 2, 5, 3])?;
    Ok(x.reshape(&[n, h * r, w * r, out_c])?)
}

/// `repeat_interleave` along the last (channel) axis: each channel duplicated `r` times in place.
fn repeat_interleave_last(x: &Array, r: i32) -> Result<Array> {
    let sh = x.shape();
    let c = sh[sh.len() - 1];
    let mut s1 = sh.to_vec();
    s1.push(1);
    let mut s2 = sh.to_vec();
    s2.push(r);
    let mut s3 = sh[..sh.len() - 1].to_vec();
    s3.push(c * r);
    Ok(broadcast_to(&x.reshape(&s1)?, &s2)?.reshape(&s3)?)
}

/// Channel-last RMSNorm over the last axis, computed in f32. `weight`/`bias` are per-channel `[C]`.
fn rms_norm(x: &Array, weight: &Array, bias: &Array, eps: f32) -> Result<Array> {
    let rank = x.shape().len();
    let ax = (rank - 1) as i32;
    let xf = x.as_dtype(F32)?;
    let var = mean_axes(&multiply(&xf, &xf)?, &[ax], true)?;
    let denom = add(&var, scalar(eps))?.sqrt()?;
    let normed = divide(&xf, &denom)?;
    Ok(add(&multiply(&normed, weight)?, bias)?)
}

/// Linear (no bias) over the last axis. `w_t` is the pre-transposed `[in, out]` weight.
fn linear_nb(x: &Array, w_t: &Array) -> Result<Array> {
    let sh = x.shape();
    let inn = sh[sh.len() - 1];
    let out = w_t.shape()[1];
    let n: i32 = sh[..sh.len() - 1].iter().product();
    let y = matmul(&x.reshape(&[n, inn])?, w_t)?;
    let mut outsh: Vec<i32> = sh[..sh.len() - 1].to_vec();
    outsh.push(out);
    Ok(y.reshape(&outsh)?)
}

/// A conv whose on-disk weight is PyTorch `[O, I/groups, H, W]`, transposed to mlx
/// `[O, H, W, I/groups]` at load. Stored f32.
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
            .transpose_axes(&[0, 2, 3, 1])?
            .as_dtype(F32)?;
        let b = if bias {
            Some(w.require(&format!("{prefix}.bias"))?.as_dtype(F32)?)
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
        let y = conv2d_op(
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

/// `ResBlock` (norm_type=rms_norm, act_fn=silu): `conv1 → SiLU → conv2(no-bias) → RMSNorm → +res`.
struct ResBlock {
    conv1: Conv,
    conv2: Conv,
    norm_w: Array,
    norm_b: Array,
    eps: f32,
}

impl ResBlock {
    fn load(w: &Weights, prefix: &str, eps: f32) -> Result<Self> {
        Ok(Self {
            conv1: Conv::load(w, &format!("{prefix}.conv1"), 1, 1, 1, true)?,
            conv2: Conv::load(w, &format!("{prefix}.conv2"), 1, 1, 1, false)?,
            norm_w: w.require(&format!("{prefix}.norm.weight"))?.as_dtype(F32)?,
            norm_b: w.require(&format!("{prefix}.norm.bias"))?.as_dtype(F32)?,
            eps,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.conv1.forward(x)?;
        let h = silu(&h)?;
        let h = self.conv2.forward(&h)?;
        let h = rms_norm(&h, &self.norm_w, &self.norm_b, self.eps)?;
        Ok(add(&h, x)?)
    }
}

/// One multiscale QKV projection: depthwise `proj_in` (kernel k, groups=channels) → grouped 1×1
/// `proj_out` (groups = 3·num_heads). Both bias-free. Operates on the NHWC `[B,H,W,3·inner]` qkv.
struct MultiscaleProj {
    proj_in: Conv,
    proj_out: Conv,
}

impl MultiscaleProj {
    fn load(w: &Weights, prefix: &str, kernel: i32, channels: i32, num_heads: i32) -> Result<Self> {
        Ok(Self {
            proj_in: Conv::load(
                w,
                &format!("{prefix}.proj_in"),
                1,
                kernel / 2,
                channels,
                false,
            )?,
            proj_out: Conv::load(w, &format!("{prefix}.proj_out"), 1, 0, 3 * num_heads, false)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        self.proj_out.forward(&self.proj_in.forward(x)?)
    }
}

/// `SanaMultiscaleLinearAttention` (residual). `inner == in_channels` for SANA-1.0 (head_dim 32).
struct LinearAttn {
    to_q: Array, // [in, inner] pre-transposed
    to_k: Array,
    to_v: Array,
    to_out: Array, // [inner·(1+scales), out] pre-transposed
    projs: Vec<MultiscaleProj>,
    norm_w: Array,
    norm_b: Array,
    head_dim: i32,
    num_heads: i32,
    norm_eps: f32,
    attn_eps: f32,
}

impl LinearAttn {
    fn load(w: &Weights, prefix: &str, cfg: &DcAeConfig, channels: i32) -> Result<Self> {
        let head_dim = cfg.attention_head_dim;
        let num_heads = channels / head_dim; // mult=1.0 → inner == channels
        let lin = |name: &str| -> Result<Array> {
            Ok(w.require(&format!("{prefix}.{name}.weight"))?
                .transpose_axes(&[1, 0])?
                .as_dtype(F32)?)
        };
        let mut projs = Vec::new();
        for (i, k) in cfg.qkv_multiscales.iter().enumerate() {
            projs.push(MultiscaleProj::load(
                w,
                &format!("{prefix}.to_qkv_multiscale.{i}"),
                *k,
                3 * channels, // proj operates over the concatenated [q,k,v] = 3·inner channels
                num_heads,
            )?);
        }
        Ok(Self {
            to_q: lin("to_q")?,
            to_k: lin("to_k")?,
            to_v: lin("to_v")?,
            to_out: lin("to_out")?,
            projs,
            norm_w: w
                .require(&format!("{prefix}.norm_out.weight"))?
                .as_dtype(F32)?,
            norm_b: w
                .require(&format!("{prefix}.norm_out.bias"))?
                .as_dtype(F32)?,
            head_dim,
            num_heads,
            norm_eps: cfg.norm_eps,
            attn_eps: cfg.attn_eps,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, h, w) = (sh[0], sh[1], sh[2]);
        let hd = self.head_dim;
        // qkv (NHWC), then concat the multiscale projections along channel.
        let q = linear_nb(x, &self.to_q)?;
        let k = linear_nb(x, &self.to_k)?;
        let v = linear_nb(x, &self.to_v)?;
        let qkv = concatenate_axis(&[q, k, v], -1)?; // [B,H,W,3·inner]
        let mut multi = vec![qkv.clone()];
        for proj in &self.projs {
            multi.push(proj.forward(&qkv)?);
        }
        let hidden = concatenate_axis(&multi, -1)?; // [B,H,W, 3·inner·(1+scales)]

        // → channels-first for the per-head reshape: [B, C_tot, H, W] → [B, heads·(1+scales), 3·hd, HW]
        let hidden = hidden.transpose_axes(&[0, 3, 1, 2])?.as_dtype(F32)?;
        let hw = h * w;
        let groups = self.num_heads * (1 + self.projs.len() as i32);
        let hidden = hidden.reshape(&[b, groups, 3 * hd, hw])?;
        // chunk(3) over the 3·hd axis → q,k,v each [B, groups, hd, HW]
        let parts = split_sections(&hidden, &[hd, 2 * hd], 2)?;
        let q = relu(&parts[0])?;
        let k = relu(&parts[1])?;
        let v = &parts[2];

        // Linear attention with 1/(Σ+eps) normalizer. The reference pads `value` with a ones-row and
        // divides by the last row; computed here as the algebraically identical numerator/denominator
        // split (all f32): num = (V·Kᵀ)·Q ; den = (Σ_hw K)·Q.
        let k_t = k.transpose_axes(&[0, 1, 3, 2])?; // [B,groups,HW,hd]
        let num = matmul(&matmul(v, &k_t)?, &q)?; // [B,groups,hd,HW]
        let k_sum = sum_axes(&k, &[3], true)?.transpose_axes(&[0, 1, 3, 2])?; // [B,groups,1,hd]
        let den = matmul(&k_sum, &q)?; // [B,groups,1,HW]
        let out = divide(&num, &add(&den, scalar(self.attn_eps))?)?; // broadcast over hd

        // → [B, inner·(1+scales), H, W] → NHWC, to_out, RMSNorm, residual.
        let out = out
            .reshape(&[b, groups * hd, h, w])?
            .transpose_axes(&[0, 2, 3, 1])?;
        let out = linear_nb(&out, &self.to_out)?;
        let out = rms_norm(&out, &self.norm_w, &self.norm_b, self.norm_eps)?;
        Ok(add(&out, x)?)
    }
}

/// `GLUMBConv` (rms_norm, residual, expand_ratio 4).
struct GluMbConv {
    conv_inverted: Conv, // 1×1, in → 2·hidden
    conv_depth: Conv,    // 3×3 depthwise, 2·hidden → 2·hidden
    conv_point: Conv,    // 1×1 no-bias, hidden → out
    norm_w: Array,
    norm_b: Array,
    hidden: i32,
    eps: f32,
}

impl GluMbConv {
    fn load(w: &Weights, prefix: &str, channels: i32, eps: f32) -> Result<Self> {
        let hidden = 4 * channels;
        Ok(Self {
            conv_inverted: Conv::load(w, &format!("{prefix}.conv_inverted"), 1, 0, 1, true)?,
            conv_depth: Conv::load(w, &format!("{prefix}.conv_depth"), 1, 1, 2 * hidden, true)?,
            conv_point: Conv::load(w, &format!("{prefix}.conv_point"), 1, 0, 1, false)?,
            norm_w: w.require(&format!("{prefix}.norm.weight"))?.as_dtype(F32)?,
            norm_b: w.require(&format!("{prefix}.norm.bias"))?.as_dtype(F32)?,
            hidden,
            eps,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.conv_inverted.forward(x)?;
        let h = silu(&h)?;
        let h = self.conv_depth.forward(&h)?;
        let parts = split_sections(&h, &[self.hidden], 3)?; // chunk(2) over channel (NHWC last axis)
        let h = multiply(&parts[0], &silu(&parts[1])?)?;
        let h = self.conv_point.forward(&h)?;
        let h = rms_norm(&h, &self.norm_w, &self.norm_b, self.eps)?;
        Ok(add(&h, x)?)
    }
}

enum Block {
    Res(ResBlock),
    Evit {
        attn: LinearAttn,
        conv_out: GluMbConv,
    },
}

impl Block {
    fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Block::Res(b) => b.forward(x),
            Block::Evit { attn, conv_out } => conv_out.forward(&attn.forward(x)?),
        }
    }
}

/// `DCUpBlock2d` (interpolate path): `nearest-upsample → conv`, + `repeat_interleave → pixel_shuffle`
/// channel shortcut.
struct UpBlock {
    conv: Conv,
    repeats: i32,
}

impl UpBlock {
    fn load(w: &Weights, prefix: &str, in_ch: i32, out_ch: i32) -> Result<Self> {
        Ok(Self {
            conv: Conv::load(w, &format!("{prefix}.conv"), 1, 1, 1, true)?,
            repeats: out_ch * 4 / in_ch,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let up = self.conv.forward(&upsample_nearest(x, 2)?)?;
        let shortcut = pixel_shuffle_2d(&repeat_interleave_last(x, self.repeats)?, 2)?;
        Ok(add(&up, &shortcut)?)
    }
}

struct Stage {
    upsample: Option<UpBlock>,
    blocks: Vec<Block>,
}

/// The full DC-AE decoder.
pub struct DcAeDecoder {
    cfg: DcAeConfig,
    conv_in: Conv,
    in_shortcut_repeats: i32,
    stages: Vec<Stage>, // storage order: shallow(0) → deep(n-1); decode iterates deep → shallow
    norm_out_w: Array,
    norm_out_b: Array,
    conv_out: Conv,
}

impl DcAeDecoder {
    pub fn from_weights(w: &Weights, cfg: DcAeConfig) -> Result<Self> {
        let n = cfg.num_stages();
        let deepest = cfg.block_out_channels[n - 1];
        let conv_in = Conv::load(w, "decoder.conv_in", 1, 1, 1, true)?;

        let mut stages = Vec::with_capacity(n);
        for i in 0..n {
            let ch = cfg.block_out_channels[i];
            // Stages 0..n-1 carry an upsample (storage slot `.0`); the deepest stage does not, so its
            // blocks start at slot `.0`. Block weights live under `decoder.up_blocks.{i}.{slot}`.
            let has_up = i + 1 < n;
            let upsample = if has_up {
                Some(UpBlock::load(
                    w,
                    &format!("decoder.up_blocks.{i}.0"),
                    cfg.block_out_channels[i + 1],
                    ch,
                )?)
            } else {
                None
            };
            let offset = if has_up { 1 } else { 0 };
            let mut blocks = Vec::new();
            for j in 0..cfg.layers_per_block[i] {
                let prefix = format!("decoder.up_blocks.{i}.{}", j + offset);
                let block = match cfg.block_types[i] {
                    BlockType::Res => Block::Res(ResBlock::load(w, &prefix, cfg.norm_eps)?),
                    BlockType::EfficientVit => Block::Evit {
                        attn: LinearAttn::load(w, &format!("{prefix}.attn"), &cfg, ch)?,
                        conv_out: GluMbConv::load(
                            w,
                            &format!("{prefix}.conv_out"),
                            ch,
                            cfg.norm_eps,
                        )?,
                    },
                };
                blocks.push(block);
            }
            stages.push(Stage { upsample, blocks });
        }

        Ok(Self {
            in_shortcut_repeats: deepest / cfg.latent_channels,
            conv_in,
            stages,
            norm_out_w: w.require("decoder.norm_out.weight")?.as_dtype(F32)?,
            norm_out_b: w.require("decoder.norm_out.bias")?.as_dtype(F32)?,
            conv_out: Conv::load(w, "decoder.conv_out", 1, 1, 1, true)?,
            cfg,
        })
    }

    /// Decode a latent `[B, latent_channels, h, w]` (channels-first, diffusers-native; **already
    /// un-scaled** by the caller) into an image `[B, H=32·h, W=32·w, 3]` (channels-last, f32).
    pub fn decode(&self, latent_nchw: &Array) -> Result<Array> {
        let latent = latent_nchw.transpose_axes(&[0, 2, 3, 1])?.as_dtype(F32)?; // → NHWC
        let shortcut = repeat_interleave_last(&latent, self.in_shortcut_repeats)?;
        let mut h = add(&self.conv_in.forward(&latent)?, &shortcut)?;
        for stage in self.stages.iter().rev() {
            if let Some(up) = &stage.upsample {
                h = up.forward(&h)?;
            }
            for block in &stage.blocks {
                h = block.forward(&h)?;
            }
        }
        let h = rms_norm(&h, &self.norm_out_w, &self.norm_out_b, self.cfg.norm_eps)?;
        let h = relu(&h)?;
        self.conv_out.forward(&h)
    }
}
