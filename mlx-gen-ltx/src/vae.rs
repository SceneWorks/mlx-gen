//! S2 — the LTX-2.3 **video VAE** (causal 3-D conv autoencoder, latent 128-ch, patch 4, 8× temporal
//! / 32× spatial). Port of the `mlx_video` reference `models/ltx/video_vae/*`, gated against it
//! (`tests/vae_parity.rs`, real `vae_{decoder,encoder}.safetensors`).
//!
//! Two distinct sub-nets, both **structure-from-config / channels-from-weights**:
//!  - **Decoder** = the reference `LTX2VideoDecoder` (`decoder.py`): `conv_in 128→1024` → 9
//!    `up_blocks` (`ResBlockGroup` + `DepthToSpaceUpsample`) → pixel-norm → SiLU → `conv_out 128→48`
//!    → unpatchify(×4). The shipped 2.3 checkpoint sets `timestep_conditioning=false`, so the
//!    decode-time noise / per-block scale-shift modulation are **inert** (no such weights) — kept
//!    config-gated, off, not dropped.
//!  - **Encoder** = the reference `VideoEncoder` (`video_vae.py`): patchify(×4) → `conv_in 48→128`
//!    → 9 `down_blocks` (`UNetMidBlock3D` + `SpaceToDepthDownsample`) → PixelNorm → SiLU →
//!    `conv_out 1024→129` → `normalize(out[:, :128])`. Wired here for the I2V sibling.
//!
//! Reference quirks carried over verbatim:
//!  - Tensors stay **NCTHW** (channels-first) throughout, transposing to channels-last only inside
//!    the conv op (mlx convs are channels-last). Conv weights are the on-disk MLX layout
//!    `[O, kt, kh, kw, I]` — no transpose.
//!  - `CausalConv3d` pads time by **frame replication**: causal → first frame ×(kt−1) at the start;
//!    non-causal → first frame ×(kt−1)/2 at the start *and* last frame ×(kt−1)/2 at the end. Spatial
//!    pad is symmetric `(k−1)/2` **zeros** (2.3 `spatial_padding_mode="zeros"`). The **encoder runs
//!    causal**, the **decoder non-causal** (`generate_av.py` calls `vae_decoder(latents)` →
//!    `causal=False`).
//!  - `pixel_norm` = `x / sqrt(mean(x² over C) + eps)` over axis 1 (no √C, no γ). Decoder eps
//!    **1e-8**, encoder PixelNorm eps **1e-6**.
//!
//! Everything runs **f32** (quality target; the reference VAE is gated in f32). The parity gate
//! honors "divergence is not rounding" — any >1% gap gets root-caused, not written off.

use mlx_rs::ops::{
    add, concatenate_axis, divide, maximum, mean_axes, multiply, pad, subtract, sum_axes,
};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::{conv3d, silu};
use mlx_gen::weights::{to_dtype, Weights};
use mlx_gen::{Error, Result};

use crate::config::{LtxVaeConfig, VaeBlock};
use mlx_gen::tiling::{TilingConfig, VaeTiling};

/// Decoder inline `pixel_norm` epsilon (`decoder.py` `ResnetBlock3DSimple.pixel_norm`).
const DEC_NORM_EPS: f32 = 1e-8;
/// Encoder `PixelNorm` epsilon (`get_norm_layer(..., eps=1e-6)`).
const ENC_NORM_EPS: f32 = 1e-6;

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// Load a conv weight+bias (`{prefix}.weight`, `{prefix}.bias`) cast to f32. The on-disk tensors are
/// already MLX conv layout `[O, kt, kh, kw, I]`, so no transpose is needed.
fn f32(w: &Weights, key: &str) -> Result<Array> {
    to_dtype(w.require(key)?, Dtype::Float32)
}

/// Force a logically-contiguous copy (see Wan `vae.rs`): host reads return the *physical* buffer, so
/// an array left strided by the final NDHWC→NCTHW transpose reads scrambled. Only needed at the
/// public decode/encode output boundary.
fn contiguous(x: &Array) -> Result<Array> {
    let shape = x.shape().to_vec();
    Ok(x.reshape(&[-1])?.reshape(&shape)?)
}

/// Slice `x` along `axis` to `[start, end)`.
fn slice_axis(x: &Array, axis: i32, start: i32, end: i32) -> Result<Array> {
    let idx: Vec<i32> = (start..end).collect();
    Ok(x.take_axis(Array::from_slice(&idx, &[end - start]), axis)?)
}

/// Temporal slice `x[:, :, start:end]` (axis 2).
fn slice_t(x: &Array, start: i32, end: i32) -> Result<Array> {
    slice_axis(x, 2, start, end)
}

/// `x / sqrt(mean(x² over C, axis 1, keepdims) + eps)` — LTX PixelNorm (no √C scale, no γ).
fn pixel_norm(x: &Array, eps: f32) -> Result<Array> {
    let sumsq = sum_axes(&multiply(x, x)?, &[1], true)?;
    let c = x.shape()[1] as f32;
    let mean = divide(&sumsq, scalar(c))?;
    let denom = add(&mean, scalar(eps))?.sqrt()?;
    Ok(divide(x, &denom)?)
}

/// 3-D conv with frame-replication temporal padding + symmetric spatial zero-pad. NCTHW I/O; weight
/// is the on-disk MLX `[O, kt, kh, kw, I]`. `causal` toggles left-only vs symmetric temporal pad.
struct CausalConv3d {
    w: Array,
    b: Array,
    kt: i32,
    ph: i32,
    pw: i32,
}

impl CausalConv3d {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let weight = f32(w, &format!("{prefix}.weight"))?;
        let sh = weight.shape(); // [O, kt, kh, kw, I]
        let (kt, kh, kw) = (sh[1], sh[2], sh[3]);
        Ok(Self {
            w: weight,
            b: f32(w, &format!("{prefix}.bias"))?,
            kt,
            ph: (kh - 1) / 2,
            pw: (kw - 1) / 2,
        })
    }

    /// Replicate `frame` (a `[B, C, 1, H, W]` slice) `n` times along the temporal axis.
    fn repeat_frame(frame: &Array, n: i32) -> Result<Array> {
        let parts: Vec<&Array> = (0..n).map(|_| frame).collect();
        Ok(concatenate_axis(&parts, 2)?)
    }

    fn forward(&self, x_ncthw: &Array, causal: bool) -> Result<Array> {
        let mut x = x_ncthw.clone();
        if self.kt > 1 {
            let t = x.shape()[2];
            if causal {
                let first = slice_t(&x, 0, 1)?;
                let pad_front = Self::repeat_frame(&first, self.kt - 1)?;
                x = concatenate_axis(&[&pad_front, &x], 2)?;
            } else {
                let ps = (self.kt - 1) / 2;
                if ps > 0 {
                    let first = slice_t(&x, 0, 1)?;
                    let last = slice_t(&x, t - 1, t)?;
                    let pad_front = Self::repeat_frame(&first, ps)?;
                    let pad_back = Self::repeat_frame(&last, ps)?;
                    x = concatenate_axis(&[&pad_front, &x, &pad_back], 2)?;
                }
            }
        }
        if self.ph > 0 || self.pw > 0 {
            x = pad(
                &x,
                &[
                    (0, 0),
                    (0, 0),
                    (0, 0),
                    (self.ph, self.ph),
                    (self.pw, self.pw),
                ][..],
                None,
                None,
            )?;
        }
        let x = x.transpose_axes(&[0, 2, 3, 4, 1])?; // NDHWC
        let y = conv3d(&x, &self.w, Some(&self.b), (1, 1, 1), (0, 0, 0))?;
        Ok(y.transpose_axes(&[0, 4, 1, 2, 3])?) // NCTHW
    }
}

// ---------------------------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------------------------

/// Decoder residual block (`ResnetBlock3DSimple`, timestep off): pixel-norm → SiLU → conv → repeat
/// → residual add. Channels constant (no shortcut). Inline pixel-norm eps 1e-8.
struct DecResBlock {
    conv1: CausalConv3d,
    conv2: CausalConv3d,
}

impl DecResBlock {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            conv1: CausalConv3d::from_weights(w, &format!("{prefix}.conv1.conv"))?,
            conv2: CausalConv3d::from_weights(w, &format!("{prefix}.conv2.conv"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = silu(&pixel_norm(x, DEC_NORM_EPS)?)?;
        let h = self.conv1.forward(&h, false)?;
        let h = silu(&pixel_norm(&h, DEC_NORM_EPS)?)?;
        let h = self.conv2.forward(&h, false)?;
        Ok(add(&h, x)?)
    }
}

/// `DepthToSpaceUpsample` with `residual=false`: conv → depth-to-space → (st>1) drop first temporal
/// frame. `stride = (st, sh, sw)`; out-channels = conv_out / prod(stride).
struct DepthToSpace {
    conv: CausalConv3d,
    st: i32,
    sh: i32,
    sw: i32,
}

impl DepthToSpace {
    fn from_weights(w: &Weights, prefix: &str, stride: (i32, i32, i32)) -> Result<Self> {
        Ok(Self {
            conv: CausalConv3d::from_weights(w, &format!("{prefix}.conv.conv"))?,
            st: stride.0,
            sh: stride.1,
            sw: stride.2,
        })
    }

    /// `(B, C·st·sh·sw, D, H, W) -> (B, C, D·st, H·sh, W·sw)`.
    fn depth_to_space(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, c_packed, d, h, w) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let (st, shp, swp) = (self.st, self.sh, self.sw);
        let c = c_packed / (st * shp * swp);
        let x = x.reshape(&[b, c, st, shp, swp, d, h, w])?;
        let x = x.transpose_axes(&[0, 1, 5, 2, 6, 3, 7, 4])?;
        Ok(x.reshape(&[b, c, d * st, h * shp, w * swp])?)
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let x = self.conv.forward(x, false)?;
        let x = self.depth_to_space(&x)?;
        if self.st > 1 {
            let t = x.shape()[2];
            slice_t(&x, 1, t)
        } else {
            Ok(x)
        }
    }
}

enum UpLayer {
    Res(Vec<DecResBlock>),
    Up(DepthToSpace),
}

/// The LTX-2.3 video decoder (`LTX2VideoDecoder`, timestep conditioning off).
struct VideoDecoder {
    conv_in: CausalConv3d,
    up_blocks: Vec<UpLayer>,
    conv_out: CausalConv3d,
    patch_size: i32,
    mean: Array, // [1, C, 1, 1, 1]
    std: Array,  // [1, C, 1, 1, 1]
}

impl VideoDecoder {
    fn from_weights(w: &Weights, cfg: &LtxVaeConfig) -> Result<Self> {
        let mut up_blocks = Vec::new();
        // `decoder_blocks` is listed in encoder order; the decoder execution path reverses it.
        for (idx, block) in cfg.decoder_blocks.iter().rev().enumerate() {
            let prefix = format!("up_blocks.{idx}");
            up_blocks.push(build_up_layer(w, &prefix, block)?);
        }
        let c = cfg.latent_channels;
        let mean = f32(w, "per_channel_statistics.mean")?.reshape(&[1, c, 1, 1, 1])?;
        let std = f32(w, "per_channel_statistics.std")?.reshape(&[1, c, 1, 1, 1])?;
        Ok(Self {
            conv_in: CausalConv3d::from_weights(w, "conv_in.conv")?,
            up_blocks,
            conv_out: CausalConv3d::from_weights(w, "conv_out.conv")?,
            patch_size: cfg.patch_size,
            mean,
            std,
        })
    }

    /// `(B, C, F', H', W')` normalized latent → `(B, 3, F, H, W)` video. Non-causal, deterministic.
    fn decode(&self, latent: &Array) -> Result<Array> {
        // Denormalize: x · std + mean.
        let x = add(&multiply(latent, &self.std)?, &self.mean)?;
        let mut x = self.conv_in.forward(&x, false)?;
        for layer in &self.up_blocks {
            x = match layer {
                UpLayer::Res(blocks) => {
                    let mut h = x;
                    for b in blocks {
                        h = b.forward(&h)?;
                    }
                    h
                }
                UpLayer::Up(u) => u.forward(&x)?,
            };
        }
        let x = pixel_norm(&x, DEC_NORM_EPS)?;
        let x = silu(&x)?;
        let x = self.conv_out.forward(&x, false)?;
        unpatchify(&x, self.patch_size)
    }
}

fn build_up_layer(w: &Weights, prefix: &str, block: &VaeBlock) -> Result<UpLayer> {
    if block.is_compress() {
        Ok(UpLayer::Up(DepthToSpace::from_weights(
            w,
            prefix,
            block.stride(),
        )?))
    } else {
        let mut blocks = Vec::new();
        for j in 0..block.num_layers {
            blocks.push(DecResBlock::from_weights(
                w,
                &format!("{prefix}.res_blocks.{j}"),
            )?);
        }
        Ok(UpLayer::Res(blocks))
    }
}

// ---------------------------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------------------------

/// Encoder residual block (`ResnetBlock3D`, PixelNorm eps 1e-6): norm → SiLU → conv → norm → SiLU →
/// conv → residual. Channels constant within a `UNetMidBlock3D` (no shortcut). Causal.
struct EncResBlock {
    conv1: CausalConv3d,
    conv2: CausalConv3d,
}

impl EncResBlock {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            conv1: CausalConv3d::from_weights(w, &format!("{prefix}.conv1.conv"))?,
            conv2: CausalConv3d::from_weights(w, &format!("{prefix}.conv2.conv"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = silu(&pixel_norm(x, ENC_NORM_EPS)?)?;
        let h = self.conv1.forward(&h, true)?;
        let h = silu(&pixel_norm(&h, ENC_NORM_EPS)?)?;
        let h = self.conv2.forward(&h, true)?;
        Ok(add(&h, x)?)
    }
}

/// `SpaceToDepthDownsample`: conv → space-to-depth (conv branch) + group-mean(space-to-depth(input))
/// skip. `out_channels` and `group_size` are derived from the conv weight + stride.
struct SpaceToDepth {
    conv: CausalConv3d,
    st: i32,
    sh: i32,
    sw: i32,
    out_channels: i32,
    group_size: i32,
}

impl SpaceToDepth {
    fn from_weights(w: &Weights, prefix: &str, stride: (i32, i32, i32)) -> Result<Self> {
        let conv = CausalConv3d::from_weights(w, &format!("{prefix}.conv.conv"))?;
        let csh = conv.w.shape(); // [O, kt, kh, kw, I]
        let (conv_out, in_channels) = (csh[0], csh[4]);
        let mult = stride.0 * stride.1 * stride.2;
        let out_channels = conv_out * mult; // after space-to-depth on the conv branch
        let group_size = in_channels * mult / out_channels;
        Ok(Self {
            conv,
            st: stride.0,
            sh: stride.1,
            sw: stride.2,
            out_channels,
            group_size,
        })
    }

    /// `(B, C, D, H, W) -> (B, C·st·sh·sw, D/st, H/sh, W/sw)`.
    fn space_to_depth(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, c, d, h, w) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let (st, shp, swp) = (self.st, self.sh, self.sw);
        let x = x.reshape(&[b, c, d / st, st, h / shp, shp, w / swp, swp])?;
        let x = x.transpose_axes(&[0, 1, 3, 5, 7, 2, 4, 6])?;
        Ok(x.reshape(&[b, c * st * shp * swp, d / st, h / shp, w / swp])?)
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        // Causal temporal pad: duplicate the first frame when downsampling time.
        if self.st == 2 {
            let first = slice_t(&x, 0, 1)?;
            x = concatenate_axis(&[&first, &x], 2)?;
        }
        // Pad the tail so D/H/W are divisible by stride (zeros, end-only).
        let sh = x.shape();
        let (d, h, w) = (sh[2], sh[3], sh[4]);
        let pad_d = (self.st - d % self.st) % self.st;
        let pad_h = (self.sh - h % self.sh) % self.sh;
        let pad_w = (self.sw - w % self.sw) % self.sw;
        if pad_d > 0 || pad_h > 0 || pad_w > 0 {
            x = pad(
                &x,
                &[(0, 0), (0, 0), (0, pad_d), (0, pad_h), (0, pad_w)][..],
                None,
                None,
            )?;
        }

        // Skip branch: space-to-depth on the input, then mean over the group axis.
        let x_in = self.space_to_depth(&x)?;
        let si = x_in.shape();
        let (b2, d2, h2, w2) = (si[0], si[2], si[3], si[4]);
        let x_in = x_in.reshape(&[b2, self.out_channels, self.group_size, d2, h2, w2])?;
        let x_in = mean_axes(&x_in, &[2], false)?; // (B, out_channels, D', H', W')

        // Conv branch: conv → space-to-depth.
        let x_conv = self.conv.forward(&x, true)?;
        let x_conv = self.space_to_depth(&x_conv)?;

        if x_conv.shape() == x_in.shape() {
            Ok(add(&x_conv, &x_in)?)
        } else {
            Ok(x_conv)
        }
    }
}

enum DownLayer {
    Res(Vec<EncResBlock>),
    Down(SpaceToDepth),
}

/// The LTX-2.3 video encoder (`VideoEncoder`, UNIFORM logvar). Causal throughout.
struct VideoEncoder {
    conv_in: CausalConv3d,
    down_blocks: Vec<DownLayer>,
    conv_out: CausalConv3d,
    patch_size: i32,
    latent_channels: i32,
    mean: Array, // [1, C, 1, 1, 1]
    std: Array,  // [1, C, 1, 1, 1]
}

impl VideoEncoder {
    fn from_weights(w: &Weights, cfg: &LtxVaeConfig) -> Result<Self> {
        let mut down_blocks = Vec::new();
        for (idx, block) in cfg.encoder_blocks.iter().enumerate() {
            let prefix = format!("down_blocks.{idx}");
            down_blocks.push(if block.is_compress() {
                DownLayer::Down(SpaceToDepth::from_weights(w, &prefix, block.stride())?)
            } else {
                let mut blocks = Vec::new();
                for j in 0..block.num_layers {
                    blocks.push(EncResBlock::from_weights(
                        w,
                        &format!("{prefix}.res_blocks.{j}"),
                    )?);
                }
                DownLayer::Res(blocks)
            });
        }
        let c = cfg.latent_channels;
        let mean = f32(w, "per_channel_statistics._mean_of_means")?.reshape(&[1, c, 1, 1, 1])?;
        let std = f32(w, "per_channel_statistics._std_of_means")?.reshape(&[1, c, 1, 1, 1])?;
        Ok(Self {
            conv_in: CausalConv3d::from_weights(w, "conv_in.conv")?,
            down_blocks,
            conv_out: CausalConv3d::from_weights(w, "conv_out.conv")?,
            patch_size: cfg.patch_size,
            latent_channels: c,
            mean,
            std,
        })
    }

    /// `(B, 3, F, H, W)` video (F = 1 + 8·k, values in [-1, 1]) → `(B, 128, F', H/32, W/32)`
    /// normalized latent means. Causal, deterministic (UNIFORM logvar → output = normalize(means);
    /// the discarded log-variance tail is not computed).
    fn encode(&self, video: &Array) -> Result<Array> {
        let mut x = patchify(video, self.patch_size)?;
        x = self.conv_in.forward(&x, true)?;
        for layer in &self.down_blocks {
            x = match layer {
                DownLayer::Res(blocks) => {
                    let mut h = x;
                    for b in blocks {
                        h = b.forward(&h)?;
                    }
                    h
                }
                DownLayer::Down(d) => d.forward(&x)?,
            };
        }
        let x = pixel_norm(&x, ENC_NORM_EPS)?;
        let x = silu(&x)?;
        // conv_out → latent_channels + 1. UNIFORM logvar: the output is normalize(means) where
        // means = the first `latent_channels`; the log-variance tail is discarded.
        let x = self.conv_out.forward(&x, true)?;
        let means = slice_c(&x, 0, self.latent_channels)?;
        let normed = divide(&subtract(&means, &self.mean)?, &self.std)?;
        Ok(normed)
    }
}

/// Channel slice `x[:, start:end]` (axis 1).
fn slice_c(x: &Array, start: i32, end: i32) -> Result<Array> {
    slice_axis(x, 1, start, end)
}

// ---------------------------------------------------------------------------------------------
// patchify / unpatchify (spatial-only, patch_size_t = 1)
// ---------------------------------------------------------------------------------------------

/// `(B, C, F, H, W) -> (B, C·p², F, H/p, W/p)` (reference `ops.patchify`, `patch_size_t=1`).
fn patchify(x: &Array, p: i32) -> Result<Array> {
    let sh = x.shape();
    let (b, c, f, h, w) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
    let (nh, nw) = (h / p, w / p);
    // (B, C, F, 1, H/p, p, W/p, p) -> transpose (0,1,3,7,5,2,4,6) -> (B, C·p·p, F, H/p, W/p).
    let x = x.reshape(&[b, c, f, 1, nh, p, nw, p])?;
    let x = x.transpose_axes(&[0, 1, 3, 7, 5, 2, 4, 6])?;
    Ok(x.reshape(&[b, c * p * p, f, nh, nw])?)
}

/// `(B, C·p², F, H, W) -> (B, C, F, H·p, W·p)` (reference `ops.unpatchify`, `patch_size_t=1`).
fn unpatchify(x: &Array, p: i32) -> Result<Array> {
    let sh = x.shape();
    let (b, c_packed, f, h, w) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
    let c = c_packed / (p * p);
    // (B, C, 1, p, p, F, H, W) -> transpose (0,1,5,2,6,4,7,3) -> (B, C, F, H·p, W·p).
    let x = x.reshape(&[b, c, 1, p, p, f, h, w])?;
    let x = x.transpose_axes(&[0, 1, 5, 2, 6, 4, 7, 3])?;
    Ok(x.reshape(&[b, c, f, h * p, w * p])?)
}

// ---------------------------------------------------------------------------------------------
// Public VAE
// ---------------------------------------------------------------------------------------------

/// The LTX-2.3 video VAE: a decoder (always) + an optional encoder (loaded only when its weights
/// are present — the I2V sibling needs it; pure T2V uses only `decode`).
pub struct LtxVideoVae {
    decoder: VideoDecoder,
    encoder: Option<VideoEncoder>,
}

impl LtxVideoVae {
    /// Build the decoder from `decoder_w` and (optionally) the encoder from `encoder_w`.
    pub fn from_weights(
        decoder_w: &Weights,
        encoder_w: Option<&Weights>,
        cfg: &LtxVaeConfig,
    ) -> Result<Self> {
        let decoder = VideoDecoder::from_weights(decoder_w, cfg)?;
        let encoder = match encoder_w {
            Some(w) => Some(VideoEncoder::from_weights(w, cfg)?),
            None => None,
        };
        Ok(Self { decoder, encoder })
    }

    /// Decode a normalized latent `(B, 128, F', H', W')` → video `(B, 3, F, 32·H', 32·W')` in
    /// roughly [-1, 1] (the caller clips + scales to uint8). Non-causal single pass.
    pub fn decode(&self, latent: &Array) -> Result<Array> {
        contiguous(&self.decoder.decode(latent)?)
    }

    /// Encode a video `(B, 3, F, H, W)` (F = 1 + 8·k, [-1, 1]) → normalized latent
    /// `(B, 128, F', H/32, W/32)`. Causal. Requires encoder weights.
    pub fn encode(&self, video: &Array) -> Result<Array> {
        let enc = self
            .encoder
            .as_ref()
            .ok_or_else(|| Error::Msg("LtxVideoVae: encode requires encoder weights".into()))?;
        contiguous(&enc.encode(video)?)
    }

    /// Decode with **tiling** for memory-bounded large/long-video decode (`cfg`). Splits the latent
    /// into overlapping spatial/temporal tiles, decodes each, and trapezoidally blends them. Falls
    /// back to the single-pass [`decode`](Self::decode) when `cfg` does not fire for these dims.
    pub fn decode_tiled(&self, latent: &Array, cfg: &TilingConfig) -> Result<Array> {
        let sh = latent.shape();
        let (f, h, w) = (sh[2], sh[3], sh[4]);
        if !cfg.needs_tiling(VaeTiling::LTX, f, h, w) {
            return self.decode(latent);
        }
        let plan = cfg.plan(VaeTiling::LTX, f, h, w);

        // Full-size accumulators (the reference allocates these too); pad-and-add each tile in turn.
        // `output` carries the batch; `weights` stays `b=1` and broadcasts on the final divide.
        let mut output: Option<Array> = None; // [b, 3, out_f, out_h, out_w]
        let mut weights: Option<Array> = None; // [1, 1, out_f, out_h, out_w]

        for t in &plan.t {
            for hh in &plan.h {
                for ww in &plan.w {
                    let tile = slice_axis(latent, 2, t.start, t.end)?;
                    let tile = slice_axis(&tile, 3, hh.start, hh.end)?;
                    let tile = slice_axis(&tile, 4, ww.start, ww.end)?;
                    let dec = self.decoder.decode(&tile)?; // [b, 3, td, hd, wd]

                    let ds = dec.shape();
                    let at = ds[2].min(t.out_stop - t.out_start);
                    let ah = ds[3].min(hh.out_stop - hh.out_start);
                    let aw = ds[4].min(ww.out_stop - ww.out_start);

                    // 1-D masks → outer product [1, 1, at, ah, aw].
                    let tm = Array::from_slice(&t.mask[..at as usize], &[1, 1, at, 1, 1]);
                    let hm = Array::from_slice(&hh.mask[..ah as usize], &[1, 1, 1, ah, 1]);
                    let wm = Array::from_slice(&ww.mask[..aw as usize], &[1, 1, 1, 1, aw]);
                    let blend = multiply(&multiply(&tm, &hm)?, &wm)?;

                    let dec = slice_axis(&dec, 2, 0, at)?;
                    let dec = slice_axis(&dec, 3, 0, ah)?;
                    let dec = slice_axis(&dec, 4, 0, aw)?;
                    let weighted = multiply(&dec, &blend)?; // [b, 3, at, ah, aw]

                    // Place at (out_start) offsets via zero-pad to the full output shape.
                    let pads = [
                        (0, 0),
                        (0, 0),
                        (t.out_start, plan.out_f - (t.out_start + at)),
                        (hh.out_start, plan.out_h - (hh.out_start + ah)),
                        (ww.out_start, plan.out_w - (ww.out_start + aw)),
                    ];
                    let weighted_full = pad(&weighted, &pads[..], None, None)?;
                    let blend_full = pad(&blend, &pads[..], None, None)?;

                    output = Some(match output {
                        None => weighted_full,
                        Some(acc) => add(&acc, &weighted_full)?,
                    });
                    weights = Some(match weights {
                        None => blend_full,
                        Some(acc) => add(&acc, &blend_full)?,
                    });
                    // Keep the lazy graph + peak memory bounded (mirrors the reference mx.eval).
                    let out_ref = output.as_ref().unwrap();
                    let w_ref = weights.as_ref().unwrap();
                    out_ref.eval()?;
                    w_ref.eval()?;
                }
            }
        }

        let output = output.expect("at least one tile");
        let weights = weights.expect("at least one tile");
        let normed = divide(&output, &maximum(&weights, scalar(1e-8))?)?;
        contiguous(&normed)
    }

    /// Whether the encoder is loaded.
    pub fn has_encoder(&self) -> bool {
        self.encoder.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_norm_matches_closed_form() {
        // 2 channels, single cell: mean(x²) = (1+4)/2 = 2.5 → denom = √(2.5 + eps).
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 2, 1, 1, 1]);
        let got = pixel_norm(&x, 0.0).unwrap();
        let got = got.as_slice::<f32>();
        let denom = (2.5f32).sqrt();
        assert!((got[0] - 1.0 / denom).abs() < 1e-6);
        assert!((got[1] - 2.0 / denom).abs() < 1e-6);
    }

    #[test]
    fn patchify_unpatchify_round_trip() {
        // (1, 1, 1, 4, 4) ascending → patchify(p=2) → unpatchify(p=2) restores the original.
        let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let x = Array::from_slice(&data, &[1, 1, 1, 4, 4]);
        let p = patchify(&x, 2).unwrap();
        assert_eq!(p.shape(), &[1, 4, 1, 2, 2]);
        let u = unpatchify(&p, 2).unwrap();
        assert_eq!(u.shape(), &[1, 1, 1, 4, 4]);
        let u = contiguous(&u).unwrap();
        for (a, b) in u.as_slice::<f32>().iter().zip(data.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn vae_block_stride_from_kind() {
        assert_eq!(
            VaeBlock {
                kind: "compress_space_res".into(),
                num_layers: 0,
                multiplier: 2
            }
            .stride(),
            (1, 2, 2)
        );
        assert_eq!(
            VaeBlock {
                kind: "compress_time".into(),
                num_layers: 0,
                multiplier: 2
            }
            .stride(),
            (2, 1, 1)
        );
        assert_eq!(
            VaeBlock {
                kind: "compress_all".into(),
                num_layers: 0,
                multiplier: 1
            }
            .stride(),
            (2, 2, 2)
        );
    }
}
