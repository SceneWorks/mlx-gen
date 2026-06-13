//! S2 — the Wan **2.1 `WanVAE`** (z16, stride 4×8×8): the 3-D causal-conv video VAE used by the
//! dense Wan2.1 / Wan2.2-14B path (the 5B uses the distinct z48 `vae22` — sc-2680). Port of the
//! `mlx_video` reference `models/wan/vae.py`, gated bit-for-bit against it (`tests/s2_parity.rs`).
//!
//! Tensors stay **NCTHW** (channels-first) throughout — mirroring the reference — and transpose to
//! channels-last only inside the conv ops (mlx convs are channels-last). Everything runs **f32**
//! (the reference upcasts the VAE to f32; f32 also sidesteps the bf16 NAX kernel history).
//!
//! Three reference quirks carried over verbatim:
//!  - The VAE "RMS_norm" is a **channel-L2 normalization** — `x / max(‖x‖₂ over C, 1e-12) · √C · γ`
//!    over axis 1 (not feature-RMS over the last axis). See [`rms_norm_channels`].
//!  - `CausalConv3d` pads time on the **left only** by `kt − st` (causal), with an optional
//!    `cache_x` left-context for the chunked encode. Spatial padding is symmetric `(kh−1)/2`.
//!  - Encode is **chunked** (frame 0 alone, then 4-frame chunks) with a persistent per-conv
//!    `feat_cache` of the last `CACHE_T` frames — reproducing the full-sequence causal result while
//!    bounding memory. Decode is a single non-causal pass (T latent → 4·T frames).

use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::{
    add, concatenate_axis, divide, maximum, minimum, multiply, pad, split, subtract, sum_axes,
};
use mlx_rs::Array;

use mlx_gen::nn::{conv2d, conv3d, silu, upsample_nearest};
use mlx_gen::tiling::{TilingConfig, VaeTiling};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::vae_common::{contiguous, scalar, slice_axis, tile_decode_accumulate, FeatCache};

/// Last-`CACHE_T` frames are carried across chunks as causal left-context during encode.
const CACHE_T: i32 = 2;
/// Channel-L2 norm floor (reference `mx.clip(..., a_min=1e-12)`).
const NORM_EPS: f32 = 1e-12;

/// Per-channel latent normalization statistics for z_dim=16 (reference `VAE_MEAN`/`VAE_STD`). These
/// are architecture constants (not learned), so they are hardcoded here and gated by the fixture.
const VAE_MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
    -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];
const VAE_STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.9160,
];

/// Wan2.1 VAE fixed structure (z16, dim_mult [1,2,4,4], 2 res-blocks/stage).
const DIM_MULT: [i32; 4] = [1, 2, 4, 4];
const NUM_RES_BLOCKS: usize = 2;
/// Decoder temporal-upsample per stage (`upsample3d` vs `upsample2d`); encoder is its mirror.
const TEMPORAL_UPSAMPLE: [bool; 3] = [true, true, false];
const TEMPORAL_DOWNSAMPLE: [bool; 3] = [false, true, true];

/// `x / max(‖x‖₂ over C, 1e-12) · √C · γ` — channel-L2 norm over axis 1. `x` is any rank with the
/// channel axis at index 1 (NCTHW or NCHW); `gamma` carries `C` elements in any shape.
fn rms_norm_channels(x: &Array, gamma: &Array) -> Result<Array> {
    let shape = x.shape();
    let nd = shape.len();
    let c = shape[1];
    let sum_sq = sum_axes(&multiply(x, x)?, &[1], true)?;
    let denom = maximum(&sum_sq, scalar(NORM_EPS))?.sqrt()?;
    let normed = divide(x, &denom)?;
    let scaled = multiply(&normed, scalar((c as f32).sqrt()))?;
    let mut wshape = vec![1i32; nd];
    wshape[1] = c;
    Ok(multiply(&scaled, &gamma.reshape(&wshape)?)?)
}

/// Last `n` frames along the temporal axis (axis 2): the reference `x[:, :, -n:]`.
fn last_t(x: &Array, n: i32) -> Result<Array> {
    let t = x.shape()[2];
    let idx: Vec<i32> = (t - n..t).collect();
    Ok(x.take_axis(Array::from_slice(&idx, &[n]), 2)?)
}

/// Temporal slice `x[:, :, start:end]` (axis 2).
fn slice_t(x: &Array, start: i32, end: i32) -> Result<Array> {
    slice_axis(x, 2, start, end)
}

/// 3-D conv with causal temporal left-pad (`kt − st`) + symmetric spatial pad `(kh−1)/2`. NCTHW
/// I/O; weight is the reference's already-MLX `[out, kt, kh, kw, in]`.
struct CausalConv3d {
    w: Array,
    b: Array,
    kt: i32,
    st: i32,
    ph: i32,
    pw: i32,
}

impl CausalConv3d {
    /// `st` is 1 everywhere except the encoder's temporal `downsample3d` `time_conv` (stride 2).
    fn from_weights(w: &Weights, prefix: &str, st: i32) -> Result<Self> {
        let weight = w.require(&format!("{prefix}.weight"))?.clone();
        let sh = weight.shape(); // [O, kt, kh, kw, I]
        let (kt, kh, kw) = (sh[1], sh[2], sh[3]);
        Ok(Self {
            w: weight,
            b: w.require(&format!("{prefix}.bias"))?.clone(),
            kt,
            st,
            ph: (kh - 1) / 2,
            pw: (kw - 1) / 2,
        })
    }

    fn forward(&self, x_ncthw: &Array, cache_x: Option<&Array>) -> Result<Array> {
        let mut x = x_ncthw.clone();
        let mut causal = self.kt - self.st;
        if let Some(cx) = cache_x {
            if causal > 0 {
                x = concatenate_axis(&[cx, &x], 2)?;
                causal = (causal - cx.shape()[2]).max(0);
            }
        }
        if causal > 0 || self.ph > 0 || self.pw > 0 {
            x = pad(
                &x,
                &[
                    (0, 0),
                    (0, 0),
                    (causal, 0),
                    (self.ph, self.ph),
                    (self.pw, self.pw),
                ][..],
                None,
                None,
            )?;
        }
        let x = x.transpose_axes(&[0, 2, 3, 4, 1])?; // NDHWC
        let y = conv3d(&x, &self.w, Some(&self.b), (self.st, 1, 1), (0, 0, 0))?;
        Ok(y.transpose_axes(&[0, 4, 1, 2, 3])?) // NCTHW
    }
}

/// Run a cached conv: feed the *previous* slot as left-context, then store this chunk's last frames.
/// Mirrors the reference's `cache_x = x[:, :, -CACHE_T:]` (+ 1-frame prepend when short) dance.
fn cached_conv(conv: &CausalConv3d, x: &Array, cache: &mut FeatCache) -> Result<Array> {
    let idx = cache.idx;
    let t = x.shape()[2];
    let mut cache_x = last_t(x, t.min(CACHE_T))?;
    if cache_x.shape()[2] < CACHE_T {
        if let Some(prev) = &cache.slots[idx] {
            cache_x = concatenate_axis(&[&last_t(prev, 1)?, &cache_x], 2)?;
        }
    }
    let y = conv.forward(x, cache.slots[idx].as_ref())?;
    cache.slots[idx] = Some(cache_x);
    cache.idx += 1;
    Ok(y)
}

/// `norm → SiLU → conv(3³) → norm → SiLU → conv(3³)` + residual (1³ skip when channels differ).
/// Reference list indices: `residual.{0,2,3,6}` (the SiLU/Dropout gaps carry no params).
struct ResidualBlock {
    norm1: Array,
    conv1: CausalConv3d,
    norm2: Array,
    conv2: CausalConv3d,
    shortcut: Option<CausalConv3d>,
}

impl ResidualBlock {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let shortcut = if w.get(&format!("{prefix}.shortcut.weight")).is_some() {
            Some(CausalConv3d::from_weights(
                w,
                &format!("{prefix}.shortcut"),
                1,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1: w.require(&format!("{prefix}.residual.0.gamma"))?.clone(),
            conv1: CausalConv3d::from_weights(w, &format!("{prefix}.residual.2"), 1)?,
            norm2: w.require(&format!("{prefix}.residual.3.gamma"))?.clone(),
            conv2: CausalConv3d::from_weights(w, &format!("{prefix}.residual.6"), 1)?,
            shortcut,
        })
    }

    fn shortcut(&self, x: &Array) -> Result<Array> {
        match &self.shortcut {
            Some(s) => s.forward(x, None),
            None => Ok(x.clone()),
        }
    }

    /// Decode path (no cache).
    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.shortcut(x)?;
        let y = self
            .conv1
            .forward(&silu(&rms_norm_channels(x, &self.norm1)?)?, None)?;
        let y = self
            .conv2
            .forward(&silu(&rms_norm_channels(&y, &self.norm2)?)?, None)?;
        Ok(add(&y, &h)?)
    }

    /// Encode path (chunked, with `feat_cache`).
    fn forward_cached(&self, x: &Array, cache: &mut FeatCache) -> Result<Array> {
        let h = self.shortcut(x)?;
        let y = silu(&rms_norm_channels(x, &self.norm1)?)?;
        let y = cached_conv(&self.conv1, &y, cache)?;
        let y = silu(&rms_norm_channels(&y, &self.norm2)?)?;
        let y = cached_conv(&self.conv2, &y, cache)?;
        Ok(add(&y, &h)?)
    }
}

/// Per-frame single-head spatial self-attention (head_dim = C). NCTHW I/O.
struct AttentionBlock {
    norm: Array,
    qkv_w: Array,
    qkv_b: Array,
    proj_w: Array,
    proj_b: Array,
}

impl AttentionBlock {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            norm: w.require(&format!("{prefix}.norm.gamma"))?.clone(),
            qkv_w: w.require(&format!("{prefix}.to_qkv.weight"))?.clone(),
            qkv_b: w.require(&format!("{prefix}.to_qkv.bias"))?.clone(),
            proj_w: w.require(&format!("{prefix}.proj.weight"))?.clone(),
            proj_b: w.require(&format!("{prefix}.proj.bias"))?.clone(),
        })
    }

    fn forward(&self, x_ncthw: &Array) -> Result<Array> {
        let sh = x_ncthw.shape();
        let (b, c, t, h, w) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let bt = b * t;
        // NCTHW -> (B·T, C, H, W), channel-L2 norm over C, then NHWC for the 1×1 convs.
        let x = x_ncthw
            .transpose_axes(&[0, 2, 1, 3, 4])?
            .reshape(&[bt, c, h, w])?;
        let normed = rms_norm_channels(&x, &self.norm)?.transpose_axes(&[0, 2, 3, 1])?;
        let qkv = conv2d(&normed, &self.qkv_w, Some(&self.qkv_b), 1, 0)?; // (BT,H,W,3C)
        let qkv = qkv.reshape(&[bt, h * w, 3 * c])?;
        let parts = split(&qkv, 3, 2)?; // q,k,v each (BT, H·W, C)
        let q = parts[0].expand_dims(1)?; // (BT, 1, H·W, C)
        let k = parts[1].expand_dims(1)?;
        let v = parts[2].expand_dims(1)?;
        let scale = (c as f32).powf(-0.5);
        let o = scaled_dot_product_attention(&q, &k, &v, scale, None, None)?;
        let o = o.reshape(&[bt, h, w, c])?;
        let o = conv2d(&o, &self.proj_w, Some(&self.proj_b), 1, 0)?; // (BT,H,W,C)
        let o = o
            .transpose_axes(&[0, 3, 1, 2])?
            .reshape(&[b, t, c, h, w])?
            .transpose_axes(&[0, 2, 1, 3, 4])?; // NCTHW
        Ok(add(&o, x_ncthw)?)
    }
}

/// Decoder spatial 2× upsample (`resample.1` = Conv2d C→C/2). `upsample3d` first doubles T via a
/// learned `time_conv` (C→2C, interleaved); `upsample2d` is spatial-only.
struct UpsampleBlock {
    conv_w: Array,
    conv_b: Array,
    time_conv: Option<CausalConv3d>,
}

impl UpsampleBlock {
    fn from_weights(w: &Weights, prefix: &str, temporal: bool) -> Result<Self> {
        let time_conv = if temporal {
            Some(CausalConv3d::from_weights(
                w,
                &format!("{prefix}.time_conv"),
                1,
            )?)
        } else {
            None
        };
        Ok(Self {
            conv_w: w.require(&format!("{prefix}.resample.1.weight"))?.clone(),
            conv_b: w.require(&format!("{prefix}.resample.1.bias"))?.clone(),
            time_conv,
        })
    }

    fn forward(&self, x_ncthw: &Array) -> Result<Array> {
        let sh = x_ncthw.shape();
        let (b, c) = (sh[0], sh[1]);
        let (mut x, mut t, h, w) = (x_ncthw.clone(), sh[2], sh[3], sh[4]);
        if let Some(tc) = &self.time_conv {
            // C→2C, then interleave the two halves into 2·T frames.
            let xt = tc.forward(&x, None)?.reshape(&[b, 2, c, t, h, w])?;
            x = xt
                .transpose_axes(&[0, 2, 3, 1, 4, 5])?
                .reshape(&[b, c, t * 2, h, w])?;
            t *= 2;
        }
        // Per-frame nearest-2× spatial upsample + 3×3 conv (C→C/2).
        let xs = x
            .transpose_axes(&[0, 2, 3, 4, 1])?
            .reshape(&[b * t, h, w, c])?;
        let up = upsample_nearest(&xs, 2)?;
        let y = conv2d(&up, &self.conv_w, Some(&self.conv_b), 1, 1)?;
        let c_out = y.shape()[3];
        Ok(y.reshape(&[b, t, h * 2, w * 2, c_out])?
            .transpose_axes(&[0, 4, 1, 2, 3])?)
    }
}

/// Encoder spatial 2× downsample (ZeroPad-(0,1,0,1) + stride-2 3×3 conv C→C). `downsample3d` adds a
/// temporal stride-2 `time_conv` with chunk-cache (first chunk passes through, later chunks fold the
/// previous chunk's last frame as left-context).
struct DownsampleBlock {
    conv_w: Array,
    conv_b: Array,
    time_conv: Option<CausalConv3d>,
}

impl DownsampleBlock {
    fn from_weights(w: &Weights, prefix: &str, temporal: bool) -> Result<Self> {
        let time_conv = if temporal {
            Some(CausalConv3d::from_weights(
                w,
                &format!("{prefix}.time_conv"),
                2,
            )?)
        } else {
            None
        };
        Ok(Self {
            conv_w: w.require(&format!("{prefix}.resample.1.weight"))?.clone(),
            conv_b: w.require(&format!("{prefix}.resample.1.bias"))?.clone(),
            time_conv,
        })
    }

    fn forward(&self, x_ncthw: &Array, cache: &mut FeatCache) -> Result<Array> {
        let sh = x_ncthw.shape();
        let (b, c, t, h, w) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let bt = b * t;
        // Per-frame ZeroPad(0,1,0,1) + valid stride-2 conv.
        let xs = x_ncthw
            .transpose_axes(&[0, 2, 3, 4, 1])?
            .reshape(&[bt, h, w, c])?;
        let xp = pad(&xs, &[(0, 0), (0, 1), (0, 1), (0, 0)][..], None, None)?;
        let y = conv2d(&xp, &self.conv_w, Some(&self.conv_b), 2, 0)?;
        let (h2, w2, c2) = (y.shape()[1], y.shape()[2], y.shape()[3]);
        let mut x = y
            .reshape(&[b, t, h2, w2, c2])?
            .transpose_axes(&[0, 4, 1, 2, 3])?; // NCTHW

        if let Some(tc) = &self.time_conv {
            let idx = cache.idx;
            if cache.slots[idx].is_none() {
                // First chunk: stash x, skip the temporal conv (no downsample this chunk).
                cache.slots[idx] = Some(x.clone());
            } else {
                let new_cache = last_t(&x, 1)?;
                let prev_last = last_t(cache.slots[idx].as_ref().unwrap(), 1)?;
                x = tc.forward(&x, Some(&prev_last))?;
                cache.slots[idx] = Some(new_cache);
            }
            cache.idx += 1;
        }
        Ok(x)
    }
}

/// One decoder up-stage entry: a residual block or a spatial/temporal upsample.
enum UpLayer {
    Res(ResidualBlock),
    Up(UpsampleBlock),
}

/// One encoder down-stage entry: a residual block or a spatial/temporal downsample.
enum DownLayer {
    Res(ResidualBlock),
    Down(DownsampleBlock),
}

/// `conv1 → [Res, Attn, Res] → upsamples → RMS+SiLU+conv` (z_dim → 3). Non-causal (single pass).
struct Decoder3d {
    conv1: CausalConv3d,
    middle: (ResidualBlock, AttentionBlock, ResidualBlock),
    upsamples: Vec<UpLayer>,
    head_norm: Array,
    head_conv: CausalConv3d,
}

impl Decoder3d {
    fn from_weights(w: &Weights) -> Result<Self> {
        // Structure (block counts, resample positions, temporal flags) is fixed by the 2.1 config;
        // channel sizes ride on the weights, so the flat `upsamples` indices are all that matter.
        let p = "decoder";
        let mut upsamples = Vec::new();
        let mut next = 0usize;
        for i in 0..DIM_MULT.len() {
            for _ in 0..(NUM_RES_BLOCKS + 1) {
                upsamples.push(UpLayer::Res(ResidualBlock::from_weights(
                    w,
                    &format!("{p}.upsamples.{next}"),
                )?));
                next += 1;
            }
            if let Some(&temporal) = TEMPORAL_UPSAMPLE.get(i) {
                upsamples.push(UpLayer::Up(UpsampleBlock::from_weights(
                    w,
                    &format!("{p}.upsamples.{next}"),
                    temporal,
                )?));
                next += 1;
            }
        }

        Ok(Self {
            conv1: CausalConv3d::from_weights(w, &format!("{p}.conv1"), 1)?,
            middle: (
                ResidualBlock::from_weights(w, &format!("{p}.middle.0"))?,
                AttentionBlock::from_weights(w, &format!("{p}.middle.1"))?,
                ResidualBlock::from_weights(w, &format!("{p}.middle.2"))?,
            ),
            upsamples,
            head_norm: w.require(&format!("{p}.head.0.gamma"))?.clone(),
            head_conv: CausalConv3d::from_weights(w, &format!("{p}.head.2"), 1)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = self.conv1.forward(x, None)?;
        x = self.middle.0.forward(&x)?;
        x = self.middle.1.forward(&x)?;
        x = self.middle.2.forward(&x)?;
        for layer in &self.upsamples {
            x = match layer {
                UpLayer::Res(r) => r.forward(&x)?,
                UpLayer::Up(u) => u.forward(&x)?,
            };
        }
        let x = silu(&rms_norm_channels(&x, &self.head_norm)?)?;
        self.head_conv.forward(&x, None)
    }
}

/// `conv1 → downsamples → [Res, Attn, Res] → RMS+SiLU+conv` (3 → z_dim·2). Chunked + cached.
struct Encoder3d {
    conv1: CausalConv3d,
    downsamples: Vec<DownLayer>,
    middle: (ResidualBlock, AttentionBlock, ResidualBlock),
    head_norm: Array,
    head_conv: CausalConv3d,
    cache_slots: usize,
}

impl Encoder3d {
    fn from_weights(w: &Weights) -> Result<Self> {
        let p = "encoder";
        let mut downsamples = Vec::new();
        let mut next = 0usize;
        let mut cache_slots = 1usize; // conv1
        for i in 0..DIM_MULT.len() {
            for _ in 0..NUM_RES_BLOCKS {
                downsamples.push(DownLayer::Res(ResidualBlock::from_weights(
                    w,
                    &format!("{p}.downsamples.{next}"),
                )?));
                next += 1;
                cache_slots += 2; // two cached convs per residual block
            }
            if let Some(&temporal) = TEMPORAL_DOWNSAMPLE.get(i) {
                downsamples.push(DownLayer::Down(DownsampleBlock::from_weights(
                    w,
                    &format!("{p}.downsamples.{next}"),
                    temporal,
                )?));
                next += 1;
                if temporal {
                    cache_slots += 1; // downsample3d time_conv
                }
            }
        }
        cache_slots += 4; // middle: 2 residual blocks × 2 convs
        cache_slots += 1; // head conv

        Ok(Self {
            conv1: CausalConv3d::from_weights(w, &format!("{p}.conv1"), 1)?,
            downsamples,
            middle: (
                ResidualBlock::from_weights(w, &format!("{p}.middle.0"))?,
                AttentionBlock::from_weights(w, &format!("{p}.middle.1"))?,
                ResidualBlock::from_weights(w, &format!("{p}.middle.2"))?,
            ),
            head_norm: w.require(&format!("{p}.head.0.gamma"))?.clone(),
            head_conv: CausalConv3d::from_weights(w, &format!("{p}.head.2"), 1)?,
            cache_slots,
        })
    }

    fn forward(&self, x: &Array, cache: &mut FeatCache) -> Result<Array> {
        let mut x = cached_conv(&self.conv1, x, cache)?;
        for layer in &self.downsamples {
            x = match layer {
                DownLayer::Res(r) => r.forward_cached(&x, cache)?,
                DownLayer::Down(d) => d.forward(&x, cache)?,
            };
        }
        x = self.middle.0.forward_cached(&x, cache)?;
        x = self.middle.1.forward(&x)?;
        x = self.middle.2.forward_cached(&x, cache)?;
        let x = silu(&rms_norm_channels(&x, &self.head_norm)?)?;
        cached_conv(&self.head_conv, &x, cache)
    }
}

/// The Wan 2.1 VAE: a decoder (always) + optional encoder (I2V), with per-channel latent
/// normalization. Decode latent → video; encode video → normalized latent.
pub struct WanVae {
    conv2: CausalConv3d,
    decoder: Decoder3d,
    encoder: Option<(CausalConv3d, Encoder3d)>, // (post-encoder conv1, encoder)
    mean: Array,                                // [1, z, 1, 1, 1]
    inv_std: Array,                             // [1, z, 1, 1, 1]
}

impl WanVae {
    /// Build from a weight map. Structure is fixed by the 2.1 config and channel sizes ride on the
    /// weights, so the same builder serves any `dim` (96 in production; tiny in the parity fixture).
    /// The encoder is loaded only if its weights are present.
    pub fn from_weights(w: &Weights) -> Result<Self> {
        let z = VAE_MEAN.len() as i32;
        let mean = Array::from_slice(&VAE_MEAN, &[1, z, 1, 1, 1]);
        let std = Array::from_slice(&VAE_STD, &[1, z, 1, 1, 1]);
        let inv_std = divide(scalar(1.0), &std)?;

        let encoder = if w.get("encoder.conv1.weight").is_some() {
            Some((
                CausalConv3d::from_weights(w, "conv1", 1)?,
                Encoder3d::from_weights(w)?,
            ))
        } else {
            None
        };

        Ok(Self {
            conv2: CausalConv3d::from_weights(w, "conv2", 1)?,
            decoder: Decoder3d::from_weights(w)?,
            encoder,
            mean,
            inv_std,
        })
    }

    /// Decode a normalized latent `[B, z, T, H, W]` → video `[B, 3, 4·T, 8·H, 8·W]` in `[-1, 1]`.
    pub fn decode(&self, z: &Array) -> Result<Array> {
        let denorm = add(&divide(z, &self.inv_std)?, &self.mean)?;
        let x = self.conv2.forward(&denorm, None)?;
        let out = self.decoder.forward(&x)?;
        contiguous(&minimum(&maximum(&out, scalar(-1.0))?, scalar(1.0))?)
    }

    /// Decode with **tiling** for memory-bounded large/long-video decode (`cfg`): split the latent
    /// into overlapping spatial/temporal tiles, decode each (conv2 + decoder + clamp), and
    /// trapezoidally blend them into the full video. Falls back to the single-pass [`decode`] when
    /// `cfg` doesn't fire for these dims. The Wan z16 VAE is **non-causal** in time (`T → 4·T`) and
    /// upsamples 8× spatially — [`VaeTiling::WAN`].
    ///
    /// Mirrors the reference `WanVAE.decode_tiled` (`models/wan/tiling.py`): **denormalize once** on
    /// the full (small) latent, then tile the denormalized latent and run only conv2+decoder+clip per
    /// tile. The full-size `output`/`weights` accumulators are filled tile-by-tile (pad-and-add) so
    /// peak memory stays bounded by one tile's decode. Shared tiling geometry: [`mlx_gen::tiling`].
    pub fn decode_tiled(&self, z: &Array, cfg: &TilingConfig) -> Result<Array> {
        let sh = z.shape();
        let (f, h, w) = (sh[2], sh[3], sh[4]);
        if !cfg.needs_tiling(VaeTiling::WAN, f, h, w) {
            return self.decode(z);
        }
        // Denormalize once (matches the reference), then tile the denormalized latent.
        let denorm = add(&divide(z, &self.inv_std)?, &self.mean)?;
        let plan = cfg.plan(VaeTiling::WAN, f, h, w);

        // NCTHW: channel axis at 1, tiled axes [2, 3, 4]. Per-tile decode = conv2 → decoder → clamp.
        tile_decode_accumulate(&denorm, &plan, [2, 3, 4], |tile| {
            let x = self.conv2.forward(tile, None)?;
            let dec = self.decoder.forward(&x)?;
            Ok(minimum(&maximum(&dec, scalar(-1.0))?, scalar(1.0))?)
        })
    }

    /// Run the chunked causal encoder + the post-encoder conv → the raw Gaussian moments
    /// `(mean, logvar)`, each `[B, z, T_lat, H/8, W/8]` (the `post_conv1` output split on the channel
    /// axis), **before** latent normalization — the reference `DiagonalGaussianDistribution.parameters`.
    fn encode_moments(&self, video: &Array) -> Result<(Array, Array)> {
        let (post_conv1, encoder) = self
            .encoder
            .as_ref()
            .ok_or_else(|| Error::Msg("WanVae: encode requires encoder weights".into()))?;

        let t = video.shape()[2];
        let num_chunks = 1 + (t - 1) / 4;
        let mut cache = FeatCache::new(encoder.cache_slots);
        let mut out: Option<Array> = None;
        for i in 0..num_chunks {
            cache.idx = 0;
            let chunk = if i == 0 {
                slice_t(video, 0, 1)
            } else {
                slice_t(video, 1 + 4 * (i - 1), 1 + 4 * i)
            }?;
            let chunk_out = encoder.forward(&chunk, &mut cache)?;
            out = Some(match out {
                None => chunk_out,
                Some(o) => concatenate_axis(&[&o, &chunk_out], 2)?,
            });
        }
        let parts = split(&post_conv1.forward(&out.unwrap(), None)?, 2, 1)?; // [mean, logvar]
        Ok((parts[0].clone(), parts[1].clone()))
    }

    /// Latent normalization by the z16 stats: `(x − mean)·inv_std`.
    fn normalize_latent(&self, x: &Array) -> Result<Array> {
        contiguous(&multiply(&subtract(x, &self.mean)?, &self.inv_std)?)
    }

    /// Encode a video `[B, 3, T, H, W]` (T = 1 + 4·k, values in `[-1, 1]`) → normalized latent
    /// `[B, z, T_lat, H/8, W/8]` via chunked causal encoding (`DiagonalGaussianDistribution.mode()` =
    /// the Gaussian mean). Requires encoder weights.
    pub fn encode(&self, video: &Array) -> Result<Array> {
        let (mean, _logvar) = self.encode_moments(video)?;
        self.normalize_latent(&mean)
    }

    /// Encode + **sample** the Gaussian (`DiagonalGaussianDistribution.sample()`): `mean +
    /// exp(0.5·clamp(logvar, −30, 20))·eps`, then latent-normalize. `eps` is standard-normal noise of
    /// the latent shape `[B, z, T_lat, H/8, W/8]`, taken as an argument so the result is deterministic
    /// (the reference draws it from the request seed). Bernini's `get_vae_features` uses this for
    /// **video** source conditioning; images use [`encode`] (`.mode()`).
    pub fn encode_sample(&self, video: &Array, eps: &Array) -> Result<Array> {
        let (mean, logvar) = self.encode_moments(video)?;
        self.normalize_latent(&reparameterize(&mean, &logvar, eps)?)
    }
}

/// `DiagonalGaussianDistribution.sample()`: `mean + exp(0.5·clamp(logvar, −30, 20))·eps`. The
/// `[−30, 20]` log-variance clamp is the diffusers default.
fn reparameterize(mean: &Array, logvar: &Array, eps: &Array) -> Result<Array> {
    let logvar = minimum(&maximum(logvar, scalar(-30.0))?, scalar(20.0))?;
    let std = multiply(&logvar, scalar(0.5))?.exp()?;
    Ok(add(mean, &multiply(&std, eps)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_channels_matches_closed_form() {
        // 2 channels, single spatial cell: ‖x‖₂ = √5, √C = √2 → out = x/√5·√2·γ.
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 2, 1, 1, 1]);
        let gamma = Array::from_slice(&[1.0f32, 1.0], &[2, 1, 1, 1]);
        let got = rms_norm_channels(&x, &gamma).unwrap();
        let got = got.as_slice::<f32>();
        let s = (2.0f32).sqrt() / (5.0f32).sqrt();
        assert!((got[0] - 1.0 * s).abs() < 1e-6);
        assert!((got[1] - 2.0 * s).abs() < 1e-6);
    }

    /// `reparameterize` = `mean + exp(0.5·clamp(logvar, −30, 20))·eps`. Checks the formula and the
    /// log-variance clamp at both ends (50 → 20, −50 → −30) against hand-computed values.
    #[test]
    fn reparameterize_matches_closed_form() {
        let mean = Array::from_slice(&[1.0f32, -2.0, 0.5, 3.0], &[4]);
        let logvar = Array::from_slice(&[0.0f32, 4.0, 50.0, -50.0], &[4]);
        let eps = Array::from_slice(&[2.0f32, 1.0, -1.0, 4.0], &[4]);
        let got = reparameterize(&mean, &logvar, &eps).unwrap();
        let got = got.as_slice::<f32>();
        // std = exp(0.5·clamp(logvar)): exp(0)=1, exp(2), exp(10) (clamped 20), exp(-15) (clamped -30).
        let want = [
            1.0 + 1.0 * 2.0,
            -2.0 + (2.0f32).exp() * 1.0,
            0.5 - (10.0f32).exp(),
            3.0 + (-15.0f32).exp() * 4.0,
        ];
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-3 * w.abs().max(1.0), "got {g} want {w}");
        }
    }
}
