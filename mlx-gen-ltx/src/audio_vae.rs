//! S3 — the LTX-2.3 **audio VAE decoder** (sc-2684). Port of the `mlx_video` reference
//! `models/ltx/audio_vae/*` as configured for the shipped `audio_vae.safetensors`: a 2-D conv
//! autoencoder decoder, **causal on the height (time) axis**, PIXEL norm, `ch 128`, `ch_mult
//! (1,2,4)`, `z_channels 8`, `out_ch 2` (stereo), `num_res_blocks 2`, nearest-2× causal upsample.
//! Decodes the audio latent `(B, 8, T, 16)` → mel/STFT-domain spectrogram `(B, 2, 4T−3, 64)`.
//!
//! Structure-from-config / channels-from-weights, like the video VAE. Runs **f32** (a post-sampling
//! quality island; the gate isolates correctness from bf16 rounding). Tensors are NHWC channels-last
//! (mlx convs are channels-last); conv weights are the on-disk `[O, kH, kW, I]` layout (no transpose).
//!
//! **`mid_block_add_attention` is `false`** for the shipped checkpoint (no `mid.attn_1` weights), so
//! the mid block is `ResnetBlock → ResnetBlock`. The [`AttnBlock`] is implemented + config-gated for a
//! future `true`-with-weights checkpoint; see [`crate::config::AudioVaeConfig`] for the reference-bug
//! note (its `load_audio_decoder` builds a random-weight mid attention).

use mlx_rs::ops::{add, broadcast_to, divide, matmul, multiply, pad, softmax_axis};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::{conv2d, silu};
use mlx_gen::weights::{to_dtype, Weights};
use mlx_gen::Result;

use crate::config::AudioVaeConfig;

/// PixelNorm epsilon (`build_normalization_layer(..., NormType.PIXEL)` → `PixelNorm(eps=1e-6)`).
const PIXEL_EPS: f32 = 1e-6;

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

fn f32(w: &Weights, key: &str) -> Result<Array> {
    to_dtype(w.require(key)?, Dtype::Float32)
}

/// Per-location RMS over the channel (last) axis: `x / sqrt(mean(x², C) + eps)`. NHWC, no learned γ.
fn pixel_norm(x: &Array) -> Result<Array> {
    let sq = multiply(x, x)?;
    let mean = mlx_rs::ops::mean_axes(&sq, &[-1], true)?;
    let rms = add(&mean, scalar(PIXEL_EPS))?.sqrt()?;
    Ok(divide(x, &rms)?)
}

/// A 2-D convolution with asymmetric (causal-on-height) or symmetric padding applied manually, then
/// `conv2d(padding=0)`. NHWC; weight `[O, kH, kW, I]`.
struct CausalConv2d {
    w: Array,
    b: Array,
    pad_top: i32,
    pad_bottom: i32,
    pad_left: i32,
    pad_right: i32,
}

impl CausalConv2d {
    /// `causal_height = true` → pad the full `kH−1` on top (time is causal); width is symmetric.
    fn load(w: &Weights, prefix: &str, causal_height: bool) -> Result<Self> {
        let weight = f32(w, &format!("{prefix}.weight"))?; // (O, kH, kW, I)
        let bias = f32(w, &format!("{prefix}.bias"))?;
        let sh = weight.shape();
        let (kh, kw) = (sh[1], sh[2]);
        let (ph, pw) = (kh - 1, kw - 1);
        let (pad_top, pad_bottom) = if causal_height {
            (ph, 0)
        } else {
            (ph / 2, ph - ph / 2)
        };
        Ok(Self {
            w: weight,
            b: bias,
            pad_top,
            pad_bottom,
            pad_left: pw / 2,
            pad_right: pw - pw / 2,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let x = if self.pad_top + self.pad_bottom + self.pad_left + self.pad_right > 0 {
            pad(
                x,
                &[
                    (0, 0),
                    (self.pad_top, self.pad_bottom),
                    (self.pad_left, self.pad_right),
                    (0, 0),
                ][..],
                None,
                None,
            )?
        } else {
            x.clone()
        };
        conv2d(&x, &self.w, Some(&self.b), 1, 0)
    }
}

/// 2-D ResNet block (`ResnetBlock`): PixelNorm → SiLU → conv → PixelNorm → SiLU → conv, plus a
/// 1×1 `nin_shortcut` when `in != out`. `temb` is unused (the audio decoder has `temb_channels = 0`).
struct ResnetBlock {
    conv1: CausalConv2d,
    conv2: CausalConv2d,
    nin_shortcut: Option<CausalConv2d>,
}

impl ResnetBlock {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        let nin_shortcut = if w
            .get(&format!("{prefix}.nin_shortcut.conv.weight"))
            .is_some()
        {
            Some(CausalConv2d::load(
                w,
                &format!("{prefix}.nin_shortcut.conv"),
                true,
            )?)
        } else {
            None
        };
        Ok(Self {
            conv1: CausalConv2d::load(w, &format!("{prefix}.conv1.conv"), true)?,
            conv2: CausalConv2d::load(w, &format!("{prefix}.conv2.conv"), true)?,
            nin_shortcut,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.conv1.forward(&silu(&pixel_norm(x)?)?)?;
        let h = self.conv2.forward(&silu(&pixel_norm(&h)?)?)?;
        let shortcut = match &self.nin_shortcut {
            Some(c) => c.forward(x)?,
            None => x.clone(),
        };
        Ok(add(&shortcut, &h)?)
    }
}

/// VANILLA self-attention over the spatial grid (`AttnBlock`): PixelNorm → 1×1 q/k/v →
/// softmax(QKᵀ/√C) → 1×1 proj_out, residual. Config-gated (off for the shipped checkpoint).
struct AttnBlock {
    q: CausalConv2d,
    k: CausalConv2d,
    v: CausalConv2d,
    proj_out: CausalConv2d,
}

impl AttnBlock {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        // 1×1 convs → non-causal (no padding).
        Ok(Self {
            q: CausalConv2d::load(w, &format!("{prefix}.q"), false)?,
            k: CausalConv2d::load(w, &format!("{prefix}.k"), false)?,
            v: CausalConv2d::load(w, &format!("{prefix}.v"), false)?,
            proj_out: CausalConv2d::load(w, &format!("{prefix}.proj_out"), false)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = pixel_norm(x)?;
        let q = self.q.forward(&h)?;
        let k = self.k.forward(&h)?;
        let v = self.v.forward(&h)?;
        let sh = q.shape();
        let (b, hh, ww, c) = (sh[0], sh[1], sh[2], sh[3]);
        let n = hh * ww;
        let q = q.reshape(&[b, n, c])?;
        let k = k.reshape(&[b, n, c])?;
        let v = v.reshape(&[b, n, c])?;
        // w_ = softmax(q @ kᵀ · C^-0.5); h_ = w_ @ v. (Reference scales q@kᵀ, not via SDPA's mask path.)
        let scale = (c as f32).powf(-0.5);
        let scores = multiply(&matmul(&q, &k.transpose_axes(&[0, 2, 1])?)?, scalar(scale))?;
        let attn = softmax_axis(&scores, -1, None)?;
        let out = matmul(&attn, &v)?.reshape(&[b, hh, ww, c])?;
        Ok(add(x, &self.proj_out.forward(&out)?)?)
    }
}

/// A contiguous index range `lo..hi` as an `Array` for `take_axis`.
fn range_idx(lo: i32, hi: i32) -> Array {
    Array::from_slice(&(lo..hi).collect::<Vec<i32>>(), &[(hi - lo).max(0)])
}

/// Nearest-neighbour 2× upsample on the (height, width) axes of NHWC `x` (`[a,a,b,b,…]` per axis),
/// dtype-agnostic (broadcast + reshape, matching `mx.repeat(·, 2, axis)`).
fn nearest2x(x: &Array) -> Result<Array> {
    let sh = x.shape();
    let (b, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
    // `reshape(-1)` after each broadcast forces a contiguous copy (a strided broadcast view otherwise
    // reads scrambled on a host `as_slice`; see the video VAE `contiguous` note).
    let x = broadcast_to(&x.reshape(&[b, h, 1, w, c])?, &[b, h, 2, w, c])?
        .reshape(&[-1])?
        .reshape(&[b, 2 * h, w, c])?;
    let x = broadcast_to(&x.reshape(&[b, 2 * h, w, 1, c])?, &[b, 2 * h, w, 2, c])?
        .reshape(&[-1])?
        .reshape(&[b, 2 * h, 2 * w, c])?;
    Ok(x)
}

/// Upsample stage (`Upsample`): nearest-2× → causal conv → **drop the first time element** (undoes
/// the encoder's causal padding, keeping length `2n−1`).
struct Upsample {
    conv: CausalConv2d,
}

impl Upsample {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            conv: CausalConv2d::load(w, &format!("{prefix}.conv.conv"), true)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let x = self.conv.forward(&nearest2x(x)?)?;
        // Drop the first element along the causal (height/time) axis.
        let h = x.shape()[1];
        Ok(x.take_axis(range_idx(1, h), 1)?)
    }
}

/// One decoder up-level: `num_res_blocks + 1` ResnetBlocks (+ optional attn) + optional Upsample.
struct UpLevel {
    blocks: Vec<ResnetBlock>,
    attn: Vec<Option<AttnBlock>>,
    upsample: Option<Upsample>,
}

/// The LTX-2.3 audio VAE decoder. `conv_in → mid → up-levels → PixelNorm → SiLU → conv_out`.
pub struct AudioDecoder {
    conv_in: CausalConv2d,
    mid_block_1: ResnetBlock,
    mid_attn: Option<AttnBlock>,
    mid_block_2: ResnetBlock,
    up: Vec<UpLevel>, // index = level (0..num_resolutions), run high→low (reversed) at decode
    conv_out: CausalConv2d,
    mean_of_means: Array, // (ch=128,)
    std_of_means: Array,
    z_channels: i32,
    out_ch: i32,
    mel_bins: i32,
    downsample_factor: i32,
}

impl AudioDecoder {
    /// Build from `audio_vae.safetensors` + the [`AudioVaeConfig`].
    pub fn from_weights(w: &Weights, cfg: &AudioVaeConfig) -> Result<Self> {
        let num_res = cfg.num_resolutions();
        // Mid attention only if the config enables it AND the weights are present (shipped: neither).
        let mid_attn = if cfg.mid_block_add_attention && w.get("mid.attn_1.q.conv.weight").is_some()
        {
            Some(AttnBlock::load(w, "mid.attn_1")?)
        } else {
            None
        };
        // Up path: level `num_res-1 .. 0`. Each level has `num_res_blocks + 1` blocks; levels != 0
        // carry an upsample. `attn` rides on the weights (the reference's `attn_resolutions` never
        // matches the audio decoder's curr_res, so it's empty for the shipped model — gate on keys).
        let mut up = Vec::with_capacity(num_res);
        for level in 0..num_res {
            let mut blocks = Vec::with_capacity((cfg.num_res_blocks + 1) as usize);
            let mut attn = Vec::with_capacity((cfg.num_res_blocks + 1) as usize);
            for i_block in 0..(cfg.num_res_blocks + 1) {
                let bp = format!("up.{level}.block.{i_block}");
                blocks.push(ResnetBlock::load(w, &bp)?);
                let ap = format!("up.{level}.attn.{i_block}");
                attn.push(if w.get(&format!("{ap}.q.conv.weight")).is_some() {
                    Some(AttnBlock::load(w, &ap)?)
                } else {
                    None
                });
            }
            let upsample = if w
                .get(&format!("up.{level}.upsample.conv.conv.weight"))
                .is_some()
            {
                Some(Upsample::load(w, &format!("up.{level}.upsample"))?)
            } else {
                None
            };
            up.push(UpLevel {
                blocks,
                attn,
                upsample,
            });
        }

        Ok(Self {
            conv_in: CausalConv2d::load(w, "conv_in.conv", true)?,
            mid_block_1: ResnetBlock::load(w, "mid.block_1")?,
            mid_attn,
            mid_block_2: ResnetBlock::load(w, "mid.block_2")?,
            up,
            conv_out: CausalConv2d::load(w, "conv_out.conv", true)?,
            mean_of_means: f32(w, "per_channel_statistics._mean_of_means")?,
            std_of_means: f32(w, "per_channel_statistics._std_of_means")?,
            z_channels: cfg.z_channels,
            out_ch: cfg.out_ch,
            mel_bins: cfg.mel_bins,
            downsample_factor: crate::positions::AUDIO_LATENT_DOWNSAMPLE_FACTOR as i32,
        })
    }

    /// Denormalize the patchified latent: `(B,T,F,C)` → patchify `(B,T,C·F)` → `·std + mean` →
    /// unpatchify `(B,T,F,C)`. Matches `AudioPatchifier` + `PerChannelStatistics.un_normalize`.
    fn denormalize(&self, sample: &Array) -> Result<Array> {
        let sh = sample.shape(); // (B, T, F, C)
        let (b, t, fbins, c) = (sh[0], sh[1], sh[2], sh[3]);
        // patchify: (B,T,F,C) → (B,T,C,F) → (B,T,C·F).
        let patched = sample
            .transpose_axes(&[0, 1, 3, 2])?
            .reshape(&[b, t, c * fbins])?;
        let std = to_dtype(&self.std_of_means, patched.dtype())?;
        let mean = to_dtype(&self.mean_of_means, patched.dtype())?;
        let denorm = add(&multiply(&patched, &std)?, &mean)?;
        // unpatchify: (B,T,C·F) → (B,T,C,F) → (B,T,F,C).
        Ok(denorm
            .reshape(&[b, t, c, fbins])?
            .transpose_axes(&[0, 1, 3, 2])?)
    }

    /// Decode an audio latent `(B, z=8, T, 16)` (NCHW) → mel spectrogram `(B, out_ch=2, T', 64)`.
    pub fn decode(&self, latent: &Array) -> Result<Array> {
        let latent = to_dtype(latent, Dtype::Float32)?;
        // (B, C, H=T, W=mel) → (B, H, W, C) NHWC (only when channels-first, like the reference).
        let mut sample = if latent.ndim() == 4 && latent.shape()[1] == self.z_channels {
            latent.transpose_axes(&[0, 2, 3, 1])?
        } else {
            latent.clone()
        };
        let lsh = sample.shape();
        let (frames, latent_mel) = (lsh[1], lsh[2]);
        sample = self.denormalize(&sample)?;

        let mut h = self.conv_in.forward(&sample)?;
        h = self.mid_block_1.forward(&h)?;
        if let Some(attn) = &self.mid_attn {
            h = attn.forward(&h)?;
        }
        h = self.mid_block_2.forward(&h)?;

        // Up path, high → low level (reversed); upsample on levels != 0.
        for level in self.up.iter().enumerate().rev() {
            let (idx, stage) = level;
            for (b_idx, block) in stage.blocks.iter().enumerate() {
                h = block.forward(&h)?;
                if let Some(attn) = &stage.attn[b_idx] {
                    h = attn.forward(&h)?;
                }
            }
            if idx != 0 {
                if let Some(up) = &stage.upsample {
                    h = up.forward(&h)?;
                }
            }
        }

        h = self.conv_out.forward(&silu(&pixel_norm(&h)?)?)?;

        // Crop/pad to the causal target frame count, then NHWC → NCHW for the vocoder.
        let target_frames = {
            let f = frames * self.downsample_factor;
            (f - (self.downsample_factor - 1)).max(1)
        };
        let target_mel = if self.mel_bins > 0 {
            self.mel_bins
        } else {
            latent_mel
        };
        let h = self.adjust(&h, target_frames, target_mel)?;
        Ok(h.transpose_axes(&[0, 3, 1, 2])?)
    }

    /// Crop-then-pad NHWC `(B, time, freq, C)` to `(B, target_time, target_freq, out_ch)`.
    fn adjust(&self, x: &Array, target_time: i32, target_freq: i32) -> Result<Array> {
        let sh = x.shape();
        let (cur_t, cur_f) = (sh[1], sh[2]);
        let crop_t = cur_t.min(target_time);
        let crop_f = cur_f.min(target_freq);
        let mut x = x
            .take_axis(range_idx(0, crop_t), 1)?
            .take_axis(range_idx(0, crop_f), 2)?
            .take_axis(range_idx(0, self.out_ch), 3)?;
        let pad_t = (target_time - x.shape()[1]).max(0);
        let pad_f = (target_freq - x.shape()[2]).max(0);
        if pad_t > 0 || pad_f > 0 {
            x = pad(
                &x,
                &[(0, 0), (0, pad_t), (0, pad_f), (0, 0)][..],
                None,
                None,
            )?;
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_norm_unit_rms() {
        // (1,1,1,4): RMS over channels → unit RMS output.
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 1, 4]);
        let y = pixel_norm(&x).unwrap();
        let s = y.as_slice::<f32>();
        let ms = s.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((ms - 1.0).abs() < 1e-4, "rms² = {ms}");
    }

    #[test]
    fn nearest2x_repeats_each_element() {
        // (1,2,1,1) over H → [a,a,b,b].
        let x = Array::from_slice(&[5.0f32, 9.0], &[1, 2, 1, 1]);
        let y = nearest2x(&x).unwrap();
        assert_eq!(y.shape(), &[1, 4, 2, 1]);
        // H repeated then W repeated: row0=5,5 row1... actually [a,a,b,b] over H, each ×2 over W.
        assert_eq!(
            y.as_slice::<f32>(),
            &[5.0, 5.0, 5.0, 5.0, 9.0, 9.0, 9.0, 9.0]
        );
    }
}
