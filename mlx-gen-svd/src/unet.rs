//! SVD `UNetSpatioTemporalConditionModel` (sc-3374) — the spatiotemporal denoising UNet. Port of
//! diffusers `unet_spatio_temporal_condition.py` + the `*SpatioTemporal` blocks in `unet_3d_blocks.py`.
//!
//! Structure: a conv stem; sinusoidal timestep + `added_time_ids` micro-conditioning → a 1280-wide
//! `emb`; a down (3× `CrossAttnDownBlockSpatioTemporal` + 1× `DownBlockSpatioTemporal`) / mid
//! (`UNetMidBlockSpatioTemporal`) / up (`UpBlockSpatioTemporal` + 3× `CrossAttnUpBlockSpatioTemporal`)
//! stack of [`SpatioTemporalResBlock`]s and [`TransformerSpatioTemporal`]s; a conv head. Predicts the
//! per-frame `v` for one denoise step. Runs NHWC for the spatial parts (`[B·F, H, W, C]`) and NDHWC
//! (`[B, F, H, W, C]`) for the temporal resnet path.
//!
//! Faithfulness: the per-block GroupNorm epsilon matches diffusers exactly — **1e-6** for the
//! `CrossAttnDownBlockSpatioTemporal` resnets, **1e-5** for the plain down / mid / all up blocks and
//! `conv_norm_out` (the `resnet_eps=1e-5` the UNet passes to `get_up_block`). The
//! `SpatioTemporalResBlock` here is temb-aware (unlike the VAE's) and blends `σ(mix)·spatial +
//! (1−σ)·temporal` (`merge_strategy="learned_with_images"`, no switch).

use mlx_rs::ops::{add, broadcast_to, concatenate_axis, multiply, sigmoid, subtract};
use mlx_rs::Array;

use mlx_gen::array::scalar;
use mlx_gen::nn::{conv2d, conv3d, group_norm, linear, silu, upsample_nearest};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::UnetConfig;
use crate::embeddings::{sinusoidal_timestep, TimestepEmbedding};
use crate::transformer::TransformerSpatioTemporal;
use crate::vae::{load_conv2d, load_conv3d};

const GN_GROUPS: i32 = 32;
/// `CrossAttnDownBlockSpatioTemporal` resnet epsilon (diffusers hardcodes `eps=1e-6` there).
const EPS_CROSS_DOWN: f32 = 1e-6;
/// Plain down / mid / up resnet + `conv_norm_out` epsilon (the `resnet_eps=1e-5` the UNet passes).
const EPS_OTHER: f32 = 1e-5;

/// `[B, D] → [B·F, D]` (diffusers `repeat_interleave(F, dim=0)`).
fn repeat_interleave_2d(x: &Array, f: i32) -> Result<Array> {
    let s = x.shape();
    let (b, d) = (s[0], s[1]);
    Ok(broadcast_to(&x.reshape(&[b, 1, d])?, &[b, f, d])?.reshape(&[b * f, d])?)
}

/// `[B, S, D] → [B·F, S, D]` (diffusers `repeat_interleave(F, dim=0)`).
fn repeat_interleave_3d(x: &Array, f: i32) -> Result<Array> {
    let s = x.shape();
    let (b, sl, d) = (s[0], s[1], s[2]);
    Ok(broadcast_to(&x.reshape(&[b, 1, sl, d])?, &[b, f, sl, d])?.reshape(&[b * f, sl, d])?)
}

/// Temb-aware spatial `ResnetBlock2D`: GroupNorm→SiLU→Conv3×3, + projected temb, GroupNorm→SiLU→
/// Conv3×3, + (1×1-conv) residual. NHWC `[B·F, H, W, C]`, `temb` `[B·F, 1280]`.
struct SpatialResnet {
    norm1_w: Array,
    norm1_b: Array,
    conv1: (Array, Array),
    temb_proj: (Array, Array),
    norm2_w: Array,
    norm2_b: Array,
    conv2: (Array, Array),
    shortcut: Option<(Array, Array)>,
    eps: f32,
}

impl SpatialResnet {
    fn from_weights(w: &Weights, prefix: &str, eps: f32) -> Result<Self> {
        let g = |n: &str| w.require(&format!("{prefix}.{n}")).cloned();
        let shortcut = match w.get(&format!("{prefix}.conv_shortcut.weight")) {
            Some(_) => Some(load_conv2d(w, &format!("{prefix}.conv_shortcut"))?),
            None => None,
        };
        Ok(Self {
            norm1_w: g("norm1.weight")?,
            norm1_b: g("norm1.bias")?,
            conv1: load_conv2d(w, &format!("{prefix}.conv1"))?,
            temb_proj: (g("time_emb_proj.weight")?, g("time_emb_proj.bias")?),
            norm2_w: g("norm2.weight")?,
            norm2_b: g("norm2.bias")?,
            conv2: load_conv2d(w, &format!("{prefix}.conv2"))?,
            shortcut,
            eps,
        })
    }

    fn forward(&self, x: &Array, temb: &Array) -> Result<Array> {
        let y = group_norm(x, &self.norm1_w, &self.norm1_b, GN_GROUPS, self.eps)?;
        let y = conv2d(&silu(&y)?, &self.conv1.0, Some(&self.conv1.1), 1, 1)?;
        let tp = linear(&silu(temb)?, &self.temb_proj.0, &self.temb_proj.1)?; // [B·F, out]
        let ts = tp.shape();
        let y = add(&y, &tp.reshape(&[ts[0], 1, 1, ts[1]])?)?;
        let y = group_norm(&y, &self.norm2_w, &self.norm2_b, GN_GROUPS, self.eps)?;
        let y = conv2d(&silu(&y)?, &self.conv2.0, Some(&self.conv2.1), 1, 1)?;
        let residual = match &self.shortcut {
            Some((cw, cb)) => conv2d(x, cw, Some(cb), 1, 0)?,
            None => x.clone(),
        };
        Ok(add(&residual, &y)?)
    }
}

/// Temb-aware temporal `TemporalResnetBlock`: Conv3d`(3,1,1)` over the frame axis, + projected temb.
/// NDHWC `[B, F, H, W, C]`, `temb` `[B, F, 1280]`.
struct TemporalResnet {
    norm1_w: Array,
    norm1_b: Array,
    conv1: (Array, Array),
    temb_proj: (Array, Array),
    norm2_w: Array,
    norm2_b: Array,
    conv2: (Array, Array),
    eps: f32,
}

impl TemporalResnet {
    fn from_weights(w: &Weights, prefix: &str, eps: f32) -> Result<Self> {
        let g = |n: &str| w.require(&format!("{prefix}.{n}")).cloned();
        Ok(Self {
            norm1_w: g("norm1.weight")?,
            norm1_b: g("norm1.bias")?,
            conv1: load_conv3d(w, &format!("{prefix}.conv1"))?,
            temb_proj: (g("time_emb_proj.weight")?, g("time_emb_proj.bias")?),
            norm2_w: g("norm2.weight")?,
            norm2_b: g("norm2.bias")?,
            conv2: load_conv3d(w, &format!("{prefix}.conv2"))?,
            eps,
        })
    }

    fn forward(&self, x: &Array, temb: &Array) -> Result<Array> {
        let y = group_norm(x, &self.norm1_w, &self.norm1_b, GN_GROUPS, self.eps)?;
        let y = conv3d(
            &silu(&y)?,
            &self.conv1.0,
            Some(&self.conv1.1),
            (1, 1, 1),
            (1, 0, 0),
        )?;
        let tp = linear(&silu(temb)?, &self.temb_proj.0, &self.temb_proj.1)?; // [B, F, out]
        let ts = tp.shape();
        let y = add(&y, &tp.reshape(&[ts[0], ts[1], 1, 1, ts[2]])?)?;
        let y = group_norm(&y, &self.norm2_w, &self.norm2_b, GN_GROUPS, self.eps)?;
        let y = conv3d(
            &silu(&y)?,
            &self.conv2.0,
            Some(&self.conv2.1),
            (1, 1, 1),
            (1, 0, 0),
        )?;
        Ok(add(x, &y)?)
    }
}

/// `SpatioTemporalResBlock` (UNet flavor): spatial pass then temporal pass, blended
/// `σ(mix)·spatial + (1−σ)·temporal`.
struct SpatioTemporalResBlock {
    spatial: SpatialResnet,
    temporal: TemporalResnet,
    mix_factor: Array,
}

impl SpatioTemporalResBlock {
    fn from_weights(w: &Weights, prefix: &str, eps: f32) -> Result<Self> {
        Ok(Self {
            spatial: SpatialResnet::from_weights(w, &format!("{prefix}.spatial_res_block"), eps)?,
            temporal: TemporalResnet::from_weights(
                w,
                &format!("{prefix}.temporal_res_block"),
                eps,
            )?,
            mix_factor: w
                .require(&format!("{prefix}.time_mixer.mix_factor"))?
                .clone(),
        })
    }

    fn forward(&self, x: &Array, temb: &Array, num_frames: i32) -> Result<Array> {
        let spatial = self.spatial.forward(x, temb)?; // [B·F, H, W, C_out]
        let sh = spatial.shape();
        let (bf, h, w_, c) = (sh[0], sh[1], sh[2], sh[3]);
        let b = bf / num_frames;
        let spatial5 = spatial.reshape(&[b, num_frames, h, w_, c])?;
        let temb5 = temb.reshape(&[b, num_frames, temb.shape()[1]])?;
        let temporal = self.temporal.forward(&spatial5, &temb5)?;

        let alpha = sigmoid(&self.mix_factor)?;
        let one_minus = subtract(scalar(1.0), &alpha)?;
        let blended = add(
            &multiply(&spatial5, &alpha)?,
            &multiply(&temporal, &one_minus)?,
        )?;
        Ok(blended.reshape(&[bf, h, w_, c])?)
    }
}

/// Test-only: load + run a single `SpatioTemporalResBlock` (for isolated parity bisection, sc-3374).
#[doc(hidden)]
pub fn debug_st_resblock(
    w: &Weights,
    prefix: &str,
    eps: f32,
    x: &Array,
    temb: &Array,
    num_frames: i32,
) -> Result<Array> {
    SpatioTemporalResBlock::from_weights(w, prefix, eps)?.forward(x, temb, num_frames)
}

/// Stride-2 downsample (`Downsample2D(use_conv=True, padding=1)`): conv 3×3, stride 2, pad 1.
fn downsample(x: &Array, conv: &(Array, Array)) -> Result<Array> {
    conv2d(x, &conv.0, Some(&conv.1), 2, 1)
}

/// Nearest-2× + conv 3×3 upsample (`Upsample2D(use_conv=True)`).
fn upsample(x: &Array, conv: &(Array, Array)) -> Result<Array> {
    conv2d(&upsample_nearest(x, 2)?, &conv.0, Some(&conv.1), 1, 1)
}

/// One down block: resnets (optionally each followed by a transformer) + an optional downsample.
struct DownBlock {
    resnets: Vec<SpatioTemporalResBlock>,
    attentions: Option<Vec<TransformerSpatioTemporal>>,
    downsampler: Option<(Array, Array)>,
}

impl DownBlock {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        num_resnets: usize,
        eps: f32,
        cross_attn: Option<i32>, // Some(heads) if a CrossAttn block
        add_down: bool,
    ) -> Result<Self> {
        let resnets = (0..num_resnets)
            .map(|j| SpatioTemporalResBlock::from_weights(w, &format!("{prefix}.resnets.{j}"), eps))
            .collect::<Result<Vec<_>>>()?;
        let attentions = match cross_attn {
            Some(heads) => Some(
                (0..num_resnets)
                    .map(|j| {
                        TransformerSpatioTemporal::from_weights(
                            w,
                            &format!("{prefix}.attentions.{j}"),
                            heads,
                            1,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?,
            ),
            None => None,
        };
        Ok(Self {
            resnets,
            attentions,
            downsampler: if add_down {
                Some(load_conv2d(w, &format!("{prefix}.downsamplers.0.conv"))?)
            } else {
                None
            },
        })
    }

    /// Returns the block output and its per-resnet (+ downsample) skip residuals.
    fn forward(
        &self,
        x: &Array,
        temb: &Array,
        context: &Array,
        num_frames: i32,
    ) -> Result<(Array, Vec<Array>)> {
        let mut x = x.clone();
        let mut res = Vec::new();
        for (i, r) in self.resnets.iter().enumerate() {
            x = r.forward(&x, temb, num_frames)?;
            if let Some(attns) = &self.attentions {
                x = attns[i].forward(&x, context, num_frames)?;
            }
            res.push(x.clone());
        }
        if let Some(conv) = &self.downsampler {
            x = downsample(&x, conv)?;
            res.push(x.clone());
        }
        Ok((x, res))
    }
}

/// The mid block: resnet → (transformer → resnet)×num_layers.
struct MidBlock {
    res0: SpatioTemporalResBlock,
    pairs: Vec<(TransformerSpatioTemporal, SpatioTemporalResBlock)>,
}

impl MidBlock {
    fn from_weights(w: &Weights, prefix: &str, heads: i32, num_layers: usize) -> Result<Self> {
        let res0 =
            SpatioTemporalResBlock::from_weights(w, &format!("{prefix}.resnets.0"), EPS_OTHER)?;
        let pairs = (0..num_layers)
            .map(|i| -> Result<_> {
                let attn = TransformerSpatioTemporal::from_weights(
                    w,
                    &format!("{prefix}.attentions.{i}"),
                    heads,
                    1,
                )?;
                let resnet = SpatioTemporalResBlock::from_weights(
                    w,
                    &format!("{prefix}.resnets.{}", i + 1),
                    EPS_OTHER,
                )?;
                Ok((attn, resnet))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { res0, pairs })
    }

    fn forward(&self, x: &Array, temb: &Array, context: &Array, num_frames: i32) -> Result<Array> {
        let mut x = self.res0.forward(x, temb, num_frames)?;
        for (attn, resnet) in &self.pairs {
            x = attn.forward(&x, context, num_frames)?;
            x = resnet.forward(&x, temb, num_frames)?;
        }
        Ok(x)
    }
}

/// One up block: per resnet, concat the popped skip then resnet (optionally + transformer); then an
/// optional upsample.
struct UpBlock {
    resnets: Vec<SpatioTemporalResBlock>,
    attentions: Option<Vec<TransformerSpatioTemporal>>,
    upsampler: Option<(Array, Array)>,
}

impl UpBlock {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        num_resnets: usize,
        eps: f32,
        cross_attn: Option<i32>,
        add_up: bool,
    ) -> Result<Self> {
        let resnets = (0..num_resnets)
            .map(|j| SpatioTemporalResBlock::from_weights(w, &format!("{prefix}.resnets.{j}"), eps))
            .collect::<Result<Vec<_>>>()?;
        let attentions = match cross_attn {
            Some(heads) => Some(
                (0..num_resnets)
                    .map(|j| {
                        TransformerSpatioTemporal::from_weights(
                            w,
                            &format!("{prefix}.attentions.{j}"),
                            heads,
                            1,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?,
            ),
            None => None,
        };
        Ok(Self {
            resnets,
            attentions,
            upsampler: if add_up {
                Some(load_conv2d(w, &format!("{prefix}.upsamplers.0.conv"))?)
            } else {
                None
            },
        })
    }

    fn forward(
        &self,
        x: &Array,
        temb: &Array,
        context: &Array,
        skips: &mut Vec<Array>,
        num_frames: i32,
    ) -> Result<Array> {
        let mut x = x.clone();
        for (i, r) in self.resnets.iter().enumerate() {
            let skip = skips.pop().expect("up block: skip residual underflow");
            x = concatenate_axis(&[&x, &skip], -1)?;
            x = r.forward(&x, temb, num_frames)?;
            if let Some(attns) = &self.attentions {
                x = attns[i].forward(&x, context, num_frames)?;
            }
        }
        if let Some(conv) = &self.upsampler {
            x = upsample(&x, conv)?;
        }
        Ok(x)
    }
}

/// The SVD spatiotemporal conditional UNet.
pub struct SvdUnet {
    conv_in: (Array, Array),
    time_embedding: TimestepEmbedding,
    add_embedding: TimestepEmbedding,
    down_blocks: Vec<DownBlock>,
    mid_block: MidBlock,
    up_blocks: Vec<UpBlock>,
    conv_norm_out_w: Array,
    conv_norm_out_b: Array,
    conv_out: (Array, Array),
    time_proj_dim: i32,
    add_time_proj_dim: i32,
}

impl SvdUnet {
    pub fn from_weights(w: &Weights, cfg: &UnetConfig) -> Result<Self> {
        let boc = &cfg.block_out_channels;
        let heads = &cfg.num_attention_heads;
        let n = boc.len();

        // Down: CrossAttn for every block but the last (a plain DownBlock).
        let mut down_blocks = Vec::with_capacity(n);
        for (i, &head) in heads.iter().enumerate() {
            let is_last = i == n - 1;
            down_blocks.push(DownBlock::from_weights(
                w,
                &format!("down_blocks.{i}"),
                cfg.layers_per_block,
                if is_last { EPS_OTHER } else { EPS_CROSS_DOWN },
                if is_last { None } else { Some(head as i32) },
                !is_last,
            )?);
        }

        let mid_block = MidBlock::from_weights(
            w,
            "mid_block",
            *heads.last().unwrap() as i32,
            cfg.transformer_layers_per_block,
        )?;

        // Up: reversed; the first is a plain UpBlock, the rest CrossAttn. All use eps 1e-5.
        let rev_heads: Vec<usize> = heads.iter().rev().copied().collect();
        let mut up_blocks = Vec::with_capacity(n);
        for (i, &head) in rev_heads.iter().enumerate() {
            let is_first = i == 0;
            let is_last = i == n - 1;
            up_blocks.push(UpBlock::from_weights(
                w,
                &format!("up_blocks.{i}"),
                cfg.layers_per_block + 1,
                EPS_OTHER,
                if is_first { None } else { Some(head as i32) },
                !is_last,
            )?);
        }

        Ok(Self {
            conv_in: load_conv2d(w, "conv_in")?,
            time_embedding: TimestepEmbedding::from_weights(w, "time_embedding")?,
            add_embedding: TimestepEmbedding::from_weights(w, "add_embedding")?,
            down_blocks,
            mid_block,
            up_blocks,
            conv_norm_out_w: w.require("conv_norm_out.weight")?.clone(),
            conv_norm_out_b: w.require("conv_norm_out.bias")?.clone(),
            conv_out: load_conv2d(w, "conv_out")?,
            time_proj_dim: boc[0] as i32,
            add_time_proj_dim: cfg.addition_time_embed_dim as i32,
        })
    }

    /// Predict per-frame `v` for one denoise step.
    /// - `sample`: NHWC-with-frames `[B, F, H, W, 8]` (4 noise latent + 4 image-latent concat).
    /// - `timestep`: the scheduler model-timestep (`0.25·ln σ`), broadcast to the batch.
    /// - `image_embeds`: CLIP image conditioning `[B, ctx, 1024]` (repeated over frames internally).
    /// - `added_time_ids`: `[B, 3]` (`[fps−1, motion_bucket_id, noise_aug_strength]`).
    ///
    /// Returns `[B, F, H, W, 4]`.
    pub fn forward(
        &self,
        sample: &Array,
        timestep: f32,
        image_embeds: &Array,
        added_time_ids: &Array,
        num_frames: i32,
    ) -> Result<Array> {
        let sh = sample.shape();
        let (b, f, h, w_) = (sh[0], sh[1], sh[2], sh[3]);
        let in_ch = sh[4];

        // Timestep embedding.
        let t = Array::from_slice(&vec![timestep; b as usize], &[b]);
        let temb = sinusoidal_timestep(&t, self.time_proj_dim)?; // [B, 320]
        let mut emb = self.time_embedding.forward(&temb)?; // [B, 1280]

        // `added_time_ids` micro-conditioning.
        let flat = added_time_ids.reshape(&[b * added_time_ids.shape()[1]])?;
        let time_embeds = sinusoidal_timestep(&flat, self.add_time_proj_dim)?; // [B·3, 256]
        let time_embeds =
            time_embeds.reshape(&[b, time_embeds.shape()[1] * added_time_ids.shape()[1]])?;
        let aug = self.add_embedding.forward(&time_embeds)?; // [B, 1280]
        emb = add(&emb, &aug)?;

        // Flatten frames; repeat conditioning over frames.
        let sample = sample.reshape(&[b * f, h, w_, in_ch])?;
        let emb = repeat_interleave_2d(&emb, f)?; // [B·F, 1280]
        let context = repeat_interleave_3d(image_embeds, f)?; // [B·F, ctx, 1024]

        // Conv stem; collect skip residuals (starting with the stem output).
        let mut x = conv2d(&sample, &self.conv_in.0, Some(&self.conv_in.1), 1, 1)?;
        let mut skips: Vec<Array> = vec![x.clone()];
        for block in &self.down_blocks {
            let (out, res) = block.forward(&x, &emb, &context, num_frames)?;
            x = out;
            skips.extend(res);
        }

        x = self.mid_block.forward(&x, &emb, &context, num_frames)?;

        for block in &self.up_blocks {
            x = block.forward(&x, &emb, &context, &mut skips, num_frames)?;
        }

        let x = group_norm(
            &x,
            &self.conv_norm_out_w,
            &self.conv_norm_out_b,
            GN_GROUPS,
            EPS_OTHER,
        )?;
        let x = conv2d(&silu(&x)?, &self.conv_out.0, Some(&self.conv_out.1), 1, 1)?;
        let os = x.shape();
        Ok(x.reshape(&[b, f, os[1], os[2], os[3]])?)
    }
}
