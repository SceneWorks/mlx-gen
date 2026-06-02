//! `UNet2DConditionModel` — the SDXL denoising U-Net. Port of the vendored `unet.UNetModel`: a
//! conv stem, sinusoidal timestep + SDXL `text_time` micro-conditioning embeddings, a down /
//! mid / up stack of [`UNetBlock2D`]s with cross-attention to the dual-CLIP text conditioning, and
//! a conv head. Runs entirely in NHWC. Predicts the noise (`eps`) for one denoise step.

mod block;
mod embeddings;
mod resnet;
mod transformer;

use mlx_rs::ops::{add, concatenate_axis};
use mlx_rs::Array;

use mlx_gen::nn::{conv2d, group_norm, silu};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::UNetConfig;
use block::{BlockSpec, UNetBlock2D};
use embeddings::{SinusoidalPositionalEncoding, TimestepEmbedding};
use transformer::Transformer2D;

// Shared with the VAE (the vendored VAE reuses the UNet `ResnetBlock2D` without a time embedding).
pub use resnet::ResnetBlock2D;

const GN_GROUPS: i32 = 32;
const GN_EPS: f32 = 1e-5;

/// Transpose a stored NCHW conv weight `[out, in, kH, kW]` to mlx's NHWC `[out, kH, kW, in]`.
pub(crate) fn nchw_to_nhwc(w: &Array) -> Result<Array> {
    Ok(w.transpose_axes(&[0, 2, 3, 1])?)
}

/// The SDXL conditional U-Net.
pub struct UNet2DConditionModel {
    conv_in_w: Array,
    conv_in_b: Array,
    timesteps: SinusoidalPositionalEncoding,
    time_embedding: TimestepEmbedding,
    add_time_proj: SinusoidalPositionalEncoding,
    add_embedding: TimestepEmbedding,
    down_blocks: Vec<UNetBlock2D>,
    mid_resnet0: ResnetBlock2D,
    mid_transformer: Transformer2D,
    mid_resnet1: ResnetBlock2D,
    up_blocks: Vec<UNetBlock2D>,
    conv_norm_out_w: Array,
    conv_norm_out_b: Array,
    conv_out_w: Array,
    conv_out_b: Array,
}

impl UNet2DConditionModel {
    /// Assemble the U-Net from a diffusers SDXL `unet/` checkpoint (keys read directly; conv weights
    /// transposed to NHWC on load). `cfg` is [`UNetConfig::sdxl_base`].
    pub fn from_weights(w: &Weights, cfg: &UNetConfig) -> Result<Self> {
        let n = cfg.num_blocks();
        let boc = &cfg.block_out_channels;
        let temb_dim_src = boc[0]; // sinusoidal timestep width

        // Down blocks: block i goes block_channels[i] -> block_channels[i+1].
        let mut down_blocks = Vec::with_capacity(n);
        for i in 0..n {
            down_blocks.push(UNetBlock2D::from_weights(
                w,
                &BlockSpec {
                    prefix: &format!("down_blocks.{i}"),
                    num_resnets: cfg.layers_per_block[i],
                    out_channels: boc[i],
                    num_heads: cfg.num_attention_heads[i],
                    transformer_layers: cfg.transformer_layers_per_block[i],
                    add_cross_attention: cfg.down_block_types[i].contains("CrossAttn"),
                    add_downsample: i < n - 1,
                    add_upsample: false,
                },
            )?);
        }

        // Mid: resnet, transformer, resnet (the vendored mid_blocks.0/1/2).
        let mid_resnet0 = ResnetBlock2D::from_weights(w, "mid_block.resnets.0")?;
        let mid_transformer = Transformer2D::from_weights(
            w,
            "mid_block.attentions.0",
            *boc.last().unwrap(),
            *cfg.num_attention_heads.last().unwrap(),
            *cfg.transformer_layers_per_block.last().unwrap(),
        )?;
        let mid_resnet1 = ResnetBlock2D::from_weights(w, "mid_block.resnets.1")?;

        // Up blocks: checkpoint up_blocks.{k} corresponds to config index `n-1-k` (the vendored
        // builds them in reversed order). add_upsample on all but the last config index (0).
        let mut up_blocks = Vec::with_capacity(n);
        for k in 0..n {
            let ci = n - 1 - k;
            up_blocks.push(UNetBlock2D::from_weights(
                w,
                &BlockSpec {
                    prefix: &format!("up_blocks.{k}"),
                    num_resnets: cfg.layers_per_block[ci] + 1,
                    out_channels: boc[ci],
                    num_heads: cfg.num_attention_heads[ci],
                    transformer_layers: cfg.transformer_layers_per_block[ci],
                    add_cross_attention: cfg.up_block_types[ci].contains("CrossAttn"),
                    add_downsample: false,
                    add_upsample: ci > 0,
                },
            )?);
        }

        Ok(Self {
            conv_in_w: nchw_to_nhwc(w.require("conv_in.weight")?)?,
            conv_in_b: w.require("conv_in.bias")?.clone(),
            timesteps: SinusoidalPositionalEncoding::timestep(temb_dim_src)?,
            time_embedding: TimestepEmbedding::from_weights(w, "time_embedding")?,
            add_time_proj: SinusoidalPositionalEncoding::timestep(
                cfg.addition_time_embed_dim.unwrap_or(256),
            )?,
            add_embedding: TimestepEmbedding::from_weights(w, "add_embedding")?,
            down_blocks,
            mid_resnet0,
            mid_transformer,
            mid_resnet1,
            up_blocks,
            conv_norm_out_w: w.require("conv_norm_out.weight")?.clone(),
            conv_norm_out_b: w.require("conv_norm_out.bias")?.clone(),
            conv_out_w: nchw_to_nhwc(w.require("conv_out.weight")?)?,
            conv_out_b: w.require("conv_out.bias")?.clone(),
        })
    }

    /// Quantize every Linear (resnets' time/shortcut projections, attention, FFN, embeddings) to
    /// Q4/Q8. Convs (`conv_in`/`conv_out`/resnet convs/up-down samplers) stay dense.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.time_embedding.quantize(bits)?;
        self.add_embedding.quantize(bits)?;
        for b in &mut self.down_blocks {
            b.quantize(bits)?;
        }
        self.mid_resnet0.quantize(bits)?;
        self.mid_transformer.quantize(bits)?;
        self.mid_resnet1.quantize(bits)?;
        for b in &mut self.up_blocks {
            b.quantize(bits)?;
        }
        Ok(())
    }

    /// Predict `eps` for one denoise step.
    /// - `x`: NHWC latents `[B, H, W, 4]`.
    /// - `timestep`: the (sigma-space) time, broadcast to the batch.
    /// - `encoder_x`: dual-CLIP text conditioning `[B, S, 2048]`.
    /// - `text_emb`: pooled conditioning `[B, 1280]`; `time_ids`: micro-conditioning `[B, 6]`.
    pub fn forward(
        &self,
        x: &Array,
        timestep: f32,
        encoder_x: &Array,
        text_emb: &Array,
        time_ids: &Array,
    ) -> Result<Array> {
        let batch = x.shape()[0];

        // Timestep embedding (broadcast the scalar time to the batch).
        let t = Array::from_slice(&vec![timestep; batch as usize], &[batch]);
        let temb = self.timesteps.forward(&t)?;
        let mut temb = self.time_embedding.forward(&temb)?;

        // SDXL `text_time` added conditioning: concat(pooled_text, flattened sinusoidal time_ids).
        let emb = self.add_time_proj.forward(time_ids)?; // [B, 6, 256]
        let es = emb.shape();
        let emb = emb.reshape(&[es[0], es[1] * es[2]])?; // flatten(1) → [B, 1536]
        let emb = concatenate_axis(&[text_emb, &emb], -1)?; // [B, 2816]
        let emb = self.add_embedding.forward(&emb)?;
        temb = add(&temb, &emb)?;

        // Conv stem.
        let mut x = conv2d(x, &self.conv_in_w, Some(&self.conv_in_b), 1, 1)?;

        // Down path — collect skip residuals (starting with the stem output).
        let mut residuals: Vec<Array> = vec![x.clone()];
        for block in &self.down_blocks {
            let (out, res) = block.forward(&x, encoder_x, &temb, None)?;
            x = out;
            residuals.extend(res);
        }

        // Mid.
        x = self.mid_resnet0.forward(&x, Some(&temb))?;
        x = self.mid_transformer.forward(&x, encoder_x)?;
        x = self.mid_resnet1.forward(&x, Some(&temb))?;

        // Up path — each block pops its skip residuals.
        for block in &self.up_blocks {
            let (out, _) = block.forward(&x, encoder_x, &temb, Some(&mut residuals))?;
            x = out;
        }

        // Conv head.
        let x = group_norm(
            &x,
            &self.conv_norm_out_w,
            &self.conv_norm_out_b,
            GN_GROUPS,
            GN_EPS,
        )?;
        let x = silu(&x)?;
        conv2d(&x, &self.conv_out_w, Some(&self.conv_out_b), 1, 1)
    }
}
