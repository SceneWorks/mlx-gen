//! `UNetBlock2D` — the down/up macro-block: a run of `ResnetBlock2D`s, optional cross-attention
//! `Transformer2D`s interleaved, and an optional downsample (stride-2 conv) or upsample
//! (nearest-2× + conv). Port of the vendored `unet.UNetBlock2D`. On the up path each resnet is fed
//! `concat(x, residual.pop())` (the U-Net skip connections); the resnets' loaded conv weights
//! already carry the post-concat channel counts, so no channel math is needed here.

use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::nn::{conv2d, upsample_nearest};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::nchw_to_nhwc;
use super::resnet::ResnetBlock2D;
use super::transformer::Transformer2D;

pub struct UNetBlock2D {
    resnets: Vec<ResnetBlock2D>,
    attentions: Option<Vec<Transformer2D>>,
    /// Downsample conv (stride 2, pad 1) on NHWC.
    downsample: Option<(Array, Array)>,
    /// Upsample conv (stride 1, pad 1) applied after a nearest-2× resize.
    upsample: Option<(Array, Array)>,
}

/// Per-block construction parameters resolved from [`crate::config::UNetConfig`].
pub struct BlockSpec<'a> {
    /// Checkpoint module prefix, e.g. `down_blocks.0` or `up_blocks.2`.
    pub prefix: &'a str,
    pub num_resnets: i32,
    pub out_channels: i32,
    pub num_heads: i32,
    pub transformer_layers: i32,
    pub add_cross_attention: bool,
    pub add_downsample: bool,
    pub add_upsample: bool,
}

impl UNetBlock2D {
    pub fn from_weights(w: &Weights, spec: &BlockSpec) -> Result<Self> {
        let resnets = (0..spec.num_resnets)
            .map(|j| ResnetBlock2D::from_weights(w, &format!("{}.resnets.{j}", spec.prefix)))
            .collect::<Result<Vec<_>>>()?;
        let attentions = if spec.add_cross_attention {
            Some(
                (0..spec.num_resnets)
                    .map(|j| {
                        Transformer2D::from_weights(
                            w,
                            &format!("{}.attentions.{j}", spec.prefix),
                            spec.out_channels,
                            spec.num_heads,
                            spec.transformer_layers,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?,
            )
        } else {
            None
        };
        let downsample = if spec.add_downsample {
            Some((
                nchw_to_nhwc(w.require(&format!("{}.downsamplers.0.conv.weight", spec.prefix))?)?,
                w.require(&format!("{}.downsamplers.0.conv.bias", spec.prefix))?
                    .clone(),
            ))
        } else {
            None
        };
        let upsample = if spec.add_upsample {
            Some((
                nchw_to_nhwc(w.require(&format!("{}.upsamplers.0.conv.weight", spec.prefix))?)?,
                w.require(&format!("{}.upsamplers.0.conv.bias", spec.prefix))?
                    .clone(),
            ))
        } else {
            None
        };
        Ok(Self {
            resnets,
            attentions,
            downsample,
            upsample,
        })
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for r in &mut self.resnets {
            r.quantize(bits)?;
        }
        if let Some(attns) = &mut self.attentions {
            for a in attns {
                a.quantize(bits)?;
            }
        }
        Ok(())
    }

    /// Run the block. `residuals` is `Some` on the up path (each resnet pops one skip tensor and
    /// concatenates it onto `x` first). Returns the block output plus the per-step output states
    /// (the down path pushes these onto the residual stack).
    pub fn forward(
        &self,
        x: &Array,
        encoder_x: &Array,
        temb: &Array,
        mut residuals: Option<&mut Vec<Array>>,
    ) -> Result<(Array, Vec<Array>)> {
        let mut x = x.clone();
        let mut output_states = Vec::with_capacity(self.resnets.len() + 1);
        for i in 0..self.resnets.len() {
            if let Some(res) = residuals.as_deref_mut() {
                let skip = res.pop().ok_or_else(|| {
                    mlx_gen::Error::Msg("sdxl unet: residual stack underflow".into())
                })?;
                x = concatenate_axis(&[&x, &skip], -1)?;
            }
            x = self.resnets[i].forward(&x, Some(temb))?;
            if let Some(attns) = &self.attentions {
                x = attns[i].forward(&x, encoder_x)?;
            }
            output_states.push(x.clone());
        }
        if let Some((w, b)) = &self.downsample {
            x = conv2d(&x, w, Some(b), 2, 1)?;
            output_states.push(x.clone());
        }
        if let Some((w, b)) = &self.upsample {
            x = conv2d(&upsample_nearest(&x, 2)?, w, Some(b), 1, 1)?;
            output_states.push(x.clone());
        }
        Ok((x, output_states))
    }

    /// Emit the diffusers paths of this block's LoRA-targetable Linears (resnet `time_emb_proj`s +
    /// each `attentions.{j}` transformer's projections), prefixed by `prefix` (e.g. `down_blocks.1`).
    pub fn lora_target_paths(&self, prefix: &str, out: &mut Vec<String>) {
        for (j, r) in self.resnets.iter().enumerate() {
            r.lora_target_paths(&format!("{prefix}.resnets.{j}"), out);
        }
        if let Some(attns) = &self.attentions {
            for (j, a) in attns.iter().enumerate() {
                a.lora_target_paths(&format!("{prefix}.attentions.{j}"), out);
            }
        }
    }
}

impl AdaptableHost for UNetBlock2D {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["resnets", j, rest @ ..] => self
                .resnets
                .get_mut(j.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            ["attentions", j, rest @ ..] => self
                .attentions
                .as_mut()?
                .get_mut(j.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            _ => None,
        }
    }
}
