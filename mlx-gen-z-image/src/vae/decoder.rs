//! VAE decoder assembly + `Vae::decode`. Port of `Decoder.__call__` / `VAE.decode`:
//! conv_in → mid-block → up-blocks → GroupNorm-out → SiLU → conv_out, with the scale/shift
//! that maps latents to the decoder's input range. NCHW throughout.

use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::Array;

use super::conv_layers::{ConvLayer, ConvNormOut};
use super::encoder::{Encoder, VaeEncoderConfig};
use super::mid_block::UNetMidBlock;
use super::up_decoder_block::UpDecoderBlock;
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

/// Per-up-block `(num_resnet_layers, add_upsample)`.
#[derive(Debug, Clone)]
pub struct VaeDecoderConfig {
    pub up_blocks: Vec<(usize, bool)>,
}

impl VaeDecoderConfig {
    /// The production Z-Image VAE decoder: 4 up-blocks of 3 resnets, upsampling on the first 3.
    pub fn default_z_image() -> Self {
        Self {
            up_blocks: vec![(3, true), (3, true), (3, true), (3, false)],
        }
    }
}

pub struct Decoder {
    conv_in: ConvLayer,
    mid_block: UNetMidBlock,
    up_blocks: Vec<UpDecoderBlock>,
    conv_norm_out: ConvNormOut,
    conv_out: ConvLayer,
}

impl Decoder {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &VaeDecoderConfig) -> Result<Self> {
        // Support an empty top-level prefix (sub-module prefixes are always non-empty).
        let p = |s: &str| {
            if prefix.is_empty() {
                s.to_string()
            } else {
                format!("{prefix}.{s}")
            }
        };
        let up_blocks = cfg
            .up_blocks
            .iter()
            .enumerate()
            .map(|(i, &(layers, up))| {
                UpDecoderBlock::from_weights(w, &p(&format!("up_blocks.{i}")), layers, up)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv_in: ConvLayer::from_weights(w, &p("conv_in"))?,
            mid_block: UNetMidBlock::from_weights(w, &p("mid_block"))?,
            up_blocks,
            conv_norm_out: ConvNormOut::from_weights(w, &p("conv_norm_out"))?,
            conv_out: ConvLayer::from_weights(w, &p("conv_out"))?,
        })
    }

    /// Quantize the decoder's only quantizable Linears — the mid-block attention (conv_in/out,
    /// up-blocks, and norms are conv/norm, not quantized).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.mid_block.quantize(bits)
    }

    /// `latents` NCHW → image NCHW (3 channels, spatial ×8).
    pub fn forward(&self, latents: &Array) -> Result<Array> {
        let mut h = self.conv_in.forward(latents)?;
        h = self.mid_block.forward(&h)?;
        for up in &self.up_blocks {
            h = up.forward(&h)?;
        }
        h = self.conv_norm_out.forward(&h)?;
        h = silu(&h)?;
        self.conv_out.forward(&h)
    }
}

/// The Z-Image VAE. `decode` undoes the latent scale/shift then runs the decoder; `encode`
/// (img2img) runs the encoder and maps the predicted mean into latent space. The encoder is
/// optional so a decode-only `Vae` can still be built from decoder weights alone.
pub struct Vae {
    decoder: Decoder,
    encoder: Option<Encoder>,
    scaling_factor: f32,
    shift_factor: f32,
}

impl Vae {
    pub const SCALING_FACTOR: f32 = 0.3611;
    pub const SHIFT_FACTOR: f32 = 0.1159;

    pub fn from_weights(w: &Weights, prefix: &str, cfg: &VaeDecoderConfig) -> Result<Self> {
        Ok(Self {
            decoder: Decoder::from_weights(w, prefix, cfg)?,
            encoder: None,
            scaling_factor: Self::SCALING_FACTOR,
            shift_factor: Self::SHIFT_FACTOR,
        })
    }

    /// Attach the img2img encoder, loaded from `prefix` (the diffusers `encoder.*` tree, remapped
    /// to the crate's internal naming by [`crate::loader::remap_vae_encoder`]).
    pub fn with_encoder(
        mut self,
        w: &Weights,
        prefix: &str,
        cfg: &VaeEncoderConfig,
    ) -> Result<Self> {
        self.encoder = Some(Encoder::from_weights(w, prefix, cfg)?);
        Ok(self)
    }

    /// Quantize the VAE's quantizable Linears (the decoder's — and, if loaded, the encoder's —
    /// mid-block spatial attention) to Q4/Q8. The VAE is otherwise all conv, so this is the full
    /// set the fork's `nn.quantize(vae, …)` hits. Output is pixel-unchanged in practice (the VAE
    /// quant is measurably 0% px on the decode), so this is for memory/`nn.quantize` faithfulness.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.decoder.quantize(bits)?;
        if let Some(encoder) = self.encoder.as_mut() {
            encoder.quantize(bits)?;
        }
        Ok(())
    }

    /// Image NCHW `[1,3,H,W]` (or `[1,3,1,H,W]`) → latent `[1,16,H/8,W/8]`. Port of the fork's
    /// `VAE.encode` composed with `VAEUtil.encode`'s 5-D→4-D fixup: run the encoder, take the
    /// distribution **mean** (first half of the channels), then map to latent space as
    /// `(mean - shift) * scaling`.
    pub fn encode(&self, image: &Array) -> Result<Array> {
        let encoder = self.encoder.as_ref().ok_or_else(|| {
            Error::Msg("z_image VAE encoder not loaded (img2img unavailable)".into())
        })?;
        let sh = image.shape();
        let image4 = if sh.len() == 5 {
            image.reshape(&[sh[0], sh[1], sh[3], sh[4]])?
        } else {
            image.clone()
        };
        let h = encoder.forward(&image4)?; // [1, 2C, H/8, W/8]
        if h.shape()[1] % 2 != 0 {
            return Err(Error::Msg(format!(
                "z-image vae encode: expected an even (2C: mean|logvar) channel count, got {}",
                h.shape()[1]
            )));
        }
        let c = h.shape()[1] / 2;
        let idx = Array::from_slice(&(0..c).collect::<Vec<i32>>(), &[c]);
        let mean = h.take_axis(&idx, 1)?; // first C channels
        Ok(multiply(
            &subtract(&mean, Array::from_slice(&[self.shift_factor], &[1]))?,
            Array::from_slice(&[self.scaling_factor], &[1]),
        )?)
    }

    /// `latents`: `(B, C, F, H, W)` (F squeezed) or `(B, C, H, W)` → image `(B, 3, 1, H·8, W·8)`.
    pub fn decode(&self, latents: &Array) -> Result<Array> {
        let sh = latents.shape();
        let latents4 = if sh.len() == 5 {
            // squeeze the (size-1) frame axis: (B,C,1,H,W) -> (B,C,H,W)
            latents.reshape(&[sh[0], sh[1], sh[3], sh[4]])?
        } else {
            latents.clone()
        };
        let scaled = add(
            &multiply(
                &latents4,
                Array::from_slice(&[1.0 / self.scaling_factor], &[1]),
            )?,
            Array::from_slice(&[self.shift_factor], &[1]),
        )?;
        let decoded = self.decoder.forward(&scaled)?;
        let d = decoded.shape();
        Ok(decoded.reshape(&[d[0], d[1], 1, d[2], d[3]])?) // restore frame axis
    }

    pub fn decoder(&self) -> &Decoder {
        &self.decoder
    }
}
