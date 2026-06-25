//! Qwen-Image causal-Conv3d VAE — encoder (image → 16-ch latent) + decoder (latent → image),
//! ported 1:1 from the fork (`qwen_vae.py`, `qwen_image_{encoder,decoder}_3d.py`). NCTHW I/O;
//! for T2I the temporal axis is a singleton (T=1).
//!
//! Latents are scaled by per-channel `mean`/`std` (fork `QwenVAE.LATENTS_{MEAN,STD}`). Weight keys
//! mirror the fork's *internal* module tree (`decoder.conv_in.conv3d.weight`,
//! `encoder.down_blocks.0.resnets.0.…`, etc.), so a `Weights` dumped from a loaded fork VAE drops
//! straight in. The on-disk-snapshot key remapping (diffusers → this tree) is applied by the loader
//! (`remap_vae_keys`).

pub mod blocks;

use mlx_rs::ops::{add, divide, multiply, split, subtract};
use mlx_rs::Array;

use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::{LatentDecoder, Result};

use blocks::{rms_norm_channels, CausalConv3d, DownBlock3D, MidBlock3D, UpBlock3D, NORM_EPS};

// fork QwenVAE.LATENTS_{MEAN,STD}, reshaped to (1, 16, 1, 1, 1) for NCTHW broadcast.
#[rustfmt::skip]
const LATENTS_MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];
#[rustfmt::skip]
const LATENTS_STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916,
];

/// Image → 32-ch (sliced to 16) latent. 4 down-stages (3 spatial-downsample), then a mid-block.
pub struct Encoder3D {
    conv_in: CausalConv3d,
    down_blocks: Vec<DownBlock3D>,
    mid_block: MidBlock3D,
    norm_out: Array,
    conv_out: CausalConv3d,
}

impl Encoder3D {
    pub fn from_weights(w: &Weights) -> Result<Self> {
        // (dim transitions 96→96→192→384→384; stages 0–2 downsample, stage 3 does not.)
        let mut down_blocks = Vec::with_capacity(4);
        for i in 0..4 {
            down_blocks.push(DownBlock3D::from_weights(
                w,
                &format!("encoder.down_blocks.{i}"),
                2,
                i < 3,
            )?);
        }
        Ok(Self {
            conv_in: CausalConv3d::from_weights(w, "encoder.conv_in", 1)?,
            down_blocks,
            mid_block: MidBlock3D::from_weights(w, "encoder.mid_block")?,
            norm_out: w.require("encoder.norm_out.weight")?.clone(),
            conv_out: CausalConv3d::from_weights(w, "encoder.conv_out", 1)?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = self.conv_in.forward(x)?;
        for block in &self.down_blocks {
            x = block.forward(&x)?;
        }
        x = self.mid_block.forward(&x)?;
        x = rms_norm_channels(&x, &self.norm_out, NORM_EPS)?;
        self.conv_out.forward(&silu(&x)?)
    }
}

/// 16-ch latent → 3-ch image. Mid-block, then 4 up-stages (3 spatial-upsample).
pub struct Decoder3D {
    conv_in: CausalConv3d,
    mid_block: MidBlock3D,
    up_blocks: Vec<UpBlock3D>,
    norm_out: Array,
    conv_out: CausalConv3d,
}

impl Decoder3D {
    pub fn from_weights(w: &Weights) -> Result<Self> {
        // up_block0/1/2 upsample (2× spatial each → 8×); up_block3 does not.
        let mut up_blocks = Vec::with_capacity(4);
        for i in 0..4 {
            up_blocks.push(UpBlock3D::from_weights(
                w,
                &format!("decoder.up_block{i}"),
                2,
                i < 3,
            )?);
        }
        Ok(Self {
            conv_in: CausalConv3d::from_weights(w, "decoder.conv_in", 1)?,
            mid_block: MidBlock3D::from_weights(w, "decoder.mid_block")?,
            up_blocks,
            norm_out: w.require("decoder.norm_out.weight")?.clone(),
            conv_out: CausalConv3d::from_weights(w, "decoder.conv_out", 1)?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = self.conv_in.forward(x)?;
        x = self.mid_block.forward(&x)?;
        for block in &self.up_blocks {
            x = block.forward(&x)?;
        }
        x = rms_norm_channels(&x, &self.norm_out, NORM_EPS)?;
        self.conv_out.forward(&silu(&x)?)
    }
}

/// The Qwen-Image VAE: `encode` (image → scaled 16-ch latent) and `decode` (latent → image).
pub struct QwenVae {
    encoder: Encoder3D,
    decoder: Decoder3D,
    quant_conv: CausalConv3d,
    post_quant_conv: CausalConv3d,
    mean: Array,
    std: Array,
}

impl QwenVae {
    pub fn from_weights(w: &Weights) -> Result<Self> {
        Ok(Self {
            encoder: Encoder3D::from_weights(w)?,
            decoder: Decoder3D::from_weights(w)?,
            quant_conv: CausalConv3d::from_weights(w, "quant_conv", 0)?,
            post_quant_conv: CausalConv3d::from_weights(w, "post_quant_conv", 0)?,
            mean: Array::from_slice(&LATENTS_MEAN, &[1, 16, 1, 1, 1]),
            std: Array::from_slice(&LATENTS_STD, &[1, 16, 1, 1, 1]),
        })
    }

    /// 16-ch latent (NCHW or NCTHW) → image `(B, 3, 1, H, W)`. Denormalizes by `std`/`mean`.
    pub fn decode(&self, latents: &Array) -> Result<Array> {
        let l = to_5d(latents)?;
        let l = add(&multiply(&l, &self.std)?, &self.mean)?;
        let l = self.post_quant_conv.forward(&l)?;
        self.decoder.forward(&l)
    }

    /// Image `(B, 3, H, W)` (or NCTHW) → scaled 16-ch latent `(B, 16, 1, H/8, W/8)`.
    pub fn encode(&self, image: &Array) -> Result<Array> {
        let x = to_5d(image)?;
        let e = self.encoder.forward(&x)?;
        let e = self.quant_conv.forward(&e)?;
        let e16 = split(&e, 2, 1)?.swap_remove(0); // keep first 16 of 32 channels
        Ok(divide(&subtract(&e16, &self.mean)?, &self.std)?)
    }
}

/// The native decoder for the Qwen-Image latent space (the behavior-preserving default of the
/// PiD decode seam, sc-7844). Delegates to the inherent [`QwenVae::decode`]; a PiD decoder for this
/// same latent space (`mlx-gen-pid`, sc-7843/7845) implements the same trait so an engine can swap
/// between them at the decode call site.
impl LatentDecoder for QwenVae {
    fn decode(&self, latents: &Array) -> Result<Array> {
        QwenVae::decode(self, latents)
    }
}

/// Add a singleton temporal axis to a 4-D `(B, C, H, W)` tensor → `(B, C, 1, H, W)`.
fn to_5d(x: &Array) -> Result<Array> {
    if x.shape().len() == 4 {
        let s = x.shape();
        Ok(x.reshape(&[s[0], s[1], 1, s[2], s[3]])?)
    } else {
        Ok(x.clone())
    }
}
