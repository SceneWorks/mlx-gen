//! DC-AE decoder configuration (epic 8485, spike sc-8486).
//!
//! Values mirror the diffusers `AutoencoderDC` config for `mit-han-lab/dc-ae-f32c32-sana-1.0`
//! (the autoencoder behind SANA-1.6B 1024px). Only the **decoder** is modelled here — the spike's
//! sole question is whether the f32 deep-compression *decode* reproduces cleanly on Metal.

/// Per-stage block kind. The SANA-1.0 decoder runs `ResBlock` in the three shallow (high-res) stages
/// and `EfficientViTBlock` (linear attention) in the three deep (low-res) stages.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlockType {
    Res,
    EfficientVit,
}

/// Decoder hyper-parameters. Stored stage order is shallow→deep (index 0 = 128-channel/full-res
/// stage … index 5 = 1024-channel/lowest-res stage), matching the on-disk `decoder.up_blocks.{i}`
/// numbering. Decode iterates them deep→shallow.
#[derive(Clone, Debug)]
pub struct DcAeConfig {
    pub in_channels: i32,
    pub latent_channels: i32,
    pub attention_head_dim: i32,
    pub block_out_channels: Vec<i32>,
    pub layers_per_block: Vec<i32>,
    pub block_types: Vec<BlockType>,
    /// One `kernel_size` per multiscale QKV projection in the EfficientViT stages (`[5]` for SANA-1.0).
    pub qkv_multiscales: Vec<i32>,
    /// `True` → upsample by nearest-`interpolate` + conv (SANA-1.0). `False` → conv + pixel-shuffle.
    pub upsample_interpolate: bool,
    /// RMS-norm epsilon (`1e-5` throughout the decoder).
    pub norm_eps: f32,
    /// Linear-attention denominator epsilon (`1e-15`).
    pub attn_eps: f32,
    /// VAE latent scaling factor (`z_decode = z / scaling_factor`). Applied by the caller, not the
    /// decoder, mirroring diffusers `Decoder.forward` (which receives an already-scaled latent).
    pub scaling_factor: f32,
}

impl DcAeConfig {
    /// `mit-han-lab/dc-ae-f32c32-sana-1.0` decoder config.
    pub fn sana_f32c32() -> Self {
        use BlockType::{EfficientVit as E, Res as R};
        Self {
            in_channels: 3,
            latent_channels: 32,
            attention_head_dim: 32,
            block_out_channels: vec![128, 256, 512, 512, 1024, 1024],
            layers_per_block: vec![3, 3, 3, 3, 3, 3],
            block_types: vec![R, R, R, E, E, E],
            qkv_multiscales: vec![5],
            upsample_interpolate: true,
            norm_eps: 1e-5,
            attn_eps: 1e-15,
            scaling_factor: 0.41407,
        }
    }

    pub fn num_stages(&self) -> usize {
        self.block_out_channels.len()
    }
}
