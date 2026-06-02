//! SDXL-base-1.0 model dimensions — the Rust mirror of the vendored `_vendor/mlx_sd/config.py`
//! dataclasses, with the constants resolved to the real
//! `stabilityai/stable-diffusion-xl-base-1.0` config.json values (the loader asserts the snapshot
//! matches). SDXL is a U-Net, not a DiT, so none of this reuses the transformer-family configs.

/// Conditional 2-D U-Net config. Defaults are SDXL-base-1.0
/// (`unet/config.json`): 3 blocks, the cross-attention stack on the inner two, `text_time`
/// added-conditioning (the SDXL pooled-text + micro-conditioning embedding).
#[derive(Clone, Debug)]
pub struct UNetConfig {
    pub in_channels: i32,
    pub out_channels: i32,
    pub conv_in_kernel: i32,
    pub conv_out_kernel: i32,
    pub block_out_channels: Vec<i32>,
    /// `layers_per_block` broadcast to one entry per block.
    pub layers_per_block: Vec<i32>,
    pub transformer_layers_per_block: Vec<i32>,
    /// Number of attention *heads* per block (diffusers `attention_head_dim`, here head-count).
    pub num_attention_heads: Vec<i32>,
    pub cross_attention_dim: Vec<i32>,
    pub norm_num_groups: i32,
    pub down_block_types: Vec<String>,
    pub up_block_types: Vec<String>,
    /// `"text_time"` for SDXL (the pooled-text + time-ids added embedding), else `None`.
    pub addition_embed_type: Option<String>,
    pub addition_time_embed_dim: Option<i32>,
    pub projection_class_embeddings_input_dim: Option<i32>,
}

impl UNetConfig {
    /// SDXL-base-1.0 U-Net (`stabilityai/stable-diffusion-xl-base-1.0`, `unet/config.json`).
    pub fn sdxl_base() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            conv_in_kernel: 3,
            conv_out_kernel: 3,
            block_out_channels: vec![320, 640, 1280],
            layers_per_block: vec![2, 2, 2],
            transformer_layers_per_block: vec![1, 2, 10],
            num_attention_heads: vec![5, 10, 20],
            cross_attention_dim: vec![2048, 2048, 2048],
            norm_num_groups: 32,
            down_block_types: vec![
                "DownBlock2D".into(),
                "CrossAttnDownBlock2D".into(),
                "CrossAttnDownBlock2D".into(),
            ],
            // diffusers config order is [CrossAttn, CrossAttn, UpBlock]; the vendored loader
            // reverses it (`up_block_types[::-1]`) and indexes by the construction index `i`
            // (= our `config_i = n-1-k`). Stored already-reversed to match — so `config_i` 0/1/2
            // map to UpBlock/CrossAttn/CrossAttn (i.e. the highest-res up block has no attention).
            up_block_types: vec![
                "UpBlock2D".into(),
                "CrossAttnUpBlock2D".into(),
                "CrossAttnUpBlock2D".into(),
            ],
            addition_embed_type: Some("text_time".into()),
            addition_time_embed_dim: Some(256),
            projection_class_embeddings_input_dim: Some(2816),
        }
    }

    pub fn num_blocks(&self) -> usize {
        self.block_out_channels.len()
    }

    /// `block_out_channels[0] * 4` — the time-embedding dimension shared across the U-Net.
    pub fn time_embed_dim(&self) -> i32 {
        self.block_out_channels[0] * 4
    }
}

/// CLIP text-encoder config. SDXL has two: CLIP-L (`text_encoder`, no projection) and OpenCLIP
/// bigG (`text_encoder_2`, with a final projection — `CLIPTextModelWithProjection`).
#[derive(Clone, Debug)]
pub struct ClipTextConfig {
    pub num_layers: i32,
    pub model_dims: i32,
    pub num_heads: i32,
    pub max_length: i32,
    pub vocab_size: i32,
    /// `Some(dim)` only for the projection variant (TE2); the pooled EOS is then projected.
    pub projection_dim: Option<i32>,
    /// `"quick_gelu"` (CLIP-L) or `"gelu"` (bigG).
    pub hidden_act: ClipActivation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClipActivation {
    QuickGelu,
    Gelu,
}

impl ClipTextConfig {
    /// SDXL `text_encoder` — CLIP-L (`text_encoder/config.json`): 768-wide, 12 layers, quick_gelu,
    /// architecture `CLIPTextModel` (no projection used for conditioning).
    pub fn sdxl_te1() -> Self {
        Self {
            num_layers: 12,
            model_dims: 768,
            num_heads: 12,
            max_length: 77,
            vocab_size: 49408,
            projection_dim: None,
            hidden_act: ClipActivation::QuickGelu,
        }
    }

    /// SDXL `text_encoder_2` — OpenCLIP bigG (`text_encoder_2/config.json`): 1280-wide, 32 layers,
    /// gelu, architecture `CLIPTextModelWithProjection` (projection 1280 → pooled conditioning).
    pub fn sdxl_te2() -> Self {
        Self {
            num_layers: 32,
            model_dims: 1280,
            num_heads: 20,
            max_length: 77,
            vocab_size: 49408,
            projection_dim: Some(1280),
            hidden_act: ClipActivation::Gelu,
        }
    }
}

/// SDXL VAE (autoencoder) config (`vae/config.json`). Note `scaling_factor = 0.13025`, distinct
/// from SD-2.1's 0.18215.
#[derive(Clone, Debug)]
pub struct VaeConfig {
    pub in_channels: i32,
    pub out_channels: i32,
    /// Encoder output channels (`2 * latent_channels` — the mean/logvar split).
    pub latent_channels_out: i32,
    /// Latent channels fed to the decoder (`latent_channels`).
    pub latent_channels_in: i32,
    pub block_out_channels: Vec<i32>,
    pub layers_per_block: i32,
    pub norm_num_groups: i32,
    pub scaling_factor: f32,
}

impl VaeConfig {
    /// SDXL-base-1.0 VAE.
    pub fn sdxl_base() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels_out: 8, // 2 * 4
            latent_channels_in: 4,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            norm_num_groups: 32,
            scaling_factor: 0.13025,
        }
    }
}

/// Diffusion (noise-schedule) config driving the discrete-Euler sampler. SDXL's
/// `scheduler/scheduler_config.json`: `scaled_linear` betas 0.00085 → 0.012 over 1000 steps.
#[derive(Clone, Debug)]
pub struct DiffusionConfig {
    pub beta_schedule: BetaSchedule,
    pub beta_start: f32,
    pub beta_end: f32,
    pub num_train_steps: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BetaSchedule {
    Linear,
    ScaledLinear,
}

impl DiffusionConfig {
    /// SDXL-base-1.0 schedule.
    pub fn sdxl_base() -> Self {
        Self {
            beta_schedule: BetaSchedule::ScaledLinear,
            beta_start: 0.00085,
            beta_end: 0.012,
            num_train_steps: 1000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdxl_unet_dims_match_config_json() {
        let c = UNetConfig::sdxl_base();
        assert_eq!(c.num_blocks(), 3);
        assert_eq!(c.time_embed_dim(), 1280);
        assert_eq!(c.transformer_layers_per_block, vec![1, 2, 10]);
        assert_eq!(c.projection_class_embeddings_input_dim, Some(2816));
        // add_embedding input = pooled(1280) + add_time_proj(6 ids * 256) = 1280 + 1536 = 2816.
        let pooled = ClipTextConfig::sdxl_te2().projection_dim.unwrap();
        let time_ids = 6 * c.addition_time_embed_dim.unwrap();
        assert_eq!(pooled + time_ids, 2816);
    }

    #[test]
    fn vae_scaling_factor_is_sdxl_specific() {
        assert_eq!(VaeConfig::sdxl_base().scaling_factor, 0.13025);
    }

    #[test]
    fn conditioning_width_is_2048() {
        // SDXL conditioning = concat(TE1.hidden[-2]=768, TE2.hidden[-2]=1280).
        let w = ClipTextConfig::sdxl_te1().model_dims + ClipTextConfig::sdxl_te2().model_dims;
        assert_eq!(w, 2048);
        assert_eq!(UNetConfig::sdxl_base().cross_attention_dim[0], 2048);
    }
}
