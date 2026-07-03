//! SVD (img2vid-xt) component configs — transcribed from the checkpoint JSON
//! (`stabilityai/stable-video-diffusion-img2vid-xt`): `unet/config.json`, `vae/config.json`,
//! `image_encoder/config.json`, `scheduler/scheduler_config.json`. Static defaults, constructed via
//! `::default()` at load — there is NO disk-JSON override path; a future checkpoint that differs would
//! need one added here.

/// `UNetSpatioTemporalConditionModel` config — only the fields the loader actually reads
/// (`SvdUnet::from_weights`). Channel counts / cross-attn dim / frame count are fixed by the weight
/// shapes and the request, so they are not carried here.
#[derive(Clone, Debug)]
pub struct UnetConfig {
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub num_attention_heads: Vec<usize>,
    pub transformer_layers_per_block: usize,
    /// Each fps/motion_bucket/noise_aug id → a 256-dim sinusoid; 3 of them concat → 768.
    pub addition_time_embed_dim: usize,
}

impl Default for UnetConfig {
    fn default() -> Self {
        Self {
            block_out_channels: vec![320, 640, 1280, 1280],
            layers_per_block: 2,
            num_attention_heads: vec![5, 10, 20, 20],
            transformer_layers_per_block: 1,
            addition_time_embed_dim: 256,
        }
    }
}

/// `AutoencoderKLTemporalDecoder` config (2D encoder + temporal decoder) — only the fields the loader
/// reads (`SvdVae::from_weights`); channel counts are fixed by the weight shapes.
#[derive(Clone, Debug)]
pub struct VaeConfig {
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub scaling_factor: f32,
}

impl Default for VaeConfig {
    fn default() -> Self {
        Self {
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            scaling_factor: 0.18215,
        }
    }
}

/// `CLIPVisionModelWithProjection` (OpenCLIP ViT-H/14) config — image conditioning encoder.
#[derive(Clone, Debug)]
pub struct ImageEncoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub patch_size: usize,
    pub image_size: usize,
    /// Output dim of the visual projection (the image embed fed to UNet cross-attn).
    pub projection_dim: usize,
    pub layer_norm_eps: f32,
}

impl Default for ImageEncoderConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1280,
            intermediate_size: 5120,
            num_hidden_layers: 32,
            num_attention_heads: 16,
            patch_size: 14,
            image_size: 224,
            projection_dim: 1024,
            layer_norm_eps: 1e-5,
        }
    }
}

/// `EulerDiscreteScheduler` (EDM) config for SVD (`use_karras_sigmas`, `timestep_type="continuous"`,
/// `prediction_type="v_prediction"`). The sigma schedule is pure Karras over the **config**
/// `sigma_min`/`sigma_max` (the betas/alphas path is unused) and the model timestep is `0.25·ln(σ)`.
#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    pub sigma_min: f32,
    pub sigma_max: f32,
    /// Karras rho (paper default 7).
    pub rho: f32,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            sigma_min: 0.002,
            sigma_max: 700.0,
            rho: 7.0,
        }
    }
}
