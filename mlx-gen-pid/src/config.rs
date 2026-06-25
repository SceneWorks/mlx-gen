//! PiD network + sampler configuration — a faithful mirror of the reference's hydra config tree
//! (`pid/_src/configs/pid/`), so a fork checkpoint loads against the same shapes.
//!
//! The numbers below are read 1:1 from the reference, NOT inferred:
//! - backbone hyperparameters: `configs/pid/defaults/model_pixeldit.py::PIXELDIT_FINETUNE_2048PX`
//!   and `configs/pid/defaults/model_pid.py::PID_SR4X`;
//! - distill-student + caption fields: `configs/pid/experiment/shared_config.py::_common_model_overrides`;
//! - the `2kto4k` dynamic-shift: `configs/pid/experiment_2kto4k/shared_config.py`.
//!
//! IMPORTANT divergence the reference's *experiment* layer applies on top of `PID_SR4X`: the distilled
//! students run with **`lq_interval = 2`** (every other patch block), not the base `1`
//! (`_common_model_overrides.net.lq_interval = 2`). [`PidConfig::lq_interval`] therefore defaults to 2.

/// Backbone (`PixDiT_T2I` + `PidNet` LQ extension) hyperparameters. Dimension-parametric so the same
/// code runs the real 1.36 B model and tiny parity fixtures.
#[derive(Debug, Clone)]
pub struct PidConfig {
    // ---- PixDiT_T2I backbone (model_pixeldit.py::PIXELDIT_FINETUNE_2048PX) ----
    /// Output pixel channels (RGB).
    pub in_channels: i32,
    /// Patch-stream attention heads.
    pub num_groups: i32,
    /// Patch-stream hidden width.
    pub hidden_size: i32,
    /// Pixel-stream (PiT) per-pixel latent width.
    pub pixel_hidden_size: i32,
    /// Pixel-stream attention projection width.
    pub pixel_attn_hidden_size: i32,
    /// Pixel-stream attention heads.
    pub pixel_num_groups: i32,
    /// Number of MMDiT patch blocks.
    pub patch_depth: i32,
    /// Number of PiT pixel blocks (run after the patch stream).
    pub pixel_depth: i32,
    /// Spatial patch size (token = `patch_size × patch_size` pixels).
    pub patch_size: i32,
    /// Caption-embedding (Gemma-2-2b-it last-hidden) width.
    pub txt_embed_dim: i32,
    /// Caption token budget (`model_max_length`).
    pub txt_max_length: i32,
    /// Apply 1-D RoPE to the text stream.
    pub use_text_rope: bool,
    /// Text RoPE base.
    pub text_rope_theta: f32,
    /// 2-D image RoPE mode (`"ntk_aware"`).
    pub rope_mode: RopeMode,
    /// NTK reference pixel height (grid = `rope_ref_h / patch_size`).
    pub rope_ref_h: i32,
    /// NTK reference pixel width.
    pub rope_ref_w: i32,

    // ---- LQ adapter (model_pid.py::PID_SR4X + experiment override) ----
    /// LQ image-branch input channels (0 = latent-only, the SR default).
    pub lq_in_channels: i32,
    /// LQ latent-branch channels (16 for the Qwen/Flux/SD3 latent spaces).
    pub lq_latent_channels: i32,
    /// LQ projection internal width.
    pub lq_hidden_dim: i32,
    /// ResBlocks after the initial 2-conv projection in each LQ branch (`num_res_blocks`, default 4).
    pub lq_num_res_blocks: i32,
    /// Inject the LQ gate every `lq_interval` patch blocks. **2 for the distilled students.**
    pub lq_interval: i32,
    /// Super-resolution factor baked into the network (4× or 8×).
    pub sr_scale: i32,
    /// VAE spatial compression (latent grid → pixel grid factor).
    pub latent_spatial_down_factor: i32,
}

/// 2-D image RoPE mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeMode {
    /// NTK-aware per-axis θ scaling (identity when the sampled grid == the reference grid).
    NtkAware,
}

impl PidConfig {
    /// Patch-stream per-head dim (`hidden_size / num_groups`).
    pub fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_groups
    }

    /// Pixel-stream per-head dim (`pixel_attn_hidden_size / pixel_num_groups`).
    pub fn pixel_head_dim(&self) -> i32 {
        self.pixel_attn_hidden_size / self.pixel_num_groups
    }

    /// NTK reference patch-grid height.
    pub fn rope_ref_grid_h(&self) -> i32 {
        self.rope_ref_h / self.patch_size
    }

    /// NTK reference patch-grid width.
    pub fn rope_ref_grid_w(&self) -> i32 {
        self.rope_ref_w / self.patch_size
    }

    /// Number of LQ gate-injection points across the patch stream (one gate per injected block).
    pub fn num_lq_outputs(&self) -> i32 {
        // Reference `LQProjection2D`: injection at blocks `0, interval, 2·interval, …` strictly below
        // `patch_depth`. Verified against `lq_projection_2d.py` when the adapter is ported.
        (self.patch_depth + self.lq_interval - 1) / self.lq_interval
    }

    /// The official `qwenimage` / `flux` / `sd3` SR4× backbone (the only released student topology).
    /// All released latent spaces share this PixDiT topology — only `lq_latent_channels`, the latent
    /// norm, and the checkpoint differ (see [`crate::registry`]).
    pub fn sr4x() -> Self {
        Self {
            in_channels: 3,
            num_groups: 24,
            hidden_size: 1536,
            pixel_hidden_size: 16,
            pixel_attn_hidden_size: 1152,
            pixel_num_groups: 16,
            patch_depth: 14,
            pixel_depth: 2,
            patch_size: 16,
            txt_embed_dim: 2304,
            txt_max_length: 300,
            use_text_rope: true,
            text_rope_theta: 10000.0,
            rope_mode: RopeMode::NtkAware,
            rope_ref_h: 1024,
            rope_ref_w: 1024,
            lq_in_channels: 0,
            lq_latent_channels: 16,
            lq_hidden_dim: 512,
            lq_num_res_blocks: 4,
            lq_interval: 2,
            sr_scale: 4,
            latent_spatial_down_factor: 8,
        }
    }
}

/// Distilled 4-step sampler configuration (`PidDistillModelConfig` + the experiment overrides in
/// `_common_model_overrides`). The reference inference path runs SDE / velocity-prediction / cfg 1.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// Flow-matching time scale fed to the timestep embedder (`t * fm_timescale`).
    pub fm_timescale: f32,
    /// The distilled time ladder (length `steps + 1`, ending at 0).
    pub student_t_list: Vec<f32>,
    /// `Sde` (released students) injects fresh noise between steps; `Ode` integrates the velocity.
    pub sample_type: SampleType,
}

/// Distilled sampler stepping mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleType {
    /// `x = (1 − t_next)·x0 + t_next·ε_new` (the released 4-step students).
    Sde,
    /// `x = x + (t_next − t_cur)·v`.
    Ode,
}

impl SamplerConfig {
    /// The released 4-step student schedule (`student_t_list=[0.999,0.866,0.634,0.342,0.0]`, SDE).
    pub fn distill_4step() -> Self {
        Self {
            fm_timescale: 1000.0,
            student_t_list: vec![0.999, 0.866, 0.634, 0.342, 0.0],
            sample_type: SampleType::Sde,
        }
    }
}

/// Caption-conditioning configuration (`_common_model_overrides`). PiD conditions on a `gemma-2-2b-it`
/// embedding of the prompt, prefixed by a fixed "Chi-prompt", then a learned `y_norm` scale.
#[derive(Debug, Clone)]
pub struct CaptionConfig {
    /// HF id of the caption text encoder.
    pub text_encoder_name: &'static str,
    /// Caption embedding width (must equal [`PidConfig::txt_embed_dim`]).
    pub caption_channels: i32,
    /// Caption token budget.
    pub model_max_length: i32,
    /// Apply the learned `y_norm` scaling to the projected caption tokens.
    pub y_norm: bool,
    /// `y_norm` scale factor.
    pub y_norm_scale_factor: f32,
    /// Whether the fixed "Chi-prompt" enhancement prefix is prepended before tokenizing.
    pub use_chi_prompt: bool,
}

impl Default for CaptionConfig {
    fn default() -> Self {
        Self {
            text_encoder_name: "gemma-2-2b-it",
            caption_channels: 2304,
            model_max_length: 300,
            y_norm: true,
            y_norm_scale_factor: 0.01,
            use_chi_prompt: true,
        }
    }
}
