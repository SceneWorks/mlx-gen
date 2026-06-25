//! Stable Diffusion 3.5 (MMDiT) configuration.
//!
//! The arch facts below were **empirically confirmed on the real weights** during the spike
//! (sc-7850): a tensor-header + `transformer/config.json` audit of every published
//! `stabilityai/stable-diffusion-3.5-*` repo. SD3.5-Large and Large-Turbo share one MMDiT
//! (38 layers, hidden 2432, 38 heads, head_dim 64); they differ only in the inference schedule
//! (Turbo is an ADD-distilled 4-step checkpoint), not in tensor layout — so one converter and one
//! config (parameterized by [`Sd3Variant`]) serves both.
//!
//! This is the **E1** slice: arch constants + the diffusers→MLX converter (see [`crate::convert`])
//! plus architecture validation. The triple-TE aggregator (E2), the MMDiT forward (E3), and the
//! 16-channel VAE (E4) are separate stories and are NOT implemented here.
//!
//! The config is **dimension-parametric** (the mlx-gen convention): the same struct describes the
//! real 8.1B Large model and a tiny parity fixture, so the converter / validator run against both.

use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, Capabilities, Modality, ModelDescriptor, Quant,
};

pub const SD3_5_LARGE_ID: &str = "sd3_5_large";
pub const SD3_5_LARGE_TURBO_ID: &str = "sd3_5_large_turbo";
pub const SD3_5_MEDIUM_ID: &str = "sd3_5_medium";

pub const DEFAULT_WIDTH: u32 = 1024;
pub const DEFAULT_HEIGHT: u32 = 1024;

/// SD3.5-Large is a true-CFG model: the reference pipeline generates at guidance ~3.5 (range 1–7)
/// over ~28–40 steps. Large-Turbo is ADD-distilled to ~4 steps at guidance 1.0 (no CFG).
pub const DEFAULT_GUIDANCE_LARGE: f32 = 3.5;
pub const DEFAULT_GUIDANCE_TURBO: f32 = 1.0;
pub const DEFAULT_STEPS_LARGE: u32 = 28;
pub const DEFAULT_STEPS_TURBO: u32 = 4;
/// SD3.5-Medium is a true-CFG model; its reference pipeline samples ~40 steps at guidance ~5 (it is
/// more guidance-sensitive than Large per Stability's model card).
pub const DEFAULT_GUIDANCE_MEDIUM: f32 = 5.0;
pub const DEFAULT_STEPS_MEDIUM: u32 = 40;

/// The base flow-match sampler name in the capability surface. An unset `req.sampler` resolves to
/// this — SD3.5's flow-match Euler over a shift-resolved logit-normal sigma schedule (the unified
/// sampler framework, epic 7114; logit-normal weighting reused from mlx-gen-ideogram per the spike).
pub const DEFAULT_SAMPLER: &str = "flow_match";

// ----------------------------------------------------------------------------------------------
// SD3.5-Large / Large-Turbo MMDiT architecture (sc-7850, real-weight confirmed)
// ----------------------------------------------------------------------------------------------

/// `SD3Transformer2DModel.num_layers` — the count of all-double-stream [`JointTransformerBlock`]s.
pub const LARGE_NUM_LAYERS: usize = 38;
/// `attention_head_dim` — per-head channel width (qk_norm RMSNorm `weight` is `[HEAD_DIM]`).
pub const LARGE_HEAD_DIM: usize = 64;
/// `num_attention_heads`.
pub const LARGE_NUM_HEADS: usize = 38;
/// `inner_dim` = `num_attention_heads * attention_head_dim` = 38 × 64 = **2432** (the hidden size).
pub const LARGE_HIDDEN: usize = LARGE_NUM_HEADS * LARGE_HEAD_DIM;
/// `patch_size` — the 2×2 latent patchify factor (`pos_embed.proj` is a `Conv2d` kernel 2, stride 2).
pub const LARGE_PATCH_SIZE: usize = 2;
/// `in_channels` — 16-channel latents (the SD3.5 16-ch VAE).
pub const LARGE_IN_CHANNELS: usize = 16;
/// `out_channels` — also 16 (`proj_out` is `[patch*patch*out_channels, hidden]` = `[64, 2432]`).
pub const LARGE_OUT_CHANNELS: usize = 16;
/// `joint_attention_dim` — the text-stream feature width fed into `context_embedder`
/// (`context_embedder.weight` is `[caption_projection_dim, joint_attention_dim]` = `[2432, 4096]`).
pub const LARGE_JOINT_ATTENTION_DIM: usize = 4096;
/// `pooled_projection_dim` — the pooled-CLIP text projection fed into the timestep/text embedder
/// (`time_text_embed.text_embedder.linear_1.weight` is `[hidden, 2048]`).
pub const LARGE_POOLED_PROJECTION_DIM: usize = 2048;
/// `caption_projection_dim` — the per-token text feature width inside the MMDiT (== hidden, 2432).
pub const LARGE_CAPTION_PROJECTION_DIM: usize = LARGE_HIDDEN;
/// `pos_embed_max_size` — the max latent edge (in patches) the learned positional table spans.
/// SD3.5-Large sets this to `192` (the SD3-base default of 96 was doubled), so the table is
/// `[1, 192*192, hidden]` = `[1, 36864, 2432]` (real-weight confirmed, sc-7850).
pub const LARGE_POS_EMBED_MAX_SIZE: usize = 192;
/// The flattened length of the learned positional table (`pos_embed_max_size^2`).
pub const LARGE_POS_EMBED_LEN: usize = LARGE_POS_EMBED_MAX_SIZE * LARGE_POS_EMBED_MAX_SIZE;
/// The Fourier timestep-embedding input width (`timestep_embedder.linear_1.weight` is `[hidden, 256]`).
pub const LARGE_TIME_PROJ_DIM: usize = 256;
/// qk-RMSNorm epsilon (diffusers `Attention(eps=1e-6)`, RMSNorm path).
pub const RMS_EPS: f32 = 1e-6;

// ----------------------------------------------------------------------------------------------
// SD3.5-Medium MMDiT-X architecture (sc-7867 / M1, real-weight confirmed from the cached
// `stabilityai/stable-diffusion-3.5-medium` `transformer/config.json` + safetensors header)
// ----------------------------------------------------------------------------------------------
//
// Medium is an **MMDiT-X**: structurally an SD3.5 MMDiT, but its FIRST 13 of 24 blocks carry a
// SECOND, image-stream-only self-attention (diffusers `attn2`) alongside the joint attention.
// Those dual-attention blocks also use an EXTENDED AdaLayerNormZero — `norm1.linear` packs 9 chunks
// (`9·hidden`) instead of 6, the extra 3 being shift/scale/gate for the `attn2` branch. The remaining
// 11 blocks (13..23) are plain MMDiT joint blocks (`norm1` = `6·hidden`, no `attn2`). The last block
// (23) is `context_pre_only` exactly as in Large. Confirmed against the real 909-tensor checkpoint.

/// `SD3Transformer2DModel.num_layers` for Medium — 24 joint blocks.
pub const MEDIUM_NUM_LAYERS: usize = 24;
/// `attention_head_dim` — per-head channel width (qk_norm RMSNorm `weight` is `[HEAD_DIM]`).
pub const MEDIUM_HEAD_DIM: usize = 64;
/// `num_attention_heads`.
pub const MEDIUM_NUM_HEADS: usize = 24;
/// `inner_dim` = `num_attention_heads * attention_head_dim` = 24 × 64 = **1536** (the hidden size).
pub const MEDIUM_HIDDEN: usize = MEDIUM_NUM_HEADS * MEDIUM_HEAD_DIM;
/// `patch_size` — the 2×2 latent patchify factor.
pub const MEDIUM_PATCH_SIZE: usize = 2;
/// `in_channels` — 16-channel latents (the SD3.5 16-ch VAE).
pub const MEDIUM_IN_CHANNELS: usize = 16;
/// `out_channels` — also 16 (`proj_out` is `[patch*patch*out_channels, hidden]` = `[64, 1536]`).
pub const MEDIUM_OUT_CHANNELS: usize = 16;
/// `joint_attention_dim` — the text-stream feature width fed into `context_embedder` (`[1536, 4096]`).
pub const MEDIUM_JOINT_ATTENTION_DIM: usize = 4096;
/// `pooled_projection_dim` — the pooled-CLIP text projection (`text_embedder.linear_1` is `[1536, 2048]`).
pub const MEDIUM_POOLED_PROJECTION_DIM: usize = 2048;
/// `caption_projection_dim` — the per-token text feature width inside the MMDiT (== hidden, 1536).
pub const MEDIUM_CAPTION_PROJECTION_DIM: usize = MEDIUM_HIDDEN;
/// `pos_embed_max_size` — the max latent edge (in patches) the learned positional table spans.
/// SD3.5-Medium sets this to `384` (1440²-capable; double Large's 192), so the table is
/// `[1, 384*384, hidden]` = `[1, 147456, 1536]` (real-weight confirmed, sc-7867).
pub const MEDIUM_POS_EMBED_MAX_SIZE: usize = 384;
/// The flattened length of the learned positional table (`pos_embed_max_size^2` = 147456).
pub const MEDIUM_POS_EMBED_LEN: usize = MEDIUM_POS_EMBED_MAX_SIZE * MEDIUM_POS_EMBED_MAX_SIZE;
/// The Fourier timestep-embedding input width (`timestep_embedder.linear_1.weight` is `[hidden, 256]`).
pub const MEDIUM_TIME_PROJ_DIM: usize = 256;
/// The number of MMDiT-X dual-attention (`attn2`) blocks — the FIRST 13 of 24 (indices `0..=12`).
/// Matches the real `dual_attention_layers = [0, 1, …, 12]` config list.
pub const MEDIUM_DUAL_ATTENTION_LAYERS: usize = 13;

/// The SD3.5 variants this crate targets. Large is the E1 target; Turbo is the same MMDiT/VAE/TE
/// tensor layout at a distilled few-step schedule, so it is a near-free addition (only the schedule
/// constants differ). Medium (M1, sc-7867) is the MMDiT-X variant — a distinct, smaller transformer
/// (24 blocks, hidden 1536) whose first 13 blocks carry a second `attn2` self-attention; its
/// converter / arch validation reuse this same scaffolding, parameterized by [`Sd3Arch::medium`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sd3Variant {
    /// Stable Diffusion 3.5 Large (8.1B), true-CFG.
    Large,
    /// Stable Diffusion 3.5 Large Turbo, ADD-distilled few-step (guidance 1.0, ~4 steps).
    LargeTurbo,
    /// Stable Diffusion 3.5 Medium (2.5B), MMDiT-X (dual-attention first 13 blocks), true-CFG.
    Medium,
}

impl Sd3Variant {
    pub fn id(self) -> &'static str {
        match self {
            Self::Large => SD3_5_LARGE_ID,
            Self::LargeTurbo => SD3_5_LARGE_TURBO_ID,
            Self::Medium => SD3_5_MEDIUM_ID,
        }
    }

    pub fn hf_model(self) -> &'static str {
        match self {
            Self::Large => "stabilityai/stable-diffusion-3.5-large",
            Self::LargeTurbo => "stabilityai/stable-diffusion-3.5-large-turbo",
            Self::Medium => "stabilityai/stable-diffusion-3.5-medium",
        }
    }

    pub fn default_steps(self) -> u32 {
        match self {
            Self::Large => DEFAULT_STEPS_LARGE,
            Self::LargeTurbo => DEFAULT_STEPS_TURBO,
            Self::Medium => DEFAULT_STEPS_MEDIUM,
        }
    }

    pub fn default_guidance(self) -> f32 {
        match self {
            Self::Large => DEFAULT_GUIDANCE_LARGE,
            Self::LargeTurbo => DEFAULT_GUIDANCE_TURBO,
            Self::Medium => DEFAULT_GUIDANCE_MEDIUM,
        }
    }

    /// True-CFG: Large / Medium run a negative prompt + a guidance scale >1. Turbo is distilled to a
    /// single (cond-only) forward — guidance 1.0, no negative prompt.
    pub fn supports_true_cfg(self) -> bool {
        matches!(self, Self::Large | Self::Medium)
    }

    /// The MMDiT arch for this variant. Large / Large-Turbo share one plain-MMDiT layout; Medium is
    /// the MMDiT-X layout.
    pub fn arch(self) -> Sd3Arch {
        match self {
            Self::Large | Self::LargeTurbo => Sd3Arch::large(),
            Self::Medium => Sd3Arch::medium(),
        }
    }

    pub fn descriptor(self) -> ModelDescriptor {
        ModelDescriptor {
            id: self.id(),
            family: "sd3",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: Capabilities {
                // Large is classic true-CFG (negative prompt + guidance); Turbo is guidance-free.
                supports_negative_prompt: self.supports_true_cfg(),
                supports_guidance: self.supports_true_cfg(),
                supports_true_cfg: self.supports_true_cfg(),
                // E1 is the converter/config slice; image-conditioning modes (img2img / inpaint)
                // are later epic stories and are not advertised yet (plain txt2img only).
                conditioning: vec![],
                supported_quants: &[Quant::Q4, Quant::Q8],
                supports_lora: true,
                supports_lokr: true,
                samplers: {
                    let mut s = curated_sampler_names();
                    s.push(DEFAULT_SAMPLER);
                    s
                },
                schedulers: {
                    let mut s = curated_scheduler_names();
                    s.push("linear");
                    s
                },
                min_size: 256,
                max_size: 1440,
                max_count: 8,
                mac_only: true,
                supports_kv_cache: false,
                // SD3.5 uses a resolution-aware flow-match shift (handled by the unified sampler).
                requires_sigma_shift: true,
            },
        }
    }
}

/// The dimension-parametric SD3.5 MMDiT architecture descriptor. Drives the converter's
/// block-count enumeration and the architecture validator's expected-shape table. The default
/// ([`Sd3Arch::large`]) is the real 8.1B Large/Large-Turbo layout; a test can construct a tiny one
/// to exercise the converter against a synthetic fixture without the multi-GB weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sd3Arch {
    pub num_layers: usize,
    pub head_dim: usize,
    pub num_heads: usize,
    pub patch_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub joint_attention_dim: usize,
    pub pooled_projection_dim: usize,
    pub caption_projection_dim: usize,
    pub pos_embed_max_size: usize,
    pub time_proj_dim: usize,
    /// The number of LEADING MMDiT-X dual-attention (`attn2`) blocks — blocks `0..dual_attention_layers`
    /// carry a second, image-stream-only self-attention plus the extended (`9·hidden`) `norm1` AdaLN.
    /// `0` for plain MMDiT (Large / Large-Turbo); `13` for Medium (`dual_attention_layers = [0..=12]`).
    pub dual_attention_layers: usize,
}

impl Sd3Arch {
    /// The real SD3.5-Large / Large-Turbo MMDiT (sc-7850, real-weight confirmed). Plain MMDiT — no
    /// dual-attention blocks.
    pub fn large() -> Self {
        Self {
            num_layers: LARGE_NUM_LAYERS,
            head_dim: LARGE_HEAD_DIM,
            num_heads: LARGE_NUM_HEADS,
            patch_size: LARGE_PATCH_SIZE,
            in_channels: LARGE_IN_CHANNELS,
            out_channels: LARGE_OUT_CHANNELS,
            joint_attention_dim: LARGE_JOINT_ATTENTION_DIM,
            pooled_projection_dim: LARGE_POOLED_PROJECTION_DIM,
            caption_projection_dim: LARGE_CAPTION_PROJECTION_DIM,
            pos_embed_max_size: LARGE_POS_EMBED_MAX_SIZE,
            time_proj_dim: LARGE_TIME_PROJ_DIM,
            dual_attention_layers: 0,
        }
    }

    /// The real SD3.5-Medium MMDiT-X (sc-7867, real-weight confirmed): 24 blocks, hidden 1536,
    /// `pos_embed_max_size` 384, and the FIRST 13 blocks carrying `attn2` dual-attention.
    pub fn medium() -> Self {
        Self {
            num_layers: MEDIUM_NUM_LAYERS,
            head_dim: MEDIUM_HEAD_DIM,
            num_heads: MEDIUM_NUM_HEADS,
            patch_size: MEDIUM_PATCH_SIZE,
            in_channels: MEDIUM_IN_CHANNELS,
            out_channels: MEDIUM_OUT_CHANNELS,
            joint_attention_dim: MEDIUM_JOINT_ATTENTION_DIM,
            pooled_projection_dim: MEDIUM_POOLED_PROJECTION_DIM,
            caption_projection_dim: MEDIUM_CAPTION_PROJECTION_DIM,
            pos_embed_max_size: MEDIUM_POS_EMBED_MAX_SIZE,
            time_proj_dim: MEDIUM_TIME_PROJ_DIM,
            dual_attention_layers: MEDIUM_DUAL_ATTENTION_LAYERS,
        }
    }

    /// `inner_dim` / hidden size = `num_heads * head_dim`.
    pub fn hidden(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Whether block `i` is an MMDiT-X dual-attention block (carries `attn2` + the extended `norm1`).
    pub fn is_dual_attention_block(&self, i: usize) -> bool {
        i < self.dual_attention_layers
    }

    /// The flattened length of the learned positional table = `pos_embed_max_size^2`.
    pub fn pos_embed_len(&self) -> usize {
        self.pos_embed_max_size * self.pos_embed_max_size
    }

    /// `proj_out` / patchify output width = `patch_size^2 * out_channels`.
    pub fn patch_out_dim(&self) -> usize {
        self.patch_size * self.patch_size * self.out_channels
    }
}

impl Default for Sd3Arch {
    fn default() -> Self {
        Self::large()
    }
}
