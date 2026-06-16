//! Ideogram 4.0 configuration — constants read directly from the official
//! `ideogram-ai/ideogram-4-fp8` checkpoint configs (sc-5984), so the Rust modules and the
//! offline converter (`tools/convert_ideogram4_to_mlx.py`) agree on every dimension.
//!
//! Pipeline (`model_index.json` = `Ideogram4Pipeline`):
//!   * `FlowMatchEulerDiscreteScheduler`
//!   * `Qwen3VLModel` text encoder (+ `Qwen2Tokenizer`)
//!   * `Ideogram4Transformer2DModel` **transformer** + **unconditional_transformer**
//!     (two full DiTs — asymmetric CFG)
//!   * `AutoencoderKLFlux2` VAE (the FLUX.2 VAE → reuse `mlx-gen-flux2`)

pub const IDEOGRAM_4_ID: &str = "ideogram_4";

/// HF repo for the gated source weights (fp8 reference release).
pub const IDEOGRAM_4_FP8_REPO: &str = "ideogram-ai/ideogram-4-fp8";

// ── Defaults / limits ────────────────────────────────────────────────────────────────────
pub const DEFAULT_WIDTH: u32 = 1024;
pub const DEFAULT_HEIGHT: u32 = 1024;
/// Native resolution range: 256–2048, multiples of 16, aspect up to 6:1.
pub const RES_MIN: u32 = 256;
pub const RES_MAX: u32 = 2048;
pub const RES_MULTIPLE: u32 = 16;
/// Euler flow-matching with asymmetric CFG. The reference `__call__` default is **128** steps;
/// the SceneWorks default is the `V4_QUALITY_48` quality preset (48 steps), a sanctioned
/// preset that renders cleanly at a fraction of the cost of 128 over two DiTs (validated:
/// 50 steps @256² is a clean image, ~8 undercooks badly). (sc-5988)
pub const DEFAULT_STEPS: u32 = 48;
/// Reference `__call__` default `guidance_scale=7.0` (asymmetric CFG: `v = g·cond + (1−g)·uncond`).
pub const DEFAULT_GUIDANCE: f32 = 7.0;
/// Reference `__call__` default `mu=0.5` — the logit-normal schedule's `known_mean`
/// (`LogitNormalSchedule::for_resolution(h, w, mu, std)`, `std=1.0`).
pub const DEFAULT_MU: f64 = 0.5;
/// Max text tokens the model accepts (Qwen3-VL context budget used by Ideogram).
pub const MAX_TEXT_TOKENS: usize = 2048;

/// `Ideogram4Transformer2DModel` dims (transformer/config.json). Single-stream, 34 layers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ideogram4DitConfig {
    pub emb_dim: i32,            // 4608 = num_heads * head_dim
    pub num_layers: i32,         // 34
    pub num_heads: i32,          // 18
    pub head_dim: i32,           // 256
    pub mlp_dim: i32,            // 12288 (SwiGLU intermediate)
    pub adaln_dim: i32,          // 512
    pub in_channels: i32,        // 128 (32-ch VAE latent, 2x2 patchified)
    pub llm_features_dim: i32,   // 53248 = 13 * 4096 (concatenated TE layers)
    pub mrope_section: [i32; 3], // [24, 20, 20]
    pub rope_theta: f32,         // 5_000_000
    pub norm_eps: f32,           // 1e-5
}

impl Ideogram4DitConfig {
    pub const fn v4() -> Self {
        Self {
            emb_dim: 4608,
            num_layers: 34,
            num_heads: 18,
            head_dim: 256,
            mlp_dim: 12288,
            adaln_dim: 512,
            in_channels: 128,
            llm_features_dim: 53248,
            mrope_section: [24, 20, 20],
            rope_theta: 5_000_000.0,
            norm_eps: 1e-5,
        }
    }
}

/// `Qwen3VLModel` text stack (text_encoder/config.json `text_config`). Text path only — the
/// vision tower is unused for text-to-image. Ideogram concatenates the hidden states from
/// [`EXTRACTED_LAYERS`] (13 of them) → `13 * 4096 = 53248` features fed to the DiT.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ideogram4TextEncoderConfig {
    pub hidden_size: i32,        // 4096
    pub num_layers: i32,         // 36
    pub num_heads: i32,          // 32
    pub num_kv_heads: i32,       // 8
    pub head_dim: i32,           // 128
    pub intermediate_size: i32,  // 12288
    pub rms_norm_eps: f32,       // 1e-6
    pub mrope_section: [i32; 3], // [24, 20, 20]
    pub rope_theta: f32,         // 5_000_000
    pub vocab_size: i32,         // 151936
}

impl Ideogram4TextEncoderConfig {
    pub const fn qwen3_vl_8b() -> Self {
        Self {
            hidden_size: 4096,
            num_layers: 36,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 12288,
            rms_norm_eps: 1e-6,
            mrope_section: [24, 20, 20],
            rope_theta: 5_000_000.0,
            vocab_size: 151936,
        }
    }
}

/// The 13 Qwen3-VL hidden-state layers Ideogram concatenates: `(0, 3, 6, …, 33, 35)`.
/// `len * hidden_size = 13 * 4096 = 53248 = Ideogram4DitConfig.llm_features_dim`.
pub const EXTRACTED_LAYERS: [usize; 13] = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35];

const _: () = assert!(
    EXTRACTED_LAYERS.len() as i32 * Ideogram4TextEncoderConfig::qwen3_vl_8b().hidden_size
        == Ideogram4DitConfig::v4().llm_features_dim
);
