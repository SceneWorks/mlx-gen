//! # mlx-gen-sd3
//!
//! The **Stable Diffusion 3.5** provider crate for [`mlx-gen`](mlx_gen). SD3.5 is an MMDiT
//! (multimodal diffusion transformer): a joint image+text double-stream transformer with a learned
//! 2D positional embedding (NO RoPE), qk-RMSNorm, adaLN modulation from pooled-text + timestep, a
//! triple text encoder (CLIP-L + CLIP-G + T5-XXL), and a 16-channel VAE. The arch was empirically
//! confirmed on the real `stabilityai/stable-diffusion-3.5-large` / `-large-turbo` weights during
//! the spike (sc-7850).
//!
//! ## Slice status: **E1** (sc-7860) + **E2** (sc-7861) + **E3** (sc-7862) + **E4** (sc-7863)
//! + **M1** (sc-7867) — converter + config + triple-TE + MMDiT-Large forward + VAE + Medium converter
//!
//! This crate currently ships:
//!
//! * [`config`] — the dimension-parametric SD3.5 MMDiT arch constants ([`config::Sd3Arch::large`]
//!   for Large / Large-Turbo and [`config::Sd3Arch::medium`] for the Medium **MMDiT-X**) and the
//!   registry descriptors.
//! * [`convert`] — the diffusers `SD3Transformer2DModel` → MLX weight converter (a 1:1 rename over
//!   the validated key set, plus offline Q4/Q8 pre-quantization) and the **architecture
//!   validation** (an exhaustive, shape-checked expected-tensor table asserted against a converted
//!   or on-disk tensor set). The same converter/validator serve Medium's MMDiT-X layout — the first
//!   13 blocks' `attn2` dual-attention tensors + the extended 9-chunk `norm1` AdaLN — driven by
//!   [`config::Sd3Arch::medium`] (real-weight confirmed, 909 transformer tensors).
//! * [`text`] — the **triple text-encoder aggregator** (E2). REUSES the existing SDXL CLIP encoder
//!   (CLIP-L + CLIP-G / OpenCLIP-bigG) and the FLUX T5-XXL encoder unchanged, and combines their
//!   outputs into SD3.5 conditioning — `pooled` `[B, 2048]` and `context` `[B, 333, 4096]` — exactly
//!   as diffusers `StableDiffusion3Pipeline.encode_prompt` does (penultimate CLIP hidden states for
//!   the context, projected pooled for the pooled vector, trailing zero-pad 2048→4096, CLIP-then-T5
//!   sequence concat).
//! * [`vae`] (E4) — the SD3.5 **16-channel VAE**: it REUSES the Z-Image 16-ch `AutoencoderKL`
//!   (structurally identical to SD3.5's per `vae/config.json`) with SD3.5's own `1.5305` / `0.0609`
//!   latent factors, plus the diffusers→MLX VAE converter + the shape-checked VAE arch validator.
//!   encode/decode apply the correct (diffusers-SD3-verified) scale/shift de-norm direction.
//!
//! * [`transformer`] (E3) — the SD3.5-Large **MMDiT forward pass** (`SD3Transformer2DModel`): patch
//!   embed + learned 2D pos_embed (NO RoPE), `(timestep + pooled-text)` adaLN modulation, 38
//!   all-double-stream joint blocks (qk-RMSNorm both streams, GELU FFN, `context_pre_only` final
//!   block), and the AdaLN-continuous output head → unpatchify. REUSES flux2's joint-attention
//!   `process_qkv`/double-stream pattern with the SD3 deltas (no RoPE, all-double topology, GELU).
//!
//! * [`loader`] / [`pipeline`] / [`model`] (E5 sc-7864 = Large; E6 sc-7865 = Large-Turbo) — the
//!   SD3.5 text-to-image vertical: the snapshot-layout loader (reusing the SDXL CLIP encoder ×2 +
//!   FLUX T5 + Z-Image VAE), the flow-match-Euler (static shift 3.0) sampling pipeline, and the
//!   [`model::Sd3Large`] [`Generator`](mlx_gen::Generator) registered under BOTH engine ids:
//!   **`sd3_5_large`** (true-CFG, 28 steps / guidance 3.5) and **`sd3_5_large_turbo`** (ADD-distilled,
//!   4 steps / guidance-baked CFG-off — same backbone + snapshot layout, distilled checkpoint, one
//!   forward per step). The pipeline's `denoise_cfg` skips the uncond forward when guidance == 1.0, so
//!   the Turbo path reuses E5's pipeline unchanged.
//!
//! Medium (M3) and native LoRA training (T1–T4) are separate epic stories and are NOT implemented
//! here.

pub mod config;
pub mod convert;
pub mod loader;
pub mod model;
pub mod pipeline;
pub mod text;
pub mod transformer;
pub mod vae;

pub use model::{Sd3Large, MODEL_ID, TURBO_MODEL_ID};

pub use config::{
    Sd3Arch, Sd3Variant, DEFAULT_GUIDANCE_LARGE, DEFAULT_GUIDANCE_MEDIUM, DEFAULT_GUIDANCE_TURBO,
    DEFAULT_HEIGHT, DEFAULT_SAMPLER, DEFAULT_STEPS_LARGE, DEFAULT_STEPS_MEDIUM,
    DEFAULT_STEPS_TURBO, DEFAULT_WIDTH, LARGE_CAPTION_PROJECTION_DIM, LARGE_HEAD_DIM, LARGE_HIDDEN,
    LARGE_IN_CHANNELS, LARGE_JOINT_ATTENTION_DIM, LARGE_NUM_HEADS, LARGE_NUM_LAYERS,
    LARGE_OUT_CHANNELS, LARGE_PATCH_SIZE, LARGE_POOLED_PROJECTION_DIM, LARGE_POS_EMBED_LEN,
    LARGE_POS_EMBED_MAX_SIZE, LARGE_TIME_PROJ_DIM, MEDIUM_CAPTION_PROJECTION_DIM,
    MEDIUM_DUAL_ATTENTION_LAYERS, MEDIUM_HEAD_DIM, MEDIUM_HIDDEN, MEDIUM_IN_CHANNELS,
    MEDIUM_JOINT_ATTENTION_DIM, MEDIUM_NUM_HEADS, MEDIUM_NUM_LAYERS, MEDIUM_OUT_CHANNELS,
    MEDIUM_PATCH_SIZE, MEDIUM_POOLED_PROJECTION_DIM, MEDIUM_POS_EMBED_LEN,
    MEDIUM_POS_EMBED_MAX_SIZE, MEDIUM_TIME_PROJ_DIM, RMS_EPS, SD3_5_LARGE_ID, SD3_5_LARGE_TURBO_ID,
    SD3_5_MEDIUM_ID,
};
pub use convert::{
    build_target_state_dict, expected_tensor_count, expected_transformer_tensors, quantize_sd3_dir,
    quantize_sd3_transformer, safetensors_header_shapes, validate_arch, validate_transformer_dir,
    ExpectedTensor,
};
pub use text::{
    build_sd3_conditioning, sd3_clip_g_config, sd3_clip_l_config, Sd3Conditioning, Sd3TextEncoders,
    CLIP_CONTEXT_DIM, CLIP_G_DIM, CLIP_L_DIM, CLIP_SEQ_LEN, CONTEXT_SEQ_LEN, JOINT_ATTENTION_DIM,
    POOLED_DIM, T5_SEQ_LEN,
};
pub use transformer::Sd3Transformer;
pub use vae::{
    build_vae_state_dict, expected_vae_tensor_count, expected_vae_tensors, load_sd3_vae,
    validate_vae_arch, validate_vae_dir, ExpectedVaeTensor, Sd3VaeArch, SD3_VAE_LATENT_CHANNELS,
    SD3_VAE_SCALE_FACTOR, SD3_VAE_SCALING_FACTOR, SD3_VAE_SHIFT_FACTOR,
};
