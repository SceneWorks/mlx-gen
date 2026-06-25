//! # mlx-gen-sd3
//!
//! The **Stable Diffusion 3.5** provider crate for [`mlx-gen`](mlx_gen). SD3.5 is an MMDiT
//! (multimodal diffusion transformer): a joint image+text double-stream transformer with a learned
//! 2D positional embedding (NO RoPE), qk-RMSNorm, adaLN modulation from pooled-text + timestep, a
//! triple text encoder (CLIP-L + CLIP-G + T5-XXL), and a 16-channel VAE. The arch was empirically
//! confirmed on the real `stabilityai/stable-diffusion-3.5-large` / `-large-turbo` weights during
//! the spike (sc-7850).
//!
//! ## Slice status: **E1** (sc-7860) + **E2** (sc-7861)
//!
//! This crate currently ships:
//!
//! * [`config`] — the dimension-parametric SD3.5-Large / Large-Turbo MMDiT arch constants and the
//!   registry descriptors.
//! * [`convert`] — the diffusers `SD3Transformer2DModel` → MLX weight converter (a 1:1 rename over
//!   the validated key set, plus offline Q4/Q8 pre-quantization) and the **architecture
//!   validation** (an exhaustive, shape-checked expected-tensor table asserted against a converted
//!   or on-disk tensor set).
//! * [`text`] — the **triple text-encoder aggregator** (E2). REUSES the existing SDXL CLIP encoder
//!   (CLIP-L + CLIP-G / OpenCLIP-bigG) and the FLUX T5-XXL encoder unchanged, and combines their
//!   outputs into SD3.5 conditioning — `pooled` `[B, 2048]` and `context` `[B, 333, 4096]` — exactly
//!   as diffusers `StableDiffusion3Pipeline.encode_prompt` does (penultimate CLIP hidden states for
//!   the context, projected pooled for the pooled vector, trailing zero-pad 2048→4096, CLIP-then-T5
//!   sequence concat).
//!
//! The MMDiT forward pass (**E3**), the 16-channel VAE (**E4**), the model/loader/pipeline wiring
//! (**E5+**), and native LoRA training (**T1–T4**) are separate epic stories and are intentionally
//! NOT implemented here. No `Generator` is registered yet.

pub mod config;
pub mod convert;
pub mod text;

pub use config::{
    Sd3Arch, Sd3Variant, DEFAULT_GUIDANCE_LARGE, DEFAULT_GUIDANCE_TURBO, DEFAULT_HEIGHT,
    DEFAULT_SAMPLER, DEFAULT_STEPS_LARGE, DEFAULT_STEPS_TURBO, DEFAULT_WIDTH,
    LARGE_CAPTION_PROJECTION_DIM, LARGE_HEAD_DIM, LARGE_HIDDEN, LARGE_IN_CHANNELS,
    LARGE_JOINT_ATTENTION_DIM, LARGE_NUM_HEADS, LARGE_NUM_LAYERS, LARGE_OUT_CHANNELS,
    LARGE_PATCH_SIZE, LARGE_POOLED_PROJECTION_DIM, LARGE_POS_EMBED_LEN, LARGE_POS_EMBED_MAX_SIZE,
    LARGE_TIME_PROJ_DIM, RMS_EPS, SD3_5_LARGE_ID, SD3_5_LARGE_TURBO_ID,
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
