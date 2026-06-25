//! # mlx-gen-sd3
//!
//! The **Stable Diffusion 3.5** provider crate for [`mlx-gen`](mlx_gen). SD3.5 is an MMDiT
//! (multimodal diffusion transformer): a joint image+text double-stream transformer with a learned
//! 2D positional embedding (NO RoPE), qk-RMSNorm, adaLN modulation from pooled-text + timestep, a
//! triple text encoder (CLIP-L + CLIP-G + T5-XXL), and a 16-channel VAE. The arch was empirically
//! confirmed on the real `stabilityai/stable-diffusion-3.5-large` / `-large-turbo` weights during
//! the spike (sc-7850).
//!
//! ## Slice status: **E1** (sc-7860) — converter + config + architecture validation
//!
//! This crate currently ships the foundational E1 slice ONLY:
//!
//! * [`config`] — the dimension-parametric SD3.5-Large / Large-Turbo MMDiT arch constants and the
//!   registry descriptors.
//! * [`convert`] — the diffusers `SD3Transformer2DModel` → MLX weight converter (a 1:1 rename over
//!   the validated key set, plus offline Q4/Q8 pre-quantization) and the **architecture
//!   validation** (an exhaustive, shape-checked expected-tensor table asserted against a converted
//!   or on-disk tensor set).
//!
//! The triple-TE aggregator (**E2**), the MMDiT forward pass (**E3**), the 16-channel VAE (**E4**),
//! the model/loader/pipeline wiring (**E5+**), and native LoRA training (**T1–T4**) are separate
//! epic stories and are intentionally NOT implemented here. No `Generator` is registered yet.

pub mod config;
pub mod convert;

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
