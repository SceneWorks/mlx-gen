//! # mlx-gen-flux2
//!
//! The **FLUX.2-klein** provider crate for [`mlx-gen`](mlx_gen). FLUX.2 shares no crate-local
//! code with the FLUX.1 family (`mlx-gen-flux`): its text encoder is a **Qwen3** dense LLM
//! (not T5+CLIP), its VAE is the 32-channel `AutoencoderKL-Flux2` with 2×2 patchify +
//! BatchNorm-stats normalization (not the 16-ch BFL VAE), and its MMDiT uses fused parallel
//! single blocks with a 4-axis RoPE. Everything shared lives in the core `mlx-gen` crate
//! (nn primitives, the `FlowMatchEuler` scheduler, adapters, quant, the `Generator` contract,
//! the registry), so this is a clean sibling crate, decoupled from FLUX.1.
//!
//! Ported from the frozen Python mflux fork (`~/repos/mflux/src/mflux/models/flux2/`) and
//! parity-gated against it. Target: **FLUX.2-klein-9b** (txt2img + edit); the config is
//! dimension-parametric so the 4b variant is a near-free follow-on.
//!
//! Slice status: **S0** — scaffold, registry ids, dimension-parametric config, the flow-match
//! schedule (reused from core), 2×2 latent pack/unpack/patchify, the 4-axis RoPE table, and the
//! latent/text id builders. `load()`/`generate()` are guarded with a clear error until the
//! text-encoder (S1), VAE (S2), and transformer (S3) modules land.

pub mod adapters;
pub mod caption_upsample;
pub mod chunk;
pub mod config;
pub mod convert;
pub mod kv_cache;
pub mod loader;
pub mod model;
pub mod model_control;
pub mod pipeline;
pub mod pos_embed;
pub mod text_encoder;
pub mod transformer;
pub mod vae;
pub mod vision;

pub use adapters::apply_flux2_adapters;
pub use caption_upsample::{
    build_upsample_input_ids, expand_pixtral_image_tokens, upsample_prompt,
    SYSTEM_MESSAGE_UPSAMPLING_I2I, SYSTEM_MESSAGE_UPSAMPLING_T2I,
};
pub use chunk::{map_seq_chunks, MemoryConfig};
pub use config::{
    Flux2Config, Flux2Quant, Flux2Variant, DEFAULT_GUIDANCE, DEFAULT_GUIDANCE_DEV, DEFAULT_HEIGHT,
    DEFAULT_STEPS, DEFAULT_STEPS_DEV, DEFAULT_WIDTH, FLUX2_DEV_CONTROL_ID, FLUX2_DEV_EDIT_ID,
    FLUX2_DEV_ID, FLUX2_KLEIN_9B_EDIT_ID, FLUX2_KLEIN_9B_ID, FLUX2_KLEIN_9B_KV_EDIT_ID,
};
pub use convert::{
    build_target_state_dict, convert_and_assemble, quantize_flux2_dit, quantize_flux2_text_encoder,
    quantize_flux2_text_encoder_dir, quantize_flux2_transformer,
};
pub use kv_cache::{CacheMode, Flux2KvCache, Stream};
pub use loader::{
    load_control_transformer_dev, load_multimodal_projector_dev, load_text_encoder,
    load_text_encoder_dev, load_tokenizer, load_tokenizer_dev, load_transformer,
    load_transformer_dev, load_vae, load_vision_tower_dev,
};
pub use model::{
    descriptor_dev, descriptor_dev_edit, descriptor_klein_9b, descriptor_klein_9b_edit,
    descriptor_klein_9b_kv_edit, load_dev, load_dev_edit, load_klein_9b, load_klein_9b_edit,
    load_klein_9b_kv_edit,
};
pub use model_control::{descriptor_dev_control, load_dev_control, Flux2DevControl};
pub use pipeline::{
    add_noise_by_interpolation, create_noise, init_time_step, pack_latents, patchify_latents,
    prepare_grid_ids, prepare_text_ids, preprocess_ref_image, schedule, timesteps_x1000,
    unpack_latents,
};
pub use pos_embed::Flux2PosEmbed;
pub use text_encoder::{Qwen3TextEncoder, Qwen3TextEncoderConfig};
pub use transformer::{
    Flux2ControlBranch, Flux2ControlTransformer, Flux2ForwardInputs, Flux2Transformer,
    Flux2TransformerConfig, CONTROL_IN_DIM,
};
pub use vae::Flux2Vae;
pub use vision::{Mistral3Projector, PixtralVisionConfig, PixtralVisionTower};
