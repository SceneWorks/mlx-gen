//! # mlx-gen-sensenova
//!
//! The **SenseNova-U1** (NEO-Unify) provider crate for [`mlx-gen`](mlx_gen). NEO-Unify is a
//! *unified* multimodal model — one network does both understanding and image generation, with
//! **no separate VAE or text encoder** (unlike every diffusion-pipeline provider). The first-class
//! target is **`sensenova/SenseNova-U1-8B-MoT`**, which powers SceneWorks **Document Studio**
//! (interleaved text-image) plus Image / image-edit / character / VQA. See epic 3180.
//!
//! ## Architecture as it actually loads (validated against the 8B-MoT checkpoint, sc-3181)
//!
//! "MoT" is **Mixture of *Transformers***, not Mixture of Experts. For the 8B-MoT checkpoint the
//! backbone is the **dense** `Qwen3` (`NEOLLMConfig`, `modeling_qwen3.py`) — there are **no expert
//! stacks and no router** in the weights. Each of the 42 decoder layers carries **two parallel
//! dense transformer paths**:
//!
//! * the **understanding** path — `input_layernorm`, `self_attn.{q,k,v,o}_proj`, the QK-norms
//!   (`q_norm`/`k_norm`) and their **spatial** counterparts (`q_norm_hw`/`k_norm_hw`),
//!   `post_attention_layernorm`, and a SwiGLU `mlp`;
//! * the **generation** path — the same modules with a `_mot_gen` suffix.
//!
//! Tokens are dispatched between the paths per-token by the `image_gen_indicators` mask;
//! **attention K/V is shared/joint across the full sequence** (only Q/O, the norms, and the MLP
//! fork per path). RoPE layers three independent rotations over `head_dim`: temporal
//! (`rope_theta`) + height + width (`rope_theta_hw`). The "vision tower" is **not** a transformer
//! here — `vision_model` (and the generation-path `fm_modules.vision_model_mot_gen`) are just a
//! Conv patch-embedder + 2D-RoPE + Conv dense-embedder. The latent→pixel path is the `fm_head`
//! (flow-matching head) → unpatchify to RGB (`use_pixel_head=false`, so the conv pixel decoders in
//! the reference are dead code for this checkpoint).
//!
//! ## Status
//!
//! sc-3181 (this slice) ships the crate scaffold plus the [`config`] parser and the [`loader`]
//! weight-map foundation (the canonical key layout + a coverage check against the real shards). The
//! backbone, vision embedder, flow-matching head, AR runtime, the generation modes, and the
//! `Generator` impl + `inventory` registration land in the following stories (sc-3182 … sc-3194).

pub mod config;
pub mod loader;
pub mod qwen3;
pub mod vision;

pub use config::{NeoChatConfig, NeoLlmConfig, NeoVisionConfig};
pub use loader::{check_coverage, expected_keys, load_raw, Coverage};
pub use qwen3::{Path, Qwen3Backbone};
pub use vision::NeoVisionEmbedder;
