//! zai-org **SCAIL-2** — native MLX provider (epic 5439, sc-5442).
//!
//! SCAIL-2 is an end-to-end controlled **character-animation / motion-transfer** model: a reference
//! image + driving video (+ color-coded segmentation masks) → an animated or identity-replaced video.
//! The backbone is **Wan2.1-14B I2V** (dense), so it reuses the [`mlx_gen_wan`] foundation (DiT blocks,
//! z16 VAE, UMT5, 3-axis RoPE, UniPC/flow schedulers) with three SCAIL-2-specific deltas:
//!
//!   1. **packed-token conditioning** — reference + driving (pose) + 28-channel color-coded masks are
//!      patch-embedded (three Conv3d stems; the mask/pose embeds are *added* to the latent embeds) and
//!      concatenated with the noisy target on the token axis (Bernini-family packed conditioning, not
//!      VACE). Only the target tokens are kept from the prediction.
//!   2. **per-source RoPE shifts** — the base 3-axis Wan RoPE with integer (T,H,W) position shifts per
//!      chunk; `replace_flag` flips the reference H-shift (animation vs. cross-identity replacement),
//!      and the pose chunk is spatially frequency-downsampled.
//!   3. **CLIP image cross-attention** — the reference image is encoded by an open-CLIP XLM-RoBERTa
//!      ViT-H/14 visual tower and injected via Wan-I2V image cross-attention (`k_img`/`v_img`).
//!
//! Weights: the turnkey `SceneWorks/scail2-mlx` snapshot (converted bf16 DiT + stock Wan2.1 VAE / UMT5
//! / CLIP). Plain single-scale CFG; macOS-only.
//!
//! Status (sc-5443): the registration + capability surface, the [`model::Scail2Dit`] DiT forward, the
//! per-chunk [`rope::ScailRope`], the CLIP/VAE/mask preprocessing, and the live [`generate`] denoise
//! loop all land here (each parity-gated against upstream on tiny seeded fixtures). Real-weight 40-layer
//! + end-to-end parity is sc-5446; Q4/Q8 load-time quant is sc-5445.

pub mod clip;
pub mod config;
pub mod convert;
pub mod generate;
pub mod lora;
pub mod model;
pub mod pipeline;
pub mod preprocess;
pub mod resize;
pub mod rope;

pub use clip::{ClipVisionConfig, ScailClip};
pub use config::Scail2Config;
pub use convert::{quantize_scail2_dit, quantize_scail2_transformer};
pub use generate::{generate, CharacterRef, Scail2Job};
pub use lora::{has_diff_patch_keys, merge_diff_patch_adapters, DiffPatchReport};
pub use model::{Scail2Dit, Scail2Inputs};
pub use preprocess::extract_and_compress_mask_to_latent;
pub use resize::{clip_preprocess, downsample_half, interpolate, Interp};
pub use rope::ScailRope;
