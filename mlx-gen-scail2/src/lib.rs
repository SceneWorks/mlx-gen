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
//! Status (sc-5442): the registration + capability surface + config/snapshot resolution land here; the
//! DiT forward / preprocessing / denoise loop are in progress.

pub mod config;
pub mod pipeline;
