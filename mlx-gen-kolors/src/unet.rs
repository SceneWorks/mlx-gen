//! Kolors U-Net (sc-3093) — the SDXL `UNet2DConditionModel` reused as-is, plus the ChatGLM3 context
//! projection. Kolors' U-Net is structurally identical to SDXL base (same blocks/channels/heads,
//! `cross_attention_dim` 2048) with two ChatGLM-driven deltas, both handled in `mlx-gen-sdxl`:
//!
//!  - an **`encoder_hid_proj`** Linear (4096→2048) that projects the ChatGLM3 context to the
//!    cross-attention width — auto-detected from the checkpoint by `UNet2DConditionModel`
//!    (`UNetConfig::kolors`);
//!  - the **5632**-wide `add_embedding.linear_1` (pooled 4096 + 6·256 time-ids), loaded by shape.
//!
//! So the wiring is: feed the encoder the ChatGLM3 **context** (`[B, S, 4096]`) and **pooled**
//! (`[B, 4096]`) + SDXL-style `time_ids` (`[B, 6]`) straight into the U-Net `forward` — the
//! projection to 2048 and the 5632 added-conditioning happen inside. Re-exported here so the Kolors
//! provider has one entry point; the T2I denoise loop + scheduler are sc-3094.

pub use mlx_gen_sdxl::{load_unet_kolors_dtype, UNet2DConditionModel};
