//! # mlx-gen-sana
//!
//! SANA (NVlabs) provider crate for [`mlx-gen`], epic 8485. **Spike sc-8486** delivers the DC-AE
//! deep-compression **decoder** (the one piece of the native-SANA port whose Metal feasibility was
//! unproven — the trunk is proven by the Clark Labs 2-bit MLX drop, and the Gemma-2 CHI text encoder
//! already ships in `mlx-gen-pid`). The full pipeline (Linear DiT trunk, flow scheduler, e2e wiring)
//! lands in sibling stories sc-8487..8490.
//!
//! Port target: diffusers `AutoencoderDC` for `mit-han-lab/dc-ae-f32c32-sana-1.0` (the autoencoder
//! behind SANA-1.6B 1024px). See [`dc_ae`] for the faithful block-by-block port.

pub mod config;
pub mod dc_ae;

pub use config::{BlockType, DcAeConfig};
pub use dc_ae::DcAeDecoder;
