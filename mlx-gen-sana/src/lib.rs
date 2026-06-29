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
//!
//! **sc-8487** adds the [`transformer`] module: the SANA **Linear Diffusion Transformer trunk**
//! (`SanaTransformer2DModel`) — ReLU linear self-attention (reusing the spike's linear-attn kernel),
//! standard caption cross-attention, GLUMBConv Mix-FFN (3×3 depthwise conv), adaLN-single timestep
//! modulation, and NoPE. Its `[B, 32, H, W]` output (f32c32 latent channels) feeds [`dc_ae`]'s
//! `DcAeDecoder::decode` directly (sc-8489 composition). Written for the bf16/fp16 weight path; the
//! 2-bit Clark Labs quant is intentionally NOT ported.

//! **sc-8488** adds the [`text_encoder`] module: SANA's text conditioning, which **reuses** PiD's
//! already-native gemma-2-2b-it CHI caption encoder ([`mlx_gen_pid::CaptionEncoder`]) rather than
//! duplicating it. SANA and PiD share the exact Gemma-2 last-hidden CHI text-encoder lineage; they
//! differ only in the CHI prompt text (quoting around `Enhanced prompt`), which is parameterized.
//! [`text_encoder::SanaTextEncoder::encode`] produces the `[1, 300, 2304]` embedding the
//! [`transformer::SanaTransformer`] trunk's `attn2` cross-attention consumes.

//! **sc-8489 (Phase A — the mlx-gen side)** adds the [`pipeline`] module: the end-to-end native SANA
//! text-to-image pipeline composing [`SanaTextEncoder`] → [`SanaTransformer`] (flow-match Euler
//! denoise via the unified epic-7114 sampler, true CFG) → [`DcAeDecoder`] into a clean
//! [`pipeline::SanaPipeline::generate`] entrypoint. Phase B (the SceneWorks worker `Generator`
//! adapter/registration) is a separate follow-up PR.

pub mod config;
pub mod dc_ae;
pub mod model;
pub mod pipeline;
pub mod text_encoder;
pub mod transformer;

pub use config::{BlockType, DcAeConfig, SanaTransformerConfig};
pub use dc_ae::DcAeDecoder;
pub use model::{descriptor as sana_descriptor, load as load_sana, Sana, MODEL_ID};
pub use pipeline::{SanaGenerateRequest, SanaPipeline};
pub use text_encoder::{
    Gemma2, Gemma2Config, SanaTextEncoder, MAX_SEQUENCE_LENGTH, SANA_CHI_PROMPT,
};
pub use transformer::SanaTransformer;
