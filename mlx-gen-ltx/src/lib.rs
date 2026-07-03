//! # mlx-gen-ltx
//!
//! LTX-2.3 **video** (text-to-video) provider crate for [`mlx-gen`]. Port of the
//! `mlx-video-with-audio` package's LTX video path (`generate_av.py`, `models/ltx/*`,
//! `models/ltx/video_vae/*`) onto Rust + `mlx-rs`.
//!
//! **Scope:** the full **AudioVideo** path (`generate_av.py`, sc-2684) — synchronized audio+video;
//! `generate()` runs the joint dual-modality denoise and returns video frames + an audio track. Built
//! on the sc-2679 video core + **single-image I2V** (sc-2685) + checkpoint-driven **Q4/Q8** quant
//! (sc-2686) + **LoRA in generate** (sc-2687 — forward-time residuals + per-pass strength over the
//! full video+audio+cross-modal surface; see [`adapters`]). LoKr (sc-2393) is a sibling story.
//!
//! This crate self-registers `ltx_2_3` into the `mlx-gen` model registry; load it with
//! `mlx_gen::load("ltx_2_3", spec)`.
//!
//! ## Status (S0–S6 complete)
//! The full text-to-video path is wired and pixel-parity vs the reference `generate_av.py`: Gemma-3
//! tokenizer (byte-exact) → [`LtxTextEncoder`] (Gemma backbone + connector) → seeded noise → the
//! 2-stage distilled denoise ([`pipeline`]: stage-1 half-res → 2× [`upsampler::LatentUpsampler`] →
//! re-noise → stage-2 full-res over the 48-layer [`transformer::LtxDiT`]) → [`vae::LtxVideoVae`]
//! decode → uint8 frames. Built on SPLIT 3-D RoPE (double-precision), an f32 position grid, the
//! distilled sigma schedules, and the legacy dtype-preserving Euler step.
//!
//! The distilled stage-1 sampler is chaos-sensitive, so e2e pixel-parity requires a **bit-exact
//! per-forward DiT** (sc-2842 — the adaLN timestep table must be built in MLX f32, not host f64). Two
//! shipped precisions, both gated bit-exact vs their reference golden: [`transformer::Precision::quant_f32`]
//! (f32 activations × quantized weights — the quality target) and [`transformer::Precision::quant_bf16`]
//! (the reference's native bf16 activations — the production-speed path). The quant geometry (**Q4**/Q8)
//! rides on the checkpoint's `split_model.json` (sc-2686). **I2V** single-image conditioning (sc-2685)
//! is wired into the same 2-stage path ([`conditioning`] + [`pipeline::generate_i2v_latents`], gated
//! bit-exact by `tests/i2v_parity.rs`). **LoRA in generate** (sc-2687) is wired (forward-time residuals
//! + per-pass strength; see [`adapters`]). LoKr (sc-2393) is a sibling.

pub mod adapters;
pub mod audio_vae;
pub mod conditioning;
pub mod config;
pub mod connector;
pub mod convert;
pub mod enhance;
pub mod gemma;
pub mod model;
pub mod pipeline;
pub mod positions;
pub mod rope;
pub mod text_encoder;
pub mod tokenizer;
pub mod training;
pub mod transformer;
pub mod upsampler;
pub mod vae;
pub mod vocoder;

pub use adapters::{apply_ltx_adapters, LtxLoraReport};
pub use audio_vae::AudioDecoder;
pub use conditioning::{
    append_keyframe_clip, apply_conditioning, apply_denoise_mask, apply_keyframes,
    keyframe_append_positions, patchify_grid, unpatchify_grid, I2vConditioning, Keyframe,
    VideoTokenState,
};
pub use config::{AudioVaeConfig, LtxConfig, LtxVaeConfig, RopeType, VaeBlock};
pub use connector::Connector;
pub use convert::{convert_and_assemble, LtxConvertOpts};
pub use enhance::{clean_response, EnhanceConfig, SampleParams};
pub use model::{apply_replacement_mask, descriptor, load, Ltx, MODEL_ID};
pub use pipeline::{
    decode_audio_track, decode_to_frames, denoise, denoise_av, denoise_av_tokens,
    generate_av_latents, generate_av_latents_iclora, generate_i2v_latents, generate_t2v,
    generate_t2v_latents, preprocess_conditioning_image, renoise, to_uint8_frames, StageClip,
    StageKeyframe, STAGE1_SIGMAS, STAGE2_SIGMAS,
};
pub use text_encoder::LtxTextEncoder;
// Tiling moved to `mlx_gen` core (shared with the Wan VAE — sc-2808). Re-export the module + config
// so `mlx_gen_ltx::tiling::*` / `mlx_gen_ltx::TilingConfig` keep resolving for existing callers.
pub use config::{VocoderConfig, VocoderGenConfig};
pub use mlx_gen::tiling::{self, TilingConfig};
pub use tokenizer::LtxTokenizer;
pub use training::{load_trainer, LtxTrainer};
pub use transformer::{to_denoised, AvDiT, LtxDiT, Precision, VideoBlock};
pub use upsampler::{upsample_latents, LatentUpsampler};
pub use vae::LtxVideoVae;
// Re-exported as `VocoderGenerator`, not `Generator`: this is the HiFi-GAN vocoder struct, and a
// bare `Generator` at the crate root would shadow the core `mlx_gen::Generator` *trait* name — an
// accidental `use mlx_gen_ltx::Generator` would then compile and mean the wrong thing (F-059).
pub use vocoder::{Generator as VocoderGenerator, LtxVocoder, VocoderWithBwe};

// sc-2963 (rollout of the Wan sc-2957 template): when on, the AvDiT's fusable elementwise *glue* —
// adaLN affine (`x·(1+scale)+shift`), the gated residuals (`x+out·gate`), the **tanh-GELU FFN
// activation**, and the split (rotate-halves) RoPE rotation — runs through `mx.compile` so MLX fuses
// each chain into a single Metal kernel. The big quantized GEMMs / SDPA / `mx.fast` norms stay eager.
//
// This is the same machine that gave Wan **+14%/step**. **Bit-exact** to the eager form. **Enabled by
// the production denoise loops** ([`pipeline::denoise`] / [`pipeline::denoise_av`]); **off by default**
// so the reference-parity gates run eager. Dtype-preserving — f32 / bf16 / quantized paths flow through
// unchanged.
//
// The toggle + its RAII [`CompileGlueGuard`] are hoisted into core (F-104); re-export core's so the
// process-global is shared with the FLUX family rather than each crate hand-rolling its own `AtomicBool`.
pub(crate) use mlx_gen::nn::compile_glue;
pub use mlx_gen::nn::{set_compile_glue, CompileGlueGuard};

#[cfg(test)]
mod compile_glue_guard_tests {
    use super::{compile_glue, set_compile_glue, CompileGlueGuard};

    #[test]
    fn guard_enables_then_restores_prior_value() {
        // Prior off → on within scope → restored off on drop (the doc's "eager by default" intent).
        set_compile_glue(false);
        {
            let _g = CompileGlueGuard::enable();
            assert!(compile_glue(), "guard enables compiled glue for its scope");
        }
        assert!(!compile_glue(), "guard restores the prior (off) on drop");

        // Restores the *prior* value, not a hardcoded false: prior on stays on after drop.
        set_compile_glue(true);
        {
            let _g = CompileGlueGuard::enable();
            assert!(compile_glue());
        }
        assert!(compile_glue(), "guard restores the prior (on) on drop");

        // Leave the global eager, as the reference-parity gates expect.
        set_compile_glue(false);
    }
}
