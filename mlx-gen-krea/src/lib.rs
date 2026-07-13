//! # mlx-gen-krea
//!
//! The **Krea 2** provider crate for [`mlx-gen`](mlx_gen). Krea 2 is Krea AI's first from-scratch
//! foundation image model (released 2026-06-22). Two surfaces share **one architecture**:
//! - **Krea 2 Turbo** (`krea_2_turbo`) — the user-facing text-to-image model (TDM-distilled few-step,
//!   CFG-free, 8 steps, up to 2048²),
//! - **Krea 2 Raw** (`krea_2_raw`) — the undistilled base. Both a **generation model** (full-CFG
//!   text-to-image: real guidance + a user negative prompt, 52 steps, resolution-dynamic mu — epic
//!   9992) AND the **LoRA-training base** (LoRAs train on Raw and apply at Turbo inference, the Lens /
//!   Z-Image precedent — epic 7565 P3). One id, both roles (generator + trainer registries).
//!
//! ## Architecture (verified against the real `krea/Krea-2-Turbo` configs + safetensors index)
//! - **DiT** — `Krea2Transformer2DModel`, a **dense single-stream** rectified-flow / v-param
//!   transformer: text + image tokens concatenated through 28 gated single-stream `transformer_blocks`
//!   (hidden 6144, GQA 48Q/12KV, head_dim 128, SwiGLU 16384, 3-axis RoPE `[32,48,48]`), a
//!   `DoubleSharedModulation` (one shared 6-factor `time_mod_proj` + per-block `scale_shift_table`),
//!   and a `text_fusion` (`TextFusionTransformer`) front-end that aggregates the 12 selected Qwen3-VL
//!   hidden layers (2 layerwise cross-layer-axis blocks → learned `projector` 12→1 → 2 token-axis
//!   refiner blocks).
//! - **Text encoder** — `Qwen3-VL-4B-Instruct` (`Qwen3VLModel`): the pipeline stacks the
//!   `text_encoder_select_layers` `[2,5,…,35]` hidden states and feeds them to the DiT's `text_fusion`.
//! - **VAE** — `AutoencoderKLQwenImage` (z_dim 16, per-channel `latents_mean`/`latents_std` de-norm) —
//!   direct reuse of `mlx-gen-qwen-image`'s `QwenVae`.
//! - **Scheduler** — `FlowMatchEulerDiscreteScheduler`, v-param, dynamic exponential time-shift; Turbo
//!   fixes mu 1.15 / 8 steps / CFG 0.
//!
//! ## Surfaces (all landed)
//! The core Turbo t2i vertical (provider scaffold, `krea_2_turbo` registration, architecture-validated
//! [`model::load`], offline Q4/Q8 [`convert`]; the single-stream DiT [`transformer`] reusing
//! `mlx-gen-boogu`'s 3-axis-RoPE blocks; the Qwen3-VL-4B [`text_encoder`]; the [`vae`] reusing
//! `mlx-gen-qwen-image`'s `QwenVae` over the core [`mlx_gen::FlowMatchSampler`] [`schedule`]) landed in
//! epic 7565 P1 (sc-7567…sc-7571). Since then the crate grew four more registered generators plus a
//! trainer and the residency split — all in this crate:
//! - **`krea_2_raw`** ([`raw_descriptor`] / [`load_raw`]) — the undistilled full-CFG base (epic 9992):
//!   real guidance + a user negative prompt, resolution-dynamic mu, 52 steps.
//! - **`krea_2_edit`** ([`edit_descriptor`] / [`load_edit`]) — Kontext-style image edit on the Raw path
//!   (epic 10871): dual conditioning (in-context VAE reference tokens + the Qwen3-VL grounded encode),
//!   one source `Reference` or a scene+person `MultiReference`.
//! - **`krea_2_turbo_edit`** ([`turbo_edit_descriptor`] / [`load_turbo_edit`]) — the same edit surface
//!   on the distilled CFG-free few-step schedule (sc-11640).
//! - **`krea_2_turbo_control`** ([`KreaTurboControl`], `model_control::load`) — pose-ControlNet on
//!   Turbo (epic 8459), a `control_scale`-scaled RMS-clamped residual branch ([`control`]).
//! - **Raw LoRA/LoKr trainer** ([`KreaRawTrainer`] / [`load_trainer`]) — LoRAs train on Raw and apply at
//!   Turbo inference (the Lens / Z-Image precedent, epic 7565 P3).
//! - **Component residency** (epic 10834 / sc-11101) — the [`KreaText`] + [`KreaHeavy`] phase split that
//!   bounds peak unified memory under `Sequential`; the img2img, PiD-decode (`mlx-gen-pid`) and
//!   `from_ldm` early-stop seams thread through it.

pub mod config;
pub mod control;
pub mod convert;
pub mod loader;
pub mod model;
pub mod model_control;
pub mod pipeline;
mod quant;
pub mod schedule;
pub mod text_encoder;
pub mod training;
pub mod transformer;
pub mod vae;

pub use config::Krea2Config;
pub use control::Krea2ControlBranch;
pub use loader::{load_text_encoder, load_transformer};
pub use model::{
    descriptor, edit_descriptor, load, load_edit, load_raw, load_turbo_edit, raw_descriptor,
    turbo_edit_descriptor, Krea, KREA_2_EDIT_ID, KREA_2_RAW_ID, KREA_2_TURBO_EDIT_ID,
    KREA_2_TURBO_ID,
};
pub use model_control::{KreaTurboControl, KREA_2_TURBO_CONTROL_ID};
pub use pipeline::{KreaHeavy, KreaPipeline, KreaText, TurboOptions};
pub use schedule::{krea_sigmas, turbo_sigmas, TURBO_MU, TURBO_STEPS};
pub use text_encoder::{KreaTeConfig, KreaTextEncoder, KreaTokenizer};
pub use training::{load_trainer, KreaRawTrainer, KREA_2_RAW_TRAINER_ID};
pub use transformer::Krea2Transformer;
pub use vae::{load_vae, QwenVae};

#[cfg(test)]
mod reexport_tests {
    //! F-077: the crate-root re-exports must cover the FULL registered surface — including the edit API
    //! (`load_edit`, `edit_descriptor`, `KREA_2_EDIT_ID`, and the Turbo-edit trio) which was previously
    //! reachable only via the `model::` path. Referencing each at the crate root pins the re-export.
    #[test]
    fn edit_surface_is_reexported_at_crate_root() {
        // Ids.
        assert_eq!(crate::KREA_2_EDIT_ID, "krea_2_edit");
        assert_eq!(crate::KREA_2_TURBO_EDIT_ID, "krea_2_turbo_edit");
        // Descriptors + loaders: referencing each function item at the crate root fails to compile if
        // the re-export is missing, and their ids must match.
        let _ = crate::load_edit;
        let _ = crate::load_turbo_edit;
        assert_eq!(crate::edit_descriptor().id, crate::KREA_2_EDIT_ID);
        assert_eq!(
            crate::turbo_edit_descriptor().id,
            crate::KREA_2_TURBO_EDIT_ID
        );
    }
}
