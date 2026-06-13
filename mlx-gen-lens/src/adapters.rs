//! Lens DiT adapter consumption (sc-3174). The model-specific piece is the key→module map (the
//! `AdaptableHost for LensTransformer` + block / joint-attention hosts in `dit/`, the Rust analog of
//! the Lens trainer's `DEFAULT_LORA_TARGET_MODULES`); per-file LoKr/LoRA dispatch, peft/kohya prefix
//! detection, stacking + mixed, and the strict no-silent-drop policy are the shared core seam
//! (sc-2534), exactly as Qwen (sc-2528) / FLUX.2 (sc-2646) / Z-Image use it.
//!
//! LoRA/LoKr are **DiT-only** for Lens — the gpt-oss text encoder and Flux.2 VAE are not adapter
//! targets, matching the trainer (`lens_train_runner` targets `img_qkv` / `txt_qkv` / `to_out.0` /
//! `to_add_out` on `LensTransformer2DModel` only). The same `LensTransformer` serves both `lens` and
//! `lens_turbo` (identical architecture), so a LoRA trained on base `microsoft/Lens` applies cleanly
//! to `Lens-Turbo`.
//!
//! **Fused QKV.** `img_qkv` / `txt_qkv` are single fused `[3·inner, in]` projections (the trainer
//! targets them as one module each), so a LoRA/LoKr on them merges into the whole fused weight — no
//! q/k/v split, unlike the BFL fused→split path FLUX.2 needs (sc-2743).

use mlx_gen::adapters::loader::{apply_adapters_strict, ApplyReport};
use mlx_gen::adapters::AdaptableHost;
use mlx_gen::runtime::AdapterSpec;
use mlx_gen::Result;

/// Apply every adapter in `specs` onto a Lens transformer `host` (stacked, mixed LoRA/LoKr), via the
/// core [`apply_adapters_strict`] — errors, never silently drops, on an unmatched target. The core
/// `Adapter::residual` runs in the host's natural dtype.
pub fn apply_lens_adapters(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
) -> Result<ApplyReport> {
    apply_adapters_strict(host, specs, "lens")
}
