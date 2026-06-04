//! FLUX.2-klein adapter consumption (sc-2646). The model-specific piece is the key→module map (the
//! `AdaptableHost for Flux2Transformer` + block/attention/feed-forward/modulation hosts in
//! `transformer.rs`, the Rust analog of the fork's `Flux2LoRAMapping`); per-file LoKr/LoRA dispatch,
//! LoRA-prefix detection, stacking + mixed, and the strict no-silent-drop policy are the shared core
//! seam (sc-2534), exactly as Qwen (sc-2528) and Z-Image (sc-2602) use it. LoRA/LoKr are
//! **transformer-only** for FLUX.2 (the VAE + Qwen3 text encoder are not adapter targets); the same
//! `Flux2Transformer` serves both the txt2img and edit variants, so this serves both.

use mlx_gen::adapters::loader::{apply_adapters_strict, ApplyReport};
use mlx_gen::adapters::AdaptableHost;
use mlx_gen::runtime::AdapterSpec;
use mlx_gen::Result;

/// Apply every adapter in `specs` onto a FLUX.2 transformer `host` (stacked, mixed LoRA/LoKr), via
/// the core [`apply_adapters_strict`] — errors, never silently drops, on an unmatched target. The
/// core `Adapter::residual` runs in the natural (fork-faithful) dtype — f32 here, since FLUX.2's
/// activations are f32 — so this change is dtype-invariant for the crate (sc-2718).
pub fn apply_flux2_adapters(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
) -> Result<ApplyReport> {
    apply_adapters_strict(host, specs, "flux2_klein_9b")
}
