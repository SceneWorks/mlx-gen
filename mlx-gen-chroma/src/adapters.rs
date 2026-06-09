//! Chroma adapter consumption (sc-3842). The model-specific piece is the key→module map
//! (`AdaptableHost for ChromaTransformer` in `transformer.rs`, in diffusers/peft naming); everything
//! else — per-file LoRA/LoKr dispatch, prefix detection, stacking, and the strict no-silent-drop
//! policy — is the shared core seam. Chroma community LoRAs are diffusers/peft (and ComfyUI/kohya)
//! over the same `transformer_blocks.*`/`single_transformer_blocks.*` paths the DiT already uses.

use mlx_gen::adapters::loader::{apply_adapters_strict, ApplyReport};
use mlx_gen::adapters::AdaptableHost;
use mlx_gen::runtime::AdapterSpec;
use mlx_gen::Result;

/// Apply every adapter in `specs` onto a Chroma transformer `host` (stacked, mixed LoRA/LoKr), via
/// the core [`apply_adapters_strict`] — errors, never silently drops, on an unmatched target.
pub fn apply_chroma_adapters(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
) -> Result<ApplyReport> {
    apply_adapters_strict(host, specs, "chroma")
}
