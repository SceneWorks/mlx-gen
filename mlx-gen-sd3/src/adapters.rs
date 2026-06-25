//! SD3.5 adapter consumption (T2 sc-7883). The model-specific piece is the key→module map (the
//! top-level `AdaptableHost for Sd3Transformer` in `transformer.rs`, routing diffusers SD3 LoRA paths
//! `transformer_blocks.N.attn.to_q` / `…attn.add_q_proj` / `…attn2.to_q` / `…ff.net.0.proj` to the
//! module tree); everything else — per-file LoKr/LoRA dispatch, LoRA-prefix detection, stacking +
//! mixed, the `LoraAdapterMeta` alpha/rank consumer, and the strict no-silent-drop policy — is the
//! shared core seam (the same one z-image/Krea inference uses).
//!
//! This is the round-trip partner of the [`crate::training`] trainer: a LoRA the trainer saves via
//! `save_lora_peft` (key_prefix `""` → bare diffusers paths + the `.alpha` tensor + `networkType`/
//! `rank`/`alpha` `__metadata__`) reloads + applies here at `sd3_5_*` generation.

use mlx_gen::adapters::loader::{apply_adapters_strict, ApplyReport};
use mlx_gen::adapters::AdaptableHost;
use mlx_gen::runtime::AdapterSpec;
use mlx_gen::Result;

/// Apply every adapter in `specs` onto an SD3.5 transformer `host` (stacked, mixed LoRA/LoKr), via the
/// core [`apply_adapters_strict`] — errors, never silently drops, on an unmatched target.
pub fn apply_sd3_adapters(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
) -> Result<ApplyReport> {
    apply_adapters_strict(host, specs, "sd3")
}
